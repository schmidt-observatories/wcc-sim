"""End-to-end WCC field simulation: Gaia -> rates -> PSF -> image -> FITS."""

from dataclasses import dataclass, field

import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.wcs import WCS

from .catalog import query_gaia
from .detectors import get_geometry, make_base_simulation
from .fitswriter import build_hdulist, write_fits
from .psf import DEFAULT_STAMP, render_oversampled_psf
from .render import (
    add_noise_and_digitize,
    bin_oversampled,
    render_scene,
    star_saturated,
)
from .starflux import rates_for_catalog, sky_and_dark_rates
from .wcsutil import build_wcs
from .wings import fit_wing_model


@dataclass
class SimulatedField:
    image_adu: np.ndarray
    image_e: np.ndarray
    image_clean: np.ndarray
    saturation_mask: np.ndarray
    wcs: WCS
    catalog: Table
    params: dict = field(default_factory=dict)

    def to_hdulist(self, write_clean=True):
        return build_hdulist(
            self.image_adu,
            self.saturation_mask,
            self.catalog,
            self.wcs,
            self.params,
            image_clean=self.image_clean if write_clean else None,
        )

    def write(self, path, write_clean=True):
        self.to_hdulist(write_clean=write_clean).writeto(path, overwrite=True)


def simulate_field(
    ra,
    dec,
    sensorfilter="zwo:r",
    focus=None,
    exptime=90.0,
    n_reads=1,
    pa=0.0,
    jitter_sigma_mas=None,
    mag_limit=21.0,
    add_noise=True,
    seed=None,
    output=None,
    catalog=None,
    shape=None,
    stamp_npix=None,
    oversample=11,
    wings=True,
    cache_dir=None,
    write_clean=True,
):
    """Simulate one WCC detector image of the Gaia field at (ra, dec).

    Parameters mirror the design spec; `shape=(ny, nx)` overrides the full
    array (useful for quick looks and tests), `catalog=` bypasses the Gaia
    query with a pre-made Table. `wings=True` (default) extends bright-star
    PSFs beyond the finite stamp with an analytic power-law wing (see
    wcc_sim.wings) so truncation stays below 0.1 sigma of the background
    noise instead of printing square "postage stamp" edges.
    """
    sim = make_base_simulation(sensorfilter)
    geom = get_geometry(sensorfilter, sim=sim)
    if focus is None:
        focus = geom.default_focus
    if focus not in (0, 1, 2):
        raise ValueError(f"focus must be 0, 1, or 2, got {focus!r}")
    if shape is None:
        shape = (geom.ny, geom.nx)
    ny, nx = shape

    wcs = build_wcs(ra, dec, geom.plate_scale_mas, pa, shape)

    half_diag_arcsec = (
        0.5 * np.hypot(nx, ny) * geom.plate_scale_mas / 1000.0
    )
    radius_arcsec = half_diag_arcsec + 10.0
    if catalog is None:
        catalog = query_gaia(
            ra, dec, radius_arcsec, mag_limit=mag_limit, cache_dir=cache_dir
        )
    catalog = catalog.copy()

    if len(catalog):
        rates, spts = rates_for_catalog(catalog, sensorfilter)
        xs, ys = wcs.world_to_pixel_values(
            np.asarray(catalog["ra"], dtype=float),
            np.asarray(catalog["dec"], dtype=float),
        )
    else:
        rates = np.array([])
        spts = np.array([], dtype=str)
        xs = np.array([])
        ys = np.array([])

    n_stamp = stamp_npix if stamp_npix is not None else DEFAULT_STAMP[focus]
    psf_os = render_oversampled_psf(
        sim,
        focus,
        oversample=oversample,
        stamp_npix=n_stamp,
        jitter_sigma_mas=jitter_sigma_mas,
    )

    sky, dark = sky_and_dark_rates(sim)
    wing = None
    wing_floor_e = None
    if wings:
        wing = fit_wing_model(bin_oversampled(psf_os, oversample))
        read_noise = float(sim.sensor.read_noise.value)
        sigma_floor = np.sqrt(
            (sky + dark) * exptime + max(int(n_reads), 1) * read_noise**2
        )
        wing_floor_e = 0.1 * sigma_floor

    image_sources = render_scene(
        shape, xs, ys, rates * exptime, psf_os, oversample,
        wing=wing, floor_e=wing_floor_e,
    )

    rng = np.random.default_rng(seed)
    out = add_noise_and_digitize(
        image_sources, sim, exptime, n_reads, rng, add_noise=add_noise
    )

    catalog["x"] = np.asarray(xs, dtype=float)
    catalog["y"] = np.asarray(ys, dtype=float)
    catalog["spt"] = spts
    catalog["rate_e_s"] = np.asarray(rates, dtype=float)
    in_image = (
        (catalog["x"] > -0.5) & (catalog["x"] < nx - 0.5)
        & (catalog["y"] > -0.5) & (catalog["y"] < ny - 0.5)
        if len(catalog)
        else np.array([], dtype=bool)
    )
    catalog["in_image"] = np.asarray(in_image, dtype=bool)
    saturated = np.zeros(len(catalog), dtype=bool)
    for i in np.flatnonzero(np.asarray(in_image, dtype=bool)):
        saturated[i] = star_saturated(
            out["satmask"], catalog["x"][i], catalog["y"][i]
        )
    catalog["saturated"] = saturated

    params = {
        "ra": float(ra),
        "dec": float(dec),
        "pa": float(pa),
        "sensorfilter": sensorfilter,
        "focus": int(focus),
        "exptime": float(exptime),
        "n_reads": int(n_reads),
        "jitter_sigma_mas": (
            float(jitter_sigma_mas)
            if jitter_sigma_mas is not None
            else float(sim.telescope.jitter_sigma.to("mas").value)
        ),
        "mag_limit": float(mag_limit),
        "seed": seed,
        "gaia_radius_arcsec": float(radius_arcsec),
        "n_sources": int(len(catalog)),
        "plate_scale_mas": float(geom.plate_scale_mas),
        "gain": float(sim.sensor.gain.to(u.electron / u.ct).value),
        "read_noise": float(sim.sensor.read_noise.value),
        "dark_e_s": dark,
        "sky_e_s": sky,
        "wings": bool(wings),
        "wing_alpha": float(wing.alpha) if wing is not None else None,
        "wing_c": float(wing.c) if wing is not None else None,
        "wing_floor_e": (
            float(wing_floor_e) if wing_floor_e is not None else None
        ),
        "well_depth": (
            float(sim.sensor.meta["well_depth"])
            if sim.sensor.meta.get("well_depth") is not None
            else None
        ),
    }

    result = SimulatedField(
        image_adu=out["image_adu"],
        image_e=out["image_e"],
        image_clean=out["image_clean"],
        saturation_mask=out["satmask"],
        wcs=wcs,
        catalog=catalog,
        params=params,
    )
    if output is not None:
        result.write(output, write_clean=write_clean)
    return result
