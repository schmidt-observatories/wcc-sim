"""End-to-end WCC field simulation: Gaia -> rates -> PSF -> image -> FITS."""

from dataclasses import dataclass, field

import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.wcs import WCS

from .catalog import query_gaia
from .chromatic import effective_psf_for_spt
from .detectors import get_geometry, make_base_simulation
from .extended import render_extended
from .fitswriter import build_hdulist, write_fits
from .psf import DEFAULT_STAMP, render_oversampled_psf
from .render import (
    add_noise_and_digitize,
    bin_oversampled,
    render_scene,
    star_saturated,
)
from .scatter import halo_for_sensorfilter
from .starflux import rates_for_catalog, sky_and_dark_rates
from .wcsutil import build_wcs
from .wings import CombinedWing, fit_wing_model


def _core_wing(wing):
    """The diffraction term of a wing model, whether or not scatter is on."""
    return getattr(wing, "core", wing)


@dataclass
class SimulatedField:
    image_adu: np.ndarray
    image_e: np.ndarray
    image_clean: np.ndarray
    saturation_mask: np.ndarray
    wcs: WCS
    catalog: Table
    params: dict = field(default_factory=dict)
    #: PSF ingredients kept for diagnostics (see wcc_sim.psfreport); not
    #: written to FITS.
    models: dict = field(default_factory=dict, repr=False)

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
    wavelength_nm=None,
    scatter=True,
    scatter_fraction=None,
    cache_dir=None,
    write_clean=True,
    report=None,
    extended_sources=None,
    chromatic=False,
):
    """Simulate one WCC detector image of the Gaia field at (ra, dec).

    Parameters mirror the design spec; `shape=(ny, nx)` overrides the full
    array (useful for quick looks and tests), `catalog=` bypasses the Gaia
    query with a pre-made Table. `wings=True` (default) extends bright-star
    PSFs beyond the finite stamp with an analytic power-law wing (see
    wcc_sim.wings) so truncation stays below 0.1 sigma of the background
    noise instead of printing square "postage stamp" edges.

    `scatter=True` (default) adds the measured scattered-light halo from the
    FRED stray-light model (see wcc_sim.scatter) as a second additive PSF
    term, reaching ~37,000 px -- past the chip diagonal, so bright-star
    contamination of faint neighbours is modelled everywhere on the array.
    Beyond ~1000 px it exceeds the diffraction wing by an order of magnitude.
    `scatter_fraction` overrides the instrument-wide scattered fraction
    (default: FRED's own 5.542e-3). The halo is drawn by the wing machinery,
    so `wings=False` turns it off as well; `params["scatter"]` reports what
    actually ran.

    `wavelength_nm` overrides the wavelength the Airy core is computed at
    (default: the sensorfilter's central wavelength). It sets PSF *geometry*
    only -- the synthetic photometry still uses the sensorfilter's bandpass,
    so the star's electron rate is unchanged. Use it to put the core at the
    450 nm of the FRED stray-light run. Ignored for focus != 0, which uses
    fixed Huygens images.

    `report=path` writes a PDF+PNG showing the PSF's radial profile split
    into its Airy and scattered-light terms (see wcc_sim.psfreport).

    `extended_sources` (list of wcc_sim.extended.SersicComponent) adds
    smooth analytic components, rendered at native resolution and
    FFT-convolved with the PSF before the noise model. `chromatic=True`
    replaces the central-wavelength PSF with spectrum-weighted effective
    PSFs — per spectral type for point sources and per (template, ebv)
    for extended components; it is a documented no-op for focus != 0
    (the defocus PSF has no wavelength model).
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
        wavelength_m=None if wavelength_nm is None else float(wavelength_nm) * 1e-9,
    )
    psf_wavelength_nm = (
        float(sim.sensor.wavelength.to("nm").value)
        if wavelength_nm is None
        else float(wavelength_nm)
    )

    sky, dark = sky_and_dark_rates(sim)
    wing = None
    wing_floor_e = None
    halo = None
    scatter_active = bool(scatter) and bool(wings)
    if scatter_active:
        halo = halo_for_sensorfilter(
            sensorfilter, frac_total=scatter_fraction, sim=sim
        )
    if wings:
        wing = fit_wing_model(bin_oversampled(psf_os, oversample))
        if halo is not None:
            wing = CombinedWing(core=wing, halo=halo)
        read_noise = float(sim.sensor.read_noise.value)
        sigma_floor = np.sqrt(
            (sky + dark) * exptime + max(int(n_reads), 1) * read_noise**2
        )
        wing_floor_e = 0.1 * sigma_floor

    chromatic_active = bool(chromatic) and focus == 0

    def _wing_for(psf):
        if not wings:
            return None
        core = fit_wing_model(bin_oversampled(psf, oversample))
        return core if halo is None else CombinedWing(core=core, halo=halo)

    if chromatic_active and len(catalog):
        image_sources = np.zeros(shape, dtype=np.float32)
        for spt in np.unique(spts):
            sel = spts == spt
            psf_spt = effective_psf_for_spt(
                sim, sensorfilter, spt, 0.0, focus, oversample,
                stamp_npix=n_stamp, jitter_sigma_mas=jitter_sigma_mas,
            )
            image_sources += render_scene(
                shape, xs[sel], ys[sel], rates[sel] * exptime, psf_spt,
                oversample, wing=_wing_for(psf_spt), floor_e=wing_floor_e,
            )
    else:
        image_sources = render_scene(
            shape, xs, ys, rates * exptime, psf_os, oversample,
            wing=wing, floor_e=wing_floor_e,
        )

    if extended_sources:
        kernels = {}
        for comp in extended_sources:
            key = (comp.template, float(comp.ebv))
            if key in kernels:
                continue
            if chromatic_active:
                psf_ext = effective_psf_for_spt(
                    sim, sensorfilter, comp.template, comp.ebv, focus,
                    oversample, stamp_npix=n_stamp,
                    jitter_sigma_mas=jitter_sigma_mas,
                )
            else:
                psf_ext = psf_os
            kernels[key] = bin_oversampled(psf_ext, oversample)
        image_sources += render_extended(
            extended_sources, wcs, shape, geom.plate_scale_mas,
            sensorfilter, kernels, wing=wing,
        ) * np.float32(exptime)

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
        "nx": int(nx),
        "ny": int(ny),
        "n_pixels": int(nx) * int(ny),
        "stamp_npix": int(n_stamp),
        "oversample": int(oversample),
        "add_noise": bool(add_noise),
        "pixel_size_um": float(geom.pixel_size_um),
        "plate_scale_mas": float(geom.plate_scale_mas),
        "gain": float(sim.sensor.gain.to(u.electron / u.ct).value),
        "read_noise": float(sim.sensor.read_noise.value),
        "dark_e_s": dark,
        "sky_e_s": sky,
        "wings": bool(wings),
        "scatter": scatter_active,
        "scatter_fraction": (
            float(halo.frac_total) if halo is not None else None
        ),
        "scatter_file": (
            str(halo.meta.get("SCATFILE")) if halo is not None else None
        ),
        "scatter_reach_px": (
            float(halo.r_px[-1]) if halo is not None else None
        ),
        "wing_alpha": float(_core_wing(wing).alpha) if wing is not None else None,
        "wing_c": float(_core_wing(wing).c) if wing is not None else None,
        "wing_floor_e": (
            float(wing_floor_e) if wing_floor_e is not None else None
        ),
        "well_depth": (
            float(sim.sensor.meta["well_depth"])
            if sim.sensor.meta.get("well_depth") is not None
            else None
        ),
        "chromatic": bool(chromatic),
        "n_extended": len(extended_sources) if extended_sources else 0,
        "wavelength_nm": psf_wavelength_nm,
    }

    result = SimulatedField(
        image_adu=out["image_adu"],
        image_e=out["image_e"],
        image_clean=out["image_clean"],
        saturation_mask=out["satmask"],
        wcs=wcs,
        catalog=catalog,
        params=params,
        models={
            "wing": wing,
            "halo": halo,
            "stamp": bin_oversampled(psf_os, oversample),
            "stamp_npix": int(n_stamp),
            "wing_floor_e": wing_floor_e,
            "sky_dark_e": (sky + dark) * exptime,
            "pixel_size_um": geom.pixel_size_um,
            "diameter_m": float(sim.telescope.diameter_primary.to("m").value),
            "f_num": float(sim.telescope.f_num),
        },
    )
    if output is not None:
        result.write(output, write_clean=write_clean)
    if report is not None:
        from .psfreport import make_psf_report

        make_psf_report(result, report)
    return result
