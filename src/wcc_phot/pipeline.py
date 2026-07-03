"""Orchestration: frames -> select -> centroid -> photometry -> light curve."""

from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits
from astropy.table import Table

from . import __version__ as wcc_phot_version
from .apphot import aperture_photometry_frame
from .centroid import centroid_stars
from .geometry import default_geometry, geometry_from_r_ap, render_model_psf
from .io import load_frame
from .lightcurve import build_lightcurve
from .psfphot import build_psf_model, psf_photometry_frame
from .select import pick_references, pick_target

# params-dict key -> (FITS card, comment), wcc_sim.fitswriter style
_CARDS = {
    "method": ("METHOD", "photometry method (aperture|psf)"),
    "target_source_id": ("TARGID", "Gaia source_id of the target"),
    "n_ref": ("NREF", "number of reference stars used"),
    "n_frames": ("NFRAMES", "number of frames"),
    "sensorfilter": ("SENSORF", "wcc_etc sensorfilter (kind:band)"),
    "focus": ("FOCUS", "[waves] defocus (0=in-focus)"),
    "r_ap": ("RAP", "[px] aperture radius"),
    "r_in": ("RIN", "[px] background annulus inner radius"),
    "r_out": ("ROUT", "[px] background annulus outer radius"),
    "centroid_box": ("CBOX", "[px] centroid box side"),
    "fit_shape": ("FITSHP", "[px] PSF fit region side"),
    "ee_fraction": ("EEFRAC", "model EE inside r_ap (empty if r_ap user-set)"),
}


@dataclass
class PhotometryResult:
    stars: Table  # target + reference selection
    measurements: Table  # one row per star per frame
    lightcurve: Table  # one row per frame
    params: dict = field(default_factory=dict)

    def to_hdulist(self):
        header = fits.Header()
        for key, (card, comment) in _CARDS.items():
            value = self.params.get(key)
            if value is not None:
                header[card] = (value, comment)
        header["WCCPHOTV"] = (wcc_phot_version, "wcc_phot version")
        hdus = [fits.PrimaryHDU(header=header)]
        for name, table in (
            ("STARS", self.stars),
            ("PHOT", self.measurements),
            ("LC", self.lightcurve),
        ):
            hdu = fits.table_to_hdu(Table(table))
            hdu.name = name
            hdus.append(hdu)
        return fits.HDUList(hdus)

    def write(self, path, overwrite=True):
        self.to_hdulist().writeto(path, overwrite=overwrite)


def run_photometry(
    frames,
    target,
    method="aperture",
    n_ref=10,
    r_ap=None,
    r_in=None,
    r_out=None,
    centroid_box=None,
    fit_shape=None,
    ee=0.95,
    oversample=5,
    iso_dmag=1.0,
    iso_radius=None,
    times=None,
    output=None,
    on_frame=None,
):
    """Aperture or PSF photometry of a target + best n_ref reference stars.

    `frames` is a sequence of wcc-sim FITS paths, HDULists, or
    SimulatedFields of the same field; `target` a Gaia source_id or
    (ra, dec) in degrees. Every star is re-centroided in every frame
    starting from its WCS-predicted position. When `r_ap` is None the
    aperture geometry comes from the model-PSF encircled energy (`ee`).
    `times` (optional, len == n frames) is recorded in the tables; frame
    index is used otherwise. `on_frame` (optional callable, e.g. a
    wcc_phot.live.LiveViewer) is called after each frame is measured with
    an event dict (frame, n_frames, time, image_e, wcs, x, y, roles,
    geom, flux_e, flux_err_e, flags, rel_flux, rel_flux_err) for live
    display or custom hooks. Returns a PhotometryResult (optionally also
    written to `output` as a STARS/PHOT/LC FITS).
    """
    if method not in ("aperture", "psf"):
        raise ValueError(f"method must be 'aperture' or 'psf', got {method!r}")
    frame_list = [load_frame(f) for f in frames]
    if not frame_list:
        raise ValueError("no frames given")
    meta0 = frame_list[0].meta
    for fr in frame_list[1:]:
        if (
            fr.meta["sensorfilter"] != meta0["sensorfilter"]
            or fr.meta["focus"] != meta0["focus"]
        ):
            raise ValueError(
                "frames mix sensorfilters/focus levels; run them separately"
            )
    if times is None:
        times = np.arange(len(frame_list), dtype=float)
    times = np.asarray(times, dtype=float)
    if times.size != len(frame_list):
        raise ValueError("len(times) must match the number of frames")

    psf_os = None
    if r_ap is None or method == "psf":
        psf_os = render_model_psf(meta0, oversample=oversample)
    if r_ap is None:
        base = default_geometry(meta0, ee=ee, oversample=oversample, psf_os=psf_os)
        geom = geometry_from_r_ap(
            base.r_ap, r_in, r_out, centroid_box, fit_shape, base.ee_fraction
        )
    else:
        geom = geometry_from_r_ap(r_ap, r_in, r_out, centroid_box, fit_shape)

    frame0 = frame_list[0]
    target_idx = pick_target(frame0.catalog, target)
    ref_idx = pick_references(
        frame0.catalog,
        target_idx,
        frame0.image_e.shape,
        geom,
        n_ref=n_ref,
        iso_dmag=iso_dmag,
        iso_radius=iso_radius,
    )
    indices = [target_idx] + ref_idx
    roles = ["target"] + ["ref"] * len(ref_idx)
    ras = np.asarray(frame0.catalog["ra"], dtype=float)[indices]
    decs = np.asarray(frame0.catalog["dec"], dtype=float)[indices]
    source_ids = np.asarray(frame0.catalog["source_id"], dtype=np.int64)[indices]
    gmags = np.asarray(frame0.catalog["phot_g_mean_mag"], dtype=float)[indices]
    stars = Table(
        {
            "star": np.arange(len(indices)),
            "role": roles,
            "source_id": source_ids,
            "ra": ras,
            "dec": decs,
            "gmag": gmags,
        }
    )

    model = build_psf_model(meta0, oversample=oversample, psf_os=psf_os) if method == "psf" else None

    rows = []
    for k, fr in enumerate(frame_list):
        x_init, y_init = fr.wcs.world_to_pixel_values(ras, decs)
        x_init = np.atleast_1d(np.asarray(x_init, dtype=float))
        y_init = np.atleast_1d(np.asarray(y_init, dtype=float))
        x, y, cflags = centroid_stars(fr, x_init, y_init, geom.centroid_box)
        flux, flux_err, bkg, aflags = aperture_photometry_frame(fr, x, y, geom)
        flags = cflags | aflags
        if method == "psf":
            flux, flux_err, x, y = psf_photometry_frame(
                fr, x, y, flux, model, geom
            )
        if on_frame is not None:
            flux_ens = float(np.sum(flux[1:]))
            err_ens = float(np.sqrt(np.sum(flux_err[1:] ** 2)))
            rel = float(flux[0]) / flux_ens
            rel_err = abs(rel) * float(
                np.hypot(flux_err[0] / flux[0], err_ens / flux_ens)
            )
            on_frame(
                {
                    "frame": k,
                    "n_frames": len(frame_list),
                    "time": float(times[k]),
                    "image_e": fr.image_e,
                    "wcs": fr.wcs,
                    "x": x,
                    "y": y,
                    "roles": roles,
                    "geom": geom,
                    "flux_e": flux,
                    "flux_err_e": flux_err,
                    "flags": flags,
                    "rel_flux": rel,
                    "rel_flux_err": rel_err,
                }
            )
        for j in range(len(indices)):
            rows.append(
                {
                    "frame": k,
                    "time": times[k],
                    "star": j,
                    "source_id": source_ids[j],
                    "role": roles[j],
                    "x_init": x_init[j],
                    "y_init": y_init[j],
                    "x": x[j],
                    "y": y[j],
                    "flux_e": flux[j],
                    "flux_err_e": flux_err[j],
                    "bkg_e_pix": bkg[j],
                    "flags": int(flags[j]),
                }
            )
    measurements = Table(rows)
    lightcurve = build_lightcurve(measurements)

    params = {
        "method": method,
        "target_source_id": int(source_ids[0]),
        "n_ref": len(ref_idx),
        "n_frames": len(frame_list),
        "sensorfilter": meta0["sensorfilter"],
        "focus": int(meta0["focus"]),
        "r_ap": geom.r_ap,
        "r_in": geom.r_in,
        "r_out": geom.r_out,
        "centroid_box": geom.centroid_box,
        "fit_shape": geom.fit_shape,
        "ee_fraction": geom.ee_fraction,
    }
    result = PhotometryResult(stars, measurements, lightcurve, params)
    if output is not None:
        result.write(output)
    return result
