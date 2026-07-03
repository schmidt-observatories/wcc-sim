"""Multi-extension FITS output: SCI (ADU) + SATMASK + CAT + optional CLEAN."""

from datetime import datetime, timezone

import numpy as np
from astropy.io import fits
from astropy.table import Table

import wcc_etc

from . import __version__ as wcc_sim_version

# params-dict key -> (FITS card, comment)
_CARDS = {
    "ra": ("RA_PNT", "[deg] pointing RA (ICRS), array center"),
    "dec": ("DEC_PNT", "[deg] pointing Dec (ICRS), array center"),
    "pa": ("PA", "[deg] position angle, E of N"),
    "sensorfilter": ("SENSORF", "wcc_etc sensorfilter (kind:band)"),
    "focus": ("FOCUS", "[waves] defocus (0=in-focus)"),
    "exptime": ("EXPTIME", "[s] total exposure time"),
    "n_reads": ("NREADS", "coadded frames (ETC semantics)"),
    "jitter_sigma_mas": ("JITTER", "[mas] jitter Gaussian sigma"),
    "mag_limit": ("MAGLIM", "Gaia G faint limit of injected sources"),
    "seed": ("SEED", "RNG seed (empty if None)"),
    "gaia_radius_arcsec": ("GAIARAD", "[arcsec] Gaia query radius"),
    "n_sources": ("NSRC", "number of injected sources"),
    "plate_scale_mas": ("PLTSCL", "[mas/pix] plate scale"),
    "gain": ("GAIN", "[e-/ADU] sensor gain"),
    "read_noise": ("RDNOISE", "[e-] read noise per frame"),
    "dark_e_s": ("DARK", "[e-/s/pix] dark current"),
    "sky_e_s": ("SKYRATE", "[e-/s/pix] sky background rate"),
    "well_depth": ("WELLDEP", "[e-] full well (empty if None)"),
    "wings": ("WINGS", "analytic PSF wing extension applied"),
    "wing_alpha": ("WINGA", "wing power-law slope (empty if wings off)"),
    "wing_c": ("WINGC", "wing amplitude [frac/pix at r=1px]"),
    "wing_floor_e": ("WINGFLR", "[e-/pix] wing truncation floor"),
    "chromatic": ("CHROMPSF", "spectrum-weighted effective PSFs used"),
    "n_extended": ("NEXTSRC", "number of extended (Sersic) components"),
}


def _header_from(params, wcs):
    header = wcs.to_header()
    for key, (card, comment) in _CARDS.items():
        if key in params:
            value = params[key]
            if value is None:
                continue
            header[card] = (value, comment)
    header["WCCSIMV"] = (wcc_sim_version, "wcc_sim version")
    header["WCCETCV"] = (wcc_etc.__version__, "wcc_etc version")
    header["DATE"] = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "file creation time (UTC)",
    )
    header["BUNIT"] = ("adu", "SCI pixel units")
    return header


def build_hdulist(image_adu, satmask, catalog, wcs, params, image_clean=None):
    primary = fits.PrimaryHDU(
        data=np.asarray(image_adu, dtype=np.float32),
        header=_header_from(params, wcs),
    )
    primary.name = "SCI"
    cat_hdu = fits.table_to_hdu(Table(catalog))  # robust for unicode columns
    cat_hdu.name = "CAT"
    hdus = [
        primary,
        fits.ImageHDU(
            data=np.asarray(satmask, dtype=np.uint8), name="SATMASK"
        ),
        cat_hdu,
    ]
    if image_clean is not None:
        clean = fits.ImageHDU(
            data=np.asarray(image_clean, dtype=np.float32), name="CLEAN"
        )
        clean.header["BUNIT"] = ("electron", "noiseless expectation image")
        hdus.append(clean)
    return fits.HDUList(hdus)


def write_fits(path, image_adu, satmask, catalog, wcs, params,
               image_clean=None, overwrite=True):
    build_hdulist(image_adu, satmask, catalog, wcs, params, image_clean).writeto(
        path, overwrite=overwrite
    )
