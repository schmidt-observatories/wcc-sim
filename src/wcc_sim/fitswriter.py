"""Multi-extension FITS output: SCI (ADU) + SATMASK + CAT + optional CLEAN, COMP."""

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
    "seed": ("SEED", "RNG seed (drawn at random if not given)"),
    "gaia_radius_arcsec": ("GAIARAD", "[arcsec] Gaia query radius"),
    "n_sources": ("NSRC", "number of injected sources"),
    "plate_scale_mas": ("PLTSCL", "[mas/pix] plate scale"),
    "n_pixels": ("NPIX", "pixels in the array"),
    "stamp_npix": ("STAMPPIX", "[pix] PSF stamp size (odd)"),
    "oversample": ("PSFOVER", "PSF stamp oversampling factor"),
    "add_noise": ("ADDNOISE", "noise realization applied"),
    "pixel_size_um": ("PIXSZUM", "[micron] detector pixel pitch"),
    "gain": ("GAIN", "[e-/ADU] sensor gain"),
    "bias": ("BIAS", "[ADU] bias level added before the ADC clip"),
    "gain_mode": ("GAINMODE", "readout gain mode (empty if single-mode)"),
    "saturation_e": ("SATLEVEL", "[e-] per-frame ceiling: min(well, ADC x gain)"),
    "read_noise": ("RDNOISE", "[e-] read noise per frame"),
    "dark_e_s": ("DARK", "[e-/s/pix] dark current"),
    "sky_e_s": ("SKYRATE", "[e-/s/pix] sky background rate"),
    "well_depth": ("WELLDEP", "[e-] full well (empty if None)"),
    "wings": ("WINGS", "analytic PSF wing extension applied"),
    "wavelength_nm": ("PSFWAVE", "[nm] field PSF wavelength (per-star: CAT)"),
    "scatter": ("SCATTER", "measured scattered-light halo applied"),
    "scatter_fraction": ("SCATFRAC", "instrument-wide scattered fraction"),
    "scatter_norm": ("SCATNORM", "SCATFRAC reference: F_FOCAL|P_FULL|USER"),
    "scatter_file": ("SCATFILE", "FRED stray-light map used"),
    "scatter_reach_px": ("SCATRCH", "[pix] modelled halo radial reach"),
    "wing_alpha": ("WINGA", "wing power-law slope (empty if wings off)"),
    "wing_c": ("WINGC", "wing amplitude [frac/pix at r=1px]"),
    "wing_floor_e": ("WINGFLR", "[e-/pix] wing truncation floor"),
    "chromatic": ("CHROMPSF", "spectrum-weighted effective PSFs used"),
    "n_extended": ("NEXTSRC", "number of extended-source components"),
    # image simulations (WCCObservation / simulate_image); per component: COMP
    "centre_ra": ("CEN_RA", "[deg] centre the pointing offset is applied to"),
    "centre_dec": ("CEN_DEC", "[deg] centre the pointing offset is applied to"),
    "offset_ra_arcsec": ("OFFSRA", "[arcsec] pointing offset in RA from CEN_RA"),
    "offset_dec_arcsec": ("OFFSDEC", "[arcsec] pointing offset in Dec from CEN_DEC"),
    "pointing_source": ("PNTSRC", "CEN_RA/CEN_DEC from: image (WCS) or user"),
    "image_total_e_s": ("IMGTOT", "[e-/s] total rate of the input image(s)"),
    "image_rate_e_s": ("IMGRATE", "[e-/s] rate on the frame after PSF + clipping"),
    "subgrid_k": ("IMGSUBK", "sub-pixels per detector pixel used (COMP)"),
    "flux_origin": ("FLUXORIG", "fluxes taken from the input image"),
    "image_bunit": ("IMGBUNIT", "BUNIT of the input image(s), as written"),
    "flux_jy": ("FLUXJY", "[Jy] total band-mean F_nu in SENSORF"),
    "flux_flam": ("FLUXFLAM", "[erg/s/cm2/A] total F_lambda at BANDPIV"),
    "flux_band": ("FLUXBAND", "[erg/s/cm2] total flux, FLUXFLAM x BANDWID"),
    "band_pivot_nm": ("BANDPIV", "[nm] pivot wavelength of SENSORF"),
    "band_width_nm": ("BANDWID", "[nm] top-hat-equivalent width of SENSORF"),
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


def build_hdulist(image_adu, satmask, catalog, wcs, params, image_clean=None,
                  components=None):
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
    if components is not None:
        comp_hdu = fits.table_to_hdu(Table(components))
        comp_hdu.name = "COMP"
        hdus.append(comp_hdu)
    return fits.HDUList(hdus)


def write_fits(path, image_adu, satmask, catalog, wcs, params,
               image_clean=None, overwrite=True):
    build_hdulist(image_adu, satmask, catalog, wcs, params, image_clean).writeto(
        path, overwrite=overwrite
    )
