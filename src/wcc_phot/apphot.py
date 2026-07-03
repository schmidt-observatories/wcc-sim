"""Aperture photometry with CCD-equation errors from the frame header rates."""

import numpy as np
from astropy.stats import SigmaClip
from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture

from .flags import FLAG_EDGE, FLAG_SATURATED


def _saturated_within(satmask, x, y, radius):
    """True if any saturated pixel lies within `radius` px of (x, y)."""
    ny, nx = satmask.shape
    r = int(np.ceil(radius))
    x0, y0 = int(round(x)), int(round(y))
    y_lo, y_hi = max(y0 - r, 0), min(y0 + r + 1, ny)
    x_lo, x_hi = max(x0 - r, 0), min(x0 + r + 1, nx)
    if y_lo >= y_hi or x_lo >= x_hi:
        return False
    return bool(satmask[y_lo:y_hi, x_lo:x_hi].any())


def aperture_photometry_frame(frame, x, y, geom):
    """Background-subtracted aperture fluxes at (x, y) on the electron image.

    Returns (flux_e, flux_err_e, bkg_e_pix, flags). Background is the
    sigma-clipped annulus median; the variance is Poisson of the star plus
    the per-pixel background variance from the header rates
    ((sky+dark)*exptime + n_reads*read_noise^2), including the
    background-estimate term (area^2/n_annulus).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    positions = np.column_stack([x, y])
    aper = CircularAperture(positions, r=geom.r_ap)
    annulus = CircularAnnulus(positions, r_in=geom.r_in, r_out=geom.r_out)

    bkg = np.asarray(
        ApertureStats(
            frame.image_e, annulus, sigma_clip=SigmaClip(sigma=3.0)
        ).median,
        dtype=float,
    )
    raw = np.asarray(ApertureStats(frame.image_e, aper).sum, dtype=float)
    area = aper.area
    flux = raw - bkg * area

    m = frame.meta
    var_pix = (
        (float(m["sky_e_s"]) + float(m["dark_e_s"])) * float(m["exptime"])
        + int(m["n_reads"]) * float(m["read_noise"]) ** 2
    )
    variance = np.clip(flux, 0, None) + area * var_pix * (
        1.0 + area / annulus.area
    )
    flux_err = np.sqrt(variance)

    ny, nx = frame.image_e.shape
    flags = np.zeros(x.size, dtype=int)
    edge = (
        (x < geom.r_out)
        | (x > nx - 1 - geom.r_out)
        | (y < geom.r_out)
        | (y > ny - 1 - geom.r_out)
    )
    flags[edge] |= FLAG_EDGE
    for i in range(x.size):
        if _saturated_within(frame.satmask, x[i], y[i], geom.r_ap):
            flags[i] |= FLAG_SATURATED
    return flux, flux_err, bkg, flags
