"""TAN WCS construction for a single WCC detector (no distortion)."""

import numpy as np
from astropy.wcs import WCS


def build_wcs(ra_deg, dec_deg, plate_scale_mas, pa_deg, shape):
    """TAN WCS centered on (ra, dec). shape is (ny, nx); pa in deg E of N."""
    ny, nx = shape
    scale_deg = plate_scale_mas / 1000.0 / 3600.0
    pa = np.deg2rad(pa_deg)
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra_deg, dec_deg]
    # FITS CRPIX is 1-based; array center in 0-based coords is ((nx-1)/2, (ny-1)/2)
    w.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    w.wcs.cd = np.array(
        [
            [-scale_deg * np.cos(pa), scale_deg * np.sin(pa)],
            [scale_deg * np.sin(pa), scale_deg * np.cos(pa)],
        ]
    )
    w.array_shape = shape
    return w
