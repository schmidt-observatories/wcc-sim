"""Per-frame re-centroiding of WCS-predicted star positions.

Center-of-mass (not a Gaussian fit) because the +1/+2-wave defocus PSFs are
centrally-depressed donuts; COM in a box sized to the PSF works for both.
"""

import warnings

import numpy as np
from photutils.centroids import centroid_com, centroid_sources

from .flags import FLAG_CENTROID


def centroid_stars(frame, x0, y0, box):
    """Refine (x0, y0) with a background-subtracted COM in a `box` px window.

    Returns (x, y, flags); a failed or runaway centroid (>box px from the
    start) falls back to the input position with FLAG_CENTROID set.
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    data = frame.image_e - np.median(frame.image_e)
    ny, nx = data.shape
    x_start = np.clip(x0, 0, nx - 1)
    y_start = np.clip(y0, 0, ny - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # partial-overlap warnings at edges
        x, y = centroid_sources(
            data, x_start, y_start, box_size=box, centroid_func=centroid_com
        )
    flags = np.zeros(x0.size, dtype=int)
    bad = (
        ~np.isfinite(x)
        | ~np.isfinite(y)
        | (np.hypot(x - x_start, y - y_start) > box)
    )
    flags[bad] |= FLAG_CENTROID
    x[bad] = x0[bad]
    y[bad] = y0[bad]
    return x, y, flags
