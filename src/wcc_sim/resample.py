"""Flux-conserving resampling of a user image onto a detector-aligned sub-grid.

The image renderer convolves on a grid with k x k sub-pixels per detector
pixel, aligned with the detector (same orientation, pixel edges coinciding).
This module only does the geometry: it takes an image with a celestial WCS,
cuts out the detector pixels it covers, and redistributes its flux onto that
sub-grid. Optics (PSF, wings) are applied afterwards.

Sub-pixel convention (k odd): detector pixel i, centre at 0-based coordinate
i, spans [i - 0.5, i + 0.5) and holds sub-pixels k*(i - x0) + p, p = 0..k-1,
whose centres sit at i + (p - (k - 1)/2) / k. An odd k puts one sub-pixel
exactly on every detector-pixel centre.

Pixel values are treated as flux per pixel (not per area): the sum over the
returned cutout equals the input sum when the image lies on the detector.
"""

import math
import warnings
from typing import NamedTuple

import numpy as np
from astropy.wcs.utils import proj_plane_pixel_area

#: Finest sub-grid: the pipeline's PSF is oversampled 11x, and 1.5 mas
#: sub-pixels are already far below lambda/D (~41 mas in the r band).
MAX_SUBGRID = 11

#: Sample points per image edge when projecting the outline onto the detector;
#: catches edges that bow under a rotated or differently projected WCS.
_EDGE_SAMPLES = 17


class Subgrid(NamedTuple):
    """Cutout on a detector-aligned sub-grid: `data` has shape ((y1 - y0) * k, (x1 - x0) * k).

    `data` is in the input's units per sub-pixel (e-/s after
    SkyImageComponent.rate_map). k = 1 means detector resolution, e.g. the output of
    render.render_subgrid.
    """

    data: np.ndarray   # value per sub-pixel, in the input's units
    y0: int            # detector pixel of the cutout's first row
    x0: int            # detector pixel of the cutout's first column
    k: int             # sub-pixels per detector pixel (odd)
    total_in: float    # input sum, before clipping to the detector


def pixel_scale_mas(wcs):
    """Mean pixel side [mas] of a celestial WCS (sqrt of the pixel area)."""
    return math.sqrt(proj_plane_pixel_area(wcs.celestial)) * 3.6e6


def choose_subgrid(input_scale_mas, det_scale_mas, override=None):
    """Sub-pixels per detector pixel: the scale ratio rounded up to odd, 1..11.

    `override` is used as given, but must be a positive odd integer.
    """
    if override is not None:
        k = int(override)
        if k != override or k < 1 or k % 2 == 0:
            raise ValueError(f"subgrid must be a positive odd integer, got {override!r}")
        return k
    # the tolerance keeps an exact ratio (e.g. 3.0) from rounding up to 4 -> 5
    k = max(1, math.ceil(det_scale_mas / input_scale_mas - 1e-9))
    if k % 2 == 0:
        k += 1
    return min(k, MAX_SUBGRID)


def _outline_on_detector(img_wcs, img_shape, det_wcs):
    """Detector (x, y) of points along the image's outer pixel edges."""
    ny, nx = img_shape
    t = np.linspace(0.0, 1.0, _EDGE_SAMPLES)
    xs = np.concatenate([-0.5 + t * nx, np.full_like(t, nx - 0.5),
                         -0.5 + t * nx, np.full_like(t, -0.5)])
    ys = np.concatenate([np.full_like(t, -0.5), -0.5 + t * ny,
                         np.full_like(t, ny - 0.5), -0.5 + t * ny])
    ra, dec = img_wcs.celestial.pixel_to_world_values(xs, ys)
    return det_wcs.world_to_pixel_values(ra, dec)


def _box(x, y):
    """Smallest (y0, y1, x0, x1), ends exclusive, of pixels containing x, y."""
    return (int(math.floor(np.min(y) + 0.5)), int(math.floor(np.max(y) + 0.5)) + 1,
            int(math.floor(np.min(x) + 0.5)), int(math.floor(np.max(x) + 0.5)) + 1)


def footprint(img_wcs, img_shape, det_wcs, det_shape):
    """Detector pixels (y0, y1, x0, x1) covered by the image, or None if off-frame.

    Ends are exclusive and the box is clipped to the frame.
    """
    x, y = _outline_on_detector(img_wcs, img_shape, det_wcs)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return None  # outline does not project (e.g. > 90 deg from the pointing)
    y0, y1, x0, x1 = _box(x, y)
    ny, nx = det_shape
    y0, y1 = max(y0, 0), min(y1, ny)
    x0, x1 = max(x0, 0), min(x1, nx)
    if y0 >= y1 or x0 >= x1:
        return None
    return y0, y1, x0, x1


def subgrid_wcs(det_wcs, y0, x0, ny, nx, k):
    """WCS of the sub-grid over detector pixels [y0, y0 + ny) x [x0, x0 + nx).

    Same projection and orientation as `det_wcs`, pixel scale divided by k.
    FITS CRPIX c (1-based) maps to sub-grid CRPIX k * (c - x0 - 0.5) + 0.5.
    """
    w = det_wcs.deepcopy()
    crpix = np.asarray(w.wcs.crpix, dtype=float)
    w.wcs.crpix = [k * (crpix[0] - x0 - 0.5) + 0.5, k * (crpix[1] - y0 - 0.5) + 0.5]
    if w.wcs.has_cd():
        w.wcs.cd = np.asarray(w.wcs.cd) / k
    else:
        w.wcs.cdelt = np.asarray(w.wcs.cdelt) / k
    w.array_shape = (ny * k, nx * k)
    return w


def resample_to_subgrid(data, img_wcs, det_wcs, det_shape, k):
    """Resample `data` (flux per pixel, celestial `img_wcs`) onto the detector sub-grid.

    Uses reproject's exact spherical-polygon overlap, which conserves surface
    brightness, and rescales by the sub-pixel / input-pixel area ratio so the
    result is flux per sub-pixel. Returns a `Subgrid`, or None (with a
    warning) if the image misses the detector; warns when part of it falls
    off the frame.
    """
    try:
        from reproject import reproject_exact
    except ImportError as err:
        raise ImportError(
            "resampling user images needs the 'reproject' package; "
            "install it with: pip install 'wcc-sim[image]'"
        ) from err

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"image must be 2-D, got shape {data.shape}")
    img_wcs = img_wcs.celestial
    total_in = float(data.sum())

    box = footprint(img_wcs, data.shape, det_wcs, det_shape)
    if box is None:
        warnings.warn("image does not overlap the detector; nothing to resample",
                      UserWarning, stacklevel=2)
        return None
    y0, y1, x0, x1 = box

    sub_wcs = subgrid_wcs(det_wcs, y0, x0, y1 - y0, x1 - x0, k)
    mean, covered = reproject_exact((data, img_wcs), sub_wcs,
                                    shape_out=((y1 - y0) * k, (x1 - x0) * k))
    # reproject_exact returns the mean over the *covered* part of each
    # sub-pixel; weighting by the covered fraction stops sub-pixels on the
    # image edge from counting as fully lit.
    fine = np.nan_to_num(mean * covered, nan=0.0)
    fine *= proj_plane_pixel_area(sub_wcs) / proj_plane_pixel_area(img_wcs)

    x, y = _outline_on_detector(img_wcs, data.shape, det_wcs)
    if _box(x, y) != box:  # the frame edge clipped the image
        kept = float(fine.sum())
        lost = 1.0 - kept / total_in if total_in else 0.0
        warnings.warn(f"image extends beyond the detector; {lost:.2%} of its "
                      "flux falls outside the frame and is dropped",
                      UserWarning, stacklevel=2)
    return Subgrid(fine, y0, x0, k, total_in)
