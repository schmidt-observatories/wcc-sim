"""Shared plotting helpers for wcc_sim and wcc_phot.

Every public plot function in the two packages takes ``ax=None``: pass an
axes to draw into your own figure, or leave it out and get a new one in the
packaged house style (``plot_default.mplstyle``). The result objects expose
the same functions as methods, ``field.plot_cutout(...)`` and
``result.plot_lightcurve(...)``, so a tutorial cell is usually one line.

This module holds what the rest of the plotting code shares: the style, the
colour tokens, the ``ax=None`` helper, and the handful of plots that every
tutorial used to rewrite by hand (an image with a catalog overlay, a radial
profile, a PSF stamp, an encircled-energy curve).

pyplot is imported only when a function has to create a figure, so the
report writers stay headless.
"""

import contextlib
import os

import numpy as np

#: The packaged house style, usable with ``plt.style.use(STYLE_PATH)``.
STYLE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "plot_default.mplstyle"
)

#: Colour tokens of the house palette, in cycle order.
TEAL = "#00798c"
RED = "#d1495b"
AMBER = "#edae49"
BLUE = "#30638e"
GREEN = "#66a182"
DARK = "#003d5b"

#: Semantic roles used across the packages.
TARGET_COLOR = RED
REF_COLOR = TEAL
CATALOG_COLOR = AMBER


#: rcParams a style context must not roll back. In a fresh Jupyter kernel the
#: inline backend is activated lazily by the first figure, and that switches
#: ``interactive`` on; a plain ``matplotlib.style.context`` entered before then
#: restores it to False on exit, and the kernel stops flushing figures to the
#: cell output from that point on.
_KEEP_RC = ("backend", "interactive")


@contextlib.contextmanager
def style_context():
    """Context manager applying the packaged style to what is drawn inside."""
    import matplotlib
    import matplotlib.style

    orig = matplotlib.rcParams.copy()
    try:
        matplotlib.style.use(STYLE_PATH)
        yield
    finally:
        for key in _KEEP_RC:
            orig[key] = matplotlib.rcParams[key]
        dict.update(matplotlib.rcParams, orig)  # as matplotlib.rc_context does


def use_style():
    """Apply the packaged style globally (for notebooks and scripts)."""
    import matplotlib.pyplot as plt

    plt.style.use(STYLE_PATH)


@contextlib.contextmanager
def axes(ax=None, **subplots_kw):
    """Yield ``ax``, or a new styled pyplot axes when it is None.

    The style is applied only to a figure this helper creates; an axes
    passed in keeps the caller's own rcParams.
    """
    if ax is not None:
        yield ax
        return
    import matplotlib.pyplot as plt

    with style_context():
        _, ax = plt.subplots(**subplots_kw)
        yield ax


# --------------------------------------------------------------------------- #
# Numerics shared by the plots and the reports                                 #
# --------------------------------------------------------------------------- #

def block_max(image, factor):
    """Downsample by taking the max of each ``factor x factor`` block.

    Max, not mean: a 61-Mpix frame has to be reduced ~8x to display, and
    averaging would wash stars and the inner halo out of the picture.
    """
    image = np.asarray(image)
    if factor <= 1:
        return image
    ny, nx = image.shape
    ny, nx = (ny // factor) * factor, (nx // factor) * factor
    return image[:ny, :nx].reshape(
        ny // factor, factor, nx // factor, factor
    ).max(axis=(1, 3))


def azimuthal_average(image, xc, yc, edges):
    """Mean pixel value in each annulus of ``edges``; NaN where empty.

    Returns ``(mean, count)`` with one entry per bin.
    """
    image = np.asarray(image, dtype=float)
    ny, nx = image.shape
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(yy - yc, xx - xc).ravel()
    idx = np.digitize(r, edges)
    n = edges.size + 1
    count = np.bincount(idx, minlength=n)
    total = np.bincount(idx, weights=image.ravel(), minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return mean[1 : edges.size], count[1 : edges.size]


def radial_profile(image, xc, yc, r_max=None, n_bins=80, log=False, bg=0.0):
    """Azimuthally averaged profile of ``image`` about ``(xc, yc)``.

    Returns ``(r, profile)`` with ``r`` the bin centres in pixels and
    ``profile`` the mean of ``image - bg`` in each annulus. ``log=True``
    uses log-spaced bins from 1 px, which is what a PSF wing wants.
    """
    ny, nx = np.shape(image)
    if r_max is None:
        r_max = min(xc, nx - 1 - xc, yc, ny - 1 - yc)
    if log:
        edges = np.geomspace(1.0, max(float(r_max), 2.0), n_bins + 1)
        r = np.sqrt(edges[:-1] * edges[1:])
    else:
        edges = np.linspace(0.0, float(r_max), n_bins + 1)
        r = 0.5 * (edges[1:] + edges[:-1])
    mean, _ = azimuthal_average(np.asarray(image, dtype=float) - bg, xc, yc, edges)
    return r, mean


def encircled_energy(image, xc=None, yc=None, total=None):
    """Cumulative fraction of ``image`` inside radius r, about ``(xc, yc)``.

    Returns ``(r_sorted, ee)``; the centre defaults to the array centre,
    which is the right choice for a PSF stamp. ``ee`` is normalised to
    ``total``, by default the sum of the image, so a stamp reaches 1; pass
    the star's true flux to see how far a truncated render falls short.
    """
    image = np.asarray(image, dtype=float)
    ny, nx = image.shape
    if xc is None:
        xc = (nx - 1) / 2.0
    if yc is None:
        yc = (ny - 1) / 2.0
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(xx - xc, yy - yc).ravel()
    order = np.argsort(r)
    ee = np.cumsum(image.ravel()[order].astype(np.float64))
    return r[order], ee / (ee[-1] if total is None else float(total))


# --------------------------------------------------------------------------- #
# Plots                                                                        #
# --------------------------------------------------------------------------- #

def plot_image(image, ax=None, catalog=None, satmask=None, stretch="asinh",
               percent=99.7, norm=None, cmap="gray_r", max_px=1200,
               title=None, **imshow_kw):
    """An image with an asinh stretch, optional catalog circles and satmask.

    ``catalog`` is a table with ``x``/``y`` columns (an ``in_image`` column,
    when present, selects the rows drawn); ``satmask`` a boolean array of
    the image's shape, drawn as a contour. ``norm`` replaces the
    ``stretch``/``percent`` default for a hand-set stretch. Frames larger
    than ``max_px`` on a side are block-max reduced for display, with the
    axes kept in native pixels. Returns the axes.
    """
    from astropy.visualization import simple_norm

    image = np.asarray(image)
    ny, nx = image.shape
    factor = max(1, int(np.ceil(max(ny, nx) / float(max_px))))
    shown = block_max(image, factor)
    extent = (-0.5, nx - 0.5, -0.5, ny - 0.5)
    with axes(ax, figsize=(7.5, 7.5 * ny / nx)) as ax:
        if norm is None:
            norm = simple_norm(shown, stretch, percent=percent)
        ax.imshow(shown, origin="lower", cmap=cmap, norm=norm, extent=extent,
                  interpolation="nearest", **imshow_kw)
        if catalog is not None and len(catalog):
            sel = (
                np.asarray(catalog["in_image"], dtype=bool)
                if "in_image" in catalog.colnames
                else np.ones(len(catalog), dtype=bool)
            )
            ax.scatter(np.asarray(catalog["x"])[sel], np.asarray(catalog["y"])[sel],
                       s=80, facecolor="none", edgecolor=CATALOG_COLOR, lw=0.9,
                       label=f"catalog ({int(sel.sum())})")
            ax.legend(loc="upper right")
        if satmask is not None and np.any(satmask):
            mask = block_max(np.asarray(satmask, dtype=np.uint8), factor)
            ax.contour(mask, levels=[0.5], colors=[RED], linewidths=0.8,
                       extent=extent, origin="lower")
        ax.set(xlabel="x [px]", ylabel="y [px]", xlim=extent[:2], ylim=extent[2:])
        ax.grid(False)
        if title:
            ax.set_title(title)
    return ax


def plot_radial_profile(image, xc, yc, ax=None, bg=0.0, r_max=None, n_bins=80,
                        log=False, **plot_kw):
    """Draw :func:`radial_profile` on log-y axes; returns the axes."""
    r, prof = radial_profile(image, xc, yc, r_max=r_max, n_bins=n_bins,
                             log=log, bg=bg)
    ok = np.isfinite(prof) & (prof > 0)
    with axes(ax, figsize=(7.5, 5)) as ax:
        ax.plot(r[ok], prof[ok], **plot_kw)
        ax.set(yscale="log", xlabel="radius [px]", ylabel="mean pixel value")
        if log:
            ax.set_xscale("log")
        if "label" in plot_kw:
            ax.legend()
    return ax


def plot_stamp(stamp, ax=None, half_width=None, decades=8, cmap="magma",
               plate_scale_mas=None, title=None):
    """A centred PSF stamp on a log10 colour scale; returns the axes.

    The extent is in pixels from the centre, or in arcsec when
    ``plate_scale_mas`` is given. ``decades`` sets the dynamic range below
    the peak; ``half_width`` crops the view.
    """
    stamp = np.asarray(stamp, dtype=float)
    n = stamp.shape[0]
    half = n // 2
    scale = 1.0 if plate_scale_mas is None else plate_scale_mas / 1000.0
    unit = "px" if plate_scale_mas is None else "arcsec"
    ext = np.array([-half - 0.5, half + 0.5, -half - 0.5, half + 0.5]) * scale
    peak = np.log10(stamp.max())
    with axes(ax, figsize=(5, 4.6)) as ax:
        im = ax.imshow(np.log10(np.maximum(stamp, 10.0 ** (peak - decades - 1))),
                       origin="lower", cmap=cmap, vmin=peak - decades, vmax=peak,
                       extent=ext, interpolation="nearest")
        if half_width is not None:
            ax.set(xlim=(-half_width * scale, half_width * scale),
                   ylim=(-half_width * scale, half_width * scale))
        ax.set(xlabel=f"$\\Delta x$ [{unit}]", ylabel=f"$\\Delta y$ [{unit}]")
        ax.grid(False)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                           label="log$_{10}$ PSF fraction / px")
        if title:
            ax.set_title(title)
    return ax


def plot_encircled_energy(image, ax=None, xc=None, yc=None, log_x=True,
                          **plot_kw):
    """Draw :func:`encircled_energy`; returns the axes."""
    r, ee = encircled_energy(image, xc=xc, yc=yc)
    # thin the curve: every pixel of a 257 px stamp is 66k points
    idx = np.unique(np.searchsorted(r, np.geomspace(max(r[1], 0.5), r[-1], 400)))
    with axes(ax, figsize=(7.5, 5)) as ax:
        ax.plot(r[idx], ee[idx], **plot_kw)
        ax.set(xlabel="radius [px]", ylabel="encircled energy", ylim=(0, 1.02))
        if log_x:
            ax.set_xscale("log")
        if "label" in plot_kw:
            ax.legend()
    return ax
