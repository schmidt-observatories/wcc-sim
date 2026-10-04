"""PSF radial-profile report: the Airy core and the scattered-light halo.

`psf_decomposition(field)` splits a simulated field's PSF into its two
physical terms on a common radial grid and measures the profile that
actually landed in the image, so the model and the render can be compared
directly. `make_psf_report(field, path)` draws that as a one-page PDF+PNG.

The headline number is the Airy-peak-to-scatter-peak contrast, defined
exactly as ``wcc_etc.scatter_psf.report_diagnostics`` defines it -- the
analytic Airy peak of a ``(1 - f_scat)`` core over the halo's central
irradiance, both in W/mm^2 per watt of source -- so it is directly
comparable with the wcc_etc build report. At 450 nm it is ~1e9.

Rendering uses a plain (pyplot-free) matplotlib Figure, so it works
headless and never opens a window.
"""

import contextlib
import os

import numpy as np

#: gks house-style tokens (see the gks-plotting skill).
_TEAL = "#00798c"
_RED = "#d1495b"
_AMBER = "#edae49"
_DARK = "#003d5b"


def _style_context():
    """The 'gks' house style when installed, matplotlib defaults otherwise."""
    import matplotlib.style

    if "gks" in matplotlib.style.available:
        return matplotlib.style.context("gks")
    return contextlib.nullcontext()


def _azimuthal_average(image, xc, yc, edges):
    """Mean pixel value in each annulus of `edges`, NaN where empty."""
    ny, nx = image.shape
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(yy - yc, xx - xc).ravel()
    idx = np.digitize(r, edges)
    n = edges.size + 1
    count = np.bincount(idx, minlength=n)
    total = np.bincount(idx, weights=image.ravel().astype(float), minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return mean[1 : edges.size], count[1 : edges.size]


def _stamp_profile(stamp):
    """Azimuthal average of a PSF stamp on integer radii.

    Integer radii, not the report's log grid: the inner log bins are
    sub-pixel wide and would come back empty, which reads as a hole in the
    Airy core rather than as too fine a grid.
    """
    n = stamp.shape[0]
    c = n // 2
    yy, xx = np.mgrid[:n, :n]
    r_int = np.rint(np.hypot(yy - c, xx - c)).astype(np.int64).ravel()
    count = np.bincount(r_int)
    total = np.bincount(r_int, weights=stamp.ravel())
    keep = (count > 0) & (total > 0)
    r = np.arange(count.size, dtype=float)
    return r[keep], total[keep] / count[keep]


def psf_decomposition(field, n_bins=140, star=0):
    """Radial profiles of the PSF's Airy and scattered-light terms.

    Parameters
    ----------
    field : SimulatedField
        Must come from a run that kept `models` (any `simulate_field` call).
    n_bins : int
        Log-spaced radial bins.
    star : int
        Catalog row to profile; the brightest star is usually row 0.

    Returns
    -------
    dict
        ``r_px`` and, on that grid, ``airy`` / ``scatter`` / ``total``
        (model, PSF fraction per detector pixel) and ``measured`` (from
        ``image_clean``, background-subtracted and divided by the star's
        electrons). Plus the peak irradiances, their ratio ``contrast``,
        the ``crossover_px`` where scatter overtakes diffraction, the drawn
        ``r_out_px`` and the ``floor_frac`` it came from.
    """
    from wcc_etc.scatter_psf import airy_irradiance

    models = field.models
    if not models:
        raise ValueError(
            "field carries no PSF models; psf_decomposition needs a field "
            "produced by simulate_field (not one re-read from FITS)"
        )
    wing, halo = models["wing"], models["halo"]
    stamp = np.asarray(models["stamp"], dtype=float)
    half = stamp.shape[0] // 2
    mm_per_px = float(models["pixel_size_um"]) / 1000.0

    ny, nx = field.image_clean.shape
    xc = float(field.catalog["x"][star])
    yc = float(field.catalog["y"][star])
    flux_e = float(field.catalog["rate_e_s"][star]) * float(field.params["exptime"])

    r_max = min(
        max(xc, nx - xc, yc, ny - yc),
        halo.r_px[-1] if halo is not None else np.inf,
    )
    edges = np.geomspace(1.0, max(r_max, 8.0), n_bins + 1)
    r_px = np.sqrt(edges[:-1] * edges[1:])  # log-centre of each bin

    # -- model terms ------------------------------------------------------- #
    core_scale = getattr(wing, "core_scale", 1.0) if wing is not None else 1.0
    core = getattr(wing, "core", wing)
    r_stamp, p_stamp = _stamp_profile(stamp)
    inner = np.exp(
        np.interp(np.log(r_px), np.log(np.maximum(r_stamp, 0.5)), np.log(p_stamp))
    )
    outer = (
        core.profile(np.maximum(r_px, 1.0))
        if core is not None
        else np.zeros_like(r_px)
    )
    airy = core_scale * np.where(r_px < half, inner, outer)
    scatter = (
        halo.profile(r_px) if halo is not None else np.zeros_like(r_px)
    )
    total = airy + scatter

    # -- what actually landed in the image --------------------------------- #
    bg = float(models["sky_dark_e"])
    measured, _ = _azimuthal_average(field.image_clean - bg, xc, yc, edges)
    measured = measured / flux_e if flux_e > 0 else measured * np.nan

    # -- peak contrast, wcc_etc's definition -------------------------------- #
    lam_m = float(field.params["wavelength_nm"]) * 1e-9
    airy_peak = float(
        airy_irradiance(
            0.0, float(models["diameter_m"]), float(models["f_num"]), lam_m,
            power=core_scale,
        )
    )
    if halo is not None:
        from wcc_etc.scatter_psf import crossover_radius

        scatter_peak = float(halo.profile(0.0)) / mm_per_px**2
        contrast = airy_peak / scatter_peak
        # On the model's own reach, not the frame's: where the two terms
        # cross is a property of the PSF, and the crossover often lies
        # outside a small subarray.
        r_model = np.geomspace(float(half), float(halo.r_px[-1]), 2000)
        above = np.flatnonzero(
            halo.profile(r_model) > core_scale * core.profile(r_model)
        )
        crossover = float(r_model[above[0]]) if above.size else None
        # The fringed crossover -- the last radius at which an Airy *ring
        # maximum* still pokes above the halo -- is what a peak-to-peak
        # requirement means, and sits well outside the crossover of the
        # azimuthally averaged profile. Both are reported because the two
        # are easy to confuse.
        halo_irradiance = halo.frac_per_px / mm_per_px**2
        r_halo_mm = halo.r_px * mm_per_px
        cross_mm = {}
        for key, envelope in (("fringed", False), ("envelope", True)):
            value = crossover_radius(
                r_halo_mm, halo_irradiance,
                float(models["diameter_m"]), float(models["f_num"]), lam_m,
                power=core_scale, envelope=envelope,
            )
            cross_mm[key] = None if not np.isfinite(value) else float(value)
    else:
        scatter_peak = None
        contrast = None
        crossover = None
        cross_mm = {"fringed": None, "envelope": None}

    floor_e = models.get("wing_floor_e")
    return {
        "r_px": r_px,
        "airy": airy,
        "scatter": scatter,
        "total": total,
        "measured": measured,
        "airy_peak_irradiance": airy_peak,
        "scatter_peak_irradiance": scatter_peak,
        "contrast": contrast,
        "crossover_px": crossover,
        "crossover_mm": None if crossover is None else crossover * mm_per_px,
        "crossover_fringed_mm": cross_mm["fringed"],
        "crossover_fringed_px": (
            None if cross_mm["fringed"] is None
            else cross_mm["fringed"] / mm_per_px
        ),
        "crossover_envelope_mm": cross_mm["envelope"],
        "stamp_half_px": float(half),
        "r_out_px": (
            wing.r_out(flux_e / wing.flux_norm(float(half)), floor_e)
            if wing is not None and floor_e
            else None
        ),
        "floor_frac": (floor_e / flux_e) if floor_e and flux_e > 0 else None,
        "flux_e": flux_e,
        "wavelength_nm": float(field.params["wavelength_nm"]),
        "mm_per_px": mm_per_px,
    }


# --------------------------------------------------------------------------- #
# Reproducibility                                                              #
# --------------------------------------------------------------------------- #

#: Everything that changes the rendered PSF, in the order a call reads best.
_REPRO_KEYS = (
    "ra", "dec", "sensorfilter", "focus", "exptime", "n_reads", "pa",
    "shape", "stamp_npix", "oversample", "wavelength_nm", "jitter_sigma_mas",
    "scatter", "scatter_fraction", "wings", "wing_floor_sigma", "chromatic",
    "add_noise", "seed",
)


def reproduction_settings(field):
    """The `simulate_field` arguments needed to reproduce this figure."""
    p = field.params
    out = {}
    for key in _REPRO_KEYS:
        if key == "shape":
            out["shape"] = (int(p["ny"]), int(p["nx"]))
        elif key in p:
            out[key] = p[key]
    return out


def reproduction_call(field, width=4):
    """A pasteable `simulate_field(...)` snippet for this figure.

    The catalog is not included: a synthetic scene cannot be recovered from
    the header, so the source list is described separately.
    """
    items = []
    for key, value in reproduction_settings(field).items():
        if isinstance(value, str):
            items.append(f"{key}={value!r}")
        elif isinstance(value, (bool, type(None))):
            items.append(f"{key}={value}")
        elif isinstance(value, tuple):
            items.append(f"{key}={value}")
        elif isinstance(value, float):
            # repr, not %g: %g would round scatter_fraction to 0.00554214 and
            # the snippet would no longer reproduce the figure it sits under
            items.append(f"{key}={value!r}")
        else:
            items.append(f"{key}={value}")
    rows = [", ".join(items[i:i + width]) for i in range(0, len(items), width)]
    body = ",\n    ".join(rows)
    return f"simulate_field(\n    {body},\n)"


# --------------------------------------------------------------------------- #
# Axis helpers                                                                 #
# --------------------------------------------------------------------------- #

def _radius_conversions(field):
    """(px -> mm, px -> arcsec) for this detector."""
    mm_per_px = float(field.params["pixel_size_um"]) / 1000.0
    arcsec_per_px = float(field.params["plate_scale_mas"]) / 1000.0
    return (lambda r: np.asarray(r) * mm_per_px,
            lambda r: np.asarray(r) * arcsec_per_px)


def _tidy_number(v):
    """Compact tick label: 1, 2, 30, 0.5, 0.02 -- never 2.0000000001."""
    if v <= 0:
        return ""
    if v >= 1:
        return f"{v:,.0f}"
    return f"{v:g}"


def _dense_log_ticks(axis, subs=(2.0, 3.0, 5.0), label_minor=True,
                     mathtext=False):
    """Decade majors plus labelled 2/3/5 minors, so the reader can interpolate."""
    import matplotlib.ticker as ticker

    fmt = (
        ticker.LogFormatterMathtext()
        if mathtext
        else ticker.FuncFormatter(lambda v, _pos: _tidy_number(v))
    )
    axis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=20))
    axis.set_major_formatter(fmt)
    axis.set_minor_locator(ticker.LogLocator(base=10.0, subs=subs, numticks=40))
    axis.set_minor_formatter(
        ticker.FuncFormatter(lambda v, _pos: _tidy_number(v))
        if label_minor
        else ticker.NullFormatter()
    )


def _ticks_outward(ax):
    ax.tick_params(which="both", direction="out", top=False, right=False)


# --------------------------------------------------------------------------- #
# Panels                                                                       #
# --------------------------------------------------------------------------- #

def _block_max(a, factor):
    """Downsample by taking the max of each factor x factor block.

    Max, not mean: a 61-Mpix frame has to be reduced ~8x to display, and
    averaging would wash the star and the inner halo out of the picture.
    """
    if factor <= 1:
        return a
    ny, nx = a.shape
    ny, nx = (ny // factor) * factor, (nx // factor) * factor
    return a[:ny, :nx].reshape(
        ny // factor, factor, nx // factor, factor
    ).max(axis=(1, 3))


#: Geometry of the slots hung around the full-frame image, as fractions of
#: the image axes itself -- percentages, not inches, so the saturation panel
#: below can be given an identically shaped box and come out at exactly the
#: same size whatever the array's aspect ratio (see `_FRAME_ROW`).
_SIDE_FRAC, _PAD_FRAC = 0.26, 0.03
_CBAR_FRAC, _CBAR_PAD_FRAC = 0.04, 0.03
_SIDE_SIZE, _SIDE_PAD = f"{_SIDE_FRAC:.0%}", f"{_PAD_FRAC:.0%}"
_CBAR_SIZE, _CBAR_PAD = f"{_CBAR_FRAC:.0%}", f"{_CBAR_PAD_FRAC:.0%}"

#: Height ratio of the frame's row to the mask's. The frame's row also has
#: to hold the horizontal cut, so it is taller by exactly that slot; the two
#: image boxes then match, and the mask needs no invisible cut slot of its
#: own pushing it down the page.
_FRAME_ROW = 1.0 + _SIDE_FRAC + _PAD_FRAC


def _side_slots(ax, share=True, anchor=None, top=True):
    """Append the (top cut, right cut, colorbar) slots around an image axes.

    `anchor` pins the whole group inside its cell -- "S" for the frame, "N"
    for the mask below it, so the two sit together instead of drifting to
    the middle of their rows when the aspect leaves slack. The right-hand
    slots are what set an aspect-locked image's size, so a panel that only
    has to *match* one reserves those and skips the top (`top=False`).
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    div = make_axes_locatable(ax)
    if anchor is not None:
        div.set_anchor(anchor)
    slots = []
    if top:
        slots.append(div.append_axes("top", size=_SIDE_SIZE, pad=_SIDE_PAD,
                                     sharex=ax if share else None))
    right = div.append_axes("right", size=_SIDE_SIZE, pad=_SIDE_PAD,
                            sharey=ax if share else None)
    cax = div.append_axes("right", size=_CBAR_SIZE, pad=_CBAR_PAD)
    return (*slots, right, cax)


def _image_panel(fig, ax, image, extent, title, background, cbar=True):
    """Log-normalized image with a colorbar in electrons."""
    from matplotlib.colors import LogNorm

    vmin = max(float(background) * 0.9, 1e-3)
    vmax = max(float(np.nanmax(image)), vmin * 10.0)
    im = ax.imshow(np.maximum(image, vmin), origin="lower", cmap="magma",
                   norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent,
                   interpolation="nearest")
    ax.set(xlabel="x [px]", ylabel="y [px]")
    ax.set_title(title, fontsize=9)
    ax.grid(False)
    _ticks_outward(ax)
    if cbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("e$^-$", fontsize=8)
        cb.ax.tick_params(labelsize=7, direction="out")
    return im


def _plot_full_frame(fig, ax, field, background):
    """The whole detector, block-reduced, with cuts through the centre.

    The two thin panels are the detector row and column through the array
    centre at native resolution -- above the image for the horizontal cut,
    to its right for the vertical one -- on the same log stretch as the
    image, so a source that is only a smudge in the block-reduced frame can
    be read off in electrons. The dashed grey lines mark where the cuts sit.
    The panels hang off an axes divider rather than a sub-gridspec so they
    track the image's aspect-locked box instead of the whole cell.
    """
    from matplotlib import ticker

    img = np.asarray(field.image_clean, dtype=float)
    ny, nx = img.shape
    factor = max(1, int(np.ceil(max(ny, nx) / 900.0)))

    im = _image_panel(fig, ax, _block_max(img, factor), [0, nx, 0, ny],
                      "", background, cbar=False)
    iy, ix = ny // 2, nx // 2
    ax.axhline(iy + 0.5, color="0.72", ls="--", lw=0.7)
    ax.axvline(ix + 0.5, color="0.72", ls="--", lw=0.7)

    ax_top, ax_right, cax = _side_slots(ax, anchor="S")

    vmin, vmax = float(im.norm.vmin), float(im.norm.vmax)
    ax_top.plot(np.arange(nx) + 0.5, np.maximum(img[iy, :], vmin), "-",
                color=_DARK, lw=0.6)
    ax_right.plot(np.maximum(img[:, ix], vmin), np.arange(ny) + 0.5, "-",
                  color=_DARK, lw=0.6)
    ax_top.set(yscale="log", ylim=(vmin, vmax * 4.0))
    ax_right.set(xscale="log", xlim=(vmin, vmax * 4.0))
    ax_top.set_ylabel("cut [e$^-$]", fontsize=7, labelpad=1.5)
    ax_right.set_xlabel("cut [e$^-$]", fontsize=7, labelpad=1.5)
    ax_top.set_title(
        f"full frame {nx} x {ny} px  ({factor}x block max)\n"
        f"centre cuts at x = {ix:,} px, y = {iy:,} px", fontsize=8.5,
    )
    for cut, axis in ((ax_top, ax_top.yaxis), (ax_right, ax_right.xaxis)):
        cut.grid(False)
        _ticks_outward(cut)
        cut.tick_params(which="major", labelsize=6, length=2.5)
        cut.tick_params(which="minor", length=1.5)
        axis.set_major_locator(ticker.LogLocator(numticks=4))
        axis.set_minor_locator(
            ticker.LogLocator(subs=tuple(range(2, 10)), numticks=40)
        )
        axis.set_major_formatter(ticker.LogFormatterMathtext())
        axis.set_minor_formatter(ticker.NullFormatter())
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False, labelrotation=90)

    cb = fig.colorbar(im, cax=cax)
    # units as a title, not a rotated label: the label would stick out into
    # the neighbouring panel's y-axis
    cax.set_title("e$^-$", fontsize=7, pad=3.0)
    cb.ax.tick_params(labelsize=6, direction="out")
    cb.ax.yaxis.set_major_locator(ticker.LogLocator(numticks=5))
    cb.ax.yaxis.set_minor_locator(ticker.NullLocator())


def _plot_zoom(fig, ax, field, dec, background, half_width=300):
    """A zoom on the star, at native resolution."""
    img = field.image_clean
    ny, nx = img.shape
    xc = int(round(float(field.catalog["x"][0])))
    yc = int(round(float(field.catalog["y"][0])))
    x0, x1 = max(xc - half_width, 0), min(xc + half_width, nx)
    y0, y1 = max(yc - half_width, 0), min(yc + half_width, ny)
    # No colorbar: the stretch is computed the same way as the full frame's
    # and the star's peak sets vmax in both, so it would be the same bar.
    _image_panel(
        fig, ax, np.asarray(img[y0:y1, x0:x1], dtype=float),
        [x0 - xc, x1 - xc, y0 - yc, y1 - yc],
        f"zoom, +/-{half_width} px at native scale", background, cbar=False,
    )
    ax.set(xlabel="$\\Delta x$ [px]", ylabel="$\\Delta y$ [px]")


def _plot_satmask(fig, ax, field, params):
    """Where the frame saturated, block-reduced with OR so nothing vanishes.

    Drawn directly under the full frame at the same size: the same block
    factor and extent, and the same (here invisible) cut and colorbar slots
    reserved around it, so the two panels are pixel-for-pixel comparable.
    """
    mask = np.asarray(field.saturation_mask, dtype=bool)
    ny, nx = mask.shape
    factor = max(1, int(np.ceil(max(ny, nx) / 900.0)))
    shown = _block_max(mask.astype(np.uint8), factor).astype(bool)
    ax.imshow(shown, origin="lower", cmap="gray_r", vmin=0, vmax=1,
              extent=[0, nx, 0, ny], interpolation="nearest")
    for slot in _side_slots(ax, share=False, anchor="N", top=False):
        slot.set_visible(False)
    n = int(mask.sum())
    r_equiv = np.sqrt(n / np.pi)
    ax.set(xlabel="x [px]", ylabel="y [px]")
    ax.set_title(
        f"saturation mask: {n:,} px "
        f"({100.0 * n / mask.size:.3f}%), r$_{{equiv}}$ = {r_equiv:,.0f} px",
        fontsize=9,
    )
    ax.grid(False)
    _ticks_outward(ax)
    return n, r_equiv


def _plot_profile(fig, ax, dec, field, params):
    """Log-log radial profile with the two model terms and the render."""
    r = dec["r_px"]
    ok = np.isfinite(dec["measured"]) & (dec["measured"] > 0)
    if ok.any():
        ax.plot(r[ok], dec["measured"][ok], "o", ms=3.0, color=_AMBER,
                mec="none", label="rendered image", zorder=5)
    pos = dec["airy"] > 0
    ax.plot(r[pos], dec["airy"][pos], "-", color=_TEAL, label="Airy core")
    if dec["contrast"] is not None:
        ax.plot(r, dec["scatter"], "-", color=_RED, label="scatter halo")
        ax.plot(r, dec["total"], "--", color=_DARK, lw=1.4, label="total")

    ax.set(xscale="log", yscale="log",
           xlabel="radius [detector pixels]",
           ylabel="PSF fraction per pixel")
    finite = dec["total"][dec["total"] > 0]
    if finite.size:
        ax.set_ylim(finite.min() * 0.2, dec["airy"].max() * 5.0)
    y_lo, y_hi = ax.get_ylim()
    ax.set_xlim(r[0] * 0.8, r[-1] * 1.3)
    x_lo, x_hi = ax.get_xlim()

    for x, text, style in (
        (dec["stamp_half_px"], "stamp edge", ":"),
        (dec["crossover_px"], "crossover", "--"),
        (dec["r_out_px"], "drawn to", "--"),
    ):
        if x is None or not (x_lo < x < x_hi):
            continue
        ax.axvline(x, color="0.5", ls=style, lw=1.0)
        ax.text(x, 0.02, f" {text}", transform=ax.get_xaxis_transform(),
                rotation=90, va="bottom", ha="left", fontsize=7, color="0.35")
    if dec["floor_frac"] and y_lo < dec["floor_frac"] < y_hi:
        ax.axhline(dec["floor_frac"], color="0.4", ls="--", lw=1.0)
        ax.text(0.985, dec["floor_frac"],
                f"{field.params.get('wing_floor_sigma', 0.1):g}"
                "$\\sigma$ draw floor ",
                transform=ax.get_yaxis_transform(), va="bottom", ha="right",
                fontsize=7, color="0.35")
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(x_lo, x_hi)

    _dense_log_ticks(ax.xaxis)
    _dense_log_ticks(ax.yaxis, label_minor=False, mathtext=True)
    _ticks_outward(ax)
    ax.tick_params(which="minor", labelsize=6.5)

    # the same radius in focal-plane mm and on-sky arcsec
    to_mm, to_arcsec = _radius_conversions(field)
    mm_per_px = to_mm(1.0)
    arcsec_per_px = to_arcsec(1.0)
    for offset, label, scale in (
        (0, "radius [arcsec on sky]", arcsec_per_px),
        (34, "radius [mm at the focal plane]", mm_per_px),
    ):
        sec = ax.secondary_xaxis(
            "top", functions=(lambda v, s=scale: v * s,
                              lambda v, s=scale: v / s),
        )
        sec.set_xlabel(label, fontsize=8, labelpad=1.5)
        _dense_log_ticks(sec.xaxis)
        sec.tick_params(which="both", direction="out", labelsize=7)
        sec.tick_params(which="minor", labelsize=6)
        if offset:
            sec.spines["top"].set_position(("outward", offset))

    ax.legend(loc="lower left", fontsize=8, ncol=2,
              title="azimuthally averaged", title_fontsize=8)


# --------------------------------------------------------------------------- #
# Summary block                                                                #
# --------------------------------------------------------------------------- #

def _px(value):
    return "n/a" if value is None else f"{value:,.0f} px"


def _mm(value):
    return "n/a" if value is None else f"{value:.2f} mm"


def _summary_columns(dec, field, params):
    """(left, right) blocks of text for the footer."""
    ny, nx = field.image_clean.shape
    mag = (
        float(field.catalog["phot_g_mean_mag"][0]) if len(field.catalog) else None
    )
    sat = int(field.saturation_mask.sum())
    left = [
        "FRAME",
        f"  array            {nx:,} x {ny:,} px "
        f"= {params['n_pixels'] / 1e6:.2f} Mpix",
        f"  pixel pitch      {params['pixel_size_um']:.2f} um "
        f"= {params['plate_scale_mas']:.3f} mas/px",
        f"  field of view    {nx * params['plate_scale_mas'] / 1000 / 60:.1f}"
        f" x {ny * params['plate_scale_mas'] / 1000 / 60:.1f} arcmin",
        f"  saturated        {sat:,} px "
        f"({100.0 * sat / params['n_pixels']:.3f}%), "
        f"r_equiv {np.sqrt(sat / np.pi):,.0f} px",
        "",
        "EXPOSURE",
        f"  exposure time    {params['exptime']:.1f} s"
        f"   x {params['n_reads']} read(s)",
        f"  background       {params['sky_e_s']:.4f} e-/s/px zodi"
        f" + {params['dark_e_s']:.4f} e-/s/px dark",
        f"  noise applied    {params['add_noise']}"
        f"      seed {params['seed']}",
        "",
        "SOURCE",
        f"  {params['n_sources']} star"
        + (f", G = {mag:.2f}" if mag is not None else "")
        + f", {dec['flux_e']:.4g} e- total",
    ]
    right = [
        "PSF",
        f"  sensorfilter     {params['sensorfilter']}"
        f"   focus {params['focus']} wave",
        f"  Airy wavelength  {dec['wavelength_nm']:.1f} nm"
        "   (PSF geometry only; the bandpass sets the flux)",
        f"  jitter           {params['jitter_sigma_mas']:.1f} mas",
        f"  stamp            {params['stamp_npix']} px "
        f"at {params['oversample']}x oversampling",
        f"  diffraction wing c={params['wing_c']:.4g}, "
        f"alpha={params['wing_alpha']:.3f}, "
        f"floor {params['wing_floor_e']:.3g} e-/px "
        f"= {params.get('wing_floor_sigma', 0.1):g} sigma",
        "",
        "SCATTERED LIGHT",
    ]
    if dec["contrast"] is None:
        right.append("  OFF")
    else:
        right += [
            f"  Airy peak        {dec['airy_peak_irradiance']:.4e}"
            " W/mm^2 per W",
            f"  scatter peak     {dec['scatter_peak_irradiance']:.4e}"
            " W/mm^2 per W",
            f"  peak contrast    {dec['contrast']:.3e}",
            f"  fraction         {params['scatter_fraction']:.4e}"
            f"   modelled to {_px(params['scatter_reach_px'])}"
            f"   drawn to {_px(dec['r_out_px'])}",
            f"  crossover        {_mm(dec['crossover_mm'])} azimuthal mean"
            f"   |   {_mm(dec['crossover_fringed_mm'])} fringed (ring maxima)",
            f"  map              {params['scatter_file']}",
        ]
    return "\n".join(left), "\n".join(right)


# --------------------------------------------------------------------------- #
# Assembly                                                                     #
# --------------------------------------------------------------------------- #

def build_psf_figure(field, zoom_half_width=300):
    """The report figure and its decomposition, without writing anything."""
    from matplotlib.figure import Figure

    dec = psf_decomposition(field)
    params = dict(field.params)
    background = float(field.models["sky_dark_e"])

    with _style_context():
        # The frame and its saturation mask share the left column, one above
        # the other, with row heights and reserved side slots chosen so the
        # two images come out the same size (see _FRAME_ROW). Everything
        # else lives to their right.
        fig = Figure(figsize=(18.0, 11.0))
        grid = fig.add_gridspec(
            2, 3, width_ratios=[1.58, 0.50, 0.94],
            height_ratios=[_FRAME_ROW, 1.0], hspace=0.15, wspace=0.124, left=0.04, right=0.985,
            top=0.91, bottom=0.05,
        )
        _plot_full_frame(fig, fig.add_subplot(grid[0, 0]), field,
                         background)
        _plot_zoom(fig, fig.add_subplot(grid[0, 1]), field, dec, background,
                   half_width=zoom_half_width)
        _plot_profile(fig, fig.add_subplot(grid[0, 2]), dec, field, params)
        _plot_satmask(fig, fig.add_subplot(grid[1, 0]), field, params)

        left_text, right_text = _summary_columns(dec, field, params)
        ax = fig.add_subplot(grid[1, 1:])
        ax.axis("off")
        opts = dict(va="top", ha="left", fontsize=7.2, family="monospace",
                    transform=ax.transAxes)
        # one block, not two stacked at fixed fractions: the axes height
        # changes with the layout and a fixed second anchor drifts away
        ax.text(0.0, 1.0, left_text + "\n\n" + right_text, color="0.15",
                **opts)
        ax.text(0.48, 1.0, "TO REPRODUCE\n" + reproduction_call(field, width=3),
                color="0.35", **opts)

        fig.suptitle(
            "WCC PSF: Airy core and scattered-light halo", fontsize=13, y=0.975
        )
    return fig, dec


def make_psf_report(field, path="wcc_sim_psf_report.pdf", dpi=200,
                    zoom_half_width=300):
    """Write the PSF radial-profile report as PDF and PNG.

    Returns the two paths written.
    """
    fig, _ = build_psf_figure(field, zoom_half_width=zoom_half_width)
    base, ext = os.path.splitext(path)
    paths = (path if ext else base + ".pdf", base + ".png")
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=dpi)
    return paths
