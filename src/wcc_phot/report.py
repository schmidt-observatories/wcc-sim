"""One-page PDF + PNG diagnostic report for a photometry run.

`make_report(result, frame=..., path=...)` renders the field with the
apertures, the relative light curve, per-star raw fluxes, centroid drifts,
annulus backgrounds, and two noise diagnostics (binned RMS vs bin size,
per-star RMS vs G against the predicted errors) into a single figure,
written as both PDF and PNG. `run_photometry(..., report=path)` and the
CLI `--report` flag call it with the first frame of the series.

Rendering uses a plain (pyplot-free) matplotlib Figure, so it works
headless and never opens a window.
"""

import contextlib
import os

import numpy as np

from . import flags as flagbits
from .live import REF_COLOR, TARGET_COLOR

_FLAG_NAMES = {
    flagbits.FLAG_CENTROID: "centroid",
    flagbits.FLAG_SATURATED: "saturated",
    flagbits.FLAG_EDGE: "edge",
}


def _style_context():
    """The 'gks' house style when installed, matplotlib defaults otherwise."""
    import matplotlib.style

    # style.context() is lazy — an unknown style raises OSError on __enter__,
    # not here — so check availability up front rather than try/except.
    if "gks" in matplotlib.style.available:
        return matplotlib.style.context("gks")
    return contextlib.nullcontext()


def _star_series(measurements, star):
    """Per-frame rows of one star, ordered by frame."""
    rows = measurements[measurements["star"] == star]
    return rows[np.argsort(rows["frame"])]


def _ref_relative_flux(series, lc):
    """A reference star's own differential LC against the other refs.

    rel_j = flux_j / (ensemble - flux_j), with the star's contribution to
    the ensemble error removed; both normalized by the median over frames.
    """
    flux = np.asarray(series["flux_e"], dtype=float)
    err = np.asarray(series["flux_err_e"], dtype=float)
    ens = np.asarray(lc["flux_ens"], dtype=float) - flux
    ens_err = np.sqrt(
        np.clip(np.asarray(lc["flux_ens_err"], dtype=float) ** 2 - err**2, 0, None)
    )
    rel = flux / ens
    rel_err = np.abs(rel) * np.hypot(err / flux, ens_err / ens)
    median = np.median(rel)
    return rel / median, rel_err / median


def _binned_rms(values):
    """(bin sizes, RMS of bin means) for consecutive-point binning."""
    values = np.asarray(values, dtype=float)
    n = values.size
    sizes = [s for s in range(1, n // 3 + 1) if n // s >= 3]
    if not sizes:
        sizes = [1]
    rms = []
    for s in sizes:
        nbin = n // s
        means = values[: nbin * s].reshape(nbin, s).mean(axis=1)
        rms.append(float(np.std(means)))
    return np.asarray(sizes), np.asarray(rms)


def compute_metrics(result):
    """Noise/quality numbers shown in the report's summary panel.

    Returns a dict: rel-flux RMS and MAD-RMS, median predicted error and
    the RMS/prediction ratio, target centroid RMS, per-star measured RMS
    and predicted errors (target first, refs in star order), and per-bit
    flagged-measurement counts.
    """
    lc = result.lightcurve
    meas = result.measurements
    stars = result.stars

    rel = np.asarray(lc["rel_flux_norm"], dtype=float)
    rel_err = np.asarray(lc["rel_flux_norm_err"], dtype=float)
    rms = float(np.std(rel))
    mad_rms = float(1.4826 * np.median(np.abs(rel - np.median(rel))))
    med_err = float(np.median(rel_err))

    star_rms, star_err = [rms], [med_err]
    for star in np.asarray(stars["star"])[1:]:
        series = _star_series(meas, star)
        if len(stars) > 2:
            rel_j, err_j = _ref_relative_flux(series, lc)
        else:  # single reference: its own LC is just the mirrored target LC
            rel_j, err_j = 1.0 / rel, rel_err / rel**2
        star_rms.append(float(np.std(rel_j)))
        star_err.append(float(np.median(err_j)))

    tgt = _star_series(meas, 0)
    cx = np.asarray(tgt["x"], dtype=float)
    cy = np.asarray(tgt["y"], dtype=float)

    flag_counts = {
        name: int(np.count_nonzero(np.asarray(meas["flags"]) & bit))
        for bit, name in _FLAG_NAMES.items()
    }
    return {
        "rms": rms,
        "mad_rms": mad_rms,
        "median_err": med_err,
        "rms_over_err": rms / med_err if med_err > 0 else np.nan,
        "centroid_rms_x": float(np.std(cx)),
        "centroid_rms_y": float(np.std(cy)),
        "star_rms": np.asarray(star_rms),
        "star_err": np.asarray(star_err),
        "flag_counts": flag_counts,
    }


def _time_axis(lc):
    """(times, axis label); 'frame' when times are just the frame index."""
    t = np.asarray(lc["time"], dtype=float)
    if np.array_equal(t, np.arange(t.size, dtype=float)):
        return t, "frame"
    return t, "time"


def _plot_field(ax, frame, result):
    from astropy.visualization import simple_norm
    from matplotlib.patches import Circle

    geom_r_out = result.params["r_out"]
    r_ap = result.params["r_ap"]
    meas = result.measurements
    stars = result.stars

    x_med = np.array([np.median(_star_series(meas, s)["x"]) for s in stars["star"]])
    y_med = np.array([np.median(_star_series(meas, s)["y"]) for s in stars["star"]])

    ny, nx = frame.image_e.shape
    pad = max(2.5 * geom_r_out, 100.0)
    x0 = int(np.clip(x_med.min() - pad, 0, nx - 1))
    x1 = int(np.clip(x_med.max() + pad, 1, nx))
    y0 = int(np.clip(y_med.min() - pad, 0, ny - 1))
    y1 = int(np.clip(y_med.max() + pad, 1, ny))
    cut = frame.image_e[y0:y1, x0:x1]

    norm = simple_norm(cut, "asinh", percent=99.5)
    ax.imshow(
        cut, origin="lower", cmap="gray_r", norm=norm,
        extent=(x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5),
    )
    for j, (x, y) in enumerate(zip(x_med, y_med)):
        color = TARGET_COLOR if j == 0 else REF_COLOR
        ax.add_patch(Circle((x, y), r_ap, fill=False, color=color, lw=1.4))
        if j == 0:
            for radius in (result.params["r_in"], geom_r_out):
                ax.add_patch(Circle((x, y), radius, fill=False, color=color,
                                    lw=0.8, ls="--", alpha=0.7))
        label = "T" if j == 0 else f"R{j}"
        # keep the label inside the crop: below the star when near the top
        # edge, x clamped away from the sides
        above = y + geom_r_out < y1 - 0.12 * (y1 - y0)
        x_lab = float(np.clip(x, x0 + 0.05 * (x1 - x0), x1 - 0.05 * (x1 - x0)))
        y_lab = y + geom_r_out if above else y - geom_r_out
        ax.annotate(
            f"{label}  G={stars['gmag'][j]:.1f}",
            (x_lab, y_lab), xytext=(0, 3 if above else -3),
            textcoords="offset points",
            ha="center", va="bottom" if above else "top",
            fontsize=8, color=color,
        )
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.set_title("first frame, apertures at median centroids", fontsize=9)


def _plot_lightcurve(ax, lc, t, tlabel):
    rel = np.asarray(lc["rel_flux_norm"], dtype=float)
    err = np.asarray(lc["rel_flux_norm_err"], dtype=float)
    ax.errorbar(t, rel, yerr=err, fmt="o", color=REF_COLOR, ms=3.5,
                lw=1, capsize=2)
    ax.axhline(1.0, color="0.4", ls="--", lw=0.8)
    ax.set_xlabel(tlabel)
    ax.set_ylabel("relative flux (norm.)")
    ax.set_title("target / reference ensemble", fontsize=9)


def _plot_binned_rms(ax, lc):
    rel = np.asarray(lc["rel_flux_norm"], dtype=float)
    sizes, rms = _binned_rms(rel)
    ax.loglog(sizes, 1e3 * rms, "o-", color=REF_COLOR, ms=3.5, lw=1.2,
              label="measured")
    ax.loglog(sizes, 1e3 * rms[0] / np.sqrt(sizes), color="0.4", ls="--",
              lw=1, label=r"white noise $\propto n^{-1/2}$")
    ax.set_xlabel("bin size [points]")
    ax.set_ylabel("RMS of bin means [ppt]")
    ax.set_title("binned RMS", fontsize=9)
    ax.legend(fontsize=7)


def _plot_raw_fluxes(ax, result, t, tlabel):
    lc = result.lightcurve
    meas = result.measurements
    for star in np.asarray(result.stars["star"])[1:]:
        flux = np.asarray(_star_series(meas, star)["flux_e"], dtype=float)
        ax.plot(t, flux / np.median(flux), color=REF_COLOR, lw=0.8, alpha=0.4)
    ens = np.asarray(lc["flux_ens"], dtype=float)
    ax.plot(t, ens / np.median(ens), color=REF_COLOR, lw=1.6, label="ensemble")
    tgt = np.asarray(lc["flux_target"], dtype=float)
    ax.plot(t, tgt / np.median(tgt), color=TARGET_COLOR, lw=1.6, label="target")
    ax.set_xlabel(tlabel)
    ax.set_ylabel("flux / median")
    ax.set_title("raw fluxes (thin: individual refs)", fontsize=9)
    ax.legend(fontsize=7)


def _plot_rms_vs_mag(ax, result, metrics):
    gmag = np.asarray(result.stars["gmag"], dtype=float)
    rms = 1e3 * metrics["star_rms"]
    err = 1e3 * metrics["star_err"]
    ax.semilogy(gmag[1:], rms[1:], "o", color=REF_COLOR, label="refs measured")
    ax.semilogy(gmag[:1], rms[:1], "o", color=TARGET_COLOR, label="target measured")
    ax.semilogy(gmag, err, "d", mfc="none", color="0.35", label="predicted")
    ax.set_xlabel("G [mag]")
    ax.set_ylabel("per-point RMS [ppt]")
    ax.set_title("noise vs magnitude", fontsize=9)
    ax.legend(fontsize=7)


def _plot_centroids(ax, result, t, tlabel, coord):
    meas = result.measurements
    for star in np.asarray(result.stars["star"]):
        series = _star_series(meas, star)
        vals = np.asarray(series[coord], dtype=float)
        vals -= np.median(vals)
        if star == 0:
            ax.plot(t, vals, color=TARGET_COLOR, lw=1.4, label="target")
        else:
            ax.plot(t, vals, color=REF_COLOR, lw=0.8, alpha=0.5)
    ax.axhline(0.0, color="0.4", ls="--", lw=0.8)
    ax.set_xlabel(tlabel)
    ax.set_ylabel(rf"$\Delta${coord} [px]")
    ax.set_title(f"{coord} centroid drift (thin: refs)", fontsize=9)
    ax.legend(fontsize=7)


def _plot_background(ax, result, t, tlabel):
    meas = result.measurements
    for star in np.asarray(result.stars["star"]):
        series = _star_series(meas, star)
        bkg = np.asarray(series["bkg_e_pix"], dtype=float)
        if star == 0:
            ax.plot(t, bkg, color=TARGET_COLOR, lw=1.4, label="target annulus")
        else:
            ax.plot(t, bkg, color=REF_COLOR, lw=0.8, alpha=0.5)
    ax.set_xlabel(tlabel)
    ax.set_ylabel("background [e-/px]")
    ax.set_title("annulus background (thin: refs)", fontsize=9)
    ax.legend(fontsize=7)


def _summary_text(result, metrics, frame=None):
    params = result.params
    stars = result.stars
    ref_g = np.asarray(stars["gmag"], dtype=float)[1:]
    lines = [
        f"target       {params['target_source_id']}  "
        f"(G = {stars['gmag'][0]:.2f})",
        f"method       {params['method']},  r_ap = {params['r_ap']:.1f} px,  "
        f"annulus {params['r_in']:.1f}-{params['r_out']:.1f} px",
        f"frames       {params['n_frames']} x {params['sensorfilter']},  "
        f"focus +{params['focus']} waves"
        + (
            f",  {float(frame.meta['exptime']):.0f} s"
            if frame is not None
            else ""
        ),
        f"references   {params['n_ref']}"
        + (
            f"  (G = {ref_g.min():.1f}-{ref_g.max():.1f})"
            if ref_g.size
            else ""
        ),
        "",
        f"rel-flux RMS        {1e3 * metrics['rms']:.2f} ppt",
        f"MAD RMS             {1e3 * metrics['mad_rms']:.2f} ppt",
        f"median pred. error  {1e3 * metrics['median_err']:.2f} ppt",
        f"RMS / prediction    {metrics['rms_over_err']:.2f}",
        f"centroid RMS (T)    x {metrics['centroid_rms_x']:.2f} px,  "
        f"y {metrics['centroid_rms_y']:.2f} px",
        "flagged             "
        + (
            ",  ".join(
                f"{name}: {count}"
                for name, count in metrics["flag_counts"].items()
                if count
            )
            or "none"
        ),
    ]
    return "\n".join(lines)


def make_report(result, frame=None, path="wcc_phot_report.pdf"):
    """Write the diagnostic report figure for a PhotometryResult.

    `frame` (optional wcc-sim FITS path, HDUList, SimulatedField, or
    Frame) fills the field panel; without it that panel is left empty.
    `path` may end in .pdf or .png — both siblings are written either
    way. Returns the list of written paths.
    """
    from matplotlib.figure import Figure

    from .io import load_frame

    if frame is not None:
        frame = load_frame(frame)
    metrics = compute_metrics(result)
    lc = result.lightcurve
    t, tlabel = _time_axis(lc)

    with _style_context():
        fig = Figure(figsize=(11, 14))
        grid = fig.add_gridspec(
            5, 2, height_ratios=[1.5, 1, 1, 1, 1],
            hspace=0.42, wspace=0.28,
            left=0.08, right=0.97, top=0.93, bottom=0.05,
        )

        ax_field = fig.add_subplot(grid[0, :])
        if frame is not None:
            _plot_field(ax_field, frame, result)
        else:
            ax_field.text(0.5, 0.5, "no frame provided", ha="center",
                          va="center", color="0.5")
            ax_field.set_axis_off()

        _plot_lightcurve(fig.add_subplot(grid[1, 0]), lc, t, tlabel)
        _plot_binned_rms(fig.add_subplot(grid[1, 1]), lc)
        _plot_raw_fluxes(fig.add_subplot(grid[2, 0]), result, t, tlabel)
        _plot_rms_vs_mag(fig.add_subplot(grid[2, 1]), result, metrics)
        _plot_centroids(fig.add_subplot(grid[3, 0]), result, t, tlabel, "x")
        _plot_centroids(fig.add_subplot(grid[3, 1]), result, t, tlabel, "y")
        _plot_background(fig.add_subplot(grid[4, 0]), result, t, tlabel)

        ax_text = fig.add_subplot(grid[4, 1])
        ax_text.set_axis_off()
        ax_text.text(0.0, 1.0, _summary_text(result, metrics, frame=frame),
                     ha="left", va="top", fontsize=9, family="monospace",
                     transform=ax_text.transAxes)

        fig.suptitle(
            f"wcc-phot report — target {result.params['target_source_id']}, "
            f"{result.params['sensorfilter']}, "
            f"{result.params['method']} photometry, "
            f"{result.params['n_frames']} frames",
            fontsize=13,
        )

        stem = os.path.splitext(path)[0]
        paths = [stem + ".pdf", stem + ".png"]
        fig.savefig(paths[0])
        fig.savefig(paths[1], dpi=200)
    return paths
