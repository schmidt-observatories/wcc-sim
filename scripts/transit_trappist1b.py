"""End-to-end TRAPPIST-1b transit simulation + photometry validation.

Run:  $PY scripts/transit_trappist1b.py [--live]
(network needed on first run for Gaia; --live opens a matplotlib window
that shows each frame + apertures and the growing light curve as the
photometry runs, and keeps it open at the end)

Injects the TRAPPIST-1b transit (lazuli_transit / NASA archive parameters)
into the Gaia catalog magnitude of TRAPPIST-1, simulates 36 x 300 s WCC
frames (zwo:i, +2 waves defocus) spanning 3 h centered on the transit,
runs wcc_phot differential aperture photometry against the reference
ensemble, and checks that the recovered light curve matches the injected
model: depth-scale factor consistent with 1, chi2/dof reasonable, and
out-of-transit scatter consistent with the error bars.

The frame is a full-width strip (2200 x 9568 px, 161" x 37") offset
(+13", +11.5") from the target so the 4 best reference stars
(G = 14.7-18.5) land in-frame; the Gaia field around TRAPPIST-1 is sparse,
so expect a UserWarning that fewer than 10 references are usable.

Writes frames, phot FITS, and a summary figure to ./transit_out/;
exits nonzero if any check fails.
"""

import argparse
import os
import sys
import time as walltime
import warnings

import numpy as np

from lazuli_transit import TransitModel
from wcc_sim import simulate_field
from wcc_sim.catalog import query_gaia
from wcc_phot import run_photometry

# TRAPPIST-1, Gaia DR3 epoch-2016 position
TARGET_ID = 2635476908753563008
RA_T1, DEC_T1 = 346.6264, -5.0435
# pointing offset [arcsec] that centers the strip on the target + best refs
DX_AS, DY_AS = 13.0, 11.5

SENSORFILTER = "zwo:i"
FOCUS = 2  # +2 waves defocus
EXPTIME = 300.0  # s
N_FRAMES = 36  # 3 h total
SHAPE = (2200, 9568)  # full-width strip
OUT = "transit_out"

# exposure mid-times [d], transit centered at t = 0
TIMES_D = ((np.arange(N_FRAMES) + 0.5) * EXPTIME - N_FRAMES * EXPTIME / 2) / 86400.0


def exposure_averaged_flux(model, times_d, exptime_s, n_sub=11):
    """Model relative flux averaged over each exposure window."""
    offsets = (np.linspace(-0.5, 0.5, n_sub) * exptime_s) / 86400.0
    grid = (times_d[:, None] + offsets[None, :]).ravel()  # batman wants 1-D
    return model.relative_flux(grid).reshape(times_d.size, n_sub).mean(axis=1)


def simulate_frames(catalog, flux):
    """One wcc-sim frame per exposure with the transit injected into the
    target's G magnitude; returns the FITS paths."""
    target_row = np.flatnonzero(
        np.asarray(catalog["source_id"], dtype=np.int64) == TARGET_ID
    )[0]
    ra_c = RA_T1 + DX_AS / 3600.0 / np.cos(np.radians(DEC_T1))
    dec_c = DEC_T1 + DY_AS / 3600.0

    paths = []
    for k, f_k in enumerate(flux):
        cat_k = catalog.copy()
        cat_k["phot_g_mean_mag"][target_row] += -2.5 * np.log10(f_k)
        path = os.path.join(OUT, f"frame_{k:03d}.fits")
        t0 = walltime.time()
        field = simulate_field(
            ra_c,
            dec_c,
            sensorfilter=SENSORFILTER,
            focus=FOCUS,
            exptime=EXPTIME,
            seed=1000 + k,
            shape=SHAPE,
            catalog=cat_k,
            output=path,
            write_clean=False,
        )
        print(
            f"frame {k + 1:2d}/{len(flux)}: t = {TIMES_D[k] * 24:+6.3f} h, "
            f"injected flux = {f_k:.6f}, "
            f"{int(field.saturation_mask.sum())} saturated px, "
            f"{walltime.time() - t0:.1f} s"
        )
        paths.append(path)
    return paths


def verify(lc, model_norm, depth_ppt):
    """(name, value, passed) rows for the three recovery checks."""
    obs = np.asarray(lc["rel_flux_norm"], dtype=float)
    err = np.asarray(lc["rel_flux_norm_err"], dtype=float)

    # depth-scale factor: obs = 1 - alpha * (1 - model_norm)
    d_mod = 1.0 - model_norm
    d_obs = 1.0 - obs
    w = 1.0 / err**2
    alpha = np.sum(w * d_mod * d_obs) / np.sum(w * d_mod**2)
    alpha_err = 1.0 / np.sqrt(np.sum(w * d_mod**2))

    chi2_dof = float(np.sum(((obs - model_norm) / err) ** 2)) / obs.size

    oot = d_mod < 0.1 * np.max(d_mod)
    oot_rms = float(np.std(d_obs[oot]))
    med_err = float(np.median(err))

    return [
        (
            f"depth scale alpha = {alpha:.3f} +/- {alpha_err:.3f} "
            f"(recovered depth {alpha * depth_ppt:.2f} ppt, "
            f"injected {depth_ppt:.2f} ppt)",
            abs(alpha - 1.0) < 3.0 * alpha_err,
        ),
        (f"chi2/dof vs injected model = {chi2_dof:.2f}", 0.5 < chi2_dof < 2.0),
        (
            f"out-of-transit RMS = {oot_rms * 1e3:.2f} ppt "
            f"(median error bar {med_err * 1e3:.2f} ppt)",
            oot_rms < 2.0 * med_err,
        ),
    ]


def plot_lightcurve(lc, model, model_norm, path, interactive=False):
    import matplotlib

    if not interactive:  # keep the backend the live viewer is using
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:  # house style if installed, matplotlib defaults otherwise
        plt.style.use("gks")
    except OSError:
        pass

    t_h = np.asarray(lc["time"], dtype=float) * 24.0
    obs = np.asarray(lc["rel_flux_norm"], dtype=float)
    err = np.asarray(lc["rel_flux_norm_err"], dtype=float)

    t_fine = np.linspace(TIMES_D[0], TIMES_D[-1], 500)
    f_fine = model.relative_flux(t_fine)
    f_fine /= np.median(exposure_averaged_flux(model, TIMES_D, EXPTIME))

    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True, figsize=(7, 5.5),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )
    ax.errorbar(t_h, obs, yerr=err, fmt="o", color="#00798c",
                ms=4, lw=1, capsize=2, label="wcc-phot recovered")
    ax.plot(t_fine * 24.0, f_fine, color="#d1495b", lw=1.5,
            label="injected model (unbinned)")
    ax.plot(t_h, model_norm, "s", color="#d1495b", ms=3, mfc="none",
            label="injected model (300 s bins)")
    ax.set_ylabel("Relative flux")
    ax.legend(loc="lower right")
    ax.set_title(
        f"TRAPPIST-1b, WCC {SENSORFILTER}, +{FOCUS} waves defocus, "
        f"{N_FRAMES} x {EXPTIME:.0f} s"
    )

    axr.errorbar(t_h, 1e3 * (obs - model_norm), yerr=1e3 * err, fmt="o",
                 color="#00798c", ms=4, lw=1, capsize=2)
    axr.axhline(0, color="0.4", ls="--", lw=1)
    axr.set_xlabel("Time from mid-transit [h]")
    axr.set_ylabel("Resid. [ppt]")

    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--live", action="store_true",
        help="live matplotlib view while the photometry runs",
    )
    args = parser.parse_args(argv)

    os.makedirs(OUT, exist_ok=True)

    model = TransitModel.from_planet("TRAPPIST-1 b")
    model.t0 = 0.0
    flux = exposure_averaged_flux(model, TIMES_D, EXPTIME)
    depth_ppt = 1e3 * (1.0 - flux.min())
    print(
        f"TRAPPIST-1b: P = {model.per:.4f} d, Rp/R* = {model.rp:.4f}, "
        f"max binned depth = {depth_ppt:.2f} ppt, "
        f"{int(np.sum(flux < 1))}/{N_FRAMES} frames in transit"
    )

    catalog = query_gaia(
        RA_T1, DEC_T1, 110.0, mag_limit=21.0,
        cache_dir=os.path.join(OUT, "cache"),
    )
    print(f"Gaia field: {len(catalog)} sources with G < 21 within 110\"")

    paths = simulate_frames(catalog, flux)

    viewer = None
    if args.live:
        from wcc_phot import LiveViewer

        viewer = LiveViewer(zoom=200)

    with warnings.catch_warnings():
        warnings.simplefilter("always", UserWarning)  # surface the n_ref note
        result = run_photometry(
            paths,
            target=TARGET_ID,
            method="aperture",
            times=TIMES_D,
            output=os.path.join(OUT, "phot.fits"),
            report=os.path.join(OUT, "phot_report.pdf"),
            on_frame=viewer,
        )
    refs = result.stars[result.stars["role"] == "ref"]
    print(
        f"photometry: r_ap = {result.params['r_ap']:.0f} px, "
        f"{len(refs)} reference stars "
        f"(G = {', '.join(f'{g:.1f}' for g in sorted(refs['gmag']))})"
    )

    # pipeline normalizes by the median over frames; do the same to the model
    model_norm = flux / np.median(flux)
    checks = verify(result.lightcurve, model_norm, depth_ppt)

    plot_lightcurve(
        result.lightcurve, model, model_norm,
        os.path.join(OUT, "transit_lightcurve.png"),
        interactive=viewer is not None,
    )

    print()
    ok = True
    for text, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {text}")
        ok &= passed
    print(
        f"\noutputs in {OUT}/: frames, phot.fits, phot_report.pdf/.png, "
        "transit_lightcurve.png"
    )
    if viewer is not None:
        print("close the live window to exit")
        viewer.hold()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
