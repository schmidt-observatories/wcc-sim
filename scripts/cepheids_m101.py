"""End-to-end M101 Cepheid period-luminosity simulation + retrieval.

Run:  $PY scripts/cepheids_m101.py [--quick]
(network needed on first run for the Gaia foreground query; --quick runs a
reduced 8-Cepheid / 8-epoch smoke configuration and always exits 0)

Injects 30 synthetic classical Cepheids (F8I supergiant template,
P = 10-60 d, demo Leavitt law M_G = -2.9 (log P - 1) - 4.2 at
mu = 29.2) plus 15 constant F8I calibration stars into a WCC field on the
M101 disk, on top of the galaxy's extended light (exponential disk +
de Vaucouleurs bulge Sersic components) and the real Gaia foreground.
Each Cepheid carries Milky-Way + internal F99 reddening applied as a
band-averaged attenuation of its count rate. 28 epochs over 180 d are
simulated in zwo:i, in focus, with chromatic effective PSFs; wcc_phot PSF
photometry recovers per-epoch fluxes, a per-epoch zero point from the
constant stars calibrates them to magnitudes, Lomb-Scargle recovers the
periods, and a weighted P-L fit recovers the distance modulus.

Checks (exit nonzero on failure, except under --quick):
  1. >= 80% of Cepheids with median per-epoch SNR > 10 recover P to 1%.
  2. P-L intercept -> distance modulus within 3 sigma of the injected
     29.2 after correcting for the known mean extinction.
  3. chi2/dof of the calibrated light curves against the injected model
     in [0.5, 2].

Writes frames, ECSV tables, and a summary figure to ./cepheid_out/.
"""

import argparse
import os
import sys
import time as walltime
import warnings

import numpy as np
from astropy.table import Table, vstack
from astropy.timeseries import LombScargle

from wcc_phot import run_photometry
from wcc_phot.io import load_frame
from wcc_sim import SersicComponent, simulate_field
from wcc_sim.catalog import COLUMNS as GAIA_COLUMNS
from wcc_sim.catalog import query_gaia
from wcc_sim.chromatic import attenuation_factor
from wcc_sim.detectors import get_geometry
from wcc_sim.wcsutil import build_wcs

# --- observation setup -----------------------------------------------------
GAL_RA, GAL_DEC = 210.80243, 54.34875  # M101 center
MU = 29.2  # distance modulus (6.9 Mpc)
FIELD_DRA_ARCMIN = 3.5  # pointing offset east along RA
SENSORFILTER = "zwo:i"
FOCUS = 0
SHAPE = (2048, 2048)  # 34.5" x 34.5"
EXPTIME = 1200.0  # s
N_READS = 2
N_EPOCHS = 28
BASELINE_D = 180.0
SEED = 42
OUT = "cepheid_out"

# --- injected population ---------------------------------------------------
N_CEPH = 30
N_CONST = 15  # constant F8I calibration stars
TEMPLATE = "F8I"
PL_SLOPE, PL_ZP = -2.9, -4.2  # demo Leavitt law: M_G = PL_SLOPE*(logP-1)+PL_ZP
P_RANGE = (10.0, 60.0)  # d
AMP_RANGE = (0.3, 0.7)  # peak-to-peak amplitude, mag
CONST_G_RANGE = (20.5, 22.0)
EBV_MW = 0.008  # SF11 foreground toward M101
EBV_INT_MED, EBV_INT_SIG = 0.10, 0.5  # internal reddening: lognormal
RISE = 0.2  # rising-branch fraction of the pulsation cycle
SYNTH_ID0 = 9_100_000_000_000_000_000

# M101 extended light: exponential disk + small de Vaucouleurs bulge, both
# centered on the (off-frame) galaxy nucleus 3.5' away.
EXTENDED = [
    SersicComponent(ra=GAL_RA, dec=GAL_DEC, n=1.0, r_eff_arcsec=250.0,
                    total_mag=8.0, ellip=0.1, pa_deg=35.0,
                    template="G2V", ebv=0.03),
    SersicComponent(ra=GAL_RA, dec=GAL_DEC, n=4.0, r_eff_arcsec=25.0,
                    total_mag=11.0, ellip=0.05, pa_deg=35.0,
                    template="K0III", ebv=0.02),
]

RA_F = GAL_RA + (FIELD_DRA_ARCMIN / 60.0) / np.cos(np.radians(GAL_DEC))
DEC_F = GAL_DEC


def cepheid_template(phase):
    """Zero-mean asymmetric light-curve template in mag (span -0.5..+0.5).

    Fast rise over the first RISE of the cycle (mag decreasing = getting
    brighter), slow linear decline over the rest.
    """
    ph = np.asarray(phase, dtype=float) % 1.0
    return np.where(
        ph < RISE,
        0.5 - ph / RISE,
        (ph - RISE) / (1.0 - RISE) - 0.5,
    )


def make_population(rng):
    """Injected Cepheid + constant-star tables (no positions yet)."""
    logp = rng.uniform(np.log10(P_RANGE[0]), np.log10(P_RANGE[1]), N_CEPH)
    ebv = EBV_MW + rng.lognormal(np.log(EBV_INT_MED), EBV_INT_SIG, N_CEPH)
    a_g = np.array(
        [
            -2.5 * np.log10(attenuation_factor(TEMPLATE, e, SENSORFILTER))
            for e in ebv
        ]
    )
    ceph = Table(
        {
            "source_id": SYNTH_ID0 + np.arange(N_CEPH, dtype=np.int64),
            "period_d": 10.0**logp,
            "logp": logp,
            "m0": MU + PL_SLOPE * (logp - 1.0) + PL_ZP + a_g,
            "amp": rng.uniform(*AMP_RANGE, N_CEPH),
            "phase0": rng.uniform(0.0, 1.0, N_CEPH),
            "ebv": ebv,
            "a_g": a_g,
        }
    )
    const = Table(
        {
            "source_id": SYNTH_ID0 + N_CEPH + np.arange(N_CONST,
                                                        dtype=np.int64),
            "gmag": rng.uniform(*CONST_G_RANGE, N_CONST),
        }
    )
    return ceph, const


def cepheid_mag(ceph, t_d):
    """(n_ceph, n_epoch) injected G magnitudes at times t_d [d]."""
    phase = (t_d[None, :] / ceph["period_d"][:, None]
             + ceph["phase0"][:, None])
    return (np.asarray(ceph["m0"])[:, None]
            + np.asarray(ceph["amp"])[:, None] * cepheid_template(phase))


def sample_positions(rng, n, shape, margin=200, min_sep=60):
    """n (x, y) positions, rejection-sampled for min pairwise separation."""
    ny, nx = shape
    xs, ys = [], []
    while len(xs) < n:
        x = rng.uniform(margin, nx - 1 - margin)
        y = rng.uniform(margin, ny - 1 - margin)
        if all(np.hypot(x - a, y - b) >= min_sep for a, b in zip(xs, ys)):
            xs.append(x)
            ys.append(y)
    return np.array(xs), np.array(ys)


def make_catalog(wcs, ceph, const, rng, shape):
    """Gaia foreground + synthetic rows (epoch mags filled in later)."""
    half_diag = 0.5 * np.hypot(*shape) * 16.87 / 1000.0 + 10.0
    gaia = query_gaia(RA_F, DEC_F, half_diag, mag_limit=21.0,
                      cache_dir=os.path.join(OUT, "cache"))
    gaia = gaia[GAIA_COLUMNS]  # drop any extras; keep the canonical set
    gaia["spt"] = np.array([""] * len(gaia), dtype="U8")

    n_synth = len(ceph) + len(const)
    xs, ys = sample_positions(rng, n_synth, shape)
    ras, decs = wcs.pixel_to_world_values(xs, ys)
    synth = Table(
        {
            "source_id": np.concatenate(
                [ceph["source_id"], const["source_id"]]
            ).astype(np.int64),
            "ra": np.asarray(ras, dtype=float),
            "dec": np.asarray(decs, dtype=float),
            "phot_g_mean_mag": np.concatenate(
                [np.asarray(ceph["m0"]), np.asarray(const["gmag"])]
            ),
            "phot_bp_mean_mag": np.full(n_synth, np.nan),
            "phot_rp_mean_mag": np.full(n_synth, np.nan),
            "spt": np.array([TEMPLATE] * n_synth, dtype="U8"),
        }
    )
    return vstack([gaia, synth], join_type="exact")


def simulate_epochs(catalog, ceph, times_d):
    """One frame per epoch with the Cepheid magnitudes set; returns paths."""
    mags = cepheid_mag(ceph, times_d)
    ceph_rows = np.array(
        [
            np.flatnonzero(
                np.asarray(catalog["source_id"], dtype=np.int64) == sid
            )[0]
            for sid in ceph["source_id"]
        ]
    )
    paths = []
    for k in range(len(times_d)):
        cat_k = catalog.copy()
        cat_k["phot_g_mean_mag"][ceph_rows] = mags[:, k]
        path = os.path.join(OUT, f"epoch_{k:03d}.fits")
        t0 = walltime.time()
        field = simulate_field(
            RA_F, DEC_F,
            sensorfilter=SENSORFILTER,
            focus=FOCUS,
            exptime=EXPTIME,
            n_reads=N_READS,
            catalog=cat_k,
            shape=SHAPE,
            extended_sources=EXTENDED,
            chromatic=True,
            seed=3000 + k,
            output=path,
            write_clean=False,
        )
        print(
            f"epoch {k + 1:2d}/{len(times_d)}: t = {times_d[k]:7.2f} d, "
            f"{int(field.saturation_mask.sum())} saturated px, "
            f"{walltime.time() - t0:.1f} s"
        )
        paths.append(path)
    return paths


def measure_star(frames, sid, times_d):
    """(flux_e, flux_err_e, flags) per epoch for one star, PSF photometry."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = run_photometry(frames, target=int(sid), method="psf",
                             n_ref=3, times=times_d)
    rows = res.measurements[res.measurements["role"] == "target"]
    rows.sort("frame")
    return (
        np.asarray(rows["flux_e"], dtype=float),
        np.asarray(rows["flux_err_e"], dtype=float),
        np.asarray(rows["flags"], dtype=int),
    )


def calibrate(flux, flux_err, const_flux, const_gmag):
    """Per-epoch zero point from the constant stars -> mags + errors.

    flux, flux_err: (n_star, n_epoch); const_flux: (n_const, n_epoch).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        inst = const_gmag[:, None] + 2.5 * np.log10(const_flux)
    zp = np.nanmedian(inst, axis=0)
    zp_err = (
        1.4826
        * np.nanmedian(np.abs(inst - zp[None, :]), axis=0)
        / np.sqrt(const_flux.shape[0])
    )
    mag = zp[None, :] - 2.5 * np.log10(flux)
    mag_err = np.hypot(1.0857 * flux_err / flux, zp_err[None, :])
    return mag, mag_err, zp, zp_err


def recover_period(t_d, mag, err):
    """Lomb-Scargle best period [d] on a dense grid over 5-100 d."""
    freq = np.linspace(1.0 / 100.0, 1.0 / 5.0, 100_000)
    power = LombScargle(t_d, mag, err).power(freq)
    return 1.0 / freq[np.argmax(power)]


def flux_mean_mag(mag):
    """Mean magnitude in flux space (matches the injected definition)."""
    return -2.5 * np.log10(np.mean(10.0 ** (-0.4 * np.asarray(mag))))


def fit_pl(logp, m, sigma, pivot=1.3):
    """Weighted linear fit m = a*(logp - pivot) + b -> (a, b, sig_a, sig_b).

    Parameter errors are scaled by sqrt(max(1, chi2/dof)) so intrinsic
    scatter (e.g. the per-star extinction spread) inflates them honestly.
    """
    x = np.column_stack([logp - pivot, np.ones_like(logp)])
    w = 1.0 / np.asarray(sigma) ** 2
    cov = np.linalg.inv(x.T @ (w[:, None] * x))
    a, b = cov @ (x.T @ (w * m))
    resid = m - (a * (logp - pivot) + b)
    chi2_dof = float(np.sum(w * resid**2)) / max(len(m) - 2, 1)
    scale = np.sqrt(max(1.0, chi2_dof))
    return (
        float(a), float(b),
        scale * float(np.sqrt(cov[0, 0])),
        scale * float(np.sqrt(cov[1, 1])),
    )


def verify(ceph, rec, mean_a_g, pivot=1.3):
    """(text, passed) rows for the three checks + the fitted numbers."""
    good = rec["snr"] > 10.0
    dp = np.abs(rec["p_rec"] - ceph["period_d"]) / ceph["period_d"]
    frac = float(np.mean(dp[good] < 0.01)) if good.any() else 0.0

    m_corr = rec["mean_mag"] - mean_a_g  # known-mean extinction correction
    a, b, sig_a, sig_b = fit_pl(
        np.log10(rec["p_rec"]), m_corr, rec["mean_mag_err"], pivot=pivot
    )
    mu_rec = b - (PL_SLOPE * (pivot - 1.0) + PL_ZP)
    chi2_dof = float(np.mean(rec["chi2_dof"]))

    checks = [
        (
            f"period recovery: {frac * 100:.0f}% of SNR>10 Cepheids "
            f"within 1% ({int(good.sum())} usable)",
            frac >= 0.80,
        ),
        (
            f"distance modulus: mu = {mu_rec:.3f} +/- {sig_b:.3f} "
            f"(injected {MU}), slope {a:.2f} +/- {sig_a:.2f} "
            f"(injected {PL_SLOPE})",
            abs(mu_rec - MU) < 3.0 * sig_b,
        ),
        (
            f"light-curve chi2/dof vs injected model = {chi2_dof:.2f}",
            0.5 < chi2_dof < 2.0,
        ),
    ]
    return checks, (a, b, sig_a, sig_b, mu_rec)


def plot_summary(ceph, rec, times_d, mags_obs, errs_obs, mag_inj, frame0,
                 fit, mean_a_g, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        plt.style.use("gks")
    except (OSError, ValueError):
        pass
    a, b, _, _, mu_rec = fit
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # (1) P-L relation
    ax = axes[0]
    lp = np.linspace(*np.log10(P_RANGE), 50)
    ax.plot(lp, MU + PL_SLOPE * (lp - 1.0) + PL_ZP, color="#d1495b",
            lw=1.5, label="injected P-L")
    ax.plot(lp, a * (lp - 1.3) + b, color="0.4", ls="--", lw=1.2,
            label=f"fit: $\\mu$ = {mu_rec:.2f}")
    ax.errorbar(np.log10(rec["p_rec"]), rec["mean_mag"] - mean_a_g,
                yerr=rec["mean_mag_err"], fmt="o", color="#00798c", ms=4,
                lw=1, capsize=2, label="recovered (ext.-corr.)")
    ax.invert_yaxis()
    ax.set_xlabel("log P [d]")
    ax.set_ylabel("mean G [mag]")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"M101 Cepheids, WCC {SENSORFILTER}")

    # (2) example folded light curve: highest-SNR Cepheid
    j = int(np.argmax(rec["snr"]))
    ax = axes[1]
    p = float(rec["p_rec"][j])
    ph = (times_d / p) % 1.0
    ax.errorbar(ph, mags_obs[j], yerr=errs_obs[j], fmt="o",
                color="#00798c", ms=4, lw=1, capsize=2)
    # overlay the injected model, folded at the *recovered* period
    tt = np.linspace(times_d.min(), times_d.max(), 2000)
    ax.plot((tt / p) % 1.0, cepheid_mag(ceph[[j]], tt)[0], ".",
            color="#d1495b", ms=1.5, alpha=0.6, label="injected")
    ax.invert_yaxis()
    ax.set_xlabel(f"phase (P = {p:.2f} d)")
    ax.set_ylabel("G [mag]")
    ax.set_title(f"Cepheid {j}: SNR = {rec['snr'][j]:.0f}")
    ax.legend(loc="upper right", fontsize=8)

    # (3) field with Cepheid positions
    ax = axes[2]
    img = frame0.image_e
    lo, hi = np.percentile(img, [5, 99.5])
    ax.imshow(img, origin="lower", cmap="gray_r",
              vmin=lo, vmax=hi)
    x, y = frame0.wcs.world_to_pixel_values(
        np.asarray(ceph["ra_field"]), np.asarray(ceph["dec_field"])
    )
    ax.scatter(x, y, s=60, facecolor="none", edgecolor="#d1495b", lw=0.8)
    ax.set_title(f"{EXPTIME:.0f} s epoch, galaxy + Cepheids")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    global N_CEPH, N_CONST, N_EPOCHS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="reduced smoke run (8 Cepheids, 8 epochs)")
    args = parser.parse_args(argv)
    if args.quick:
        N_CEPH, N_CONST, N_EPOCHS = 8, 8, 8

    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    geom = get_geometry(SENSORFILTER)
    wcs = build_wcs(RA_F, DEC_F, geom.plate_scale_mas, 0.0, SHAPE)
    times_d = np.sort(rng.uniform(0.0, BASELINE_D, N_EPOCHS))

    ceph, const = make_population(rng)
    catalog = make_catalog(wcs, ceph, const, rng, SHAPE)
    # record the field positions for the summary plot
    idx = {int(s): i for i, s in enumerate(catalog["source_id"])}
    ceph["ra_field"] = [catalog["ra"][idx[int(s)]] for s in ceph["source_id"]]
    ceph["dec_field"] = [catalog["dec"][idx[int(s)]] for s in ceph["source_id"]]
    ceph.write(os.path.join(OUT, "cepheids_injected.ecsv"),
               format="ascii.ecsv", overwrite=True)
    print(
        f"field: ({RA_F:.4f}, {DEC_F:.4f}), {len(catalog)} catalog rows "
        f"({len(catalog) - len(ceph) - len(const)} Gaia), "
        f"{N_CEPH} Cepheids P = {ceph['period_d'].min():.1f}-"
        f"{ceph['period_d'].max():.1f} d, "
        f"G = {ceph['m0'].min():.1f}-{ceph['m0'].max():.1f}"
    )

    paths = simulate_epochs(catalog, ceph, times_d)
    frames = [load_frame(p) for p in paths]

    # --- photometry: Cepheids + constant stars -----------------------------
    const_flux = np.zeros((len(const), N_EPOCHS))
    for i, sid in enumerate(const["source_id"]):
        f, _, _ = measure_star(frames, sid, times_d)
        const_flux[i] = f
        print(f"constant {i + 1:2d}/{len(const)} measured")

    flux = np.zeros((N_CEPH, N_EPOCHS))
    flux_err = np.zeros_like(flux)
    flags = np.zeros(flux.shape, dtype=int)
    for j, sid in enumerate(ceph["source_id"]):
        flux[j], flux_err[j], flags[j] = measure_star(frames, sid, times_d)
        print(f"cepheid {j + 1:2d}/{N_CEPH} measured")

    mags, errs, zp, zp_err = calibrate(
        flux, flux_err, const_flux, np.asarray(const["gmag"])
    )
    print(
        f"zero point: rms over epochs {np.std(zp):.4f} mag, "
        f"median per-epoch error {np.median(zp_err):.4f} mag"
    )

    # --- recovery -----------------------------------------------------------
    mag_inj = cepheid_mag(ceph, times_d)
    rec = Table(
        {
            "source_id": ceph["source_id"],
            "p_rec": np.zeros(N_CEPH),
            "mean_mag": np.zeros(N_CEPH),
            "mean_mag_err": np.zeros(N_CEPH),
            "snr": np.zeros(N_CEPH),
            "chi2_dof": np.zeros(N_CEPH),
        }
    )
    for j in range(N_CEPH):
        ok = (flags[j] == 0) & np.isfinite(mags[j])
        rec["p_rec"][j] = recover_period(times_d[ok], mags[j][ok],
                                         errs[j][ok])
        rec["mean_mag"][j] = flux_mean_mag(mags[j][ok])
        rec["mean_mag_err"][j] = float(np.median(errs[j][ok])
                                       / np.sqrt(ok.sum()))
        rec["snr"][j] = float(np.median(flux[j][ok] / flux_err[j][ok]))
        rec["chi2_dof"][j] = float(
            np.mean(((mags[j][ok] - mag_inj[j][ok]) / errs[j][ok]) ** 2)
        )
    rec.write(os.path.join(OUT, "cepheids_recovered.ecsv"),
              format="ascii.ecsv", overwrite=True)

    mean_a_g = float(np.mean(ceph["a_g"]))
    checks, fit = verify(ceph, rec, mean_a_g)
    plot_summary(ceph, rec, times_d, mags, errs, mag_inj, frames[0], fit,
                 mean_a_g, os.path.join(OUT, "cepheid_pl.png"))

    print()
    ok = True
    for text, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {text}")
        ok &= passed
    print(f"\noutputs in {OUT}/: frames, ECSV tables, cepheid_pl.png")
    if args.quick:
        print("(--quick: checks are informational, exiting 0)")
        sys.exit(0)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
