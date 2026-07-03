# TRAPPIST-1b end-to-end transit simulation + photometry validation

**Date:** 2026-07-02
**Deliverable:** `scripts/transit_trappist1b.py` — a single end-to-end script that
simulates a TRAPPIST-1b transit as observed by the Lazuli WCC and verifies that
`wcc_phot` recovers the injected signal.

## Goal

Demonstrate, end to end, that a known transit injected at the catalog level
survives the full chain — wcc_etc count rates → wcc_sim image rendering with
+2-wave defocus PSF and full noise model → wcc_phot differential aperture
photometry — and comes out with the expected depth and noise.

## Observation setup

| Item | Value | Why |
|---|---|---|
| Target | TRAPPIST-1 (Gaia DR3 2635476908753563008, G=15.62, M5V template) | user-specified |
| Planet | TRAPPIST-1 b via `lazuli_transit.TransitModel.from_planet` (NASA archive cache: P=1.5108 d, Rp/R*=0.0859 → 7.38 ppt depth, a/R*=20.84, i=89.73°, T14≈36 min) | user-pointed at lazuli-transit |
| Filter | `zwo:i` | brightest band for an M5V: 16.4 ke-/s → 0.46 ppt photon+sky noise per exposure (vs 0.67 ppt in r) |
| Focus | +2 waves defocus (`focus=2`) | user-specified; EE95 radius 35 px, brightest ref peaks at 3.1 ke- ≪ 16.3 ke- well → nothing saturates |
| Exposure | 300 s × 36 frames = 3 h, transit centered (t0=0, times = frame centers −1.5 h … +1.5 h) | user-specified |
| Frame | `shape=(2200, 9568)` (full-width strip, 161″×37″), pointing offset (+13″, +11.5″) from the target | Gaia field is sparse (14 stars G<21 within 110″); the strip catches the target + the 4 best refs (G=14.7, 16.5, 17.3, 18.5) at ~⅓ the memory of full frames (36 frames ≈ 6 GB in wcc_phot vs 17 GB) |

## Injection mechanism

`simulate_field(catalog=...)` bypasses the Gaia query, and
`rates_for_catalog` scales rates analytically as `10^(-0.4 (G - 15))` with the
spectral type set only by BP−RP. So per frame k:

1. exposure-averaged relative flux `f_k` = mean of `TransitModel.relative_flux`
   over 5 sub-samples spanning the 300 s exposure;
2. `cat_k = cat.copy()`; target row `phot_g_mean_mag += -2.5 log10(f_k)`;
3. `simulate_field(..., catalog=cat_k, seed=1000+k, output=frame_k.fits, write_clean=False)`.

This is an exact flux injection — no other star or noise term changes.
Frames go to `transit_out/` on disk; `run_photometry` gets the paths.

## Photometry

`run_photometry(paths, target=2635476908753563008, method="aperture",
times=times_days, output=transit_out/phot.fits)` with default model-PSF-driven
geometry (r_ap = EE95 ≈ 36 px, annulus 54–90 px). Reference selection finds the
4 usable refs (a UserWarning about <10 refs is expected and fine).

## Verification ("results as expected")

The pipeline normalizes `rel_flux` by its median over frames, so apply the
*identical* normalization to the injected exposure-averaged model
(`model_norm = f / median(f at the 36 frame times)`) — zero free parameters.
Then:

1. **Depth ratio** — least-squares scale α of `(1 − model_norm)` against
   `(1 − rel_flux_norm)`: PASS if |α − 1| < 3σ_α.
2. **Chi-square** — χ²/dof of residuals against `rel_flux_norm_err`:
   PASS if 0.5 < χ²/dof < 2.
3. **Out-of-transit scatter** — OOT RMS < 2 × median error bar.

Script prints a summary table + PASS/FAIL per check, exits nonzero on FAIL,
and saves `transit_out/transit_lightcurve.png` (light curve + injected model +
residual panel, gks plot style).

## Alternatives considered

- **Full frames (9568×6380)** — captures 2 more (faint) refs but ~17 GB in
  `run_photometry`'s eager frame list on a 24 GB machine; rejected.
- **Injection via `rate_e_s` post-hoc scaling** — `simulate_field` recomputes
  rates from mags, so mag-space injection is the supported path.
- **PSF photometry** — aperture is the default and the robust mode at +2 waves;
  PSF mode can be flipped on with one argument later if wanted.
