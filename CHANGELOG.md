# Changelog

## 0.1.0 (unreleased)

First public release.

### Simulator (`wcc_sim`)
- `simulate_field`: Gaia DR3 cone search (cached on disk, astrometry
  propagated to the observation epoch, XHIP bright-star supplement), wcc-etc
  count rates per spectral template, oversampled PSF rendering with sub-pixel
  placement, the ETC noise model with full-well and ADC saturation, and a
  multi-extension FITS product (`SCI`, `SATMASK`, `CAT`, `CLEAN`) with a TAN
  WCS.
- PSF modes: in-focus Airy at each star's effective wavelength, +1/+2-wave
  Zemax Huygens defocus, optional Gaussian jitter, spectrum-weighted
  chromatic PSFs.
- Analytic diffraction-wing extension beyond the PSF stamp, and the measured
  scattered-light halo (azimuthally averaged stray-light profile) as an
  additive term that reaches across the whole array.
- Extended sources: analytic Sersic components with reddening.
- Detectors: Sony IMX455 (`zwo:*`) and Hamamatsu HWK4123 qCMOS (`qcmos:*`,
  selectable gain mode); the per-frame saturation ceiling is recorded as
  `SATLEVEL`.
- `wcc-sim` command line.

### Photometry (`wcc_phot`)
- Differential aperture and PSF photometry on a series of frames: target and
  reference selection, per-frame centroiding, quality flags, relative light
  curve, one-page diagnostic report, live viewer.
- `wcc-phot` command line.

### Plotting
- `.plot_*` methods on `SimulatedField` (`plot`, `plot_cutout`,
  `plot_psf_profile`, `plot_satmask`, `psf_report`) and `PhotometryResult`
  (`plot`, `plot_lightcurve`, `plot_field`, `plot_binned_rms`,
  `plot_raw_fluxes`, `plot_rms_vs_mag`, `plot_centroids`, `plot_background`,
  `report`), each taking `ax=None` and returning the axes. The report panels
  behind them are public functions in `wcc_sim.psfreport` and
  `wcc_phot.report`; `wcc_sim.plotting` adds `plot_image`, `plot_stamp`,
  `plot_radial_profile`, `plot_encircled_energy` for plain arrays and ships
  the house style as `plot_default.mplstyle` (no user-installed style needed).
- Tutorial `10_plotting.ipynb`; the other notebooks use the methods.

### Validation
- TRAPPIST-1b transit recovery and M101 Cepheid period-luminosity scripts.
- Monte Carlo cross-check of the wcc-etc signal and noise prediction against
  simulated frames (`tests/test_etc_crosscheck.py`).

### Fixes from the September 2026 audit
- qCMOS saturation ceiling (#17), Gaia synchronous-query truncation (#19),
  in-focus PSF wavelength (#20), synthetic BP-RP to template table with
  colour-interpolated rates (#21), input validation and edge cases (#22),
  catalog masks and spherical matching (#10), light-curve guards (#12),
  defocus PSF centring (#18, with wcc-etc), one delivered-frame contract
  `(ADU - bias) * gain` for objects and files (#11), recorded seed for
  unseeded runs and a stale-Gaia-cache warning (#24).

### Documentation
- Docs site carries the Schmidt Observatory System logo and the stock Read the
  Docs theme (the Schmidt Sciences stylesheet was never loaded). New glossary
  page; the index opens in plain words; the concepts diagram names each stage
  in words with the module beside it; the quickstart uses `field.plot()`; the
  installation page lists the extras and no longer claims a wcc-etc 0.6.0
  floor; the tutorials no longer name a personal conda kernel. Tutorial 01 is
  split into a short `01_getting_started` and a `02_verification` that holds
  the assertions; the other tutorials are renumbered 03 to 10.
