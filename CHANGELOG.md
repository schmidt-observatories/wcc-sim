# Changelog

## Unreleased

- `wcc_sim.GalSimComponent`: an extended source described by any
  `galsim.GSObject` (bulge+disk sums, shears, RealGalaxy), normalized by
  `total_mag`/`template`/`ebv` like `SersicComponent` and rendered through the
  same PSF convolution.
- `wcc_sim.ImageComponent`: a calibrated sky image with a WCS (HST drizzled
  mosaics in e-/s with an AB zero point) reprojected onto the frame with
  `reproject`, surface brightness conserved, each pixel treated as an
  AB-flat source (`starflux.rate_for_ab`). `wcc_sim.phangs` lists, downloads
  (once, into `~/.cache/wcc_sim/phangs`) and loads PHANGS-HST mosaics as
  ImageComponents, mapping WCC g/r/i to F435W-or-F438W/F555W/F814W.
- galsim and reproject are an optional extra: `pip install wcc-sim[galaxies]`.
- `wcc_sim.empty_catalog()`: public way to skip the Gaia query (previously
  the private `catalog._empty_table`).
- Tutorial notebook `11_galaxies_phangs`: full-frame g/r/i WCC images of
  NGC 1300 and NGC 4321 from PHANGS-HST and a colour composite.
- `wcc_sim.simulate_image`: simulate a WCC exposure of a sky image (FITS with a
  celestial WCS and physical `BUNIT`: Jy/pixel, Jy/arcsec2, MJy/sr, F_lambda or
  electron/s), e.g. a POLARIS radiative-transfer model. The image is resampled
  flux-conservingly (`reproject_exact`) onto a detector-aligned k x k sub-grid
  and PSF-convolved per sub-pixel phase (`render.render_subgrid`, identical to
  one `add_star` per sub-pixel), so compact structure finer than the detector
  pixel (a planet next to its star) is kept. Built from public pieces:
  `SkyImageComponent` (validated image, `fitsimage.ImageInput`),
  `detector_frame` (pointing with `offset_arcsec` and `pa`) and
  `WCCObservation` (`rate_image()`, noiseless e-/s; `expose()`, noise and
  saturation, returning a `SimulatedField`).
- FITS output of image simulations: a `COMP` table (one row per component) and
  the SCI cards `PNTSRC`, `CEN_RA`/`CEN_DEC`, `OFFSRA`/`OFFSDEC`, `IMGTOT`,
  `IMGRATE`, `IMGSUBK`, `IMGBUNIT`, `FLUXORIG`, `FLUXJY`, `FLUXFLAM`,
  `FLUXBAND`, `BANDPIV`, `BANDWID`. `fluxrate.band_fluxes` converts a count
  rate back to the flux in the detector band.
- `simulate_field(..., offset_arcsec=(dRA, dDec))`: offset pointing (e.g.
  dithers), applied on the sky before `pa`; the default (0, 0) gives the same
  output as before.
- `render.unit_stamp`: the PSF stamp construction factored out of `add_star`
  so stars and images use the same stamp; stars are rendered bit-identically.
- Optional extra `pip install wcc-sim[image]` (reproject, without galsim); the
  `test` extra now includes reproject.
- Tutorial notebook `12_image_sources`: a synthetic star + inclined ring +
  planet image through each step, with flux-conservation checks.

## 0.1.0 (2026-10-12)

First public release (PyPI). 0.1.0rc1 went to TestPyPI the same day.

### Packaging
- `wcc_etc>=0.1.0` is a declared dependency (it is on PyPI since 0.1.0),
  imported through its public `KIND_NAMES` and `SENSORFILTER_FOCUS` names, and
  `wcc_sim.require_wcc_etc()` turns a missing or too-old wcc_etc into a readable
  ImportError. CI installs wcc-etc from PyPI and a second job builds the wheel
  from the checkout, installs it into an empty venv and runs an offline
  simulation. Tagged `v*` pushes publish to TestPyPI, and final tags to PyPI, by trusted
  publishing.

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
- The packaged scatter profile, its module docstring and the `SCATFILE`
  header card label the halo as the wcc-etc 450 nm scatter map instead of
  naming the ray-trace run. `CITATION.cff` removed.
