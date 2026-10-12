![Logo](docs/logo_schmidt_observatory_system.png)

[![Tests](https://github.com/schmidt-observatories/wcc-sim/actions/workflows/tests.yml/badge.svg)](https://github.com/schmidt-observatories/wcc-sim/actions/workflows/tests.yml)

# wcc-sim
End-to-end image simulator and differential-photometry pipeline for the
Wide-field Context Camera (WCC) on Lazuli.

`wcc-sim` queries Gaia DR3 for a pointing, converts Gaia photometry to detector
count rates through the [wcc-etc](https://github.com/schmidt-observatories/wcc-etc)
instrument model, renders every star with the in-focus (Airy) or +1/+2-wave
defocus (Zemax Huygens) PSF plus its diffraction wings and the scattered-light
halo, adds photon/sky/dark/read noise with full-well and ADC saturation, and
writes a multi-extension FITS image (SCI + SATMASK + CAT + CLEAN) with a TAN
WCS. The sibling `wcc-phot` package extracts a relative light curve from a
series of such frames.

# Installation

## **1. Install wcc-etc**

`wcc_etc` provides the instrument model and is not on PyPI, so install it from
its repository first:
```sh
git clone https://github.com/schmidt-observatories/wcc-etc.git
cd wcc-etc && pip install -e . && cd ..
```

## **2. Clone and install wcc-sim**
```sh
git clone https://github.com/schmidt-observatories/wcc-sim.git
cd wcc-sim
pip install -e .
```

With optional extras:
```sh
pip install -e ".[test]"   # pytest
pip install -e ".[docs]"   # documentation build (Sphinx, nbsphinx, etc.)
```

Python 3.11 or newer is required. A default run queries the Gaia archive, so
it needs network access. Pass `cache_dir=` (CLI: `--cache-dir`) to keep the
Gaia result on disk so repeat runs work offline.

# Quick Start

```python
from wcc_sim import simulate_field

# Simulate one 90 s frame of the field at (RA, Dec) = (150.1, 2.2) deg on the
# Sony IMX455 detector in the r band, +1 wave defocused, and save it as FITS.
field = simulate_field(150.1, 2.2, sensorfilter="zwo:r", focus=1,
                       exptime=90, seed=42, output="field_1wave.fits")

field.image_adu       # the detector image (ADU)
field.catalog         # the Gaia table with pixel positions and count rates
field.params          # everything the run used (rates, noise, PSF, wing floor)
field.plot(catalog=True)      # the image with the catalog circled (ax=None -> new figure)
field.plot_psf_profile()      # Airy core, scatter halo and the rendered radial profile
```

The same from the command line:
```sh
wcc-sim --ra 150.1 --dec 2.2 --sensorfilter zwo:r --focus 1 \
        --exptime 90 --seed 42 -o field_1wave.fits
```

Then recover a light curve from a series of frames of the same field:
```python
from wcc_phot import run_photometry

result = run_photometry(["f0.fits", "f1.fits", "f2.fits"], target=(150.1, 2.2),
                        method="aperture", n_ref=10)
result.lightcurve     # per frame: rel_flux_norm, errors, flags
result.plot()         # the relative light curve; result.plot_field(frame), .report(path)
```
```sh
wcc-phot f*.fits --ra 150.1 --dec 2.2 --method psf -o phot.fits --report report.pdf
```

`sensorfilter` is `kind:band` as in wcc-etc: `zwo:*` is the Sony IMX455
(9568 x 6380 px, 16.87 mas/pix) and `qcmos:*` the Hamamatsu HWK4123
(4096 x 2304 px, 20.64 mas/pix). `focus=0|1|2` selects in-focus, +1 wave or
+2 waves of defocus. `extended_sources=[SersicComponent(...)]` adds analytic
galaxy light and `chromatic=True` switches to spectrum-weighted PSFs (in focus
only).

# Tutorial
See the `notebooks/` directory for example tutorials (executed outputs
included; the Gaia queries they need are cached in `notebooks/gaia_cache/`,
so they run offline):

| Notebook | What it covers |
|---|---|
| `01_wcc_sim_example.ipynb` | Getting started: simulate a field, inspect the catalog, image and parameters, and verify count rates against wcc-etc. |
| `02_psf_focus_modes.ipynb` | In-focus Airy vs +1/+2-wave defocus PSFs and what they do to peak pixel and saturation. |
| `03_detectors_and_filters.ipynb` | The two detectors, the filter sets, and Gaia-to-count-rate conversion per spectral type. |
| `04_noise_and_saturation.ipynb` | Photon, sky, dark and read noise; full well and ADC clipping; the SATMASK extension. |
| `05_astrometry_and_catalogs.ipynb` | The TAN WCS and its round trip, where stars land, position angle, edges and user-supplied catalogs. |
| `06_cli_and_fits.ipynb` | The `wcc-sim` command line and the layout of the output FITS file. |
| `07_psf_wings.ipynb` | Diffraction wings beyond the PSF stamp: the wing model, seam smoothness, encircled energy and how far stars extend. |
| `08_wcc_phot_photometry.ipynb` | `wcc-phot`: simulate a dithered series, pick a target, aperture and PSF photometry, centroid drift, the live viewer. |
| `09_plotting.ipynb` | Every `.plot_*` method once: images with catalog and saturation overlays, cutouts, the PSF radial profile, stamps, encircled energy, the photometry panels. |

Ready-to-run CLI examples are in `scripts/example_*.sh`. Two end-to-end
validations live next to them: `scripts/transit_trappist1b.py` injects the
TRAPPIST-1b transit into the Gaia catalog, simulates 36 x 300 s defocused
frames, runs `wcc-phot` and checks the recovered depth; `scripts/cepheids_m101.py`
places synthetic Cepheids on the M101 disk (Sersic galaxy light plus the real
Gaia foreground) and recovers a period-luminosity relation.

# Notes
Instrument configuration (detectors, filters, throughput, PSFs, the stray-light
model) comes from wcc-etc and is not duplicated here. The one data product
wcc-sim ships is the azimuthally averaged scattered-light profile in
`src/wcc_sim/data/psfs/`, regenerated by `scripts/build_scatter_profile.py`.

## Model assumptions and validation status

wcc-sim takes its instrument model from wcc-etc. Some choices in that model
are deliberate margins or simplifications rather than best estimates, and
they move the numbers you get:

- **Read noise carries a x2 margin.** Both detector configurations multiply
  the datasheet read noise by `read_noise_margin = 2` to cover unmodelled
  noise sources (RTS, non-linearity), and every simulated frame inherits it.
  In a 90 s `zwo:r` frame most of the per-pixel variance is read noise, so
  faint-source depth is about 0.6 mag shallower than the datasheet value
  would give. There is no switch for it yet (issue #23).
- **Unobscured pupil.** The collecting area is pi (D/2)^2 and the in-focus
  Airy PSF has no central obscuration or spiders. The defocused PSFs are
  Zemax Huygens images of the real system.
- **Constant zodiacal background** at V = 22.5 mag/arcsec^2, with no
  dependence on ecliptic latitude or Sun angle.
- **Gaia G normalisation** uses a synthetic Vega-system zero point about 1%
  below the published DR3 value, so all count rates are about 1% low.
- **The scattered-light halo is monochromatic.** It comes from a single
  450 nm stray-light run and is applied with the same shape and fraction in
  every band; surface scatter falls with wavelength, so the halo is
  over-predicted in the red bands.
- **"N waves" of defocus** is a Fringe Zernike Z4 coefficient of N waves
  (2N waves peak-to-valley), the convention of the Zemax runs.

Not modelled: cosmic rays, PRNU, non-linearity, random telegraph noise,
charge diffusion and intrapixel response, Earthshine, the pointing dependence
of the zodiacal light, and giant or reddened-star templates (every star gets
a dwarf template).

**Validation status.** The count-rate chain, PSF energy, astrometry and
noise model have been checked against first-principles calculations and
against the wcc-etc exposure-time calculator, including a Monte Carlo
cross-check of the ETC's signal and noise predictions on simulated frames.
Nothing has yet been compared with measured data; issue #25 holds the plan.

## Documentation

The full documentation site (installation, a user guide, rendered tutorial
notebooks, a CLI reference and an auto-generated API reference) is built with
**Sphinx** and the *Read the Docs* theme. It lives under `docs/sphinx/`:

```bash
pip install -e ".[docs]"   # Sphinx + theme + nbsphinx (+ the package itself)
brew install pandoc        # or: conda install pandoc / apt-get install pandoc
cd docs/sphinx
make html                  # output in _build/html/index.html
```

The rendered site is published on Read the Docs at
<https://wcc-sim.readthedocs.io/>; `.readthedocs.yaml` configures the build.

## Tests

```sh
pip install -e ".[test]"
python -m pytest
```

## Citation and acknowledgements
If you use wcc-sim, please cite it using the metadata in `CITATION.cff`
(GitHub renders it as a "Cite this repository" button).

This work has made use of data from the European Space Agency (ESA) mission
*Gaia* (https://www.cosmos.esa.int/gaia), processed by the Gaia Data
Processing and Analysis Consortium (DPAC); the Gaia tables redistributed in
`notebooks/gaia_cache/` are subject to the
[Gaia data credits](https://www.cosmos.esa.int/web/gaia-users/credits).
Stellar spectra are the Pickles (1998, PASP 110, 863) library; the Gaia
passbands come from the SVO Filter Profile Service (Rodrigo & Solano 2020,
http://svo2.cab.inta-csic.es/theory/fps/); reddening uses the Fitzpatrick
(1999) curve via [dust_extinction](https://dust-extinction.readthedocs.io).
Simulations are built on [Astropy](https://www.astropy.org),
[photutils](https://photutils.readthedocs.io) and
[synphot](https://synphot.readthedocs.io).
