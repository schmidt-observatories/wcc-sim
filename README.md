# wcc-sim

End-to-end photometric image simulator for the Lazuli Wide-field Context
Camera (WCC). Queries Gaia DR3 for a given RA/Dec, converts Gaia photometry
to detector count rates through the [wcc-etc](../wcc-etc4/wcc-etc) instrument
model, renders every star with the in-focus (Airy) or +1/+2-wave defocus
(Zemax Huygens) PSF, adds photon/sky/dark/read noise with full-well + ADC
saturation, and writes a multi-extension FITS image (SCI + SATMASK + CAT +
CLEAN) with a TAN WCS.

## Install

Requires the `py313` env (wcc_etc installed editable there):

    ~/anaconda3/envs/py313/bin/python -m pip install -e . --no-deps

## Usage

Python:

    from wcc_sim import simulate_field
    field = simulate_field(150.1, 2.2, sensorfilter="zwo:r", focus=1,
                           exptime=90, seed=42, output="field_1wave.fits")

CLI:

    wcc-sim --ra 150.1 --dec 2.2 --sensorfilter zwo:r --focus 1 \
            --exptime 90 --seed 42 -o field_1wave.fits

Detectors: `zwo:*` = Sony IMX455, 9568x6380 px, 16.87 mas/pix;
`qcmos:*` = Hamamatsu HWK4123, 4096x2304 px, 20.64 mas/pix.
`--focus 0|1|2` selects in-focus / +1 wave / +2 waves defocus PSFs.

## Photometry (wcc-phot)

The sibling `wcc_phot` package extracts differential aperture or PSF
photometry from a series of wcc-sim frames of the same field: it picks the
target (Gaia source_id or RA/Dec) plus the N=10 best reference stars
(unsaturated, isolated, away from edges, closest in G to the target),
re-centroids every star in every frame from its WCS-predicted position,
and builds a relative light curve against the reference ensemble.

Python:

    from wcc_phot import run_photometry
    result = run_photometry(["f0.fits", "f1.fits"], target=(150.1, 2.2),
                            method="aperture", n_ref=10)
    result.lightcurve  # per frame: rel_flux_norm, errors, flags

CLI:

    wcc-phot f*.fits --ra 150.1 --dec 2.2 --method psf -o phot.fits

Add `--live` for a matplotlib window that updates as each frame is
analyzed (image + apertures on the left, growing light curve on the
right); in Python, pass `on_frame=wcc_phot.LiveViewer()` or any callable
to hook custom displays. See `scripts/example_photometry.sh` and
`notebooks/08_wcc_phot_photometry.ipynb`.

Default aperture geometry (radius, annulus, centroid box) comes from the
95% encircled-energy radius of the wcc-sim PSF model for the frames'
sensorfilter/focus/jitter; PSF mode fits that same model with photutils.
Output FITS extensions: STARS (selection), PHOT (per star per frame),
LC (per frame).

## Tests

    ~/anaconda3/envs/py313/bin/python -m pytest

## Documentation

Sphinx docs (user guide, tutorial notebooks, CLI reference, API) live in
`docs/sphinx/`. Build with the same env (needs `pip install -e ".[docs]"`
once, plus a `pandoc` binary):

    cd docs/sphinx && make html
    open _build/html/index.html

Tutorial notebooks are in `notebooks/` (executed outputs included; the Gaia
queries they need are cached in `notebooks/gaia_cache/`, so they run offline).
Ready-to-run CLI examples are in `scripts/example_*.sh`.
