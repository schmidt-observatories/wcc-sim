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

## Tests

    ~/anaconda3/envs/py313/bin/python -m pytest
