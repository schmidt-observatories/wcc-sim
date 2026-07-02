#!/usr/bin/env bash
# Quick-look example: one fast 1024x1024 subarray image of a Kepler-field
# pointing. The Gaia query is cached in smoke_out/cache, so only the first
# run needs network. Run from the repository root with the wcc-sim env active.
set -euo pipefail
mkdir -p smoke_out

wcc-sim --ra 291.0 --dec 44.5 --sensorfilter zwo:r \
        --exptime 90 --seed 42 --shape 1024 1024 \
        --cache-dir smoke_out/cache \
        -o smoke_out/quicklook.fits
