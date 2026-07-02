#!/usr/bin/env bash
# Detector-comparison example: the same pointing on both WCC detectors —
# Sony IMX455 in r (16.87 mas/px) and Hamamatsu qCMOS broadband
# (20.64 mas/px) — as full-frame images for a side-by-side look.
# Run from the repository root with the wcc-sim env active.
set -euo pipefail
mkdir -p smoke_out

wcc-sim --ra 291.0 --dec 44.5 --sensorfilter zwo:r \
        --exptime 90 --seed 42 --cache-dir smoke_out/cache \
        -o smoke_out/field_zwo_r.fits

wcc-sim --ra 291.0 --dec 44.5 --sensorfilter qcmos:bb \
        --exptime 90 --seed 42 --cache-dir smoke_out/cache \
        -o smoke_out/field_qcmos_bb.fits
