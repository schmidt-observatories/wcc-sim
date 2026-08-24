#!/usr/bin/env bash
# Focus-sweep example: the same field at all three focus levels (in-focus
# Airy, +1 wave, +2 waves defocus), one FITS per level — the CLI equivalent
# of scripts/smoke_run.py. Run from the repository root with the wcc-sim env
# active; only the first run needs network (queries are cached).
set -euo pipefail
mkdir -p smoke_out

#for focus in 0 1 2; do
#  wcc-sim --ra 291.0 --dec 44.5 --sensorfilter zwo:r --focus "$focus" \
#          --exptime 90 --seed 42 --shape 1024 1024 \
#          --cache-dir smoke_out/cache \
#          -o "smoke_out/field_focus${focus}.fits"
#done

for focus in 0 1 2; do
  wcc-sim --ra 291.0 --dec 44.5 --sensorfilter zwo:r --focus "$focus" \
          --exptime 90 --seed 42 \
          --cache-dir smoke_out/cache \
          -o "smoke_out/field_focus${focus}.fits"
done
