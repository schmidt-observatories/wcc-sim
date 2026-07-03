#!/usr/bin/env bash
# Photometry example: simulate a short dithered series of the same Kepler-field
# pointing, then extract a differential light curve of a bright clean star
# with wcc-phot (aperture mode; swap --method psf to fit the PSF model).
# Add --live to watch the frames, apertures, and light curve update as each
# image is analyzed. Run from the repository root with the wcc-sim env active;
# only the first run needs network (Gaia queries are cached).
set -euo pipefail
mkdir -p smoke_out

# 5 x 90 s frames with sub-pixel pointing offsets (the 16.87 mas px = 4.7e-6 deg)
i=0
for ra in 291.0 291.000002 290.999998 291.000001 290.999999; do
  wcc-sim --ra "$ra" --dec 44.5 --sensorfilter zwo:r \
          --exptime 90 --seed "$((42 + i))" \
          --shape 4096 4096 --cache-dir smoke_out/cache \
          -o "smoke_out/series${i}.fits"
  i=$((i + 1))
done

# target = the unsaturated in-image star closest to G=18.5 (the IMX455 well
# is ~16 ke-, so brighter stars saturate at 90 s in focus)
TARGET=$(python - <<'EOF'
import numpy as np
from astropy.io import fits

cat = fits.getdata("smoke_out/series0.fits", "CAT")
ok = cat["in_image"] & ~cat["saturated"]
best = np.argmin(np.abs(cat["phot_g_mean_mag"][ok] - 18.5))
print(int(cat["source_id"][ok][best]))
EOF
)

wcc-phot smoke_out/series*.fits \
         --source-id "$TARGET" --n-ref 10 --method aperture \
         --lc-csv smoke_out/lightcurve.ecsv \
         -o smoke_out/photometry.fits
