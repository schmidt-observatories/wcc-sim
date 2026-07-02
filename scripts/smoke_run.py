"""Manual end-to-end smoke run on real Gaia data (network required).

Run:  $PY scripts/smoke_run.py
Simulates a 1024x1024 subarray on a Kepler-field pointing at all three
focus levels and writes FITS to ./smoke_out/.
"""

import os
import time

from wcc_sim import simulate_field

os.makedirs("smoke_out", exist_ok=True)
for focus in (0, 1, 2):
    t0 = time.time()
    f = simulate_field(
        ra=291.0, dec=44.5, sensorfilter="zwo:r", focus=focus,
        exptime=90.0, seed=42, shape=(1024, 1024),
        cache_dir="smoke_out/cache",
        output=f"smoke_out/field_focus{focus}.fits",
    )
    print(
        f"focus={focus}: {f.params['n_sources']} Gaia sources, "
        f"{int(f.saturation_mask.sum())} saturated px, "
        f"{time.time() - t0:.1f} s"
    )
