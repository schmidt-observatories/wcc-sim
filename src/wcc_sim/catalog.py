"""Gaia DR3 cone-search catalog with a simple on-disk cache."""

import os
import time
import warnings

import numpy as np
from astropy.table import Table

COLUMNS = [
    "source_id",
    "ra",
    "dec",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "pmra",
    "pmdec",
    "parallax",
    "radial_velocity",
]

#: Reference epoch of Gaia DR3 positions (Julian year).
GAIA_EPOCH = 2016.0

#: The Gaia archive caps *synchronous* TAP jobs at this many rows and returns
#: an arbitrary subset of an unordered result with no warning. Queries here
#: run asynchronously (uncapped), but a result of exactly this length is still
#: treated as possibly truncated: it raises a warning and is never cached.
SYNC_ROW_CAP = 2000

#: Pointings are rounded to this many arcsec in the cache key, so the frames
#: of a dithered series (sub-pixel to a few arcsec apart) share one query. The
#: cone radius carries a 10 arcsec margin over the detector half-diagonal.
CACHE_ROUND_ARCSEC = 1.0


def build_adql(ra_deg, dec_deg, radius_arcsec, mag_limit):
    radius_deg = radius_arcsec / 3600.0
    return (
        f"SELECT {', '.join(COLUMNS)} FROM gaiadr3.gaia_source "
        f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})) "
        f"AND phot_g_mean_mag <= {mag_limit} "
        "ORDER BY phot_g_mean_mag"
    )


def _run_query(adql, timeout_s=600.0, poll_s=2.0):
    """Asynchronous Gaia TAP query (network). Kept thin so tests can patch it.

    Asynchronous because synchronous jobs are capped at SYNC_ROW_CAP rows
    (ROW_LIMIT = -1 does not lift that). The job is polled so a hung archive
    raises TimeoutError instead of blocking forever.
    """
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = -1
    job = Gaia.launch_job_async(adql, background=True)
    t0 = time.monotonic()
    while job.get_phase(update=True) not in ("COMPLETED", "ERROR", "ABORTED"):
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError(
                f"Gaia query did not finish within {timeout_s:.0f} s "
                f"(job {job.jobid}, phase {job.get_phase()})"
            )
        time.sleep(poll_s)
    return job.get_results()


def _cache_paths(cache_dir, ra_deg, dec_deg, radius_arcsec, mag_limit):
    """(rounded-pointing key, legacy exact key) cache files, in lookup order."""
    step = CACHE_ROUND_ARCSEC / 3600.0
    ra_r = round(ra_deg / step) * step
    dec_r = round(dec_deg / step) * step
    rounded = f"gaia_{ra_r:.6f}_{dec_r:+.6f}_{radius_arcsec:.1f}_{mag_limit:.2f}"
    legacy = f"gaia_{ra_deg:.6f}_{dec_deg:+.6f}_{radius_arcsec:.1f}_{mag_limit:.2f}"
    return [os.path.join(cache_dir, k + ".ecsv") for k in (rounded, legacy)]


def _empty_table():
    return Table({c: np.array([], dtype=np.int64 if c == "source_id" else float)
                  for c in COLUMNS})


def query_gaia(ra_deg, dec_deg, radius_arcsec, mag_limit=21.0, cache_dir=None):
    """Gaia DR3 sources within radius_arcsec of (ra, dec), G <= mag_limit."""
    cache_file = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        candidates = _cache_paths(cache_dir, ra_deg, dec_deg, radius_arcsec, mag_limit)
        cache_file = candidates[0]
        for path in candidates:
            if os.path.exists(path):
                cached = Table.read(path, format="ascii.ecsv")
                if set(COLUMNS).issubset(cached.colnames):
                    return cached
                # written before the astrometry columns existed: using it would
                # contribute zero proper motion for the whole field.
                # brightcat.query_bright carries the same guard for the XHIP
                # cache; the two are deliberately duplicated (different
                # signatures and cache keys) and must stay in step.

    result = _run_query(build_adql(ra_deg, dec_deg, radius_arcsec, mag_limit))
    complete = True
    if len(result) >= SYNC_ROW_CAP:
        complete = False
        warnings.warn(
            f"Gaia returned {len(result)} rows for the cone at ({ra_deg}, "
            f"{dec_deg}), which is the archive's synchronous-job cap of "
            f"{SYNC_ROW_CAP}: the catalog may be truncated (the faintest "
            "stars are dropped first). It will not be cached.",
            UserWarning,
        )
    if len(result) == 0:
        warnings.warn(
            f"No Gaia sources within {radius_arcsec:.0f}\" of "
            f"({ra_deg}, {dec_deg}) to G={mag_limit}; image will be sky-only.",
            UserWarning,
        )
        result = _empty_table()
    # Normalize column names (astroquery capitalization varies across versions).
    result.rename_columns(result.colnames, [c.lower() for c in result.colnames])

    if cache_file is not None and complete:
        result.write(cache_file, format="ascii.ecsv", overwrite=True)
    return result
