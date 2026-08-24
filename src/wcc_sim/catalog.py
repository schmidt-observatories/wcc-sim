"""Gaia DR3 cone-search catalog with a simple on-disk cache."""

import os
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


def build_adql(ra_deg, dec_deg, radius_arcsec, mag_limit):
    radius_deg = radius_arcsec / 3600.0
    return (
        f"SELECT {', '.join(COLUMNS)} FROM gaiadr3.gaia_source "
        f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})) "
        f"AND phot_g_mean_mag <= {mag_limit}"
    )


def _run_query(adql):
    """Synchronous Gaia TAP query (network). Kept thin so tests can patch it."""
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = -1
    job = Gaia.launch_job(adql)
    return job.get_results()


def _empty_table():
    return Table({c: np.array([], dtype=np.int64 if c == "source_id" else float)
                  for c in COLUMNS})


def query_gaia(ra_deg, dec_deg, radius_arcsec, mag_limit=21.0, cache_dir=None):
    """Gaia DR3 sources within radius_arcsec of (ra, dec), G <= mag_limit."""
    cache_file = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        key = f"gaia_{ra_deg:.6f}_{dec_deg:+.6f}_{radius_arcsec:.1f}_{mag_limit:.2f}"
        cache_file = os.path.join(cache_dir, key + ".ecsv")
        if os.path.exists(cache_file):
            cached = Table.read(cache_file, format="ascii.ecsv")
            if set(COLUMNS).issubset(cached.colnames):
                return cached
            # written before the astrometry columns existed: using it would
            # contribute zero proper motion for the whole field.
            # brightcat.query_bright carries the same guard for the XHIP
            # cache; the two are deliberately duplicated (different
            # signatures and cache keys) and must stay in step.

    result = _run_query(build_adql(ra_deg, dec_deg, radius_arcsec, mag_limit))
    if len(result) == 0:
        warnings.warn(
            f"No Gaia sources within {radius_arcsec:.0f}\" of "
            f"({ra_deg}, {dec_deg}) to G={mag_limit}; image will be sky-only.",
            UserWarning,
        )
        result = _empty_table()
    # Normalize column names (astroquery capitalization varies across versions).
    result.rename_columns(result.colnames, [c.lower() for c in result.colnames])

    if cache_file is not None:
        result.write(cache_file, format="ascii.ecsv", overwrite=True)
    return result
