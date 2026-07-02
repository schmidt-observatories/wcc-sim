import numpy as np
import pytest
from astropy.table import Table

RA0, DEC0 = 150.1, 2.2


@pytest.fixture
def canned_catalog():
    """5 stars around the pointing: solar, red, blue, faint, missing-BP.

    Offsets are 1.5 arcsec so every star lands inside a 256x256 subarray
    (256 px x 16.87 mas/px spans only ~4.3 arcsec).
    """
    d = 1.5 / 3600.0  # 1.5 arcsec offsets
    return Table(
        {
            "source_id": np.arange(5, dtype=np.int64),
            "ra": [RA0, RA0 + d, RA0 - d, RA0, RA0 + d / 2],
            "dec": [DEC0, DEC0 + d / 2, DEC0 - d / 2, DEC0 + d, DEC0 - d],
            "phot_g_mean_mag": [12.0, 15.0, 15.0, 20.5, 17.0],
            "phot_bp_mean_mag": [12.4, 16.2, 14.9, 21.0, np.nan],
            "phot_rp_mean_mag": [11.6, 14.0, 15.0, 20.2, 16.4],
        }
    )
