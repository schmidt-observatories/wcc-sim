import numpy as np
import pytest
from astropy.table import Table

RA0, DEC0 = 150.1, 2.2

# zwo (IMX455) plate scale, deg per pixel offset used by the phot fixtures
_PIX_DEG = 16.87e-3 / 3600.0


def _phot_star_positions():
    """(dx, dy) pixel offsets from the field center for `phot_catalog`."""
    return [
        (0, 0),  # 0: target
        (60, 0), (-60, 20), (0, 62), (0, -60), (55, 55),  # 1-5: refs
        (-55, -55), (58, -52), (-30, -60), (30, 60), (-90, 0),  # 6-10: refs
        (100, -100),  # 11: saturated
        (-75, 60), (-63, 60),  # 12-13: close pair (12 px apart)
        (118, 30),  # 14: too close to the frame edge
    ]


@pytest.fixture(scope="session")
def phot_catalog():
    """15 stars for photometry tests on a 256x256 zwo subarray.

    Index 0 is the G=18.5 target (bright but unsaturated at 90 s — the
    IMX455 well is only ~16 ke-); 1-10 are clean references with
    |G - G_target| = 0.15..1.50 (so the expected selection order is
    exactly 1..10); 11 saturates; 12-13 are a close pair; 14 sits at
    the edge.
    """
    offsets = _phot_star_positions()
    g = np.array(
        [18.5] + [18.5 + 0.15 * k for k in range(1, 11)] + [13.0, 19.5, 19.6, 18.8]
    )
    dec = np.array([DEC0 + dy * _PIX_DEG for _, dy in offsets])
    cos_dec = np.cos(np.radians(DEC0))
    ra = np.array([RA0 + dx * _PIX_DEG / cos_dec for dx, _ in offsets])
    return Table(
        {
            "source_id": np.arange(len(offsets), dtype=np.int64),
            "ra": ra,
            "dec": dec,
            "phot_g_mean_mag": g,
            "phot_bp_mean_mag": g + 0.35,
            "phot_rp_mean_mag": g - 0.35,
        }
    )


@pytest.fixture(scope="session")
def phot_frames(phot_catalog):
    """Three noisy 256x256 frames of `phot_catalog` with sub-pixel dithers."""
    from wcc_sim import simulate_field

    dithers_px = [(0.0, 0.0), (0.37, -0.21), (-0.29, 0.33)]
    frames = []
    for k, (dx, dy) in enumerate(dithers_px):
        frames.append(
            simulate_field(
                RA0 + dx * _PIX_DEG,
                DEC0 + dy * _PIX_DEG,
                sensorfilter="zwo:r",
                catalog=phot_catalog,
                shape=(256, 256),
                stamp_npix=33,
                exptime=90.0,
                seed=100 + k,
            )
        )
    return frames


@pytest.fixture(scope="session")
def single_star_frame():
    """Noiseless frame with only the G=18.5 target, full 129 px PSF stamp."""
    from wcc_sim import simulate_field

    catalog = Table(
        {
            "source_id": np.array([0], dtype=np.int64),
            "ra": [RA0],
            "dec": [DEC0],
            "phot_g_mean_mag": [18.5],
            "phot_bp_mean_mag": [18.85],
            "phot_rp_mean_mag": [18.15],
        }
    )
    return simulate_field(
        RA0,
        DEC0,
        sensorfilter="zwo:r",
        catalog=catalog,
        shape=(256, 256),
        stamp_npix=129,
        exptime=90.0,
        add_noise=False,
    )


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
