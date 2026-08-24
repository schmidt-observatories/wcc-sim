"""Space-motion propagation, checked against an external reference.

The reference rows are HIP 71683 and 71681 (alpha Cen A and B) from VizieR
I/239/hip_main: positions at epoch J1991.25, and VizieR's own computed J2000
positions in the _RA.icrs / _DE.icrs columns. VizieR's computation is proper
motion only, which is what makes it a clean check on the pm path.
"""

import numpy as np
import pytest
from astropy.table import Table

HIP_EPOCH = 1991.25

#: ra, dec at J1991.25; pmra (dRA/dt cos dec), pmdec [mas/yr]; plx [mas]; rv [km/s]
HIP = Table({
    "hip": [71683, 71681],
    "ra": [219.92041034, 219.91412833],
    "dec": [-60.83514707, -60.83947139],
    "pmra": [-3678.19, -3600.35],
    "pmdec": [481.84, 952.11],
    "parallax": [742.12, 742.12],
    "radial_velocity": [-21.4, -18.6],
})

#: VizieR's own _RA.icrs / _DE.icrs for the same two rows (J2000, pm only)
VIZIER_J2000 = np.array([[219.90206584, -60.83397468],
                         [219.89617026, -60.83715604]])


def _sep_mas(table, reference):
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    got = SkyCoord(table["ra"], table["dec"], unit="deg")
    ref = SkyCoord(reference[:, 0], reference[:, 1], unit="deg")
    return got.separation(ref).to(u.mas).value


def test_proper_motion_only_reproduces_vizier_j2000():
    """The external check: pm-only propagation must land on VizieR's own
    numbers, which are also pm-only. Tolerance is 0.5 mas; the observed
    agreement is 0.000 mas."""
    from wcc_sim.astrometry import propagate

    out = propagate(HIP, HIP_EPOCH, 2000.0)
    assert _sep_mas(out, VIZIER_J2000) == pytest.approx([0.0, 0.0], abs=0.5)


def test_parallax_and_rv_add_perspective_acceleration():
    """Adding distance and radial velocity must move alpha Cen A by ~4.6 mas
    at J2000 and ~75 mas at J2026.6 -- 4.5 px at 16.869 mas/px, which is why
    the RV column is queried at all."""
    from wcc_sim.astrometry import propagate

    for epoch, expected in ((2000.0, 4.6), (2026.6, 75.3)):
        pm_only = propagate(HIP, HIP_EPOCH, epoch)
        full = propagate(HIP, HIP_EPOCH, epoch, parallax="parallax",
                         rv="radial_velocity")
        assert _sep_mas(full, np.column_stack(
            [pm_only["ra"], pm_only["dec"]]))[0] == pytest.approx(expected, rel=0.1)


def test_alpha_cen_crosses_the_field_between_catalog_epochs():
    """Why the merge propagates before matching: 24.75 years of Gaia-minus-
    Hipparcos epoch difference is 92 arcsec of motion, and the WCC field is
    162 arcsec wide."""
    from wcc_sim.astrometry import propagate

    moved = propagate(HIP, HIP_EPOCH, 2016.0)
    assert _sep_mas(moved, np.column_stack([HIP["ra"], HIP["dec"]]))[0] / 1000.0 \
        == pytest.approx(92.0, rel=0.05)


def test_missing_parallax_falls_back_to_proper_motion_only():
    """A stand-in distance is used for rows with no parallax; it must not
    perturb the answer."""
    from wcc_sim.astrometry import propagate

    no_plx = HIP.copy()
    no_plx["parallax"] = 0.0
    a = propagate(no_plx, HIP_EPOCH, 2026.6, parallax="parallax",
                  rv="radial_velocity")
    b = propagate(HIP, HIP_EPOCH, 2026.6)
    assert _sep_mas(a, np.column_stack([b["ra"], b["dec"]])) == \
        pytest.approx([0.0, 0.0], abs=1e-3)


def test_missing_parallax_emits_no_erfa_warning():
    """The stand-in distance makes ERFA report 'distance overridden'; that is
    expected and must not reach the user."""
    import warnings
    from wcc_sim.astrometry import propagate

    no_plx = HIP.copy()
    no_plx["parallax"] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        propagate(no_plx, HIP_EPOCH, 2026.6, parallax="parallax")


def test_equal_epochs_and_empty_tables_are_no_ops():
    from wcc_sim.astrometry import propagate

    same = propagate(HIP, 2000.0, 2000.0)
    assert np.array_equal(np.asarray(same["ra"]), np.asarray(HIP["ra"]))
    assert len(propagate(HIP[:0], 1991.25, 2000.0)) == 0


def test_rows_without_proper_motion_stay_put():
    from wcc_sim.astrometry import propagate

    still = HIP.copy()
    still["pmra"] = 0.0
    still["pmdec"] = 0.0
    out = propagate(still, HIP_EPOCH, 2026.6)
    assert np.allclose(np.asarray(out["ra"]), np.asarray(HIP["ra"]))


def test_masked_proper_motion_does_not_move_the_star():
    """A MaskedColumn hides a real number under its mask, and np.asarray()
    hands that number back. Gaia and XHIP both mark unmeasured astrometry
    this way, so a leak here moves a star by a motion nobody measured."""
    from astropy.table import MaskedColumn
    from wcc_sim.astrometry import propagate

    masked = HIP.copy()
    masked["pmra"] = MaskedColumn([-3678.19, -3600.35], mask=[False, True])
    masked["pmdec"] = MaskedColumn([481.84, 952.11], mask=[False, True])
    out = propagate(masked, HIP_EPOCH, 2026.6)
    assert out["ra"][1] == pytest.approx(HIP["ra"][1], abs=1e-12)
    assert out["dec"][1] == pytest.approx(HIP["dec"][1], abs=1e-12)
    assert abs(out["ra"][0] - HIP["ra"][0]) > 1e-4          # row 0 still moves


def test_masked_parallax_falls_back_to_proper_motion_only():
    from astropy.table import MaskedColumn
    from wcc_sim.astrometry import propagate

    masked = HIP.copy()
    masked["parallax"] = MaskedColumn([742.12, 742.12], mask=[True, False])
    out = propagate(masked, HIP_EPOCH, 2026.6, parallax="parallax",
                    rv="radial_velocity")
    pm_only = propagate(HIP, HIP_EPOCH, 2026.6)
    seps = _sep_mas(out, np.column_stack([pm_only["ra"], pm_only["dec"]]))
    assert seps[0] == pytest.approx(0.0, abs=1e-3)   # masked plx -> pm only
    assert seps[1] > 1.0                             # real plx -> RV applies


def test_a_missing_position_column_raises_instead_of_yielding_nan():
    """A mistyped column name must not come back as a silently NaN position."""
    from wcc_sim.astrometry import propagate

    renamed = HIP.copy()
    renamed.rename_column("ra", "RA")
    with pytest.raises(ValueError, match="ra"):
        propagate(renamed, HIP_EPOCH, 2000.0)


# --------------------------------------------------------------------------- #
# Cross-match                                                                  #
# --------------------------------------------------------------------------- #

def _cat(*pairs):
    return Table({"ra": [p[0] for p in pairs], "dec": [p[1] for p in pairs]})


def test_crossmatch_pairs_within_the_radius_only():
    from wcc_sim.astrometry import crossmatch

    a = _cat((10.0, 0.0), (10.01, 0.0))            # second is 36" away
    b = _cat((10.0001, 0.0))                       # 0.36" from the first
    idx_a, idx_b = crossmatch(a, b, 2.0)
    assert list(idx_a) == [0] and list(idx_b) == [0]


def test_crossmatch_is_one_to_one_and_keeps_the_closer_pair():
    """Two bright rows cannot both claim one Gaia row; the closer wins and
    the other is left unmatched, so it gets added rather than dropped."""
    from wcc_sim.astrometry import crossmatch

    a = _cat((10.0002, 0.0), (10.0001, 0.0))       # 0.72" and 0.36" away
    b = _cat((10.0, 0.0))
    idx_a, idx_b = crossmatch(a, b, 5.0)
    assert list(idx_a) == [1] and list(idx_b) == [0]


def test_crossmatch_handles_empty_inputs():
    from wcc_sim.astrometry import crossmatch

    for a, b in ((_cat(), _cat((10.0, 0.0))), (_cat((10.0, 0.0)), _cat()),
                 (_cat(), _cat())):
        idx_a, idx_b = crossmatch(a, b, 2.0)
        assert len(idx_a) == 0 and len(idx_b) == 0
