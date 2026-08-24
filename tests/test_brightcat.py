"""The bright-star supplement: query, normalization, and the merge policy.

Nothing here touches the network -- brightcat._run_query is patched, the way
tests/test_catalog.py already patches the Gaia one. The fixture rows are the
real XHIP values for alpha Cen A and B.
"""

import numpy as np
import pytest
from astropy.table import Table

#: XHIP V/137D rows for HIP 71683 / 71681. Positions are epoch J1991.25
#: despite the RAJ2000 column name.
XHIP_ROWS = Table({
    "HIP": np.array([71683, 71681], dtype=np.int32),
    "RAJ2000": [219.92041034, 219.91412833],
    "DEJ2000": [-60.83514707, -60.83947139],
    "pmRA": [-3678.19, -3600.35],
    "pmDE": [481.84, 952.11],
    "Plx": [742.12, 742.12],
    "RV": [-21.4, -18.6],
    "Vmag": [-0.01, 1.35],
    "B-V": [0.710, 0.900],
    "SpType": ["G2V", "K1V"],
    "Comp": ["A", "B"],
})


@pytest.fixture
def patched_query(monkeypatch):
    """brightcat._run_query -> the alpha Cen rows, with a call counter."""
    import wcc_sim.brightcat as bc

    calls = []

    def fake(ra, dec, radius):
        calls.append((ra, dec, radius))
        return XHIP_ROWS.copy()

    monkeypatch.setattr(bc, "_run_query", fake)
    return calls


def test_query_bright_returns_the_rows(patched_query):
    from wcc_sim.brightcat import query_bright

    out = query_bright(219.9, -60.83, 110.0)
    assert len(out) == 2
    assert set(("HIP", "Vmag", "B-V", "RV")).issubset(out.colnames)
    assert len(patched_query) == 1


def test_query_bright_caches_to_disk(patched_query, tmp_path):
    """A second call with the same cone must not hit the service again, so a
    repeat run of a script works offline."""
    from wcc_sim.brightcat import query_bright

    first = query_bright(219.9, -60.83, 110.0, cache_dir=str(tmp_path))
    second = query_bright(219.9, -60.83, 110.0, cache_dir=str(tmp_path))
    assert len(patched_query) == 1
    assert list(first["HIP"]) == list(second["HIP"])
    assert len(list(tmp_path.glob("xhip_*.ecsv"))) == 1


def test_query_bright_warns_and_degrades_when_the_service_fails(monkeypatch):
    """A VizieR outage must not take the whole simulation down: warn, return
    nothing, and let the caller continue with Gaia alone."""
    import wcc_sim.brightcat as bc

    def boom(ra, dec, radius):
        raise ConnectionError("VizieR closed the connection")

    monkeypatch.setattr(bc, "_run_query", boom)
    with pytest.warns(UserWarning, match="bright-star query failed"):
        out = bc.query_bright(219.9, -60.83, 110.0)
    assert len(out) == 0
    assert set(bc.XHIP_COLUMNS).issubset(out.colnames)


# --------------------------------------------------------------------------- #
# Normalization to Gaia-shaped rows                                            #
# --------------------------------------------------------------------------- #

GAIA_LIKE_COLUMNS = (
    "source_id", "ra", "dec", "phot_g_mean_mag", "phot_bp_mean_mag",
    "phot_rp_mean_mag", "pmra", "pmdec", "parallax", "radial_velocity",
    "spt", "catalog",
)


def test_to_gaia_like_produces_the_pipeline_column_set():
    from wcc_sim.brightcat import to_gaia_like

    out = to_gaia_like(XHIP_ROWS)
    assert set(GAIA_LIKE_COLUMNS).issubset(out.colnames)
    assert list(out["catalog"]) == ["hipparcos", "hipparcos"]


def test_to_gaia_like_marks_provenance_with_a_negative_source_id():
    """A negative id cannot collide with a Gaia source_id, so a merged row's
    origin is readable straight off the catalog."""
    from wcc_sim.brightcat import to_gaia_like

    assert list(to_gaia_like(XHIP_ROWS)["source_id"]) == [-71683, -71681]


def test_to_gaia_like_converts_v_to_g_through_the_template():
    """alpha Cen A: V = -0.01, B-V = 0.71 -> the template nearest that colour,
    whose synthetic G-V is about -0.166, so G is about -0.18. The published
    (BP-RP) colour term gives -0.15 for the same star; the 0.03 mag spread is
    the accuracy of this path, and 3% in rate."""
    from wcc_sim.brightcat import to_gaia_like

    out = to_gaia_like(XHIP_ROWS)
    assert out["phot_g_mean_mag"][0] == pytest.approx(-0.18, abs=0.04)
    assert out["phot_g_mean_mag"][1] == pytest.approx(1.35 - 0.2, abs=0.1)


def test_to_gaia_like_colours_round_trip_to_the_same_template():
    """The BP and RP magnitudes exist only so the untouched rate path picks
    the same template this row was built from."""
    from wcc_sim.brightcat import to_gaia_like
    from wcc_sim.starflux import spt_from_bp_rp

    out = to_gaia_like(XHIP_ROWS)
    recovered = spt_from_bp_rp(
        np.asarray(out["phot_bp_mean_mag"]) - np.asarray(out["phot_rp_mean_mag"])
    )
    assert list(recovered) == list(out["spt"])


def test_to_gaia_like_keeps_positions_at_the_catalog_epoch():
    """Normalization must not move anything: propagation is the merge's job."""
    from wcc_sim.brightcat import to_gaia_like

    out = to_gaia_like(XHIP_ROWS)
    assert out["ra"][0] == pytest.approx(219.92041034, abs=1e-8)


def test_to_gaia_like_drops_rows_without_a_v_magnitude():
    """No magnitude means no rate; the row would render as a zero-flux star."""
    from wcc_sim.brightcat import to_gaia_like

    rows = XHIP_ROWS.copy()
    rows["Vmag"] = [np.nan, 1.35]
    with pytest.warns(UserWarning, match="no V magnitude"):
        out = to_gaia_like(rows)
    assert list(out["source_id"]) == [-71681]


def test_to_gaia_like_falls_back_to_g2v_without_a_colour():
    from wcc_sim.brightcat import to_gaia_like

    rows = XHIP_ROWS.copy()
    rows["B-V"] = [np.nan, 0.9]
    assert to_gaia_like(rows)["spt"][0] == "G2V"
