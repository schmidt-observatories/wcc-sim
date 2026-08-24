"""The bright-star supplement: query, normalization, and the merge policy.

Nothing here touches the network -- brightcat._run_query is patched, the way
tests/test_catalog.py already patches the Gaia one. The fixture rows are the
real XHIP values for alpha Cen A and B.
"""

import numpy as np
import pytest
from astropy.table import MaskedColumn, Table

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
    assert out.meta["bright_query_ok"] is False


def test_query_bright_requeries_a_cache_missing_columns(patched_query, tmp_path):
    """The guard catalog.query_gaia already has. The next time XHIP_COLUMNS
    grows, a cache written before that would be handed back and raise
    KeyError out of to_gaia_like mid-simulation."""
    from wcc_sim.brightcat import query_bright

    stale = XHIP_ROWS.copy()
    del stale["RV"]
    key = f"xhip_{219.9:.6f}_{-60.83:+.6f}_{110.0:.1f}.ecsv"
    stale.write(tmp_path / key, format="ascii.ecsv")

    out = query_bright(219.9, -60.83, 110.0, cache_dir=str(tmp_path))
    assert len(patched_query) == 1                  # re-queried, not trusted
    assert "RV" in out.colnames
    # and the stale file is overwritten, so it self-heals
    from astropy.table import Table as _Table
    assert "RV" in _Table.read(tmp_path / key, format="ascii.ecsv").colnames


def test_query_bright_degrades_on_a_malformed_response(monkeypatch):
    """The deliberately broad `except` covers only the network call. A VizieR
    schema surprise -- a column simply absent from the response -- would sail
    past it and raise KeyError out of to_gaia_like instead, taking the run
    down anyway."""
    import wcc_sim.brightcat as bc

    truncated = XHIP_ROWS.copy()
    del truncated["Vmag"]
    monkeypatch.setattr(bc, "_run_query", lambda ra, dec, radius: truncated)
    with pytest.warns(UserWarning, match="unexpected columns|missing columns"):
        out = bc.query_bright(219.9, -60.83, 110.0)
    assert len(out) == 0
    assert out.meta["bright_query_ok"] is False
    assert set(bc.XHIP_COLUMNS).issubset(out.colnames)


def test_a_malformed_response_is_not_cached(monkeypatch, tmp_path):
    """Writing it would make the next offline run fail the same way."""
    import wcc_sim.brightcat as bc

    truncated = XHIP_ROWS.copy()
    del truncated["Vmag"]
    monkeypatch.setattr(bc, "_run_query", lambda ra, dec, radius: truncated)
    with pytest.warns(UserWarning):
        bc.query_bright(219.9, -60.83, 110.0, cache_dir=str(tmp_path))
    assert list(tmp_path.glob("xhip_*.ecsv")) == []


def test_query_bright_flags_an_empty_cone_as_a_successful_query(monkeypatch):
    """The WCC field is 162" x 108", so most pointings hold no XHIP row at
    all. That must not be reported the way a VizieR outage is."""
    import wcc_sim.brightcat as bc

    monkeypatch.setattr(bc, "_run_query",
                        lambda ra, dec, radius: bc._empty_table())
    out = bc.query_bright(219.9, -60.83, 110.0)
    assert len(out) == 0
    assert out.meta["bright_query_ok"] is True


def test_query_bright_flag_survives_the_cache(patched_query, tmp_path):
    """The flag has to reach `merge` on a repeat offline run too."""
    from wcc_sim.brightcat import query_bright

    query_bright(219.9, -60.83, 110.0, cache_dir=str(tmp_path))
    cached = query_bright(219.9, -60.83, 110.0, cache_dir=str(tmp_path))
    assert len(patched_query) == 1
    assert cached.meta["bright_query_ok"] is True


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


def test_to_gaia_like_derives_g_continuously_in_colour():
    """The magnitude conversion must be interpolated, not stepped. Two red
    stars 0.015 mag apart in B-V straddle the M2V/M4V boundary; with the
    nearest template's G-V their derived G differed by 0.55 mag -- a factor
    1.66 in rendered flux -- for the same V."""
    from wcc_sim.brightcat import to_gaia_like

    rows = XHIP_ROWS.copy()
    rows["Vmag"] = [6.0, 6.0]
    rows["B-V"] = [1.530, 1.545]
    g = to_gaia_like(rows)["phot_g_mean_mag"]
    assert abs(g[0] - g[1]) < 0.1


def test_to_gaia_like_still_picks_the_discrete_template_for_the_sed():
    """Only the magnitude is interpolated: the SED, and so the BP/RP
    round-trip, still comes from the nearest template."""
    from wcc_sim.brightcat import to_gaia_like

    rows = XHIP_ROWS.copy()
    rows["B-V"] = [1.530, 1.545]
    assert list(to_gaia_like(rows)["spt"]) == ["M2V", "M4V"]


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


def test_to_gaia_like_drops_rows_without_a_finite_position():
    """A non-finite RA/Dec reaches match_to_catalog_sky, which raises
    "Matching coordinates cannot contain NaN entries" -- one bad row would
    take down a whole simulation in a module whose contract is that a
    bright-catalog failure is never fatal."""
    from wcc_sim.brightcat import to_gaia_like

    rows = XHIP_ROWS.copy()
    rows["RAJ2000"] = [np.nan, 219.91412833]
    with pytest.warns(UserWarning, match="no usable position"):
        out = to_gaia_like(rows)
    assert list(out["source_id"]) == [-71681]


def test_to_gaia_like_drops_a_masked_position_after_a_cache_round_trip(tmp_path):
    """The masked-column case: a masked DEJ2000 read back as 0.0 would place
    the star on the celestial equator instead of dropping it."""
    from wcc_sim.brightcat import to_gaia_like

    rows = XHIP_ROWS.copy()
    rows["DEJ2000"] = MaskedColumn([-60.83514707, -60.83947139],
                                   mask=[False, True])
    path = tmp_path / "xhip_pos.ecsv"
    rows.write(path, format="ascii.ecsv")
    back = Table.read(path, format="ascii.ecsv")

    with pytest.warns(UserWarning, match="no usable position"):
        out = to_gaia_like(back)
    assert list(out["source_id"]) == [-71683]


def test_to_gaia_like_falls_back_to_g2v_without_a_colour():
    from wcc_sim.brightcat import to_gaia_like

    rows = XHIP_ROWS.copy()
    rows["B-V"] = [np.nan, 0.9]
    assert to_gaia_like(rows)["spt"][0] == "G2V"


def test_masked_v_magnitude_is_dropped_after_a_cache_round_trip(tmp_path):
    """The cache is ECSV, and astropy reads a masked float back with 0.0 under
    the mask. A star with no measured V would come back as V = 0.0 -- a
    spurious first-magnitude star -- unless the mask itself is read."""
    from wcc_sim.brightcat import to_gaia_like

    rows = XHIP_ROWS.copy()
    rows["Vmag"] = MaskedColumn([-0.01, 8.0], mask=[False, True])
    path = tmp_path / "xhip.ecsv"
    rows.write(path, format="ascii.ecsv")
    back = Table.read(path, format="ascii.ecsv")

    with pytest.warns(UserWarning, match="no V magnitude"):
        out = to_gaia_like(back)
    assert list(out["source_id"]) == [-71683]


def test_masked_colour_falls_back_to_g2v_after_a_cache_round_trip(tmp_path):
    """A masked B-V read back as 0.0 would select A0V, not the documented
    G2V fallback -- the star would be rendered with the wrong spectrum."""
    from wcc_sim.brightcat import to_gaia_like

    rows = XHIP_ROWS.copy()
    rows["B-V"] = MaskedColumn([0.710, 0.900], mask=[True, False])
    path = tmp_path / "xhip_colour.ecsv"
    rows.write(path, format="ascii.ecsv")
    back = Table.read(path, format="ascii.ecsv")

    assert to_gaia_like(back)["spt"][0] == "G2V"


def test_masked_columns_are_honoured_without_a_round_trip():
    """The in-memory case too: np.asarray on a MaskedColumn returns the value
    under the mask, whatever that value happens to be."""
    from wcc_sim.brightcat import to_gaia_like

    rows = XHIP_ROWS.copy()
    rows["Vmag"] = MaskedColumn([-0.01, 8.0], mask=[False, True])
    with pytest.warns(UserWarning, match="no V magnitude"):
        out = to_gaia_like(rows)
    assert list(out["source_id"]) == [-71683]


# --------------------------------------------------------------------------- #
# The merge                                                                    #
# --------------------------------------------------------------------------- #

def _gaia(*rows):
    """A minimal Gaia-shaped table: (source_id, ra, dec, g)."""
    return Table({
        "source_id": np.array([r[0] for r in rows], dtype=np.int64),
        "ra": [r[1] for r in rows],
        "dec": [r[2] for r in rows],
        "phot_g_mean_mag": [r[3] for r in rows],
        "phot_bp_mean_mag": [r[3] + 0.4 for r in rows],
        "phot_rp_mean_mag": [r[3] - 0.4 for r in rows],
        "pmra": [0.0] * len(rows),
        "pmdec": [0.0] * len(rows),
        "parallax": [0.0] * len(rows),
        "radial_velocity": [0.0] * len(rows),
    })


def test_merge_adds_bright_stars_gaia_does_not_have():
    """The gap-filling case, and the whole point of the feature."""
    from wcc_sim.brightcat import merge

    gaia = _gaia((1, 219.95, -60.9, 14.3))
    merged, info = merge(gaia, XHIP_ROWS, epoch=2000.0)
    assert info["n_bright_added"] == 2
    assert info["n_bright_replaced"] == 0
    assert info["bright_catalog"] == "hipparcos"
    assert info["epoch"] == pytest.approx(2000.0)
    assert len(merged) == 3
    assert list(merged["catalog"]).count("hipparcos") == 2


def test_merge_completes_when_a_bright_row_has_no_position():
    """The failure this guards: the NaN row used to reach crossmatch and
    raise, so the merge -- and the simulation -- died."""
    from wcc_sim.brightcat import merge

    rows = XHIP_ROWS.copy()
    rows["RAJ2000"] = [np.nan, 219.91412833]
    with pytest.warns(UserWarning, match="no usable position"):
        merged, info = merge(_gaia((1, 219.95, -60.9, 14.3)), rows,
                             epoch=2000.0)
    assert info["n_bright_added"] == 1
    assert len(merged) == 2


def test_merge_puts_the_bright_rows_first():
    """Row 0 is the row the PSF report decomposes, so the brightest added
    star belongs there."""
    from wcc_sim.brightcat import merge

    merged, _ = merge(_gaia((1, 219.95, -60.9, 14.3)), XHIP_ROWS, epoch=2000.0)
    assert merged["source_id"][0] == -71683
    assert merged["phot_g_mean_mag"][0] < merged["phot_g_mean_mag"][1]


def test_merge_replaces_a_matched_gaia_row_when_the_star_is_bright():
    """Brighter than the threshold, Gaia's photometry is where the saturation
    systematics live, so the Hipparcos row wins."""
    from wcc_sim.brightcat import merge

    # a Gaia entry at alpha Cen A's J2000 position with a nonsense magnitude
    gaia = _gaia((1, 219.90206584, -60.83397468, 11.0))
    merged, info = merge(gaia, XHIP_ROWS[:1], epoch=2000.0)
    assert info["n_bright_replaced"] == 1
    assert info["n_bright_added"] == 0
    assert len(merged) == 1
    assert merged["source_id"][0] == -71683


def test_merge_keeps_the_gaia_row_for_a_faint_match():
    """Fainter than the threshold Gaia is the better source, so the duplicate
    is dropped rather than added twice."""
    from wcc_sim.brightcat import merge

    faint = XHIP_ROWS[:1].copy()
    faint["Vmag"] = [8.0]
    gaia = _gaia((1, 219.90206584, -60.83397468, 7.8))
    merged, info = merge(gaia, faint, epoch=2000.0)
    assert (info["n_bright_added"], info["n_bright_replaced"]) == (0, 0)
    assert list(merged["source_id"]) == [1]


def test_merge_propagates_before_matching():
    """Un-propagated, alpha Cen A sits 92 arcsec from itself between the two
    catalog epochs and would be added as a second star. With propagation the
    same star is recognised as one."""
    from wcc_sim.brightcat import merge

    gaia = _gaia((1, 219.90206584, -60.83397468, 11.0))   # J2000 position
    merged, info = merge(gaia, XHIP_ROWS[:1], epoch=2000.0)
    assert info["n_bright_added"] == 0                     # matched, not added


def test_merge_defaults_to_the_gaia_epoch():
    """epoch=None leaves Gaia positions untouched and brings the bright rows
    to them, so existing simulations do not move."""
    from wcc_sim.brightcat import merge
    from wcc_sim.catalog import GAIA_EPOCH

    gaia = _gaia((1, 219.95, -60.9, 14.3))
    merged, info = merge(gaia, XHIP_ROWS, epoch=None)
    assert info["epoch"] == pytest.approx(GAIA_EPOCH)
    assert merged["ra"][list(merged["source_id"]).index(1)] == \
        pytest.approx(219.95, abs=1e-9)


def test_merge_with_no_bright_rows_leaves_the_catalog_alone():
    """An empty bright query must not reorder or drop anything: this is the
    property that keeps every un-supplemented simulation byte-identical."""
    from wcc_sim.brightcat import _empty_table, merge

    gaia = _gaia((1, 219.95, -60.9, 14.3), (2, 219.96, -60.91, 12.0))
    merged, info = merge(gaia, _empty_table(), epoch=None)
    assert list(merged["source_id"]) == [1, 2]      # input order, unsorted
    assert (info["n_bright_added"], info["n_bright_replaced"]) == (0, 0)
    assert list(merged["spt"]) == ["", ""]      # no override for Gaia rows


def test_merge_reports_an_empty_but_successful_query_as_consulted():
    """A hand-made table, or one from a cone that genuinely holds no bright
    star, counts as consulted: n_bright_added = 0 with the catalog named
    means "nothing to add", which is the common healthy case."""
    from wcc_sim.brightcat import _empty_table, merge

    merged, info = merge(_gaia((1, 219.95, -60.9, 14.3)), _empty_table(),
                         epoch=None)
    assert info["bright_catalog"] == "hipparcos"
    assert info["n_bright_added"] == 0


def test_merge_reports_a_failed_query_as_not_consulted(monkeypatch):
    """Only a query that never reached the service reports None, so the
    report distinguishes "no bright star here" from "we do not know"."""
    import wcc_sim.brightcat as bc

    def boom(ra, dec, radius):
        raise ConnectionError("VizieR closed the connection")

    monkeypatch.setattr(bc, "_run_query", boom)
    with pytest.warns(UserWarning, match="bright-star query failed"):
        empty = bc.query_bright(219.9, -60.83, 110.0)
    merged, info = bc.merge(_gaia((1, 219.95, -60.9, 14.3)), empty, epoch=None)
    assert info["bright_catalog"] is None
    assert info["n_bright_added"] == 0
    assert list(merged["source_id"]) == [1]


def test_merge_marks_gaia_rows_with_an_empty_spt_override():
    """rates_for_catalog treats a non-empty spt as an override; Gaia rows must
    keep using their own BP-RP."""
    from wcc_sim.brightcat import merge

    merged, _ = merge(_gaia((1, 219.95, -60.9, 14.3)), XHIP_ROWS, epoch=2000.0)
    by_id = dict(zip(map(int, merged["source_id"]), merged["spt"]))
    assert by_id[1] == ""
    assert by_id[-71683] != ""
