import numpy as np
import pytest
from astropy.table import Table

COLS = ["source_id", "ra", "dec", "phot_g_mean_mag",
        "phot_bp_mean_mag", "phot_rp_mean_mag"]


def fake_table(n=3):
    return Table(
        {
            "source_id": np.arange(n, dtype=np.int64),
            "ra": np.full(n, 150.1),
            "dec": np.full(n, 2.2),
            "phot_g_mean_mag": np.linspace(12, 18, n),
            "phot_bp_mean_mag": np.linspace(12.3, 18.5, n),
            "phot_rp_mean_mag": np.linspace(11.6, 17.2, n),
            "pmra": np.full(n, 1.0),
            "pmdec": np.full(n, 1.0),
            "parallax": np.full(n, 1.0),
            "radial_velocity": np.full(n, 1.0),
        }
    )


def test_build_adql_contains_cone_and_maglimit():
    from wcc_sim.catalog import build_adql

    q = build_adql(150.1, 2.2, 100.0, 20.5)
    assert "gaiadr3.gaia_source" in q
    assert "CIRCLE" in q and "150.1" in q
    assert "phot_g_mean_mag <= 20.5" in q
    # bright stars must survive any truncation
    assert q.rstrip().endswith("ORDER BY phot_g_mean_mag")


def test_query_gaia_uses_run_query(monkeypatch):
    from wcc_sim import catalog

    monkeypatch.setattr(catalog, "_run_query", lambda adql: fake_table())
    t = catalog.query_gaia(150.1, 2.2, 100.0, mag_limit=21.0)
    assert len(t) == 3
    assert set(COLS) <= set(t.colnames)


def test_query_gaia_cache_roundtrip(monkeypatch, tmp_path):
    from wcc_sim import catalog

    calls = []

    def counting(adql):
        calls.append(1)
        return fake_table()

    monkeypatch.setattr(catalog, "_run_query", counting)
    t1 = catalog.query_gaia(150.1, 2.2, 100.0, cache_dir=tmp_path)
    t2 = catalog.query_gaia(150.1, 2.2, 100.0, cache_dir=tmp_path)
    assert len(calls) == 1  # second call served from cache
    assert len(t1) == len(t2) == 3


def test_query_gaia_empty_warns(monkeypatch):
    from wcc_sim import catalog

    monkeypatch.setattr(catalog, "_run_query", lambda adql: Table())
    with pytest.warns(UserWarning, match="No Gaia sources"):
        t = catalog.query_gaia(150.1, 2.2, 100.0)
    assert len(t) == 0
    assert set(COLS) <= set(t.colnames)


def test_columns_include_the_astrometry_needed_to_propagate():
    """Without proper motion, epoch propagation would silently do nothing."""
    from wcc_sim.catalog import COLUMNS

    for name in ("pmra", "pmdec", "parallax", "radial_velocity"):
        assert name in COLUMNS


def test_gaia_epoch_is_dr3s_reference_epoch():
    from wcc_sim.catalog import GAIA_EPOCH

    assert GAIA_EPOCH == pytest.approx(2016.0)


def test_a_cache_written_before_the_new_columns_is_requeried(monkeypatch, tmp_path):
    """An old cache lacks pmra; using it would contribute zero proper motion
    for the whole field, which is worse than a re-query."""
    import wcc_sim.catalog as cat

    stale = Table({"source_id": np.array([1], dtype=np.int64), "ra": [10.0],
                   "dec": [0.0], "phot_g_mean_mag": [12.0],
                   "phot_bp_mean_mag": [12.4], "phot_rp_mean_mag": [11.6]})
    key = "gaia_10.000000_+0.000000_100.0_21.00.ecsv"
    stale.write(tmp_path / key, format="ascii.ecsv")

    calls = []

    def fake(adql):
        calls.append(adql)
        fresh = stale.copy()
        for name in ("pmra", "pmdec", "parallax", "radial_velocity"):
            fresh[name] = [1.0]
        return fresh

    monkeypatch.setattr(cat, "_run_query", fake)
    out = cat.query_gaia(10.0, 0.0, 100.0, cache_dir=str(tmp_path))
    assert len(calls) == 1
    assert "pmra" in out.colnames


def test_query_at_sync_row_cap_warns_and_is_not_cached(monkeypatch, tmp_path):
    """Exactly 2000 rows is the synchronous-job cap: suspect, never cached."""
    from wcc_sim import catalog

    monkeypatch.setattr(
        catalog, "_run_query", lambda adql: fake_table(catalog.SYNC_ROW_CAP)
    )
    with pytest.warns(UserWarning, match="truncated"):
        t = catalog.query_gaia(150.1, 2.2, 100.0, cache_dir=tmp_path)
    assert len(t) == catalog.SYNC_ROW_CAP
    assert list(tmp_path.glob("*.ecsv")) == []


def test_dithered_pointings_share_one_cache_entry(monkeypatch, tmp_path):
    """Sub-arcsec dithers must not re-query Gaia frame after frame."""
    from wcc_sim import catalog

    calls = []

    def counting(adql):
        calls.append(1)
        return fake_table()

    monkeypatch.setattr(catalog, "_run_query", counting)
    px_deg = 16.87e-3 / 3600.0
    for k in range(4):
        catalog.query_gaia(150.1 + 0.3 * k * px_deg, 2.2 - 0.2 * k * px_deg,
                           100.0, cache_dir=tmp_path)
    assert len(calls) == 1


def test_legacy_exact_key_cache_is_still_read(monkeypatch, tmp_path):
    """Caches written before pointing rounding keep notebooks offline."""
    from wcc_sim import catalog

    ra, dec = 150.1000031, 2.2000017  # not on the 1 arcsec grid
    legacy = tmp_path / f"gaia_{ra:.6f}_{dec:+.6f}_100.0_21.00.ecsv"
    fake_table(5).write(legacy, format="ascii.ecsv")
    monkeypatch.setattr(
        catalog, "_run_query", lambda adql: pytest.fail("should hit the cache")
    )
    assert len(catalog.query_gaia(ra, dec, 100.0, cache_dir=tmp_path)) == 5
