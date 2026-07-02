import warnings

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
        }
    )


def test_build_adql_contains_cone_and_maglimit():
    from wcc_sim.catalog import build_adql

    q = build_adql(150.1, 2.2, 100.0, 20.5)
    assert "gaiadr3.gaia_source" in q
    assert "CIRCLE" in q and "150.1" in q
    assert "phot_g_mean_mag <= 20.5" in q


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
