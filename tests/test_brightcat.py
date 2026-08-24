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
