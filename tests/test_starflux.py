import numpy as np
import pytest
from astropy.table import Table

SENSORFILTER = "zwo:r"


def test_gaia_g_bandpass_loads():
    from wcc_sim.starflux import gaia_g_bandpass

    bp = gaia_g_bandpass()
    # Transmissive at 600 nm, dead at 200 and 1200 nm
    assert bp(6000).value > 0.1
    assert bp(2000).value == pytest.approx(0.0, abs=1e-6)
    assert bp(12000).value == pytest.approx(0.0, abs=1e-6)


def test_spt_from_bp_rp_nearest_and_nan():
    from wcc_sim.starflux import spt_from_bp_rp

    spts = spt_from_bp_rp([0.82, 2.7, -0.3, np.nan])
    assert spts[0] == "G2V"
    assert spts[1] == "M4V"
    assert spts[2] in ("O5V", "O9V")
    assert spts[3] == "G2V"  # missing color falls back to G2V


def test_spt_from_bp_rp_masked_column():
    from astropy.table import MaskedColumn

    from wcc_sim.starflux import spt_from_bp_rp

    col = MaskedColumn([0.82, 2.7], mask=[False, True])
    assert list(spt_from_bp_rp(col)) == ["G2V", "G2V"]


def test_rate_scales_exactly_with_magnitude():
    from wcc_sim.starflux import REF_MAG, rate_for_spt, rates_for_catalog

    base = rate_for_spt("G2V", SENSORFILTER)
    assert base > 0
    cat = Table(
        {
            "phot_g_mean_mag": [REF_MAG, REF_MAG + 5.0],
            "phot_bp_mean_mag": [15.82, 20.82],
            "phot_rp_mean_mag": [15.0, 20.0],
        }
    )
    rates, spts = rates_for_catalog(cat, SENSORFILTER)
    assert rates[0] == pytest.approx(base, rel=1e-12)
    assert rates[0] / rates[1] == pytest.approx(100.0, rel=1e-9)
    assert list(spts) == ["G2V", "G2V"]


def test_rate_memoized():
    from wcc_sim import starflux

    starflux.rate_for_spt("K5V", SENSORFILTER)
    n = len(starflux._RATE_CACHE)
    starflux.rate_for_spt("K5V", SENSORFILTER)
    assert len(starflux._RATE_CACHE) == n


def test_sky_and_dark_rates_positive():
    from wcc_sim.detectors import make_base_simulation
    from wcc_sim.starflux import sky_and_dark_rates

    sky, dark = sky_and_dark_rates(make_base_simulation(SENSORFILTER))
    assert sky > 0
    assert dark > 0


def test_rate_reference_pin():
    """Regression pin for the full Gaia-G -> Pickles -> throughput chain.

    Guards against silent changes to the flux normalization (bandpass
    units, magsys, Vega zero point). If wcc_etc throughput data is
    deliberately updated, re-record this value with:
    python -c "from wcc_sim.starflux import rate_for_spt; print(rate_for_spt('G2V', 'zwo:r'))"
    """
    from wcc_sim.starflux import rate_for_spt

    assert rate_for_spt("G2V", "zwo:r") == pytest.approx(38484.388331215, rel=1e-3)


def test_spt_override_column(monkeypatch):
    import wcc_sim.starflux as sf

    rates = {"G2V": 1.0, "M2III": 2.0}
    monkeypatch.setattr(sf, "rate_for_spt", lambda spt, f: rates[str(spt)])
    catalog = Table(
        {
            "phot_g_mean_mag": [15.0, 15.0, 15.0],
            "phot_bp_mean_mag": [np.nan] * 3,
            "phot_rp_mean_mag": [np.nan] * 3,
            # NaN colors -> base type G2V; row 1 overridden (and longer
            # than the base dtype width — must not be truncated)
            "spt": ["", "M2III", ""],
        }
    )
    out_rates, spts = sf.rates_for_catalog(catalog, "zwo:r")
    assert list(spts) == ["G2V", "M2III", "G2V"]
    assert out_rates == pytest.approx([1.0, 2.0, 1.0])
