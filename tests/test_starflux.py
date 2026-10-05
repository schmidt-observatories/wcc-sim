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

    spts = spt_from_bp_rp([0.86, 2.94, -0.55, np.nan])
    assert spts[0] == "G2V"
    assert spts[1] == "M4V"    # a real M4V (BP-RP 2.94) used to get M5V
    assert spts[2] == "O5V"
    assert spts[3] == "G2V"  # missing color falls back to G2V


def test_spt_from_bp_rp_masked_column():
    from astropy.table import MaskedColumn

    from wcc_sim.starflux import spt_from_bp_rp

    col = MaskedColumn([0.86, 2.7], mask=[False, True])
    assert list(spt_from_bp_rp(col)) == ["G2V", "G2V"]


def test_rate_scales_exactly_with_magnitude():
    from wcc_sim.starflux import (REF_MAG, bp_rp_for_spt, rate_for_spt,
                                  rates_for_catalog)

    base = rate_for_spt("G2V", SENSORFILTER)
    assert base > 0
    bp_rp = bp_rp_for_spt("G2V")[0]   # at a template's own colour the
    cat = Table(                      # interpolation is that template's rate
        {
            "phot_g_mean_mag": [REF_MAG, REF_MAG + 5.0],
            "phot_bp_mean_mag": [REF_MAG + bp_rp, REF_MAG + 5.0 + bp_rp],
            "phot_rp_mean_mag": [REF_MAG, REF_MAG + 5.0],
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


# --------------------------------------------------------------------------- #
# Synthetic Johnson colours (for V-magnitude catalogs like Hipparcos)          #
# --------------------------------------------------------------------------- #

def test_synthetic_b_v_of_the_solar_template_is_the_solar_value():
    """The whole V -> G path rests on this: Pickles G2V through Johnson B and
    V must give the Sun's B-V = 0.65, or the templates are being integrated
    through the wrong passbands."""
    from wcc_sim.starflux import _synthetic_color_table

    spts, b_v, _ = _synthetic_color_table()
    assert b_v[list(spts).index("G2V")] == pytest.approx(0.65, abs=0.01)


def test_synthetic_g_minus_v_is_near_the_published_colour_term():
    """-0.164 synthetic vs -0.14 from the published (BP-RP) relation; the gap
    is that relation's own scatter, so the tolerance is deliberately loose."""
    from wcc_sim.starflux import g_minus_v

    assert g_minus_v("G2V")[0] == pytest.approx(-0.15, abs=0.03)


def test_every_template_has_a_synthetic_colour():
    from wcc_sim.starflux import _spt_table, _synthetic_color_table

    assert set(_spt_table()[0]) == set(_synthetic_color_table()[0])


def test_spt_from_b_v_picks_the_nearest_template():
    from wcc_sim.starflux import spt_from_b_v

    assert spt_from_b_v(0.65)[0] == "G2V"
    assert list(spt_from_b_v([0.0, 1.45])) == ["A0V", "M2V"]


def test_g_minus_v_at_b_v_is_continuous_in_colour():
    """The bug this pins. Taking the *nearest template's* G-V makes the
    derived magnitude a step function of colour: between M2V (B-V 1.461,
    G-V -0.859) and M4V (1.618, -1.408) it jumps 0.549 mag at the midpoint,
    a factor 1.66 in rendered flux, for a 0.015 mag change in B-V -- inside
    Hipparcos's own B-V error for a red star, and enough to flip the
    replace_mag = 6.0 decision."""
    import numpy as np

    from wcc_sim.starflux import _synthetic_color_table, g_minus_v_at_b_v

    _, b_v, _ = _synthetic_color_table()
    grid = np.arange(b_v.min() - 0.1, b_v.max() + 0.1, 0.001)
    values = g_minus_v_at_b_v(grid)
    window = 20                                   # 0.001 mag steps -> 0.02
    jump = np.max(np.abs(values[window:] - values[:-window]))
    assert jump < 0.1, f"0.02 mag in B-V moves G-V by {jump:.3f} mag"


def test_g_minus_v_at_b_v_matches_the_table_at_a_tabulated_colour():
    """Interpolation must not shift the anchors: at a template's own colour
    the interpolated value is that template's."""
    from wcc_sim.starflux import g_minus_v, g_minus_v_at_b_v

    assert g_minus_v_at_b_v(0.6502)[0] == pytest.approx(g_minus_v("G2V")[0],
                                                        abs=1e-6)


def test_g_minus_v_at_b_v_clamps_outside_the_table():
    """A colour redder or bluer than any template gets the end value rather
    than an extrapolation off the end of the sequence."""
    from wcc_sim.starflux import g_minus_v, g_minus_v_at_b_v

    assert g_minus_v_at_b_v(5.0)[0] == pytest.approx(g_minus_v("M5V")[0])
    assert g_minus_v_at_b_v(-5.0)[0] == pytest.approx(g_minus_v("O5V")[0])


def test_g_minus_v_at_b_v_falls_back_to_g2v_without_a_colour():
    """Matching spt_from_b_v's own NaN fallback, so a colourless row is
    treated as solar by both halves of the conversion."""
    import numpy as np

    from wcc_sim.starflux import g_minus_v, g_minus_v_at_b_v

    assert g_minus_v_at_b_v(np.nan)[0] == pytest.approx(g_minus_v("G2V")[0])


def test_g_minus_v_at_b_v_is_not_confused_by_the_non_monotonic_sequence():
    """The synthetic sequence is not sorted in B-V: O9V (-0.322) sits redward
    of B0V (-0.332), and M0V (1.346) blueward of K7V (1.394). np.interp on the
    unsorted table would return nonsense there."""
    from wcc_sim.starflux import g_minus_v, g_minus_v_at_b_v

    assert g_minus_v_at_b_v(-0.3218)[0] == pytest.approx(g_minus_v("O9V")[0],
                                                         abs=1e-6)
    assert g_minus_v_at_b_v(1.3937)[0] == pytest.approx(g_minus_v("K7V")[0],
                                                        abs=1e-6)


def test_spt_from_b_v_falls_back_to_g2v_without_a_colour():
    """Same convention as spt_from_bp_rp, so a missing colour behaves the
    same whichever catalog the row came from."""
    from wcc_sim.starflux import spt_from_b_v

    assert spt_from_b_v(np.nan)[0] == "G2V"


def test_bp_rp_for_spt_round_trips_through_the_type_lookup():
    """to_gaia_like sets BP-RP from this, so spt_from_bp_rp must return the
    type it came from -- otherwise a merged row's rate uses another template."""
    from wcc_sim.starflux import bp_rp_for_spt, spt_from_bp_rp

    for spt in ("A0V", "G2V", "K5V", "M4V"):
        assert spt_from_bp_rp(bp_rp_for_spt(spt))[0] == spt


# --------------------------------------------------------------------------- #
# Colour interpolation and masked catalog values (#21, #10)                   #
# --------------------------------------------------------------------------- #

def test_synthetic_bp_rp_matches_the_issue_numbers():
    """K7V 1.72 and M4V 2.90 are what the templates actually integrate to;
    the hand-typed table said 1.55 and 2.71."""
    from wcc_sim.starflux import bp_rp_for_spt

    assert bp_rp_for_spt("K7V")[0] == pytest.approx(1.72, abs=0.01)
    assert bp_rp_for_spt("M4V")[0] == pytest.approx(2.90, abs=0.01)
    assert bp_rp_for_spt("G2V")[0] == pytest.approx(0.86, abs=0.01)


def test_rate_is_continuous_in_colour():
    """The bug this pins: nearest-template rates stepped by 40% between M2V
    and M4V. 0.02 mag of BP-RP must never move the rate by more than 5%."""
    from wcc_sim.starflux import _spt_table, log_rate_at_bp_rp

    _, colors = _spt_table()
    grid = np.arange(colors.min() - 0.1, colors.max() + 0.1, 0.001)
    log_rate = log_rate_at_bp_rp(grid, SENSORFILTER)
    window = 20                                   # 0.001 mag steps -> 0.02
    jump = np.max(np.abs(log_rate[window:] - log_rate[:-window]))
    assert 10 ** jump < 1.05, f"0.02 mag in BP-RP moves the rate by {10**jump:.3f}x"


def test_rate_at_a_template_colour_is_that_template():
    from wcc_sim.starflux import bp_rp_for_spt, log_rate_at_bp_rp, rate_for_spt

    for spt in ("A0V", "K7V", "M4V"):
        got = 10 ** log_rate_at_bp_rp(bp_rp_for_spt(spt), SENSORFILTER)[0]
        assert got == pytest.approx(rate_for_spt(spt, SENSORFILTER), rel=1e-9)


def test_rate_clamps_outside_the_colour_range_and_falls_back_without_one():
    from wcc_sim.starflux import log_rate_at_bp_rp, rate_for_spt

    assert 10 ** log_rate_at_bp_rp(9.0, SENSORFILTER)[0] == pytest.approx(
        rate_for_spt("M5V", SENSORFILTER))
    assert 10 ** log_rate_at_bp_rp(np.nan, SENSORFILTER)[0] == pytest.approx(
        rate_for_spt("G2V", SENSORFILTER))


def test_masked_bp_is_a_missing_colour_not_the_value_under_the_mask():
    """#10: a masked BP with 0 underneath used to select O5V (24% off the
    documented G2V fallback). Masked and NaN must give identical results."""
    from astropy.table import MaskedColumn

    from wcc_sim.starflux import color_fallback, rates_for_catalog

    masked = Table({
        "phot_g_mean_mag": [18.0],
        "phot_bp_mean_mag": MaskedColumn([0.0], mask=[True]),
        "phot_rp_mean_mag": [17.65],
    })
    nan = Table({
        "phot_g_mean_mag": [18.0],
        "phot_bp_mean_mag": [np.nan],
        "phot_rp_mean_mag": [17.65],
    })
    r_masked, s_masked = rates_for_catalog(masked, SENSORFILTER)
    r_nan, s_nan = rates_for_catalog(nan, SENSORFILTER)
    assert list(s_masked) == list(s_nan) == ["G2V"]
    assert r_masked[0] == pytest.approx(r_nan[0], rel=1e-12)
    assert color_fallback(masked).tolist() == [True]


def test_masked_g_gives_a_nan_rate_not_a_bright_star():
    from astropy.table import MaskedColumn

    from wcc_sim.starflux import rates_for_catalog

    cat = Table({
        "phot_g_mean_mag": MaskedColumn([0.0, 15.0], mask=[True, False]),
        "phot_bp_mean_mag": [15.9, 15.9],
        "phot_rp_mean_mag": [15.0, 15.0],
    })
    rates, _ = rates_for_catalog(cat, SENSORFILTER)
    assert np.isnan(rates[0]) and np.isfinite(rates[1])


def test_color_fallback_respects_the_spt_override():
    from wcc_sim.starflux import color_fallback

    cat = Table({
        "phot_bp_mean_mag": [np.nan, np.nan, 16.0],
        "phot_rp_mean_mag": [15.0, 15.0, 15.0],
        "spt": ["", "F8I", ""],
    })
    assert color_fallback(cat).tolist() == [True, False, False]


def test_spt_from_b_v_honours_a_masked_column():
    from astropy.table import MaskedColumn

    from wcc_sim.starflux import g_minus_v, g_minus_v_at_b_v, spt_from_b_v

    col = MaskedColumn([0.0, 0.65], mask=[True, False])
    assert list(spt_from_b_v(col)) == ["G2V", "G2V"]
    assert g_minus_v_at_b_v(col)[0] == pytest.approx(g_minus_v("G2V")[0])


def test_masked_values_survive_the_ecsv_cache_round_trip(tmp_path):
    """The Gaia cache is ECSV; a masked BP must still be a missing colour
    after astropy reads it back (it stores 0.0 under the mask)."""
    from astropy.table import MaskedColumn

    from wcc_sim.starflux import rates_for_catalog

    cat = Table({
        "phot_g_mean_mag": [18.0],
        "phot_bp_mean_mag": MaskedColumn([0.0], mask=[True]),
        "phot_rp_mean_mag": [17.65],
    })
    path = tmp_path / "cache.ecsv"
    cat.write(path, format="ascii.ecsv")
    back = Table.read(path, format="ascii.ecsv")
    r0, s0 = rates_for_catalog(cat, SENSORFILTER)
    r1, s1 = rates_for_catalog(back, SENSORFILTER)
    assert list(s1) == list(s0) == ["G2V"]
    assert r1[0] == pytest.approx(r0[0], rel=1e-12)
