import pytest
from astropy import units as u

from wcc_sim import fluxrate
from wcc_sim.detectors import make_base_simulation
from wcc_sim.fluxrate import flux_to_rate

SF = "zwo:r"


def test_jy_per_pixel_reference_value():
    # flat F_nu source of 1 Jy through telescope + WCC r (wcc_etc 0.1.0)
    assert flux_to_rate("Jy/pixel", SF) == pytest.approx(1.0767e7, rel=1e-3)


@pytest.mark.parametrize("unit, scale", [("mJy/pix", 1e-3), ("Jy", 1.0), ("Jy/pix", 1.0)])
def test_flux_density_per_pixel_variants(unit, scale):
    assert flux_to_rate(unit, SF) == pytest.approx(scale * flux_to_rate("Jy/pixel", SF))


def test_surface_brightness_uses_pixel_area():
    area = 0.0123
    per_pixel = flux_to_rate("Jy/pixel", SF)
    assert flux_to_rate("Jy/arcsec2", SF, area) == pytest.approx(area * per_pixel)
    sr = (1 * u.MJy / u.sr).to_value(u.Jy / u.arcsec**2)
    assert flux_to_rate("MJy/sr", SF, area) == pytest.approx(sr * area * per_pixel)


def test_flambda_matches_fnu_at_pivot():
    pivot = make_base_simulation(SF).sensor.bandpass.pivot()
    flam_in_jy = (1 * u.erg / u.s / u.cm**2 / u.AA).to_value(
        u.Jy, equivalencies=u.spectral_density(pivot))
    assert flux_to_rate("erg/(s cm2 Angstrom)", SF) == pytest.approx(
        flam_in_jy * flux_to_rate("Jy", SF), rel=0.03)
    assert flux_to_rate("erg/(s cm2 Angstrom arcsec2)", SF, 2.0) == pytest.approx(
        2.0 * flux_to_rate("erg/(s cm2 Angstrom)", SF))


@pytest.mark.parametrize("unit", ["electron/s", "electron/(s pix)"])
def test_counts_pass_through(unit):
    assert flux_to_rate(unit, SF) == 1.0


def test_halpha_filters_scale_with_width():
    r2, r6, r20 = (flux_to_rate("Jy/pix", f"zwo:halpha{w}") for w in (2, 6, 20))
    assert r6 / r2 == pytest.approx(3.0, rel=0.03)
    assert r20 / r2 == pytest.approx(10.0, rel=0.03)


@pytest.mark.parametrize(
    "unit, sf, area, match",
    [
        ("Jy/arcsec2", SF, None, "pixel area"),
        ("erg/(s cm2)", SF, None, "unsupported"),      # a line flux, not a density
        ("adu", SF, None, "unsupported"),
        ("Jy/pixel", "zwo:nope", None, "unknown sensorfilter"),
    ],
)
def test_errors(unit, sf, area, match):
    with pytest.raises(ValueError, match=match):
        flux_to_rate(unit, sf, area)


def test_rates_are_cached(monkeypatch):
    calls = []

    def fake(sensorfilter, flat_unit, mag, magsys):
        calls.append(sensorfilter)
        return 42.0

    fluxrate._rate_per_jy.cache_clear()
    monkeypatch.setattr(fluxrate, "_source_rate", fake)
    assert flux_to_rate("Jy", "zwo:g") == 42.0
    assert flux_to_rate("mJy/pix", "zwo:g") == pytest.approx(0.042)
    assert calls == ["zwo:g"]
    fluxrate._rate_per_jy.cache_clear()   # don't leak the fake value


def test_band_fluxes_inverts_flux_to_rate():
    from wcc_sim.fluxrate import band_fluxes

    rate = 3.5 * flux_to_rate("Jy", SF)
    fl = band_fluxes(rate, SF)
    assert fl["f_nu_jy"] == pytest.approx(3.5)
    pivot = fl["pivot_nm"] * u.nm
    flam = (3.5 * u.Jy).to_value(u.erg / u.s / u.cm**2 / u.AA,
                                 equivalencies=u.spectral_density(pivot))
    assert fl["f_lambda"] == pytest.approx(flam)
    assert fl["f_band"] == pytest.approx(flam * fl["width_nm"] * 10.0)   # nm -> A
    assert 500 < fl["pivot_nm"] < 700 and fl["width_nm"] > 50


def test_band_fluxes_unknown_filter():
    from wcc_sim.fluxrate import band_fluxes

    with pytest.raises(ValueError, match="unknown sensorfilter"):
        band_fluxes(1.0, "zwo:nope")
