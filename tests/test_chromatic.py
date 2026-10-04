import numpy as np
import pytest

OS, STAMP = 3, 33  # small/fast render for tests


def _sim():
    from wcc_sim.detectors import make_base_simulation

    return make_base_simulation("zwo:r")


def _spectrum(name):
    from wcc_etc.scene import get_scene_element

    return get_scene_element(name, mag=15.0).spectrum


def _rms_radius(psf):
    n = psf.shape[0]
    c = n // 2
    yy, xx = np.mgrid[:n, :n]
    return float(np.sqrt((psf * ((yy - c) ** 2 + (xx - c) ** 2)).sum() / psf.sum()))


def test_band_nodes_weights_normalized():
    from wcc_sim.chromatic import band_nodes, band_support

    sim = _sim()
    waves_m, weights = band_nodes(sim, _spectrum("G2V"), n_nodes=5)
    assert waves_m.shape == (5,) and weights.shape == (5,)
    assert weights.sum() == pytest.approx(1.0)
    lo, hi = band_support(sim)
    assert np.all(waves_m > lo * 1e-10) and np.all(waves_m < hi * 1e-10)


def test_red_spectrum_shifts_nodes_red():
    from wcc_sim.chromatic import band_nodes

    sim = _sim()
    _, w_blue = band_nodes(sim, _spectrum("A0V"), n_nodes=5)
    _, w_red = band_nodes(sim, _spectrum("M0V"), n_nodes=5)
    waves_m, _ = band_nodes(sim, _spectrum("A0V"), n_nodes=5)
    assert np.sum(waves_m * w_red) > np.sum(waves_m * w_blue)


def test_single_node_is_monochromatic_at_the_effective_wavelength():
    """n_nodes=1 is one Airy render at lambda_eff, not at the pivot (#20)."""
    from wcc_sim.chromatic import effective_psf, effective_wavelength_m
    from wcc_sim.psf import render_oversampled_psf

    sim = _sim()
    spec = _spectrum("G2V")
    pivot = render_oversampled_psf(sim, 0, oversample=OS, stamp_npix=STAMP)
    mono = render_oversampled_psf(sim, 0, oversample=OS, stamp_npix=STAMP,
                                  wavelength_m=effective_wavelength_m(sim, spec))
    eff = effective_psf(sim, 0, spec, oversample=OS, stamp_npix=STAMP, n_nodes=1)
    assert np.array_equal(eff, mono)
    assert not np.array_equal(eff, pivot)


def test_defocus_is_passthrough():
    from wcc_sim.chromatic import effective_psf
    from wcc_sim.psf import render_oversampled_psf

    sim = _sim()
    mono = render_oversampled_psf(sim, 1, oversample=OS, stamp_npix=65)
    eff = effective_psf(sim, 1, _spectrum("M0V"), oversample=OS,
                        stamp_npix=65, n_nodes=5)
    assert np.array_equal(eff, mono)


def test_red_effective_psf_wider_in_focus():
    from wcc_sim.chromatic import effective_psf

    sim = _sim()
    blue = effective_psf(sim, 0, _spectrum("A0V"), oversample=OS,
                         stamp_npix=STAMP, n_nodes=5)
    red = effective_psf(sim, 0, _spectrum("M0V"), oversample=OS,
                        stamp_npix=STAMP, n_nodes=5)
    assert _rms_radius(red) > _rms_radius(blue)
    assert red.sum() == pytest.approx(1.0, rel=1e-6)


def test_effective_psf_for_spt_is_cached():
    from wcc_sim.chromatic import effective_psf_for_spt

    sim = _sim()
    a = effective_psf_for_spt(sim, "zwo:r", "G2V", 0.0, 0, OS, STAMP, n_nodes=3)
    b = effective_psf_for_spt(sim, "zwo:r", "G2V", 0.0, 0, OS, STAMP, n_nodes=3)
    assert a is b


def test_bad_n_nodes_raises():
    from wcc_sim.chromatic import effective_psf

    with pytest.raises(ValueError):
        effective_psf(_sim(), 0, _spectrum("G2V"), n_nodes=0)


def test_attenuation_zero_ebv_is_one():
    from wcc_sim.chromatic import attenuation_factor

    assert attenuation_factor("G2V", 0.0, "zwo:i") == 1.0


def test_attenuation_monotonic_and_band_dependent():
    from wcc_sim.chromatic import attenuation_factor

    a1 = attenuation_factor("G2V", 0.1, "zwo:i")
    a2 = attenuation_factor("G2V", 0.3, "zwo:i")
    assert 0.0 < a2 < a1 < 1.0
    # zwo:i (~690-855 nm) is redder than V: attenuation must be milder
    # than the full A_V = R_V * E(B-V) dimming, but real (< 1).
    av_floor = 10.0 ** (-0.4 * 3.1 * 0.1)
    assert av_floor < a1 < 1.0


def test_attenuation_negative_ebv_raises():
    import pytest as _pytest

    from wcc_sim.chromatic import attenuation_factor

    with _pytest.raises(ValueError):
        attenuation_factor("G2V", -0.1, "zwo:i")


def test_effective_wavelength_matches_the_etc():
    """Issue #20: the sim must render the PSF where the ETC does."""
    from wcc_sim.chromatic import effective_wavelength_nm_for_spt
    from wcc_sim.starflux import make_star_simulation

    for spt in ("O5V", "G2V", "M5V"):
        etc = float(make_star_simulation(spt, "zwo:bb").effective_wavelength.to("nm").value)
        assert effective_wavelength_nm_for_spt("zwo:bb", spt) == pytest.approx(etc, abs=1.0)
    pivot = float(make_star_simulation("M5V", "zwo:bb").sensor.wavelength.to("nm").value)
    assert effective_wavelength_nm_for_spt("zwo:bb", "M5V") - pivot > 100.0

