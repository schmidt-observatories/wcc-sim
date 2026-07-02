import numpy as np
import pytest


@pytest.fixture(scope="module")
def base_sim():
    from wcc_sim.detectors import make_base_simulation

    return make_base_simulation("zwo:r")


@pytest.fixture(scope="module")
def psf_os(base_sim):
    from wcc_sim.psf import render_oversampled_psf

    return render_oversampled_psf(base_sim, focus=0, oversample=11, stamp_npix=33)


@pytest.fixture(scope="module")
def wing(psf_os):
    from wcc_sim.render import bin_oversampled
    from wcc_sim.wings import fit_wing_model

    return fit_wing_model(bin_oversampled(psf_os, 11))


def _synthetic_power_law_stamp(n, c, alpha):
    half = n // 2
    yy, xx = np.mgrid[:n, :n]
    r = np.maximum(np.hypot(yy - half, xx - half), 1.0)
    return c * r**alpha


def test_fit_recovers_synthetic_power_law():
    from wcc_sim.wings import fit_wing_model

    stamp = _synthetic_power_law_stamp(65, c=1e-3, alpha=-3.0)
    model = fit_wing_model(stamp)
    assert model.alpha == pytest.approx(-3.0, abs=0.02)
    assert model.c == pytest.approx(1e-3, rel=0.05)
    assert model.r_in == pytest.approx(32.0)


def test_fit_airy_wing_slope_near_minus_3(wing):
    # jitter-blurred Airy envelope falls as ~r^-3
    assert -3.6 < wing.alpha < -2.5


def test_fit_rejects_shallow_slope():
    from wcc_sim.wings import fit_wing_model

    stamp = _synthetic_power_law_stamp(65, c=1e-3, alpha=-1.0)
    with pytest.raises(ValueError, match="alpha"):
        fit_wing_model(stamp)


def test_r_out_scales_with_flux():
    from wcc_sim.wings import WingModel

    m = WingModel(c=0.05, alpha=-3.0, r_in=16.0)
    r1 = m.r_out(1e6, floor_e=1.0)
    r8 = m.r_out(8e6, floor_e=1.0)
    assert r1 == pytest.approx((1e6 * 0.05) ** (1.0 / 3.0), rel=1e-6)
    assert r8 == pytest.approx(2.0 * r1, rel=1e-6)  # 8x flux, r^-3 wing
    assert m.r_out(1.0, floor_e=1e6) == pytest.approx(16.0)  # clamps to r_in


def test_energy_beyond_analytic():
    from wcc_sim.wings import WingModel

    m = WingModel(c=0.05, alpha=-3.0, r_in=16.0)
    assert m.energy_beyond(16.0) == pytest.approx(2 * np.pi * 0.05 / 16.0)


def test_wing_removes_postage_stamp_edge(psf_os, wing):
    from wcc_sim.render import add_star

    half = 16  # 33 px stamp
    flux = 1e9
    bare = np.zeros((256, 256), dtype=np.float32)
    add_star(bare, psf_os, x=128.0, y=128.0, flux_e=flux, oversample=11)
    assert bare[128, 128 + half + 2] == 0.0  # current truncation

    img = np.zeros((256, 256), dtype=np.float32)
    add_star(
        img, psf_os, x=128.0, y=128.0, flux_e=flux, oversample=11,
        wing=wing, floor_e=0.5,
    )
    v_in = img[128, 128 + half - 1]
    v_out = img[128, 128 + half + 1]
    assert v_out > 0.0
    assert 0.5 < v_in / v_out < 2.0  # smooth across the old stamp boundary
    # halo is azimuthally smooth: same radius along y as along x
    assert img[128 + half + 5, 128] == pytest.approx(
        img[128, 128 + half + 5], rel=0.1
    )


def test_wing_halo_is_circular_not_square(psf_os, wing):
    from wcc_sim.render import add_star

    img = np.zeros((512, 512), dtype=np.float32)
    add_star(
        img, psf_os, x=256.0, y=256.0, flux_e=1e9, oversample=11,
        wing=wing, floor_e=1.0,
    )
    r_out = wing.r_out(1e9 / (1.0 + wing.energy_beyond(16.0)), 1.0)
    yy, xx = np.mgrid[:512, :512]
    rr = np.hypot(yy - 256.0, xx - 256.0)
    assert (img[rr > r_out + 1.5] == 0.0).all()
    ring = (rr > 16 * np.sqrt(2) + 2) & (rr < r_out - 2)
    assert (img[ring] > 0.0).all()  # no zero gaps between stamp and halo


def test_faint_star_gets_no_halo(psf_os, wing):
    from wcc_sim.render import add_star

    img = np.zeros((256, 256), dtype=np.float32)
    add_star(
        img, psf_os, x=128.0, y=128.0, flux_e=10.0, oversample=11,
        wing=wing, floor_e=1.0,
    )
    outside = img.copy()
    outside[128 - 16 : 128 + 17, 128 - 16 : 128 + 17] = 0.0
    assert (outside == 0.0).all()
    assert img.sum() == pytest.approx(10.0, rel=0.05)


def test_wing_flux_conserved(psf_os, wing):
    from wcc_sim.render import add_star

    flux = 1e9
    img = np.zeros((640, 640), dtype=np.float32)
    add_star(
        img, psf_os, x=320.0, y=320.0, flux_e=flux, oversample=11,
        wing=wing, floor_e=5.0,
    )
    assert img.sum() / flux == pytest.approx(1.0, abs=0.01)


def test_wing_star_clipped_at_image_edge_no_crash(psf_os, wing):
    from wcc_sim.render import add_star

    img = np.zeros((128, 128), dtype=np.float32)
    add_star(
        img, psf_os, x=3.0, y=125.0, flux_e=1e9, oversample=11,
        wing=wing, floor_e=1.0,
    )
    assert np.isfinite(img).all()
    assert img.sum() > 0.0


def test_render_scene_passes_wing_through(psf_os, wing):
    from wcc_sim.render import render_scene

    img = render_scene(
        (256, 256), [128.0], [128.0], [1e9], psf_os, 11,
        wing=wing, floor_e=1.0,
    )
    assert img[128, 128 + 20] > 0.0  # halo beyond the 33 px stamp


def test_pipeline_wings_default_on(canned_catalog):
    from wcc_sim.pipeline import simulate_field

    RA0, DEC0 = 150.1, 2.2
    field = simulate_field(
        RA0, DEC0, sensorfilter="zwo:r", focus=0, exptime=90.0,
        catalog=canned_catalog, shape=(256, 256), stamp_npix=33,
        add_noise=False, seed=0,
    )
    assert field.params["wings"] is True
    # G=12 star sits at the field center; its halo must extend beyond the
    # 33 px stamp: pixels ~30 px away exceed the uniform sky+dark floor.
    i = int(np.argmin(field.catalog["phot_g_mean_mag"]))
    x0 = int(round(field.catalog["x"][i]))
    y0 = int(round(field.catalog["y"][i]))
    bg = field.image_clean[5, 5]
    assert field.image_clean[y0, x0 + 30] > bg * 1.5


def test_pipeline_wings_off_reproduces_truncation(canned_catalog):
    from wcc_sim.pipeline import simulate_field

    RA0, DEC0 = 150.1, 2.2
    field = simulate_field(
        RA0, DEC0, sensorfilter="zwo:r", focus=0, exptime=90.0,
        catalog=canned_catalog, shape=(256, 256), stamp_npix=33,
        add_noise=False, seed=0, wings=False,
    )
    assert field.params["wings"] is False
    i = int(np.argmin(field.catalog["phot_g_mean_mag"]))
    x0 = int(round(field.catalog["x"][i]))
    y0 = int(round(field.catalog["y"][i]))
    bg = field.image_clean[5, 5]
    assert field.image_clean[y0, x0 + 30] == pytest.approx(bg, rel=1e-4)
