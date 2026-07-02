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


def test_add_star_conserves_flux_interior(psf_os):
    from wcc_sim.render import add_star

    img = np.zeros((128, 128), dtype=np.float32)
    add_star(img, psf_os, x=64.3, y=63.7, flux_e=1000.0, oversample=11)
    assert img.sum() == pytest.approx(1000.0, rel=2e-3)


def test_add_star_subpixel_centroid(psf_os):
    from wcc_sim.render import add_star

    img = np.zeros((128, 128), dtype=np.float32)
    x_true, y_true = 64.30, 63.70
    add_star(img, psf_os, x=x_true, y=y_true, flux_e=1e6, oversample=11)
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
    xc = (img * xx).sum() / img.sum()
    yc = (img * yy).sum() / img.sum()
    assert xc == pytest.approx(x_true, abs=0.05)
    assert yc == pytest.approx(y_true, abs=0.05)


def test_add_star_edge_clipped_no_crash(psf_os):
    from wcc_sim.render import add_star

    img = np.zeros((128, 128), dtype=np.float32)
    add_star(img, psf_os, x=2.0, y=126.0, flux_e=1000.0, oversample=11)
    assert 0 < img.sum() < 1000.0
    add_star(img, psf_os, x=-500.0, y=64.0, flux_e=1000.0, oversample=11)  # skipped


def test_render_scene_multiple(psf_os):
    from wcc_sim.render import render_scene

    img = render_scene(
        (128, 128), [30.0, 90.0], [40.0, 80.0], [1000.0, 2000.0], psf_os, 11
    )
    assert img.dtype == np.float32
    assert img.sum() == pytest.approx(3000.0, rel=2e-3)


def test_noise_statistics_sky_only(base_sim):
    from wcc_sim.render import add_noise_and_digitize

    rng = np.random.default_rng(42)
    sources = np.zeros((256, 256), dtype=np.float32)
    exptime, n_reads = 90.0, 1
    out = add_noise_and_digitize(sources, base_sim, exptime, n_reads, rng)
    lam = float(out["image_clean"][0, 0])  # uniform sky+dark expectation
    rn = float(base_sim.sensor.read_noise.value)
    resid = out["image_e"] - out["image_clean"]
    assert resid.mean() == pytest.approx(0.0, abs=5 * np.sqrt(lam + rn**2) / 256)
    assert resid.var() == pytest.approx(lam + rn**2, rel=0.05)


def test_read_noise_scales_with_n_reads(base_sim):
    from wcc_sim.render import add_noise_and_digitize

    rng = np.random.default_rng(1)
    sources = np.zeros((256, 256), dtype=np.float32)
    out4 = add_noise_and_digitize(sources, base_sim, 90.0, 4, rng)
    out1 = add_noise_and_digitize(
        sources, base_sim, 90.0, 1, np.random.default_rng(1)
    )
    rn = float(base_sim.sensor.read_noise.value)
    v4 = out4["image_e"].var() - out4["image_clean"][0, 0]
    v1 = out1["image_e"].var() - out1["image_clean"][0, 0]
    assert v4 - v1 == pytest.approx(3 * rn**2, rel=0.15)


def test_saturation_flag_and_clip(base_sim):
    from wcc_sim.render import add_noise_and_digitize

    rng = np.random.default_rng(0)
    sources = np.zeros((32, 32), dtype=np.float32)
    sources[16, 16] = 1e9  # hopelessly saturated pixel
    out = add_noise_and_digitize(sources, base_sim, 90.0, 1, rng)
    assert out["satmask"][16, 16]
    assert not out["satmask"][0, 0]
    gain = float(base_sim.sensor.gain.value)
    adc_max = float(base_sim.sensor.adc_max.value)
    assert out["image_adu"][16, 16] <= adc_max + 1e-3


def test_no_noise_mode(base_sim):
    from wcc_sim.render import add_noise_and_digitize

    sources = np.zeros((16, 16), dtype=np.float32)
    out = add_noise_and_digitize(
        sources, base_sim, 90.0, 1, np.random.default_rng(0), add_noise=False
    )
    assert np.array_equal(out["image_e"], out["image_clean"])


def test_star_saturated_window():
    from wcc_sim.render import star_saturated

    mask = np.zeros((128, 128), dtype=bool)
    assert not star_saturated(mask, 64.0, 64.0)
    mask[86, 64] = True  # ring pixel 22 px from center (2-wave defocus ring)
    assert star_saturated(mask, 64.0, 64.0)
    assert not star_saturated(mask, 64.0, 20.0)  # 44 px away, outside radius
    assert not star_saturated(mask, -500.0, 64.0)  # fully off-array window


def test_invalid_n_reads_raises(base_sim):
    from wcc_sim.render import add_noise_and_digitize

    with pytest.raises(ValueError, match="n_reads"):
        add_noise_and_digitize(
            np.zeros((8, 8), dtype=np.float32), base_sim, 90.0, 0,
            np.random.default_rng(0),
        )
