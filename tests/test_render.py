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


# -- unit_stamp / render_subgrid ----------------------------------------------- #

def _moffat_psf(n_stamp, os_, beta=2.5, alpha_px=1.5):
    """Normalised Moffat PSF oversampled by os_ (centre in the middle sample)."""
    n = n_stamp * os_
    c = (n - 1) / 2
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - c, yy - c) / os_
    psf = (1 + (r / alpha_px) ** 2) ** (-beta)
    return psf / psf.sum()


def _old_add_star_stamp(psf_os, sx, sy, oversample, flux_e, wing, blend, half):
    """add_star's stamp formula before unit_stamp was factored out (reference)."""
    from wcc_sim.render import bin_oversampled

    stamp = bin_oversampled(np.roll(psf_os, (sy, sx), axis=(0, 1)), oversample)
    if wing is None:
        return stamp * flux_e
    flux_e = flux_e / wing.flux_norm(float(half))
    return flux_e * (blend.core_scale * (stamp * blend.w + blend.model * (1.0 - blend.w))
                     + blend.halo)


@pytest.mark.parametrize("use_wing", [False, True])
def test_add_star_bit_identical_after_unit_stamp(use_wing):
    from wcc_sim.render import add_star, bin_oversampled, wing_blend
    from wcc_sim.wings import fit_wing_model

    os_, n_stamp = 11, 21
    psf = _moffat_psf(n_stamp, os_)
    wing = fit_wing_model(bin_oversampled(psf, os_)) if use_wing else None
    blend = wing_blend(n_stamp, wing) if use_wing else None
    for x, y, f in [(30.0, 30.0, 1000.0), (31.27, 29.64, 55.5), (40.49, 20.51, 1.0)]:
        image = np.zeros((64, 64))
        add_star(image, psf, x, y, f, os_, wing=wing, blend=blend)
        x0, y0 = int(round(x)), int(round(y))
        ref = _old_add_star_stamp(psf, int(round((x - x0) * os_)), int(round((y - y0) * os_)),
                                  os_, f, wing, blend, n_stamp // 2)
        h = n_stamp // 2
        assert np.array_equal(image[y0 - h:y0 + h + 1, x0 - h:x0 + h + 1], ref)


def _render_both(k, data, wing_on, y0=10, x0=10, n_stamp=15, canvas=64):
    """render_subgrid vs one add_star per sub-pixel, on the same canvas."""
    from wcc_sim.observation import WCCObservation
    from wcc_sim.render import add_star, bin_oversampled, render_subgrid, wing_blend
    from wcc_sim.resample import Subgrid
    from wcc_sim.wings import fit_wing_model

    psf = _moffat_psf(n_stamp, k)
    wing = fit_wing_model(bin_oversampled(psf, k)) if wing_on else None
    blend = wing_blend(n_stamp, wing) if wing_on else None
    sub = Subgrid(data, y0, x0, k, float(data.sum()))
    out = render_subgrid(sub, psf, wing, blend)
    fast = np.zeros((canvas, canvas))
    WCCObservation._add_cutout(fast, out)
    slow = np.zeros((canvas, canvas))
    shift = (k - 1) // 2
    for a, b in zip(*np.nonzero(data)):
        x = x0 + b // k + (b % k - shift) / k
        y = y0 + a // k + (a % k - shift) / k
        add_star(slow, psf, x, y, data[a, b], k, wing=wing, blend=blend)
    return sub, out, fast, slow


@pytest.mark.parametrize("wing_on", [False, True])
def test_render_subgrid_equals_add_star_per_subpixel(wing_on):
    data = np.random.default_rng(4).uniform(0.0, 1.0, (24, 24))   # 8 x 8 px, k = 3
    _, _, fast, slow = _render_both(3, data, wing_on)
    np.testing.assert_allclose(fast, slow, rtol=1e-12, atol=1e-14 * slow.max())


def test_render_subgrid_point_limit_k11():
    data = np.zeros((33, 33))      # 3 x 3 px at k = 11
    data[17, 9] = 500.0            # one sub-pixel, off-centre
    _, _, fast, slow = _render_both(11, data, wing_on=True)
    np.testing.assert_allclose(fast, slow, rtol=1e-12, atol=1e-14 * slow.max())


def test_render_subgrid_flux_and_origin():
    data = np.random.default_rng(5).uniform(0.0, 1.0, (15, 15))   # k = 5, 3 x 3 px
    sub, out, _, _ = _render_both(5, data, wing_on=False)
    assert out.k == 1 and out.total_in == sub.total_in
    assert (out.y0, out.x0) == (sub.y0 - 7, sub.x0 - 7)           # half of a 15-px stamp
    assert out.data.shape == (3 + 14, 3 + 14)
    assert out.data.sum() == pytest.approx(data.sum(), rel=1e-12)


def test_render_subgrid_peak_lands_in_right_pixel():
    data = np.zeros((15, 15))
    data[7, 12] = 1.0              # detector pixel (1, 2) of the cutout, centre sub-pixel
    _, _, fast, _ = _render_both(5, data, wing_on=False, y0=20, x0=30)
    assert np.unravel_index(np.argmax(fast), fast.shape) == (21, 32)


def test_render_subgrid_rejects_mismatched_psf():
    from wcc_sim.render import render_subgrid
    from wcc_sim.resample import Subgrid

    sub = Subgrid(np.ones((9, 9)), 0, 0, 3, 81.0)
    with pytest.raises(ValueError, match="multiple of"):
        render_subgrid(sub, _moffat_psf(7, 5))           # 35 samples, not / 3
    with pytest.raises(ValueError, match="odd"):
        render_subgrid(sub, _moffat_psf(14, 3))          # even stamp
