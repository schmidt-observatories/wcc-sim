import numpy as np
import pytest


@pytest.fixture(scope="module")
def base_sim():
    from wcc_sim.detectors import make_base_simulation

    return make_base_simulation("zwo:r")


def bin_os(a, os_):
    n = a.shape[0] // os_
    return a.reshape(n, os_, n, os_).sum(axis=(1, 3))


def test_airy_normalized_and_centered(base_sim):
    from wcc_sim.psf import render_oversampled_psf

    os_ = 11
    p = render_oversampled_psf(base_sim, focus=0, oversample=os_, stamp_npix=65)
    assert p.shape == (65 * os_, 65 * os_)
    assert p.sum() == pytest.approx(1.0, rel=1e-9)
    cy, cx = np.unravel_index(p.argmax(), p.shape)
    assert abs(cy - p.shape[0] // 2) <= 1
    assert abs(cx - p.shape[1] // 2) <= 1


@pytest.mark.parametrize("focus", [1, 2])
def test_defocus_renders_and_is_broad(base_sim, focus):
    from wcc_sim.psf import render_oversampled_psf

    os_ = 5  # keep test fast
    p = render_oversampled_psf(base_sim, focus=focus, oversample=os_)
    det = bin_os(p, os_)
    assert p.sum() == pytest.approx(1.0, rel=1e-9)
    # Defocused PSF: peak detector pixel well below an in-focus core
    assert det.max() < 0.05


def test_stamp_captures_energy(base_sim):
    """Peak-pixel proxy: if the default stamp loses wings, renormalization
    inflates the peak vs a larger window."""
    from wcc_sim.psf import DEFAULT_STAMP, render_oversampled_psf

    os_ = 5
    p_small = bin_os(
        render_oversampled_psf(base_sim, 0, os_, DEFAULT_STAMP[0]), os_
    )
    p_big = bin_os(
        render_oversampled_psf(base_sim, 0, os_, DEFAULT_STAMP[0] + 192), os_
    )
    assert p_small.max() / p_big.max() == pytest.approx(1.0, abs=5e-3)


def test_defocus_source_data_within_default_stamp():
    """>=99.5% of the raw Huygens energy fits in the default 257-px stamp."""
    from wcc_etc import DEFOCUS_2WAVE_PATH, DefocusPSF

    # DefocusPSF knows the file format (FITS now, text before) and the
    # source pixel scale; the ETC has no public accessor for the raw grid.
    psf = DefocusPSF(DEFOCUS_2WAVE_PATH)
    d = psf._data
    half_um = 257 * 3.76 / 2.0
    half_px = int(half_um / psf.src_um_per_pix)
    c = d.shape[0] // 2
    lo, hi = max(0, c - half_px), min(d.shape[0], c + half_px + 1)
    frac = d[lo:hi, lo:hi].sum() / d.sum()
    assert frac >= 0.995


def test_bad_focus_raises(base_sim):
    from wcc_sim.psf import render_oversampled_psf

    with pytest.raises(ValueError, match="focus"):
        render_oversampled_psf(base_sim, focus=3)


def test_even_stamp_npix_raises(base_sim):
    from wcc_sim.psf import render_oversampled_psf

    with pytest.raises(ValueError, match="odd"):
        render_oversampled_psf(base_sim, focus=0, stamp_npix=64)


def _rms_radius(psf):
    n = psf.shape[0]
    c = n // 2
    yy, xx = np.mgrid[:n, :n]
    r2 = (yy - c) ** 2 + (xx - c) ** 2
    return float(np.sqrt((psf * r2).sum() / psf.sum()))


def test_explicit_wavelength_matches_default():
    from wcc_sim.detectors import make_base_simulation
    from wcc_sim.psf import render_oversampled_psf

    sim = make_base_simulation("zwo:r")
    a = render_oversampled_psf(sim, 0, oversample=3, stamp_npix=33)
    b = render_oversampled_psf(
        sim, 0, oversample=3, stamp_npix=33,
        wavelength_m=float(sim.sensor.wavelength.to("m").value),
    )
    assert np.array_equal(a, b)


def test_longer_wavelength_widens_airy():
    from wcc_sim.detectors import make_base_simulation
    from wcc_sim.psf import render_oversampled_psf

    sim = make_base_simulation("zwo:r")
    blue = render_oversampled_psf(sim, 0, oversample=3, stamp_npix=33,
                                  wavelength_m=550e-9)
    red = render_oversampled_psf(sim, 0, oversample=3, stamp_npix=33,
                                 wavelength_m=900e-9)
    assert _rms_radius(red) > _rms_radius(blue)


@pytest.mark.parametrize("sensorfilter, focus", [("zwo:r", 1), ("zwo:r", 2), ("qcmos:bb", 2)])
def test_defocus_psf_is_centred_on_its_centroid(sensorfilter, focus):
    """Issue #18: the raw Huygens images sit ~(+0.5, -0.7) px off centre."""
    from wcc_sim.detectors import make_base_simulation
    from wcc_sim.psf import centroid_offset, render_oversampled_psf

    os_ = 11
    sim = make_base_simulation(sensorfilter)
    p = render_oversampled_psf(sim, focus=focus, oversample=os_, jitter_sigma_mas=0.0)
    dx, dy = centroid_offset(p)
    assert abs(dx / os_) < 0.02 and abs(dy / os_) < 0.02
    assert p.sum() == pytest.approx(1.0, rel=1e-9)


def test_defocused_star_lands_on_its_catalog_position():
    """End to end: noiseless 2-wave star, centroid within 0.05 px of the WCS."""
    from astropy.table import Table

    from wcc_sim import simulate_field

    ra0, dec0 = 150.1, 2.2
    cat = Table({
        "source_id": np.array([0], dtype=np.int64), "ra": [ra0], "dec": [dec0],
        "phot_g_mean_mag": [14.0], "phot_bp_mean_mag": [14.4],
        "phot_rp_mean_mag": [13.6],
    })
    f = simulate_field(ra0, dec0, sensorfilter="zwo:r", focus=2, catalog=cat,
                       shape=(301, 301), add_noise=False, wings=False,
                       jitter_sigma_mas=0.0)
    img = f.image_clean - np.median(f.image_clean)
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
    w = np.clip(img, 0, None)
    cx, cy = (w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()
    x0, y0 = f.catalog["x"][0], f.catalog["y"][0]
    assert cx == pytest.approx(x0, abs=0.05)
    assert cy == pytest.approx(y0, abs=0.05)
