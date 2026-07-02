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
        render_oversampled_psf(base_sim, 0, os_, DEFAULT_STAMP[0] + 32), os_
    )
    assert p_small.max() / p_big.max() == pytest.approx(1.0, abs=5e-3)


def test_defocus_source_data_within_default_stamp():
    """>=99.5% of the raw Huygens energy fits in the default 257-px stamp."""
    from wcc_etc import DEFOCUS_2WAVE_PATH
    from wcc_etc.psfsim import load_huygens_psf

    d = load_huygens_psf(DEFOCUS_2WAVE_PATH)  # 256x256 @ 4.0 um/px
    half_um = 257 * 3.76 / 2.0
    half_px = int(half_um / 4.0)
    c = d.shape[0] // 2
    lo, hi = max(0, c - half_px), min(d.shape[0], c + half_px + 1)
    frac = d[lo:hi, lo:hi].sum() / d.sum()
    assert frac >= 0.995


def test_bad_focus_raises(base_sim):
    from wcc_sim.psf import render_oversampled_psf

    with pytest.raises(ValueError, match="focus"):
        render_oversampled_psf(base_sim, focus=3)
