import numpy as np
import pytest

from tests.conftest import DEC0, RA0

PLATE_MAS = 16.87  # zwo plate scale, close enough for unit tests


def _comp(**kw):
    from wcc_sim.extended import SersicComponent

    base = dict(ra=RA0, dec=DEC0, n=1.0, r_eff_arcsec=0.1, total_mag=15.0)
    base.update(kw)
    return SersicComponent(**base)


def test_validation():
    from wcc_sim.extended import SersicComponent

    with pytest.raises(ValueError):  # both normalizations
        _comp(sb_mag_arcsec2=20.0)
    with pytest.raises(ValueError):  # neither
        SersicComponent(ra=RA0, dec=DEC0, n=1.0, r_eff_arcsec=0.1)
    with pytest.raises(ValueError):
        _comp(n=0.0)
    with pytest.raises(ValueError):
        _comp(r_eff_arcsec=0.0)
    with pytest.raises(ValueError):
        _comp(ellip=1.0)
    with pytest.raises(ValueError):
        _comp(ebv=-0.1)


def test_total_over_amplitude_matches_numeric_integral():
    from astropy.modeling.models import Sersic2D

    from wcc_sim.extended import sersic_total_over_amplitude

    n, r_eff, ellip = 1.0, 6.0, 0.3
    factor = sersic_total_over_amplitude(n, r_eff, ellip)
    mod = Sersic2D(amplitude=1.0, r_eff=r_eff, n=n, x_0=0.0, y_0=0.0,
                   ellip=ellip, theta=0.0)
    g = np.linspace(-200.0, 200.0, 2001)  # 0.2 px sampling to r ~ 33 r_eff
    numeric = mod(g[None, :], g[:, None]).sum() * (g[1] - g[0]) ** 2
    assert numeric == pytest.approx(factor, rel=2e-3)


def test_component_amplitude_total_mag_roundtrip():
    from wcc_sim.extended import component_amplitude, sersic_total_over_amplitude
    from wcc_sim.starflux import REF_MAG, rate_for_spt

    comp = _comp(total_mag=REF_MAG)
    amp = component_amplitude(comp, "zwo:r", PLATE_MAS)
    r_eff_pix = comp.r_eff_arcsec * 1000.0 / PLATE_MAS
    total = amp * sersic_total_over_amplitude(comp.n, r_eff_pix, comp.ellip)
    assert total == pytest.approx(rate_for_spt("G2V", "zwo:r"), rel=1e-6)


def test_component_amplitude_surface_brightness():
    from wcc_sim.extended import component_amplitude
    from wcc_sim.starflux import REF_MAG, rate_for_spt

    comp = _comp(total_mag=None, sb_mag_arcsec2=REF_MAG)
    amp = component_amplitude(comp, "zwo:r", PLATE_MAS)
    pix_arcsec2 = (PLATE_MAS / 1000.0) ** 2
    assert amp == pytest.approx(
        rate_for_spt("G2V", "zwo:r") * pix_arcsec2, rel=1e-6
    )


def test_reddening_dims_amplitude():
    from wcc_sim.extended import component_amplitude

    a0 = component_amplitude(_comp(), "zwo:r", PLATE_MAS)
    a1 = component_amplitude(_comp(ebv=0.3), "zwo:r", PLATE_MAS)
    assert a1 < a0


SHAPE = (256, 256)


def _wcs(shape=SHAPE):
    from wcc_sim.wcsutil import build_wcs

    return build_wcs(RA0, DEC0, PLATE_MAS, 0.0, shape)


def _moments(img):
    ny, nx = img.shape
    yy, xx = np.mgrid[:ny, :nx]
    t = img.sum()
    cx, cy = (img * xx).sum() / t, (img * yy).sum() / t
    return (
        (img * (xx - cx) ** 2).sum() / t,
        (img * (yy - cy) ** 2).sum() / t,
    )


def test_profile_flux_conserved_exponential():
    from wcc_sim.extended import (
        component_amplitude,
        render_component_profile,
        sersic_total_over_amplitude,
    )

    comp = _comp(n=1.0, r_eff_arcsec=0.05, total_mag=18.0)  # r_eff ~ 3 px
    img = render_component_profile(comp, _wcs(), SHAPE, PLATE_MAS, "zwo:r")
    amp = component_amplitude(comp, "zwo:r", PLATE_MAS)
    r_eff_pix = comp.r_eff_arcsec * 1000.0 / PLATE_MAS
    total = amp * sersic_total_over_amplitude(comp.n, r_eff_pix, comp.ellip)
    assert img.dtype == np.float32
    assert img.sum() == pytest.approx(total, rel=0.005)


def test_profile_flux_conserved_devauc():
    from wcc_sim.extended import (
        component_amplitude,
        render_component_profile,
        sersic_total_over_amplitude,
    )

    comp = _comp(n=4.0, r_eff_arcsec=0.02, total_mag=18.0)  # cuspy center
    img = render_component_profile(comp, _wcs(), SHAPE, PLATE_MAS, "zwo:r")
    amp = component_amplitude(comp, "zwo:r", PLATE_MAS)
    r_eff_pix = comp.r_eff_arcsec * 1000.0 / PLATE_MAS
    total = amp * sersic_total_over_amplitude(comp.n, r_eff_pix, comp.ellip)
    # n=4 keeps ~1% of its flux beyond the 128 px frame half-width;
    # require the rendered sum to land between 97% and 100.5% of analytic.
    assert 0.97 * total < img.sum() < 1.005 * total


def test_profile_orientation():
    from wcc_sim.extended import render_component_profile

    ex = _comp(ellip=0.6, pa_deg=0.0, r_eff_arcsec=0.2)
    ey = _comp(ellip=0.6, pa_deg=90.0, r_eff_arcsec=0.2)
    ix = render_component_profile(ex, _wcs(), SHAPE, PLATE_MAS, "zwo:r")
    iy = render_component_profile(ey, _wcs(), SHAPE, PLATE_MAS, "zwo:r")
    vxx_x, vyy_x = _moments(ix)
    vxx_y, vyy_y = _moments(iy)
    assert vxx_x > vyy_x  # pa=0: major axis along +x
    assert vyy_y > vxx_y  # pa=90: rotated onto +y


def test_far_off_frame_component_skipped():
    from wcc_sim.extended import render_component_profile

    comp = _comp(dec=DEC0 + 5.0, r_eff_arcsec=0.5)
    assert (
        render_component_profile(comp, _wcs(), SHAPE, PLATE_MAS, "zwo:r")
        is None
    )
