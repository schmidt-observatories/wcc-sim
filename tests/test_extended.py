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


def _mono_kernel(stamp_npix=33, oversample=3):
    from wcc_sim.detectors import make_base_simulation
    from wcc_sim.psf import render_oversampled_psf
    from wcc_sim.render import bin_oversampled

    sim = make_base_simulation("zwo:r")
    psf = render_oversampled_psf(sim, 0, oversample=oversample,
                                 stamp_npix=stamp_npix)
    return bin_oversampled(psf, oversample)


def test_render_extended_conserves_flux():
    from wcc_sim.extended import render_extended, render_component_profile

    comp = _comp(n=1.0, r_eff_arcsec=0.05, total_mag=18.0)
    kern = _mono_kernel()
    img = render_extended([comp], _wcs(), SHAPE, PLATE_MAS, "zwo:r",
                          {(comp.template, comp.ebv): kern})
    prof = render_component_profile(comp, _wcs(), SHAPE, PLATE_MAS, "zwo:r")
    assert img.dtype == np.float32
    # unit-sum kernel: convolution preserves flux (edge losses negligible
    # for a compact centered component)
    assert img.sum() == pytest.approx(prof.sum(), rel=0.005)
    # convolution spreads the profile: peak must drop
    assert img.max() < prof.max()


def test_render_extended_groups_by_template_and_ebv():
    from wcc_sim.extended import render_extended

    c1 = _comp(n=1.0, r_eff_arcsec=0.1, total_mag=17.0, template="G2V")
    c2 = _comp(n=1.0, r_eff_arcsec=0.2, total_mag=17.5, template="K0V")
    kern = _mono_kernel()
    kernels = {("G2V", 0.0): kern, ("K0V", 0.0): kern}
    both = render_extended([c1, c2], _wcs(), SHAPE, PLATE_MAS, "zwo:r",
                           kernels)
    solo1 = render_extended([c1], _wcs(), SHAPE, PLATE_MAS, "zwo:r", kernels)
    solo2 = render_extended([c2], _wcs(), SHAPE, PLATE_MAS, "zwo:r", kernels)
    assert np.allclose(both, solo1 + solo2, atol=1e-4)


def test_render_extended_wing_renormalization():
    from wcc_sim.extended import render_extended
    from wcc_sim.wings import WingModel

    comp = _comp(n=1.0, r_eff_arcsec=0.05, total_mag=18.0)
    kern = _mono_kernel()
    kernels = {(comp.template, comp.ebv): kern}
    wing = WingModel(c=0.01, alpha=-3.0, r_in=float(kern.shape[0] // 2))
    plain = render_extended([comp], _wcs(), SHAPE, PLATE_MAS, "zwo:r",
                            kernels)
    winged = render_extended([comp], _wcs(), SHAPE, PLATE_MAS, "zwo:r",
                             kernels, wing=wing)
    factor = 1.0 + wing.energy_beyond(kern.shape[0] // 2)
    assert winged.sum() == pytest.approx(plain.sum() / factor, rel=1e-4)


def test_render_extended_all_off_frame():
    from wcc_sim.extended import render_extended

    comp = _comp(dec=DEC0 + 5.0)
    img = render_extended([comp], _wcs(), SHAPE, PLATE_MAS, "zwo:r",
                          {(comp.template, comp.ebv): _mono_kernel()})
    assert img.sum() == 0.0


def test_galsim_component_matches_sersic():
    """galsim.Sersic through GalSimComponent reproduces the analytic
    SersicComponent: same total flux and same off-cusp pixel values."""
    galsim = pytest.importorskip("galsim")
    from wcc_sim.extended import GalSimComponent, render_component_profile

    r_eff = 0.1  # arcsec, ~6 px
    ana = _comp(n=1.0, r_eff_arcsec=r_eff, ellip=0.3, pa_deg=40.0, total_mag=15.0)
    # astropy ellip keeps the major axis at r_eff and shrinks the minor one;
    # galsim's area-preserving shear() would not, so compress y instead.
    prof = (
        galsim.Sersic(n=1.0, half_light_radius=r_eff)
        .transform(1.0, 0.0, 0.0, 0.7)
        .rotate(40.0 * galsim.degrees)
    )
    gs = GalSimComponent(ra=RA0, dec=DEC0, profile=prof, total_mag=15.0)
    a = render_component_profile(ana, _wcs(), SHAPE, PLATE_MAS, "zwo:r")
    g = render_component_profile(gs, _wcs(), SHAPE, PLATE_MAS, "zwo:r")
    assert g.sum() == pytest.approx(a.sum(), rel=0.02)
    off = a > 0.01 * a.max()  # away from the refined cusp pixels
    off &= a < 0.5 * a.max()
    assert np.allclose(g[off], a[off], rtol=0.03)

    with pytest.raises(TypeError):
        GalSimComponent(ra=RA0, dec=DEC0, profile=1.0, total_mag=15.0)
    far = GalSimComponent(ra=RA0 + 1.0, dec=DEC0, profile=prof, total_mag=15.0)
    assert render_component_profile(far, _wcs(), SHAPE, PLATE_MAS, "zwo:r") is None


def test_galsim_cusp_keeps_flux():
    """A compact n=4 profile (r_eff = 1 px) through GalSimComponent keeps
    its flux: without the subgrid cusp refinement the pixel-center sampling
    lost ~20% of it."""
    galsim = pytest.importorskip("galsim")
    from wcc_sim.extended import (
        GalSimComponent, component_total_rate, render_component_profile,
    )

    r_eff = PLATE_MAS / 1000.0  # one pixel
    ana = _comp(n=4.0, r_eff_arcsec=r_eff, total_mag=15.0)
    gs = GalSimComponent(ra=RA0, dec=DEC0, total_mag=15.0,
                         profile=galsim.Sersic(n=4.0, half_light_radius=r_eff))
    a = render_component_profile(ana, _wcs(), SHAPE, PLATE_MAS, "zwo:r")
    g = render_component_profile(gs, _wcs(), SHAPE, PLATE_MAS, "zwo:r")
    total = component_total_rate(gs, "zwo:r")
    assert g.sum() == pytest.approx(total, rel=0.02)
    assert g.sum() == pytest.approx(a.sum(), rel=0.02)
    assert g.max() == pytest.approx(a.max(), rel=0.02)

    # a profile wider than the frame draws only the frame intersection
    big = GalSimComponent(ra=RA0, dec=DEC0, total_mag=10.0,
                          profile=galsim.Sersic(n=4.0, half_light_radius=10.0))
    img = render_component_profile(big, _wcs(), SHAPE, PLATE_MAS, "zwo:r")
    assert img.shape == SHAPE and img.dtype == np.float32 and img.max() > 0


def test_rate_for_ab_scales():
    from wcc_sim.starflux import rate_for_ab

    r20 = rate_for_ab(20.0, "zwo:r")
    assert r20 > 0
    assert rate_for_ab(22.5, "zwo:r") == pytest.approx(r20 * 10 ** (-1.0), rel=1e-9)


def test_image_component_conserves_flux():
    """A flat patch of known AB surface brightness on a coarser WCS lands in
    the frame with the right total e-/s, whatever the pixel-area ratio."""
    pytest.importorskip("reproject")
    from astropy.wcs import WCS

    from wcc_sim.extended import ImageComponent, render_component_profile
    from wcc_sim.starflux import rate_for_ab

    # input: 0.04"/px like HST, 200x200 px, centred on the frame centre
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [100.5, 100.5]
    w.wcs.cdelt = [-0.04 / 3600, 0.04 / 3600]
    data = np.zeros((200, 200), dtype=np.float32)
    data[80:120, 80:120] = 2.0   # 40x40 input px = 1.6" square
    data[:5, :5] = np.nan         # footprint hole, must render as 0
    zp = 25.0
    comp = ImageComponent(data=data, wcs=w, zp_ab=zp)
    img = render_component_profile(comp, _wcs(), SHAPE, PLATE_MAS, "zwo:r")
    expected = 2.0 * 40 * 40 * rate_for_ab(zp, "zwo:r")
    assert img.sum() == pytest.approx(expected, rel=0.02)
    assert img.min() >= 0.0

    far = WCS(w.to_header())
    far.wcs.crval = [RA0 + 1.0, DEC0]
    assert render_component_profile(
        ImageComponent(data=data, wcs=far, zp_ab=zp), _wcs(), SHAPE, PLATE_MAS, "zwo:r"
    ) is None


def test_missing_optional_dependency_names_the_extra(monkeypatch):
    import sys

    from astropy.wcs import WCS
    from wcc_sim.extended import ImageComponent, render_component_profile

    monkeypatch.setitem(sys.modules, "reproject", None)   # simulate not installed
    comp = ImageComponent(data=np.zeros((4, 4), dtype=np.float32), wcs=WCS(naxis=2), zp_ab=25.0)
    with pytest.raises(ImportError, match=r"wcc-sim\[galaxies\]"):
        render_component_profile(comp, _wcs(), SHAPE, PLATE_MAS, "zwo:r")


def test_image_component_rejects_integer_data():
    from astropy.wcs import WCS
    from wcc_sim.extended import ImageComponent

    with pytest.raises(ValueError, match="float"):
        ImageComponent(data=np.zeros((4, 4), dtype=np.int32), wcs=WCS(naxis=2), zp_ab=25.0)
