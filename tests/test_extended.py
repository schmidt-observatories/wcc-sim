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
