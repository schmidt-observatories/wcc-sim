import numpy as np
import pytest


def make(pa=0.0):
    from wcc_sim.wcsutil import build_wcs

    return build_wcs(150.1, 2.2, 16.87, pa, (6380, 9568))


def test_center_maps_to_crpix():
    w = make()
    x, y = w.world_to_pixel_values(150.1, 2.2)
    assert x == pytest.approx((9568 - 1) / 2, abs=1e-6)
    assert y == pytest.approx((6380 - 1) / 2, abs=1e-6)


def test_roundtrip():
    w = make(pa=33.0)
    ra, dec = w.pixel_to_world_values(100.0, 200.0)
    x, y = w.world_to_pixel_values(ra, dec)
    assert x == pytest.approx(100.0, abs=1e-8)
    assert y == pytest.approx(200.0, abs=1e-8)


def test_plate_scale_and_east_left():
    w = make()
    ra0, dec0 = w.pixel_to_world_values(4783.5, 3189.5)
    ra1, dec1 = w.pixel_to_world_values(4784.5, 3189.5)  # +1 px in x
    dra = (ra1 - ra0) * np.cos(np.deg2rad(dec0)) * 3600 * 1000
    assert dra == pytest.approx(-16.87, rel=1e-3)  # RA decreases with +x
    _, dec2 = w.pixel_to_world_values(4783.5, 3190.5)  # +1 px in y
    assert (dec2 - dec0) * 3600 * 1000 == pytest.approx(16.87, rel=1e-3)


def test_pa_rotates_dec_axis():
    w = make(pa=90.0)
    ra0, dec0 = w.pixel_to_world_values(4783.5, 3189.5)
    ra1, dec1 = w.pixel_to_world_values(4784.5, 3189.5)
    # with pa=90, +x now points along +Dec
    assert (dec1 - dec0) * 3600 * 1000 == pytest.approx(16.87, rel=1e-3)
