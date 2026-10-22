import importlib.util
import sys

import numpy as np
import pytest
from astropy.wcs import WCS

from wcc_sim.resample import (
    MAX_SUBGRID,
    choose_subgrid,
    footprint,
    pixel_scale_mas,
    resample_to_subgrid,
    subgrid_wcs,
)
from wcc_sim.wcsutil import build_wcs

needs_reproject = pytest.mark.skipif(
    importlib.util.find_spec("reproject") is None, reason="reproject not installed"
)

RA0, DEC0 = 291.0, 44.5
PLATE = 16.869  # zwo plate scale [mas/pix]
DET_SHAPE = (65, 65)  # odd: the pointing falls on a pixel centre


def det(pa=0.0, shape=DET_SHAPE):
    return build_wcs(RA0, DEC0, PLATE, pa, shape)


def image_wcs(scale_mas, shape, ra=RA0, dec=DEC0, rot_deg=0.0):
    """TAN image WCS centred on (ra, dec), North up / East left, rotated by rot_deg."""
    ny, nx = shape
    s = scale_mas / 3.6e6
    c, sn = np.cos(np.radians(rot_deg)), np.sin(np.radians(rot_deg))
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    w.wcs.cd = np.array([[-s * c, s * sn], [s * sn, s * c]])
    w.array_shape = shape
    return w


# -- choose_subgrid --------------------------------------------------------- #

@pytest.mark.parametrize(
    "input_mas, expected",
    [(0.42, MAX_SUBGRID), (3.0, 7), (5.0, 5), (PLATE / 3, 3), (PLATE, 1), (40.0, 1)],
)
def test_choose_subgrid(input_mas, expected):
    assert choose_subgrid(input_mas, PLATE) == expected


def test_choose_subgrid_override():
    assert choose_subgrid(0.42, PLATE, override=3) == 3
    for bad in (4, 0, -1, 2.5):
        with pytest.raises(ValueError):
            choose_subgrid(0.42, PLATE, override=bad)


# -- geometry ---------------------------------------------------------------- #

def test_pixel_scale_mas():
    assert pixel_scale_mas(image_wcs(0.42, (8, 8), rot_deg=30)) == pytest.approx(0.42)


def test_footprint_centred_and_off_detector():
    w = image_wcs(2.0, (21, 21))  # 42 mas wide = 2.5 detector px, centred
    y0, y1, x0, x1 = footprint(w, (21, 21), det(), DET_SHAPE)
    cy, cx = (DET_SHAPE[0] - 1) / 2, (DET_SHAPE[1] - 1) / 2
    assert y0 <= cy < y1 and x0 <= cx < x1
    assert 2 <= y1 - y0 <= 4 and 2 <= x1 - x0 <= 4
    far = image_wcs(2.0, (21, 21), dec=DEC0 + 0.1)  # 360" away
    assert footprint(far, (21, 21), det(), DET_SHAPE) is None


def test_subgrid_wcs_maps_subpixel_centres():
    k, y0, x0 = 5, 10, 20
    d = det(pa=17.0)
    sub = subgrid_wcs(d, y0, x0, 4, 6, k)
    for a, b in [(0, 0), (7, 13), (19, 29)]:  # (row, col) on the sub-grid
        ra, dec = sub.pixel_to_world_values(b, a)
        x, y = d.world_to_pixel_values(ra, dec)
        assert x == pytest.approx(x0 - 0.5 + (b + 0.5) / k, abs=1e-8)
        assert y == pytest.approx(y0 - 0.5 + (a + 0.5) / k, abs=1e-8)


# -- resampling --------------------------------------------------------------- #

@needs_reproject
@pytest.mark.parametrize(
    "scale_mas, shape, rot_deg, pa",
    [
        (2.0, (40, 40), 0.0, 0.0),     # finer than the sub-grid
        (8.0, (12, 12), 0.0, 0.0),     # coarser than the sub-grid (k = 3)
        (2.0, (40, 30), 30.0, 0.0),    # rotated image
        (2.0, (40, 30), 0.0, 63.0),    # rotated detector
    ],
)
def test_flux_conserved(scale_mas, shape, rot_deg, pa):
    rng = np.random.default_rng(3)
    data = rng.uniform(0.1, 1.0, shape)
    w = image_wcs(scale_mas, shape, rot_deg=rot_deg)
    k = choose_subgrid(pixel_scale_mas(w), PLATE)
    sg = resample_to_subgrid(data, w, det(pa=pa), DET_SHAPE, k)
    assert sg.k == k
    assert sg.data.shape[0] % k == 0 and sg.data.shape[1] % k == 0
    assert sg.total_in == pytest.approx(data.sum())
    assert sg.data.sum() == pytest.approx(data.sum(), rel=1e-6)


@needs_reproject
def test_single_pixel_lands_in_expected_subpixel():
    k = 5
    d = det()
    iy, ix, p, q = 31, 33, 4, 1          # detector pixel and sub-pixel offsets
    y_t = iy + (p - (k - 1) / 2) / k     # target sub-pixel centre, detector coords
    x_t = ix + (q - (k - 1) / 2) / k
    ra, dec = d.pixel_to_world_values(x_t, y_t)
    # input pixels 1/5 of a sub-pixel, the bright one centred on the target
    w = image_wcs(PLATE / k / 5, (11, 11), ra=float(ra), dec=float(dec))
    data = np.zeros((11, 11))
    data[5, 5] = 1.0
    sg = resample_to_subgrid(data, w, d, DET_SHAPE, k)
    a, b = np.unravel_index(np.argmax(sg.data), sg.data.shape)
    assert (sg.y0 + a // k, sg.x0 + b // k) == (iy, ix)
    assert (a % k, b % k) == (p, q)
    assert sg.data[a, b] == pytest.approx(1.0, abs=1e-6)


@needs_reproject
def test_off_detector_returns_none_with_warning():
    w = image_wcs(2.0, (21, 21), dec=DEC0 + 0.1)
    with pytest.warns(UserWarning, match="does not overlap"):
        assert resample_to_subgrid(np.ones((21, 21)), w, det(), DET_SHAPE, 3) is None


@needs_reproject
def test_partly_off_detector_keeps_inside_flux():
    d = det()
    # centre the image on the detector's left edge (x = -0.5): half inside
    ra, dec = d.pixel_to_world_values(-0.5, (DET_SHAPE[0] - 1) / 2)
    w = image_wcs(2.0, (20, 20), ra=float(ra), dec=float(dec))
    with pytest.warns(UserWarning, match="beyond the detector"):
        sg = resample_to_subgrid(np.ones((20, 20)), w, d, DET_SHAPE, 9)
    assert sg.x0 == 0
    assert sg.total_in == pytest.approx(400.0)
    assert sg.data.sum() == pytest.approx(200.0, rel=1e-4)


@needs_reproject
def test_hamster_like_map():
    w = image_wcs(0.4185, (512, 512))   # 0.0586 au/px at 140 pc
    k = choose_subgrid(pixel_scale_mas(w), PLATE)
    data = np.random.default_rng(5).uniform(0.0, 1.0, (512, 512))
    sg = resample_to_subgrid(data, w, det(), DET_SHAPE, k)
    assert k == MAX_SUBGRID
    assert sg.data.shape == (13 * k, 13 * k)   # 214 mas = 12.7 px -> 13 px
    assert sg.data.sum() == pytest.approx(data.sum(), rel=1e-6)


def test_missing_reproject_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "reproject", None)
    with pytest.raises(ImportError, match=r"wcc-sim\[image\]"):
        resample_to_subgrid(np.ones((4, 4)), image_wcs(2.0, (4, 4)), det(), DET_SHAPE, 3)
