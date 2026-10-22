import warnings

import numpy as np
import pytest
from astropy.wcs import WCS

from wcc_sim.detectors import detector_frame
from wcc_sim.extended import ExtendedComponent
from wcc_sim.fitsimage import ImageInput
from wcc_sim.imagesource import SkyImageComponent
from wcc_sim.resample import choose_subgrid

RA0, DEC0 = 291.0, 44.5
SF = "zwo:r"


def sky_image(scale_mas=2.0, n=40, ra=RA0, dec=DEC0, band=None):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    s = scale_mas / 3.6e6
    w.wcs.cd = [[-s, 0.0], [0.0, s]]
    h = w.to_header()
    h["BUNIT"] = "Jy/pixel"
    if band:
        h["BAND"] = band
    data = np.random.default_rng(1).uniform(0.0, 1e-3, (n, n))
    return data, h


def frame(**kw):
    return detector_frame(SF, (RA0, DEC0), shape=(65, 65), **kw)


def test_extended_component_is_abstract():
    with pytest.raises(TypeError):
        ExtendedComponent()


def test_kw_only_base_allows_positional_subclass_fields():
    data, h = sky_image()
    comp = SkyImageComponent(ImageInput.from_fits((data, h)), label="x")
    assert comp.label == "x" and comp.kind == "image"


def test_rate_map_conserves_rate_and_chooses_k():
    data, h = sky_image()
    comp = SkyImageComponent.from_fits((data, h))
    det = frame()
    sub = comp.rate_map(det.wcs, det.shape, det.plate_scale_mas, SF)
    expected = comp.image.rate_image(SF).sum()
    assert sub.total_in == pytest.approx(expected)
    assert sub.data.sum() == pytest.approx(expected, rel=1e-6)
    assert sub.k == choose_subgrid(2.0, det.plate_scale_mas)
    assert comp.rate_map(det.wcs, det.shape, det.plate_scale_mas, SF, k=3).k == 3


def test_footprint_matches_resample_and_off_detector():
    data, h = sky_image()
    comp = SkyImageComponent.from_fits((data, h))
    det = frame()
    sub = comp.rate_map(det.wcs, det.shape, det.plate_scale_mas, SF)
    y0, y1, x0, x1 = comp.footprint(det.wcs, det.shape)
    assert (y0, x0) == (sub.y0, sub.x0)
    assert sub.data.shape == ((y1 - y0) * sub.k, (x1 - x0) * sub.k)
    far = SkyImageComponent.from_fits(sky_image(dec=DEC0 + 0.1))
    assert far.footprint(det.wcs, det.shape) is None
    with pytest.warns(UserWarning, match="does not overlap"):
        assert far.rate_map(det.wcs, det.shape, det.plate_scale_mas, SF) is None


def test_from_fits_label_and_properties(tmp_path):
    from astropy.io import fits

    data, h = sky_image()
    path = tmp_path / "model.fits"
    fits.PrimaryHDU(data, h).writeto(path)
    comp = SkyImageComponent.from_fits(path)
    assert comp.label == "model.fits"
    assert comp.centre_radec == pytest.approx((RA0, DEC0), abs=1e-6)
    assert SkyImageComponent.from_fits(path, label="mine").label == "mine"


def test_band_warning_only_for_other_filter():
    data, h = sky_image(band="zwo:halpha2")
    comp = SkyImageComponent.from_fits((data, h))
    det = frame()
    with pytest.warns(UserWarning, match="flat-spectrum transfer"):
        comp.rate_map(det.wcs, det.shape, det.plate_scale_mas, SF)
    det2 = detector_frame("zwo:halpha2", (RA0, DEC0), shape=(65, 65))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        comp.rate_map(det2.wcs, det2.shape, det2.plate_scale_mas, "zwo:halpha2")


def test_to_record_bunit_as_written():
    import numpy as np
    from astropy.wcs import WCS

    from wcc_sim.fitsimage import ImageInput
    from wcc_sim.imagesource import SkyImageComponent

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.cdelt = [-1e-6, 1e-6]
    h = w.to_header()
    data = np.ones((8, 8))
    h["BUNIT"] = "mJy/pixel"
    assert SkyImageComponent(ImageInput.from_fits((data, h))).to_record()["bunit"] == "mJy/pixel"
    del h["BUNIT"]
    rec = SkyImageComponent(ImageInput.from_fits((data, h), flux_unit="Jy/pix")).to_record()
    assert rec["bunit"] == rec["unit"] == "Jy / pix"
