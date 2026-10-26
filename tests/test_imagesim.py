import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from wcc_sim import simulate_image

RA0, DEC0 = 291.0, 44.5


def write_image(path, n=40, scale_mas=5.58):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    s = scale_mas / 3.6e6
    w.wcs.cd = [[-s, 0.0], [0.0, s]]
    h = w.to_header()
    h["BUNIT"] = "Jy/pixel"
    data = np.full((n, n), 1e-4)
    data[n // 2, n // 2] = 1e-2
    fits.PrimaryHDU(data, h).writeto(path)
    return path


def test_pointing_from_image(tmp_path):
    path = write_image(tmp_path / "in.fits")
    out = tmp_path / "out.fits"
    field = simulate_image(path, focus=0, shape=(161, 161), exptime=10.0, seed=1,
                           wings=False, output=out)
    assert field.params["pointing_source"] == "image"
    assert field.params["ra"] == pytest.approx(RA0) and field.params["dec"] == pytest.approx(DEC0)
    assert field.params["image_rate_e_s"] == pytest.approx(field.params["image_total_e_s"],
                                                           rel=1e-5)
    with fits.open(out) as h:
        assert h["SCI"].header["PNTSRC"] == "image"
        assert h["COMP"].data["source"][0] == "in.fits"


def test_user_pointing_and_offset(tmp_path):
    path = write_image(tmp_path / "in.fits")
    field = simulate_image(path, ra=RA0, dec=DEC0 + 1e-4, offset_arcsec=(0.0, -0.36),
                           focus=0, shape=(161, 161), exptime=10.0, seed=1, wings=False)
    p = field.params
    assert p["pointing_source"] == "user"
    assert (p["centre_ra"], p["centre_dec"]) == pytest.approx((RA0, DEC0 + 1e-4))
    assert p["dec"] == pytest.approx(DEC0, abs=1e-8)            # 0.36" = 1e-4 deg back


def test_ra_without_dec_rejected(tmp_path):
    with pytest.raises(ValueError, match="both ra and dec"):
        simulate_image(write_image(tmp_path / "in.fits"), ra=RA0)
