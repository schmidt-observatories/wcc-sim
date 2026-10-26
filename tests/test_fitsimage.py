import dataclasses

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from wcc_sim.fitsimage import ImageInput, image_unit, image_wcs, read_image_fits

RA0, DEC0 = 212.04231062531, -41.3979370253
SCALE_MAS = 5.58036  # pixel scale of a converted POLARIS model
N = 32


def sky_header(shape=(N, N), bunit="Jy/pixel"):
    """Header with a TAN WCS centred on the image, as for a converted model image."""
    ny, nx = shape[-2:]
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [(nx + 1) / 2, (ny + 1) / 2]
    s = SCALE_MAS / 3.6e6
    w.wcs.cd = [[-s, 0.0], [0.0, s]]
    h = w.to_header()
    if bunit is not None:
        h["BUNIT"] = bunit
    return h


def image(shape=(N, N)):
    data = np.ones(shape)
    data[..., N // 2, N // 2] = 100.0
    return data


# -- input forms ------------------------------------------------------------- #

def test_all_input_forms_agree(tmp_path):
    data, hdr = image(), sky_header()
    path = tmp_path / "img.fits"
    fits.PrimaryHDU(data, hdr).writeto(path)
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data, hdr)])
    results = [
        ImageInput.from_fits(path),
        ImageInput.from_fits(str(path)),
        ImageInput.from_fits(hdul, ext=1),
        ImageInput.from_fits(fits.ImageHDU(data, hdr)),
        ImageInput.from_fits((data, hdr)),
    ]
    for img in results:
        np.testing.assert_array_equal(img.data, data)
        assert img.unit == u.Jy / u.pix
        assert img.centre_radec == pytest.approx(results[0].centre_radec)
    assert results[0].source == "img.fits"
    assert results[3].source == "<HDU>"
    assert results[4].source == "<array>"


def test_bad_input_type():
    with pytest.raises(TypeError):
        ImageInput.from_fits(42)


# -- data checks --------------------------------------------------------------- #

def test_length_one_axes_are_dropped():
    data, hdr = read_image_fits((image((1, 1, N, N)), sky_header()))
    assert data.shape == (N, N)
    assert ImageInput.from_fits((image((1, 1, N, N)), sky_header())).shape == (N, N)


def test_cube_is_rejected():
    with pytest.raises(ValueError, match=r"2-D.*\(2, 32, 32\)"):
        ImageInput.from_fits((np.ones((2, N, N)), sky_header()))


@pytest.mark.parametrize(
    "value, match",
    [(np.nan, "non-finite"), (-1e-9, "negative"), (None, "all zero")],
)
def test_bad_data_rejected_by_reader_and_constructor(value, match):
    data = np.zeros((N, N)) if value is None else image()
    if value is not None:
        data[3, 4] = value
    hdr = sky_header()
    with pytest.raises(ValueError, match=match):
        ImageInput.from_fits((data, hdr))
    with pytest.raises(ValueError, match=match):
        ImageInput(data, image_wcs(hdr), u.Jy / u.pix)


def test_data_read_only_and_frozen():
    img = ImageInput.from_fits((image(), sky_header()))
    with pytest.raises(ValueError):
        img.data[0, 0] = 5.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        img.unit = u.Jy
    assert img != ImageInput.from_fits((image(), sky_header()))  # identity, not values


def test_constructor_copies_data():
    data = image()
    img = ImageInput(data, image_wcs(sky_header()), "Jy/pixel")
    data[0, 0] = 7.0
    assert img.data[0, 0] == 1.0
    assert img.unit == u.Jy / u.pix   # strings are parsed


# -- WCS ---------------------------------------------------------------------- #

def test_properties():
    img = ImageInput.from_fits((image(), sky_header()))
    assert img.shape == (N, N)
    assert img.pixel_scale_mas == pytest.approx(SCALE_MAS)
    assert img.total.unit == u.Jy / u.pix
    assert img.total.value == pytest.approx(N * N - 1 + 100.0)
    assert img.centre_radec == pytest.approx((RA0, DEC0), abs=1e-10)


def test_missing_celestial_wcs():
    hdr = fits.Header({"CTYPE1": "PARAM", "CTYPE2": "PARAM", "BUNIT": "Jy/pixel"})
    with pytest.raises(ValueError, match="CTYPE"):
        ImageInput.from_fits((image(), hdr))


def test_alternate_wcs_only_gets_hint():
    hdr = fits.Header({"CTYPE1": "PARAM", "CTYPE2": "PARAM",
                       "CTYPE1A": "RA---TAN", "CTYPE2A": "DEC--TAN",
                       "CRVAL1A": 0.0, "CRVAL2A": 0.0, "CDELT1A": -1e-6, "CDELT2A": 1e-6,
                       "BUNIT": "Jy/pixel"})
    with pytest.raises(ValueError, match="alternate WCS"):
        image_wcs(hdr)


def test_shape_mismatch():
    hdr = sky_header()
    hdr["NAXIS"], hdr["NAXIS1"], hdr["NAXIS2"] = 2, N + 1, N
    with pytest.raises(ValueError, match="does not match"):
        ImageInput.from_fits((image(), hdr))


def test_header_is_a_copy():
    hdr = sky_header()
    img = ImageInput.from_fits((image(), hdr))
    img.header["BUNIT"] = "adu"
    assert hdr["BUNIT"] == "Jy/pixel"


# -- units -------------------------------------------------------------------- #

def test_bunit_parsed():
    assert image_unit(sky_header()) == u.Jy / u.pix
    assert image_unit(sky_header(bunit="Jy/pix")) == u.Jy / u.pix


def test_missing_bunit_requires_flux_unit():
    hdr = sky_header(bunit=None)
    with pytest.raises(ValueError, match="no BUNIT"):
        ImageInput.from_fits((image(), hdr))
    assert ImageInput.from_fits((image(), hdr), flux_unit="Jy/pix").unit == u.Jy / u.pix


def test_flux_unit_must_agree_with_bunit():
    hdr = sky_header()
    assert ImageInput.from_fits((image(), hdr), flux_unit="Jy/pix").unit == u.Jy / u.pix
    with pytest.raises(ValueError, match="conflicts"):
        ImageInput.from_fits((image(), hdr), flux_unit="mJy/pix")


def test_nonsense_bunit():
    with pytest.raises(ValueError, match="cannot parse"):
        image_unit(sky_header(bunit="bananas per fortnight"))


# -- detector rates ----------------------------------------------------------- #

def test_rate_scale_and_rate_image():
    from wcc_sim.fluxrate import flux_to_rate

    img = ImageInput.from_fits((image(), sky_header()))
    scale = img.rate_scale("zwo:r")
    assert scale == pytest.approx(flux_to_rate("Jy/pixel", "zwo:r"))
    assert img.rate_image("zwo:r").sum() == pytest.approx(img.total.value * scale)


def test_rate_scale_surface_brightness_uses_pixel_area():
    from wcc_sim.fluxrate import flux_to_rate

    img = ImageInput.from_fits((image(), sky_header(bunit="Jy/arcsec2")))
    area = (SCALE_MAS / 1000) ** 2
    assert img.rate_scale("zwo:r") == pytest.approx(area * flux_to_rate("Jy", "zwo:r"))


def test_rate_scale_warns_on_band_mismatch():
    import warnings

    hdr = sky_header()
    hdr["BAND"] = "zwo:halpha2"
    img = ImageInput.from_fits((image(), hdr))
    assert img.band == "zwo:halpha2"
    with pytest.warns(UserWarning, match="flat-spectrum transfer"):
        img.rate_scale("zwo:r")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        img.rate_scale("zwo:halpha2")   # same band: no band-mismatch warning
