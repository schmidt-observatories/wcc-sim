import warnings

import numpy as np
import pytest
from astropy.wcs import WCS

from wcc_sim.detectors import detector_frame
from wcc_sim.fitsimage import ImageInput
from wcc_sim.imagesource import SkyImageComponent
from wcc_sim.observation import WCCObservation
from wcc_sim.resample import Subgrid

RA0, DEC0 = 291.0, 44.5
SF = "zwo:r"


def component(ra=RA0, dec=DEC0, scale_mas=5.58, n=60, value=1e-3):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    s = scale_mas / 3.6e6
    w.wcs.cd = [[-s, 0.0], [0.0, s]]
    h = w.to_header()
    h["BUNIT"] = "Jy/pixel"
    data = np.full((n, n), value)
    data[n // 2, n // 2] = 100 * value       # a "star"
    return SkyImageComponent(ImageInput.from_fits((data, h)))


def frame(shape=(161, 161)):   # > 129 px stamp + component
    return detector_frame(SF, (RA0, DEC0), shape=shape)


def test_psf_cached_and_no_wing_option():
    obs = WCCObservation(component(), frame(), focus=0, wings=False)
    psf3, wing, blend = obs.psf(3)
    assert obs.psf(3)[0] is psf3 and wing is None and blend is None
    assert psf3.shape[0] % 3 == 0
    obs_w = WCCObservation(component(), frame(), focus=0, wings=True, scatter=False)
    assert obs_w.psf(3)[1] is not None


def test_rate_image_conserves_rate_without_wings():
    comp = component()
    obs = WCCObservation(comp, frame(), focus=0, wings=False)
    img = obs.rate_image()
    assert img.dtype == np.float32 and img.shape == (161, 161)
    total = comp.image.rate_image(SF).sum()
    assert img.sum() == pytest.approx(total, rel=1e-5)
    assert obs.params["subgrid_k"] == "5"


def test_rate_image_with_wings_close_to_total():
    comp = component()
    img = WCCObservation(comp, frame(), focus=0, wings=True, scatter=False).rate_image()
    assert img.sum() == pytest.approx(comp.image.rate_image(SF).sum(), rel=0.01)


def test_two_components_add_and_off_detector_skipped():
    a, b = component(), component(value=2e-3)
    single = WCCObservation(a, frame(), focus=0, wings=False).rate_image()
    both = WCCObservation([a, b], frame(), focus=0, wings=False).rate_image()
    other = WCCObservation(b, frame(), focus=0, wings=False).rate_image()
    np.testing.assert_allclose(both, single + other, rtol=1e-6)
    far = component(dec=DEC0 + 0.1)
    with pytest.warns(UserWarning, match="does not overlap"):
        img = WCCObservation([a, far], frame(), focus=0, wings=False).rate_image()
    np.testing.assert_allclose(img, single, rtol=1e-6)


@pytest.mark.parametrize("y0, x0", [(-3, -3), (-3, 8), (8, -3), (8, 8)])
def test_add_cutout_clips_at_edges(y0, x0):
    image = np.zeros((10, 10), dtype=np.float32)
    WCCObservation._add_cutout(image, Subgrid(np.ones((5, 5)), y0, x0, 1, 25.0))
    expected = np.zeros((10, 10))
    expected[max(y0, 0):min(y0 + 5, 10), max(x0, 0):min(x0 + 5, 10)] = 1.0
    np.testing.assert_array_equal(image, expected)


def test_add_cutout_fully_outside_is_ignored():
    image = np.zeros((10, 10), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        WCCObservation._add_cutout(image, Subgrid(np.ones((3, 3)), 20, 20, 1, 9.0))
    assert not image.any()


# -- expose / SimulatedField / FITS -------------------------------------------- #

def test_expose_noiseless_matches_rate_image():
    from wcc_sim.starflux import sky_and_dark_rates

    obs = WCCObservation(component(), frame(), focus=0, wings=False)
    field = obs.expose(exptime=50.0, add_noise=False, seed=3)
    sky, dark = sky_and_dark_rates(obs.sim)
    np.testing.assert_allclose(field.image_clean - np.float32((sky + dark) * 50.0),
                               obs.rate_image() * np.float32(50.0), rtol=1e-5, atol=1e-2)
    assert len(field.catalog) == 0 and "rate_e_s" in field.catalog.colnames
    assert field.params["exptime"] == 50.0 and field.params["n_sources"] == 0


def test_expose_reproducible_and_seed_recorded():
    obs = WCCObservation(component(), frame(), focus=0, wings=False)
    a, b = obs.expose(exptime=10.0, seed=7), obs.expose(exptime=10.0, seed=7)
    np.testing.assert_array_equal(a.image_adu, b.image_adu)
    c = obs.expose(exptime=10.0)
    assert isinstance(c.params["seed"], int)
    np.testing.assert_array_equal(obs.expose(exptime=10.0, seed=c.params["seed"]).image_adu,
                                  c.image_adu)


@pytest.mark.parametrize("kw", [{"exptime": 0.0}, {"exptime": np.inf}, {"n_reads": 0},
                                {"n_reads": 1.5}, {"n_reads": True}])
def test_expose_rejects_bad_inputs(kw):
    with pytest.raises(ValueError):
        WCCObservation(component(), frame(), focus=0, wings=False).expose(**kw)


def test_components_table_and_fluxes():
    from wcc_sim.fluxrate import flux_to_rate

    a, far = component(), component(dec=DEC0 + 0.1)
    obs = WCCObservation([a, far], frame(), focus=0, wings=False)
    with pytest.warns(UserWarning, match="does not overlap"):
        table = obs.components_table()
    assert list(table["kind"]) == ["image", "image"]
    assert list(table["subgrid_k"]) == [5, -1]
    assert table["on_frame_e_s"][0] == pytest.approx(table["total_e_s"][0], rel=1e-5)
    assert table["on_frame_e_s"][1] == 0.0
    jy = a.image.data.sum()                                   # Jy/pixel image
    assert table["f_nu_jy"] == pytest.approx([jy, jy], rel=1e-9)
    assert table["rate_per_unit"][0] == pytest.approx(flux_to_rate("Jy/pixel", SF))


def test_subgrid_override():
    obs = WCCObservation(component(), frame(), focus=0, wings=False, subgrid=3)
    obs.rate_image()
    assert obs.params["subgrid_k"] == "3"


def test_write_and_read_back(tmp_path):
    from astropy.io import fits

    det = detector_frame(SF, (RA0, DEC0), offset_arcsec=(0.1, -0.05), pa=20, shape=(161, 161))
    field = WCCObservation(component(), det, focus=0).expose(exptime=20.0, seed=1)
    path = tmp_path / "img.fits"
    field.write(path)
    with fits.open(path) as h:
        assert [x.name for x in h] == ["SCI", "SATMASK", "CAT", "CLEAN", "COMP"]
        hdr = h["SCI"].header
        assert hdr["SENSORF"] == SF and hdr["PA"] == 20.0
        assert hdr["OFFSRA"] == 0.1 and hdr["OFFSDEC"] == -0.05
        assert hdr["FLUXORIG"] == "input image (BUNIT)"
        assert hdr["IMGBUNIT"] == "Jy/pixel"                     # as written, not 'Jy / pix'
        assert h["COMP"].data["bunit"][0] == "Jy/pixel"
        assert hdr["FLUXJY"] == pytest.approx(field.components["f_nu_jy"][0])
        assert hdr["IMGTOT"] == pytest.approx(field.params["image_total_e_s"])
        assert "PSFOVER" not in hdr and "WINGFLR" not in hdr
        assert len(h["CAT"].data) == 0 and len(h["COMP"].data) == 1


def test_band_fluxes_in_params():
    from wcc_sim.fluxrate import band_fluxes

    field = WCCObservation(component(), frame(), focus=0, wings=False).expose(
        exptime=10.0, add_noise=False)
    expected = band_fluxes(field.params["image_total_e_s"], SF)
    assert field.params["flux_jy"] == pytest.approx(expected["f_nu_jy"])
    assert field.params["flux_band"] == pytest.approx(expected["f_band"])
