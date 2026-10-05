import numpy as np
import pytest

from wcc_phot.apphot import aperture_photometry_frame
from wcc_phot.centroid import centroid_stars
from wcc_phot.flags import FLAG_CENTROID
from wcc_phot.geometry import default_geometry, render_model_psf
from wcc_phot.io import load_frame
from wcc_phot.psfphot import build_psf_model, psf_photometry_frame

OVERSAMPLE = 5


@pytest.fixture(scope="module")
def star_frame(single_star_frame):
    return load_frame(single_star_frame)


@pytest.fixture(scope="module")
def model_psf(star_frame):
    return render_model_psf(star_frame.meta, oversample=OVERSAMPLE)


@pytest.fixture(scope="module")
def geom(star_frame, model_psf):
    return default_geometry(
        star_frame.meta, oversample=OVERSAMPLE, psf_os=model_psf
    )


def _true_position(single_star_frame):
    return single_star_frame.wcs.world_to_pixel_values(
        float(single_star_frame.catalog["ra"][0]),
        float(single_star_frame.catalog["dec"][0]),
    )


def test_default_geometry_sane(geom):
    assert geom.r_ap >= 2.0
    assert geom.r_ap <= geom.r_in < geom.r_out
    assert geom.centroid_box % 2 == 1 and geom.fit_shape % 2 == 1
    assert 0.9 < geom.ee_fraction <= 1.0


def test_centroid_recovers_offset_start(single_star_frame, star_frame):
    x_true, y_true = _true_position(single_star_frame)
    x, y, flags = centroid_stars(
        star_frame, x_true + 1.5, y_true - 1.2, box=21
    )
    assert flags[0] == 0
    assert x[0] == pytest.approx(x_true, abs=0.1)
    assert y[0] == pytest.approx(y_true, abs=0.1)


def test_centroid_empty_region_falls_back(star_frame):
    x, y, flags = centroid_stars(star_frame, 30.0, 30.0, box=13)
    assert flags[0] & FLAG_CENTROID
    assert x[0] == 30.0 and y[0] == 30.0


def test_aperture_flux_matches_model_ee(single_star_frame, star_frame, geom):
    x_true, y_true = _true_position(single_star_frame)
    flux, flux_err, bkg, flags = aperture_photometry_frame(
        star_frame, x_true, y_true, geom
    )
    expected = (
        float(single_star_frame.catalog["rate_e_s"][0])
        * star_frame.meta["exptime"]
    )
    assert flags[0] == 0
    assert flux[0] / expected == pytest.approx(geom.ee_fraction, rel=0.02)
    assert flux_err[0] > 0
    sky_dark = (
        star_frame.meta["sky_e_s"] + star_frame.meta["dark_e_s"]
    ) * star_frame.meta["exptime"]
    # annulus sits on the PSF wings, so bkg is sky+dark plus a little star
    assert 0 < bkg[0] < sky_dark + 3.0


def test_psf_flux_matches_total(single_star_frame, star_frame, geom, model_psf):
    x_true, y_true = _true_position(single_star_frame)
    model = build_psf_model(
        star_frame.meta, oversample=OVERSAMPLE, psf_os=model_psf
    )
    flux, flux_err, x_fit, y_fit, fflags = psf_photometry_frame(
        star_frame,
        np.array([x_true + 0.3]),
        np.array([y_true - 0.2]),
        np.array([1000.0]),
        model,
        geom,
    )
    expected = (
        float(single_star_frame.catalog["rate_e_s"][0])
        * star_frame.meta["exptime"]
    )
    assert flux[0] == pytest.approx(expected, rel=0.02)
    assert x_fit[0] == pytest.approx(x_true, abs=0.05)
    assert y_fit[0] == pytest.approx(y_true, abs=0.05)
    assert list(fflags) == [0]  # a clean fit carries no FLAG_FIT
