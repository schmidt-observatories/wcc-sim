"""Regression tests for the input edge cases of audit issue #22."""

import numpy as np
import pytest
from astropy.table import Table

from tests.conftest import DEC0, RA0

SHAPE = (64, 64)


def run(canned_catalog, **kw):
    from wcc_sim import simulate_field

    kw.setdefault("catalog", canned_catalog)
    kw.setdefault("shape", SHAPE)
    kw.setdefault("stamp_npix", 33)
    kw.setdefault("seed", 1)
    kw.setdefault("add_noise", False)
    return simulate_field(RA0, DEC0, sensorfilter="zwo:r", **kw)


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"exptime": -1.0}, "exptime"),
        ({"exptime": float("nan")}, "exptime"),
        ({"n_reads": 0}, "n_reads"),
        ({"focus": True}, "focus"),
        ({"focus": 3}, "focus"),
        ({"oversample": 0}, "oversample"),
        ({"oversample": 10}, "oversample"),  # even: shifts every star
        ({"shape": (0, 64)}, "shape"),
        ({"scatter_fraction": 0.0}, "scatter_fraction"),
        ({"scatter_fraction": 2.0}, "scatter_fraction"),
        ({"wing_floor_sigma": 0.0}, "wing_floor_sigma"),
        ({"wing_floor_sigma": -0.1}, "wing_floor_sigma"),
    ],
)
def test_bad_inputs_are_rejected_up_front(canned_catalog, kw, match):
    with pytest.raises(ValueError, match=match):
        run(canned_catalog, **kw)


def test_bad_pointing_is_rejected_before_the_query(monkeypatch):
    """dec=95 or ra=400 must not reach Gaia (or astropy's WCS)."""
    from wcc_sim import pipeline

    monkeypatch.setattr(
        pipeline, "query_gaia", lambda *a, **k: pytest.fail("queried Gaia")
    )
    with pytest.raises(ValueError, match="dec"):
        pipeline.simulate_field(RA0, 95.0, shape=SHAPE)
    with pytest.raises(ValueError, match="ra"):
        pipeline.simulate_field(400.0, DEC0, shape=SHAPE)


def test_misspelled_spt_override_names_the_template(canned_catalog):
    cat = canned_catalog.copy()
    cat["spt"] = ["", "G2X", "", "", ""]
    with pytest.raises(ValueError, match="G2X"):
        run(cat)


def test_star_on_the_far_side_of_the_sky_is_dropped_not_fatal(canned_catalog):
    """A user-catalog star >90 deg away projects to NaN; it must be dropped."""
    cat = canned_catalog.copy()
    cat["ra"][1] = (RA0 + 180.0) % 360.0
    cat["dec"][1] = -DEC0
    with pytest.warns(UserWarning, match="do not project"):
        f = run(cat)
    assert len(f.catalog) == len(canned_catalog) - 1
    assert np.isfinite(f.catalog["x"]).all()


def test_mag_limit_is_not_recorded_for_a_user_catalog(canned_catalog):
    f = run(canned_catalog, mag_limit=12.0)
    assert f.params["mag_limit"] is None
    assert "MAGLIM" not in f.to_hdulist()[0].header


def test_run_photometry_accepts_a_single_frame(phot_frames, tmp_path):
    """A bare path used to be iterated character by character."""
    from wcc_phot import run_photometry

    path = tmp_path / "f0.fits"
    phot_frames[0].write(str(path))
    r_path = run_photometry(str(path), target=0, n_ref=3, method="aperture")
    r_obj = run_photometry(phot_frames[0], target=0, n_ref=3, method="aperture")
    assert len(r_path.lightcurve) == len(r_obj.lightcurve) == 1


def test_mag_limit_still_recorded_for_gaia_queries(monkeypatch, canned_catalog):
    from wcc_sim import pipeline

    monkeypatch.setattr(pipeline, "query_gaia", lambda *a, **k: canned_catalog)
    f = pipeline.simulate_field(RA0, DEC0, shape=SHAPE, stamp_npix=33,
                                mag_limit=19.5, add_noise=False)
    assert f.params["mag_limit"] == 19.5
