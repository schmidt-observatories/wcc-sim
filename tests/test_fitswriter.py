import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table


def make_inputs():
    from wcc_sim.wcsutil import build_wcs

    image = np.ones((64, 64), dtype=np.float32)
    mask = np.zeros((64, 64), dtype=bool)
    mask[1, 2] = True
    cat = Table(
        {
            "source_id": np.array([1, 2], dtype=np.int64),
            "ra": [150.1, 150.11],
            "dec": [2.2, 2.21],
            "x": [10.0, 20.0],
            "y": [30.0, 40.0],
            "phot_g_mean_mag": [15.0, 16.0],
            "spt": ["G2V", "K5V"],
            "rate_e_s": [1e4, 4e3],
        }
    )
    wcs = build_wcs(150.1, 2.2, 16.87, 0.0, (64, 64))
    params = {
        "ra": 150.1, "dec": 2.2, "pa": 0.0, "sensorfilter": "zwo:r",
        "focus": 0, "exptime": 90.0, "n_reads": 1, "jitter_sigma_mas": 10.0,
        "mag_limit": 21.0, "seed": 42, "gaia_radius_arcsec": 100.0,
        "n_sources": 2, "plate_scale_mas": 16.87,
    }
    return image, mask, cat, wcs, params


def test_hdulist_structure_and_header():
    from wcc_sim.fitswriter import build_hdulist

    image, mask, cat, wcs, params = make_inputs()
    hdul = build_hdulist(image, mask, cat, wcs, params, image_clean=image * 2)
    names = [h.name for h in hdul]
    assert names == ["SCI", "SATMASK", "CAT", "CLEAN"]
    h = hdul["SCI"].header
    assert h["SENSORF"] == "zwo:r"
    assert h["FOCUS"] == 0
    assert h["EXPTIME"] == 90.0
    assert h["CTYPE1"] == "RA---TAN"
    assert "WCCSIMV" in h and "WCCETCV" in h
    assert hdul["SATMASK"].data.dtype == np.uint8
    assert hdul["SATMASK"].data[1, 2] == 1
    assert list(hdul["CAT"].data["spt"]) == ["G2V", "K5V"]


def test_write_and_read_roundtrip(tmp_path):
    from wcc_sim.fitswriter import write_fits

    image, mask, cat, wcs, params = make_inputs()
    path = tmp_path / "out.fits"
    write_fits(str(path), image, mask, cat, wcs, params)
    with fits.open(path) as hdul:
        assert hdul[0].header["RA_PNT"] == pytest.approx(150.1)
        assert hdul[0].data.dtype.newbyteorder("=") == np.float32
        assert len(hdul["CAT"].data) == 2
        assert "CLEAN" not in [h.name for h in hdul]
