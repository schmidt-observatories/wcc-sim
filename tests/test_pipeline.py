import numpy as np
import pytest

from tests.conftest import DEC0, RA0

SHAPE = (256, 256)  # small subarray for speed


def run(canned_catalog, **kw):
    from wcc_sim import simulate_field

    kw.setdefault("catalog", canned_catalog)
    kw.setdefault("shape", SHAPE)
    kw.setdefault("stamp_npix", 33)
    kw.setdefault("seed", 42)
    return simulate_field(RA0, DEC0, sensorfilter="zwo:r", **kw)


def test_field_structure(canned_catalog):
    f = run(canned_catalog)
    assert f.image_adu.shape == SHAPE
    assert f.image_adu.dtype == np.float32
    assert f.saturation_mask.dtype == bool
    assert {"x", "y", "spt", "rate_e_s", "in_image", "saturated"} <= set(
        f.catalog.colnames
    )
    assert f.params["sensorfilter"] == "zwo:r"
    assert f.params["focus"] == 0  # zwo:r default is in-focus


def test_star_lands_at_wcs_position(canned_catalog):
    f = run(canned_catalog, add_noise=False)
    # star 0 is at the pointing center
    x, y = f.wcs.world_to_pixel_values(RA0, DEC0)
    img = f.image_clean - np.median(f.image_clean)
    cy, cx = np.unravel_index(img.argmax(), img.shape)
    assert cx == pytest.approx(x, abs=1.0)
    assert cy == pytest.approx(y, abs=1.0)


def test_photometric_closure_vs_etc(canned_catalog):
    """Total injected flux of an interior star matches the ETC rate to <1%."""
    from wcc_sim.starflux import rate_for_spt

    single = canned_catalog[[1]]  # G=15 star, 1.5" from center (interior)
    f = run(single, add_noise=False, exptime=90.0)
    sky_dark = np.median(f.image_clean)  # uniform background level
    star_e = (f.image_clean - sky_dark).sum()
    spt = f.catalog["spt"][0]
    expected = rate_for_spt(spt, "zwo:r") * 90.0
    assert expected > 0  # guard: a zero rate would make the closure check vacuous
    assert star_e == pytest.approx(expected, rel=0.01)


def test_focus_override_changes_psf(canned_catalog):
    f0 = run(canned_catalog, add_noise=False, focus=0)
    f1 = run(canned_catalog, add_noise=False, focus=1, stamp_npix=257)
    assert f1.params["focus"] == 1
    assert f1.image_clean.max() < f0.image_clean.max()  # defocus spreads light


def test_bright_star_saturates(canned_catalog):
    bright = canned_catalog[[0]].copy()
    bright["phot_g_mean_mag"][0] = 5.0
    f = run(bright, exptime=90.0)
    assert f.saturation_mask.sum() > 0
    assert bool(f.catalog["saturated"][0])


def test_reproducible_with_seed(canned_catalog):
    f1 = run(canned_catalog, seed=7)
    f2 = run(canned_catalog, seed=7)
    assert np.array_equal(f1.image_adu, f2.image_adu)


def test_write_fits(canned_catalog, tmp_path):
    path = tmp_path / "field.fits"
    run(canned_catalog, output=str(path))
    from astropy.io import fits

    with fits.open(path) as hdul:
        assert [h.name for h in hdul] == ["SCI", "SATMASK", "CAT", "CLEAN"]
        assert hdul["SCI"].header["NSRC"] == len(hdul["CAT"].data)


def test_empty_catalog_sky_only():
    from astropy.table import Table

    from wcc_sim import simulate_field
    from wcc_sim.catalog import _empty_table

    f = simulate_field(
        RA0, DEC0, sensorfilter="zwo:r", catalog=_empty_table(),
        shape=SHAPE, seed=1,
    )
    assert f.image_adu.std() > 0  # noise present
    assert len(f.catalog) == 0
