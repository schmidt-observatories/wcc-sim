"""Issue #11: photometry sees one delivered product whatever the input form.

`load_frame` used to hand photometry the pre-ADC electron image for a
SimulatedField but `SCI * GAIN` for a FITS file, so a noiseless bright star
had no ceiling from the object and the ADC ceiling from the file, and the
zero clip of the ADU image dropped negative read-noise pixels only on the
file path. Every path now loads ``(ADU - bias) * gain``.
"""

import numpy as np
import pytest
from astropy.table import Table

from tests.conftest import DEC0, RA0


def _field(catalog, **kw):
    from wcc_sim import simulate_field

    kw.setdefault("shape", (33, 33))
    kw.setdefault("stamp_npix", 33)
    kw.setdefault("wings", False)
    return simulate_field(RA0, DEC0, sensorfilter="zwo:r", catalog=catalog, **kw)


def _bright():
    return Table({
        "source_id": np.array([0], dtype=np.int64), "ra": [RA0], "dec": [DEC0],
        "phot_g_mean_mag": [8.0], "phot_bp_mean_mag": [8.4], "phot_rp_mean_mag": [7.6],
    })


@pytest.mark.parametrize(
    "catalog, kw",
    [
        (_bright(), dict(exptime=90.0, add_noise=False)),      # well/ADC-capped
        (_bright(), dict(exptime=90.0, seed=1)),               # capped, with noise
        (_bright()[:0], dict(exptime=0.001, seed=42, shape=(64, 64))),  # sky-only
    ],
    ids=["bright-noiseless", "bright-noisy", "sky-only"],
)
def test_object_hdulist_and_path_deliver_the_same_pixels(catalog, kw, tmp_path):
    from wcc_phot.io import load_frame

    f = _field(catalog, **kw)
    path = tmp_path / "f.fits"
    f.write(str(path))
    frames = [load_frame(f), load_frame(f.to_hdulist()), load_frame(str(path))]
    for fr in frames[1:]:
        assert np.allclose(fr.image_e, frames[0].image_e, atol=1e-3)
        assert np.array_equal(fr.satmask, frames[0].satmask)
    if not f.params["add_noise"]:  # the delivered image carries the ceiling
        ceiling = f.params["saturation_e"] * f.params["n_reads"]
        assert frames[0].image_e.max() <= ceiling * (1 + 1e-6)
    # the delivered image is what the ADC clipped: nothing below -bias*gain
    assert frames[0].image_e.min() >= -f.params["bias"] * f.params["gain"] - 1e-6


def _hdulist(image_adu, params):
    from wcc_sim.fitswriter import build_hdulist
    from wcc_sim.wcsutil import build_wcs

    cat = Table({"source_id": np.array([1], dtype=np.int64), "ra": [RA0], "dec": [DEC0]})
    wcs = build_wcs(RA0, DEC0, 16.87, 0.0, image_adu.shape)
    base = {
        "exptime": 1.0, "n_reads": 1, "gain": 2.0, "read_noise": 3.0, "sky_e_s": 0.1,
        "dark_e_s": 0.01, "focus": 0, "sensorfilter": "zwo:r", "jitter_sigma_mas": 10.0,
        "plate_scale_mas": 16.87,
    }
    return build_hdulist(image_adu, np.zeros(image_adu.shape, bool), cat, wcs, {**base, **params})


def test_delivered_electrons_subtract_the_bias():
    from wcc_phot.io import frame_meta, load_frame

    hdul = _hdulist(np.full((8, 8), 105.0, np.float32), {"bias": 100.0})
    assert hdul["SCI"].header["BIAS"] == 100.0
    assert frame_meta(hdul)["bias"] == 100.0
    assert np.allclose(load_frame(hdul).image_e, 10.0)


def test_file_without_bias_card_loads_as_zero_bias():
    from wcc_phot.io import load_frame

    hdul = _hdulist(np.full((8, 8), 105.0, np.float32), {})
    assert "BIAS" not in hdul["SCI"].header
    assert np.allclose(load_frame(hdul).image_e, 210.0)


def test_photometry_agrees_between_object_and_file(phot_frames, tmp_path):
    """Acceptance: the same flux, error and flags from the object and its file."""
    from wcc_phot import run_photometry

    path = tmp_path / "f0.fits"
    phot_frames[0].write(str(path))
    kw = dict(target=0, n_ref=3, method="aperture")
    a = run_photometry(phot_frames[0], **kw).lightcurve
    b = run_photometry(str(path), **kw).lightcurve
    for col in ("rel_flux", "rel_flux_norm_err", "n_ref", "flags"):
        assert np.allclose(a[col], b[col], rtol=1e-6, equal_nan=True), col
