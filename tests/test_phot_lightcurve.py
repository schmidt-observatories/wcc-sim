"""Regression tests for the differential-photometry guards of issue #12."""

import numpy as np
import pytest
from astropy.table import Table

from wcc_phot.flags import FLAG_FIT, FLAG_NOFLUX, FLAG_SATURATED
from wcc_phot.lightcurve import build_lightcurve, ensemble_ratio, usable_references


def _measurements(n_frames=3, n_ref=3, target=1000.0, ref=5000.0):
    rows = []
    for k in range(n_frames):
        rows.append(dict(frame=k, time=float(k), star=0, role="target",
                         flux_e=target, flux_err_e=np.sqrt(target), flags=0,
                         x=10.0, y=10.0))
        for j in range(1, n_ref + 1):
            rows.append(dict(frame=k, time=float(k), star=j, role="ref",
                             flux_e=ref, flux_err_e=np.sqrt(ref), flags=0,
                             x=10.0 * j, y=10.0))
    return Table(rows)


def test_zero_target_is_a_measurement_of_zero_not_a_crash():
    rel, rel_err, ens, _, ok = ensemble_ratio(0.0, 3.0, [100.0, 100.0], [10.0, 10.0])
    assert ok and rel == 0.0 and rel_err == pytest.approx(3.0 / 200.0)


def test_zero_or_invalid_ensemble_gives_nan_and_a_flag():
    rel, rel_err, _, _, ok = ensemble_ratio(100.0, 10.0, [0.0, 0.0], [1.0, 1.0])
    assert not ok and np.isnan(rel) and np.isnan(rel_err)
    rel, _, _, _, ok = ensemble_ratio(np.nan, 10.0, [100.0], [10.0])
    assert not ok and np.isnan(rel)
    meas = _measurements()
    meas["flux_e"][(meas["frame"] == 1) & (meas["role"] == "ref")] = 0.0
    lc = build_lightcurve(meas)  # must not raise
    assert np.isnan(lc["rel_flux"][1]) and (lc["flags"][1] & FLAG_NOFLUX)
    assert np.isfinite(lc["rel_flux_norm"][0]) and lc["flags"][0] == 0


def test_error_propagation_is_the_quotient_variance():
    t, et, r, er = 1000.0, 31.6, np.array([4000.0, 6000.0]), np.array([63.2, 77.5])
    rel, rel_err, _, _, _ = ensemble_ratio(t, et, r, er)
    R, eR = r.sum(), np.sqrt((er**2).sum())
    assert rel_err == pytest.approx(np.sqrt(et**2 / R**2 + t**2 * eR**2 / R**4), rel=1e-12)


def test_reference_flagged_in_one_frame_leaves_the_ensemble_in_every_frame():
    """Dropping a ref from one frame only would step the light curve by its
    share of the ensemble; a constant target must stay constant."""
    meas = _measurements(n_frames=4, n_ref=3)
    hit = (meas["frame"] == 2) & (meas["star"] == 3)
    meas["flux_e"][hit] = 0.0
    meas["flags"][hit] = FLAG_SATURATED
    assert usable_references(meas) == [1, 2]
    lc = build_lightcurve(meas)
    assert list(lc["n_ref"]) == [2, 2, 2, 2]
    assert np.ptp(lc["rel_flux"]) == 0.0
    assert np.all(lc["flags"] == 0)  # the target itself was never flagged


def test_all_references_flagged_falls_back_to_all_with_a_warning():
    meas = _measurements(n_frames=2, n_ref=2)
    meas["flags"][meas["role"] == "ref"] = FLAG_SATURATED
    with pytest.warns(UserWarning, match="every reference star is flagged"):
        assert usable_references(meas) == [1, 2]


def test_fit_flags_fold_photutils_status_into_one_bit():
    from wcc_phot.psfphot import fit_flags

    result = Table({"flux_fit": [10.0, np.nan, 5.0], "flags": [0, 0, 2]})
    assert list(fit_flags(result)) == [0, FLAG_FIT, FLAG_FIT]


def test_compute_metrics_uses_only_accepted_frames():
    from wcc_phot.pipeline import PhotometryResult
    from wcc_phot.report import compute_metrics

    meas = _measurements(n_frames=5, n_ref=3)
    meas["flux_e"][(meas["frame"] == 4) & (meas["role"] == "ref")] = 0.0
    lc = build_lightcurve(meas)
    stars = Table({"star": [0, 1, 2, 3], "role": ["target"] + ["ref"] * 3,
                   "gmag": [15.0] * 4, "used": [False, True, True, True]})
    metrics = compute_metrics(PhotometryResult(stars, meas, lc, {}))
    assert np.isfinite(metrics["rms"]) and np.all(np.isfinite(metrics["star_rms"]))
    assert metrics["flag_counts"]["noflux"] == 0  # the LC flag, not a measurement flag
