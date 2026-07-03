import dataclasses

import numpy as np
import pytest

from wcc_phot import run_photometry
from wcc_phot.io import load_frame

PHOT_KW = dict(r_ap=6.0, centroid_box=13, fit_shape=13)


def test_end_to_end_aperture(phot_frames):
    result = run_photometry(phot_frames, target=0, **PHOT_KW)
    assert result.params["n_ref"] == 10
    assert list(result.stars["role"]) == ["target"] + ["ref"] * 10
    assert list(result.stars["source_id"]) == list(range(11))

    assert len(result.measurements) == 11 * 3
    target_rows = result.measurements[result.measurements["role"] == "target"]
    assert np.all(target_rows["flags"] == 0)
    # re-centroided positions stay near the WCS prediction
    assert np.all(
        np.hypot(
            result.measurements["x"] - result.measurements["x_init"],
            result.measurements["y"] - result.measurements["y_init"],
        )
        < 1.0
    )

    lc = result.lightcurve
    assert len(lc) == 3
    assert np.all(lc["n_ref"] == 10)
    # constant star: normalized flux consistent with the propagated errors
    assert np.all(
        np.abs(lc["rel_flux_norm"] - 1.0) < 5 * lc["rel_flux_norm_err"]
    )
    assert np.all(lc["rel_flux_norm_err"] < 0.02)


def test_end_to_end_psf(phot_frames):
    result = run_photometry(phot_frames, target=0, method="psf", **PHOT_KW)
    lc = result.lightcurve
    assert np.all(np.isfinite(result.measurements["flux_e"]))
    assert np.all(result.measurements["flux_e"] > 0)
    assert np.all(np.abs(lc["rel_flux_norm"] - 1.0) < 0.02)


def test_times_recorded(phot_frames):
    times = [2460000.5, 2460000.6, 2460000.7]
    result = run_photometry(phot_frames, target=0, times=times, **PHOT_KW)
    assert list(result.lightcurve["time"]) == times
    with pytest.raises(ValueError, match="len\\(times\\)"):
        run_photometry(phot_frames, target=0, times=[1.0], **PHOT_KW)


def test_mixed_focus_raises(phot_frames):
    f0 = load_frame(phot_frames[0])
    f1 = load_frame(phot_frames[1])
    f1 = dataclasses.replace(f1, meta={**f1.meta, "focus": 2})
    with pytest.raises(ValueError, match="mix"):
        run_photometry([f0, f1], target=0, **PHOT_KW)


def test_fits_roundtrip(phot_frames, tmp_path):
    from astropy.io import fits

    path = tmp_path / "phot.fits"
    result = run_photometry(phot_frames, target=0, output=str(path), **PHOT_KW)
    with fits.open(path) as hdul:
        assert [h.name for h in hdul] == ["PRIMARY", "STARS", "PHOT", "LC"]
        header = hdul[0].header
        assert header["TARGID"] == 0
        assert header["METHOD"] == "aperture"
        assert header["NREF"] == 10
        assert len(hdul["LC"].data) == len(result.lightcurve)
        assert len(hdul["PHOT"].data) == len(result.measurements)


def test_cli(phot_frames, tmp_path, capsys):
    from wcc_phot.cli import main

    paths = []
    for k, frame in enumerate(phot_frames):
        path = tmp_path / f"frame{k}.fits"
        frame.write(str(path))
        paths.append(str(path))
    out = tmp_path / "phot.fits"
    csv = tmp_path / "lc.ecsv"
    code = main(
        paths
        + [
            "--source-id", "0",
            "--r-ap", "6",
            "--centroid-box", "13",
            "-o", str(out),
            "--lc-csv", str(csv),
        ]
    )
    assert code == 0
    assert out.exists() and csv.exists()
    assert "3 frames" in capsys.readouterr().out


def test_cli_requires_target(tmp_path):
    from wcc_phot.cli import main

    with pytest.raises(SystemExit):
        main(["nonexistent.fits", "-o", str(tmp_path / "x.fits")])
