import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from wcc_phot import make_report, run_photometry
from wcc_phot.report import _binned_rms, _style_context, compute_metrics

PHOT_KW = dict(r_ap=6.0, centroid_box=13, fit_shape=13)


@pytest.fixture(scope="module")
def phot_result(phot_frames):
    return run_photometry(phot_frames, target=0, **PHOT_KW)


def test_make_report_writes_pdf_and_png(phot_result, phot_frames, tmp_path):
    out = tmp_path / "report.pdf"
    paths = make_report(phot_result, frame=phot_frames[0], path=str(out))
    assert paths == [str(tmp_path / "report.pdf"), str(tmp_path / "report.png")]
    for p in paths:
        assert (tmp_path / p).stat().st_size > 0


def test_make_report_without_frame(phot_result, tmp_path):
    paths = make_report(phot_result, path=str(tmp_path / "nofield.png"))
    assert all((tmp_path / p).stat().st_size > 0 for p in paths)


def test_run_photometry_report_hookup(phot_frames, tmp_path):
    run_photometry(
        phot_frames, target=0, report=str(tmp_path / "auto.pdf"), **PHOT_KW
    )
    assert (tmp_path / "auto.pdf").stat().st_size > 0
    assert (tmp_path / "auto.png").stat().st_size > 0


def test_make_report_without_gks_style(phot_result, tmp_path, monkeypatch):
    # simulate an environment where the 'gks' house style is not installed;
    # _style_context must fall back to defaults rather than raise on __enter__
    import matplotlib.style

    monkeypatch.setattr(
        matplotlib.style,
        "available",
        [s for s in matplotlib.style.available if s != "gks"],
    )
    with _style_context():  # would raise OSError before the fix
        pass
    paths = make_report(phot_result, path=str(tmp_path / "nogks.png"))
    assert all((tmp_path / p).stat().st_size > 0 for p in paths)


def test_compute_metrics(phot_result):
    metrics = compute_metrics(phot_result)
    n_stars = len(phot_result.stars)
    assert metrics["star_rms"].shape == (n_stars,)
    assert metrics["star_err"].shape == (n_stars,)
    assert np.all(metrics["star_err"] > 0)
    # constant stars: scatter should be within a few times the prediction
    assert metrics["rms_over_err"] < 5
    assert metrics["flag_counts"] == {
        "centroid": 0, "saturated": 0, "edge": 0, "fit": 0, "noflux": 0
    }


def test_binned_rms_white_noise_scaling():
    rng = np.random.default_rng(7)
    values = rng.normal(1.0, 1e-3, 3000)
    sizes, rms = _binned_rms(values)
    assert sizes[0] == 1
    expected = rms[0] / np.sqrt(sizes[-1])
    assert rms[-1] == pytest.approx(expected, rel=0.5)
