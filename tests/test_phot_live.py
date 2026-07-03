import matplotlib

matplotlib.use("Agg", force=True)  # headless: no window, draw paths still run

import numpy as np
import pytest

from wcc_phot import LiveViewer, run_photometry

PHOT_KW = dict(r_ap=6.0, centroid_box=13, fit_shape=13)

EVENT_KEYS = {
    "frame", "n_frames", "time", "image_e", "wcs", "x", "y", "roles",
    "geom", "flux_e", "flux_err_e", "flags", "rel_flux", "rel_flux_err",
}


def test_on_frame_events(phot_frames):
    events = []
    result = run_photometry(
        phot_frames, target=0, on_frame=events.append, **PHOT_KW
    )
    assert len(events) == 3
    for k, event in enumerate(events):
        assert EVENT_KEYS <= set(event)
        assert event["frame"] == k and event["n_frames"] == 3
        assert event["roles"][0] == "target"
        assert len(event["x"]) == 11
    # the streamed rel_flux matches the final light curve
    assert [e["rel_flux"] for e in events] == pytest.approx(
        list(result.lightcurve["rel_flux"])
    )


def test_live_viewer_updates(phot_frames):
    viewer = LiveViewer(pause=0.001, zoom=80)
    run_photometry(phot_frames, target=0, on_frame=viewer, **PHOT_KW)
    assert viewer._fig is not None
    assert len(viewer._fig.axes) == 2
    assert len(viewer._circles) == 11
    assert len(viewer._rel) == 3
    assert np.all(np.isfinite(viewer._rel))
    viewer.hold()  # returns immediately on a non-GUI backend
    viewer.close()
    assert viewer._fig is None


def test_cli_live_flag(phot_frames, tmp_path, capsys):
    from wcc_phot.cli import main

    paths = []
    for k, frame in enumerate(phot_frames):
        path = tmp_path / f"frame{k}.fits"
        frame.write(str(path))
        paths.append(str(path))
    code = main(
        paths
        + [
            "--source-id", "0",
            "--r-ap", "6",
            "--centroid-box", "13",
            "--live", "--live-pause", "0.001",
            "-o", str(tmp_path / "phot.fits"),
        ]
    )
    assert code == 0
    assert "3 frames" in capsys.readouterr().out
