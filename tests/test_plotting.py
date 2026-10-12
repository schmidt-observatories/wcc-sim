"""Every ``.plot_*`` method returns an Axes, honours ``ax=``, and the reports still write."""

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from wcc_phot import run_photometry
from wcc_sim import plotting

PHOT_KW = dict(r_ap=6.0, centroid_box=13, fit_shape=13)


@pytest.fixture(scope="module")
def phot_result(phot_frames):
    return run_photometry(phot_frames, target=0, **PHOT_KW)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------------------- #
# plotting module                                                              #
# --------------------------------------------------------------------------- #

def test_style_is_packaged_and_loads():
    import matplotlib.style

    assert plotting.STYLE_PATH.endswith("plot_default.mplstyle")
    with plotting.style_context():
        assert matplotlib.rcParams["xtick.direction"] == "in"
    plotting.use_style()
    assert matplotlib.rcParams["legend.frameon"] is False


def test_axes_helper_returns_given_axes_untouched():
    _, ax = plt.subplots()
    with plotting.axes(ax) as got:
        assert got is ax
    with plotting.axes(None) as new:
        assert new is not ax


def test_radial_profile_and_encircled_energy_on_a_gaussian():
    n = 101
    yy, xx = np.mgrid[:n, :n] - n // 2
    sigma = 4.0
    g = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    r, prof = plotting.radial_profile(g, n // 2, n // 2, r_max=30, n_bins=30)
    assert prof[0] > prof[-1] > 0
    assert np.all(np.diff(prof[np.isfinite(prof)]) <= 1e-9)
    r_ee, ee = plotting.encircled_energy(g)
    assert ee[-1] == pytest.approx(1.0)
    # 1 - exp(-r^2 / 2 sigma^2): half the energy inside r = sigma * sqrt(2 ln 2)
    i = np.searchsorted(r_ee, sigma * np.sqrt(2 * np.log(2)))
    assert ee[i] == pytest.approx(0.5, abs=0.03)


def test_plot_image_downsamples_and_overlays(single_star_frame):
    f = single_star_frame
    ax = plotting.plot_image(f.image_adu, catalog=f.catalog,
                             satmask=f.saturation_mask, max_px=64)
    # the display is block-reduced but the axes stay in native pixels
    assert ax.get_xlim() == pytest.approx((-0.5, 255.5))
    assert len(ax.collections) >= 1  # the catalog scatter
    _, ax2 = plt.subplots()
    assert plotting.plot_image(f.image_e, ax=ax2) is ax2


def test_stamp_and_profile_plots_return_axes(single_star_frame):
    stamp = single_star_frame.models["stamp"]
    assert plotting.plot_stamp(stamp, half_width=20, plate_scale_mas=16.87).has_data()
    assert plotting.plot_encircled_energy(stamp, label="airy").has_data()
    ax = plotting.plot_radial_profile(single_star_frame.image_clean, 127.5, 127.5,
                                      bg=0.0, r_max=60, log=True)
    assert ax.get_yscale() == "log" and ax.get_xscale() == "log"


# --------------------------------------------------------------------------- #
# SimulatedField methods                                                       #
# --------------------------------------------------------------------------- #

def test_field_plot_methods(single_star_frame, tmp_path):
    f = single_star_frame
    for which in ("adu", "e", "clean"):
        assert f.plot(which=which, catalog=True, satmask=True).has_data()
    _, ax = plt.subplots()
    assert f.plot(ax=ax) is ax
    assert f.plot_cutout(half_width=40).has_data()
    assert f.plot_cutout(which="adu", half_width=40).has_data()
    assert f.plot_satmask().has_data()
    ax = f.plot_psf_profile()
    assert ax.get_xscale() == "log" and len(ax.lines) >= 2
    with pytest.raises(ValueError):
        f.plot(which="nope")
    pdf, png = f.psf_report(str(tmp_path / "psf.pdf"))
    assert (tmp_path / "psf.pdf").stat().st_size > 0
    assert (tmp_path / "psf.png").stat().st_size > 0


# --------------------------------------------------------------------------- #
# PhotometryResult methods                                                     #
# --------------------------------------------------------------------------- #

def test_result_plot_methods(phot_result, phot_frames, tmp_path):
    r = phot_result
    for name in ("plot", "plot_lightcurve", "plot_binned_rms", "plot_raw_fluxes",
                 "plot_rms_vs_mag", "plot_background"):
        ax = getattr(r, name)()
        assert ax.has_data(), name
    assert r.plot_centroids(coord="y").get_ylabel().endswith("y [px]")
    assert r.plot_field(phot_frames[0]).images
    _, ax = plt.subplots()
    assert r.plot_lightcurve(ax=ax) is ax
    paths = r.report(str(tmp_path / "rep.pdf"), frame=phot_frames[0])
    assert all((tmp_path / p).stat().st_size > 0 for p in paths)


def test_live_viewer_colours_come_from_plotting():
    from wcc_phot import live

    assert live.TARGET_COLOR == plotting.TARGET_COLOR
    assert live.REF_COLOR == plotting.REF_COLOR


def test_style_context_keeps_interactive_state():
    # The inline Jupyter backend switches `interactive` on lazily, from
    # inside the first figure's creation. A context that rolled that back on
    # exit would silence every later cell, so the context must not restore it.
    import matplotlib

    was = matplotlib.rcParams["interactive"]
    try:
        with plotting.style_context():
            matplotlib.rcParams["interactive"] = not was  # what activation does
            assert matplotlib.rcParams["xtick.direction"] == "in"
        assert matplotlib.rcParams["interactive"] == (not was)
    finally:
        matplotlib.rcParams["interactive"] = was
