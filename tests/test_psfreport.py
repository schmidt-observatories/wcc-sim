import matplotlib.colors
import numpy as np
import pytest
from astropy.table import Table

RA0, DEC0 = 150.1, 2.2
SHAPE = (2001, 2001)


@pytest.fixture(scope="module")
def mag0_catalog():
    return Table({
        "source_id": np.array([0], dtype=np.int64),
        "ra": [RA0], "dec": [DEC0],
        "phot_g_mean_mag": [0.0],
        "phot_bp_mean_mag": [0.35],
        "phot_rp_mean_mag": [-0.35],
    })


def _run(catalog, **kw):
    from wcc_sim import simulate_field

    opts = dict(
        sensorfilter="zwo:g", focus=0, exptime=60.0, catalog=catalog,
        shape=SHAPE, add_noise=False, seed=0, wavelength_nm=450.0,
    )
    opts.update(kw)
    return simulate_field(RA0, DEC0, **opts)


@pytest.fixture(scope="module")
def mag0_field(mag0_catalog):
    return _run(mag0_catalog)


@pytest.fixture(scope="module")
def decomposition(mag0_field):
    from wcc_sim.psfreport import psf_decomposition

    return psf_decomposition(mag0_field)


# --------------------------------------------------------------------------- #
# The wavelength override                                                      #
# --------------------------------------------------------------------------- #

def test_wavelength_override_is_recorded(mag0_field):
    assert mag0_field.params["wavelength_nm"] == pytest.approx(450.0)


def test_wavelength_override_narrows_the_psf(mag0_catalog):
    """450 nm diffracts less than the filter's own 477 nm centre."""
    blue = _run(mag0_catalog, wavelength_nm=450.0)
    red = _run(mag0_catalog, wavelength_nm=700.0)
    assert blue.image_clean.max() > red.image_clean.max()


def test_wavelength_override_leaves_the_photometry_alone(mag0_catalog):
    """It sets PSF geometry only -- the bandpass and hence the rate is fixed."""
    blue = _run(mag0_catalog, wavelength_nm=450.0)
    red = _run(mag0_catalog, wavelength_nm=700.0)
    assert float(blue.catalog["rate_e_s"][0]) == pytest.approx(
        float(red.catalog["rate_e_s"][0]), rel=1e-12
    )


def test_default_wavelength_is_the_sensor_central_wavelength(mag0_catalog):
    field = _run(mag0_catalog, wavelength_nm=None)
    assert field.params["wavelength_nm"] == pytest.approx(477.33, abs=0.1)


# --------------------------------------------------------------------------- #
# The decomposition                                                            #
# --------------------------------------------------------------------------- #

def test_airy_to_scatter_peak_contrast_is_about_1e9(decomposition):
    """The number to eyeball: ~1e9 at 450 nm, jitter aside.

    Same definition as wcc_etc.scatter_psf.report_diagnostics -- the
    analytic Airy peak for a (1 - f_scat) core over the halo's central
    irradiance, both in W/mm^2 per watt of source.
    """
    assert decomposition["contrast"] == pytest.approx(1.07e9, rel=0.05)


def test_peak_irradiances_are_in_physical_units(decomposition):
    assert decomposition["airy_peak_irradiance"] == pytest.approx(1.714e4, rel=0.01)
    assert decomposition["scatter_peak_irradiance"] == pytest.approx(
        1.597e-5, rel=0.01
    )


def test_components_sum_to_the_total(decomposition):
    total = decomposition["airy"] + decomposition["scatter"]
    assert np.allclose(total, decomposition["total"], rtol=1e-12)


def test_scatter_overtakes_the_airy_wing_at_the_crossover(decomposition):
    r_x = decomposition["crossover_px"]
    assert 800.0 < r_x < 2000.0
    r = decomposition["r_px"]
    airy, scat = decomposition["airy"], decomposition["scatter"]
    assert scat[r < 0.5 * r_x].max() < airy[r < 0.5 * r_x].max()
    inner = r > 2.0 * r_x
    if inner.any():
        assert (scat[inner] > airy[inner]).all()


def test_measured_profile_tracks_the_model(decomposition):
    """End-to-end: what landed in the image matches the two-term model.

    This is the real check on the whole chain -- stamp, crossfade, on-stamp
    halo and the drawn halo beyond it all have to line up with the profile
    the model predicts.
    """
    r = decomposition["r_px"]
    measured, total = decomposition["measured"], decomposition["total"]
    ok = np.isfinite(measured) & (measured > 0) & (r > 100.0) & (r < 900.0)
    assert ok.sum() > 10
    ratio = measured[ok] / total[ok]
    assert np.median(ratio) == pytest.approx(1.0, abs=0.25)


def test_decomposition_without_scatter_reports_no_halo(mag0_catalog):
    from wcc_sim.psfreport import psf_decomposition

    dec = psf_decomposition(_run(mag0_catalog, scatter=False))
    assert dec["contrast"] is None
    assert not dec["scatter"].any()


# --------------------------------------------------------------------------- #
# The report itself                                                            #
# --------------------------------------------------------------------------- #

def test_report_writes_pdf_and_png(mag0_field, tmp_path):
    from wcc_sim.psfreport import make_psf_report

    out = tmp_path / "psf.pdf"
    make_psf_report(mag0_field, str(out))
    assert out.exists() and out.stat().st_size > 5000
    assert (tmp_path / "psf.png").exists()


def test_pipeline_report_flag_writes_the_file(mag0_catalog, tmp_path):
    out = tmp_path / "auto.pdf"
    _run(mag0_catalog, report=str(out))
    assert out.exists()


def test_cli_report_and_wavelength_flags(mag0_catalog, tmp_path, monkeypatch):
    import wcc_sim.cli as cli

    captured = {}
    real = cli.simulate_field

    def spy(*a, **kw):
        captured.update(kw)
        kw["catalog"] = mag0_catalog
        kw["shape"] = (513, 513)
        return real(*a, **kw)

    monkeypatch.setattr(cli, "simulate_field", spy)
    out = tmp_path / "cli.fits"
    rep = tmp_path / "cli_report.pdf"
    assert cli.main([
        "--ra", str(RA0), "--dec", str(DEC0), "--sensorfilter", "zwo:g",
        "--exptime", "60", "--wavelength", "450", "--report", str(rep),
        "-o", str(out),
    ]) == 0
    assert captured["wavelength_nm"] == pytest.approx(450.0)
    assert rep.exists()


# --------------------------------------------------------------------------- #
# Reproducibility metadata                                                     #
# --------------------------------------------------------------------------- #

def test_params_record_what_is_needed_to_reproduce(mag0_field):
    p = mag0_field.params
    assert p["nx"] == SHAPE[1] and p["ny"] == SHAPE[0]
    assert p["n_pixels"] == SHAPE[0] * SHAPE[1]
    assert p["stamp_npix"] == 129  # focus-0 default
    assert p["oversample"] == 11
    assert p["add_noise"] is False


def test_reproduction_call_round_trips(mag0_field, mag0_catalog):
    """The snippet in the report must actually reproduce the figure."""
    from wcc_sim.psfreport import reproduction_call

    src = reproduction_call(mag0_field)
    assert "simulate_field(" in src
    assert "wavelength_nm=450.0" in src
    assert "scatter=True" in src
    assert "exptime=60.0" in src
    assert f"shape=({SHAPE[0]}, {SHAPE[1]})" in src


def test_reproduction_settings_cover_every_psf_knob(mag0_field):
    from wcc_sim.psfreport import reproduction_settings

    s = reproduction_settings(mag0_field)
    for key in ("sensorfilter", "focus", "exptime", "n_reads", "wavelength_nm",
                "jitter_sigma_mas", "stamp_npix", "oversample", "scatter",
                "scatter_fraction", "wings", "seed", "add_noise", "shape"):
        assert key in s, key


# --------------------------------------------------------------------------- #
# Figure structure                                                             #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def figure(mag0_field):
    from wcc_sim.psfreport import build_psf_figure

    fig, dec = build_psf_figure(mag0_field)
    return fig, dec


def test_figure_has_full_frame_and_zoom_image_panels(figure):
    fig, _ = figure
    titles = [ax.get_title() for ax in fig.axes]
    assert any("full frame" in t for t in titles)
    assert any("zoom" in t for t in titles)


def test_full_frame_panel_spans_the_whole_detector(figure, mag0_field):
    fig, _ = figure
    ny, nx = mag0_field.image_clean.shape
    full = [ax for ax in fig.axes if "full frame" in ax.get_title()][0]
    x0, x1 = full.get_xlim()
    assert x1 - x0 == pytest.approx(nx, rel=0.02)


def _full_frame_image_axes(fig, nx):
    """The log-stretched image that spans the whole array."""
    from matplotlib.colors import LogNorm

    return [ax for ax in fig.axes
            if ax.images and isinstance(ax.images[0].norm, LogNorm)
            and ax.get_xlim()[1] > 0.9 * nx][0]


def _cut_axes(fig):
    """The horizontal cut (above the frame) and the vertical one (right)."""
    top = [ax for ax in fig.axes if ax.get_ylabel().startswith("cut")]
    right = [ax for ax in fig.axes if ax.get_xlabel().startswith("cut")]
    assert len(top) == 1 and len(right) == 1, (top, right)
    return top[0], right[0]


def test_full_frame_has_centre_cut_panels(figure, mag0_field):
    """One cut panel above the frame, one to its right, both log in e-."""
    fig, _ = figure
    ny, nx = mag0_field.image_clean.shape
    top, right = _cut_axes(fig)
    assert top.get_xlim() == pytest.approx((0, nx), abs=1.0)
    assert right.get_ylim() == pytest.approx((0, ny), abs=1.0)
    assert top.get_yscale() == "log" and right.get_xscale() == "log"


def test_centre_cuts_are_the_real_row_and_column(figure, mag0_field):
    """Native resolution, not the block-reduced display, and both peak at the star."""
    fig, _ = figure
    img = np.asarray(mag0_field.image_clean, dtype=float)
    ny, nx = img.shape
    top, right = _cut_axes(fig)

    x, row = top.lines[0].get_data()
    assert row.size == nx
    assert np.nanmax(row) == pytest.approx(img[ny // 2, :].max(), rel=1e-6)
    assert x[int(np.argmax(row))] == pytest.approx(0.5 * nx, abs=1.5)

    col, y = right.lines[0].get_data()
    assert col.size == ny
    assert np.nanmax(col) == pytest.approx(img[:, nx // 2].max(), rel=1e-6)
    assert y[int(np.argmax(col))] == pytest.approx(0.5 * ny, abs=1.5)


def test_full_frame_marks_where_the_cuts_were_taken(figure, mag0_field):
    """A thin dashed grey line on the image per cut, through the centre."""
    fig, _ = figure
    ny, nx = mag0_field.image_clean.shape
    ax = _full_frame_image_axes(fig, nx)
    lines = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
    assert len(lines) == 2
    for ln in lines:
        assert ln.get_linewidth() <= 1.0
        assert 0.5 <= float(matplotlib.colors.to_rgb(ln.get_color())[0]) < 1.0
    ys = [ln.get_ydata()[0] for ln in lines if ln.get_ydata()[0] == ln.get_ydata()[1]]
    xs = [ln.get_xdata()[0] for ln in lines if ln.get_xdata()[0] == ln.get_xdata()[1]]
    assert ys == pytest.approx([ny // 2 + 0.5])
    assert xs == pytest.approx([nx // 2 + 0.5])


def _drawn(fig, ax):
    """The image's on-canvas box, after aspect and the axes divider."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(fig).draw()
    return ax.images[0].get_window_extent()


def test_saturation_mask_sits_under_the_frame_at_the_same_size(figure, mag0_field):
    """Same width and height as the full frame, directly below it."""
    fig, _ = figure
    ny, nx = mag0_field.image_clean.shape
    frame = _drawn(fig, _full_frame_image_axes(fig, nx))
    mask = _drawn(fig, [ax for ax in fig.axes
                        if "saturat" in ax.get_title()][0])
    assert mask.width == pytest.approx(frame.width, abs=1.0)
    assert mask.height == pytest.approx(frame.height, abs=1.0)
    assert mask.x0 == pytest.approx(frame.x0, abs=1.0)
    assert mask.y1 < frame.y0                          # below, not beside
    assert frame.y0 - mask.y1 < 0.5 * frame.height     # and right below


def test_the_full_frame_gets_a_big_share_of_the_page(figure, mag0_field):
    """It is the point of the report -- it dwarfs the zoom, not the reverse.

    A square test array cannot fill a landscape cell, so this is deliberately
    a floor rather than the ~11% of the page the real 3:2 detector gets.
    """
    fig, _ = figure
    ny, nx = mag0_field.image_clean.shape
    frame = _drawn(fig, _full_frame_image_axes(fig, nx))
    zoom = _drawn(fig, [ax for ax in fig.axes if "zoom" in ax.get_title()][0])
    page = fig.get_window_extent()
    assert frame.width * frame.height > 2.0 * zoom.width * zoom.height
    assert frame.width * frame.height > 0.07 * page.width * page.height


def _profile_axes(fig):
    return [ax for ax in fig.axes
            if ax.get_xlabel().startswith("radius")][0]


def _all_x_labels(fig):
    """Primary axis labels plus those of secondary (twinned) axes."""
    labels = []
    for ax in fig.axes:
        labels.append(ax.get_xlabel())
        labels += [child.get_xlabel() for child in ax.child_axes]
    return labels


def test_profile_has_mm_and_arcsec_axes(figure):
    fig, _ = figure
    labels = _all_x_labels(fig)
    assert any("mm" in t for t in labels), labels
    assert any("arcsec" in t for t in labels), labels
    assert any(t.startswith("radius [detector pixels]") for t in labels)


def test_profile_secondary_axes_have_their_own_ticks(figure):
    fig, _ = figure
    ax = _profile_axes(fig)
    assert len(ax.child_axes) == 2
    for child in ax.child_axes:
        assert len([t for t in child.get_xticklabels() if t.get_text()]) >= 2


def test_profile_secondary_axes_use_the_right_plate_scale(figure, mag0_field):
    """1000 px must read as 3.76 mm and 16.87 arcsec."""
    from wcc_sim.psfreport import _radius_conversions

    to_mm, to_arcsec = _radius_conversions(mag0_field)
    assert to_mm(1000.0) == pytest.approx(3.76, rel=1e-3)
    assert to_arcsec(1000.0) == pytest.approx(16.869, rel=1e-3)


def test_profile_ticks_point_outward(figure):
    fig, _ = figure
    ax = _profile_axes(fig)
    for tick in ax.xaxis.get_major_ticks():
        assert tick._apply_params  # sanity: params were applied
    assert ax.xaxis.get_tick_params(which="major")["direction"] == "out"
    assert ax.yaxis.get_tick_params(which="minor")["direction"] == "out"


def test_profile_labels_more_than_just_decades(figure):
    fig, _ = figure
    ax = _profile_axes(fig)
    minor = [t for t in ax.get_xticklabels(minor=True) if t.get_text()]
    assert len(minor) >= 4


def test_summary_reports_pixel_count_and_exposure(figure, mag0_field):
    fig, _ = figure
    text = " ".join(
        t.get_text() for ax in fig.axes for t in ax.texts
    )
    n = mag0_field.params["n_pixels"]
    assert f"{SHAPE[1]:,} x {SHAPE[0]:,} px" in text
    assert f"{n / 1e6:.2f} Mpix" in text
    assert "exposure time    60.0 s" in text
    assert "simulate_field(" in text


# --------------------------------------------------------------------------- #
# Crossover definitions and the saturation panel                               #
# --------------------------------------------------------------------------- #

def test_decomposition_reports_both_crossover_definitions(decomposition):
    """The mean-profile crossover and the fringed one are different numbers.

    wcc_etc's report quotes the *fringed* crossover -- the last radius where
    an Airy ring maximum still pokes above the halo, which is what a
    peak-to-peak requirement means. The azimuthally averaged profile crosses
    considerably earlier, and the two get confused.
    """
    d = decomposition
    assert d["crossover_mm"] == pytest.approx(d["crossover_px"] * 0.00376, rel=1e-6)
    assert d["crossover_fringed_mm"] > d["crossover_mm"]
    assert d["crossover_fringed_mm"] == pytest.approx(6.1, rel=0.10)


def test_crossover_moves_inward_when_the_halo_is_brighter(mag0_catalog):
    """A brighter halo overtakes the Airy wing sooner, not later."""
    from wcc_sim.psfreport import psf_decomposition

    faint = psf_decomposition(_run(mag0_catalog, scatter_fraction=5.542e-3))
    bright = psf_decomposition(_run(mag0_catalog, scatter_fraction=7.130e-3))
    assert bright["crossover_mm"] < faint["crossover_mm"]
    assert bright["crossover_fringed_mm"] < faint["crossover_fringed_mm"]
    assert bright["crossover_fringed_mm"] == pytest.approx(4.9, rel=0.10)


def test_figure_has_a_saturation_mask_panel(figure):
    fig, _ = figure
    titles = [ax.get_title() for ax in fig.axes]
    assert any("saturat" in t.lower() for t in titles), titles


def test_summary_reports_the_saturated_equivalent_radius(figure, mag0_field):
    fig, _ = figure
    text = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    n = int(mag0_field.saturation_mask.sum())
    assert "equiv" in text.lower()
    assert f"{n:,}" in text


def test_summary_names_the_background_as_zodi(figure):
    fig, _ = figure
    text = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "zodi" in text.lower()


def test_profile_table_records_the_focal_plane_scatter_fraction():
    """FRED gives two normalizations; both must be available, not just one."""
    from wcc_sim.scatter import load_scatter_table

    meta = load_scatter_table().meta
    assert meta["P_FULL"] == pytest.approx(5.542138e-3, rel=1e-5)
    assert meta["F_FOCAL"] == pytest.approx(7.130e-3, rel=1e-3)
    assert meta["THRUPUT"] == pytest.approx(0.7773, rel=1e-3)
