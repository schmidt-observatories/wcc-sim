import numpy as np
import pytest

# A synthetic halo with an analytic answer: p(r) = c * r^alpha, tabulated
# log-spaced the way the FRED builder tabulates the real one.
C_SYN, ALPHA_SYN = 1e-4, -2.5
R_MIN_SYN, R_MAX_SYN = 10.0, 40000.0


def _analytic_energy_between(r0, r1):
    """2*pi * int_r0^r1 r * c r^alpha dr, for the synthetic power law."""
    k = ALPHA_SYN + 2.0
    return 2.0 * np.pi * C_SYN * (r1**k - r0**k) / k


def _synthetic_halo():
    from wcc_sim.scatter import ScatterHalo

    r = np.geomspace(R_MIN_SYN, R_MAX_SYN, 300)
    p = C_SYN * r**ALPHA_SYN
    # the halo is truncated at R_MAX_SYN, so its energy is the finite integral
    total = _analytic_energy_between(R_MIN_SYN, R_MAX_SYN)
    return ScatterHalo(r_px=r, frac_per_px=p, frac_total=total)


def test_profile_reproduces_tabulated_power_law():
    halo = _synthetic_halo()
    for r in [50.0, 500.0, 5000.0, 30000.0]:
        assert halo.profile(r) == pytest.approx(C_SYN * r**ALPHA_SYN, rel=0.01)


def test_profile_accepts_arrays_and_preserves_shape():
    halo = _synthetic_halo()
    r = np.array([[100.0, 200.0], [400.0, 800.0]])
    out = halo.profile(r)
    assert out.shape == r.shape
    assert np.all(np.diff(out.ravel()) < 0.0)


def test_profile_is_flat_below_the_tabulated_inner_radius():
    # the FRED map has 0.4 mm cells (~106 det px) so it cannot resolve the
    # halo near the core; the innermost tabulated value must be held, not
    # extrapolated to a divergence.
    halo = _synthetic_halo()
    inner = halo.profile(R_MIN_SYN)
    assert halo.profile(0.0) == pytest.approx(inner, rel=1e-6)
    assert halo.profile(R_MIN_SYN / 4.0) == pytest.approx(inner, rel=1e-6)


def test_energy_beyond_matches_analytic_integral():
    halo = _synthetic_halo()
    for r in [100.0, 1000.0, 10000.0]:
        expect = _analytic_energy_between(r, R_MAX_SYN)
        assert halo.energy_beyond(r) == pytest.approx(expect, rel=0.02)


def test_profile_is_zero_beyond_the_modelled_edge():
    # the FRED map ends; extrapolating a power law past it would invent flux
    halo = _synthetic_halo()
    assert halo.profile(R_MAX_SYN * 1.01) == 0.0
    assert halo.energy_beyond(R_MAX_SYN) == 0.0


def test_r_out_clamps_to_the_modelled_edge():
    halo = _synthetic_halo()
    assert halo.r_out(flux_e=1e30, floor_e=1.0) == pytest.approx(R_MAX_SYN)


def test_energy_beyond_inner_radius_recovers_frac_total():
    halo = _synthetic_halo()
    assert halo.energy_beyond(R_MIN_SYN) == pytest.approx(halo.frac_total, rel=0.02)


def test_r_out_inverts_the_profile():
    halo = _synthetic_halo()
    r = halo.r_out(flux_e=1e9, floor_e=1.0)
    assert halo.profile(r) * 1e9 == pytest.approx(1.0, rel=0.05)


def test_r_out_grows_with_flux():
    halo = _synthetic_halo()
    assert halo.r_out(1e10, 1.0) > halo.r_out(1e8, 1.0)


def test_r_out_is_zero_when_halo_is_everywhere_below_the_floor():
    halo = _synthetic_halo()
    assert halo.r_out(flux_e=1.0, floor_e=1e6) == 0.0


def test_scaled_changes_the_total_scatter_fraction_not_the_shape():
    halo = _synthetic_halo()
    twice = halo.scaled(2.0 * halo.frac_total)
    assert twice.frac_total == pytest.approx(2.0 * halo.frac_total)
    assert twice.profile(1000.0) == pytest.approx(2.0 * halo.profile(1000.0), rel=1e-6)


# --------------------------------------------------------------------------- #
# Building the real halo from the FRED .fgd that wcc_etc ships                 #
# --------------------------------------------------------------------------- #

P_FULL_FRED = 5.542137968166e-3   # FRED "integrated power" for a 1 W source
IRR_PEAK_FRED = 1.65013733315822e-05  # peak of the map [W/mm^2 per W]
MM_PER_PIX_IMX = 0.00376


@pytest.fixture(scope="module")
def table():
    from wcc_sim.scatter import build_scatter_table

    try:
        return build_scatter_table()
    except FileNotFoundError:
        pytest.skip("FRED .fgd map not available (only a local wcc_etc checkout ships it)")


@pytest.fixture(scope="module")
def halo(table):
    from wcc_sim.scatter import scatter_halo

    return scatter_halo(MM_PER_PIX_IMX, table=table)


def test_builder_recovers_fred_integrated_power(table):
    assert table.meta["P_FULL"] == pytest.approx(P_FULL_FRED, rel=1e-6)


def test_builder_records_the_map_peak_irradiance(table):
    assert table.meta["IRRPEAK"] == pytest.approx(IRR_PEAK_FRED, rel=1e-6)


def test_airy_to_scatter_peak_contrast_is_about_1e9(table):
    """The acceptance number: Airy peak / scatter peak ~ 1e9 at 450 nm."""
    from wcc_etc.scatter_psf import airy_irradiance

    airy_peak = float(
        airy_irradiance(np.array([1e-9]), 3.065, 15.0, 450e-9, power=1.0)[0]
    )
    contrast = airy_peak / table.meta["IRRPEAK"]
    assert contrast == pytest.approx(1.045e9, rel=0.02)


def test_table_profile_is_monotone_decreasing(table):
    assert np.all(np.diff(np.asarray(table["frac_per_mm2"])) < 0.0)


def test_halo_carries_the_scattered_fraction_inside_its_modelled_disc(halo, table):
    # absolute FRED brightness is preserved, so the deposited energy is
    # frac_total * FDISC; the rest lands off the modelled disc (and off any
    # detector) and is deliberately not deposited
    assert halo.frac_total == pytest.approx(P_FULL_FRED, rel=1e-9)
    assert halo.energy_beyond(0.0) == pytest.approx(
        P_FULL_FRED * table.meta["FDISC"], rel=2e-3
    )


def test_profile_integral_matches_the_exact_2d_map_integral(table):
    """The real validation: azimuthal average, units and integration at once.

    The 1D profile's own integral over the disc must reproduce the map's exact
    2D integral over the same disc. Agreement here means the halo loses
    nothing measurable by being stored as a radial profile.
    """
    assert table.meta["FDISCPRF"] == pytest.approx(table.meta["FDISC"], rel=1e-3)


def test_most_of_the_scattered_power_is_inside_the_modelled_disc(table):
    # the remainder sits in the FRED map's long-axis corners, 37,000+ px out
    assert 0.85 < table.meta["FDISC"] < 0.90


def test_profile_scales_with_pixel_area_at_fixed_physical_radius(table):
    from wcc_sim.scatter import scatter_halo

    fine = scatter_halo(MM_PER_PIX_IMX, table=table)
    coarse = scatter_halo(2.0 * MM_PER_PIX_IMX, table=table)
    r_mm = 5.0
    assert coarse.profile(r_mm / (2.0 * MM_PER_PIX_IMX)) == pytest.approx(
        4.0 * fine.profile(r_mm / MM_PER_PIX_IMX), rel=0.02
    )


def test_halo_reaches_past_the_chip_diagonal(halo):
    # IMX455 is 9568 x 6380 px -> 11500 px diagonal; a star in one corner
    # must still deposit halo in the opposite corner, with no truncation
    assert halo.r_px[-1] > 11500.0


def test_halo_dominates_the_diffraction_wing_at_the_chip_half_diagonal(halo):
    """The reason this exists: the r^-3 wing badly under-predicts far flux."""
    from wcc_sim.detectors import make_base_simulation
    from wcc_sim.psf import render_oversampled_psf
    from wcc_sim.render import bin_oversampled
    from wcc_sim.wings import fit_wing_model

    sim = make_base_simulation("zwo:r")
    wing = fit_wing_model(
        bin_oversampled(render_oversampled_psf(sim, 0, oversample=11), 11)
    )
    r = 5750.0  # chip half-diagonal
    assert halo.profile(r) / wing.profile(r) > 5.0


def test_scatter_fraction_can_be_overridden(table):
    from wcc_sim.scatter import scatter_halo

    halo = scatter_halo(MM_PER_PIX_IMX, frac_total=1e-2, table=table)
    assert halo.energy_beyond(0.0) == pytest.approx(
        1e-2 * table.meta["FDISC"], rel=2e-3
    )


# --------------------------------------------------------------------------- #
# The packaged profile and the sensor convenience                              #
# --------------------------------------------------------------------------- #

def test_packaged_profile_reproduces_the_builder(table):
    from wcc_sim.scatter import load_scatter_table

    packaged = load_scatter_table()
    assert np.allclose(packaged["r_mm"], table["r_mm"], rtol=1e-12)
    assert np.allclose(packaged["frac_per_mm2"], table["frac_per_mm2"], rtol=1e-12)
    assert packaged.meta["SCATFILE"] == table.meta["SCATFILE"]


def test_halo_for_sensorfilter_uses_the_detector_pixel_pitch():
    from wcc_sim.scatter import halo_for_sensorfilter, load_scatter_table

    halo = halo_for_sensorfilter("zwo:r")
    reach_mm = float(load_scatter_table()["r_mm"][-1])
    assert halo.r_px[-1] == pytest.approx(reach_mm / MM_PER_PIX_IMX, rel=1e-6)


def test_halo_for_sensorfilter_differs_between_detectors():
    from wcc_sim.scatter import halo_for_sensorfilter

    imx = halo_for_sensorfilter("zwo:r")
    hwk = halo_for_sensorfilter("qcmos:bb")
    assert imx.r_px[-1] != pytest.approx(hwk.r_px[-1], rel=1e-3)


# --------------------------------------------------------------------------- #
# Combining the halo with the core's diffraction wing                          #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def core_wing():
    from wcc_sim.detectors import make_base_simulation
    from wcc_sim.psf import render_oversampled_psf
    from wcc_sim.render import bin_oversampled
    from wcc_sim.wings import fit_wing_model

    sim = make_base_simulation("zwo:r")
    return fit_wing_model(
        bin_oversampled(render_oversampled_psf(sim, 0, oversample=11), 11)
    )


@pytest.fixture(scope="module")
def combined(core_wing, halo):
    from wcc_sim.wings import CombinedWing

    return CombinedWing(core=core_wing, halo=halo)


def test_combined_profile_is_the_scaled_core_plus_the_halo(combined, core_wing, halo):
    r = 800.0
    scale = 1.0 - halo.frac_total
    assert combined.profile(r) == pytest.approx(
        scale * core_wing.profile(r) + halo.profile(r), rel=1e-9
    )


def test_combined_energy_beyond_is_the_sum_of_the_terms(combined, core_wing, halo):
    r = 64.0
    scale = 1.0 - halo.frac_total
    assert combined.energy_beyond(r) == pytest.approx(
        scale * core_wing.energy_beyond(r) + halo.energy_beyond(r), rel=1e-9
    )


def test_combined_r_out_reaches_further_than_the_diffraction_wing_alone(
    combined, core_wing
):
    # the halo is shallower than r^-3, so the brighter the star the further
    # ahead of the diffraction-only radius the halo pushes the floor crossing
    assert combined.r_out(1e10, 1.0) > 1.15 * core_wing.r_out(1e10, 1.0)
    assert combined.r_out(1e12, 1.0) > 3.0 * core_wing.r_out(1e12, 1.0)


def test_faint_star_halo_radius_is_unchanged_by_scatter(combined, core_wing):
    # at 200 px the diffraction wing still dominates, so faint stars are
    # unaffected and cost nothing extra to draw
    assert combined.r_out(1e8, 1.0) == pytest.approx(
        core_wing.r_out(1e8, 1.0), rel=0.05
    )


def test_combined_r_out_is_clamped_to_the_modelled_halo_edge(combined, halo):
    assert combined.r_out(1e16, 1.0) == pytest.approx(halo.r_px[-1], rel=1e-9)


def test_combined_r_out_lands_where_the_profile_meets_the_floor(combined):
    flux, floor = 1e10, 1.0
    r = combined.r_out(flux, floor)
    assert flux * combined.profile(r) == pytest.approx(floor, rel=0.10)


def test_wing_model_flux_norm_is_the_existing_renormalization(core_wing):
    # unchanged behaviour for the no-scatter path
    assert core_wing.flux_norm(64.0) == pytest.approx(
        1.0 + core_wing.energy_beyond(64.0), rel=1e-12
    )


def test_combined_flux_norm_conserves_flux_minus_the_light_that_leaves_the_disc(
    combined, halo
):
    """Off-disc scatter is lost, not redistributed into the drawn PSF."""
    lost = halo.frac_total - halo.energy_beyond(0.0)
    scale = 1.0 - halo.frac_total
    drawn = scale * (1.0 + combined.core.energy_beyond(64.0)) + halo.energy_beyond(0.0)
    assert combined.flux_norm(64.0) == pytest.approx(drawn / (1.0 - lost), rel=1e-9)


def test_combined_profile_is_halo_dominated_far_out(combined, halo):
    r = 5750.0  # chip half-diagonal
    assert halo.profile(r) / combined.profile(r) > 0.8


# --------------------------------------------------------------------------- #
# Rendering: the halo must land inside the stamp footprint too                 #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def psf_os():
    from wcc_sim.detectors import make_base_simulation
    from wcc_sim.psf import render_oversampled_psf

    sim = make_base_simulation("zwo:r")
    return render_oversampled_psf(sim, focus=0, oversample=11, stamp_npix=33)


def test_wing_blend_tabulates_the_halo_on_the_stamp_grid(combined, halo):
    from wcc_sim.render import wing_blend

    blend = wing_blend(33, combined)
    assert blend.core_scale == pytest.approx(1.0 - halo.frac_total)
    # halo sampled at the pixel 10 px to the right of the stamp centre
    assert blend.halo[16, 26] == pytest.approx(halo.profile(10.0), rel=1e-9)
    assert blend.halo[16, 16] == pytest.approx(halo.profile(0.0), rel=1e-9)


def test_wing_blend_without_scatter_is_unchanged(core_wing):
    from wcc_sim.render import wing_blend

    blend = wing_blend(33, core_wing)
    assert blend.core_scale == 1.0
    assert not blend.halo.any()  # nothing added inside the stamp


def test_combined_wing_reduces_to_the_diffraction_path_as_scatter_vanishes(
    psf_os, core_wing, halo
):
    """Isolates the scatter plumbing from the pre-existing crossfade.

    With the scattered fraction driven to zero the combined path must
    reproduce the diffraction-only rendering pixel for pixel, which is what
    proves the core scaling, the on-stamp halo term and `flux_norm` are
    consistent with each other.
    """
    from wcc_sim.render import add_star
    from wcc_sim.wings import CombinedWing

    def render(wing):
        img = np.zeros((1024, 1024), dtype=np.float32)
        add_star(img, psf_os, x=512.0, y=512.0, flux_e=2e8, oversample=11,
                 wing=wing, floor_e=1.0)
        return img

    plain = render(core_wing)
    vanishing = render(CombinedWing(core=core_wing, halo=halo.scaled(1e-12)))
    assert vanishing.sum() / plain.sum() == pytest.approx(1.0, rel=1e-5)
    assert np.abs(vanishing - plain).max() < 1e-6 * plain.max()


def test_add_star_flux_deficit_with_scatter_is_the_undrawn_sub_floor_halo(
    psf_os, combined, core_wing, halo
):
    """Flux is conserved up to halo the floor says not to draw.

    Two deficits are expected and both are accounted for: the pre-existing
    crossfade/square-footprint approximation (~0.5% here, present without
    scatter too) and the halo beyond r_out, which is deliberately not drawn
    because it sits under the noise floor.
    """
    from wcc_sim.render import add_star

    flux, floor = 2e8, 1.0

    def render(wing):
        img = np.zeros((1024, 1024), dtype=np.float32)
        add_star(img, psf_os, x=512.0, y=512.0, flux_e=flux, oversample=11,
                 wing=wing, floor_e=floor)
        return img.sum() / flux

    with_scatter = render(combined)
    baseline_deficit = 1.0 - render(core_wing)
    undrawn = halo.energy_beyond(combined.r_out(flux / combined.flux_norm(16.0), floor))

    assert with_scatter == pytest.approx(1.0, abs=0.02)
    assert 0.0 < 1.0 - with_scatter < baseline_deficit + undrawn + 0.002


def test_scatter_deposits_far_more_flux_than_diffraction_alone(
    psf_os, combined, core_wing
):
    """The point of the exercise: contamination 1500 px from a bright star."""
    from wcc_sim.render import add_star

    flux, r = 1e12, 1500
    with_scatter = np.zeros((3201, 3201), dtype=np.float32)
    add_star(with_scatter, psf_os, x=1600.0, y=1600.0, flux_e=flux,
             oversample=11, wing=combined, floor_e=1e-3)
    diffraction_only = np.zeros((3201, 3201), dtype=np.float32)
    add_star(diffraction_only, psf_os, x=1600.0, y=1600.0, flux_e=flux,
             oversample=11, wing=core_wing, floor_e=1e-3)
    ratio = with_scatter[1600, 1600 + r] / diffraction_only[1600, 1600 + r]
    assert ratio > 2.0


def test_halo_stays_azimuthally_smooth_far_out(psf_os, combined):
    from wcc_sim.render import add_star

    img = np.zeros((1024, 1024), dtype=np.float32)
    add_star(img, psf_os, x=512.0, y=512.0, flux_e=1e10, oversample=11,
             wing=combined, floor_e=1e-3)
    r = 400
    along_x = img[512, 512 + r]
    along_y = img[512 + r, 512]
    diagonal = img[512 + int(r / np.sqrt(2)), 512 + int(r / np.sqrt(2))]
    assert along_y == pytest.approx(along_x, rel=0.05)
    assert diagonal == pytest.approx(along_x, rel=0.05)


# --------------------------------------------------------------------------- #
# Pipeline and CLI wiring                                                      #
# --------------------------------------------------------------------------- #

RA0, DEC0 = 150.1, 2.2


def _field(catalog, **kw):
    from wcc_sim.pipeline import simulate_field

    opts = dict(
        sensorfilter="zwo:r", focus=0, exptime=90.0, catalog=catalog,
        shape=(256, 256), stamp_npix=33, add_noise=False, seed=0,
    )
    opts.update(kw)
    return simulate_field(RA0, DEC0, **opts)


def test_pipeline_scatter_is_on_by_default(canned_catalog):
    field = _field(canned_catalog)
    assert field.params["scatter"] is True
    assert field.params["scatter_fraction"] == pytest.approx(P_FULL_FRED)
    assert "26-0212" in field.params["scatter_file"]


@pytest.fixture(scope="module")
def bright_star_catalog():
    from astropy.table import Table

    return Table({
        "source_id": np.array([0], dtype=np.int64),
        "ra": [RA0], "dec": [DEC0],
        "phot_g_mean_mag": [8.0],
        "phot_bp_mean_mag": [8.35],
        "phot_rp_mean_mag": [7.65],
    })


C_BRIGHT = 900  # centre of the bright-star frame


@pytest.fixture(scope="module")
def bright_pair(bright_star_catalog):
    """One G=8 star rendered with and without the scatter halo."""
    shape = (2 * C_BRIGHT + 1, 2 * C_BRIGHT + 1)
    on = _field(bright_star_catalog, shape=shape)
    off = _field(bright_star_catalog, shape=shape, scatter=False)
    return on, off


def _cut(field, r):
    """Background-subtracted pixel r px to the right of the star."""
    return float(
        field.image_clean[C_BRIGHT, C_BRIGHT + r] - field.image_clean[3, 3]
    )


def test_pipeline_scatter_raises_the_far_wing_of_a_bright_star(bright_pair):
    on, off = bright_pair
    assert _cut(on, 500) / _cut(off, 500) > 1.3


def test_pipeline_scatter_slightly_dims_the_near_wing(bright_pair):
    """Energy moved into the halo leaves the core: (1 - f_scat) * core.

    The halo is spread over ~37,000 px, so inside the diffraction-dominated
    region turning scatter on makes the star marginally fainter, not brighter.
    """
    on, off = bright_pair
    assert _cut(on, 60) < _cut(off, 60)


def test_scatter_extends_the_drawn_reach_of_a_bright_star(bright_pair):
    """Measured: r_out grows from 597 px to 770 px for this G=8 star."""
    on, off = bright_pair
    assert _cut(off, 650) == 0.0        # diffraction wing already under floor
    assert _cut(on, 650) > 0.0          # halo is still above it
    assert _cut(on, 700) > 0.0
    assert _cut(on, 800) == 0.0         # past the combined r_out


def test_drawn_halo_reach_is_limited_by_the_wing_floor(bright_pair):
    """The halo is modelled to 37,000 px but only drawn above 0.1 sigma.

    For a G=8 star at 90 s the floor bites at ~770 px, far inside the model's
    reach -- so what limits the halo in an image is `wing_floor_e`, not the
    FRED map.
    """
    on, _ = bright_pair
    assert on.params["scatter_reach_px"] > 36000.0
    assert _cut(on, 800) == 0.0


def test_pipeline_scatter_off_matches_the_pre_scatter_rendering(canned_catalog):
    off = _field(canned_catalog, scatter=False)
    assert off.params["scatter_fraction"] is None
    assert off.params["wing_alpha"] is not None  # diffraction wing still fitted


def test_pipeline_scatter_fraction_can_be_overridden(canned_catalog):
    field = _field(canned_catalog, scatter_fraction=2e-2)
    assert field.params["scatter_fraction"] == pytest.approx(2e-2)


def test_pipeline_no_wings_disables_scatter(canned_catalog):
    # the halo is drawn by the wing machinery, so turning wings off turns
    # scatter off too -- and params must say so rather than claiming it ran
    field = _field(canned_catalog, wings=False)
    assert field.params["scatter"] is False


def test_pipeline_records_scatter_in_the_fits_header(canned_catalog, tmp_path):
    from astropy.io import fits

    out = tmp_path / "scatter.fits"
    _field(canned_catalog, output=str(out))
    hdr = fits.getheader(out, 0)
    assert hdr["SCATTER"] is True
    assert hdr["SCATFRAC"] == pytest.approx(P_FULL_FRED)


def test_cli_no_scatter_flag(canned_catalog, tmp_path, monkeypatch):
    from astropy.io import fits

    import wcc_sim.cli as cli

    captured = {}
    real = cli.simulate_field

    def spy(*a, **kw):
        captured.update(kw)
        kw["catalog"] = canned_catalog
        kw["shape"] = (256, 256)
        return real(*a, **kw)

    monkeypatch.setattr(cli, "simulate_field", spy)
    out = tmp_path / "cli.fits"
    assert cli.main([
        "--ra", str(RA0), "--dec", str(DEC0), "--no-scatter",
        "--stamp-npix", "33", "-o", str(out),
    ]) == 0
    assert captured["scatter"] is False
    assert fits.getheader(out, 0)["SCATTER"] is False


def test_extended_kernel_uses_flux_norm(canned_catalog):
    """Extended components must follow the same normalization as add_star."""
    from wcc_sim.extended import SersicComponent

    comp = SersicComponent(
        ra=RA0, dec=DEC0, n=1.0, r_eff_arcsec=0.5, total_mag=15.0,
    )
    on = _field(canned_catalog, extended_sources=[comp])
    off = _field(canned_catalog, extended_sources=[comp], scatter=False)
    # the scatter term scales the core down, so the convolved extended flux
    # must differ between the two -- it must not silently ignore the halo
    assert on.image_clean.sum() != pytest.approx(off.image_clean.sum(), rel=1e-9)


# --------------------------------------------------------------------------- #
# Cross-validation against wcc_etc's own 2D resampler                          #
# --------------------------------------------------------------------------- #

def test_profile_matches_wcc_etc_make_total_psf(halo):
    """The 1D profile must agree with the 2D map it replaces.

    ``wcc_etc.scatter_psf.make_total_psf`` resamples the FRED map onto the
    detector grid by flux-conserving 2D cubic interpolation -- a completely
    different code path from the azimuthal averaging here, and the one that
    produced the reference combined-PSF FITS. Absolute surface brightness,
    not just shape, has to match: this is what pins the halo's normalization.
    """
    from wcc_etc.scatter_psf import make_total_psf

    res = make_total_psf(
        sensor="imx455", extent=1001, wavelength=450e-9, jitter_sigma_mas=0.0,
        keep_components=True, renormalize=False, verbose=False, inner_npix=51,
    )
    scatter = np.asarray(res.scatter, dtype=float) * res.desired_power
    n = scatter.shape[0]
    c = n // 2
    yy, xx = np.mgrid[:n, :n]
    r_int = np.hypot(yy - c, xx - c).astype(np.int32).ravel()
    count = np.bincount(r_int, minlength=c + 2)
    total = np.bincount(r_int, weights=scatter.ravel(), minlength=c + 2)
    reference = np.where(count > 0, total / np.maximum(count, 1), 0.0)

    for r in [100, 150, 200, 300, 400, 500]:
        assert halo.profile(float(r)) == pytest.approx(reference[r], rel=0.01)
