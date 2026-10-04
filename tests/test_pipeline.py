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
    kw.setdefault("sensorfilter", "zwo:r")
    return simulate_field(RA0, DEC0, **kw)


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


def _injected_star_e(field):
    """Total star electrons in a frame, over the uniform sky+dark level."""
    return (field.image_clean - np.median(field.image_clean)).sum()


def test_photometric_closure_vs_etc(canned_catalog):
    """Total injected flux of an interior star matches the ETC rate to <1%.

    Run without the scattered-light halo: this pins the ETC -> simulation
    flux calibration, which should not be entangled with stray light. The
    scatter-on case is covered by the companion test below.
    """
    from wcc_sim.starflux import rate_for_spt

    single = canned_catalog[[1]]  # G=15 star, 1.5" from center (interior)
    # 65 px stamp: the 33 px speed default leaves ~1% of a red star's flux
    # outside the wing-model normalisation (0.65% remains even at 129 px;
    # that residual is the wing fit's, see #13/#14), not a rate error.
    f = run(single, add_noise=False, exptime=90.0, scatter=False, stamp_npix=65)
    spt = f.catalog["spt"][0]
    expected = rate_for_spt(spt, "zwo:r") * 90.0
    assert expected > 0  # guard: a zero rate would make the closure check vacuous
    assert _injected_star_e(f) == pytest.approx(expected, rel=0.01)


def test_scatter_removes_the_scattered_fraction_from_the_star(canned_catalog):
    """With scatter on, a faint star keeps (1 - f_scat) of its flux.

    The scattered light is real and leaves the core, but for a faint star the
    halo sits below the drawing floor everywhere, so it is not deposited --
    the frame legitimately holds ~0.55% less than the ETC total. Folding it
    back into the core would make a star's brightness depend on the noise
    floor.
    """
    from wcc_sim.scatter import halo_for_sensorfilter

    single = canned_catalog[[1]]
    on = run(single, add_noise=False, exptime=90.0)
    off = run(single, add_noise=False, exptime=90.0, scatter=False)
    f_scat = halo_for_sensorfilter("zwo:r").frac_total

    deficit = 1.0 - _injected_star_e(on) / _injected_star_e(off)
    assert deficit == pytest.approx(f_scat, abs=0.001)


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


def test_extended_source_adds_flux(canned_catalog):
    from wcc_sim import SersicComponent
    from wcc_sim.extended import component_amplitude, sersic_total_over_amplitude

    comp = SersicComponent(ra=RA0, dec=DEC0, n=1.0, r_eff_arcsec=0.05,
                           total_mag=16.0)
    single = canned_catalog[[3]]  # one faint star only
    base = run(single, add_noise=False)
    ext = run(single, add_noise=False, extended_sources=[comp])
    extra = float(ext.image_clean.sum() - base.image_clean.sum())
    plate = base.params["plate_scale_mas"]
    amp = component_amplitude(comp, "zwo:r", plate)
    r_eff_pix = comp.r_eff_arcsec * 1000.0 / plate
    expected = (
        amp * sersic_total_over_amplitude(comp.n, r_eff_pix, comp.ellip)
        * base.params["exptime"]
    )
    # wing renormalization holds back the stamp-truncated fraction: for
    # stamp_npix=33/oversample=11 zwo:r in-focus PSF, energy_beyond(half=16)
    # is ~2.4% (verified via wcc_sim.wings.WingModel.energy_beyond using the
    # fitted params in base.params), so rel=0.02 is slightly too tight here.
    assert extra == pytest.approx(expected, rel=0.03)
    assert ext.params["n_extended"] == 1
    assert base.params["n_extended"] == 0


def test_extended_in_fits_header(canned_catalog, tmp_path):
    from wcc_sim import SersicComponent

    comp = SersicComponent(ra=RA0, dec=DEC0, n=1.0, r_eff_arcsec=0.05,
                           total_mag=16.0)
    path = tmp_path / "ext.fits"
    run(canned_catalog, extended_sources=[comp], output=str(path))
    from astropy.io import fits

    header = fits.getheader(path, "SCI")
    assert header["NEXTSRC"] == 1
    assert header["CHROMPSF"] is False


def test_chromatic_changes_in_focus_image(canned_catalog):
    mono = run(canned_catalog, add_noise=False)
    chrom = run(canned_catalog, add_noise=False, chromatic=True)
    assert chrom.params["chromatic"] is True
    assert not np.array_equal(mono.image_clean, chrom.image_clean)
    # rates are untouched: total flux agrees to the wing-truncation level
    assert chrom.image_clean.sum() == pytest.approx(
        mono.image_clean.sum(), rel=0.01
    )


def test_chromatic_defocus_is_passthrough(canned_catalog):
    mono = run(canned_catalog, add_noise=False, focus=1, stamp_npix=65)
    chrom = run(canned_catalog, add_noise=False, focus=1, stamp_npix=65,
                chromatic=True)
    assert np.array_equal(mono.image_clean, chrom.image_clean)


def test_in_focus_stars_use_their_templates_effective_wavelength(canned_catalog):
    """Issue #20: an M dwarf's Airy core is built redward of the pivot."""
    from wcc_sim.chromatic import effective_wavelength_nm_for_spt

    f = run(canned_catalog, sensorfilter="zwo:bb", add_noise=False)
    pivot = f.params["wavelength_nm"]
    for spt, lam in zip(f.catalog["spt"], f.catalog["psf_wavelength_nm"]):
        assert lam == pytest.approx(effective_wavelength_nm_for_spt("zwo:bb", spt), abs=1e-6)
    red = f.catalog["psf_wavelength_nm"][np.char.startswith(f.catalog["spt"].astype(str), "M")]
    assert len(red) and (red - pivot > 50.0).all()


def test_wavelength_override_pins_every_star(canned_catalog):
    f = run(canned_catalog, add_noise=False, wavelength_nm=450.0)
    assert f.params["wavelength_nm"] == 450.0
    assert (f.catalog["psf_wavelength_nm"] == 450.0).all()


def test_m_dwarf_peak_pixel_is_lower_than_a_pivot_render(canned_catalog):
    """The physics behind #20: ~20% lower peak for M5V in the broad band."""
    spts = run(canned_catalog).catalog["spt"].astype(str)
    red = canned_catalog[[int(np.flatnonzero(np.char.startswith(spts, "M"))[0])]]
    eff = run(red, sensorfilter="zwo:bb", add_noise=False, wings=False)
    piv = run(red, sensorfilter="zwo:bb", add_noise=False, wings=False,
              wavelength_nm=eff.params["wavelength_nm"])
    assert eff.image_clean.max() < 0.9 * piv.image_clean.max()
