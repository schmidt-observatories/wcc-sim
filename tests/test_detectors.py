import pytest


def test_geometry_imx():
    from wcc_sim.detectors import get_geometry

    g = get_geometry("zwo:r")
    assert (g.nx, g.ny) == (9568, 6380)
    assert g.pixel_size_um == pytest.approx(3.76)
    assert g.plate_scale_mas == pytest.approx(16.87, abs=0.02)
    assert g.default_focus == 0


def test_geometry_qcmos():
    from wcc_sim.detectors import get_geometry

    g = get_geometry("qcmos:bb")
    assert (g.nx, g.ny) == (4096, 2304)
    assert g.plate_scale_mas == pytest.approx(20.64, abs=0.02)


def test_geometry_defocus_default():
    from wcc_sim.detectors import get_geometry

    assert get_geometry("zwo:r+1").default_focus == 1
    assert get_geometry("zwo:r-1").default_focus in (1, 2)  # per sensor_info


def test_unknown_sensorfilter_lists_valid_keys():
    from wcc_sim.detectors import get_geometry

    with pytest.raises(ValueError, match="zwo:r"):
        get_geometry("nope:x")


def test_unknown_sensorfilter_with_explicit_sim():
    from wcc_sim.detectors import get_geometry, make_base_simulation

    sim = make_base_simulation("zwo:r")
    with pytest.raises(ValueError, match="zwo:r"):
        get_geometry("nope:x", sim=sim)


def test_qcmos_saturation_ceiling_follows_the_gain_mode():
    """#17: the 12-bit ADC x 0.1348 e-/ADU clips at 552 e- in the 32x high
    gain mode, 15x below the 7500 e- well; the 1x low gain mode lets the
    well clip first."""
    from wcc_sim.detectors import make_base_simulation, saturation_level_e

    high = saturation_level_e(make_base_simulation("qcmos:bb", gain_mode="high").sensor)
    low = saturation_level_e(make_base_simulation("qcmos:bb", gain_mode="low").sensor)
    assert high == pytest.approx(4095 * 0.1348, rel=1e-3)
    assert low == pytest.approx(7500.0)
    assert saturation_level_e(make_base_simulation("zwo:r").sensor) > 1e4


def test_unknown_gain_mode_is_rejected():
    from wcc_sim.detectors import make_base_simulation

    with pytest.raises(ValueError, match="gain_mode"):
        make_base_simulation("qcmos:bb", gain_mode="medium")


def test_require_wcc_etc_rejects_old_version(monkeypatch):
    import wcc_etc
    import wcc_sim

    wcc_sim.require_wcc_etc()  # the installed one passes
    monkeypatch.setattr(wcc_etc, "__version__", "0.0.9")
    with pytest.raises(ImportError, match="needs wcc_etc >= 0.1.0, found 0.0.9"):
        wcc_sim.require_wcc_etc()


# -- detector_frame ----------------------------------------------------------- #

def test_detector_frame_zero_offset_and_full_shape():
    from wcc_sim.detectors import detector_frame, get_geometry

    det = detector_frame("zwo:r", (150.1, 2.2))
    geom = get_geometry("zwo:r")
    assert det.shape == (geom.ny, geom.nx)
    assert det.pointing == pytest.approx((150.1, 2.2))
    assert det.plate_scale_mas == pytest.approx(geom.plate_scale_mas)


def test_detector_frame_offset_east_moves_target_west():
    from wcc_sim.detectors import detector_frame

    det = detector_frame("zwo:r", (150.1, 2.2), offset_arcsec=(1.0, 0.0), shape=(201, 201))
    x, y = det.wcs.world_to_pixel_values(150.1, 2.2)
    # detector centred 1" East of the target -> target 1" West = +x (East is left)
    assert x - 100 == pytest.approx(1000.0 / det.plate_scale_mas, rel=1e-4)
    assert y - 100 == pytest.approx(0.0, abs=1e-3)


def test_detector_frame_pa_and_cache():
    from wcc_sim import detectors
    from wcc_sim.wcsutil import build_wcs

    det = detectors.detector_frame("zwo:r", (150.1, 2.2), pa=90.0, shape=(11, 11))
    ref = build_wcs(150.1, 2.2, det.plate_scale_mas, 90.0, (11, 11))
    assert det.wcs.wcs.cd.ravel().tolist() == pytest.approx(ref.wcs.cd.ravel().tolist())
    before = detectors._cached_geometry.cache_info().hits
    detectors.detector_frame("zwo:r", (150.1, 2.2), shape=(11, 11))
    assert detectors._cached_geometry.cache_info().hits == before + 1


def test_detector_frame_records_centre_and_offset():
    from wcc_sim.detectors import detector_frame

    f = detector_frame("zwo:r", (10.0, 20.0), offset_arcsec=(1.0, 2.0), shape=(5, 5))
    assert f.centre == (10.0, 20.0) and f.offset_arcsec == (1.0, 2.0)
    assert f.pointing[1] == pytest.approx(20.0 + 2.0 / 3600, abs=1e-6)


def test_detector_frame_zero_offset_is_exactly_build_wcs():
    from wcc_sim.detectors import detector_frame, get_geometry
    from wcc_sim.wcsutil import build_wcs

    ra, dec = 212.04231062531, -41.3979370253
    f = detector_frame("zwo:r", (ra, dec), pa=33.0, shape=(64, 48))
    ref = build_wcs(ra, dec, get_geometry("zwo:r").plate_scale_mas, 33.0, (64, 48))
    assert f.pointing == (ra, dec)
    assert f.wcs.to_header() == ref.to_header()
