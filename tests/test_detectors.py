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
