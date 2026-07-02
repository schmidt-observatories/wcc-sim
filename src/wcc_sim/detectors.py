"""Detector geometry for the WCC focal plane (array dims live here, not in wcc_etc)."""

from dataclasses import dataclass

from wcc_etc import Simulation
from wcc_etc.io import _KIND_NAMES, _SENSORFILTER_FOCUS
from wcc_etc.scene import get_scene

# Full-frame pixel counts (manufacturer values; wcc_etc Sensor only stores area).
ARRAY_DIMS = {
    "zwo": (9568, 6380),   # Sony IMX455 / ZWO ASI6200MM
    "qcmos": (4096, 2304), # Hamamatsu HWK4123
}

FOCUS_WAVES = {"0wave": 0, "1wave": 1, "2wave": 2}


@dataclass(frozen=True)
class DetectorGeometry:
    nx: int
    ny: int
    pixel_size_um: float
    plate_scale_mas: float
    default_focus: int


def make_base_simulation(sensorfilter):
    """A wcc_etc Simulation for this sensorfilter (G2V mag-15 source, zodi bg).

    Used for geometry, PSF context parameters, and sky/dark rates. Raises
    ValueError listing valid keys for an unknown sensorfilter.
    """
    if sensorfilter not in _SENSORFILTER_FOCUS:
        raise ValueError(
            f"Unknown sensorfilter {sensorfilter!r}. "
            f"Valid keys: {sorted(_SENSORFILTER_FOCUS)}"
        )
    return Simulation.from_sensorfilter(sensorfilter, get_scene("G2V", mag=15.0))


def get_geometry(sensorfilter, sim=None):
    if sim is None:
        sim = make_base_simulation(sensorfilter)
    kind = sensorfilter.split(":", 1)[0]
    kind = _KIND_NAMES.get(kind, kind)
    nx, ny = ARRAY_DIMS[kind]
    plate_mas = (
        sim.sensor.get_plate_scale(sim.telescope).to("arcsec/pix").value * 1000.0
    )
    return DetectorGeometry(
        nx=nx,
        ny=ny,
        pixel_size_um=float(sim.sensor.pixel_size.value),
        plate_scale_mas=plate_mas,
        default_focus=FOCUS_WAVES[_SENSORFILTER_FOCUS[sensorfilter]],
    )
