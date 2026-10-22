__version__ = "0.1.0"

#: Oldest wcc_etc this package works with (public ``wcc_etc.io`` names).
WCC_ETC_MINIMUM = "0.1.0"


def require_wcc_etc(minimum=WCC_ETC_MINIMUM):
    """Raise ImportError with a readable message if wcc_etc is missing or too old."""
    try:
        import wcc_etc
    except ImportError as err:
        raise ImportError(
            f"wcc_sim {__version__} needs the wcc_etc package (>= {minimum}). "
            "Install it with: pip install wcc-etc"
        ) from err
    from packaging.version import Version

    found = getattr(wcc_etc, "__version__", "0")
    if Version(found) < Version(minimum):
        raise ImportError(
            f"wcc_sim {__version__} needs wcc_etc >= {minimum}, found {found}. "
            "Upgrade it with: pip install --upgrade wcc-etc"
        )


require_wcc_etc()

from .catalog import empty_catalog  # noqa: E402,F401
from .extended import GalSimComponent, ImageComponent, SersicComponent  # noqa: E402,F401
from .pipeline import SimulatedField, simulate_field  # noqa: E402,F401
from .detectors import detector_frame  # noqa: E402,F401
from .imagesim import simulate_image  # noqa: E402,F401
from .imagesource import SkyImageComponent  # noqa: E402,F401
from .observation import WCCObservation  # noqa: E402,F401
