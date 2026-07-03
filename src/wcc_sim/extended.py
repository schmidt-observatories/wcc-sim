"""Analytic extended-source (Sersic) components for wcc-sim frames.

Components are rendered in e-/s/pix on the NATIVE detector grid and
convolved once per (template, ebv) group with an effective-PSF kernel
(FFT). The native grid is adequate because the in-focus PSF FWHM is
~3 px; the cuspy Sersic center is the exception and is re-evaluated on a
refined subgrid. Rendering on the 11x point-source grid would need ~TB of
memory for a full frame; sampling profiles as dense point grids would need
~1e6 stamp placements per component. See the design spec
(docs/superpowers/specs/2026-07-02-extended-sources-cepheids-design.md).
"""

from dataclasses import dataclass

import numpy as np
from astropy.modeling.models import Sersic2D
from scipy.signal import fftconvolve
from scipy.special import gamma, gammaincinv

from .chromatic import attenuation_factor
from .starflux import REF_MAG, rate_for_spt

_REFINE = 9         # subgrid factor for the central cusp
_REFINE_HALF_MIN = 8   # px; refinement box half-width bounds
_REFINE_HALF_MAX = 64


@dataclass(frozen=True)
class SersicComponent:
    """One elliptical Sersic component.

    Exactly one of `total_mag` (integrated Gaia-G vegamag, same
    normalization convention as point sources) or `sb_mag_arcsec2`
    (surface brightness at r_eff, mag/arcsec^2) sets the flux. `pa_deg`
    is the major-axis angle in degrees CCW from the +x detector axis.
    n=1 is an exponential disk, n=4 a de Vaucouleurs bulge.
    """

    ra: float
    dec: float
    n: float
    r_eff_arcsec: float
    ellip: float = 0.0
    pa_deg: float = 0.0
    total_mag: float = None
    sb_mag_arcsec2: float = None
    template: str = "G2V"
    ebv: float = 0.0

    def __post_init__(self):
        if (self.total_mag is None) == (self.sb_mag_arcsec2 is None):
            raise ValueError(
                "set exactly one of total_mag / sb_mag_arcsec2"
            )
        if not self.n > 0.0:
            raise ValueError(f"n must be > 0, got {self.n}")
        if not self.r_eff_arcsec > 0.0:
            raise ValueError(
                f"r_eff_arcsec must be > 0, got {self.r_eff_arcsec}"
            )
        if not 0.0 <= self.ellip < 1.0:
            raise ValueError(f"ellip must be in [0, 1), got {self.ellip}")
        if self.ebv < 0.0:
            raise ValueError(f"ebv must be >= 0, got {self.ebv}")


def sersic_total_over_amplitude(n, r_eff_pix, ellip):
    """F_total / amplitude for a Sersic2D profile (analytic).

    Sersic2D's `amplitude` is the surface brightness at r_eff per pixel
    area; integrating the profile over the plane gives
    2 pi n r_eff^2 (1-ellip) e^bn bn^(-2n) Gamma(2n).
    """
    bn = float(gammaincinv(2.0 * n, 0.5))
    return float(
        2.0 * np.pi * n * r_eff_pix**2 * (1.0 - ellip)
        * np.exp(bn) * bn ** (-2.0 * n) * gamma(2.0 * n)
    )


def component_amplitude(comp, sensorfilter, plate_scale_mas):
    """Sersic2D amplitude in e-/s/pix for the (reddened) component."""
    att = attenuation_factor(comp.template, comp.ebv, sensorfilter)
    rate_ref = rate_for_spt(comp.template, sensorfilter)
    pix_arcsec = plate_scale_mas / 1000.0
    if comp.total_mag is not None:
        total = rate_ref * 10.0 ** (-0.4 * (comp.total_mag - REF_MAG)) * att
        r_eff_pix = comp.r_eff_arcsec / pix_arcsec
        return total / sersic_total_over_amplitude(
            comp.n, r_eff_pix, comp.ellip
        )
    sb_rate = rate_ref * 10.0 ** (-0.4 * (comp.sb_mag_arcsec2 - REF_MAG)) * att
    return sb_rate * pix_arcsec**2
