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


def _r999_pix(n, r_eff_pix):
    """Radius enclosing 99.9% of the Sersic flux, in pixels."""
    bn = float(gammaincinv(2.0 * n, 0.5))
    return r_eff_pix * (float(gammaincinv(2.0 * n, 0.999)) / bn) ** n


def render_component_profile(comp, wcs, shape, plate_scale_mas, sensorfilter):
    """Component surface brightness in e-/s/pix on the native grid
    (unconvolved); None if the 99.9%-flux footprint misses the frame.

    The profile is evaluated at pixel centers in row chunks (full-frame
    float64 temporaries would be ~GB); a box around the center is
    re-evaluated on a _REFINE x subgrid and averaged, since a Sersic cusp
    changes across a pixel while the rest of the profile does not.
    """
    ny, nx = shape
    x0, y0 = (float(v) for v in wcs.world_to_pixel_values(comp.ra, comp.dec))
    r_eff_pix = comp.r_eff_arcsec * 1000.0 / plate_scale_mas
    dx = max(0.0, -x0, x0 - (nx - 1))
    dy = max(0.0, -y0, y0 - (ny - 1))
    if np.hypot(dx, dy) > _r999_pix(comp.n, r_eff_pix):
        return None

    amp = component_amplitude(comp, sensorfilter, plate_scale_mas)
    mod = Sersic2D(
        amplitude=amp, r_eff=r_eff_pix, n=comp.n, x_0=x0, y_0=y0,
        ellip=comp.ellip, theta=np.radians(comp.pa_deg),
    )

    img = np.empty(shape, dtype=np.float32)
    xx = np.arange(nx, dtype=float)[None, :]
    for y_lo in range(0, ny, 1024):
        y_hi = min(y_lo + 1024, ny)
        yy = np.arange(y_lo, y_hi, dtype=float)[:, None]
        img[y_lo:y_hi] = mod(xx, yy)

    half = int(np.clip(np.ceil(2.0 * r_eff_pix), _REFINE_HALF_MIN,
                       _REFINE_HALF_MAX))
    bx_lo, bx_hi = max(int(x0) - half, 0), min(int(x0) + half + 1, nx)
    by_lo, by_hi = max(int(y0) - half, 0), min(int(y0) + half + 1, ny)
    if bx_lo < bx_hi and by_lo < by_hi:
        off = (np.arange(_REFINE) + 0.5) / _REFINE - 0.5
        fx = (np.arange(bx_lo, bx_hi, dtype=float)[:, None] + off).ravel()
        fy = (np.arange(by_lo, by_hi, dtype=float)[:, None] + off).ravel()
        fine = mod(fx[None, :], fy[:, None])
        img[by_lo:by_hi, bx_lo:bx_hi] = fine.reshape(
            by_hi - by_lo, _REFINE, bx_hi - bx_lo, _REFINE
        ).mean(axis=(1, 3))
    return img


def render_extended(components, wcs, shape, plate_scale_mas, sensorfilter,
                    kernels, wing=None):
    """PSF-convolved sum of all components, e-/s/pix on the native grid.

    `kernels` maps (template, ebv) -> native-resolution PSF kernel; the
    caller decides whether those are chromatic effective PSFs or copies of
    the monochromatic one. Kernels are normalized to unit sum here and,
    when a wing model is given, scaled by 1/wing.flux_norm(half) so
    extended flux follows the same stamp-truncation convention as
    add_star. No wing halo is drawn: for smooth extended light the halo
    is a sub-noise redistribution.
    """
    image = np.zeros(shape, dtype=np.float32)
    groups = {}
    for comp in components:
        groups.setdefault((comp.template, float(comp.ebv)), []).append(comp)
    for key, comps in groups.items():
        sub = None
        for comp in comps:
            prof = render_component_profile(
                comp, wcs, shape, plate_scale_mas, sensorfilter
            )
            if prof is not None:
                sub = prof if sub is None else sub + prof
        if sub is None:
            continue
        kern = np.asarray(kernels[key], dtype=np.float64)
        kern = kern / kern.sum()
        if wing is not None:
            kern = kern / wing.flux_norm(kern.shape[0] // 2)
        image += fftconvolve(sub, kern, mode="same").astype(np.float32)
    return image
