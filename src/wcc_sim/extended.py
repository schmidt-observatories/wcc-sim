"""Extended-source components for wcc-sim frames: analytic Sersic profiles,
arbitrary GalSim profiles (GalSimComponent) and calibrated sky images with a
WCS such as HST mosaics (ImageComponent). The last two need the optional
`galaxies` extra (galsim, reproject).

Components are rendered in e-/s/pix on the NATIVE detector grid and
convolved once per (template, ebv) group with an effective-PSF kernel
(FFT). The native grid is adequate because the in-focus PSF FWHM is
~3 px; the cuspy Sersic center is the exception and is re-evaluated on a
refined subgrid. Rendering on the 11x point-source grid would need ~TB of
memory for a full frame; sampling profiles as dense point grids would need
~1e6 stamp placements per component. See the design spec
(docs/superpowers/specs/2026-07-02-extended-sources-cepheids-design.md).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from astropy import units as u
from astropy.modeling.models import Sersic2D
from scipy.signal import fftconvolve
from scipy.special import gamma, gammaincinv

from .chromatic import attenuation_factor
from .starflux import REF_MAG, rate_for_ab, rate_for_spt

_REFINE = 9         # subgrid factor for the central cusp
_REFINE_HALF_MIN = 8   # px; refinement box half-width bounds
_REFINE_HALF_MAX = 64


@dataclass(frozen=True, kw_only=True)
class ExtendedComponent(ABC):
    """Base of extended sources: something whose light can be mapped onto the detector.

    Only what every extended source has lives here: a `label` and the methods.
    Fields are keyword-only so subclasses can still take positional fields.
    Components describe the source only; observation settings (filter, detector
    WCS, sub-grid k) are method arguments.
    """

    label: str | None = None

    kind: ClassVar[str] = "extended"

    @abstractmethod
    def rate_map(self, det_wcs, det_shape, plate_scale_mas, sensorfilter, k=None):
        """Unconvolved e-/s per sub-pixel on the detector sub-grid (resample.Subgrid;
        total_in = the source's total e-/s), or None if it misses the detector."""

    @abstractmethod
    def footprint(self, det_wcs, det_shape):
        """Detector pixels (y0, y1, x0, x1) the source covers, or None."""

    def to_record(self):
        """Description of the source for the COMP table (FITS output)."""
        return {"label": self.label or "", "kind": self.kind}


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


@dataclass(frozen=True, eq=False)
class GalSimComponent:
    """One extended component described by a galsim GSObject.

    `profile` is any galsim.GSObject in arcsec (e.g. galsim.Sersic,
    galsim.Add of a bulge and disk, a sheared/rotated profile, or a
    RealGalaxy). Its own flux is ignored: `total_mag` (integrated Gaia-G
    vegamag, same convention as point sources and SersicComponent) sets the
    flux through the `template` spectrum and `ebv` reddening. galsim's
    position-angle convention (CCW from +x) matches SersicComponent.
    Profiles smaller than a pixel belong in the point-source catalog, not
    here: the renderer samples surface brightness, it does not integrate it.
    """

    ra: float
    dec: float
    profile: object
    total_mag: float
    template: str = "G2V"
    ebv: float = 0.0

    def __post_init__(self):
        if not hasattr(self.profile, "drawImage"):
            raise TypeError("profile must be a galsim.GSObject")
        if self.ebv < 0.0:
            raise ValueError(f"ebv must be >= 0, got {self.ebv}")


@dataclass(frozen=True, eq=False)
class ImageComponent:
    """A calibrated sky image (e.g. an HST drizzled mosaic) as an extended
    source, reprojected onto the frame.

    `data` is a 2-D array with WCS `wcs`; a pixel of value v has AB
    magnitude `zp_ab - 2.5 log10(v)`, so for an HST product in e-/s
    `zp_ab = -2.5 log10(PHOTFLAM) - 5 log10(PHOTPLAM) - 2.408` (see
    wcc_sim.phangs.ab_zeropoint). Each pixel is treated as an AB-flat
    source, i.e. the image's own band is taken to be the WCC band it is
    simulated in. `template`/`ebv` only pick the PSF kernel in chromatic
    mode. NaNs (outside the mosaic footprint) and negative pixels render as
    zero.
    """

    data: object
    wcs: object
    zp_ab: float
    template: str = "G2V"
    ebv: float = 0.0

    def __post_init__(self):
        if np.ndim(self.data) != 2:
            raise ValueError("data must be a 2-D array")
        if not np.issubdtype(np.asarray(self.data).dtype, np.floating):
            raise ValueError("data must be a float array (NaN marks no data)")
        if self.ebv < 0.0:
            raise ValueError(f"ebv must be >= 0, got {self.ebv}")


def component_total_rate(comp, sensorfilter):
    """Integrated e-/s of a component with `total_mag` set (reddened)."""
    att = attenuation_factor(comp.template, comp.ebv, sensorfilter)
    rate_ref = rate_for_spt(comp.template, sensorfilter)
    return rate_ref * 10.0 ** (-0.4 * (comp.total_mag - REF_MAG)) * att


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
    pix_arcsec = plate_scale_mas / 1000.0
    if comp.total_mag is not None:
        total = component_total_rate(comp, sensorfilter)
        r_eff_pix = comp.r_eff_arcsec / pix_arcsec
        return total / sersic_total_over_amplitude(
            comp.n, r_eff_pix, comp.ellip
        )
    att = attenuation_factor(comp.template, comp.ebv, sensorfilter)
    rate_ref = rate_for_spt(comp.template, sensorfilter)
    sb_rate = rate_ref * 10.0 ** (-0.4 * (comp.sb_mag_arcsec2 - REF_MAG)) * att
    return sb_rate * pix_arcsec**2


def _r999_pix(n, r_eff_pix):
    """Radius enclosing 99.9% of the Sersic flux, in pixels."""
    bn = float(gammaincinv(2.0 * n, 0.5))
    return r_eff_pix * (float(gammaincinv(2.0 * n, 0.999)) / bn) ** n


def render_galsim_profile(comp, wcs, shape, plate_scale_mas, sensorfilter):
    """GalSimComponent surface brightness in e-/s/pix on the native grid
    (unconvolved); None if its footprint misses the frame.

    The profile is drawn with galsim (`method="no_pixel"`: surface
    brightness sampled at pixel centers times pixel area, so the pixel
    response enters once through the PSF kernel in render_extended)
    directly onto the part of the frame inside its getGoodImageSize box, so
    a profile larger than the frame never allocates more than the frame.
    The central cusp box is re-drawn on a _REFINE x subgrid and summed, as
    the Sersic path does, so compact or high-n profiles keep their flux.
    """
    import galsim

    if not isinstance(comp.profile, galsim.GSObject):
        raise TypeError("profile must be a galsim.GSObject")
    ny, nx = shape
    x0, y0 = (float(v) for v in wcs.world_to_pixel_values(comp.ra, comp.dec))
    pix_arcsec = plate_scale_mas / 1000.0
    prof = comp.profile.withFlux(component_total_rate(comp, sensorfilter))
    half = int(prof.getGoodImageSize(pix_arcsec)) // 2 + 1
    ix0, iy0 = int(np.floor(x0 + 0.5)), int(np.floor(y0 + 0.5))
    b = galsim.BoundsI(max(0, ix0 - half), min(nx - 1, ix0 + half),
                       max(0, iy0 - half), min(ny - 1, iy0 + half))
    if not b.isDefined():
        return None
    stamp = galsim.Image(b, scale=pix_arcsec, dtype=np.float32)
    prof.drawImage(stamp, method="no_pixel", center=galsim.PositionD(x0, y0))
    # cusp: fine pixels carry SB x fine area, so sum (not mean) them
    h = _REFINE_HALF_MIN
    bf = galsim.BoundsI(ix0 - h, ix0 + h, iy0 - h, iy0 + h)
    ov = bf & b
    if ov.isDefined():
        nf = (2 * h + 1) * _REFINE
        fine = prof.drawImage(
            nx=nf, ny=nf, scale=pix_arcsec / _REFINE, method="no_pixel",
            offset=((x0 - ix0) * _REFINE, (y0 - iy0) * _REFINE),
        ).array
        coarse = fine.reshape(2 * h + 1, _REFINE, 2 * h + 1, _REFINE).sum(axis=(1, 3))
        stamp[ov].array[:] = coarse[ov.ymin - bf.ymin: ov.ymax - bf.ymin + 1,
                                    ov.xmin - bf.xmin: ov.xmax - bf.xmin + 1]
    img = np.zeros(shape, dtype=np.float32)
    img[b.ymin:b.ymax + 1, b.xmin:b.xmax + 1] = stamp.array
    return img


def render_image_profile(comp, wcs, shape, plate_scale_mas, sensorfilter):
    """ImageComponent surface brightness in e-/s/pix on the native grid
    (unconvolved); None if the frame misses the image.

    The input is first cut to the frame's footprint (so a 12000^2 HST
    mosaic is not interpolated whole), bilinearly reprojected onto the frame
    WCS with `reproject`, and scaled by the pixel-area ratio so surface
    brightness is conserved. No pixel-response convolution is applied here:
    as for the other components it enters once through the PSF kernel.
    """
    from astropy.coordinates import SkyCoord
    from astropy.nddata import Cutout2D, NoOverlapError
    from astropy.wcs.utils import proj_plane_pixel_area
    from reproject import reproject_interp

    ny, nx = shape
    pix_out_arcsec2 = (plate_scale_mas / 1000.0) ** 2
    pix_in_arcsec2 = proj_plane_pixel_area(comp.wcs) * 3600.0**2
    diag = np.hypot(nx, ny) * plate_scale_mas / 1000.0 * 1.05 * u.arcsec
    center = SkyCoord(*wcs.wcs.crval, unit="deg")
    try:
        cut = Cutout2D(comp.data, center, diag, wcs=comp.wcs, mode="partial",
                       fill_value=np.nan)
    except NoOverlapError:
        return None
    img = reproject_interp((cut.data, cut.wcs), wcs, shape_out=shape,
                           return_footprint=False, block_size=(1024, 1024))
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(img, 0.0, None, out=img)
    img *= rate_for_ab(comp.zp_ab, sensorfilter) * pix_out_arcsec2 / pix_in_arcsec2
    return img.astype(np.float32)


def render_component_profile(comp, wcs, shape, plate_scale_mas, sensorfilter):
    """Component surface brightness in e-/s/pix on the native grid
    (unconvolved); None if the 99.9%-flux footprint misses the frame.

    GalSimComponents are drawn by render_galsim_profile and ImageComponents
    by render_image_profile. For Sersic
    components the profile is evaluated at pixel centers in row chunks (full-frame
    float64 temporaries would be ~GB); a box around the center is
    re-evaluated on a _REFINE x subgrid and averaged, since a Sersic cusp
    changes across a pixel while the rest of the profile does not.
    """
    if isinstance(comp, GalSimComponent):
        return render_galsim_profile(comp, wcs, shape, plate_scale_mas, sensorfilter)
    if isinstance(comp, ImageComponent):
        return render_image_profile(comp, wcs, shape, plate_scale_mas, sensorfilter)
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
