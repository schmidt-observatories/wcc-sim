"""Place PSF stamps on the detector grid and apply the ETC noise model."""

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from wcc_etc.psfsim import saturation_mask_from_image_e

from .detectors import saturation_level_e
from .starflux import sky_and_dark_rates


def bin_oversampled(stamp_os, oversample):
    n = stamp_os.shape[0] // oversample
    return (
        stamp_os[: n * oversample, : n * oversample]
        .reshape(n, oversample, n, oversample)
        .sum(axis=(1, 3))
    )


@dataclass(frozen=True)
class StampBlend:
    """Everything about one stamp geometry that does not depend on the star.

    `w` is a cosine taper (1 in the core, 0 at r >= half-width) and `model`
    is the *diffraction* wing on the stamp grid; blending the stamp's outer
    annulus into the smooth model removes the Airy-ring texture that would
    otherwise end abruptly at the square stamp edge. `halo` is the measured
    scattered-light profile on the same grid, and `core_scale` the weight on
    the diffraction term -- both inert (0 and 1) when there is no scatter
    model, which reproduces the diffraction-only rendering exactly.
    """

    w: np.ndarray
    model: np.ndarray
    halo: np.ndarray
    core_scale: float


def wing_blend(n_stamp, wing, blend_start=0.8):
    """Precompute the stamp->wing crossfade and halo for one stamp geometry."""
    half = n_stamp // 2
    yy, xx = np.mgrid[:n_stamp, :n_stamp]
    r = np.hypot(yy - half, xx - half)
    t = np.clip((r - blend_start * half) / ((1.0 - blend_start) * half), 0.0, 1.0)
    w = 0.5 * (1.0 + np.cos(np.pi * t))
    # The scatter halo is added across the whole stamp, not crossfaded in at
    # the edge: it is a separate additive term of the PSF, present under the
    # core as well (where the core outruns it by ~1e9 and it does not matter).
    core = getattr(wing, "core", wing)
    halo_model = getattr(wing, "halo", None)
    return StampBlend(
        w=w,
        model=core.profile(np.maximum(r, 1.0)),
        halo=(
            np.zeros_like(r) if halo_model is None else halo_model.profile(r)
        ),
        core_scale=1.0 if halo_model is None else 1.0 - halo_model.frac_total,
    )


def _add_wing_halo(image, x0, y0, half, scaled_flux, wing, floor_e):
    """Add flux*profile(r) outside the square stamp footprint, in row chunks."""
    ny, nx = image.shape
    r_out = wing.r_out(scaled_flux, floor_e)
    if r_out <= half:
        return
    r_out_i = int(np.ceil(r_out))
    by_lo, by_hi = max(y0 - r_out_i, 0), min(y0 + r_out_i + 1, ny)
    bx_lo, bx_hi = max(x0 - r_out_i, 0), min(x0 + r_out_i + 1, nx)
    if by_lo >= by_hi or bx_lo >= bx_hi:
        return
    xx = np.arange(bx_lo, bx_hi, dtype=float)[None, :] - x0
    for cy_lo in range(by_lo, by_hi, 1024):
        cy_hi = min(cy_lo + 1024, by_hi)
        yy = np.arange(cy_lo, cy_hi, dtype=float)[:, None] - y0
        rr = np.hypot(yy, xx)
        m = (rr <= r_out) & ((np.abs(yy) > half) | (np.abs(xx) > half))
        if not m.any():
            continue
        halo = np.zeros(rr.shape, dtype=np.float32)
        halo[m] = scaled_flux * wing.profile(rr[m])
        image[cy_lo:cy_hi, bx_lo:bx_hi] += halo


def add_star(image, psf_os, x, y, flux_e, oversample, wing=None, floor_e=None,
             blend=None):
    """Add one star at float pixel (x, y), sub-pixel placed, edge-clipped.

    Sub-pixel shift is a fine-grid np.roll (error <= 1/(2*oversample) px);
    rolled wrap-around energy is negligible because stamp edges are ~0.

    With a `wing` model (see wcc_sim.wings), the stamp's outer annulus is
    crossfaded into the smooth wing profile and the halo is continued on a
    circular footprint out to where flux_e * profile(r) < floor_e. Star flux
    is renormalized by `wing.flux_norm(half)` for all stars so photometry
    stays magnitude-independent; the sub-floor halo of faint stars is simply
    not drawn.

    A CombinedWing additionally carries the measured scattered-light halo,
    which is added across the stamp as well as beyond it.
    """
    ny, nx = image.shape
    n_stamp = psf_os.shape[0] // oversample
    half = n_stamp // 2
    x0, y0 = int(round(x)), int(round(y))
    if x0 + half < 0 or x0 - half >= nx or y0 + half < 0 or y0 - half >= ny:
        return
    sx = int(round((x - x0) * oversample))
    sy = int(round((y - y0) * oversample))
    stamp = bin_oversampled(np.roll(psf_os, (sy, sx), axis=(0, 1)), oversample)
    if wing is not None:
        if blend is None:
            blend = wing_blend(n_stamp, wing)
        flux_e = flux_e / wing.flux_norm(float(half))
        stamp = flux_e * (
            blend.core_scale
            * (stamp * blend.w + blend.model * (1.0 - blend.w))
            + blend.halo
        )
    else:
        stamp = stamp * flux_e

    y_lo, y_hi = y0 - half, y0 + half + 1
    x_lo, x_hi = x0 - half, x0 + half + 1
    iy_lo, ix_lo = max(y_lo, 0), max(x_lo, 0)
    iy_hi, ix_hi = min(y_hi, ny), min(x_hi, nx)
    image[iy_lo:iy_hi, ix_lo:ix_hi] += stamp[
        iy_lo - y_lo : iy_hi - y_lo, ix_lo - x_lo : ix_hi - x_lo
    ]

    if wing is not None and floor_e is not None:
        _add_wing_halo(image, x0, y0, half, flux_e, wing, floor_e)


def render_scene(shape, xs, ys, fluxes_e, psf_os, oversample, wing=None,
                 floor_e=None):
    """Sum of PSF stamps (e-) for all stars on a float32 (ny, nx) grid."""
    image = np.zeros(shape, dtype=np.float32)
    blend = None
    if wing is not None:
        blend = wing_blend(psf_os.shape[0] // oversample, wing)
    for x, y, f in zip(np.atleast_1d(xs), np.atleast_1d(ys), np.atleast_1d(fluxes_e)):
        add_star(image, psf_os, float(x), float(y), float(f), oversample,
                 wing=wing, floor_e=floor_e, blend=blend)
    return image


def star_saturated(satmask, x, y, radius=32):
    """True if any saturated pixel lies within `radius` px of (x, y).

    A window is used rather than the central pixel alone because the
    defocused PSFs are centrally depressed: a bright star can saturate
    its ring while its central pixel stays below full well. The 2-wave
    defocus PSF's bright ring extends to ~29 px from center.
    """
    ny, nx = satmask.shape
    x0, y0 = int(round(x)), int(round(y))
    y_lo, y_hi = max(y0 - radius, 0), min(y0 + radius + 1, ny)
    x_lo, x_hi = max(x0 - radius, 0), min(x0 + radius + 1, nx)
    if y_lo >= y_hi or x_lo >= x_hi:
        return False
    return bool(satmask[y_lo:y_hi, x_lo:x_hi].any())


def add_noise_and_digitize(image_sources_e, sim, exptime, n_reads, rng, add_noise=True):
    """Apply sky+dark, per-frame saturation, Poisson + read noise, ADU conversion.

    ETC semantics (wcc_etc.simulation): read-noise variance scales with
    n_reads; saturation is evaluated on the per-frame expectation.
    """
    n_reads = int(n_reads)
    if n_reads < 1:
        raise ValueError(f"n_reads must be >= 1, got {n_reads}")
    sensor = sim.sensor
    gain = float(sensor.gain.to(u.electron / u.ct).value)
    read_noise = float(sensor.read_noise.value)
    adc_max = float(sensor.adc_max.to(u.ct).value)
    bias = float(sensor.bias_level.to(u.ct).value)
    sky, dark = sky_and_dark_rates(sim)

    image_clean = image_sources_e + np.float32((sky + dark) * exptime)

    per_frame = image_clean / n_reads
    satmask = saturation_mask_from_image_e(sensor, per_frame)

    cap_frame_e = saturation_level_e(sensor)
    expectation = np.minimum(image_clean, np.float32(n_reads * cap_frame_e))

    if add_noise:
        image_e = rng.poisson(expectation.astype(np.float64)).astype(np.float32)
        image_e += rng.normal(
            0.0, read_noise * np.sqrt(n_reads), size=image_e.shape
        ).astype(np.float32)
    else:
        image_e = image_clean.copy()

    image_adu = np.clip(image_e / gain + bias, 0.0, n_reads * adc_max).astype(
        np.float32
    )
    return {
        "image_clean": image_clean.astype(np.float32),
        "image_e": image_e,
        "image_adu": image_adu,
        "satmask": np.asarray(satmask, dtype=bool),
    }
