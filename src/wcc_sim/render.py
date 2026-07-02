"""Place PSF stamps on the detector grid and apply the ETC noise model."""

import numpy as np
from astropy import units as u
from wcc_etc.psfsim import saturation_mask_from_image_e

from .starflux import sky_and_dark_rates


def bin_oversampled(stamp_os, oversample):
    n = stamp_os.shape[0] // oversample
    return (
        stamp_os[: n * oversample, : n * oversample]
        .reshape(n, oversample, n, oversample)
        .sum(axis=(1, 3))
    )


def add_star(image, psf_os, x, y, flux_e, oversample):
    """Add one star at float pixel (x, y), sub-pixel placed, edge-clipped.

    Sub-pixel shift is a fine-grid np.roll (error <= 1/(2*oversample) px);
    rolled wrap-around energy is negligible because stamp edges are ~0.
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
    stamp = stamp * flux_e

    y_lo, y_hi = y0 - half, y0 + half + 1
    x_lo, x_hi = x0 - half, x0 + half + 1
    iy_lo, ix_lo = max(y_lo, 0), max(x_lo, 0)
    iy_hi, ix_hi = min(y_hi, ny), min(x_hi, nx)
    image[iy_lo:iy_hi, ix_lo:ix_hi] += stamp[
        iy_lo - y_lo : iy_hi - y_lo, ix_lo - x_lo : ix_hi - x_lo
    ]


def render_scene(shape, xs, ys, fluxes_e, psf_os, oversample):
    """Sum of PSF stamps (e-) for all stars on a float32 (ny, nx) grid."""
    image = np.zeros(shape, dtype=np.float32)
    for x, y, f in zip(np.atleast_1d(xs), np.atleast_1d(ys), np.atleast_1d(fluxes_e)):
        add_star(image, psf_os, float(x), float(y), float(f), oversample)
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

    well = sensor.meta.get("well_depth")
    cap_frame_e = adc_max * gain if well is None else min(float(well), adc_max * gain)
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
