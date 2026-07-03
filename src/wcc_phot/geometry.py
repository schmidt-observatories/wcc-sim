"""Aperture/centroid geometry, with defaults derived from the wcc-sim PSF model."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PhotGeometry:
    """Pixel-space geometry shared by centroiding and photometry."""

    r_ap: float  # aperture radius [px]
    r_in: float  # background annulus inner radius [px]
    r_out: float  # background annulus outer radius [px]
    centroid_box: int  # COM centroid box side [px, odd]
    fit_shape: int  # PSF-fit region side [px, odd]
    ee_fraction: float | None = None  # model EE inside r_ap (None if user-set)


def _odd(value):
    n = int(np.ceil(value))
    return n + 1 if n % 2 == 0 else n


def geometry_from_r_ap(
    r_ap,
    r_in=None,
    r_out=None,
    centroid_box=None,
    fit_shape=None,
    ee_fraction=None,
):
    """PhotGeometry from an aperture radius; unset pieces get scaled defaults."""
    r_ap = float(r_ap)
    if r_ap <= 0:
        raise ValueError(f"r_ap must be positive, got {r_ap}")
    r_in = float(r_in) if r_in is not None else 1.5 * r_ap
    r_out = float(r_out) if r_out is not None else 2.5 * r_ap
    if not r_ap <= r_in < r_out:
        raise ValueError(
            f"need r_ap <= r_in < r_out, got {r_ap}, {r_in}, {r_out}"
        )
    centroid_box = (
        int(centroid_box) if centroid_box is not None else _odd(2 * r_ap + 1)
    )
    fit_shape = int(fit_shape) if fit_shape is not None else _odd(2 * r_ap + 1)
    if centroid_box % 2 == 0 or fit_shape % 2 == 0:
        raise ValueError("centroid_box and fit_shape must be odd")
    return PhotGeometry(r_ap, r_in, r_out, centroid_box, fit_shape, ee_fraction)


def render_model_psf(meta, oversample=5):
    """Oversampled model PSF for a frame's sensorfilter/focus/jitter."""
    from wcc_sim.detectors import make_base_simulation
    from wcc_sim.psf import render_oversampled_psf

    sim = make_base_simulation(meta["sensorfilter"])
    return render_oversampled_psf(
        sim,
        int(meta["focus"]),
        oversample=oversample,
        jitter_sigma_mas=float(meta["jitter_sigma_mas"]),
    )


def ee_radius(psf_det, fraction=0.95):
    """Radius [px] enclosing `fraction` of a centered detector-sampled PSF."""
    n = psf_det.shape[0]
    center = (n - 1) / 2
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - center, yy - center).ravel()
    order = np.argsort(r)
    cum = np.cumsum(psf_det.ravel()[order])
    cum /= cum[-1]
    i = int(np.searchsorted(cum, fraction))
    return float(r[order][min(i, r.size - 1)])


def default_geometry(meta, ee=0.95, oversample=5, psf_os=None):
    """Model-PSF-driven geometry: r_ap = EE(ee) radius, annulus 1.5-2.5 r_ap."""
    from wcc_sim.render import bin_oversampled

    if psf_os is None:
        psf_os = render_model_psf(meta, oversample=oversample)
    psf_det = bin_oversampled(psf_os, oversample)
    r_ap = float(np.ceil(max(ee_radius(psf_det, ee), 2.0)))

    n = psf_det.shape[0]
    center = (n - 1) / 2
    yy, xx = np.mgrid[:n, :n]
    inside = np.hypot(xx - center, yy - center) <= r_ap
    ee_fraction = float(psf_det[inside].sum() / psf_det.sum())
    return geometry_from_r_ap(r_ap, ee_fraction=ee_fraction)
