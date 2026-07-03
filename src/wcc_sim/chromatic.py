"""Spectrum-weighted effective PSFs and band-averaged extinction factors.

Convolution is linear, so the band-integrated image of a source with photon
spectrum S(lambda) equals its surface-brightness model convolved with ONE
effective PSF: sum_i w_i PSF(lambda_i), with w_i proportional to the
integral of S(lambda) T(lambda) over node sub-bands (T = total throughput,
sim.sensor.bandpass). Per-wavelength image convolution is never needed for
a component with a single spectrum.

Only the in-focus Airy PSF has a wavelength model; wcc_etc.DefocusPSF is a
fixed measured Huygens image and the defocused PSF is geometry-dominated,
so focus=1,2 return the monochromatic stamp (documented passthrough).

Reddening enters twice, consistently: attenuation_factor() gives the
band-averaged rate multiplier for a Fitzpatrick (1999) R_V=3.1 curve, and
band_nodes(ebv=...) reddens the weights so a dusty source also gets a
redder effective PSF. wcc_etc itself is never modified.
"""

from functools import lru_cache

import numpy as np
from astropy import units as u
from dust_extinction.parameter_averages import F99
from wcc_etc.scene import get_scene_element

from .detectors import make_base_simulation
from .psf import DEFAULT_STAMP, render_oversampled_psf

_N_WAVE = 4096  # fine-grid points for band integrals
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy<2 has no trapezoid
_EFF_PSF_CACHE = {}


def band_support(sim, frac=0.005):
    """(lo, hi) wavelength bounds in Angstrom where throughput > frac*peak."""
    bp = sim.sensor.bandpass
    ws = bp.waveset
    t = bp(ws).value
    m = t > frac * t.max()
    return float(ws[m].min().value), float(ws[m].max().value)


def _weighted_flux(sim, spectrum, ebv=0.0, rv=3.1):
    """(wave_aa, S*T [*extinction]) on a fine grid across the band."""
    lo, hi = band_support(sim)
    wave = np.linspace(lo, hi, _N_WAVE) * u.AA
    st = (spectrum(wave) * sim.sensor.bandpass(wave)).value
    if ebv > 0.0:
        st = st * F99(Rv=rv).extinguish(wave, Ebv=ebv)
    return wave.value, st


def band_nodes(sim, spectrum, n_nodes=7, ebv=0.0, rv=3.1):
    """Node wavelengths [m] and unit-sum weights for the effective PSF.

    Weights are S(lambda)*T(lambda) integrals over n_nodes equal sub-bands;
    each node sits at its sub-band's flux-weighted mean wavelength.
    """
    if n_nodes < 1:
        raise ValueError(f"n_nodes must be >= 1, got {n_nodes}")
    wave_aa, st = _weighted_flux(sim, spectrum, ebv=ebv, rv=rv)
    edges = np.linspace(wave_aa[0], wave_aa[-1], n_nodes + 1)
    idx = np.clip(np.searchsorted(edges, wave_aa) - 1, 0, n_nodes - 1)
    weights = np.zeros(n_nodes)
    waves_aa = 0.5 * (edges[:-1] + edges[1:])  # fallback: bin centers
    for i in range(n_nodes):
        s = st[idx == i]
        weights[i] = s.sum()
        if s.sum() > 0.0:
            waves_aa[i] = np.average(wave_aa[idx == i], weights=s)
    total = weights.sum()
    if total <= 0.0:
        raise ValueError("spectrum has no flux inside the bandpass")
    return waves_aa * 1e-10, weights / total


def effective_psf(sim, focus, spectrum=None, oversample=11, stamp_npix=None,
                  jitter_sigma_mas=None, n_nodes=7, ebv=0.0):
    """Spectrum-weighted oversampled PSF; monochromatic passthrough when
    focus != 0 (no wavelength model), n_nodes == 1, or spectrum is None."""
    if n_nodes < 1:
        raise ValueError(f"n_nodes must be >= 1, got {n_nodes}")
    if focus != 0 or n_nodes == 1 or spectrum is None:
        return render_oversampled_psf(
            sim, focus, oversample=oversample, stamp_npix=stamp_npix,
            jitter_sigma_mas=jitter_sigma_mas,
        )
    waves_m, weights = band_nodes(sim, spectrum, n_nodes=n_nodes, ebv=ebv)
    psf = None
    for w_m, wt in zip(waves_m, weights):
        p = render_oversampled_psf(
            sim, focus, oversample=oversample, stamp_npix=stamp_npix,
            jitter_sigma_mas=jitter_sigma_mas, wavelength_m=w_m,
        )
        psf = wt * p if psf is None else psf + wt * p
    return psf


def effective_psf_for_spt(sim, sensorfilter, spt, ebv, focus, oversample,
                          stamp_npix=None, jitter_sigma_mas=None, n_nodes=7):
    """Cached effective PSF for one spectral template (+ optional reddening).

    `spt` is coerced to str for the wcc_etc type-is-str dispatch. The cache
    key resolves jitter/stamp defaults first so equivalent calls share.
    """
    if stamp_npix is None:
        stamp_npix = DEFAULT_STAMP[focus]
    if jitter_sigma_mas is None:
        jitter_sigma_mas = float(sim.telescope.jitter_sigma.to("mas").value)
    key = (
        sensorfilter, str(spt), round(float(ebv), 4), int(focus),
        int(oversample), int(stamp_npix), round(float(jitter_sigma_mas), 3),
        int(n_nodes),
    )
    if key not in _EFF_PSF_CACHE:
        spectrum = get_scene_element(str(spt), mag=15.0).spectrum
        _EFF_PSF_CACHE[key] = effective_psf(
            sim, focus, spectrum, oversample=oversample,
            stamp_npix=stamp_npix, jitter_sigma_mas=jitter_sigma_mas,
            n_nodes=n_nodes, ebv=float(ebv),
        )
    return _EFF_PSF_CACHE[key]


@lru_cache(maxsize=256)
def attenuation_factor(template, ebv, sensorfilter, rv=3.1):
    """Band-averaged flux attenuation of `template` for a given E(B-V).

    a = int S T 10^(-0.4 A(lambda)) dlambda / int S T dlambda with a
    Fitzpatrick (1999) curve; the exact broadband dimming factor to apply
    to the unreddened wcc_etc count rate.
    """
    ebv = float(ebv)
    if ebv < 0.0:
        raise ValueError(f"ebv must be >= 0, got {ebv}")
    if ebv == 0.0:
        return 1.0
    sim = make_base_simulation(sensorfilter)
    spectrum = get_scene_element(str(template), mag=15.0).spectrum
    wave_aa, st = _weighted_flux(sim, spectrum)
    trans = F99(Rv=rv).extinguish(wave_aa * u.AA, Ebv=ebv)
    return float(_trapezoid(st * trans, wave_aa) / _trapezoid(st, wave_aa))
