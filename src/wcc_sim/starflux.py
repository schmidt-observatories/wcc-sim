"""Convert Gaia photometry to WCC detector count rates via wcc_etc.

Each star: BP-RP -> nearest Pickles dwarf type; that spectrum normalized to
the star's G mag (vegamag) in the Gaia DR3 G bandpass; integrated through the
instrument throughput by wcc_etc. Rates are memoized per (spt, sensorfilter)
at G = REF_MAG and scaled analytically per star.
"""

import os
from functools import lru_cache

import numpy as np
from astropy import units as u
from synphot import Empirical1D, SpectralElement
from wcc_etc import Simulation
from wcc_etc.scene import get_scene

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REF_MAG = 15.0

_RATE_CACHE = {}


@lru_cache(maxsize=1)
def gaia_g_bandpass():
    """Gaia DR3 G passband (SVO FPS: GAIA/GAIA3.G) as a SpectralElement."""
    wave, trans = np.loadtxt(os.path.join(DATA_DIR, "gaia_dr3_g.dat"), unpack=True)
    return SpectralElement(
        Empirical1D,
        points=wave * u.AA,
        lookup_table=trans,
        keep_neg=False,
        fill_value=0,  # Without this, synphot extrapolates edge values outside tabulated range.
    )


@lru_cache(maxsize=1)
def _spt_table():
    path = os.path.join(DATA_DIR, "bp_rp_to_spt.csv")
    spts = np.loadtxt(path, delimiter=",", skiprows=1, usecols=0, dtype=str)
    colors = np.loadtxt(path, delimiter=",", skiprows=1, usecols=1)
    return spts, colors


def spt_from_bp_rp(bp_rp):
    """Nearest-neighbor Pickles dwarf type for BP-RP; NaN -> 'G2V'."""
    spts, colors = _spt_table()
    bp_rp = np.atleast_1d(
        np.ma.filled(np.ma.masked_invalid(bp_rp), np.nan).astype(float)
    )
    out = np.full(bp_rp.shape, "G2V", dtype=object)
    ok = np.isfinite(bp_rp)
    idx = np.abs(bp_rp[ok, None] - colors[None, :]).argmin(axis=1)
    out[ok] = spts[idx]
    return out.astype(str)


def make_star_simulation(spt, sensorfilter, mag=REF_MAG):
    """wcc_etc Simulation for one Pickles star normalized in Gaia G (vegamag)."""
    scene = get_scene(spt, mag=mag, magsys="vegamag", bandpass=gaia_g_bandpass())
    return Simulation.from_sensorfilter(sensorfilter, scene)


def rate_for_spt(spt, sensorfilter):
    """Total point-source rate (e-/s) at G = REF_MAG, memoized.

    `spt` is coerced to a plain `str`: wcc_etc.scene dispatches on an exact
    `type(x) is str` check, so a `numpy.str_` (as produced by indexing the
    array `spt_from_bp_rp` returns) silently falls through to a spectrum-less
    scene and a rate of 0.0 instead of raising.
    """
    spt = str(spt)
    key = (spt, sensorfilter)
    if key not in _RATE_CACHE:
        sim = make_star_simulation(spt, sensorfilter)
        _RATE_CACHE[key] = float(
            sim._count_rate_components()["source_rate_total"]
        )
    return _RATE_CACHE[key]


def rates_for_catalog(catalog, sensorfilter):
    """Per-star (rate_e_s, spt) arrays for a Gaia catalog Table.

    An optional `spt` column overrides the BP-RP lookup row-wise (empty
    string = no override).
    """
    g = np.asarray(catalog["phot_g_mean_mag"], dtype=float)
    bp = np.ma.filled(np.ma.masked_invalid(
        np.asarray(catalog["phot_bp_mean_mag"], dtype=float)), np.nan)
    rp = np.ma.filled(np.ma.masked_invalid(
        np.asarray(catalog["phot_rp_mean_mag"], dtype=float)), np.nan)
    spts = spt_from_bp_rp(bp - rp)
    if "spt" in catalog.colnames:
        # Optional per-row override (e.g. supergiant templates for injected
        # Cepheids — the BP-RP table maps to dwarfs only). Empty string
        # means "no override". Merge via object dtype: assigning into the
        # fixed-width array from spt_from_bp_rp would silently truncate
        # longer type names.
        override = np.array(
            [str(s).strip() for s in catalog["spt"]], dtype=object
        )
        merged = spts.astype(object)
        use = override != ""
        merged[use] = override[use]
        spts = merged.astype(str)
    rates = np.array(
        [
            rate_for_spt(spt, sensorfilter) * 10.0 ** (-0.4 * (gmag - REF_MAG))
            for spt, gmag in zip(spts, g)
        ]
    )
    return rates, spts


def sky_and_dark_rates(sim):
    """(sky_e_s_per_pix, dark_e_s_per_pix) for a base simulation."""
    sky = float(sim._count_rate_components()["background_rate_per_pix"])
    dark = float(sim.sensor.dark_current.to(u.electron / u.pix / u.s).value)
    return sky, dark
