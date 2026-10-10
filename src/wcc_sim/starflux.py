"""Convert Gaia photometry to WCC detector count rates via wcc_etc.

Each star: BP-RP -> nearest Pickles dwarf template for the SED (and the PSF
colour); the count rate is interpolated in log(rate) between the templates
bracketing the star's colour. Each template is normalized to G = REF_MAG
(vegamag) in the Gaia DR3 G bandpass and integrated through the instrument
throughput by wcc_etc; rates are memoized per (spt, sensorfilter) and scaled
analytically per star.
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


@lru_cache(maxsize=3)
def gaia_bandpass(band="g"):
    """Gaia DR3 passband (SVO FPS GAIA/GAIA3.G, .Gbp, .Grp) as a SpectralElement."""
    band = band.lower()
    if band not in ("g", "bp", "rp"):
        raise ValueError(f"band must be 'g', 'bp' or 'rp', got {band!r}")
    wave, trans = np.loadtxt(
        os.path.join(DATA_DIR, f"gaia_dr3_{band}.dat"), unpack=True
    )
    return SpectralElement(
        Empirical1D,
        points=wave * u.AA,
        lookup_table=trans,
        keep_neg=False,
        fill_value=0,  # Without this, synphot extrapolates edge values outside tabulated range.
    )


def gaia_g_bandpass():
    """Gaia DR3 G passband (SVO FPS: GAIA/GAIA3.G) as a SpectralElement."""
    return gaia_bandpass("g")


def column_floats(cat, name, default=np.nan):
    """Column `name` as float, with masked and non-finite entries -> `default`.

    `np.asarray` on a MaskedColumn returns the data *under* the mask (0.0 for
    a value astropy read back from an ECSV cache), so a magnitude that was
    never measured would be used as if it had been: a masked BP with 0
    underneath turned a G = 18 star into an O5V. Read the mask itself.
    """
    col = cat[name]
    masked = np.ma.getmaskarray(np.ma.asarray(col))
    values = np.asarray(np.ma.getdata(col), dtype=float)
    return np.where(masked | ~np.isfinite(values), default, values)


def _finite(values):
    """Array-like, possibly masked -> 1-d float array with NaN where missing."""
    arr = np.ma.masked_invalid(np.ma.asarray(values, dtype=float))
    return np.atleast_1d(np.ma.filled(arr, np.nan))


@lru_cache(maxsize=1)
def _synthetic_color_table():
    """(spts, Johnson B-V, Gaia G - Johnson V) per Pickles template.

    Vendored rather than computed: the synphot integrations cost ~8 s and
    need the remote Johnson curves. Regenerate with
    scripts/build_spt_colors.py.
    """
    path = os.path.join(DATA_DIR, "spt_synthetic_colors.csv")
    spts = np.loadtxt(path, delimiter=",", skiprows=1, usecols=0, dtype=str)
    b_v, g_v = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2),
                          unpack=True)
    return spts, b_v, g_v


@lru_cache(maxsize=1)
def _spt_table():
    """(spts, synthetic Gaia DR3 BP-RP) per Pickles dwarf template.

    Synthetic, from the same spectra the rates come from, so a template's
    colour maps back to itself. The hand-typed table this replaced held
    B-V for the O and B rows and was 0.2 mag off at K7V and M4V, so a real
    M4V star (BP-RP ~2.9) got the M5V template.
    """
    path = os.path.join(DATA_DIR, "spt_synthetic_colors.csv")
    spts = np.loadtxt(path, delimiter=",", skiprows=1, usecols=0, dtype=str)
    colors = np.loadtxt(path, delimiter=",", skiprows=1, usecols=3)
    return spts, colors


def spt_from_bp_rp(bp_rp):
    """Nearest-neighbor Pickles dwarf type for BP-RP; NaN/masked -> 'G2V'."""
    spts, colors = _spt_table()
    bp_rp = _finite(bp_rp)
    out = np.full(bp_rp.shape, "G2V", dtype=object)
    ok = np.isfinite(bp_rp)
    idx = np.abs(bp_rp[ok, None] - colors[None, :]).argmin(axis=1)
    out[ok] = spts[idx]
    return out.astype(str)


def spt_from_b_v(b_v):
    """Nearest-neighbor Pickles dwarf type for Johnson B-V; NaN/masked -> 'G2V'.

    The B-V counterpart of spt_from_bp_rp, for catalogs (Hipparcos) that
    give Johnson photometry instead of Gaia's.
    """
    spts, colors, _ = _synthetic_color_table()
    b_v = _finite(b_v)
    out = np.full(b_v.shape, "G2V", dtype=object)
    ok = np.isfinite(b_v)
    if ok.any():
        out[ok] = spts[np.abs(b_v[ok, None] - colors[None, :]).argmin(axis=1)]
    return out.astype(str)


def _by_spt(values, spt):
    spts, _, _ = _synthetic_color_table()
    index = {str(s): i for i, s in enumerate(spts)}
    return np.array([values[index[str(s)]] for s in np.atleast_1d(spt)],
                    dtype=float)


def g_minus_v(spt):
    """Synthetic Gaia G - Johnson V for a Pickles type."""
    return _by_spt(_synthetic_color_table()[2], spt)


@lru_cache(maxsize=1)
def _g_v_vs_b_v():
    """(B-V, G-V) sorted in B-V, for interpolation.

    The synthetic sequence is not monotonic in colour -- O9V (-0.3218) sits
    redward of B0V (-0.3323), M0V (1.3458) blueward of K7V (1.3937) -- so
    np.interp on the table order would return nonsense there.
    """
    _, b_v, g_v = _synthetic_color_table()
    order = np.argsort(b_v)
    return b_v[order], g_v[order]


def g_minus_v_at_b_v(b_v):
    """Synthetic Gaia G - Johnson V interpolated in B-V.

    Taking the nearest template's G-V makes the derived G a step function of
    colour: between M2V (B-V 1.461, G-V -0.859) and M4V (1.618, -1.408) it
    jumps 0.55 mag at the midpoint, well inside Hipparcos's own B-V error for
    a red star. The template still sets the SED; only this conversion is
    interpolated. NaN colour falls back to the G2V entry, matching
    spt_from_b_v's own fallback.

    Colours outside the tabulated range are clamped to the end templates
    (np.interp's default), which is the right behaviour: extrapolating a
    colour-colour sequence off its end is worse than saturating it.
    """
    colors, values = _g_v_vs_b_v()
    b_v = _finite(b_v)
    out = np.full(b_v.shape, g_minus_v("G2V")[0], dtype=float)
    ok = np.isfinite(b_v)
    if ok.any():
        out[ok] = np.interp(b_v[ok], colors, values)
    return out


def bp_rp_for_spt(spt):
    """The BP-RP that spt_from_bp_rp maps back to this type."""
    spts, colors = _spt_table()
    index = {str(s): i for i, s in enumerate(spts)}
    return np.array([colors[index[str(s)]] for s in np.atleast_1d(spt)],
                    dtype=float)


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
        try:
            sim = make_star_simulation(spt, sensorfilter)
        except Exception as exc:  # synphot: "Cannot determine filename."
            raise ValueError(
                f"unknown spectral template {spt!r} (check the catalog 'spt' "
                "override column; names are wcc_etc Pickles types such as "
                "'G2V' or 'F8I')"
            ) from exc
        _RATE_CACHE[key] = float(
            sim._count_rate_components()["source_rate_total"]
        )
    return _RATE_CACHE[key]


def log_rate_at_bp_rp(bp_rp, sensorfilter):
    """log10 of the rate (e-/s) at G = REF_MAG, interpolated in BP-RP.

    Snapping to the nearest template made the rate a step function of
    colour: M2V -> M4V is -40% in r and +40% in z at fixed G, about +-20%
    rate error across the M dwarfs from the lookup alone. Interpolating
    log(rate) between the two bracketing templates removes the steps; the
    nearest template (`spt_from_bp_rp`) still sets the SED and the PSF
    colour. Colours outside the table clamp to the end templates. A missing
    colour gets the G2V rate, matching spt_from_bp_rp's fallback.
    """
    bp_rp = _finite(bp_rp)
    out = np.full(bp_rp.shape, np.log10(rate_for_spt("G2V", sensorfilter)))
    ok = np.isfinite(bp_rp)
    if ok.any():
        spts, colors = _spt_table()
        order = np.argsort(colors)
        log_rates = [np.log10(rate_for_spt(spts[i], sensorfilter)) for i in order]
        out[ok] = np.interp(bp_rp[ok], colors[order], log_rates)
    return out


def bp_rp_from_catalog(catalog):
    """Per-row BP-RP, NaN wherever either magnitude is masked or non-finite."""
    return (column_floats(catalog, "phot_bp_mean_mag")
            - column_floats(catalog, "phot_rp_mean_mag"))


def color_fallback(catalog):
    """True where a row's SED and rate come from the G2V fallback.

    A row has no usable BP-RP and no `spt` override. Written to the CAT
    extension as `spt_fallback` so the provenance is visible downstream.
    """
    fallback = ~np.isfinite(bp_rp_from_catalog(catalog))
    if "spt" in catalog.colnames:
        fallback &= np.array([str(s).strip() == "" for s in catalog["spt"]],
                             dtype=bool)
    return fallback


def rates_for_catalog(catalog, sensorfilter):
    """Per-star (rate_e_s, spt) arrays for a Gaia catalog Table.

    Masked or non-finite BP/RP mean "no colour": the G2V template and rate.
    A masked or non-finite G gives a NaN rate; callers decide whether to drop
    the row. An optional `spt` column overrides the BP-RP lookup row-wise
    (empty string = no override) and uses that template's own rate rather
    than the colour interpolation.
    """
    g = column_floats(catalog, "phot_g_mean_mag")
    bp_rp = bp_rp_from_catalog(catalog)
    spts = spt_from_bp_rp(bp_rp)
    log_rate = log_rate_at_bp_rp(bp_rp, sensorfilter)
    if "spt" in catalog.colnames:
        # Optional per-row override (e.g. supergiant templates for injected
        # Cepheids -- the BP-RP table maps to dwarfs only). Empty string
        # means "no override". Merge via object dtype: assigning into the
        # fixed-width array from spt_from_bp_rp would silently truncate
        # longer type names.
        override = np.array(
            [str(s).strip() for s in catalog["spt"]], dtype=object
        )
        use = override != ""
        merged = spts.astype(object)
        merged[use] = override[use]
        spts = merged.astype(str)
        log_rate[use] = [np.log10(rate_for_spt(s, sensorfilter))
                         for s in override[use]]
    rates = 10.0 ** (log_rate - 0.4 * (g - REF_MAG))
    return rates, spts


def sky_and_dark_rates(sim):
    """(sky_e_s_per_pix, dark_e_s_per_pix) for a base simulation."""
    sky = float(sim._count_rate_components()["background_rate_per_pix"])
    dark = float(sim.sensor.dark_current.to(u.electron / u.pix / u.s).value)
    return sky, dark
