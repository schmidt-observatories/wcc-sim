"""Pixel values in physical units -> WCC detector count rates (e-/s per pixel).

`flux_to_rate(unit, sensorfilter, pixel_area_arcsec2)` returns the factor that
turns an image in `unit` into e-/s per pixel for one WCC sensorfilter. Supported:

- flux density per pixel (Jy/pixel, mJy/pix, or a plain Jy),
- surface brightness (Jy/arcsec2, MJy/sr), needs the pixel area,
- F_lambda per pixel or per arcsec2 (erg/(s cm2 Angstrom) [/pix | /arcsec2]),
- electron/s [/pix], already a detector rate.

The rate per unit flux comes from wcc_etc: a flat spectrum (flat in F_nu for
Jy units, flat in F_lambda for F_lambda units) normalised to 1 Jy or
1 erg/s/cm2/A (zero points from astropy's ABmag / STmag) and integrated through
the telescope + WCC throughput. Pixel values are thus taken as the band-mean
flux density of a flat spectrum. For an image normalised in the same WCC filter
(e.g. a model normalised in zwo:halpha2) this is exact by construction: the
band-mean F_nu is defined by equal photon counts. For another band it is a
flat-spectrum transfer.

`band_fluxes(rate_e_s, sensorfilter)` goes the other way: the flat-spectrum flux
in that filter that gives a count rate, as F_nu [Jy], F_lambda at the pivot and
band-integrated [erg/s/cm2] (F_lambda x top-hat-equivalent width).
"""

from functools import lru_cache

from astropy import units as u

_FLAM = u.erg / u.s / u.cm**2 / u.AA
_COUNTS = u.electron / u.s

_SUPPORTED = ("flux density per pixel (e.g. Jy/pixel), surface brightness "
              "(e.g. Jy/arcsec2, MJy/sr), F_lambda (erg/(s cm2 Angstrom) [/pix | "
              "/arcsec2]) or electron/s [/pix]")


def _source_rate(sensorfilter, flat_unit, mag, magsys):
    """e-/s of a flat-spectrum source (wcc_etc), validating the sensorfilter first."""
    from wcc_etc import Simulation
    from wcc_etc.io import SENSORFILTER_FOCUS
    from wcc_etc.scene import get_scene

    if sensorfilter not in SENSORFILTER_FOCUS:
        raise ValueError(f"unknown sensorfilter {sensorfilter!r}; valid: "
                         f"{', '.join(sorted(SENSORFILTER_FOCUS))}")
    scene = get_scene("flat", mag=mag, magsys=magsys, flat_unit=flat_unit)
    sim = Simulation.from_sensorfilter(sensorfilter, scene)
    return float(sim._count_rate_components()["source_rate_total"])


@lru_cache(maxsize=None)
def _rate_per_jy(sensorfilter):
    """e-/s of a source flat in F_nu with 1 Jy (AB magnitude from astropy)."""
    return _source_rate(sensorfilter, "fnu", float((1 * u.Jy).to_value(u.ABmag)), "abmag")


@lru_cache(maxsize=None)
def _rate_per_flam(sensorfilter):
    """e-/s of a source flat in F_lambda with 1 erg/s/cm2/A (STmag from astropy)."""
    return _source_rate(sensorfilter, "flam", float((1 * _FLAM).to_value(u.STmag)),
                        u.STmag)


def flux_to_rate(unit, sensorfilter, pixel_area_arcsec2=None):
    """e-/s per pixel per unit of pixel value, for `unit` in WCC `sensorfilter`.

    `pixel_area_arcsec2` (the input image's pixel area) is required for surface
    brightness units. Raises ValueError for unsupported units (e.g. a line flux
    erg/s/cm2, or adu) and unknown sensorfilters.
    """
    unit = u.Unit(unit)
    for base, rate in ((u.Jy, _rate_per_jy), (_FLAM, _rate_per_flam), (_COUNTS, None)):
        if unit.is_equivalent(base / u.pix):
            per_pixel = (1 * unit).to_value(base / u.pix)
        elif unit.is_equivalent(base):
            per_pixel = (1 * unit).to_value(base)           # plain unit: per pixel
        elif unit.is_equivalent(base / u.arcsec**2):
            if pixel_area_arcsec2 is None:
                raise ValueError(f"surface brightness {unit} needs the pixel area "
                                 "(pixel_area_arcsec2)")
            per_pixel = (1 * unit).to_value(base / u.arcsec**2) * pixel_area_arcsec2
        else:
            continue
        return per_pixel * (rate(sensorfilter) if rate is not None else 1.0)
    raise ValueError(f"unsupported pixel unit {unit}; use {_SUPPORTED}")


@lru_cache(maxsize=None)
def _band(sensorfilter):
    """(pivot, top-hat-equivalent width) of the WCC bandpass, as astropy Quantities."""
    from .detectors import make_base_simulation

    _rate_per_jy(sensorfilter)                   # validates the sensorfilter
    bp = make_base_simulation(sensorfilter).sensor.bandpass
    return bp.pivot().to(u.nm), bp.rectwidth().to(u.nm)


def band_fluxes(rate_e_s, sensorfilter):
    """Flat-F_nu flux in WCC `sensorfilter` that gives `rate_e_s` [e-/s].

    Returns a dict: f_nu_jy [Jy] (band-mean F_nu), f_lambda [erg/s/cm2/A] at the
    pivot, f_band [erg/s/cm2] = f_lambda x top-hat-equivalent width (the
    band-integrated flux), pivot_nm and width_nm.
    """
    pivot, width = _band(sensorfilter)
    f_nu = float(rate_e_s) / _rate_per_jy(sensorfilter) * u.Jy
    f_lam = f_nu.to(_FLAM, equivalencies=u.spectral_density(pivot))
    return {"f_nu_jy": float(f_nu.value), "f_lambda": float(f_lam.value),
            "f_band": float((f_lam * width).to_value(u.erg / u.s / u.cm**2)),
            "pivot_nm": float(pivot.value), "width_nm": float(width.value)}
