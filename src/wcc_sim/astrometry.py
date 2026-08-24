"""Space-motion propagation and cross-matching, for merging sky catalogs.

Catalogs disagree on epoch -- Gaia DR3 is at J2016.0, Hipparcos at J1991.25
-- and on column names, so both are parameters here. Nothing in this module
knows which catalog it is looking at.
"""

import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

#: Stand-in distance for rows with no usable parallax. ERFA ignores a
#: distance this large and propagates proper motion alone, which is exactly
#: the wanted behaviour for an unmeasured parallax.
_FAR_PC = 1e6


def as_time(epoch):
    """A Julian year (2016.0) or anything Time accepts -> Time."""
    if isinstance(epoch, Time):
        return epoch
    return Time(f"J{float(epoch)}")


def _column(cat, name, default):
    """Column `name` as float with masked and non-finite entries filled."""
    if name is None or name not in cat.colnames:
        return np.full(len(cat), default, dtype=float)
    values = np.ma.masked_invalid(np.asarray(cat[name], dtype=float))
    return np.ma.filled(values, default)


def propagate(cat, from_epoch, to_epoch, ra="ra", dec="dec", pmra="pmra",
              pmdec="pmdec", parallax=None, rv=None):
    """Copy of `cat` with positions moved from one epoch to another.

    `pmra` is dRA/dt * cos(dec) in mas/yr, as Gaia and Hipparcos both
    tabulate it. Rows without proper motion stay where they are. When
    `parallax` and `rv` name columns, they add perspective acceleration --
    75 mas, 4.5 px, for alpha Cen over 35 years -- and are ignored where the
    parallax is not positive.
    """
    out = cat.copy()
    if not len(out):
        return out
    t0, t1 = as_time(from_epoch), as_time(to_epoch)
    if abs((t1 - t0).to_value(u.yr)) < 1e-9:
        return out

    plx = _column(out, parallax, 0.0)
    known = plx > 0.0
    distance = np.where(known, 1000.0 / np.where(known, plx, 1.0), _FAR_PC)
    coords = SkyCoord(
        ra=_column(out, ra, np.nan) * u.deg,
        dec=_column(out, dec, np.nan) * u.deg,
        pm_ra_cosdec=_column(out, pmra, 0.0) * u.mas / u.yr,
        pm_dec=_column(out, pmdec, 0.0) * u.mas / u.yr,
        distance=distance * u.pc,
        radial_velocity=np.where(known, _column(out, rv, 0.0), 0.0) * u.km / u.s,
        obstime=t0,
    )
    with warnings.catch_warnings():
        # _FAR_PC rows: ERFA says it ignored the distance, which is the point
        warnings.filterwarnings("ignore", message=".*distance overridden.*")
        moved = coords.apply_space_motion(new_obstime=t1)
    out[ra] = moved.ra.deg
    out[dec] = moved.dec.deg
    return out
