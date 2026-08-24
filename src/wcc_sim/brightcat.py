"""Bright-star supplement: the stars Gaia DR3 does not have.

Gaia's brightest source is G = 1.73 and it holds only 150 sources brighter
than G = 3, so the naked-eye stars whose scattered-light halo drives WCC
stray-light requirements are missing from it entirely -- alpha Cen A and B,
for instance, have no DR3 entry at all. This module fills that end from
XHIP (VizieR V/137D, Hipparcos plus radial velocities), normalizes the rows
into the Gaia-shaped table the rest of the pipeline expects, and merges them
by positional cross-match.
"""

import os
import warnings

import numpy as np
from astropy import units as u
from astropy.table import Table, vstack

from .astrometry import crossmatch, propagate
from .catalog import GAIA_EPOCH
from .starflux import bp_rp_for_spt, g_minus_v, spt_from_b_v

#: VizieR table: the Extended Hipparcos Compilation (Anderson & Francis 2012).
XHIP_CATALOG = "V/137D/XHIP"

XHIP_COLUMNS = [
    "HIP", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "Plx", "RV", "Vmag", "B-V",
    "SpType", "Comp",
]

#: Epoch of the XHIP positions. The columns are named RAJ2000/DEJ2000, but
#: J2000 there is the equinox, not the epoch: the values are byte-identical
#: to I/239/hip_main's RAICRS/DEICRS, which are documented as J1991.25.
#: Reading them as epoch J2000 puts alpha Cen 32 arcsec off.
XHIP_EPOCH = 1991.25

_DTYPES = {"HIP": np.int64, "SpType": "U32", "Comp": "U8"}


def _empty_table():
    return Table({c: np.array([], dtype=_DTYPES.get(c, float))
                  for c in XHIP_COLUMNS})


def _missing_columns(table):
    """The XHIP_COLUMNS `table` does not have.

    `to_gaia_like` indexes these by name, so anything short of the full set
    raises KeyError mid-simulation. Both the query and the cache check for
    them; the two checks must stay in step (see `catalog.query_gaia` for the
    Gaia twin -- deliberately duplicated, the signatures and cache keys
    differ too much to share).
    """
    return [c for c in XHIP_COLUMNS if c not in table.colnames]


def _run_query(ra_deg, dec_deg, radius_arcsec):
    """Synchronous VizieR cone search (network). Thin so tests can patch it."""
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    vizier = Vizier(columns=XHIP_COLUMNS, row_limit=-1)
    found = vizier.query_region(
        SkyCoord(ra_deg, dec_deg, unit="deg"),
        radius=radius_arcsec * u.arcsec,
        catalog=XHIP_CATALOG,
    )
    return found[0] if len(found) else _empty_table()


def query_bright(ra_deg, dec_deg, radius_arcsec, cache_dir=None):
    """XHIP rows within radius_arcsec of (ra, dec), cached like query_gaia.

    A failure of the service is not fatal: it warns and returns an empty
    table so the caller can carry on with Gaia alone.

    A response missing any of XHIP_COLUMNS is treated the same way: the
    broad `except` covers only the network call, and a KeyError out of
    `to_gaia_like` later would defeat it.

    An empty result is ambiguous on its own -- the WCC field is 162" x 108",
    so most pointings hold no XHIP row and an empty table is the *normal*
    outcome -- so the distinction is recorded on the returned table's
    `meta["bright_query_ok"]`: True whenever the service or the cache was
    actually consulted, False when the query failed. `merge` reads it, and
    reports "consulted, nothing to add" rather than "not filled".
    """
    cache_file = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        key = f"xhip_{ra_deg:.6f}_{dec_deg:+.6f}_{radius_arcsec:.1f}"
        cache_file = os.path.join(cache_dir, key + ".ecsv")
        if os.path.exists(cache_file):
            cached = Table.read(cache_file, format="ascii.ecsv")
            if not _missing_columns(cached):
                cached.meta["bright_query_ok"] = True
                return cached
            # written before XHIP_COLUMNS last grew: using it would raise
            # KeyError out of to_gaia_like. Re-query and overwrite instead.
    try:
        out = _run_query(ra_deg, dec_deg, radius_arcsec)
    except Exception as exc:  # network, service, or VOTable parse failure
        warnings.warn(
            f"bright-star query failed ({type(exc).__name__}: {exc}); "
            "continuing with Gaia only",
            UserWarning,
        )
        failed = _empty_table()
        failed.meta["bright_query_ok"] = False
        return failed
    # A schema surprise -- a requested column simply absent from the response
    # -- sails past the `except` above and raises KeyError out of
    # to_gaia_like instead, defeating the point of degrading gracefully.
    missing = _missing_columns(out)
    if missing:
        warnings.warn(
            f"bright-star query returned unexpected columns (missing "
            f"{missing}); continuing with Gaia only",
            UserWarning,
        )
        bad = _empty_table()
        bad.meta["bright_query_ok"] = False
        return bad
    out.meta["bright_query_ok"] = True
    if cache_file is not None:
        out.write(cache_file, format="ascii.ecsv", overwrite=True)
    return out


def _floats(cat, name, default=np.nan):
    """Column `name` as float, with masked and non-finite entries replaced.

    `np.asarray` on a MaskedColumn returns the data *under* the mask -- 0.0
    for a value astropy read back from the ECSV cache -- so a magnitude that
    was never measured would be used as if it had been. Read the mask itself.
    """
    col = cat[name]
    masked = np.ma.getmaskarray(np.ma.asarray(col))
    values = np.asarray(np.ma.getdata(col), dtype=float)
    return np.where(masked | ~np.isfinite(values), default, values)


def to_gaia_like(bright):
    """XHIP rows as a Gaia-shaped catalog, positions still at XHIP_EPOCH.

    The magnitude path is the point of this function. Hipparcos gives
    Johnson V and B-V; the rate model is normalized in Gaia G. So: pick the
    Pickles template whose synthetic B-V is nearest the star's, then
    G = V + (G-V) of that template. BP and RP are set from the same
    template's tabulated BP-RP, so `spt_from_bp_rp` independently recovers
    the type and `rates_for_catalog` needs no change at all. No empirical
    colour relation is involved.

    Rows without a usable V magnitude or a finite position are dropped with
    a warning, one per reason.
    """
    v = _floats(bright, "Vmag")
    ra = _floats(bright, "RAJ2000")
    dec = _floats(bright, "DEJ2000")
    # A non-finite position is not merely a useless row: it reaches
    # crossmatch -> match_to_catalog_sky, which raises "Matching coordinates
    # cannot contain NaN entries" instead of skipping it, so one bad row
    # would take down a run this module promises never to break.
    reasons = (
        (~np.isfinite(v),
         "no V magnitude: without one there is no count rate"),
        (~np.isfinite(ra) | ~np.isfinite(dec),
         "no usable position: a non-finite RA/Dec cannot be cross-matched"),
    )
    keep = np.ones(len(bright), dtype=bool)
    for bad, why in reasons:
        keep &= ~bad
        if bad.any():
            warnings.warn(
                f"{int(bad.sum())} bright-catalog row(s) have {why}; dropped",
                UserWarning,
            )
    rows = bright[keep]
    v = v[keep]
    spt = spt_from_b_v(_floats(rows, "B-V"))
    g = v + g_minus_v(spt)
    bp_rp = bp_rp_for_spt(spt)
    return Table({
        "source_id": -np.asarray(rows["HIP"], dtype=np.int64),
        "ra": ra[keep],
        "dec": dec[keep],
        "phot_g_mean_mag": g,
        "phot_bp_mean_mag": g + 0.5 * bp_rp,
        "phot_rp_mean_mag": g - 0.5 * bp_rp,
        "pmra": _floats(rows, "pmRA", 0.0),
        "pmdec": _floats(rows, "pmDE", 0.0),
        "parallax": _floats(rows, "Plx", 0.0),
        "radial_velocity": _floats(rows, "RV", 0.0),
        "spt": spt,
        "catalog": np.full(len(rows), "hipparcos"),
    })


#: Columns the merged table carries beyond the Gaia query's own.
_PROVENANCE = {"spt": "", "catalog": "gaia"}


def _with_provenance(gaia):
    """Gaia rows, plus the two columns the merged table needs.

    `spt` is empty because `rates_for_catalog` reads a non-empty value as an
    override; Gaia rows must keep deriving their type from BP-RP.
    """
    out = gaia.copy()
    for name, value in _PROVENANCE.items():
        if name not in out.colnames:
            out[name] = np.full(len(out), value)
    return out


def merge(gaia, bright, epoch=None, replace_mag=6.0, match_radius_arcsec=2.0,
          gaia_epoch=GAIA_EPOCH):
    """Gaia plus the bright rows it is missing, all at one epoch.

    Returns `(merged, info)`. Both catalogs are propagated to a common epoch
    *before* matching: Gaia is at J2016.0 and Hipparcos at J1991.25, and a
    3.7 arcsec/yr star is 92 arcsec from itself across that gap, so an
    un-propagated match would add it twice.

    Policy, per matched pair: brighter than `replace_mag` in G the bright row
    replaces the Gaia one, which is where DR3's saturation systematics live;
    fainter, Gaia wins and the duplicate is dropped. Unmatched bright rows
    are added -- the gap-filling case. Added rows are sorted brightest-first
    and prepended, so row 0 (the row the PSF report decomposes) is the
    brightest star in the field.

    `epoch=None` means the Gaia epoch, so Gaia positions do not move and an
    empty bright table gives back the input catalog untouched.

    `bright_catalog` in the returned info separates "consulted" from "not
    consulted", never "empty" from "non-empty": a successful query over a
    cone with no bright star reports `"hipparcos"` with
    `n_bright_added = 0`, and only a failed query reports `None`. The
    distinction rides on `bright.meta["bright_query_ok"]`, set by
    `query_bright`; a hand-built table without the key counts as a
    successful supply.
    """
    to_epoch = float(gaia_epoch if epoch is None else
                     (epoch.jyear if hasattr(epoch, "jyear") else epoch))
    consulted = bool(getattr(bright, "meta", {}).get("bright_query_ok", True))
    info = {"bright_catalog": "hipparcos" if consulted else None,
            "n_bright_added": 0, "n_bright_replaced": 0, "epoch": to_epoch}

    gaia_moved = propagate(gaia, gaia_epoch, to_epoch, parallax="parallax",
                           rv="radial_velocity")
    if not len(bright):
        return _with_provenance(gaia_moved), info

    rows = to_gaia_like(bright)
    if not len(rows):
        return _with_provenance(gaia_moved), info
    rows = propagate(rows, XHIP_EPOCH, to_epoch, parallax="parallax",
                     rv="radial_velocity")
    rows.sort("phot_g_mean_mag")

    idx_bright, idx_gaia = crossmatch(rows, gaia_moved, match_radius_arcsec)
    bright_g = np.asarray(rows["phot_g_mean_mag"], dtype=float)
    wins = bright_g[idx_bright] < float(replace_mag) if len(idx_bright) else \
        np.array([], dtype=bool)

    take_bright = np.ones(len(rows), dtype=bool)
    take_bright[idx_bright[~wins]] = False          # matched and faint: drop
    drop_gaia = np.zeros(len(gaia_moved), dtype=bool)
    drop_gaia[idx_gaia[wins]] = True                # matched and bright: replace

    info["bright_catalog"] = "hipparcos"
    info["n_bright_replaced"] = int(wins.sum())
    info["n_bright_added"] = int(take_bright.sum()) - int(wins.sum())
    merged = vstack(
        [rows[take_bright], _with_provenance(gaia_moved[~drop_gaia])],
        join_type="exact",
    )
    return merged, info
