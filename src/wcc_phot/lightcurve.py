"""Differential light curve from the per-star, per-frame measurement table."""

import warnings

import numpy as np
from astropy.table import Table

from .flags import FLAG_NOFLUX


def ensemble_ratio(flux_t, err_t, ref_flux, ref_err):
    """Target / reference-ensemble flux for one frame.

    Returns (rel, rel_err, flux_ens, err_ens, ok). The ensemble is the sum
    of the given reference fluxes. The error uses
    var(T/R) = var(T)/R^2 + T^2 var(R)/R^4, which never divides by the
    target flux, so a zero target is a valid measurement of zero rather
    than a ZeroDivisionError. `ok` is False, and rel/rel_err are NaN, when
    the target flux is not finite or the ensemble is not finite or <= 0.
    """
    flux_t, err_t = float(flux_t), float(err_t)
    ref_flux = np.asarray(ref_flux, dtype=float)
    ref_err = np.asarray(ref_err, dtype=float)
    flux_ens = float(np.sum(ref_flux))
    err_ens = float(np.sqrt(np.sum(ref_err**2)))
    ok = bool(np.isfinite(flux_t) and np.isfinite(flux_ens) and flux_ens > 0)
    if not ok:
        return np.nan, np.nan, flux_ens, err_ens, False
    rel = flux_t / flux_ens
    rel_err = np.sqrt(err_t**2 / flux_ens**2 + flux_t**2 * err_ens**2 / flux_ens**4)
    return rel, float(rel_err), flux_ens, err_ens, True


def usable_references(measurements):
    """Star indices of the references with flags == 0 in every frame.

    A fixed subset keeps the ensemble baseline stable: dropping a reference
    from some frames only would step the light curve by that star's share
    of the ensemble, an artificial transit. If no reference is clean in
    every frame, all of them are used and a warning says so.
    """
    refs = measurements[measurements["role"] == "ref"]
    stars = np.unique(np.asarray(refs["star"]))
    clean = [
        int(s) for s in stars
        if not np.any(np.asarray(refs["flags"][refs["star"] == s]) != 0)
    ]
    if not clean and len(stars):
        warnings.warn(
            "every reference star is flagged in at least one frame; using "
            "all of them in the ensemble",
            UserWarning,
        )
        clean = [int(s) for s in stars]
    return clean


def build_lightcurve(measurements, refs=None):
    """One row per frame: target, reference-ensemble, and relative fluxes.

    The ensemble is the straight sum of the usable reference fluxes
    (`refs`, default `usable_references`); rel_flux = flux_target /
    flux_ens with errors propagated (see `ensemble_ratio`), and
    rel_flux_norm is rel_flux divided by its median over the frames where
    it is defined. A frame whose ratio is undefined keeps its row with NaN
    fluxes and FLAG_NOFLUX set instead of aborting the series. `flags` is
    the target's own flags plus FLAG_NOFLUX; reference flags are reported
    through the selection, not here.
    """
    if refs is None:
        refs = usable_references(measurements)
    refs = np.asarray(refs, dtype=int)
    rows = []
    for k in np.unique(measurements["frame"]):
        mk = measurements[measurements["frame"] == k]
        target = mk[mk["role"] == "target"][0]
        ref_rows = mk[(mk["role"] == "ref") & np.isin(mk["star"], refs)]
        rel, rel_err, flux_ens, err_ens, ok = ensemble_ratio(
            target["flux_e"], target["flux_err_e"],
            ref_rows["flux_e"], ref_rows["flux_err_e"],
        )
        rows.append(
            {
                "frame": int(k),
                "time": float(target["time"]),
                "flux_target": float(target["flux_e"]),
                "flux_target_err": float(target["flux_err_e"]),
                "flux_ens": flux_ens,
                "flux_ens_err": err_ens,
                "rel_flux": rel,
                "rel_flux_err": rel_err,
                "n_ref": len(ref_rows),
                "flags": int(target["flags"]) | (0 if ok else FLAG_NOFLUX),
            }
        )
    lc = Table(rows)
    rel = np.asarray(lc["rel_flux"], dtype=float)
    median = float(np.nanmedian(rel)) if np.isfinite(rel).any() else np.nan
    if not (np.isfinite(median) and median != 0):
        median = np.nan
    lc["rel_flux_norm"] = rel / median
    lc["rel_flux_norm_err"] = np.asarray(lc["rel_flux_err"], dtype=float) / median
    return lc
