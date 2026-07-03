"""Differential light curve from the per-star, per-frame measurement table."""

import numpy as np
from astropy.table import Table


def build_lightcurve(measurements):
    """One row per frame: target, reference-ensemble, and relative fluxes.

    The ensemble is the straight sum of the reference fluxes;
    rel_flux = flux_target / flux_ens with errors propagated, and
    rel_flux_norm is rel_flux divided by its median over frames.
    """
    rows = []
    for k in np.unique(measurements["frame"]):
        mk = measurements[measurements["frame"] == k]
        target = mk[mk["role"] == "target"][0]
        refs = mk[mk["role"] == "ref"]
        flux_t = float(target["flux_e"])
        err_t = float(target["flux_err_e"])
        flux_ens = float(np.sum(refs["flux_e"]))
        err_ens = float(np.sqrt(np.sum(np.asarray(refs["flux_err_e"]) ** 2)))
        rel = flux_t / flux_ens
        rel_err = abs(rel) * np.hypot(err_t / flux_t, err_ens / flux_ens)
        rows.append(
            {
                "frame": int(k),
                "time": float(target["time"]),
                "flux_target": flux_t,
                "flux_target_err": err_t,
                "flux_ens": flux_ens,
                "flux_ens_err": err_ens,
                "rel_flux": rel,
                "rel_flux_err": rel_err,
                "n_ref": len(refs),
                "flags": int(np.bitwise_or.reduce(np.asarray(mk["flags"]))),
            }
        )
    lc = Table(rows)
    median = float(np.nanmedian(lc["rel_flux"]))
    lc["rel_flux_norm"] = lc["rel_flux"] / median
    lc["rel_flux_norm_err"] = lc["rel_flux_err"] / median
    return lc
