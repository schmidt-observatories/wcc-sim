"""Target matching and best-N reference-star selection from a frame catalog."""

import warnings

import numpy as np
from astropy.coordinates import angular_separation


def pick_target(catalog, target, tol_arcsec=2.0):
    """Catalog row index of the target.

    `target` is a Gaia source_id (int) or an (ra, dec) pair in degrees;
    coordinate targets must match a catalog entry within `tol_arcsec`.
    """
    if len(catalog) == 0:
        raise ValueError("empty catalog")
    if not isinstance(target, (tuple, list, np.ndarray)):
        matches = np.flatnonzero(
            np.asarray(catalog["source_id"], dtype=np.int64) == int(target)
        )
        if matches.size == 0:
            raise ValueError(f"source_id {target} not in catalog")
        return int(matches[0])
    ra, dec = (float(v) for v in target)
    cat_ra = np.asarray(catalog["ra"], dtype=float)
    cat_dec = np.asarray(catalog["dec"], dtype=float)
    # Spherical, not a flat RA difference: (0.0001, 0) and (359.9999, 0)
    # are 0.72 arcsec apart. Rows without a finite position never match.
    sep = np.degrees(angular_separation(
        np.radians(cat_ra), np.radians(cat_dec), np.radians(ra), np.radians(dec)
    )) * 3600.0
    sep = np.where(np.isfinite(sep), sep, np.inf)
    i = int(np.argmin(sep))
    if not sep[i] <= tol_arcsec:
        raise ValueError(
            f"no catalog source within {tol_arcsec}\" of ({ra}, {dec}); "
            f"nearest is {sep[i]:.2f}\" away"
        )
    return i


def pick_references(
    catalog, target_idx, shape, geom, n_ref=10, iso_dmag=1.0, iso_radius=None
):
    """Indices of the best `n_ref` comparison stars.

    Criteria: on-image, unsaturated, annulus + centroid box fully inside the
    frame, and isolated (no catalog neighbor within `iso_radius` px brighter
    than G + iso_dmag — saturated/off-frame neighbors count too). Survivors
    are ranked by |G - G_target|, photometrically most similar first.
    """
    ny, nx = shape
    if iso_radius is None:
        iso_radius = geom.r_out
    g = np.asarray(catalog["phot_g_mean_mag"], dtype=float)
    x = np.asarray(catalog["x"], dtype=float)
    y = np.asarray(catalog["y"], dtype=float)

    margin = max(geom.r_out, geom.centroid_box / 2)
    ok = np.asarray(catalog["in_image"], dtype=bool).copy()
    ok &= ~np.asarray(catalog["saturated"], dtype=bool)
    ok &= (x >= margin) & (x <= nx - 1 - margin)
    ok &= (y >= margin) & (y <= ny - 1 - margin)
    ok &= np.isfinite(g)
    ok[target_idx] = False

    for i in np.flatnonzero(ok):
        d = np.hypot(x - x[i], y - y[i])
        d[i] = np.inf
        if np.any((d < iso_radius) & (g < g[i] + iso_dmag)):
            ok[i] = False

    candidates = np.flatnonzero(ok)
    if candidates.size == 0:
        raise ValueError("no usable reference stars in the field")
    order = np.argsort(np.abs(g[candidates] - g[target_idx]))
    refs = candidates[order][:n_ref]
    if refs.size < n_ref:
        warnings.warn(
            f"only {refs.size} of {n_ref} requested reference stars usable",
            UserWarning,
        )
    return [int(i) for i in refs]
