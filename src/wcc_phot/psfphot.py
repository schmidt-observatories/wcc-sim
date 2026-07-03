"""PSF-model photometry using the wcc-sim PSF as a photutils ImagePSF."""

import warnings

import numpy as np
from astropy.table import Table
from photutils.background import LocalBackground
from photutils.psf import ImagePSF, PSFPhotometry

from .geometry import render_model_psf


def build_psf_model(meta, oversample=5, psf_os=None):
    """ImagePSF for the frame's sensorfilter/focus/jitter.

    The oversampled stamp is normalized so data.sum()/oversample^2 == 1,
    making the fitted `flux` parameter the total detector-integrated
    electrons.
    """
    if psf_os is None:
        psf_os = render_model_psf(meta, oversample=oversample)
    data = psf_os * (oversample**2 / psf_os.sum())
    return ImagePSF(data, oversampling=oversample)


def psf_photometry_frame(frame, x, y, flux0, model, geom):
    """Fit flux + position per star; returns (flux, flux_err, x_fit, y_fit).

    Initial positions/fluxes come from the centroid + aperture pass; the
    local background is estimated in the same annulus as aperture mode.
    """
    m = frame.meta
    var_read = int(m["n_reads"]) * float(m["read_noise"]) ** 2
    error = np.sqrt(np.clip(frame.image_e, 0, None) + var_read)

    phot = PSFPhotometry(
        model,
        fit_shape=(geom.fit_shape, geom.fit_shape),
        localbkg_estimator=LocalBackground(geom.r_in, geom.r_out),
        aperture_radius=geom.r_ap,
    )
    init = Table(
        {
            "x": np.atleast_1d(np.asarray(x, dtype=float)),
            "y": np.atleast_1d(np.asarray(y, dtype=float)),
            "flux": np.clip(np.atleast_1d(np.asarray(flux0, dtype=float)), 1.0, None),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # fitter chatter for flagged stars
        result = phot(frame.image_e, error=error, init_params=init)
    result.sort("id")
    return (
        np.asarray(result["flux_fit"], dtype=float),
        np.asarray(result["flux_err"], dtype=float),
        np.asarray(result["x_fit"], dtype=float),
        np.asarray(result["y_fit"], dtype=float),
    )
