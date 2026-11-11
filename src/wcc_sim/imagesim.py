"""simulate_image: one call from a sky image (FITS) to a simulated WCC exposure.

A thin wrapper: SkyImageComponent.from_fits -> detector_frame -> WCCObservation
-> expose. Fluxes are taken from the input image (its BUNIT); scale them
beforehand (e.g. normalise the model in the detector band). For
several images, or several exposures of one set-up, use WCCObservation directly.
"""

from .detectors import detector_frame
from .imagesource import SkyImageComponent
from .observation import DEFAULT_MAX_MEMORY_GB, WCCObservation


def simulate_image(image, *, ra=None, dec=None, offset_arcsec=(0.0, 0.0), pa=0.0, ext=0,
                   flux_unit=None, sensorfilter="zwo:r", focus=None, exptime=90.0,
                   n_reads=1, gain_mode=None, jitter_sigma_mas=None, wavelength_nm=None,
                   stamp_npix=None, wings=True, scatter=True, scatter_fraction=None,
                   subgrid=None, shape=None, add_noise=True, seed=None, output=None,
                   write_clean=True, max_memory_gb=DEFAULT_MAX_MEMORY_GB):
    """Simulate one WCC exposure of a sky image -> SimulatedField (catalogue empty).

    The image is resampled flux-conservingly onto a detector-aligned k x k
    sub-grid, PSF-convolved per sub-pixel phase, binned to detector pixels and
    passed through the noise model of simulate_field. The header records the
    pointing (PNTSRC, CEN_RA/DEC, OFFSRA/DEC), the total rate (IMGTOT) and the
    total flux in the detector band (FLUXJY, FLUXFLAM, FLUXBAND); the COMP
    extension describes the image.
    
    The input image must have the standard, celestial WCS information (TAN: CTYPE1/2,
    CRVAL1/2, CRPIX1/2 and CDELT1/2 or a CD/PC matrix) with finite, non-negative
    values. The Meaning of the pixel values must be given in the header by BUNIT 
    or in `flux_unit`. Supported units: flux density per pixel (Jy/pixel, mJy/pix, plain Jy), 
    surface brightness (Jy/arcsec2, MJy/sr), F_lambda (erg/(s cm2 Angstrom) [/pix | /arcsec2]),
    electron/s [/pix].


    Parameters
    ----------
    image : str, Path, HDUList, ImageHDU/PrimaryHDU or (data, header)
        The sky image, see above.
    ra, dec : float, optional
        Detector centre [deg, ICRS]. Default: image centre from input image's WCS header.
    offset_arcsec : (float, float), default (0, 0)
        Pointing offset (dRA, dDec) [arcsec], e.g., dither; applied before rotation `pa`.
    pa : float, default 0
        Detector position angle [deg, East of North].
    ext : int or str, default 0
        HDU index or name to read when `image` is a path or HDUList.
    flux_unit : str or astropy unit, optional
        Pixel unit if input image lacks BUNIT. Must agree with BUNIT if present.
    sensorfilter : str, default "zwo:r"
        Detector and filter, e.g. "zwo:halpha2", "qcmos:g".
    focus : int, optional
        Focus offset in waves: 0 (in focus, Airy PSF) or 1 / 2 (Zemax Huygens
        defocus PSFs). Defaults to the filter's value from wcc_etc.
    exptime : float, default 90
        Exposure time [s].
    n_reads : int, default 1
        Number of reads (frames), >= 1. Defines read-noise. Saturation is evaluated
        per frame, that is, exptime / n_reads.
    gain_mode : str, optional
        Readout gain (default for qCMOS is 32x photon-number-resolving mode, "low" = 1x).
        Ignored for the ZWO.
    jitter_sigma_mas : float, optional
        Gaussian pointing jitter, sigma [mas], convolved into the PSF. Default none.
    wavelength_nm : float, optional
        Wavelength [nm] of in-focus PSF. Default: filter's central
        wavelength. Affects PSF geometry only; ignored for focus != 0.
    stamp_npix : int, optional
        PSF stamp size [detector px, odd]; default 129 in focus, 257 defocused.
    wings : bool, default True
        Extend the PSF beyond the stamp with the analytic diffraction wing.
    scatter : bool, default True
        Add scattered-light halo (wcc_sim.scatter); requires `wings`.
        Note: halo is calculated only out to stamp border.
    scatter_fraction : float, optional
        Scattered fraction of the halo; default wcc_sim.scatter's F_FOCAL.
    subgrid : int, optional
        Sub-pixels per detector pixel per axis, k (positive odd integer). Default:
        detector / image pixel-scale ratio rounded up to odd (max 11).
    shape : (int, int), optional
        Detector (sub)array (ny, nx) centred on the pointing; default the full
        detector (ZWO 6380 x 9568 px, qCMOS 2304 x 4096 px). Image flux outside the
        array is dropped.
    add_noise : bool, default True
        Poisson and read noise.
    seed : int, optional
        Random seed; recorded in params["seed"] so simulations can be exactly repeated.
    output : str or Path, optional
        Write the multi-extension FITS file (SCI, SATMASK, CAT, CLEAN, COMP).
    write_clean : bool, default True
        Include noiseless CLEAN extension in `output`.
    max_memory_gb : float or None, default 4
        Limit [GB = 1e9 bytes] for the estimated peak memory of the resampling.
        None disables the check.

    Returns
    -------
    SimulatedField
        ADU image, e- image, saturation mask, empty star catalogue, `components`
        table (one row per image) and `params` (settings and derived values).

    Examples
    --------
    >>> field = simulate_image("test_polaris_im.fits", sensorfilter="zwo:halpha2",
    ...                        offset_arcsec=(0.5, -0.2), pa=30, exptime=300,
    ...                        n_reads=30, shape=(201, 201), seed=1,
    ...                        output="polaris_sim.fits")       # doctest: +SKIP
    """
    if (ra is None) != (dec is None):
        raise ValueError("give both ra and dec, or neither (image centre)")
    comp = SkyImageComponent.from_fits(image, ext=ext, flux_unit=flux_unit)
    centre = comp.centre_radec if ra is None else (float(ra), float(dec))
    frame = detector_frame(sensorfilter, centre, offset_arcsec=offset_arcsec, pa=pa,
                           shape=shape)
    obs = WCCObservation(comp, frame, focus=focus, jitter_sigma_mas=jitter_sigma_mas,
                         wavelength_nm=wavelength_nm, stamp_npix=stamp_npix, wings=wings,
                         scatter=scatter, scatter_fraction=scatter_fraction,
                         gain_mode=gain_mode, subgrid=subgrid,
                         max_memory_gb=max_memory_gb)
    field = obs.expose(exptime=exptime, n_reads=n_reads, seed=seed, add_noise=add_noise)
    field.params["pointing_source"] = "image" if ra is None else "user"
    if output is not None:
        field.write(output, write_clean=write_clean)
    return field
