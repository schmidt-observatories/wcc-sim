"""simulate_image: one call from a sky image (FITS) to a simulated WCC exposure.

A thin wrapper: SkyImageComponent.from_fits -> detector_frame -> WCCObservation
-> expose. Fluxes are taken from the input image (its BUNIT); scale them
beforehand (e.g. normalise the model in the detector band). For
several images, or several exposures of one set-up, use WCCObservation directly.
"""

from .detectors import detector_frame
from .imagesource import SkyImageComponent
from .observation import WCCObservation


def simulate_image(image, *, ra=None, dec=None, offset_arcsec=(0.0, 0.0), pa=0.0, ext=0,
                   flux_unit=None, sensorfilter="zwo:r", focus=None, exptime=90.0,
                   n_reads=1, gain_mode=None, jitter_sigma_mas=None, wavelength_nm=None,
                   stamp_npix=None, wings=True, scatter=True, scatter_fraction=None,
                   subgrid=None, shape=None, add_noise=True, seed=None, output=None,
                   write_clean=True):
    """Simulate one WCC exposure of a sky image -> SimulatedField (catalogue empty).

    `image` is a FITS path, HDUList, HDU or (data, header): 2-D, celestial WCS,
    BUNIT (or `flux_unit`). The detector is centred on (`ra`, `dec`), default
    the image centre from its WCS, moved by `offset_arcsec` (dRA, dDec) and
    rotated by `pa` (deg E of N). `shape=(ny, nx)` gives a subarray (default
    the full detector); `subgrid` overrides the odd sub-grid factor k. The
    other arguments are those of simulate_field. The header records the
    pointing (PNTSRC, CEN_RA/DEC, OFFSRA/DEC), the total rate (IMGTOT) and the
    total flux in the detector band (FLUXJY, FLUXFLAM, FLUXBAND); the COMP
    extension describes the image. `output` writes the FITS file.
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
                         gain_mode=gain_mode, subgrid=subgrid)
    field = obs.expose(exptime=exptime, n_reads=n_reads, seed=seed, add_noise=add_noise)
    field.params["pointing_source"] = "image" if ra is None else "user"
    if output is not None:
        field.write(output, write_clean=write_clean)
    return field
