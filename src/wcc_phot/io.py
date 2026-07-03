"""Load WCC frames (wcc-sim FITS path, HDUList, or SimulatedField) for photometry."""

import os
import warnings
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

# meta key -> SCI header card (subset of wcc_sim.fitswriter._CARDS we need)
_META_CARDS = {
    "exptime": "EXPTIME",
    "n_reads": "NREADS",
    "gain": "GAIN",
    "read_noise": "RDNOISE",
    "sky_e_s": "SKYRATE",
    "dark_e_s": "DARK",
    "focus": "FOCUS",
    "sensorfilter": "SENSORF",
    "jitter_sigma_mas": "JITTER",
    "plate_scale_mas": "PLTSCL",
}


@dataclass
class Frame:
    """One WCC exposure: electron image + catalog + WCS + header metadata."""

    image_e: np.ndarray  # [e-] SCI * GAIN
    satmask: np.ndarray  # bool
    catalog: Table
    wcs: WCS
    meta: dict


def load_frame(source):
    """Build a Frame from a FITS path, an open HDUList, or a SimulatedField."""
    if isinstance(source, Frame):
        return source
    if hasattr(source, "image_e") and hasattr(source, "params"):
        meta = {key: source.params[key] for key in _META_CARDS}
        return Frame(
            image_e=np.asarray(source.image_e, dtype=float),
            satmask=np.asarray(source.saturation_mask, dtype=bool),
            catalog=source.catalog.copy(),
            wcs=source.wcs,
            meta=meta,
        )
    if isinstance(source, (str, os.PathLike)):
        with fits.open(source) as hdul:
            return _frame_from_hdulist(hdul)
    return _frame_from_hdulist(source)


def frame_meta(source):
    """Read a frame's header metadata without loading the pixel data.

    Same dispatch as load_frame (Frame / SimulatedField / FITS path /
    HDUList) but for a path it reads only the SCI header, so a frame
    series can be validated up front without holding every image in
    memory. Returns the same meta dict load_frame builds.
    """
    if isinstance(source, Frame):
        return source.meta
    if hasattr(source, "image_e") and hasattr(source, "params"):
        return {key: source.params[key] for key in _META_CARDS}
    if isinstance(source, (str, os.PathLike)):
        header = fits.getheader(source, "SCI")
    else:
        header = source["SCI"].header
    try:
        return {key: header[card] for key, card in _META_CARDS.items()}
    except KeyError as err:
        raise ValueError(
            f"missing header card {err} — is this a wcc-sim FITS file?"
        ) from None


def _frame_from_hdulist(hdul):
    header = hdul["SCI"].header
    try:
        meta = {key: header[card] for key, card in _META_CARDS.items()}
    except KeyError as err:
        raise ValueError(
            f"missing header card {err} — is this a wcc-sim FITS file?"
        ) from None
    image_e = np.asarray(hdul["SCI"].data, dtype=float) * float(meta["gain"])
    satmask = np.asarray(hdul["SATMASK"].data, dtype=bool)
    catalog = Table(hdul["CAT"].data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # FITSFixedWarning from DATE etc.
        wcs = WCS(header)
    return Frame(image_e, satmask, catalog, wcs, meta)
