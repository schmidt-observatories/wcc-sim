"""Validated input images: 2-D flux per pixel, a celestial WCS and an explicit unit.

`ImageInput.from_fits` is the entry point for user-supplied sky images (e.g.
model images converted from radiative-transfer output). Every `ImageInput` is checked on
construction, so code downstream (resampling, rendering) can rely on:

- `data`: 2-D float64, finite, no negative values, not all zero, read-only;
- `wcs`: a celestial WCS with exactly two axes, matching the data shape;
- `unit`: an astropy unit, never assumed -- from BUNIT, or given explicitly
  when the file has none.

Which units can be turned into detector counts is decided later (flux
conversion), not here.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

from .resample import pixel_scale_mas

_WCS_NEEDS = ("CTYPE1/2 = 'RA---TAN'/'DEC--TAN', CRVAL1/2, CRPIX1/2 and "
              "CDELT1/2 or a CD/PC matrix")


def _as_image_array(data, shape_in=None):
    """`data` as a validated 2-D float64 array (length-1 axes dropped)."""
    if data is None:
        raise ValueError("the image HDU has no data")
    arr = np.asarray(data, dtype=float)
    shape_in = arr.shape if shape_in is None else shape_in
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(
            f"image must be 2-D after dropping length-1 axes, got shape {arr.shape} "
            f"(from {shape_in}); select one plane first")
    bad = int(np.count_nonzero(~np.isfinite(arr)))
    if bad:
        raise ValueError(f"image contains {bad} non-finite pixel(s) (NaN or inf)")
    neg = int(np.count_nonzero(arr < 0))
    if neg:
        raise ValueError(
            f"image contains {neg} negative pixel(s) (minimum {arr.min():.6g}); an "
            "intensity image is expected, e.g. not a Stokes Q/U plane")
    if not np.any(arr):
        raise ValueError("image is all zero")
    return arr


def _read(image, ext):
    """(data, header copy, source label) from a path, HDUList, HDU or (data, header)."""
    if isinstance(image, (str, Path)):
        with fits.open(image) as hdul:
            hdu = hdul[ext]
            return hdu.data, hdu.header.copy(), Path(image).name
    if isinstance(image, fits.HDUList):
        hdu = image[ext]
        return hdu.data, hdu.header.copy(), "<HDU>"
    if hasattr(image, "header") and hasattr(image, "data"):
        return image.data, image.header.copy(), "<HDU>"
    if isinstance(image, (tuple, list)) and len(image) == 2:
        data, header = image
        if header is None:
            header = fits.Header()
        # fits.Header(header) would share the cards with the caller's header
        header = header.copy() if isinstance(header, fits.Header) else fits.Header(header)
        return data, header, "<array>"
    raise TypeError(
        "image must be a FITS path, an HDUList, an image HDU or a (data, header) "
        f"tuple, got {type(image).__name__}")


def read_image_fits(image, ext=0):
    """(2-D float64 data, header copy) of `image`; see `ImageInput` for the checks.

    `image` is a path, an HDUList (`ext` selects the HDU), an image HDU or a
    (data, header) tuple.
    """
    data, header, _ = _read(image, ext)
    return _as_image_array(data), header


def _check_wcs(wcs, ctypes=()):
    """Raise ValueError unless `wcs` is celestial with exactly two axes."""
    if not wcs.has_celestial:
        found = ", ".join(ctypes) if ctypes else "none"
        raise ValueError(
            f"no celestial WCS: need {_WCS_NEEDS}; found CTYPEs: {found}")
    if wcs.naxis != 2:
        raise ValueError(f"celestial WCS must have 2 axes, got {wcs.naxis}")


def image_wcs(header):
    """The celestial WCS of `header` (2 axes); ValueError says what is missing."""
    ctypes = [f"{k}={header[k]!r}" for k in header if k.startswith("CTYPE")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        full = WCS(header)
    if not full.has_celestial:
        alt = [c for c in ctypes if c.split("=")[0][-1].isalpha()
               and ("RA--" in c or "DEC-" in c)]
        hint = ""
        if alt:
            hint = (f"; the sky WCS is only an alternate WCS ({', '.join(alt)}), as in "
                    "POLARIS output -- convert it to a 2-D image with a celestial WCS first")
        found = ", ".join(c for c in ctypes if not c.split("=")[0][-1].isalpha()) or "none"
        raise ValueError(f"no celestial WCS: need {_WCS_NEEDS}; found CTYPEs: "
                         f"{found}{hint}")
    cel = full.celestial
    _check_wcs(cel)
    return cel


def _parse_unit(value):
    """An astropy unit from a unit or a FITS/generic unit string."""
    if isinstance(value, u.UnitBase):
        return value
    text = str(value).strip()
    for fmt in ("fits", None):
        try:
            return u.Unit(text, format=fmt) if fmt else u.Unit(text)
        except ValueError:
            continue
    raise ValueError(f"cannot parse unit {value!r}")


def image_unit(header, flux_unit=None):
    """Pixel-value unit: BUNIT, or `flux_unit` when the file has none. Never assumed.

    With both present they must agree; a file's BUNIT is not silently overridden.
    """
    bunit = str(header.get("BUNIT", "")).strip()
    if bunit:
        unit = _parse_unit(bunit)
        if flux_unit is not None and _parse_unit(flux_unit) != unit:
            raise ValueError(f"flux_unit {flux_unit!r} conflicts with the file's "
                             f"BUNIT {bunit!r}")
        return unit
    if flux_unit is None:
        raise ValueError("the image has no BUNIT keyword; give its pixel unit "
                         "explicitly with flux_unit= (e.g. 'Jy/pixel')")
    return _parse_unit(flux_unit)


@dataclass(frozen=True, eq=False)  # eq=False: == is identity, no field-wise array compare
class ImageInput:
    """A validated 2-D sky image: flux per pixel, celestial WCS, explicit unit."""

    data: np.ndarray            # 2-D float64, read-only
    wcs: WCS                    # celestial, 2 axes
    unit: u.UnitBase            # pixel-value unit
    header: fits.Header | None = None
    source: str = "<array>"     # file name, "<HDU>" or "<array>"

    def __post_init__(self):
        arr = _as_image_array(self.data).copy()
        arr.flags.writeable = False
        object.__setattr__(self, "data", arr)

        if not isinstance(self.wcs, WCS):
            raise TypeError(f"wcs must be an astropy WCS, got {type(self.wcs).__name__}")
        _check_wcs(self.wcs)
        ny, nx = arr.shape
        if self.wcs.array_shape is not None and tuple(self.wcs.array_shape) != arr.shape:
            raise ValueError(f"WCS array shape {tuple(self.wcs.array_shape)} does not "
                             f"match the data shape {arr.shape}")
        if self.header is not None and "NAXIS1" in self.header and "NAXIS2" in self.header:
            naxis = (int(self.header["NAXIS2"]), int(self.header["NAXIS1"]))
            if naxis != (ny, nx):
                raise ValueError(f"header NAXIS2/NAXIS1 = {naxis} does not match the "
                                 f"data shape {arr.shape}")

        object.__setattr__(self, "unit", _parse_unit(self.unit))

    @classmethod
    def from_fits(cls, image, ext=0, flux_unit=None):
        """Read, check and wrap `image` (path, HDUList, image HDU or (data, header)).

        `flux_unit` is required when the file has no BUNIT and must agree with
        BUNIT when both are present.
        """
        data, header, source = _read(image, ext)
        return cls(_as_image_array(data, np.shape(data)), image_wcs(header),
                   image_unit(header, flux_unit), header, source)

    @property
    def shape(self):
        """(ny, nx)."""
        return self.data.shape

    @property
    def pixel_scale_mas(self):
        """Mean pixel side [mas]."""
        return pixel_scale_mas(self.wcs)

    @property
    def total(self):
        """Sum of all pixels as a Quantity in `unit`."""
        return float(self.data.sum()) * self.unit

    @property
    def centre_radec(self):
        """(ra, dec) [deg] of the image centre."""
        ny, nx = self.shape
        ra, dec = self.wcs.pixel_to_world_values((nx - 1) / 2, (ny - 1) / 2)
        return float(ra), float(dec)

    @property
    def band(self):
        """The header's BAND keyword (band the pixel values refer to), or None."""
        if self.header is None:
            return None
        return str(self.header.get("BAND", "")).strip() or None

    def rate_scale(self, sensorfilter):
        """e-/s per pixel per unit of pixel value in WCC `sensorfilter` (flux_to_rate).

        Warns if the image says it was normalised in another band (BAND): the
        conversion is then a flat-spectrum transfer between the two bands.
        """
        from .fluxrate import flux_to_rate   # wcc_etc only when rates are needed

        if self.band is not None and self.band != sensorfilter:
            warnings.warn(f"image normalised in {self.band}, simulated in "
                          f"{sensorfilter}: flat-spectrum transfer between the bands",
                          UserWarning, stacklevel=2)
        area = (self.pixel_scale_mas / 1000.0) ** 2   # arcsec2, for surface brightness
        return flux_to_rate(self.unit, sensorfilter, pixel_area_arcsec2=area)

    def rate_image(self, sensorfilter):
        """The image in e-/s per (input) pixel for WCC `sensorfilter`."""
        return self.data * self.rate_scale(sensorfilter)
