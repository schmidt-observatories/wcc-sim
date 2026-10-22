"""User-supplied sky images as extended sources (SkyImageComponent).

A SkyImageComponent wraps a validated ImageInput (fitsimage.py): pixel values in
physical units (BUNIT), a celestial WCS and an optional BAND. Its rate_map
converts the image to e-/s for a WCC filter (fluxrate.flux_to_rate, flat
spectrum; warns if BAND differs) and resamples it flux-conservingly onto the
detector sub-grid (resample.py). Pointing, offset and rotation are not part of
the component: they are in the detector WCS (detectors.detector_frame).

Not to be confused with extended.ImageComponent (an AB-calibrated image, e.g. a
PHANGS-HST mosaic, for simulate_field's extended_sources): that one is
reprojected bilinearly onto the frame and convolved at detector resolution;
SkyImageComponent is rendered on the sub-grid (render.render_subgrid), which
keeps compact structure in images finer than the detector pixel.
"""

from dataclasses import dataclass

from .extended import ExtendedComponent
from .fitsimage import ImageInput
from .resample import choose_subgrid, footprint, resample_to_subgrid


@dataclass(frozen=True)
class SkyImageComponent(ExtendedComponent):
    """A sky image (ImageInput) as an extended source."""

    image: ImageInput

    kind = "image"

    @classmethod
    def from_fits(cls, image, ext=0, flux_unit=None, *, label=None):
        """From a FITS path, HDUList, HDU or (data, header); label defaults to the source."""
        img = ImageInput.from_fits(image, ext=ext, flux_unit=flux_unit)
        return cls(img, label=label if label is not None else img.source)

    @property
    def centre_radec(self):
        return self.image.centre_radec

    @property
    def band(self):
        return self.image.band

    def rate_map(self, det_wcs, det_shape, plate_scale_mas, sensorfilter, k=None):
        """e-/s per sub-pixel on the detector sub-grid (Subgrid), or None if off-frame.

        The image is converted to e-/s per input pixel first, then resampled, so
        Subgrid.data is in e-/s per sub-pixel and Subgrid.total_in is the whole
        image's e-/s before clipping to the detector.
        """
        rate = self.image.rate_image(sensorfilter)
        k = choose_subgrid(self.image.pixel_scale_mas, plate_scale_mas, override=k)
        return resample_to_subgrid(rate, self.image.wcs, det_wcs, det_shape, k)

    def footprint(self, det_wcs, det_shape):
        return footprint(self.image.wcs, self.image.shape, det_wcs, det_shape)

    def to_record(self):
        ra, dec = self.centre_radec
        ny, nx = self.image.shape
        # bunit: the file's BUNIT as written (unit: astropy's parsed form); a file
        # without BUNIT (unit from flux_unit=) gets the parsed unit
        bunit = str(self.image.header.get("BUNIT", "")).strip() or self.image.unit.to_string()
        return super().to_record() | {
            "source": self.image.source, "bunit": bunit, "unit": self.image.unit.to_string(),
            "band": self.band or "", "centre_ra": ra, "centre_dec": dec,
            "nx": nx, "ny": ny, "pixel_scale_mas": self.image.pixel_scale_mas}
