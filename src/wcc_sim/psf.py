"""Render the WCC PSF (in-focus Airy or 1/2-wave defocus) on an oversampled grid.

wcc_etc PSFSource.render(ctx) returns a detector-sampled PSF; we get an
oversampled one by describing a finer detector to the context (pixel size and
plate scale divided by `oversample`, npix multiplied by it). Jitter blur is
applied by wcc_etc at the fine plate scale.
"""

from wcc_etc import DEFOCUS_1WAVE_PATH, DEFOCUS_2WAVE_PATH, AiryPSF, DefocusPSF
from wcc_etc.psfsim import DetectorPSFContext

# Detector-pixel stamp sizes (odd). Defocus Huygens data spans ~272 IMX px.
DEFAULT_STAMP = {0: 65, 1: 257, 2: 257}


def make_psf_source(focus):
    if focus == 0:
        return AiryPSF()
    if focus == 1:
        return DefocusPSF(DEFOCUS_1WAVE_PATH)
    if focus == 2:
        return DefocusPSF(DEFOCUS_2WAVE_PATH)
    raise ValueError(f"focus must be 0, 1, or 2 (waves of defocus), got {focus!r}")


def render_oversampled_psf(
    sim, focus, oversample=11, stamp_npix=None, jitter_sigma_mas=None
):
    """Normalized PSF on a (stamp_npix*oversample)^2 fine grid, centered."""
    source = make_psf_source(focus)
    if stamp_npix is None:
        stamp_npix = DEFAULT_STAMP[focus]
    if stamp_npix % 2 == 0:
        raise ValueError(f"stamp_npix must be odd, got {stamp_npix}")
    sensor, telescope = sim.sensor, sim.telescope
    if jitter_sigma_mas is None:
        jitter_sigma_mas = float(telescope.jitter_sigma.to("mas").value)
    plate_mas = (
        sensor.get_plate_scale(telescope).to("arcsec/pix").value * 1000.0
    )
    n_fine = stamp_npix * oversample
    ctx = DetectorPSFContext(
        npix=n_fine,
        pixel_size_um=float(sensor.pixel_size.value) / oversample,
        plate_scale_mas=plate_mas / oversample,
        wavelength_m=float(sensor.wavelength.to("m").value),
        diameter_m=float(telescope.diameter_primary.to("m").value),
        fnum=float(telescope.f_num),
        jitter_sigma_mas=jitter_sigma_mas,
        oversample=3,  # internal supersampling of the fine grid (Airy path)
    )
    return source.render(ctx)
