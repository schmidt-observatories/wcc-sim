"""PSF and contrast helpers shared by the H-alpha high-contrast notebooks.

Everything runs through the wcc_etc instrument model (via wcc_sim). The PSF is
the ETC's jitter-blurred Airy pattern, optionally degraded to a Strehl ratio
S < 1 with the core + Gaussian-halo model from
notebooks_scrap/jitter_strehl_sensitivity.ipynb, plus the FRED scattered-light
halo from wcc_sim.scatter as an additive radial term.

Contrast is evaluated with circular apertures on the 11x oversampled PSF
(so aperture sums are accurate at 1-3 detector-pixel radii) while the
noise counts detector pixels. Run `python hci_psf.py` for the self-checks.
"""

import numpy as np
from scipy.special import erf

from wcc_etc import AiryPSF, Simulation
from wcc_etc.psfsim import PSFSource, DetectorPSFContext, normalize_psf, grid_center
from wcc_etc.scene import get_scene
from wcc_sim.starflux import make_star_simulation, sky_and_dark_rates
from wcc_sim.detectors import get_geometry
from wcc_sim.render import bin_oversampled
from wcc_sim.scatter import halo_for_sensorfilter

HALO_K = 4.0                 # Strehl-halo FWHM in Airy FWHM (as in the jitter/Strehl study)
LAM_STREHL_REF_NM = 615.0    # the S = 0.8 requirement is taken to be referenced at r
HA_WAVE_A = 6563.0
OVERSAMPLE = 11
STAMP_NPIX = 81              # 1.37" square on IMX455: separations to 500 mas fit with margin

# ---------------------------------------------------------------- PSF model --

def airy_fwhm_mas(wavelength_m, diameter_m):
    return 1.028 * wavelength_m / diameter_m * 206264.806 * 1e3


def wfe_from_strehl(S, lam_nm):
    return lam_nm / (2 * np.pi) * np.sqrt(-np.log(S))


def strehl_from_wfe(wfe_nm, lam_nm):
    return np.exp(-(2 * np.pi * wfe_nm / lam_nm) ** 2)


def strehl_at(S_ref, lam_nm, lam_ref_nm=LAM_STREHL_REF_NM):
    """Strehl at lam_nm for the wavefront error that gives S_ref at lam_ref_nm."""
    return strehl_from_wfe(wfe_from_strehl(S_ref, lam_ref_nm), lam_nm)


def pixel_gaussian(n, sigma_pix, center=None):
    """Pixel-integrated circular Gaussian (exact per pixel via erf), sums to ~1."""
    c = (n - 1) / 2.0 if center is None else center
    cx, cy = (c, c) if np.ndim(c) == 0 else c
    e = np.arange(n + 1) - 0.5
    fx = 0.5 * erf((e - cx) / (np.sqrt(2) * sigma_pix))
    fy = 0.5 * erf((e - cy) / (np.sqrt(2) * sigma_pix))
    g = np.outer(np.diff(fy), np.diff(fx))
    return g / g.sum()


class StrehlPSF(PSFSource):
    """S * Airy_jit + (1 - S) * Gaussian halo, halo sigma = hypot(k FWHM/2.355, jitter)."""

    def __init__(self, strehl=1.0, halo_k=HALO_K):
        self.strehl, self.halo_k = float(strehl), float(halo_k)

    def cache_key(self):
        return ("StrehlPSF", round(self.strehl, 4), round(self.halo_k, 3))

    def render(self, ctx):
        core = AiryPSF().render(ctx)
        if self.strehl >= 1.0:
            return core
        sig_halo_mas = self.halo_k * airy_fwhm_mas(ctx.wavelength_m, ctx.diameter_m) / 2.3548
        sig_pix = np.hypot(sig_halo_mas, ctx.jitter_sigma_mas) / ctx.plate_scale_mas
        center = ctx.center if ctx.center is not None else grid_center(ctx.npix)
        halo = pixel_gaussian(ctx.npix, sig_pix, center)
        return normalize_psf(self.strehl * core + (1 - self.strehl) * halo)


class Instrument:
    """One sensorfilter's geometry, rates and PSF renderer for a host star."""

    def __init__(self, sensorfilter, host_spt="K7V", host_mag=11.7):
        self.sensorfilter = sensorfilter
        self.sim = make_star_simulation(host_spt, sensorfilter, mag=host_mag)
        self.host_spt, self.host_mag = host_spt, host_mag
        self.star_rate = float(self.sim._count_rate_components()["source_rate_total"])
        self.sky_rate, self.dark_rate = sky_and_dark_rates(self.sim)
        self.read_noise = float(self.sim.sensor.read_noise.value)
        self.well_depth = float(self.sim.sensor.meta["well_depth"])
        geom = get_geometry(sensorfilter, sim=self.sim)
        self.plate_mas = geom.plate_scale_mas
        self.pixel_um = geom.pixel_size_um
        self.wavelength_m = float(self.sim.effective_wavelength.to("m").value)
        self.diameter_m = float(self.sim.telescope.diameter_primary.to("m").value)
        self.fnum = float(self.sim.telescope.f_num)
        self.fwhm_mas = airy_fwhm_mas(self.wavelength_m, self.diameter_m)
        self.halo = halo_for_sensorfilter(sensorfilter, sim=self.sim)
        bp = self.sim.sensor.bandpass
        w = bp.waveset
        t = bp(w).value
        self.band_eqw_nm = float(np.trapezoid(t, w.value) / t.max() / 10.0)
        self._cache = {}

    def line_rate(self, flux_cgs, wave_A=HA_WAVE_A, fwhm_A=2.0):
        """Detector rate (e-/s) for an emission line of integrated flux [erg/s/cm^2]."""
        scene = get_scene("emission", mag=None, background=None,
                          lines=[dict(wave=wave_A, flux=float(flux_cgs), fwhm=fwhm_A)])
        sim = Simulation.from_sensorfilter(self.sensorfilter, scene)
        return float(sim._count_rate_components()["source_rate_total"])

    def fine_psf(self, jitter_mas, strehl=1.0, halo_k=HALO_K, stamp_npix=STAMP_NPIX):
        """Unit PSF on the oversampled grid; centre on fine pixel (n-1)/2 (odd n)."""
        key = (round(float(jitter_mas), 4), round(float(strehl), 4), halo_k, stamp_npix)
        if key not in self._cache:
            ctx = DetectorPSFContext(
                npix=stamp_npix * OVERSAMPLE,
                pixel_size_um=self.pixel_um / OVERSAMPLE,
                plate_scale_mas=self.plate_mas / OVERSAMPLE,
                wavelength_m=self.wavelength_m,
                diameter_m=self.diameter_m,
                fnum=self.fnum,
                jitter_sigma_mas=float(jitter_mas),
                oversample=3,
            )
            self._cache[key] = StrehlPSF(strehl, halo_k).render(ctx)
        return self._cache[key]

    def peak_fraction(self, jitter_mas, strehl=1.0, halo_k=HALO_K):
        """Fraction of the star's light in the brightest detector pixel."""
        return float(bin_oversampled(self.fine_psf(jitter_mas, strehl, halo_k), OVERSAMPLE).max())

    def frame_time(self, jitter_mas, strehl=1.0, well_fraction=0.5, halo_k=HALO_K):
        """Per-frame exposure [s] that fills the peak pixel to well_fraction of the well."""
        return well_fraction * self.well_depth / (self.star_rate * self.peak_fraction(jitter_mas, strehl, halo_k))


# ------------------------------------------------------------- apertures ----

def _fine_grid(n):
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[:n, :n]
    return xx - c, yy - c


def aperture_fractions(inst, fine_psf, sep_mas, r_ap_px, n_angles=12):
    """(star fraction in an off-axis aperture, planet enclosed fraction) per separation.

    The star fraction is the PSF fraction (core + Strehl halo + FRED halo) inside a
    circle of radius r_ap_px [detector px] centred sep_mas from the star, averaged
    over n_angles position angles. The planet fraction is the same aperture centred
    on the planet's own PSF (identical shape).
    """
    n = fine_psf.shape[0]
    c = (n - 1) / 2.0
    fine_plate = inst.plate_mas / OVERSAMPLE
    xx, yy = _fine_grid(n)
    r_fine = np.hypot(xx, yy)
    # FRED halo is tabulated per detector pixel: convert to per fine pixel
    halo_fine = inst.halo.profile(r_fine / OVERSAMPLE) / OVERSAMPLE ** 2
    total = fine_psf + halo_fine
    r_ap_fine = r_ap_px * OVERSAMPLE
    planet_frac = float(fine_psf[r_fine <= r_ap_fine].sum())
    seps = np.atleast_1d(sep_mas).astype(float)
    star_frac = np.empty(seps.size)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    h = int(np.ceil(r_ap_fine)) + 1

    def ap_sum(x0, y0):
        # sum inside the circle, evaluated on a small box around it
        i0, j0 = int(round(y0 + c)), int(round(x0 + c))
        if i0 - h < 0 or j0 - h < 0 or i0 + h + 1 > n or j0 + h + 1 > n:
            raise ValueError("aperture falls off the stamp; increase STAMP_NPIX")
        sl = (slice(i0 - h, i0 + h + 1), slice(j0 - h, j0 + h + 1))
        m = np.hypot(xx[sl] - x0, yy[sl] - y0) <= r_ap_fine
        return total[sl][m].sum()

    for i, s in enumerate(seps):
        d = s / fine_plate
        star_frac[i] = np.mean([ap_sum(d * np.cos(th), d * np.sin(th)) for th in angles])
    return star_frac, planet_frac


def contrast_curve(inst, jitter_mas, strehl, sep_mas, total_time_s, r_ap_px=(1.0, 1.5, 2.0, 2.5, 3.0),
                   well_fraction=0.5, halo_k=HALO_K, snr=5.0, planet_rate=None):
    """Photon-noise 5-sigma contrast (ideal PSF subtraction) and raw contrast vs separation.

    Returns a dict of arrays over sep_mas: 'contrast' (planet/star total flux at
    `snr`, aperture radius re-optimised per separation), 'raw' (planet flux that
    equals the stellar light in the same aperture), 'r_ap_px', plus the scalar
    frame time, number of frames and, if planet_rate [e-/s] is given, the SNR.
    """
    psf = inst.fine_psf(jitter_mas, strehl, halo_k)
    t_frame = inst.frame_time(jitter_mas, strehl, well_fraction, halo_k)
    n_frames = max(int(np.ceil(total_time_s / t_frame)), 1)
    T = total_time_s
    seps = np.atleast_1d(sep_mas).astype(float)
    best = np.full(seps.size, np.inf)
    raw = np.full(seps.size, np.inf)
    rbest = np.zeros(seps.size)
    snr_pl = np.zeros(seps.size)
    for r_ap in np.atleast_1d(r_ap_px):
        sfrac, pfrac = aperture_fractions(inst, psf, seps, r_ap)
        n_pix = np.pi * r_ap ** 2
        var = (inst.star_rate * sfrac * T + (inst.sky_rate + inst.dark_rate) * n_pix * T
               + n_frames * inst.read_noise ** 2 * n_pix)
        c = snr * np.sqrt(var) / (inst.star_rate * T * pfrac)
        better = c < best
        best[better], rbest[better] = c[better], r_ap
        raw[better] = (sfrac / pfrac)[better]
        if planet_rate is not None:
            sig = planet_rate * T * pfrac
            snr_pl[better] = (sig / np.sqrt(var + sig))[better]
    out = dict(sep_mas=seps, contrast=best, raw=raw, r_ap_px=rbest, t_frame_s=t_frame, n_frames=n_frames,
               peak_fraction=inst.peak_fraction(jitter_mas, strehl, halo_k))
    if planet_rate is not None:
        out["snr"] = snr_pl
    return out


def mismatch_floor(inst, jitter_sci, jitter_ref, strehl_sci, strehl_ref, sep_mas, r_ap_px=1.5, halo_k=HALO_K,
                   snr=5.0, window_mas=None, n_window=7):
    """Contrast floor from subtracting a reference PSF with different jitter/Strehl.

    Residual = |PSF_sci - PSF_ref| summed in the aperture; the planet must exceed
    `snr` times that residual. A pure PSF-shape quantity: independent of star
    brightness and exposure time. The residual of smeared Airy rings changes sign
    across a ring, so at a single separation it can pass through zero; the floor
    is therefore the RMS of the residual over n_window separations spanning
    +/- window_mas/2 (default: one Airy FWHM) around each requested separation.
    """
    if window_mas is None:
        window_mas = inst.fwhm_mas
    p_sci = inst.fine_psf(jitter_sci, strehl_sci, halo_k)
    p_ref = inst.fine_psf(jitter_ref, strehl_ref, halo_k)
    seps = np.atleast_1d(sep_mas).astype(float)
    offs = np.linspace(-window_mas / 2, window_mas / 2, n_window)
    grid = np.clip(seps[:, None] + offs[None, :], r_ap_px * inst.plate_mas, None)
    s_sci, pfrac = aperture_fractions(inst, p_sci, grid.ravel(), r_ap_px)
    s_ref, _ = aperture_fractions(inst, p_ref, grid.ravel(), r_ap_px)
    resid = np.sqrt(np.mean((s_sci - s_ref).reshape(grid.shape) ** 2, axis=1))
    return snr * resid / pfrac


# ------------------------------------------------------------ self-checks ---

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    inst = Instrument("zwo:halpha2")
    psf = inst.fine_psf(10.0)
    assert abs(psf.sum() - 1) < 1e-6
    n = psf.shape[0]
    assert np.unravel_index(psf.argmax(), psf.shape) == ((n - 1) // 2, (n - 1) // 2), "PSF not centred"
    psf8 = inst.fine_psf(10.0, 0.8)
    assert abs(psf8.sum() - 1) < 1e-6 and psf8.max() < psf.max()
    # line rate is linear in flux and independent of the bandwidth (line inside every H-alpha band)
    r1, r2 = inst.line_rate(1e-16), inst.line_rate(4e-16)
    assert abs(r2 / r1 - 4) < 1e-6
    r20 = Instrument("zwo:halpha20").line_rate(1e-16)
    assert abs(r20 / r1 - 1) < 0.05, (r1, r20)
    # planet enclosed fraction grows with aperture; star fraction falls with separation on average
    _, p1 = aperture_fractions(inst, psf, [180.0], 1.0)
    s, p3 = aperture_fractions(inst, psf, [100.0, 300.0], 3.0)
    assert p3 > p1 and s[0] > s[1]
    cc = contrast_curve(inst, 10.0, 1.0, [100.0, 200.0, 400.0], 3600.0)
    assert np.all(np.diff(cc["contrast"]) < 0), cc["contrast"]
    assert cc["t_frame_s"] > 0 and cc["n_frames"] >= 1
    assert np.all(mismatch_floor(inst, 10.0, 10.0, 1.0, 1.0, [180.0]) == 0)
    assert np.all(mismatch_floor(inst, 10.0, 12.0, 1.0, 1.0, [180.0]) > 0)
    print("hci_psf self-checks passed:", f"star {inst.star_rate:.4g} e-/s, line 1e-16 -> {r1:.3f} e-/s,",
          f"t_frame {cc['t_frame_s']:.2f} s, 5-sigma contrast at 200 mas in 1 h: {cc['contrast'][1]:.2e}")
