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


# ------------------------------------------------- pupil-plane wavefront PSF --

from scipy.ndimage import gaussian_filter, shift as nd_shift


class WfePSF:
    """PSF from an unobscured circular pupil with a phase screen, on the fine grid.

    The FFT size is chosen so that one lambda/D spans the same number of fine
    pixels as in the ETC's Airy rendering, so these PSFs drop into
    :func:`aperture_fractions` unchanged. Phase screens are in nm of wavefront.
    Tip/tilt is projected out of every screen (registration is treated
    separately). The pupil edge is anti-aliased by 4x supersampling.
    """

    def __init__(self, inst, n_pup=128, stamp_npix=STAMP_NPIX, seed=0):
        self.inst = inst
        self.n_pup = n_pup
        fine_plate_mas = inst.plate_mas / OVERSAMPLE
        lam_over_d_mas = inst.wavelength_m / inst.diameter_m * 206264.806e3
        self.n_fft = int(round(lam_over_d_mas / fine_plate_mas * n_pup))
        self.n_fine = stamp_npix * OVERSAMPLE
        self.rng = np.random.default_rng(seed)
        ss = 4
        yy, xx = (np.mgrid[:n_pup * ss, :n_pup * ss] + 0.5) / (n_pup * ss) - 0.5
        fine_pupil = (np.hypot(xx, yy) <= 0.5).astype(float)
        self.pupil = fine_pupil.reshape(n_pup, ss, n_pup, ss).mean(axis=(1, 3))
        yy, xx = (np.mgrid[:n_pup, :n_pup] + 0.5) / n_pup - 0.5
        self.x, self.y = xx, yy                       # pupil coords in units of D
        self.inside = self.pupil > 0.5
        self.rho = np.hypot(xx, yy) / 0.5              # normalised radius
        self.theta = np.arctan2(yy, xx)

    # -- screens ----------------------------------------------------------------
    def _remove_ptt(self, screen):
        m = self.inside
        A = np.stack([np.ones(m.sum()), self.x[m], self.y[m]], axis=1)
        coef, *_ = np.linalg.lstsq(A, screen[m], rcond=None)
        out = screen - (coef[0] + coef[1] * self.x + coef[2] * self.y)
        return out * self.inside

    def rms_nm(self, screen):
        return float(np.sqrt(np.mean(screen[self.inside] ** 2)))

    def scaled(self, screen, rms_nm):
        return screen * (rms_nm / self.rms_nm(screen))

    def psd_screen(self, alpha=2.5, f_min=1.0, f_max=None, rms_nm=1.0):
        """Random screen with PSD ~ f^-alpha between f_min and f_max cycles/D."""
        n = self.n_pup
        fx = np.fft.fftfreq(n) * n                  # cycles per pupil width (= per D)
        f = np.hypot(fx[None, :], fx[:, None])
        f_max = n / 2 if f_max is None else f_max
        amp = np.where((f >= f_min) & (f <= f_max), np.maximum(f, 1e-9) ** (-alpha / 2), 0.0)
        phase = self.rng.uniform(0, 2 * np.pi, (n, n))
        screen = np.real(np.fft.ifft2(amp * np.exp(1j * phase))) * self.pupil
        return self.scaled(self._remove_ptt(screen), rms_nm)

    def zernike_screen(self, coeffs_nm):
        """Low-order screen: dict of {'focus','astig0','astig45','coma_x','coma_y'} -> RMS nm each."""
        r, t = self.rho, self.theta
        modes = {
            'focus': np.sqrt(3) * (2 * r ** 2 - 1),
            'astig0': np.sqrt(6) * r ** 2 * np.cos(2 * t),
            'astig45': np.sqrt(6) * r ** 2 * np.sin(2 * t),
            'coma_x': np.sqrt(8) * (3 * r ** 3 - 2 * r) * np.cos(t),
            'coma_y': np.sqrt(8) * (3 * r ** 3 - 2 * r) * np.sin(t),
        }
        screen = sum(float(a) * modes[k] for k, a in coeffs_nm.items()) * self.pupil
        return self._remove_ptt(screen)

    def strehl(self, screen):
        phi = 2 * np.pi * screen[self.inside] * 1e-9 / self.inst.wavelength_m
        return float(np.abs(np.mean(np.exp(1j * phi))) ** 2)

    def screen_for_strehl(self, target_S, alpha=2.5, f_min=1.0, f_max=None):
        """Static PSD screen scaled so that the exact pupil-average Strehl equals target_S."""
        screen = self.psd_screen(alpha, f_min, f_max, rms_nm=1.0)
        lo, hi = 0.0, 300.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.strehl(self.scaled(screen, mid)) > target_S:
                lo = mid
            else:
                hi = mid
        return self.scaled(screen, 0.5 * (lo + hi))

    # -- PSF --------------------------------------------------------------------
    def psf(self, screen_nm, jitter_mas=0.0):
        """Unit-normalised PSF on the fine grid, centred on fine pixel (n-1)/2."""
        phi = 2 * np.pi * screen_nm * 1e-9 / self.inst.wavelength_m
        E = np.zeros((self.n_fft, self.n_fft), dtype=complex)
        E[: self.n_pup, : self.n_pup] = self.pupil * np.exp(1j * phi)
        F = np.fft.fftshift(np.fft.fft2(E))
        I = np.abs(F) ** 2
        c = self.n_fft // 2
        h = (self.n_fine - 1) // 2
        psf = I[c - h: c + h + 1, c - h: c + h + 1]
        if jitter_mas > 0:
            psf = gaussian_filter(psf, jitter_mas / (self.inst.plate_mas / OVERSAMPLE), mode='nearest')
        # normalised on the stamp, as the ETC's Airy stamp is (the wings outside hold ~0.8%)
        return psf / psf.sum()


def azimuthal_mean_image(image):
    """Image of the azimuthal mean about the centre (a radial-profile fit with no free parameters)."""
    n = image.shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - c, yy - c)
    ri = np.rint(r).astype(int)
    prof = np.bincount(ri.ravel(), weights=image.ravel()) / np.bincount(ri.ravel())
    return prof[ri]


def speckle_floor(inst, psf_sci, psf_ref, sep_mas, r_ap_px=2.0, snr=5.0, radial_subtract=True,
                  n_angles=24, window_mas=None, n_window=7, planet_psf=None):
    """Contrast floor from the residual PSF_sci - PSF_ref, as a planet/star flux ratio.

    The residual is optionally cleaned of its azimuthal mean (what a radial-profile
    fit or an annulus-scaled reference removes). The floor is `snr` times the RMS of
    the residual aperture sum over `n_angles` position angles and `n_window`
    separations spanning one Airy FWHM, divided by the planet's enclosed fraction.
    With radial_subtract=True this is the asymmetric (speckle) term only.
    """
    if window_mas is None:
        window_mas = inst.fwhm_mas
    resid = psf_sci - psf_ref
    if radial_subtract:
        resid = resid - azimuthal_mean_image(resid)
    n = resid.shape[0]
    c = (n - 1) / 2.0
    xx, yy = _fine_grid(n)
    fine_plate = inst.plate_mas / OVERSAMPLE
    r_ap_fine = r_ap_px * OVERSAMPLE
    h = int(np.ceil(r_ap_fine)) + 1
    ref = psf_sci if planet_psf is None else planet_psf
    pfrac = float(ref[np.hypot(xx, yy) <= r_ap_fine].sum())
    seps = np.atleast_1d(sep_mas).astype(float)
    offs = np.linspace(-window_mas / 2, window_mas / 2, n_window)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    out = np.empty(seps.size)
    for i, s in enumerate(seps):
        vals = []
        for o in offs:
            d = max(s + o, r_ap_px * inst.plate_mas) / fine_plate
            for th in angles:
                x0, y0 = d * np.cos(th), d * np.sin(th)
                i0, j0 = int(round(y0 + c)), int(round(x0 + c))
                sl = (slice(i0 - h, i0 + h + 1), slice(j0 - h, j0 + h + 1))
                m = np.hypot(xx[sl] - x0, yy[sl] - y0) <= r_ap_fine
                vals.append(resid[sl][m].sum())
        out[i] = np.sqrt(np.mean(np.square(vals)))
    return snr * out / pfrac


def shifted(psf, dx_mas, inst, dy_mas=0.0):
    """PSF shifted by (dx, dy) mas on the fine grid (linear interpolation)."""
    fp = inst.plate_mas / OVERSAMPLE
    return nd_shift(psf, (dy_mas / fp, dx_mas / fp), order=1, mode='constant')


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
    # pupil-plane PSF: flat wavefront must reproduce the ETC Airy aperture light at 160 mas
    w = WfePSF(inst)
    flat = w.psf(np.zeros((w.n_pup, w.n_pup)), jitter_mas=10.0)
    assert abs(flat.sum() - 1) < 1e-3, flat.sum()
    assert np.unravel_index(flat.argmax(), flat.shape) == ((n - 1) // 2, (n - 1) // 2)
    s_fft, p_fft = aperture_fractions(inst, flat, [160.0, 300.0], 2.0)
    s_airy, p_airy = aperture_fractions(inst, psf, [160.0, 300.0], 2.0)
    assert np.all(np.abs(s_fft / s_airy - 1) < 0.05), (s_fft, s_airy)
    assert abs(p_fft / p_airy - 1) < 0.02, (p_fft, p_airy)
    stat = w.screen_for_strehl(0.822)
    assert abs(w.strehl(stat) - 0.822) < 1e-3 and 40 < w.rms_nm(stat) < 52, w.rms_nm(stat)
    # a symmetric residual is removed by the radial subtraction; a shifted one is not
    sym = speckle_floor(inst, inst.fine_psf(10.0), inst.fine_psf(12.0), [160.0], radial_subtract=True)
    sym0 = speckle_floor(inst, inst.fine_psf(10.0), inst.fine_psf(12.0), [160.0], radial_subtract=False)
    assert sym[0] < 0.3 * sym0[0], (sym, sym0)
    reg = speckle_floor(inst, psf, shifted(psf, 1.0, inst), [160.0])
    assert reg[0] > 0 and speckle_floor(inst, psf, psf, [160.0])[0] == 0
    print(f"WfePSF: n_fft {w.n_fft}, flat-wavefront aperture light vs ETC Airy at 160/300 mas: "
          f"{s_fft[0] / s_airy[0]:.3f}, {s_fft[1] / s_airy[1]:.3f}; S = 0.822 screen = {w.rms_nm(stat):.1f} nm RMS; "
          f"symmetric residual after radial subtraction: {sym[0] / sym0[0]:.3f} of before")
    print("hci_psf self-checks passed:", f"star {inst.star_rate:.4g} e-/s, line 1e-16 -> {r1:.3f} e-/s,",
          f"t_frame {cc['t_frame_s']:.2f} s, 5-sigma contrast at 200 mas in 1 h: {cc['contrast'][1]:.2e}")
