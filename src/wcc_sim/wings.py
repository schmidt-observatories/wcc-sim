"""Analytic PSF wing model extending the finite stamps to large radii.

The rendered stamps truncate the PSF at the stamp boundary, which shows up
as a square "postage stamp" edge around bright stars once the display
stretch reaches the wing level (~100 sigma above sky for a G=8 star with
the 129 px in-focus stamp). The wing beyond the stamp is smooth and follows
a power law -- r^-3 for the jitter-blurred Airy envelope, steeper for the
defocused Huygens PSFs -- so it can be fit from the outer annulus of the
stamp itself and evaluated directly on detector pixels, out to a per-star
radius where it drops below the noise floor.

These fits describe the diffraction wing only. The measured scattered-light
halo lives in :mod:`wcc_sim.scatter` and is *added* to this diffraction wing
by :class:`CombinedWing` rather than replacing it -- the two are physically
separate terms of the same PSF, and beyond ~1000 px the scatter halo is the
larger of them by an order of magnitude.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WingModel:
    """Azimuthally averaged power-law wing: profile(r) = c * r**alpha.

    `profile` is the PSF fraction per detector pixel at radius r (px).
    Requires alpha < -2 so the total wing energy converges.
    """

    c: float
    alpha: float
    r_in: float  # px; stamp half-width the model was fit inside

    def profile(self, r):
        return self.c * np.asarray(r, dtype=float) ** self.alpha

    def r_out(self, flux_e, floor_e):
        """Radius (px) where flux_e * profile(r) falls to floor_e."""
        if flux_e <= 0.0 or floor_e <= 0.0:
            return float(self.r_in)
        r = (floor_e / (flux_e * self.c)) ** (1.0 / self.alpha)
        return max(float(r), float(self.r_in))

    def energy_beyond(self, r):
        """PSF fraction integrated over the plane outside radius r."""
        return 2.0 * np.pi * self.c * r ** (self.alpha + 2.0) / (-(self.alpha + 2.0))

    def flux_norm(self, r_stamp):
        """Divide star flux by this so stamp + drawn wing integrate to 1.

        The rendered stamp is normalized to sum 1, so it already claims all
        the energy; the wing drawn beyond it is extra, and this divides it
        back out.
        """
        return 1.0 + self.energy_beyond(r_stamp)


def fit_wing_model(stamp, r_fit=(0.65, 0.98)):
    """Fit a WingModel to the outer annulus of a detector-sampled PSF stamp.

    `r_fit` bounds the fit annulus as fractions of the stamp half-width;
    the default range sits outside the PSF core for all three focus modes
    while staying clear of the stamp corners.
    """
    stamp = np.asarray(stamp, dtype=float)
    n = stamp.shape[0]
    half = n // 2
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - half, xx - half)
    m = (r >= r_fit[0] * half) & (r <= r_fit[1] * half) & (stamp > 0)
    if m.sum() < 16:
        raise ValueError(f"stamp ({n}x{n}) too small to fit a wing model")
    coeffs = np.column_stack([np.log(r[m]), np.ones(int(m.sum()))])
    (alpha, lnc), *_ = np.linalg.lstsq(coeffs, np.log(stamp[m]), rcond=None)
    if alpha >= -2.0:
        raise ValueError(
            f"fitted wing slope alpha={alpha:.2f} >= -2: wing energy diverges; "
            "stamp outer annulus is not in the power-law wing regime"
        )
    return WingModel(c=float(np.exp(lnc)), alpha=float(alpha), r_in=float(half))


@dataclass(frozen=True)
class CombinedWing:
    """Diffraction wing plus the measured scattered-light halo.

    The total PSF is ``(1 - f_scat) * core + f_scat * halo`` (the convention
    of ``wcc_etc.scatter_psf.make_total_psf``), so beyond the stamp the two
    terms simply add: the core's fitted power law scaled down by the
    scattered fraction, plus the halo at its absolute FRED brightness.

    Exposes the same interface as :class:`WingModel`, so
    :func:`wcc_sim.render.add_star` needs to know nothing about scatter.
    """

    core: WingModel
    halo: object  # wcc_sim.scatter.ScatterHalo

    @property
    def core_scale(self):
        """Weight on the diffraction term: 1 - the scattered fraction."""
        return 1.0 - self.halo.frac_total

    @property
    def r_in(self):
        return self.core.r_in

    def profile(self, r):
        return self.core_scale * self.core.profile(r) + self.halo.profile(r)

    def energy_beyond(self, r):
        return (
            self.core_scale * self.core.energy_beyond(r)
            + self.halo.energy_beyond(r)
        )

    def flux_norm(self, r_stamp):
        """Divide star flux by this so the drawn PSF carries the right energy.

        The scattered light falling outside the halo's modelled disc lands off
        the focal plane entirely, so it is neither drawn nor folded back into
        the star -- the target is ``1 - lost``, not 1.
        """
        lost = self.halo.frac_total - self.halo.energy_beyond(0.0)
        drawn = (
            self.core_scale * (1.0 + self.core.energy_beyond(r_stamp))
            + self.halo.energy_beyond(0.0)
        )
        return drawn / (1.0 - lost)

    def r_out(self, flux_e, floor_e):
        """Radius where the summed profile falls to ``floor_e``.

        Inverted on a log grid rather than analytically: the sum of a power
        law and a tabulated profile has no closed form, and this is evaluated
        once per star. The crossing is then interpolated between the
        bracketing grid points -- returning the grid point itself would clip
        the halo a fraction of a grid step early, which is a thin annulus of
        real flux at exactly the floor.
        """
        if flux_e <= 0.0 or floor_e <= 0.0:
            return float(self.r_in)
        r_edge = float(self.halo.r_px[-1])
        r = np.geomspace(max(float(self.r_in), 1.0), r_edge, 512)
        p = flux_e * self.profile(r)
        above = np.flatnonzero(p >= floor_e)
        if above.size == 0:
            return float(self.r_in)
        i = int(above[-1])
        if i + 1 >= r.size or p[i + 1] <= 0.0:
            return max(float(r[i]), float(self.r_in))
        # log-log interpolation of the profile between r[i] and r[i+1]
        t = (np.log(floor_e) - np.log(p[i])) / (np.log(p[i + 1]) - np.log(p[i]))
        r_cross = np.exp(np.log(r[i]) + t * (np.log(r[i + 1]) - np.log(r[i])))
        return max(float(r_cross), float(self.r_in))
