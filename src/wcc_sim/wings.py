"""Analytic PSF wing model extending the finite stamps to large radii.

The rendered stamps truncate the PSF at the stamp boundary, which shows up
as a square "postage stamp" edge around bright stars once the display
stretch reaches the wing level (~100 sigma above sky for a G=8 star with
the 129 px in-focus stamp). The wing beyond the stamp is smooth and follows
a power law -- r^-3 for the jitter-blurred Airy envelope, steeper for the
defocused Huygens PSFs -- so it can be fit from the outer annulus of the
stamp itself and evaluated directly on detector pixels, out to a per-star
radius where it drops below the noise floor.

These fits describe the diffraction wing only; scattered-light halos are
not in the PSF models yet. When scattered-light PSFs land, replace the
power-law fit here with the measured extended profile (same interface).
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
