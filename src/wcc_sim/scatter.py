"""Scattered-light halo as an azimuthally averaged radial profile.

The Lazuli stray-light budget comes from a FRED ray trace of the telescope
(``26-0212_tele image plane, scatter rays, full area, center_450nm``), which
wcc_etc ships and combines with a diffraction core in
``wcc_etc.scatter_psf.make_total_psf``:

    total = (1 - f_scat) * core + f_scat * scatter / P_full

``f_scat`` is the *instrument-wide* scattered fraction (5.542e-3 for the
26-0212 run) -- a property of the optics, not of any stamp.

Why a 1D profile and not a 2D kernel
------------------------------------
The FRED grid has 0.4 mm cells, which at the WCC's 3.76 um pixels is 106
detector pixels per cell, and the map is azimuthally symmetric to 0.3-2%
rms at every radius from 1 mm to 130 mm. So the halo carries no azimuthal
information worth keeping and is intrinsically smooth on a ~100 px scale:
a radial profile reproduces it to within its own symmetry scatter, in ~10 kB
instead of the ~10^9 pixels a full-plane 2D kernel would need. Convolving
with such a kernel would also cost more than the rest of the simulation put
together, and would forfeit the sub-pixel star placement that
:func:`wcc_sim.render.add_star` gets from the oversampled core stamp.

The halo is therefore added as a second, additive term rather than spliced
onto the core: no join radius, no crossfade, and no amplitude step to
reconcile. It reaches to 138 mm = 36,800 detector pixels, well beyond the
11,500 px chip diagonal, so a star in one corner of the array still deposits
a correct halo in the opposite corner.

The profile keeps FRED's absolute surface brightness and is truncated at that
reach rather than renormalized to unit energy. Renormalizing would inflate
the halo by ~1/0.875, corrupting the far-field brightness that is the whole
reason for using the map; and extrapolating a power law past the edge invents
41% of a unit of flux the map does not contain. The 12.5% of the scattered
power that lies outside the modelled disc sits in the FRED map's long-axis
corners, 37,000+ px off-axis -- off any detector, so it is not deposited and
not counted.

The FRED run is monochromatic at 450 nm and the halo is treated as
achromatic here. Real surface-scatter BRDF falls roughly as lambda^-2, so
the halo is over-predicted in redder bands; ``WAVELEN`` is recorded in the
profile metadata so that can be revisited.
"""

import os
from dataclasses import dataclass, field

import numpy as np
from astropy.table import Table

#: Packaged azimuthal profile, built from the FRED map by
#: ``scripts/build_scatter_profile.py``. The 24 MB ``.fgd`` and the 75 MB
#: combined-PSF FITS are deliberately not in the repo: runtime needs neither.
PROFILE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "psfs", "lazuli_scatter_450nm_profile.ecsv"
)


@dataclass(frozen=True)
class ScatterHalo:
    """Azimuthally averaged scattered-light halo on detector pixels.

    Parameters
    ----------
    r_px : ndarray
        Ascending radii [detector px] at which the profile is tabulated.
    frac_per_px : ndarray
        PSF fraction per detector pixel at ``r_px`` -- already carrying the
        scattered fraction, so it is directly comparable with
        :meth:`wcc_sim.wings.WingModel.profile`.
    frac_total : float
        The instrument-wide scattered fraction this profile represents.
        Stored as metadata; :meth:`energy_beyond` integrates the table
        itself rather than trusting it.

    Inside ``r_px[0]`` the profile is held flat: the FRED cells are ~106
    detector pixels across, so the map genuinely does not resolve the halo
    near the core, and extrapolating a power law inward would invent a
    divergence. It does not matter -- the diffraction core outruns the halo
    by ~10^9 there.
    """

    r_px: np.ndarray
    frac_per_px: np.ndarray
    frac_total: float
    meta: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        r = np.asarray(self.r_px, dtype=float)
        p = np.asarray(self.frac_per_px, dtype=float)
        if r.ndim != 1 or r.shape != p.shape or r.size < 2:
            raise ValueError("r_px and frac_per_px must be 1D and the same length")
        if np.any(np.diff(r) <= 0):
            raise ValueError("r_px must be strictly increasing")
        if np.any(p <= 0) or not np.all(np.isfinite(p)):
            raise ValueError("frac_per_px must be positive and finite")
        object.__setattr__(self, "r_px", r)
        object.__setattr__(self, "frac_per_px", p)
        # Local log-log slope of each segment.
        object.__setattr__(
            self, "_slopes", np.diff(np.log(p)) / np.diff(np.log(r))
        )
        object.__setattr__(self, "_enc", self._enclosed_beyond_nodes())
        # Integer-radius lookup for the hot path in render._add_wing_halo.
        # Quantizing to whole pixels costs <=2% at the inner edge and ~0.1%
        # beyond r~1000 px -- inside the map's own azimuthal scatter.
        r_max = int(np.ceil(r[-1]))
        object.__setattr__(self, "_lut", self._eval(np.arange(r_max + 1, dtype=float)))

    # -- construction helpers ------------------------------------------------ #

    def _eval(self, r):
        """Log-log interpolation of the table: flat inside, zero past the edge."""
        r = np.asarray(r, dtype=float)
        out = np.zeros(r.shape, dtype=float)
        inner = r <= self.r_px[0]
        mid = ~inner & (r <= self.r_px[-1])
        out[inner] = self.frac_per_px[0]
        if mid.any():
            out[mid] = np.exp(
                np.interp(
                    np.log(r[mid]), np.log(self.r_px), np.log(self.frac_per_px)
                )
            )
        return out

    def _segment_energy(self, i, r0, r1):
        """``2 pi int_r0^r1 r p(r) dr`` inside tabulated segment(s) ``i``.

        Each segment is a power law in (r, p), so this is exact there rather
        than merely a trapezoid.
        """
        a = self._slopes[i]
        p_i = self.frac_per_px[i]
        r_i = self.r_px[i]
        k = a + 2.0
        return np.where(
            np.abs(k) > 1e-8,
            2.0 * np.pi * p_i * r_i ** (-a) * (r1**k - r0**k) / np.where(k, k, 1.0),
            2.0 * np.pi * p_i * r_i**2.0 * np.log(r1 / r0),
        )

    def _enclosed_beyond_nodes(self):
        """Energy outside each tabulated radius; zero past the modelled edge."""
        i = np.arange(self.r_px.size - 1)
        seg = self._segment_energy(i, self.r_px[:-1], self.r_px[1:])
        enc = np.zeros(self.r_px.size, dtype=float)
        enc[:-1] = np.cumsum(seg[::-1])[::-1]
        return enc

    # -- the WingModel-compatible interface --------------------------------- #

    def profile(self, r):
        """PSF fraction per detector pixel at radius r [px]; 0 past the edge."""
        r = np.asarray(r, dtype=float)
        idx = np.clip(np.rint(r).astype(np.int64), 0, self._lut.size - 1)
        return np.where(r > self.r_px[-1], 0.0, self._lut[idx])

    def energy_beyond(self, r):
        """PSF fraction of the halo lying outside radius r [px]."""
        r = float(r)
        if r >= self.r_px[-1]:
            return 0.0
        if r <= self.r_px[0]:
            # flat inner disc: the annulus r..r_px[0] plus everything beyond
            disc = np.pi * self.frac_per_px[0] * (self.r_px[0] ** 2 - r**2)
            return float(self._enc[0] + disc)
        # exact within the containing segment, then the tabulated remainder
        i = int(np.searchsorted(self.r_px, r, side="right") - 1)
        return float(
            self._segment_energy(i, r, self.r_px[i + 1]) + self._enc[i + 1]
        )

    def r_out(self, flux_e, floor_e):
        """Radius [px] where ``flux_e * profile(r)`` falls to ``floor_e``.

        Returns 0.0 when even the halo's peak is below the floor, so the
        caller simply draws nothing.
        """
        if flux_e <= 0.0 or floor_e <= 0.0:
            return 0.0
        target = floor_e / flux_e
        if target >= self.frac_per_px[0]:
            return 0.0
        if target <= self.frac_per_px[-1]:
            return float(self.r_px[-1])  # halo is drawn to its modelled edge
        # profile is monotone decreasing; invert on the reversed table
        return float(
            np.exp(
                np.interp(
                    np.log(target),
                    np.log(self.frac_per_px[::-1]),
                    np.log(self.r_px[::-1]),
                )
            )
        )

    # -- rescaling ----------------------------------------------------------- #

    def scaled(self, frac_total):
        """Same halo shape, renormalized to a different scattered fraction."""
        factor = float(frac_total) / self.frac_total
        return ScatterHalo(
            r_px=self.r_px,
            frac_per_px=self.frac_per_px * factor,
            frac_total=float(frac_total),
            meta=dict(self.meta),
        )


# --------------------------------------------------------------------------- #
# Building the profile from the FRED map                                       #
# --------------------------------------------------------------------------- #

#: FRED cell size [mm]; the innermost bin is one cell wide and holds the peak.
_FGD_CELL_MM = 0.4


def build_scatter_table(fgd_path=None, n_bins=300):
    """Azimuthally average a FRED ``.fgd`` stray-light map into a radial table.

    Reads the map through wcc_etc's own ``read_fgd`` / ``fgd_grid`` so the two
    packages cannot drift apart on the grid convention (FRED's axis limits are
    cell *edges*, so samples sit at cell centers).

    Only annuli that lie entirely inside the map are used. The map is
    rectangular (+/-335 x +/-140 mm), so averaging over partial annuli beyond
    the short half-axis and then multiplying by a full 2*pi*r biases the
    integral high -- by 13% when measured against FRED's exact 2D integral.

    Returns
    -------
    astropy.table.Table
        Columns ``r_mm`` and ``frac_per_mm2`` (halo fraction per mm^2,
        normalized so the profile integrates to 1 over the plane), with the
        provenance and the pre-normalization diagnostics in ``.meta``.
    """

    from wcc_etc.scatter_help import read_fgd
    from wcc_etc.scatter_psf import (
        DEFAULT_SCATTER_FGD,
        arcsec_per_mm,
        fgd_grid,
        fgd_integrated_power,
    )

    path = DEFAULT_SCATTER_FGD if fgd_path is None else fgd_path
    header, data = read_fgd(path)
    x_mm, y_mm = fgd_grid(header, np.asarray(data))
    d = np.asarray(data, dtype=float)
    d = np.where(d > 1e307, np.nan, d)  # FRED HOLE_VALUE
    if not np.isfinite(d).all():
        raise ValueError(f"{path} contains HOLE_VALUE cells; cannot average")

    dx = float(x_mm[1] - x_mm[0])
    dy = float(y_mm[1] - y_mm[0])
    p_full = float(d.sum() * dx * dy)
    p_header = fgd_integrated_power(path)
    if p_header is not None and abs(p_full / p_header - 1.0) > 1e-3:
        raise ValueError(
            f"integrated power {p_full:.6e} disagrees with the FRED header "
            f"{p_header:.6e}: the grid convention is wrong"
        )

    xx, yy = np.meshgrid(x_mm, y_mm)
    r = np.hypot(xx, yy).ravel()
    w = d.ravel()

    # Annuli fully inside the map, plus a central bin one FRED cell wide.
    r_max = min(float(np.abs(x_mm).max()), float(np.abs(y_mm).max()))
    edges = np.concatenate(
        [[0.0], np.geomspace(_FGD_CELL_MM, r_max, int(n_bins))]
    )
    idx = np.digitize(r, edges)
    n = edges.size + 1
    count = np.bincount(idx, minlength=n)
    irr_sum = np.bincount(idx, weights=w, minlength=n)
    r_sum = np.bincount(idx, weights=r, minlength=n)
    keep = np.zeros(n, dtype=bool)
    keep[1 : edges.size] = count[1 : edges.size] > 0  # drop under/overflow bins
    r_bin = r_sum[keep] / count[keep]
    irr_bin = irr_sum[keep] / count[keep]

    # Keep FRED's absolute surface brightness -- do NOT renormalize the shape.
    # The profile's own integral is checked against the map's exact 2D integral
    # over the same disc; they agree to ~1e-4, which is the real validation
    # that the azimuthal average and the units are right.
    shape = ScatterHalo(r_px=r_bin, frac_per_px=irr_bin / p_full, frac_total=1.0)
    f_disc_profile = shape.energy_beyond(0.0)
    f_disc_exact = float(
        d[np.hypot(xx, yy) < r_bin[-1]].sum() * dx * dy / p_full
    )

    table = Table({"r_mm": r_bin, "frac_per_mm2": irr_bin / p_full})
    # FRED quotes two normalizations and they differ by the mirror throughput:
    # "integrated power" is per watt *launched at the aperture*, while
    # "fraction of total" is per watt *reaching the image plane*. A unit-sum
    # PSF describes detected photons, so F_FOCAL is the one that belongs in
    # `total = (1 - f) * core + f * halo`.
    f_focal = _fgd_fraction_of_total(path)
    table.meta.update(
        {
            "SCATFILE": os.path.basename(path),
            "P_FULL": p_full,
            "F_FOCAL": f_focal,
            "THRUPUT": (p_full / f_focal) if f_focal else None,
            "IRRPEAK": float(d.max()),
            "FDISC": f_disc_exact,
            "FDISCPRF": float(f_disc_profile),
            "WAVELEN": 450.0,
            "CELLMM": dx,
            "RMAXMM": r_max,
            "ASPMM": float(arcsec_per_mm(3.065, 15.0)),
            "NBINS": int(r_bin.size),
            "COMMENT": (
                "azimuthally averaged FRED scatter halo, absolute surface "
                "brightness; frac_per_mm2 integrates to FDISC over r<RMAXMM"
            ),
        }
    )
    return table


def _fgd_fraction_of_total(path, max_lines=200):
    """FRED's ``fraction of total`` [fraction, not percent], or None.

    This is the scattered power divided by the power that reaches the image
    plane, so ``integrated power / fraction of total`` recovers the optical
    throughput (0.777 for the 26-0212 run -- four reflectors at ~94%).
    """
    import re

    pattern = re.compile(r"fraction of total:\s*([0-9.eE+-]+)\s*%")
    with open(path, errors="replace") as fh:
        for i, line in enumerate(fh):
            if i > max_lines or line.startswith("BeginData"):
                break
            m = pattern.search(line)
            if m:
                return float(m.group(1)) / 100.0
    return None


def load_scatter_table(path=None):
    """Read the packaged azimuthal scatter profile."""
    return Table.read(PROFILE_PATH if path is None else path, format="ascii.ecsv")


def scatter_halo(mm_per_pixel, frac_total=None, table=None):
    """A :class:`ScatterHalo` on the pixel grid of a detector.

    Parameters
    ----------
    mm_per_pixel : float
        Detector pixel pitch at the focal plane [mm]; sets the conversion from
        the profile's physical units to fraction per detector pixel.
    frac_total : float, optional
        Instrument-wide scattered fraction. Defaults to FRED's own integrated
        power for the map the profile was built from (5.542e-3).
    table : astropy.table.Table, optional
        A profile table; defaults to the packaged one.
    """
    table = load_scatter_table() if table is None else table
    mm_per_pixel = float(mm_per_pixel)
    if mm_per_pixel <= 0.0:
        raise ValueError(f"mm_per_pixel must be > 0, got {mm_per_pixel!r}")
    frac = float(table.meta["P_FULL"] if frac_total is None else frac_total)
    return ScatterHalo(
        r_px=np.asarray(table["r_mm"], dtype=float) / mm_per_pixel,
        frac_per_px=(
            np.asarray(table["frac_per_mm2"], dtype=float)
            * mm_per_pixel**2
            * frac
        ),
        frac_total=frac,
        meta=dict(table.meta),
    )


def halo_for_sensorfilter(sensorfilter, frac_total=None, table=None, sim=None):
    """A :class:`ScatterHalo` on the pixel grid of a wcc_etc sensorfilter."""
    from .detectors import get_geometry

    geom = get_geometry(sensorfilter, sim=sim)
    return scatter_halo(
        geom.pixel_size_um / 1000.0, frac_total=frac_total, table=table
    )
