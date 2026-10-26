"""An observation of extended sources with one WCC detector (WCCObservation).

Brings together what is observed (ExtendedComponents, e.g. SkyImageComponent),
where the detector looks (a DetectorFrame: WCS with pointing, offset and pa,
shape, pixel scale, filter) and how the instrument is set up (focus, jitter,
PSF wings, scattered-light halo). The PSF is built for whatever sub-grid
factor k a component needs, the same way simulate_field builds it, and the
components are rendered with render.render_subgrid and placed into the frame.

WCCObservation is the process (e-/s, no exposure time); expose() turns it into
one exposure, a SimulatedField (the product: ADU, noise, saturation, FITS
output), the same type simulate_field returns. One observation can be exposed
many times; PSF and rendering are done once.
"""

import secrets

import numpy as np
from astropy import units as u
from astropy.table import Table

from .detectors import get_geometry, make_base_simulation, saturation_level_e
from .fluxrate import band_fluxes, flux_to_rate
from .pipeline import SimulatedField, _core_wing
from .psf import DEFAULT_STAMP, render_oversampled_psf
from .render import add_noise_and_digitize, bin_oversampled, render_subgrid, wing_blend
from .scatter import halo_for_sensorfilter
from .starflux import sky_and_dark_rates
from .wings import CombinedWing, fit_wing_model

#: CAT columns of simulate_field; image fields write them with zero rows.
_CAT_COLUMNS = (("source_id", np.int64), ("ra", float), ("dec", float),
                ("phot_g_mean_mag", float), ("x", float), ("y", float), ("spt", "U8"),
                ("spt_fallback", bool), ("rate_e_s", float), ("psf_wavelength_nm", float),
                ("in_image", bool), ("saturated", bool))


def _common(table, column):
    """The column's value if all rows share it, 'mixed (see COMP)' if not, None if absent."""
    if column not in table.colnames or not len(table):
        return None
    values = sorted(set(str(v) for v in table[column]))
    return values[0] if len(values) == 1 else "mixed (see COMP)"


class WCCObservation:
    """Extended sources observed with a placed WCC detector."""

    def __init__(self, components, frame, *, focus=None, jitter_sigma_mas=None,
                 wavelength_nm=None, stamp_npix=None, wings=True, scatter=True,
                 scatter_fraction=None, gain_mode=None, subgrid=None):
        if not isinstance(components, (list, tuple)):
            components = [components]
        self.components = list(components)
        self.frame = frame
        self.jitter_sigma_mas = jitter_sigma_mas
        self.wavelength_nm = wavelength_nm
        self.wings = bool(wings)
        self.scatter = bool(scatter) and self.wings   # as in simulate_field
        self.scatter_fraction = scatter_fraction
        self.gain_mode = gain_mode
        self.subgrid = subgrid
        self.sim = make_base_simulation(frame.sensorfilter, gain_mode=gain_mode)
        self.geom = get_geometry(frame.sensorfilter, sim=self.sim)
        self.focus = self.geom.default_focus if focus is None else int(focus)
        self.stamp_npix = DEFAULT_STAMP[self.focus] if stamp_npix is None else stamp_npix
        self._psf_cache = {}
        self._halo = None
        self._subs = None    # per component: (rate_map Subgrid, rendered k = 1 Subgrid)

    # -- PSF ---------------------------------------------------------------------- #

    def _get_halo(self):
        if self.scatter and self._halo is None:
            self._halo = halo_for_sensorfilter(
                self.frame.sensorfilter, frac_total=self.scatter_fraction, sim=self.sim)
        return self._halo

    def psf(self, k):
        """(psf_k, wing, blend) for sub-grid factor k, built once per k.

        psf_k is the PSF oversampled by k (render_oversampled_psf); with wings, the
        wing is fitted to the binned stamp and combined with the scattered-light
        halo, exactly as in simulate_field.
        """
        if k not in self._psf_cache:
            wave_m = None if self.wavelength_nm is None else float(self.wavelength_nm) * 1e-9
            psf_k = render_oversampled_psf(
                self.sim, self.focus, oversample=k, stamp_npix=self.stamp_npix,
                jitter_sigma_mas=self.jitter_sigma_mas, wavelength_m=wave_m)
            wing = blend = None
            if self.wings:
                wing = fit_wing_model(bin_oversampled(psf_k, k))
                if self.scatter:
                    wing = CombinedWing(core=wing, halo=self._get_halo())
                blend = wing_blend(self.stamp_npix, wing)
            self._psf_cache[k] = (psf_k, wing, blend)
        return self._psf_cache[k]

    # -- rendering ---------------------------------------------------------------- #

    def _rendered(self):
        """Per component (rate_map Subgrid, rendered Subgrid), or (None, None); cached."""
        if self._subs is None:
            f = self.frame
            self._subs = []
            for comp in self.components:
                sub = comp.rate_map(f.wcs, f.shape, f.plate_scale_mas, f.sensorfilter,
                                    k=self.subgrid)
                cut = None if sub is None else render_subgrid(sub, *self.psf(sub.k))
                self._subs.append((sub, cut))
        return self._subs

    def rate_cutouts(self):
        """One detector-resolution Subgrid (k = 1, e-/s per pixel) per component.

        None for a component that misses the detector (rate_map warns).
        """
        return [cut for _, cut in self._rendered()]

    def rate_image(self):
        """Noiseless count-rate image [e-/s per detector pixel] of the whole frame."""
        image = np.zeros(self.frame.shape, dtype=np.float32)
        for cut in self.rate_cutouts():
            if cut is not None:
                self._add_cutout(image, cut)
        return image

    @staticmethod
    def _add_cutout(image, cut):
        """Add cut.data at detector pixel (cut.y0, cut.x0), clipped to the frame."""
        ny, nx = image.shape
        h, w = cut.data.shape
        y_lo, x_lo = max(cut.y0, 0), max(cut.x0, 0)
        y_hi, x_hi = min(cut.y0 + h, ny), min(cut.x0 + w, nx)
        if y_lo >= y_hi or x_lo >= x_hi:
            return 0.0
        part = cut.data[y_lo - cut.y0:y_hi - cut.y0, x_lo - cut.x0:x_hi - cut.x0]
        image[y_lo:y_hi, x_lo:x_hi] += part
        return float(part.sum())

    # -- records ------------------------------------------------------------------ #

    @property
    def params(self):
        """Settings of this observation, with simulate_field's key names."""
        f = self.frame
        ny, nx = f.shape
        centre = f.centre if f.centre is not None else f.pointing
        jitter = (float(self.jitter_sigma_mas) if self.jitter_sigma_mas is not None
                  else float(self.sim.telescope.jitter_sigma.to("mas").value))
        wave = (float(self.sim.sensor.wavelength.to("nm").value)
                if self.wavelength_nm is None else float(self.wavelength_nm))
        return {"ra": float(f.pointing[0]), "dec": float(f.pointing[1]), "pa": float(f.pa),
                "sensorfilter": f.sensorfilter, "focus": self.focus,
                "jitter_sigma_mas": jitter, "wavelength_nm": wave,
                "nx": int(nx), "ny": int(ny), "n_pixels": int(nx) * int(ny),
                "plate_scale_mas": float(f.plate_scale_mas),
                "stamp_npix": int(self.stamp_npix), "wings": self.wings,
                "scatter": self.scatter, "gain_mode": self.gain_mode,
                "n_sources": 0, "n_extended": len(self.components),
                "centre_ra": float(centre[0]), "centre_dec": float(centre[1]),
                "offset_ra_arcsec": float(f.offset_arcsec[0]),
                "offset_dec_arcsec": float(f.offset_arcsec[1]),
                "subgrid_k": ",".join(str(k) for k in sorted(self._psf_cache))}

    def components_table(self):
        """One row per component (COMP extension): source, rates, k, placement, fluxes.

        total_e_s is the component's whole rate, on_frame_e_s what lands on the
        frame after PSF and clipping; the fluxes are the flat-F_nu fluxes in the
        detector filter that give total_e_s (fluxrate.band_fluxes). A component
        that misses the detector has subgrid_k = y0 = x0 = -1.
        """
        sf = self.frame.sensorfilter
        scratch = np.zeros(self.frame.shape, dtype=np.float32)
        rows = []
        for comp, (sub, cut) in zip(self.components, self._rendered()):
            row = comp.to_record()
            if hasattr(comp, "image"):
                # flux_to_rate, not image.rate_scale: rate_map already warned on BAND
                img = comp.image
                row["rate_per_unit"] = flux_to_rate(
                    img.unit, sf, pixel_area_arcsec2=(img.pixel_scale_mas / 1000.0) ** 2)
            if sub is None:
                total = (float(comp.image.data.sum()) * row["rate_per_unit"]
                         if hasattr(comp, "image") else 0.0)
                place = {"subgrid_k": -1, "y0": -1, "x0": -1, "on_frame_e_s": 0.0}
            else:
                total = float(sub.total_in)
                place = {"subgrid_k": int(sub.k), "y0": int(sub.y0), "x0": int(sub.x0),
                         "on_frame_e_s": self._add_cutout(scratch, cut)}
            fl = band_fluxes(total, sf)
            rows.append(row | place | {"total_e_s": total, "f_nu_jy": fl["f_nu_jy"],
                                       "f_lambda": fl["f_lambda"], "f_band": fl["f_band"]})
        table = Table(rows=rows) if rows else Table()
        # e-/s columns carry the unit in their names (not a FITS-standard unit)
        for name, unit in (("f_nu_jy", u.Jy), ("f_lambda", u.erg / u.s / u.cm**2 / u.AA),
                           ("f_band", u.erg / u.s / u.cm**2), ("pixel_scale_mas", u.mas)):
            if name in table.colnames:
                table[name].unit = unit
        return table

    # -- exposure ------------------------------------------------------------------ #

    def expose(self, exptime=90.0, n_reads=1, seed=None, add_noise=True):
        """One exposure: rate_image() x exptime, sky + dark, noise, ADU -> SimulatedField.

        Same detector model as simulate_field (render.add_noise_and_digitize);
        the catalogue is empty (no stars), the components are in
        SimulatedField.components (written as the COMP extension). A missing seed
        is drawn and recorded so the exposure can be repeated.
        """
        if not (np.isfinite(exptime) and exptime > 0.0):
            raise ValueError(f"exptime must be a positive finite number of seconds, "
                             f"got {exptime!r}")
        if isinstance(n_reads, bool) or int(n_reads) != n_reads or n_reads < 1:
            raise ValueError(f"n_reads must be a positive integer, got {n_reads!r}")
        exptime, n_reads = float(exptime), int(n_reads)
        if seed is None:
            seed = secrets.randbits(32)
        image_e = self.rate_image() * np.float32(exptime)
        out = add_noise_and_digitize(image_e, self.sim, exptime, n_reads,
                                     np.random.default_rng(seed), add_noise=add_noise)

        comps = self.components_table()
        sensor = self.sim.sensor
        sky, dark = sky_and_dark_rates(self.sim)
        halo = self._get_halo()
        k_first = sorted(self._psf_cache)[0] if self._psf_cache else None
        psf_k, wing, _ = self._psf_cache[k_first] if k_first else (None, None, None)
        total = float(np.sum(comps["total_e_s"])) if len(comps) else 0.0
        fl = band_fluxes(total, self.frame.sensorfilter)
        params = self.params | {
            "exptime": exptime, "n_reads": n_reads, "seed": seed,
            "add_noise": bool(add_noise), "mag_limit": None, "oversample": None,
            "pixel_size_um": float(self.geom.pixel_size_um),
            "gain": float(sensor.gain.to(u.electron / u.ct).value),
            "bias": float(sensor.bias_level.to(u.ct).value),
            "saturation_e": float(saturation_level_e(sensor)),
            "read_noise": float(sensor.read_noise.value),
            "dark_e_s": dark, "sky_e_s": sky,
            "well_depth": (float(sensor.meta["well_depth"])
                           if sensor.meta.get("well_depth") is not None else None),
            "scatter_fraction": float(halo.frac_total) if halo is not None else None,
            "scatter_norm": halo.meta.get("SCATNORM") if halo is not None else None,
            "scatter_file": str(halo.meta.get("SCATFILE")) if halo is not None else None,
            "scatter_reach_px": float(halo.r_px[-1]) if halo is not None else None,
            "wing_alpha": float(_core_wing(wing).alpha) if wing is not None else None,
            "wing_c": float(_core_wing(wing).c) if wing is not None else None,
            "wing_floor_e": None, "chromatic": False,
            "image_total_e_s": total,
            "image_rate_e_s": float(np.sum(comps["on_frame_e_s"])) if len(comps) else 0.0,
            "flux_origin": "input image (BUNIT)",
            "image_bunit": _common(comps, "bunit"),
            "flux_jy": fl["f_nu_jy"], "flux_flam": fl["f_lambda"], "flux_band": fl["f_band"],
            "band_pivot_nm": fl["pivot_nm"], "band_width_nm": fl["width_nm"],
        }
        tel = self.sim.telescope
        models = {
            "wing": wing, "halo": halo,
            "stamp": bin_oversampled(psf_k, k_first) if psf_k is not None else None,
            "stamp_npix": int(self.stamp_npix), "wing_floor_e": None,
            "sky_dark_e": (sky + dark) * exptime,
            "pixel_size_um": self.geom.pixel_size_um,
            "diameter_m": float(tel.diameter_primary.to("m").value),
            "f_num": float(tel.f_num),
        }
        catalog = Table(names=[c for c, _ in _CAT_COLUMNS], dtype=[d for _, d in _CAT_COLUMNS])
        return SimulatedField(
            image_adu=out["image_adu"], image_e=out["image_e"],
            image_clean=out["image_clean"], saturation_mask=out["satmask"],
            wcs=self.frame.wcs, catalog=catalog, params=params, models=models,
            components=comps)
