# wcc_phot — differential photometry pipeline for WCC image series

**Date:** 2026-07-02
**Status:** implemented without interactive review (user AFK); decisions below
are defaults chosen to match repo conventions — flag anything to change.

## Goal

A sibling subpackage `wcc_phot` (same repo, `src/wcc_phot/`, CLI `wcc-phot`)
that extracts **aperture or PSF photometry** of a **target star and the top
N=10 best reference stars** from a series of WCC images, **re-centroiding
every star in every frame**, and produces a differential (relative) light
curve of the target.

## Scope decisions (made in lieu of Q&A)

1. **Input**: a time series of wcc-sim output FITS files (paths, open
   `HDUList`s, or in-memory `SimulatedField`s) of the same field. The
   pipeline leans on what wcc-sim already provides: `SCI` (ADU) + `GAIN`
   → electrons, the `CAT` extension for star positions/magnitudes/flags,
   the TAN WCS for per-frame predicted pixel positions, and header cards
   (`EXPTIME`, `NREADS`, `RDNOISE`, `SKYRATE`, `DARK`, `FOCUS`, `SENSORF`,
   `JITTER`, `PLTSCL`) for error estimates and PSF-model construction.
   No blind source detection in v1.
2. **Target selection**: by Gaia `source_id` or `(ra, dec)` — nearest CAT
   entry within 2 arcsec (error if none).
3. **"Best" reference stars** (selected once, on the first frame):
   - `in_image`, not `saturated`, not the target;
   - centroid box + annulus fully inside the frame (edge margin);
   - isolated: no CAT neighbor within `iso_radius` (default = annulus
     outer radius) that is brighter than `G_star + 1` mag;
   - survivors ranked by |G − G_target| (photometrically most similar
     first); take the top `n_ref` (default 10). Fewer available → warn
     and continue.
4. **Centroiding**: every star, every frame. Predicted position = catalog
   (ra, dec) through that frame's WCS; refined with background-subtracted
   center-of-mass (`photutils.centroids.centroid_sources` +
   `centroid_com`) in a box sized to the PSF. COM (not a Gaussian fit)
   because the +1/+2-wave defocus PSFs are donuts. Failed centroid →
   fall back to the WCS position and set a flag bit.
5. **Aperture photometry**: circular aperture + sigma-clipped-median
   annulus background on the electron image. Default geometry is derived
   from the wcc-sim PSF model for the frame's (sensorfilter, focus,
   jitter): `r_ap` = 95% encircled-energy radius, annulus at
   1.5–2.5 × `r_ap`, centroid box ≈ 2 × `r_ap` (odd). All overridable.
   Fluxes are *not* aperture-corrected (cancels in differential
   photometry); the EE fraction at `r_ap` is recorded in `params`.
6. **PSF photometry**: `photutils.psf.PSFPhotometry` with an `ImagePSF`
   model built from `wcc_sim.psf.render_oversampled_psf` for the frame's
   focus/filter/jitter (oversample 5 to keep memory modest). Initial
   (x, y, flux) from the centroid + aperture estimate; local background
   from the same annulus. Fitted flux ≈ total flux.
7. **Errors**: standard CCD equation using header rates — Poisson of the
   star + per-pixel background variance ((sky+dark)·exptime +
   nreads·rdnoise²) over the aperture, plus the background-estimate term
   (npix²/n_ann). PSF fluxes carry the fit covariance error.
8. **Saturation**: any saturated pixel within the aperture (or fit
   region) sets a flag bit for that star/frame; measurements are kept
   but flagged.
9. **Light curve**: ensemble = straight sum of reference fluxes;
   `rel_flux = F_target / ΣF_ref`, error-propagated, plus
   `rel_flux_norm` (divided by its median over frames). Frames carry an
   index and optional user-supplied `times` (wcc-sim headers have no
   observation time).
10. **Output**: `PhotometryResult` dataclass with `.stars` (selection
    table), `.measurements` (long table: one row per star per frame),
    `.lightcurve` (one row per frame), `.params`; `to_hdulist()`/
    `.write()` produce a FITS with STARS/PHOT/LC binary-table extensions
    mirroring the wcc-sim fitswriter style. CLI can also dump the light
    curve as ECSV.

## Architecture

```
src/wcc_phot/
  __init__.py    exports run_photometry, PhotometryResult, __version__
  io.py          Frame dataclass + load_frame(path|HDUList|SimulatedField)
  geometry.py    PhotGeometry (r_ap, annulus, boxes) + model-PSF EE defaults
  select.py      pick_target(), pick_references()
  centroid.py    centroid_stars(frame, x0, y0, box) -> x, y, flags
  apphot.py      aperture_photometry_frame(frame, x, y, geom)
  psfphot.py     build_psf_model(meta), psf_photometry_frame(...)
  lightcurve.py  build_lightcurve(measurements)
  pipeline.py    run_photometry() orchestration + PhotometryResult
  cli.py         wcc-phot entry point
```

Data flow: `load_frame` → (frame 0) `pick_target` + `pick_references` →
per frame: WCS-predict → `centroid_stars` → aperture or PSF measure →
stack into `measurements` → `build_lightcurve`.

Each module is independently testable; `wcc_phot` imports `wcc_sim` only
in `geometry.py`/`psfphot.py` (PSF model) — the rest works from FITS
contents alone.

## Error handling

- Target not matched within 2″ → `ValueError`.
- < n_ref usable references → `UserWarning`, proceed with what's there;
  0 references → `ValueError`.
- Centroid failure / off-image / saturated-in-aperture → per-row `flags`
  bitmask (1=centroid fallback, 2=saturated, 4=edge-clipped), never a crash.
- Frames with mismatched sensorfilter/focus → `ValueError` (one PSF model
  per run).

## Testing

Simulated series via `simulate_field` (256×256 subarray, stamp 33,
focus 0, canned ~14-star catalog, per-frame sub-pixel pointing offsets):

- selection: target by id and by ra/dec; saturated/edge/crowded refs
  rejected; Δmag ranking; n_ref truncation warning
- centroid: recovers injected sub-pixel shifts to <0.1 px (noiseless)
- aperture: noiseless single star → flux ≈ EE(r_ap)·rate·exptime to ~2%
- PSF: noiseless → fitted flux ≈ rate·exptime to ~2%
- end-to-end: 3 noisy frames → rel_flux_norm scatter consistent with
  propagated errors; FITS round-trip; CLI smoke test

## Follow-ups (not v1)

- Sphinx page + tutorial notebook (docs mirror the wcc-etc4 style)
- Inverse-variance ensemble weighting / iterative ref rejection
- ePSF built from the frames themselves (data-driven PSF phot)
