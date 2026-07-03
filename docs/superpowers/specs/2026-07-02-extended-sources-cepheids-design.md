# Extended sources, chromatic effective PSF, and an M101 Cepheid P-L demo

**Date:** 2026-07-02
**Deliverables:**
- `wcc_sim.extended` — analytic extended-source components rendered into
  `simulate_field` images.
- `wcc_sim.chromatic` — spectrum-weighted effective PSFs and band-averaged
  extinction factors.
- `scripts/cepheids_m101.py` — end-to-end demo: Cepheids injected into an
  M101 outer-disk field, recovered with `wcc_phot` PSF photometry, fitted
  Leavitt law and distance modulus verified against the injected values.

## Goal

Extend the point-source-only simulator so a frame can contain smooth
extended emission (galaxy light), with the source spectrum handled
correctly in the PSF convolution, and demonstrate the capability with a
Cepheid period-luminosity retrieval in M101 including realistic reddening.

## Physics decisions

### Chromatic PSF: spectrum-weighted effective PSF, not per-λ image convolution

Convolution is linear, so for a component with a single spectrum S(λ) the
exact band-integrated image equals the surface-brightness model convolved
with one **effective PSF**:

```
PSF_eff = Σ_i w_i · PSF(λ_i),   w_i ∝ ∫_bin S(λ) T(λ) dλ,   Σ w_i = 1
```

with T(λ) the total throughput (`sim.sensor.bandpass`) and λ_i ≈ 7 nodes
spanning the band. Per-wavelength convolution of the image is only needed
when the spectrum varies within a component — modeled here as multiple
components instead.

- **focus=0 (Airy):** rendered per node — `DetectorPSFContext` already takes
  `wavelength_m`. In-focus FWHM varies ~±10% across a broad band, so this
  matters.
- **focus=1,2 (defocus):** `wcc_etc.DefocusPSF` is a fixed Huygens image
  zoomed to pixel scale with no wavelength dependence, and the defocused PSF
  is geometry-dominated. `effective_psf` returns the monochromatic stamp for
  these — documented behavior, not a warning.

Point sources get the same treatment: with `chromatic=True`, each Pickles
spectral type gets its own cached effective PSF stamp (types per field are
few, so the render count is bounded). Default is `chromatic=False` so
existing results, tests, and the TRAPPIST-1b script are bit-unchanged.

### Extended-source rendering: native-grid FFT convolution

Alternatives considered and rejected:
- **Oversampled full-frame grid (11×):** ~7 TB float32 per full frame.
- **Dense point-source sampling via `add_star`:** ~10⁶ stamp placements per
  component, slow, adds quantization noise.

Chosen approach: render each component's surface-brightness profile
analytically in e⁻/s/pix on the **native** pixel grid, then FFT-convolve
with the binned effective-PSF stamp (`scipy.signal.fftconvolve`,
`mode="same"`). The in-focus PSF FWHM is ~3.2 px at 16.87 mas/pix, so the
native grid samples the kernel adequately. Sérsic centers are cuspy: the
profile (not the convolution) is evaluated on a 9× refined sub-grid in a
small box (~±2 r_eff, capped) around each center and bin-summed, matching
the pixel-integration accuracy of the point-source path.

Flux normalization mirrors `add_star`: total component flux is divided by
`(1 + wing.energy_beyond(half))` when wings are enabled so extended and
point photometry stay mutually consistent. No wing halo is drawn for
extended components — the redistributed energy is smooth and far below the
per-pixel noise floor.

### Reddening: band-averaged attenuation factor, wcc_etc untouched

`attenuation_factor(template, ebv, sensorfilter, rv=3.1)` applies a
Fitzpatrick (1999) curve (`dust_extinction.parameter_averages.F99`) to the
synphot template spectrum and returns the single multiplier

```
a = ∫ S(λ) 10^{-0.4 A(λ)} T(λ) dλ / ∫ S(λ) T(λ) dλ
```

applied to the unreddened wcc_etc rate. This is exact for broadband
photometry and avoids pushing modified spectra through `wcc_etc.scene`'s
`type(x) is str` dispatch. The same E(B-V) also feeds the effective-PSF
weights (a reddened spectrum is a redder weighting).

## Components

### `wcc_sim/chromatic.py`

- `band_nodes(sim, n_nodes=7)` → node wavelengths + normalized weights for a
  given source spectrum: sub-band integrals of S(λ)·T(λ) via synphot.
- `effective_psf(sim, focus, spectrum, oversample, stamp_npix, jitter_sigma_mas, n_nodes=7)`
  → oversampled effective PSF (Airy path); monochromatic passthrough for
  focus=1,2.
- `attenuation_factor(template, ebv, sensorfilter, rv=3.1)` → float.
- Spectra are obtained from wcc_etc scene elements (the same Pickles files
  the rate calculation uses), so weights and rates are self-consistent.

### `wcc_sim/extended.py`

- `SersicComponent` dataclass: `ra, dec` (deg), `total_mag` (Gaia G, vegamag
  — same normalization convention as point sources) **or**
  `sb_mag_arcsec2` at r_eff, `n`, `r_eff_arcsec`, `ellip`, `pa_deg`,
  `template="G2V"`, `ebv=0.0`. `n=1` gives an exponential disk, `n=4` a
  de Vaucouleurs bulge.
- `render_extended(components, wcs, shape, sensorfilter, sim, psf_eff_by_key, wing=None)`
  → float32 e⁻/s/pix image. Components are grouped by `(template, ebv)`;
  each group is summed unconvolved, then convolved once with that group's
  effective PSF. Profile evaluation uses `astropy.modeling.models.Sersic2D`
  with the central refinement described above; total rate comes from
  `rate_for_spt(template, sensorfilter)` scaled by
  `10^(-0.4 (mag - REF_MAG))` and the attenuation factor.

### `simulate_field` changes (`wcc_sim/pipeline.py`)

New kwargs, all defaulting to current behavior:
- `extended_sources=None` — list of `SersicComponent`; rendered image (×
  exptime) is added to `image_sources` before `add_noise_and_digitize`, so
  extended light participates fully in Poisson noise and saturation.
- `chromatic=False` — when True, point sources are rendered with per-spt
  effective PSFs and extended components with per-(template, ebv) effective
  PSFs; when False, everything uses the current central-wavelength stamp.

`params` records `n_extended`, `chromatic`, and a compact per-component
summary; the FITS catalog table is unchanged (extended components are not
catalog rows — they carry no photometry ground truth beyond `params`).

### `scripts/cepheids_m101.py`

Structure mirrors `transit_trappist1b.py`: config block → per-epoch
simulation to `cepheid_out/` → `wcc_phot` run → verification with printed
PASS/FAIL and nonzero exit on failure → gks-style summary figure.

| Item | Value | Why |
|---|---|---|
| Field | M101 outer-disk pointing (~5′ from center, HST-Key-Project-like) | disk SB ~22–23 mag/arcsec² keeps crowding honest but photometrable |
| Galaxy model | exponential disk (n=1) + faint bulge tail (n=4), normalized to the M101 SB profile at the field radius, F8I-ish composite template | 2 components exercise the multi-template convolution path |
| Foreground | real (cached) Gaia catalog at the pointing | reference stars for wcc_phot come from here |
| Cepheids | ~30, P log-uniform 10–60 d, I-band Leavitt law + μ=29.2, asymmetric sawtooth light curve, F8I template | long-period end is what WCC can reach at 6.9 Mpc |
| Reddening | MW foreground E(B−V)=0.008 + per-Cepheid internal lognormal (median ~0.1) | realistic scatter source in the P-L |
| Epochs | ~25 over ~90 d, exposure set so the P-L faint end reaches SNR ≳ 15/epoch (expected ~1 h, n_reads > 1) | period recovery needs per-epoch detections |
| Filter/focus | `zwo:i`, focus=0, `chromatic=True` | in-focus for point-source depth; chromatic path exercised |
| Photometry | `wcc_phot` `method="psf"` | LocalBackground annulus absorbs the smooth galaxy light |

Cepheids are injected exactly as the transit script injects: per-epoch
catalog copies with `phot_g_mean_mag` set from the phase of each Cepheid's
light curve, reddening applied as a mag offset from `attenuation_factor`.
The BP−RP→spt lookup covers dwarfs only, so `rates_for_catalog` gains one
small extension: an optional `spt` column in the input catalog overrides
the color-based lookup, letting Cepheid rows carry `F8I`. Leavitt-law, light-curve-template,
and reddening-draw logic lives in the script — demo science, not
instrument physics.

**Verification (PASS/FAIL):**
1. **Period recovery** — ≥80% of Cepheids with median per-epoch SNR > 10
   have a Lomb-Scargle peak within 1% of the injected period.
2. **Distance modulus** — P-L fit to phase-averaged recovered mags,
   corrected for the known mean extinction, gives μ within 3σ of 29.2.
3. **Noise sanity** — χ²/dof of light-curve residuals against the injected
   model in [0.5, 2].

Figure: recovered P-L relation with injected relation overlaid + one
example phase-folded light curve + a field cutout showing Cepheids on the
galaxy background.

## Error handling

- `SersicComponent` validates: exactly one of `total_mag` /
  `sb_mag_arcsec2`; `n > 0`; `r_eff_arcsec > 0`; `0 ≤ ellip < 1`; unknown
  `template` fails fast via the existing wcc_etc lookup error.
- `effective_psf` raises on `n_nodes < 1`; `n_nodes=1` reproduces the
  monochromatic stamp exactly (test anchor).
- Components fully outside the padded frame are skipped (cheap bounding-box
  check) rather than rendered.
- `chromatic=True` with focus≠0 is allowed and documented as a
  monochromatic passthrough (defocus PSF has no wavelength model).

## Testing

- **chromatic:** weights sum to 1; `n_nodes=1` equals monochromatic;
  M-dwarf effective PSF measurably wider than A-star (in focus);
  `attenuation_factor(ebv=0) == 1`; a hand-integrated F99 case matches.
- **extended:** Sérsic total flux conserved through render+convolve
  (large-aperture sum vs. analytic total within 0.5%); ellipticity/PA
  orientation check; center refinement converges (9× vs 27× within
  tolerance); component off-frame is a no-op.
- **pipeline:** `simulate_field(extended_sources=[...])` on a small `shape`
  adds the expected mean surface brightness; `extended_sources=None,
  chromatic=False` is bit-identical to current output (regression guard).
- **demo:** `scripts/cepheids_m101.py` is the end-to-end integration test;
  its three checks are the acceptance criteria.

## Out of scope (later PRs)

- FITS-image extended-source input (real morphology maps).
- Spatially varying PSF across the field.
- Chromatic defocus PSFs (needs wavelength-resolved Huygens data in
  wcc_etc).
