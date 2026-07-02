# wcc-sim: End-to-end WCC photometric image simulator — Design

**Date:** 2026-07-02
**Status:** Draft for review

## Purpose

Simulate realistic full-array 2D images from the Lazuli Wide-field Context Camera
(WCC) for a given RA/Dec pointing: query Gaia for the stars in the field, convert
Gaia photometry to detector count rates using the `wcc_etc` instrument model,
render each star with the appropriate PSF (in-focus Airy, or +1/+2 waves defocus
Huygens PSFs), add realistic noise (photon, sky background, dark, read), apply
saturation, and write a multi-extension FITS file with WCS and a source catalog.

## Decisions made with Gummi

- **Scope per call:** one detector image per call (specify `sensorfilter`, e.g.
  `zwo:r` or `qcmos:bb`). A thin loop handles multiple detectors/focus levels.
- **Gaia → flux:** map each star's BP−RP color to the nearest Pickles spectral
  type (bundled in wcc-etc), normalize to its G mag in the Gaia G bandpass,
  integrate through the WCC filter via the ETC.
- **Detector model v1:** photon + sky + dark + per-frame read noise (always);
  full-well/ADC saturation with mask; 10 mas Gaussian jitter blur (overridable).
  No flat-field/PRNU, no cosmic rays in v1.
- **Interface:** Python API + thin CLI (chosen as the recommended default when
  Gummi was away; revisit if he prefers config-file driven).

## Architecture

New standalone package **`wcc_sim`** (src layout, this repo) that depends on
`wcc_etc` (installed editable from
`/Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc`).
wcc_etc remains the single source of truth for instrument physics (throughputs,
sensors, PSFs, count rates, noise semantics). wcc_sim adds only what wcc_etc
lacks: Gaia catalogs, WCS, multi-source scene rendering, and FITS output.

Alternatives considered: extending wcc_etc itself (rejected: bloats the ETC with
astroquery/WCS deps, couples releases) and a standalone reimplementation
(rejected: duplicates instrument physics that would drift).

### Key facts inherited from wcc_etc (v0.6.0)

- `Sensor.from_name("zwo:r")` etc.; kinds `zwo` (Sony IMX455) and `qcmos` (HWK4123).
- Telescope: D = 3.065 m, f/15, jitter_sigma = 10 mas (`data/config/lazuli.toml`).
- Plate scales: IMX 3.76 µm → **16.87 mas/pix**; HWK 4.6 µm → **20.64 mas/pix**.
- Array dims (manufacturer, NOT stored in `Sensor`): IMX455 **9568×6380**,
  HWK4123 **4096×2304**. wcc_sim stores these in its own small table.
- PSFs: `AiryPSF()` (0 waves, analytic), `DefocusPSF(DEFOCUS_1WAVE_PATH)`,
  `DefocusPSF(DEFOCUS_2WAVE_PATH)` (Zemax Huygens, 500 nm, 4 µm/pix source
  sampling). Focus level per sensorfilter comes from `io.sensor_info`;
  `Simulation.from_sensorfilter` auto-selects.
- Count-rate primitive: `Simulation._count_rate_components()` →
  `{"source_rate_total", "background_rate_per_pix", "diffuse_rate_per_pix"}` in e-/s.
- Noise semantics: read noise per frame (variance × n_reads); saturation
  evaluated per frame (per-frame time = time/n_reads); well depth and
  `adc_max = 2**bit_depth − 1` on `Sensor`.
- FOV of a full IMX array ≈ 2.7′ × 1.8′; HWK ≈ 1.4′ × 0.8′.

## Public API

```python
from wcc_sim import simulate_field

field = simulate_field(
    ra=150.1, dec=2.2,            # deg (ICRS), pointing = array center
    sensorfilter="zwo:r",         # any key in wcc_etc.io.sensor_info
    focus=None,                   # None → sensor_info focus_level; or 0|1|2 waves
    exptime=90.0,                 # s, total
    n_reads=1,                    # coadded frames (ETC semantics)
    pa=0.0,                       # position angle, deg E of N
    jitter_sigma_mas=None,        # None → telescope default (10 mas)
    mag_limit=21.0,               # Gaia G faint limit
    add_noise=True,
    seed=None,
    output="field.fits",          # None → don't write
)
```

Returns a `SimulatedField` dataclass: `image_adu`, `image_e`, `image_clean`
(noiseless e-), `saturation_mask`, `wcs`, `catalog` (astropy Table with
per-star x, y, G, BP−RP, SpT, rate_e_s), `params` (dict of everything),
`.to_hdulist()`, `.write(path)`.

CLI (console script `wcc-sim`):

```
wcc-sim --ra 150.1 --dec 2.2 --sensorfilter zwo:r --focus 1 \
        --exptime 90 --n-reads 1 --mag-limit 21 --seed 42 -o field_1wave.fits
```

## Modules

### `catalog.py` — Gaia query
- `query_gaia(ra, dec, radius, mag_limit) -> astropy.table.Table` using
  `astroquery.gaia.Gaia` (DR3 cone search). Columns: `source_id, ra, dec,
  phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag`.
- Query radius = half-diagonal of the array + 10″ margin (PSF wings).
- Optional on-disk cache (`~/.wcc_sim/cache/`, keyed by rounded
  ra/dec/radius/mag_limit) so repeated runs are offline and fast.
- v1 ignores proper motion (epoch offsets < 1 px for almost all stars at these
  plate scales over ~decade baselines; revisit if needed).

### `starflux.py` — Gaia photometry → count rates
- Bundled data: Gaia DR3 G passband CSV; BP−RP → spectral-type lookup table
  (Mamajek mean dwarf colors, O9V–M9V).
- `spt_from_bp_rp(bp_rp) -> str`: nearest-neighbor in the lookup; stars with
  missing BP or RP default to **G2V**.
- `rate_for_star(G, spt, sensorfilter) -> float` (e-/s): build a wcc_etc
  `Scene` with the Pickles spectrum for `spt`, `mag=15.0`,
  `magsys="vegamag"`, `bandpass=<Gaia G SpectralElement>`; take
  `source_rate_total` from `Simulation._count_rate_components()`; scale by
  `10**(-0.4*(G-15))`. Rates memoized per `(spt, sensorfilter)` — one synphot
  integration per unique spectral type, so thousands of stars are cheap.
- Sky background and dark come from the same `Simulation` (`zodi` default,
  22.5 mag/arcsec² johnson_v): `background_rate_per_pix` and
  `sensor.dark_current`.

### `wcsutil.py` — WCS
- `build_wcs(ra, dec, plate_scale_mas, pa, shape) -> astropy.wcs.WCS`:
  TAN projection, CRPIX at array center, CD matrix from plate scale + position
  angle, RA axis flipped (East-left convention). No distortion in v1.

### `psf.py` — PSF preparation
- `get_psf(focus, simulation, npix, oversample=11) -> np.ndarray` (oversampled,
  normalized): focus 0 → `AiryPSF`, 1 → `DefocusPSF(DEFOCUS_1WAVE_PATH)`,
  2 → `DefocusPSF(DEFOCUS_2WAVE_PATH)`, rendered through wcc_etc's
  `DetectorPSFContext` with the jitter blur applied. Rendered **once per
  image** (PSF is field-invariant in wcc_etc).
- Stamp size default: 128 detector px in focus, 256 for defocus (must capture
  ≥ 99.5% encircled energy; verified in tests, configurable).

### `render.py` — scene renderer
- `render_scene(catalog, wcs, psf_oversampled, exptime, shape) -> image_clean`:
  per star, compute (x, y) via WCS; split into integer pixel + subpixel
  fraction; realize the subpixel shift by slicing the oversampled PSF grid,
  bin ×oversample down to detector sampling, scale by `rate_e_s * exptime`,
  and add into the full float32 array (edge stamps clipped). Stars > stamp/2
  outside the array are skipped.
- `add_noise_and_digitize(image_clean, sim, n_reads, rng)`:
  1. expectation e- = sources + (sky + dark) × exptime (uniform terms)
  2. per-frame saturation: pixels with expectation/n_reads ≥ well_depth are
     flagged; summed e- clipped at n_reads × well_depth
  3. Poisson realization of the (clipped) expectation
  4. read noise: Gaussian σ = read_noise × √n_reads per pixel
  5. ADU = clip(e-/gain + bias_level, 0, n_reads × adc_max), float32
- Full IMX frame is 9568×6380 float32 ≈ 244 MB per image plane — acceptable;
  arrays are float32 throughout.

### `fitswriter.py` — FITS output
Multi-extension FITS:
- `SCI` (primary): image in ADU. Header: full WCS; RA/DEC/PA; SENSORF, FOCUS,
  EXPTIME, NREADS, JITTER, MAGLIM, SEED; gain, read noise, dark, well depth,
  plate scale; Gaia query metadata (radius, n_sources, DR); `wcc_etc` and
  `wcc_sim` versions; DATE.
- `SATMASK`: uint8 saturation mask.
- `CAT`: BinTableHDU of injected sources (source_id, ra, dec, x, y, G, BP−RP,
  SpT, rate_e_s, flagged saturated or off-edge).
- `CLEAN` (optional, on by default): noiseless expectation image in e-.

### `pipeline.py` / `cli.py`
- `simulate_field(...)` orchestrates: sensor dims + plate scale → WCS → Gaia
  query → rates → PSF → render → noise → FITS.
- `cli.py`: argparse console entry point `wcc-sim` mapping 1:1 onto
  `simulate_field`.

## Error handling

- Unknown `sensorfilter` → `ValueError` listing valid keys from `sensor_info`.
- Unimplemented sensorfilter (per `sensor_info["implemented"]`) → warning, proceed.
- Empty Gaia result / Gaia service down: empty → warn and render sky-only
  image; network error → clear exception advising the cache or `catalog=`
  injection (API accepts a pre-made Table to bypass the query).
- `focus` not in {0, 1, 2} → `ValueError`.

## Testing (pytest, fully offline)

- Gaia is mocked with canned Tables (recorded once); no network in tests.
- **Photometric closure:** single star on a small subarray → aperture
  photometry (photutils / `FitsImg`) recovers the ETC `signal_e` for the same
  (mag, spt, filter, exptime) to ≲ 1%.
- **Noise statistics:** sky-only image variance ≈ (sky+dark)·t + n_reads·RN²
  per pixel.
- **WCS:** pixel→sky→pixel round-trip; a star at the pointing center lands at
  CRPIX; PA rotation sanity check.
- **Saturation:** bright star saturates, mask set, CAT flag set; per-frame
  semantics (same star, n_reads=4 vs 1).
- **PSF energy capture:** default stamp sizes hold ≥ 99.5% encircled energy
  for all three focus levels.
- **FITS integrity:** all extensions present, header keywords round-trip,
  CLEAN + noise realization consistent with SCI.
- End-to-end smoke test on a 512×512 subarray with a canned catalog.

## Out of scope (v1)

Focal-plane mosaics (single detector per call; `sensor_info` centers make the
extension natural later), proper-motion epoch propagation, field-dependent
PSFs, flat-field/PRNU, cosmic rays, hot pixels, galaxies/extended sources,
non-linearity correction, per-frame (video-mode) output.
