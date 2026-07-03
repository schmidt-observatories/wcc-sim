# Extended Sources + Chromatic Effective PSF + M101 Cepheid Demo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add analytic extended-source (Sérsic) rendering with spectrum-weighted chromatic effective PSFs and F99 reddening to `wcc_sim`, and demonstrate it with an end-to-end M101 Cepheid period-luminosity retrieval script.

**Architecture:** Extended components are rendered analytically in e⁻/s/pix on the native pixel grid and FFT-convolved once per (template, E(B−V)) group with an effective PSF (Σᵢ wᵢ·PSF(λᵢ), weights ∝ spectrum × throughput per sub-band). Point sources optionally get per-spectral-type effective PSFs (`chromatic=True`). Reddening is a band-averaged rate multiplier computed with `dust_extinction`, so `wcc_etc` is never modified. The demo mirrors `scripts/transit_trappist1b.py`: simulate epochs → `wcc_phot` PSF photometry → verification with PASS/FAIL exit code.

**Tech Stack:** Python 3.13 (`conda activate py313`), numpy, scipy, astropy (`Sersic2D`), synphot, dust_extinction (new dependency), wcc_etc (unmodified), photutils via wcc_phot.

**Spec:** `docs/superpowers/specs/2026-07-02-extended-sources-cepheids-design.md`

## Global Constraints

- Work on branch `feature/extended-sources` (already created; spec is committed there).
- Run everything in the `py313` conda env: `source ~/.zshrc; conda activate py313`. Tests: `python -m pytest`.
- **Do not modify wcc_etc.** Reddening and chromatic weights wrap around it.
- Defaults must not change behavior: `simulate_field(...)` with `extended_sources=None, chromatic=False` (the defaults) must leave all existing tests passing and images unchanged.
- `wcc_etc.scene` dispatches on `type(x) is str` — always coerce spectral-type values with `str(...)` before passing them (see `starflux.rate_for_spt` for precedent).
- `numpy >= 2` in this env: use `np.trapezoid`, not `np.trapz`.
- Follow repo style: module docstring explaining the physics decision, short numpy-ish docstrings, no type annotations in `src/wcc_sim` (matches existing files).
- The working tree also carries **unrelated uncommitted changes** (`src/wcc_phot/*`, `scripts/transit_trappist1b.py`, README) from a different PR-in-progress. `git add` only the files named in each task — never `git add -A`.

---

### Task 1: `render_oversampled_psf` gains a `wavelength_m` override

**Files:**
- Modify: `src/wcc_sim/psf.py:28-54`
- Test: `tests/test_psf.py` (append)

**Interfaces:**
- Produces: `render_oversampled_psf(sim, focus, oversample=11, stamp_npix=None, jitter_sigma_mas=None, wavelength_m=None)` — `wavelength_m=None` keeps the current behavior (sensor central wavelength); a float renders the Airy PSF at that wavelength (meters). Task 2 consumes this.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_psf.py`, matching its existing imports)

```python
def _rms_radius(psf):
    n = psf.shape[0]
    c = n // 2
    yy, xx = np.mgrid[:n, :n]
    r2 = (yy - c) ** 2 + (xx - c) ** 2
    return float(np.sqrt((psf * r2).sum() / psf.sum()))


def test_explicit_wavelength_matches_default():
    from wcc_sim.detectors import make_base_simulation
    from wcc_sim.psf import render_oversampled_psf

    sim = make_base_simulation("zwo:r")
    a = render_oversampled_psf(sim, 0, oversample=3, stamp_npix=33)
    b = render_oversampled_psf(
        sim, 0, oversample=3, stamp_npix=33,
        wavelength_m=float(sim.sensor.wavelength.to("m").value),
    )
    assert np.array_equal(a, b)


def test_longer_wavelength_widens_airy():
    from wcc_sim.detectors import make_base_simulation
    from wcc_sim.psf import render_oversampled_psf

    sim = make_base_simulation("zwo:r")
    blue = render_oversampled_psf(sim, 0, oversample=3, stamp_npix=33,
                                  wavelength_m=550e-9)
    red = render_oversampled_psf(sim, 0, oversample=3, stamp_npix=33,
                                 wavelength_m=900e-9)
    assert _rms_radius(red) > _rms_radius(blue)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_psf.py -k wavelength -v`
Expected: FAIL with `TypeError: render_oversampled_psf() got an unexpected keyword argument 'wavelength_m'`

- [ ] **Step 3: Implement** — in `src/wcc_sim/psf.py`, change the signature and the context wavelength:

```python
def render_oversampled_psf(
    sim, focus, oversample=11, stamp_npix=None, jitter_sigma_mas=None,
    wavelength_m=None,
):
    """Normalized PSF on a (stamp_npix*oversample)^2 fine grid, centered.

    `wavelength_m` overrides the sensor central wavelength (Airy path only;
    DefocusPSF is a fixed Huygens image and ignores wavelength).
    """
```

and inside the `DetectorPSFContext(...)` call replace the `wavelength_m=` line with:

```python
        wavelength_m=(
            float(sim.sensor.wavelength.to("m").value)
            if wavelength_m is None
            else float(wavelength_m)
        ),
```

(Note: `sensor, telescope = sim.sensor, sim.telescope` already exists above; keep using `sensor`/`telescope` as the file does — only the wavelength line changes plus the new parameter.)

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_psf.py -v`
Expected: all PASS (old + 2 new)

- [ ] **Step 5: Commit**

```bash
git add src/wcc_sim/psf.py tests/test_psf.py
git commit -m "feat: wavelength override for rendered Airy PSF"
```

---

### Task 2: `wcc_sim.chromatic` — band nodes + effective PSF

**Files:**
- Create: `src/wcc_sim/chromatic.py`
- Test: `tests/test_chromatic.py` (create)

**Interfaces:**
- Consumes: `render_oversampled_psf(..., wavelength_m=)` from Task 1; `make_base_simulation` from `wcc_sim.detectors`; `wcc_etc.scene.get_scene_element(name, mag=...).spectrum` (a synphot `SourceSpectrum`); `sim.sensor.bandpass` (synphot `SpectralElement`).
- Produces (Tasks 3, 7, 8 rely on these exact names):
  - `band_support(sim, frac=0.005)` → `(lo_aa, hi_aa)` floats, Å.
  - `band_nodes(sim, spectrum, n_nodes=7, ebv=0.0, rv=3.1)` → `(waves_m, weights)` ndarrays, `weights.sum() == 1`.
  - `effective_psf(sim, focus, spectrum=None, oversample=11, stamp_npix=None, jitter_sigma_mas=None, n_nodes=7, ebv=0.0)` → oversampled PSF ndarray. Passthrough to the monochromatic render when `focus != 0`, `n_nodes == 1`, or `spectrum is None`.
  - `effective_psf_for_spt(sim, sensorfilter, spt, ebv, focus, oversample, stamp_npix=None, jitter_sigma_mas=None, n_nodes=7)` → cached effective PSF (module-level `_EFF_PSF_CACHE` dict).

- [ ] **Step 1: Write the failing tests** — create `tests/test_chromatic.py`:

```python
import numpy as np
import pytest

OS, STAMP = 3, 33  # small/fast render for tests


def _sim():
    from wcc_sim.detectors import make_base_simulation

    return make_base_simulation("zwo:r")


def _spectrum(name):
    from wcc_etc.scene import get_scene_element

    return get_scene_element(name, mag=15.0).spectrum


def _rms_radius(psf):
    n = psf.shape[0]
    c = n // 2
    yy, xx = np.mgrid[:n, :n]
    return float(np.sqrt((psf * ((yy - c) ** 2 + (xx - c) ** 2)).sum() / psf.sum()))


def test_band_nodes_weights_normalized():
    from wcc_sim.chromatic import band_nodes, band_support

    sim = _sim()
    waves_m, weights = band_nodes(sim, _spectrum("G2V"), n_nodes=5)
    assert waves_m.shape == (5,) and weights.shape == (5,)
    assert weights.sum() == pytest.approx(1.0)
    lo, hi = band_support(sim)
    assert np.all(waves_m > lo * 1e-10) and np.all(waves_m < hi * 1e-10)


def test_red_spectrum_shifts_nodes_red():
    from wcc_sim.chromatic import band_nodes

    sim = _sim()
    _, w_blue = band_nodes(sim, _spectrum("A0V"), n_nodes=5)
    _, w_red = band_nodes(sim, _spectrum("M1V"), n_nodes=5)
    waves_m, _ = band_nodes(sim, _spectrum("A0V"), n_nodes=5)
    assert np.sum(waves_m * w_red) > np.sum(waves_m * w_blue)


def test_single_node_is_monochromatic():
    from wcc_sim.chromatic import effective_psf
    from wcc_sim.psf import render_oversampled_psf

    sim = _sim()
    mono = render_oversampled_psf(sim, 0, oversample=OS, stamp_npix=STAMP)
    eff = effective_psf(sim, 0, _spectrum("G2V"), oversample=OS,
                        stamp_npix=STAMP, n_nodes=1)
    assert np.array_equal(eff, mono)


def test_defocus_is_passthrough():
    from wcc_sim.chromatic import effective_psf
    from wcc_sim.psf import render_oversampled_psf

    sim = _sim()
    mono = render_oversampled_psf(sim, 1, oversample=OS, stamp_npix=65)
    eff = effective_psf(sim, 1, _spectrum("M1V"), oversample=OS,
                        stamp_npix=65, n_nodes=5)
    assert np.array_equal(eff, mono)


def test_red_effective_psf_wider_in_focus():
    from wcc_sim.chromatic import effective_psf

    sim = _sim()
    blue = effective_psf(sim, 0, _spectrum("A0V"), oversample=OS,
                         stamp_npix=STAMP, n_nodes=5)
    red = effective_psf(sim, 0, _spectrum("M1V"), oversample=OS,
                        stamp_npix=STAMP, n_nodes=5)
    assert _rms_radius(red) > _rms_radius(blue)
    assert red.sum() == pytest.approx(1.0, rel=1e-6)


def test_effective_psf_for_spt_is_cached():
    from wcc_sim.chromatic import effective_psf_for_spt

    sim = _sim()
    a = effective_psf_for_spt(sim, "zwo:r", "G2V", 0.0, 0, OS, STAMP, n_nodes=3)
    b = effective_psf_for_spt(sim, "zwo:r", "G2V", 0.0, 0, OS, STAMP, n_nodes=3)
    assert a is b


def test_bad_n_nodes_raises():
    from wcc_sim.chromatic import effective_psf

    with pytest.raises(ValueError):
        effective_psf(_sim(), 0, _spectrum("G2V"), n_nodes=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_chromatic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wcc_sim.chromatic'`

- [ ] **Step 3: Implement** — create `src/wcc_sim/chromatic.py`:

```python
"""Spectrum-weighted effective PSFs and band-averaged extinction factors.

Convolution is linear, so the band-integrated image of a source with photon
spectrum S(lambda) equals its surface-brightness model convolved with ONE
effective PSF: sum_i w_i PSF(lambda_i), with w_i proportional to the
integral of S(lambda) T(lambda) over node sub-bands (T = total throughput,
sim.sensor.bandpass). Per-wavelength image convolution is never needed for
a component with a single spectrum.

Only the in-focus Airy PSF has a wavelength model; wcc_etc.DefocusPSF is a
fixed measured Huygens image and the defocused PSF is geometry-dominated,
so focus=1,2 return the monochromatic stamp (documented passthrough).

Reddening enters twice, consistently: attenuation_factor() gives the
band-averaged rate multiplier for a Fitzpatrick (1999) R_V=3.1 curve, and
band_nodes(ebv=...) reddens the weights so a dusty source also gets a
redder effective PSF. wcc_etc itself is never modified.
"""

from functools import lru_cache

import numpy as np
from astropy import units as u
from dust_extinction.parameter_averages import F99
from wcc_etc.scene import get_scene_element

from .detectors import make_base_simulation
from .psf import DEFAULT_STAMP, render_oversampled_psf

_N_WAVE = 4096  # fine-grid points for band integrals
_EFF_PSF_CACHE = {}


def band_support(sim, frac=0.005):
    """(lo, hi) wavelength bounds in Angstrom where throughput > frac*peak."""
    bp = sim.sensor.bandpass
    ws = bp.waveset
    t = bp(ws).value
    m = t > frac * t.max()
    return float(ws[m].min().value), float(ws[m].max().value)


def _weighted_flux(sim, spectrum, ebv=0.0, rv=3.1):
    """(wave_aa, S*T [*extinction]) on a fine grid across the band."""
    lo, hi = band_support(sim)
    wave = np.linspace(lo, hi, _N_WAVE) * u.AA
    st = (spectrum(wave) * sim.sensor.bandpass(wave)).value
    if ebv > 0.0:
        st = st * F99(Rv=rv).extinguish(wave, Ebv=ebv)
    return wave.value, st


def band_nodes(sim, spectrum, n_nodes=7, ebv=0.0, rv=3.1):
    """Node wavelengths [m] and unit-sum weights for the effective PSF.

    Weights are S(lambda)*T(lambda) integrals over n_nodes equal sub-bands;
    each node sits at its sub-band's flux-weighted mean wavelength.
    """
    if n_nodes < 1:
        raise ValueError(f"n_nodes must be >= 1, got {n_nodes}")
    wave_aa, st = _weighted_flux(sim, spectrum, ebv=ebv, rv=rv)
    edges = np.linspace(wave_aa[0], wave_aa[-1], n_nodes + 1)
    idx = np.clip(np.searchsorted(edges, wave_aa) - 1, 0, n_nodes - 1)
    weights = np.zeros(n_nodes)
    waves_aa = 0.5 * (edges[:-1] + edges[1:])  # fallback: bin centers
    for i in range(n_nodes):
        s = st[idx == i]
        weights[i] = s.sum()
        if s.sum() > 0.0:
            waves_aa[i] = np.average(wave_aa[idx == i], weights=s)
    total = weights.sum()
    if total <= 0.0:
        raise ValueError("spectrum has no flux inside the bandpass")
    return waves_aa * 1e-10, weights / total


def effective_psf(sim, focus, spectrum=None, oversample=11, stamp_npix=None,
                  jitter_sigma_mas=None, n_nodes=7, ebv=0.0):
    """Spectrum-weighted oversampled PSF; monochromatic passthrough when
    focus != 0 (no wavelength model), n_nodes == 1, or spectrum is None."""
    if n_nodes < 1:
        raise ValueError(f"n_nodes must be >= 1, got {n_nodes}")
    if focus != 0 or n_nodes == 1 or spectrum is None:
        return render_oversampled_psf(
            sim, focus, oversample=oversample, stamp_npix=stamp_npix,
            jitter_sigma_mas=jitter_sigma_mas,
        )
    waves_m, weights = band_nodes(sim, spectrum, n_nodes=n_nodes, ebv=ebv)
    psf = None
    for w_m, wt in zip(waves_m, weights):
        p = render_oversampled_psf(
            sim, focus, oversample=oversample, stamp_npix=stamp_npix,
            jitter_sigma_mas=jitter_sigma_mas, wavelength_m=w_m,
        )
        psf = wt * p if psf is None else psf + wt * p
    return psf


def effective_psf_for_spt(sim, sensorfilter, spt, ebv, focus, oversample,
                          stamp_npix=None, jitter_sigma_mas=None, n_nodes=7):
    """Cached effective PSF for one spectral template (+ optional reddening).

    `spt` is coerced to str for the wcc_etc type-is-str dispatch. The cache
    key resolves jitter/stamp defaults first so equivalent calls share.
    """
    if stamp_npix is None:
        stamp_npix = DEFAULT_STAMP[focus]
    if jitter_sigma_mas is None:
        jitter_sigma_mas = float(sim.telescope.jitter_sigma.to("mas").value)
    key = (
        sensorfilter, str(spt), round(float(ebv), 4), int(focus),
        int(oversample), int(stamp_npix), round(float(jitter_sigma_mas), 3),
        int(n_nodes),
    )
    if key not in _EFF_PSF_CACHE:
        spectrum = get_scene_element(str(spt), mag=15.0).spectrum
        _EFF_PSF_CACHE[key] = effective_psf(
            sim, focus, spectrum, oversample=oversample,
            stamp_npix=stamp_npix, jitter_sigma_mas=jitter_sigma_mas,
            n_nodes=n_nodes, ebv=float(ebv),
        )
    return _EFF_PSF_CACHE[key]
```

(`lru_cache`, `make_base_simulation`, and `_weighted_flux`'s `rv` are also used by Task 3's `attenuation_factor`, which lands in this same module.)

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_chromatic.py tests/test_psf.py -v`
Expected: all PASS (~1-2 min: several small Airy renders)

- [ ] **Step 5: Commit**

```bash
git add src/wcc_sim/chromatic.py tests/test_chromatic.py
git commit -m "feat: spectrum-weighted chromatic effective PSFs (wcc_sim.chromatic)"
```

---

### Task 3: `attenuation_factor` (F99 reddening) + dust_extinction dependency

**Files:**
- Modify: `src/wcc_sim/chromatic.py` (append function)
- Modify: `pyproject.toml:11-19` (add `dust_extinction` to `dependencies`)
- Test: `tests/test_chromatic.py` (append)

**Interfaces:**
- Produces: `attenuation_factor(template, ebv, sensorfilter, rv=3.1)` → float in (0, 1]; `== 1.0` exactly when `ebv == 0`. Consumed by Tasks 5 and 9.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_chromatic.py`):

```python
def test_attenuation_zero_ebv_is_one():
    from wcc_sim.chromatic import attenuation_factor

    assert attenuation_factor("G2V", 0.0, "zwo:i") == 1.0


def test_attenuation_monotonic_and_band_dependent():
    from wcc_sim.chromatic import attenuation_factor

    a1 = attenuation_factor("G2V", 0.1, "zwo:i")
    a2 = attenuation_factor("G2V", 0.3, "zwo:i")
    assert 0.0 < a2 < a1 < 1.0
    # zwo:i (~690-855 nm) is redder than V: attenuation must be milder
    # than the full A_V = R_V * E(B-V) dimming, but real (< 1).
    av_floor = 10.0 ** (-0.4 * 3.1 * 0.1)
    assert av_floor < a1 < 1.0


def test_attenuation_negative_ebv_raises():
    import pytest as _pytest

    from wcc_sim.chromatic import attenuation_factor

    with _pytest.raises(ValueError):
        attenuation_factor("G2V", -0.1, "zwo:i")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_chromatic.py -k attenuation -v`
Expected: FAIL with `ImportError: cannot import name 'attenuation_factor'`

- [ ] **Step 3: Implement** — append to `src/wcc_sim/chromatic.py`:

```python
@lru_cache(maxsize=256)
def attenuation_factor(template, ebv, sensorfilter, rv=3.1):
    """Band-averaged flux attenuation of `template` for a given E(B-V).

    a = int S T 10^(-0.4 A(lambda)) dlambda / int S T dlambda with a
    Fitzpatrick (1999) curve; the exact broadband dimming factor to apply
    to the unreddened wcc_etc count rate.
    """
    ebv = float(ebv)
    if ebv < 0.0:
        raise ValueError(f"ebv must be >= 0, got {ebv}")
    if ebv == 0.0:
        return 1.0
    sim = make_base_simulation(sensorfilter)
    spectrum = get_scene_element(str(template), mag=15.0).spectrum
    wave_aa, st = _weighted_flux(sim, spectrum)
    trans = F99(Rv=rv).extinguish(wave_aa * u.AA, Ebv=ebv)
    return float(np.trapezoid(st * trans, wave_aa) / np.trapezoid(st, wave_aa))
```

Then in `pyproject.toml` add `"dust_extinction",` to the `dependencies` list (after `"photutils",`).

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_chromatic.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/wcc_sim/chromatic.py tests/test_chromatic.py pyproject.toml
git commit -m "feat: band-averaged F99 attenuation factor for reddened rates"
```

---

### Task 4: `rates_for_catalog` honors an `spt` override column

**Files:**
- Modify: `src/wcc_sim/starflux.py:82-96`
- Test: `tests/test_starflux.py` (append)

**Interfaces:**
- Produces: `rates_for_catalog(catalog, sensorfilter)` unchanged signature; if the catalog has an `spt` column, non-empty entries override the BP−RP lookup row-wise (empty string → fall back). Needed so demo Cepheids carry `F8I` (the BP−RP table maps to dwarfs only).

- [ ] **Step 1: Write the failing test** (append to `tests/test_starflux.py`):

```python
def test_spt_override_column(monkeypatch):
    import wcc_sim.starflux as sf

    rates = {"G2V": 1.0, "M2III": 2.0}
    monkeypatch.setattr(sf, "rate_for_spt", lambda spt, f: rates[str(spt)])
    catalog = Table(
        {
            "phot_g_mean_mag": [15.0, 15.0, 15.0],
            "phot_bp_mean_mag": [np.nan] * 3,
            "phot_rp_mean_mag": [np.nan] * 3,
            # NaN colors -> base type G2V; row 1 overridden (and longer
            # than the base dtype width — must not be truncated)
            "spt": ["", "M2III", ""],
        }
    )
    out_rates, spts = sf.rates_for_catalog(catalog, "zwo:r")
    assert list(spts) == ["G2V", "M2III", "G2V"]
    assert out_rates == pytest.approx([1.0, 2.0, 1.0])
```

(Confirm `tests/test_starflux.py` already imports `numpy as np`, `pytest`, and `Table`; add any missing import at the top.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_starflux.py::test_spt_override_column -v`
Expected: FAIL — `spts` comes back `["G2V", "G2V", "G2V"]` (override ignored)

- [ ] **Step 3: Implement** — in `src/wcc_sim/starflux.py`, inside `rates_for_catalog` after `spts = spt_from_bp_rp(bp - rp)` insert:

```python
    if "spt" in catalog.colnames:
        # Optional per-row override (e.g. supergiant templates for injected
        # Cepheids — the BP-RP table maps to dwarfs only). Empty string
        # means "no override". Merge via object dtype: assigning into the
        # fixed-width array from spt_from_bp_rp would silently truncate
        # longer type names.
        override = np.array(
            [str(s).strip() for s in catalog["spt"]], dtype=object
        )
        merged = spts.astype(object)
        use = override != ""
        merged[use] = override[use]
        spts = merged.astype(str)
```

Also update the docstring line to: `"""Per-star (rate_e_s, spt) arrays for a Gaia catalog Table.\n\n    An optional `spt` column overrides the BP-RP lookup row-wise (empty\n    string = no override).\n    """`

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_starflux.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/wcc_sim/starflux.py tests/test_starflux.py
git commit -m "feat: optional spt override column in rates_for_catalog"
```

---

### Task 5: `wcc_sim.extended` — SersicComponent + analytic normalization

**Files:**
- Create: `src/wcc_sim/extended.py`
- Test: `tests/test_extended.py` (create)

**Interfaces:**
- Consumes: `rate_for_spt`, `REF_MAG` (starflux); `attenuation_factor` (Task 3).
- Produces (Tasks 6-9 rely on these exact names):
  - `SersicComponent(ra, dec, n, r_eff_arcsec, ellip=0.0, pa_deg=0.0, total_mag=None, sb_mag_arcsec2=None, template="G2V", ebv=0.0)` — frozen dataclass, validates on construction. `pa_deg` = major-axis angle, degrees CCW from the +x detector axis.
  - `sersic_total_over_amplitude(n, r_eff_pix, ellip)` → float (F_total / amplitude, analytic).
  - `component_amplitude(comp, sensorfilter, plate_scale_mas)` → Sérsic2D amplitude in e⁻/s/pix (reddening applied).

- [ ] **Step 1: Write the failing tests** — create `tests/test_extended.py`:

```python
import numpy as np
import pytest

from tests.conftest import DEC0, RA0

PLATE_MAS = 16.87  # zwo plate scale, close enough for unit tests


def _comp(**kw):
    from wcc_sim.extended import SersicComponent

    base = dict(ra=RA0, dec=DEC0, n=1.0, r_eff_arcsec=0.1, total_mag=15.0)
    base.update(kw)
    return SersicComponent(**base)


def test_validation():
    from wcc_sim.extended import SersicComponent

    with pytest.raises(ValueError):  # both normalizations
        _comp(sb_mag_arcsec2=20.0)
    with pytest.raises(ValueError):  # neither
        SersicComponent(ra=RA0, dec=DEC0, n=1.0, r_eff_arcsec=0.1)
    with pytest.raises(ValueError):
        _comp(n=0.0)
    with pytest.raises(ValueError):
        _comp(r_eff_arcsec=0.0)
    with pytest.raises(ValueError):
        _comp(ellip=1.0)
    with pytest.raises(ValueError):
        _comp(ebv=-0.1)


def test_total_over_amplitude_matches_numeric_integral():
    from astropy.modeling.models import Sersic2D

    from wcc_sim.extended import sersic_total_over_amplitude

    n, r_eff, ellip = 1.0, 6.0, 0.3
    factor = sersic_total_over_amplitude(n, r_eff, ellip)
    mod = Sersic2D(amplitude=1.0, r_eff=r_eff, n=n, x_0=0.0, y_0=0.0,
                   ellip=ellip, theta=0.0)
    g = np.linspace(-200.0, 200.0, 2001)  # 0.2 px sampling to r ~ 33 r_eff
    numeric = mod(g[None, :], g[:, None]).sum() * (g[1] - g[0]) ** 2
    assert numeric == pytest.approx(factor, rel=2e-3)


def test_component_amplitude_total_mag_roundtrip():
    from wcc_sim.extended import component_amplitude, sersic_total_over_amplitude
    from wcc_sim.starflux import REF_MAG, rate_for_spt

    comp = _comp(total_mag=REF_MAG)
    amp = component_amplitude(comp, "zwo:r", PLATE_MAS)
    r_eff_pix = comp.r_eff_arcsec * 1000.0 / PLATE_MAS
    total = amp * sersic_total_over_amplitude(comp.n, r_eff_pix, comp.ellip)
    assert total == pytest.approx(rate_for_spt("G2V", "zwo:r"), rel=1e-6)


def test_component_amplitude_surface_brightness():
    from wcc_sim.extended import component_amplitude
    from wcc_sim.starflux import REF_MAG, rate_for_spt

    comp = _comp(total_mag=None, sb_mag_arcsec2=REF_MAG)
    amp = component_amplitude(comp, "zwo:r", PLATE_MAS)
    pix_arcsec2 = (PLATE_MAS / 1000.0) ** 2
    assert amp == pytest.approx(
        rate_for_spt("G2V", "zwo:r") * pix_arcsec2, rel=1e-6
    )


def test_reddening_dims_amplitude():
    from wcc_sim.extended import component_amplitude

    a0 = component_amplitude(_comp(), "zwo:r", PLATE_MAS)
    a1 = component_amplitude(_comp(ebv=0.3), "zwo:r", PLATE_MAS)
    assert a1 < a0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extended.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wcc_sim.extended'`

- [ ] **Step 3: Implement** — create `src/wcc_sim/extended.py`:

```python
"""Analytic extended-source (Sersic) components for wcc-sim frames.

Components are rendered in e-/s/pix on the NATIVE detector grid and
convolved once per (template, ebv) group with an effective-PSF kernel
(FFT). The native grid is adequate because the in-focus PSF FWHM is
~3 px; the cuspy Sersic center is the exception and is re-evaluated on a
refined subgrid. Rendering on the 11x point-source grid would need ~TB of
memory for a full frame; sampling profiles as dense point grids would need
~1e6 stamp placements per component. See the design spec
(docs/superpowers/specs/2026-07-02-extended-sources-cepheids-design.md).
"""

from dataclasses import dataclass

import numpy as np
from astropy.modeling.models import Sersic2D
from scipy.signal import fftconvolve
from scipy.special import gamma, gammaincinv

from .chromatic import attenuation_factor
from .starflux import REF_MAG, rate_for_spt

_REFINE = 9         # subgrid factor for the central cusp
_REFINE_HALF_MIN = 8   # px; refinement box half-width bounds
_REFINE_HALF_MAX = 64


@dataclass(frozen=True)
class SersicComponent:
    """One elliptical Sersic component.

    Exactly one of `total_mag` (integrated Gaia-G vegamag, same
    normalization convention as point sources) or `sb_mag_arcsec2`
    (surface brightness at r_eff, mag/arcsec^2) sets the flux. `pa_deg`
    is the major-axis angle in degrees CCW from the +x detector axis.
    n=1 is an exponential disk, n=4 a de Vaucouleurs bulge.
    """

    ra: float
    dec: float
    n: float
    r_eff_arcsec: float
    ellip: float = 0.0
    pa_deg: float = 0.0
    total_mag: float = None
    sb_mag_arcsec2: float = None
    template: str = "G2V"
    ebv: float = 0.0

    def __post_init__(self):
        if (self.total_mag is None) == (self.sb_mag_arcsec2 is None):
            raise ValueError(
                "set exactly one of total_mag / sb_mag_arcsec2"
            )
        if not self.n > 0.0:
            raise ValueError(f"n must be > 0, got {self.n}")
        if not self.r_eff_arcsec > 0.0:
            raise ValueError(
                f"r_eff_arcsec must be > 0, got {self.r_eff_arcsec}"
            )
        if not 0.0 <= self.ellip < 1.0:
            raise ValueError(f"ellip must be in [0, 1), got {self.ellip}")
        if self.ebv < 0.0:
            raise ValueError(f"ebv must be >= 0, got {self.ebv}")


def sersic_total_over_amplitude(n, r_eff_pix, ellip):
    """F_total / amplitude for a Sersic2D profile (analytic).

    Sersic2D's `amplitude` is the surface brightness at r_eff per pixel
    area; integrating the profile over the plane gives
    2 pi n r_eff^2 (1-ellip) e^bn bn^(-2n) Gamma(2n).
    """
    bn = float(gammaincinv(2.0 * n, 0.5))
    return float(
        2.0 * np.pi * n * r_eff_pix**2 * (1.0 - ellip)
        * np.exp(bn) * bn ** (-2.0 * n) * gamma(2.0 * n)
    )


def component_amplitude(comp, sensorfilter, plate_scale_mas):
    """Sersic2D amplitude in e-/s/pix for the (reddened) component."""
    att = attenuation_factor(comp.template, comp.ebv, sensorfilter)
    rate_ref = rate_for_spt(comp.template, sensorfilter)
    pix_arcsec = plate_scale_mas / 1000.0
    if comp.total_mag is not None:
        total = rate_ref * 10.0 ** (-0.4 * (comp.total_mag - REF_MAG)) * att
        r_eff_pix = comp.r_eff_arcsec / pix_arcsec
        return total / sersic_total_over_amplitude(
            comp.n, r_eff_pix, comp.ellip
        )
    sb_rate = rate_ref * 10.0 ** (-0.4 * (comp.sb_mag_arcsec2 - REF_MAG)) * att
    return sb_rate * pix_arcsec**2
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_extended.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/wcc_sim/extended.py tests/test_extended.py
git commit -m "feat: SersicComponent with analytic flux normalization and reddening"
```

---

### Task 6: `render_component_profile` — native-grid evaluation with central refinement

**Files:**
- Modify: `src/wcc_sim/extended.py` (append)
- Test: `tests/test_extended.py` (append)

**Interfaces:**
- Consumes: `component_amplitude` (Task 5); a `WCS` built with `wcc_sim.wcsutil.build_wcs(ra, dec, plate_scale_mas, pa, shape)`.
- Produces: `render_component_profile(comp, wcs, shape, plate_scale_mas, sensorfilter)` → float32 (ny, nx) e⁻/s/pix array, **unconvolved**; returns `None` when the component's 99.9%-flux radius misses the frame entirely. Task 7 consumes.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_extended.py`):

```python
SHAPE = (256, 256)


def _wcs(shape=SHAPE):
    from wcc_sim.wcsutil import build_wcs

    return build_wcs(RA0, DEC0, PLATE_MAS, 0.0, shape)


def _moments(img):
    ny, nx = img.shape
    yy, xx = np.mgrid[:ny, :nx]
    t = img.sum()
    cx, cy = (img * xx).sum() / t, (img * yy).sum() / t
    return (
        (img * (xx - cx) ** 2).sum() / t,
        (img * (yy - cy) ** 2).sum() / t,
    )


def test_profile_flux_conserved_exponential():
    from wcc_sim.extended import (
        component_amplitude,
        render_component_profile,
        sersic_total_over_amplitude,
    )

    comp = _comp(n=1.0, r_eff_arcsec=0.05, total_mag=18.0)  # r_eff ~ 3 px
    img = render_component_profile(comp, _wcs(), SHAPE, PLATE_MAS, "zwo:r")
    amp = component_amplitude(comp, "zwo:r", PLATE_MAS)
    r_eff_pix = comp.r_eff_arcsec * 1000.0 / PLATE_MAS
    total = amp * sersic_total_over_amplitude(comp.n, r_eff_pix, comp.ellip)
    assert img.dtype == np.float32
    assert img.sum() == pytest.approx(total, rel=0.005)


def test_profile_flux_conserved_devauc():
    from wcc_sim.extended import (
        component_amplitude,
        render_component_profile,
        sersic_total_over_amplitude,
    )

    comp = _comp(n=4.0, r_eff_arcsec=0.02, total_mag=18.0)  # cuspy center
    img = render_component_profile(comp, _wcs(), SHAPE, PLATE_MAS, "zwo:r")
    amp = component_amplitude(comp, "zwo:r", PLATE_MAS)
    r_eff_pix = comp.r_eff_arcsec * 1000.0 / PLATE_MAS
    total = amp * sersic_total_over_amplitude(comp.n, r_eff_pix, comp.ellip)
    # n=4 keeps ~1% of its flux beyond the 128 px frame half-width;
    # require the rendered sum to land between 97% and 100.5% of analytic.
    assert 0.97 * total < img.sum() < 1.005 * total


def test_profile_orientation():
    from wcc_sim.extended import render_component_profile

    ex = _comp(ellip=0.6, pa_deg=0.0, r_eff_arcsec=0.2)
    ey = _comp(ellip=0.6, pa_deg=90.0, r_eff_arcsec=0.2)
    ix = render_component_profile(ex, _wcs(), SHAPE, PLATE_MAS, "zwo:r")
    iy = render_component_profile(ey, _wcs(), SHAPE, PLATE_MAS, "zwo:r")
    vxx_x, vyy_x = _moments(ix)
    vxx_y, vyy_y = _moments(iy)
    assert vxx_x > vyy_x  # pa=0: major axis along +x
    assert vyy_y > vxx_y  # pa=90: rotated onto +y


def test_far_off_frame_component_skipped():
    from wcc_sim.extended import render_component_profile

    comp = _comp(dec=DEC0 + 5.0, r_eff_arcsec=0.5)
    assert (
        render_component_profile(comp, _wcs(), SHAPE, PLATE_MAS, "zwo:r")
        is None
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extended.py -k profile -v`
Expected: FAIL with `ImportError: cannot import name 'render_component_profile'`

- [ ] **Step 3: Implement** — append to `src/wcc_sim/extended.py`:

```python
def _r999_pix(n, r_eff_pix):
    """Radius enclosing 99.9% of the Sersic flux, in pixels."""
    bn = float(gammaincinv(2.0 * n, 0.5))
    return r_eff_pix * (float(gammaincinv(2.0 * n, 0.999)) / bn) ** n


def render_component_profile(comp, wcs, shape, plate_scale_mas, sensorfilter):
    """Component surface brightness in e-/s/pix on the native grid
    (unconvolved); None if the 99.9%-flux footprint misses the frame.

    The profile is evaluated at pixel centers in row chunks (full-frame
    float64 temporaries would be ~GB); a box around the center is
    re-evaluated on a _REFINE x subgrid and averaged, since a Sersic cusp
    changes across a pixel while the rest of the profile does not.
    """
    ny, nx = shape
    x0, y0 = (float(v) for v in wcs.world_to_pixel_values(comp.ra, comp.dec))
    r_eff_pix = comp.r_eff_arcsec * 1000.0 / plate_scale_mas
    dx = max(0.0, -x0, x0 - (nx - 1))
    dy = max(0.0, -y0, y0 - (ny - 1))
    if np.hypot(dx, dy) > _r999_pix(comp.n, r_eff_pix):
        return None

    amp = component_amplitude(comp, sensorfilter, plate_scale_mas)
    mod = Sersic2D(
        amplitude=amp, r_eff=r_eff_pix, n=comp.n, x_0=x0, y_0=y0,
        ellip=comp.ellip, theta=np.radians(comp.pa_deg),
    )

    img = np.empty(shape, dtype=np.float32)
    xx = np.arange(nx, dtype=float)[None, :]
    for y_lo in range(0, ny, 1024):
        y_hi = min(y_lo + 1024, ny)
        yy = np.arange(y_lo, y_hi, dtype=float)[:, None]
        img[y_lo:y_hi] = mod(xx, yy)

    half = int(np.clip(np.ceil(2.0 * r_eff_pix), _REFINE_HALF_MIN,
                       _REFINE_HALF_MAX))
    bx_lo, bx_hi = max(int(x0) - half, 0), min(int(x0) + half + 1, nx)
    by_lo, by_hi = max(int(y0) - half, 0), min(int(y0) + half + 1, ny)
    if bx_lo < bx_hi and by_lo < by_hi:
        off = (np.arange(_REFINE) + 0.5) / _REFINE - 0.5
        fx = (np.arange(bx_lo, bx_hi, dtype=float)[:, None] + off).ravel()
        fy = (np.arange(by_lo, by_hi, dtype=float)[:, None] + off).ravel()
        fine = mod(fx[None, :], fy[:, None])
        img[by_lo:by_hi, bx_lo:bx_hi] = fine.reshape(
            by_hi - by_lo, _REFINE, bx_hi - bx_lo, _REFINE
        ).mean(axis=(1, 3))
    return img
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_extended.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/wcc_sim/extended.py tests/test_extended.py
git commit -m "feat: native-grid Sersic profile rendering with central refinement"
```

---

### Task 7: `render_extended` — grouped FFT convolution with wing renormalization

**Files:**
- Modify: `src/wcc_sim/extended.py` (append)
- Test: `tests/test_extended.py` (append)

**Interfaces:**
- Consumes: `render_component_profile` (Task 6); `wcc_sim.wings.WingModel.energy_beyond(r)`.
- Produces: `render_extended(components, wcs, shape, plate_scale_mas, sensorfilter, kernels, wing=None)` → float32 (ny, nx) e⁻/s/pix, PSF-convolved. `kernels` is `{(template, ebv): 2-D native-resolution kernel}` — one entry per distinct `(comp.template, float(comp.ebv))` key; kernels are normalized to unit sum internally, then scaled by `1/(1 + wing.energy_beyond(half))` when `wing` is given (the same flux-truncation convention as `add_star`; no halo is drawn for extended light). Task 8 consumes.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_extended.py`):

```python
def _mono_kernel(stamp_npix=33, oversample=3):
    from wcc_sim.detectors import make_base_simulation
    from wcc_sim.psf import render_oversampled_psf
    from wcc_sim.render import bin_oversampled

    sim = make_base_simulation("zwo:r")
    psf = render_oversampled_psf(sim, 0, oversample=oversample,
                                 stamp_npix=stamp_npix)
    return bin_oversampled(psf, oversample)


def test_render_extended_conserves_flux():
    from wcc_sim.extended import render_extended, render_component_profile

    comp = _comp(n=1.0, r_eff_arcsec=0.05, total_mag=18.0)
    kern = _mono_kernel()
    img = render_extended([comp], _wcs(), SHAPE, PLATE_MAS, "zwo:r",
                          {(comp.template, comp.ebv): kern})
    prof = render_component_profile(comp, _wcs(), SHAPE, PLATE_MAS, "zwo:r")
    assert img.dtype == np.float32
    # unit-sum kernel: convolution preserves flux (edge losses negligible
    # for a compact centered component)
    assert img.sum() == pytest.approx(prof.sum(), rel=0.005)
    # convolution spreads the profile: peak must drop
    assert img.max() < prof.max()


def test_render_extended_groups_by_template_and_ebv():
    from wcc_sim.extended import render_extended

    c1 = _comp(n=1.0, r_eff_arcsec=0.1, total_mag=17.0, template="G2V")
    c2 = _comp(n=1.0, r_eff_arcsec=0.2, total_mag=17.5, template="K0V")
    kern = _mono_kernel()
    kernels = {("G2V", 0.0): kern, ("K0V", 0.0): kern}
    both = render_extended([c1, c2], _wcs(), SHAPE, PLATE_MAS, "zwo:r",
                           kernels)
    solo1 = render_extended([c1], _wcs(), SHAPE, PLATE_MAS, "zwo:r", kernels)
    solo2 = render_extended([c2], _wcs(), SHAPE, PLATE_MAS, "zwo:r", kernels)
    assert np.allclose(both, solo1 + solo2, atol=1e-4)


def test_render_extended_wing_renormalization():
    from wcc_sim.extended import render_extended
    from wcc_sim.wings import WingModel

    comp = _comp(n=1.0, r_eff_arcsec=0.05, total_mag=18.0)
    kern = _mono_kernel()
    kernels = {(comp.template, comp.ebv): kern}
    wing = WingModel(c=0.01, alpha=-3.0, r_in=float(kern.shape[0] // 2))
    plain = render_extended([comp], _wcs(), SHAPE, PLATE_MAS, "zwo:r",
                            kernels)
    winged = render_extended([comp], _wcs(), SHAPE, PLATE_MAS, "zwo:r",
                             kernels, wing=wing)
    factor = 1.0 + wing.energy_beyond(kern.shape[0] // 2)
    assert winged.sum() == pytest.approx(plain.sum() / factor, rel=1e-4)


def test_render_extended_all_off_frame():
    from wcc_sim.extended import render_extended

    comp = _comp(dec=DEC0 + 5.0)
    img = render_extended([comp], _wcs(), SHAPE, PLATE_MAS, "zwo:r",
                          {(comp.template, comp.ebv): _mono_kernel()})
    assert img.sum() == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extended.py -k render_extended -v`
Expected: FAIL with `ImportError: cannot import name 'render_extended'`

- [ ] **Step 3: Implement** — append to `src/wcc_sim/extended.py`:

```python
def render_extended(components, wcs, shape, plate_scale_mas, sensorfilter,
                    kernels, wing=None):
    """PSF-convolved sum of all components, e-/s/pix on the native grid.

    `kernels` maps (template, ebv) -> native-resolution PSF kernel; the
    caller decides whether those are chromatic effective PSFs or copies of
    the monochromatic one. Kernels are normalized to unit sum here and,
    when a WingModel is given, scaled by 1/(1 + energy_beyond(half)) so
    extended flux follows the same stamp-truncation convention as
    add_star. No wing halo is drawn: for smooth extended light the halo
    is a sub-noise redistribution.
    """
    image = np.zeros(shape, dtype=np.float32)
    groups = {}
    for comp in components:
        groups.setdefault((comp.template, float(comp.ebv)), []).append(comp)
    for key, comps in groups.items():
        sub = None
        for comp in comps:
            prof = render_component_profile(
                comp, wcs, shape, plate_scale_mas, sensorfilter
            )
            if prof is not None:
                sub = prof if sub is None else sub + prof
        if sub is None:
            continue
        kern = np.asarray(kernels[key], dtype=np.float64)
        kern = kern / kern.sum()
        if wing is not None:
            kern = kern / (1.0 + wing.energy_beyond(kern.shape[0] // 2))
        image += fftconvolve(sub, kern, mode="same").astype(np.float32)
    return image
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_extended.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/wcc_sim/extended.py tests/test_extended.py
git commit -m "feat: render_extended — grouped FFT convolution with wing renormalization"
```

---

### Task 8: `simulate_field` integration (`extended_sources`, `chromatic`) + FITS cards

**Files:**
- Modify: `src/wcc_sim/pipeline.py` (signature, PSF/rendering section, params)
- Modify: `src/wcc_sim/fitswriter.py:14-37` (two new cards)
- Modify: `src/wcc_sim/__init__.py` (export `SersicComponent`)
- Test: `tests/test_pipeline.py` (append)

**Interfaces:**
- Consumes: `effective_psf_for_spt` (Task 2), `render_extended` (Task 7), `bin_oversampled`, `fit_wing_model`.
- Produces: `simulate_field(..., extended_sources=None, chromatic=False)`. `params` gains `"chromatic"` (bool) and `"n_extended"` (int). FITS header gains `CHROMPSF` and `NEXTSRC`. `wcc_sim.SersicComponent` importable from the package root. Task 9 consumes.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_pipeline.py`):

```python
def test_extended_source_adds_flux(canned_catalog):
    from wcc_sim import SersicComponent
    from wcc_sim.extended import component_amplitude, sersic_total_over_amplitude

    comp = SersicComponent(ra=RA0, dec=DEC0, n=1.0, r_eff_arcsec=0.05,
                           total_mag=16.0)
    single = canned_catalog[[3]]  # one faint star only
    base = run(single, add_noise=False)
    ext = run(single, add_noise=False, extended_sources=[comp])
    extra = float(ext.image_clean.sum() - base.image_clean.sum())
    plate = base.params["plate_scale_mas"]
    amp = component_amplitude(comp, "zwo:r", plate)
    r_eff_pix = comp.r_eff_arcsec * 1000.0 / plate
    expected = (
        amp * sersic_total_over_amplitude(comp.n, r_eff_pix, comp.ellip)
        * base.params["exptime"]
    )
    # wing renormalization holds back the stamp-truncated fraction
    assert extra == pytest.approx(expected, rel=0.02)
    assert ext.params["n_extended"] == 1
    assert base.params["n_extended"] == 0


def test_extended_in_fits_header(canned_catalog, tmp_path):
    from wcc_sim import SersicComponent

    comp = SersicComponent(ra=RA0, dec=DEC0, n=1.0, r_eff_arcsec=0.05,
                           total_mag=16.0)
    path = tmp_path / "ext.fits"
    run(canned_catalog, extended_sources=[comp], output=str(path))
    from astropy.io import fits

    header = fits.getheader(path, "SCI")
    assert header["NEXTSRC"] == 1
    assert header["CHROMPSF"] is False


def test_chromatic_changes_in_focus_image(canned_catalog):
    mono = run(canned_catalog, add_noise=False)
    chrom = run(canned_catalog, add_noise=False, chromatic=True)
    assert chrom.params["chromatic"] is True
    assert not np.array_equal(mono.image_clean, chrom.image_clean)
    # rates are untouched: total flux agrees to the wing-truncation level
    assert chrom.image_clean.sum() == pytest.approx(
        mono.image_clean.sum(), rel=0.01
    )


def test_chromatic_defocus_is_passthrough(canned_catalog):
    mono = run(canned_catalog, add_noise=False, focus=1, stamp_npix=65)
    chrom = run(canned_catalog, add_noise=False, focus=1, stamp_npix=65,
                chromatic=True)
    assert np.array_equal(mono.image_clean, chrom.image_clean)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline.py -k "extended or chromatic" -v`
Expected: FAIL with `TypeError: simulate_field() got an unexpected keyword argument 'extended_sources'`

- [ ] **Step 3: Implement.** In `src/wcc_sim/pipeline.py`:

3a. Imports — add:

```python
from .chromatic import effective_psf_for_spt
from .extended import render_extended
```

3b. Signature — after `wings=True,` add:

```python
    extended_sources=None,
    chromatic=False,
```

and extend the docstring with:

```
    `extended_sources` (list of wcc_sim.extended.SersicComponent) adds
    smooth analytic components, rendered at native resolution and
    FFT-convolved with the PSF before the noise model. `chromatic=True`
    replaces the central-wavelength PSF with spectrum-weighted effective
    PSFs — per spectral type for point sources and per (template, ebv)
    for extended components; it is a documented no-op for focus != 0
    (the defocus PSF has no wavelength model).
```

3c. Rendering — replace the single `image_sources = render_scene(...)` call (pipeline.py:133-136) with:

```python
    chromatic_active = bool(chromatic) and focus == 0

    def _wing_for(psf):
        return fit_wing_model(bin_oversampled(psf, oversample)) if wings else None

    if chromatic_active and len(catalog):
        image_sources = np.zeros(shape, dtype=np.float32)
        for spt in np.unique(spts):
            sel = spts == spt
            psf_spt = effective_psf_for_spt(
                sim, sensorfilter, spt, 0.0, focus, oversample,
                stamp_npix=n_stamp, jitter_sigma_mas=jitter_sigma_mas,
            )
            image_sources += render_scene(
                shape, xs[sel], ys[sel], rates[sel] * exptime, psf_spt,
                oversample, wing=_wing_for(psf_spt), floor_e=wing_floor_e,
            )
    else:
        image_sources = render_scene(
            shape, xs, ys, rates * exptime, psf_os, oversample,
            wing=wing, floor_e=wing_floor_e,
        )

    if extended_sources:
        kernels = {}
        for comp in extended_sources:
            key = (comp.template, float(comp.ebv))
            if key in kernels:
                continue
            if chromatic_active:
                psf_ext = effective_psf_for_spt(
                    sim, sensorfilter, comp.template, comp.ebv, focus,
                    oversample, stamp_npix=n_stamp,
                    jitter_sigma_mas=jitter_sigma_mas,
                )
            else:
                psf_ext = psf_os
            kernels[key] = bin_oversampled(psf_ext, oversample)
        image_sources += render_extended(
            extended_sources, wcs, shape, geom.plate_scale_mas,
            sensorfilter, kernels, wing=wing,
        ) * np.float32(exptime)
```

Notes: the `else` branch is byte-for-byte the current call — defaults stay bit-identical. The existing mono `psf_os`/`wing`/`wing_floor_e` computations above stay unchanged (the mono wing also renormalizes the extended kernels; per-template wing differences are negligible there).

3d. Params — add to the `params` dict:

```python
        "chromatic": bool(chromatic),
        "n_extended": len(extended_sources) if extended_sources else 0,
```

3e. `src/wcc_sim/fitswriter.py` — add to `_CARDS`:

```python
    "chromatic": ("CHROMPSF", "spectrum-weighted effective PSFs used"),
    "n_extended": ("NEXTSRC", "number of extended (Sersic) components"),
```

3f. `src/wcc_sim/__init__.py` — export alongside the existing names:

```python
from .extended import SersicComponent
```

(and add `"SersicComponent"` to `__all__` if the module defines one).

- [ ] **Step 4: Run the new tests AND the full suite (regression guard)**

Run: `python -m pytest tests/ -x -q`
Expected: everything PASSES; no existing test changes behavior.

- [ ] **Step 5: Commit**

```bash
git add src/wcc_sim/pipeline.py src/wcc_sim/fitswriter.py src/wcc_sim/__init__.py tests/test_pipeline.py
git commit -m "feat: extended sources and chromatic PSFs in simulate_field"
```

---

### Task 9: `scripts/cepheids_m101.py` — end-to-end Cepheid P-L demo

**Files:**
- Create: `scripts/cepheids_m101.py`
- No unit test — the script's three PASS/FAIL checks are the integration test (mirrors `scripts/transit_trappist1b.py`).

**Interfaces:**
- Consumes: `simulate_field(extended_sources=, chromatic=)`, `SersicComponent`, `attenuation_factor`, `query_gaia`, `build_wcs`, `get_geometry`; `wcc_phot.run_photometry`, `wcc_phot.io.load_frame`; `astropy.timeseries.LombScargle`.
- Produces: `cepheid_out/` with epoch FITS frames, `cepheids_injected.ecsv`, `cepheids_recovered.ecsv`, `cepheid_pl.png`; exit 0 iff all checks pass. `--quick` runs a reduced config (8 epochs, 8 Cepheids) for smoke-testing and always exits 0.

- [ ] **Step 1: Sanity-check the two galaxy templates exist** (fail fast before writing 300 lines):

Run: `python -c "from wcc_etc.scene import get_scene_element; [get_scene_element(s, mag=15.0) for s in ('G2V','K0III','F8I')]; print('ok')"`
Expected: `ok`. If `K0III` fails, substitute `K3III`, then `K0V`, in the `EXTENDED` list below.

- [ ] **Step 2: Write the script** — create `scripts/cepheids_m101.py`:

```python
"""End-to-end M101 Cepheid period-luminosity simulation + retrieval.

Run:  $PY scripts/cepheids_m101.py [--quick]
(network needed on first run for the Gaia foreground query; --quick runs a
reduced 8-Cepheid / 8-epoch smoke configuration and always exits 0)

Injects 30 synthetic classical Cepheids (F8I supergiant template,
P = 10-60 d, demo Leavitt law M_G = -2.9 (log P - 1) - 4.2 at
mu = 29.2) plus 15 constant F8I calibration stars into a WCC field on the
M101 disk, on top of the galaxy's extended light (exponential disk +
de Vaucouleurs bulge Sersic components) and the real Gaia foreground.
Each Cepheid carries Milky-Way + internal F99 reddening applied as a
band-averaged attenuation of its count rate. 28 epochs over 180 d are
simulated in zwo:i, in focus, with chromatic effective PSFs; wcc_phot PSF
photometry recovers per-epoch fluxes, a per-epoch zero point from the
constant stars calibrates them to magnitudes, Lomb-Scargle recovers the
periods, and a weighted P-L fit recovers the distance modulus.

Checks (exit nonzero on failure, except under --quick):
  1. >= 80% of Cepheids with median per-epoch SNR > 10 recover P to 1%.
  2. P-L intercept -> distance modulus within 3 sigma of the injected
     29.2 after correcting for the known mean extinction.
  3. chi2/dof of the calibrated light curves against the injected model
     in [0.5, 2].

Writes frames, ECSV tables, and a summary figure to ./cepheid_out/.
"""

import argparse
import os
import sys
import time as walltime
import warnings

import numpy as np
from astropy.table import Table, vstack
from astropy.timeseries import LombScargle

from wcc_phot import run_photometry
from wcc_phot.io import load_frame
from wcc_sim import SersicComponent, simulate_field
from wcc_sim.catalog import COLUMNS as GAIA_COLUMNS
from wcc_sim.catalog import query_gaia
from wcc_sim.chromatic import attenuation_factor
from wcc_sim.detectors import get_geometry
from wcc_sim.wcsutil import build_wcs

# --- observation setup -----------------------------------------------------
GAL_RA, GAL_DEC = 210.80243, 54.34875  # M101 center
MU = 29.2  # distance modulus (6.9 Mpc)
FIELD_DRA_ARCMIN = 3.5  # pointing offset east along RA
SENSORFILTER = "zwo:i"
FOCUS = 0
SHAPE = (2048, 2048)  # 34.5" x 34.5"
EXPTIME = 1200.0  # s
N_READS = 2
N_EPOCHS = 28
BASELINE_D = 180.0
SEED = 42
OUT = "cepheid_out"

# --- injected population ---------------------------------------------------
N_CEPH = 30
N_CONST = 15  # constant F8I calibration stars
TEMPLATE = "F8I"
PL_SLOPE, PL_ZP = -2.9, -4.2  # demo Leavitt law: M_G = PL_SLOPE*(logP-1)+PL_ZP
P_RANGE = (10.0, 60.0)  # d
AMP_RANGE = (0.3, 0.7)  # peak-to-peak amplitude, mag
CONST_G_RANGE = (20.5, 22.0)
EBV_MW = 0.008  # SF11 foreground toward M101
EBV_INT_MED, EBV_INT_SIG = 0.10, 0.5  # internal reddening: lognormal
RISE = 0.2  # rising-branch fraction of the pulsation cycle
SYNTH_ID0 = 9_100_000_000_000_000_000

# M101 extended light: exponential disk + small de Vaucouleurs bulge, both
# centered on the (off-frame) galaxy nucleus 3.5' away.
EXTENDED = [
    SersicComponent(ra=GAL_RA, dec=GAL_DEC, n=1.0, r_eff_arcsec=250.0,
                    total_mag=8.0, ellip=0.1, pa_deg=35.0,
                    template="G2V", ebv=0.03),
    SersicComponent(ra=GAL_RA, dec=GAL_DEC, n=4.0, r_eff_arcsec=25.0,
                    total_mag=11.0, ellip=0.05, pa_deg=35.0,
                    template="K0III", ebv=0.02),
]

RA_F = GAL_RA + (FIELD_DRA_ARCMIN / 60.0) / np.cos(np.radians(GAL_DEC))
DEC_F = GAL_DEC


def cepheid_template(phase):
    """Zero-mean asymmetric light-curve template in mag (span -0.5..+0.5).

    Fast rise over the first RISE of the cycle (mag decreasing = getting
    brighter), slow linear decline over the rest.
    """
    ph = np.asarray(phase, dtype=float) % 1.0
    return np.where(
        ph < RISE,
        0.5 - ph / RISE,
        (ph - RISE) / (1.0 - RISE) - 0.5,
    )


def make_population(rng):
    """Injected Cepheid + constant-star tables (no positions yet)."""
    logp = rng.uniform(np.log10(P_RANGE[0]), np.log10(P_RANGE[1]), N_CEPH)
    ebv = EBV_MW + rng.lognormal(np.log(EBV_INT_MED), EBV_INT_SIG, N_CEPH)
    a_g = np.array(
        [
            -2.5 * np.log10(attenuation_factor(TEMPLATE, e, SENSORFILTER))
            for e in ebv
        ]
    )
    ceph = Table(
        {
            "source_id": SYNTH_ID0 + np.arange(N_CEPH, dtype=np.int64),
            "period_d": 10.0**logp,
            "logp": logp,
            "m0": MU + PL_SLOPE * (logp - 1.0) + PL_ZP + a_g,
            "amp": rng.uniform(*AMP_RANGE, N_CEPH),
            "phase0": rng.uniform(0.0, 1.0, N_CEPH),
            "ebv": ebv,
            "a_g": a_g,
        }
    )
    const = Table(
        {
            "source_id": SYNTH_ID0 + N_CEPH + np.arange(N_CONST,
                                                        dtype=np.int64),
            "gmag": rng.uniform(*CONST_G_RANGE, N_CONST),
        }
    )
    return ceph, const


def cepheid_mag(ceph, t_d):
    """(n_ceph, n_epoch) injected G magnitudes at times t_d [d]."""
    phase = (t_d[None, :] / ceph["period_d"][:, None]
             + ceph["phase0"][:, None])
    return (np.asarray(ceph["m0"])[:, None]
            + np.asarray(ceph["amp"])[:, None] * cepheid_template(phase))


def sample_positions(rng, n, shape, margin=200, min_sep=60):
    """n (x, y) positions, rejection-sampled for min pairwise separation."""
    ny, nx = shape
    xs, ys = [], []
    while len(xs) < n:
        x = rng.uniform(margin, nx - 1 - margin)
        y = rng.uniform(margin, ny - 1 - margin)
        if all(np.hypot(x - a, y - b) >= min_sep for a, b in zip(xs, ys)):
            xs.append(x)
            ys.append(y)
    return np.array(xs), np.array(ys)


def make_catalog(wcs, ceph, const, rng, shape):
    """Gaia foreground + synthetic rows (epoch mags filled in later)."""
    half_diag = 0.5 * np.hypot(*shape) * 16.87 / 1000.0 + 10.0
    gaia = query_gaia(RA_F, DEC_F, half_diag, mag_limit=21.0,
                      cache_dir=os.path.join(OUT, "cache"))
    gaia = gaia[GAIA_COLUMNS]  # drop any extras; keep the canonical set
    gaia["spt"] = np.array([""] * len(gaia), dtype="U8")

    n_synth = len(ceph) + len(const)
    xs, ys = sample_positions(rng, n_synth, shape)
    ras, decs = wcs.pixel_to_world_values(xs, ys)
    synth = Table(
        {
            "source_id": np.concatenate(
                [ceph["source_id"], const["source_id"]]
            ).astype(np.int64),
            "ra": np.asarray(ras, dtype=float),
            "dec": np.asarray(decs, dtype=float),
            "phot_g_mean_mag": np.concatenate(
                [np.asarray(ceph["m0"]), np.asarray(const["gmag"])]
            ),
            "phot_bp_mean_mag": np.full(n_synth, np.nan),
            "phot_rp_mean_mag": np.full(n_synth, np.nan),
            "spt": np.array([TEMPLATE] * n_synth, dtype="U8"),
        }
    )
    return vstack([gaia, synth], join_type="exact")


def simulate_epochs(catalog, ceph, times_d):
    """One frame per epoch with the Cepheid magnitudes set; returns paths."""
    mags = cepheid_mag(ceph, times_d)
    ceph_rows = np.array(
        [
            np.flatnonzero(
                np.asarray(catalog["source_id"], dtype=np.int64) == sid
            )[0]
            for sid in ceph["source_id"]
        ]
    )
    paths = []
    for k in range(len(times_d)):
        cat_k = catalog.copy()
        cat_k["phot_g_mean_mag"][ceph_rows] = mags[:, k]
        path = os.path.join(OUT, f"epoch_{k:03d}.fits")
        t0 = walltime.time()
        field = simulate_field(
            RA_F, DEC_F,
            sensorfilter=SENSORFILTER,
            focus=FOCUS,
            exptime=EXPTIME,
            n_reads=N_READS,
            catalog=cat_k,
            shape=SHAPE,
            extended_sources=EXTENDED,
            chromatic=True,
            seed=3000 + k,
            output=path,
            write_clean=False,
        )
        print(
            f"epoch {k + 1:2d}/{len(times_d)}: t = {times_d[k]:7.2f} d, "
            f"{int(field.saturation_mask.sum())} saturated px, "
            f"{walltime.time() - t0:.1f} s"
        )
        paths.append(path)
    return paths


def measure_star(frames, sid, times_d):
    """(flux_e, flux_err_e, flags) per epoch for one star, PSF photometry."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = run_photometry(frames, target=int(sid), method="psf",
                             n_ref=3, times=times_d)
    rows = res.measurements[res.measurements["role"] == "target"]
    rows.sort("frame")
    return (
        np.asarray(rows["flux_e"], dtype=float),
        np.asarray(rows["flux_err_e"], dtype=float),
        np.asarray(rows["flags"], dtype=int),
    )


def calibrate(flux, flux_err, const_flux, const_gmag):
    """Per-epoch zero point from the constant stars -> mags + errors.

    flux, flux_err: (n_star, n_epoch); const_flux: (n_const, n_epoch).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        inst = const_gmag[:, None] + 2.5 * np.log10(const_flux)
    zp = np.nanmedian(inst, axis=0)
    zp_err = (
        1.4826
        * np.nanmedian(np.abs(inst - zp[None, :]), axis=0)
        / np.sqrt(const_flux.shape[0])
    )
    mag = zp[None, :] - 2.5 * np.log10(flux)
    mag_err = np.hypot(1.0857 * flux_err / flux, zp_err[None, :])
    return mag, mag_err, zp, zp_err


def recover_period(t_d, mag, err):
    """Lomb-Scargle best period [d] on a dense grid over 5-100 d."""
    freq = np.linspace(1.0 / 100.0, 1.0 / 5.0, 100_000)
    power = LombScargle(t_d, mag, err).power(freq)
    return 1.0 / freq[np.argmax(power)]


def flux_mean_mag(mag):
    """Mean magnitude in flux space (matches the injected definition)."""
    return -2.5 * np.log10(np.mean(10.0 ** (-0.4 * np.asarray(mag))))


def fit_pl(logp, m, sigma, pivot=1.3):
    """Weighted linear fit m = a*(logp - pivot) + b -> (a, b, sig_a, sig_b).

    Parameter errors are scaled by sqrt(max(1, chi2/dof)) so intrinsic
    scatter (e.g. the per-star extinction spread) inflates them honestly.
    """
    x = np.column_stack([logp - pivot, np.ones_like(logp)])
    w = 1.0 / np.asarray(sigma) ** 2
    cov = np.linalg.inv(x.T @ (w[:, None] * x))
    a, b = cov @ (x.T @ (w * m))
    resid = m - (a * (logp - pivot) + b)
    chi2_dof = float(np.sum(w * resid**2)) / max(len(m) - 2, 1)
    scale = np.sqrt(max(1.0, chi2_dof))
    return (
        float(a), float(b),
        scale * float(np.sqrt(cov[0, 0])),
        scale * float(np.sqrt(cov[1, 1])),
    )


def verify(ceph, rec, mean_a_g, pivot=1.3):
    """(text, passed) rows for the three checks + the fitted numbers."""
    good = rec["snr"] > 10.0
    dp = np.abs(rec["p_rec"] - ceph["period_d"]) / ceph["period_d"]
    frac = float(np.mean(dp[good] < 0.01)) if good.any() else 0.0

    m_corr = rec["mean_mag"] - mean_a_g  # known-mean extinction correction
    a, b, sig_a, sig_b = fit_pl(
        np.log10(rec["p_rec"]), m_corr, rec["mean_mag_err"], pivot=pivot
    )
    mu_rec = b - (PL_SLOPE * (pivot - 1.0) + PL_ZP)
    chi2_dof = float(np.mean(rec["chi2_dof"]))

    checks = [
        (
            f"period recovery: {frac * 100:.0f}% of SNR>10 Cepheids "
            f"within 1% ({int(good.sum())} usable)",
            frac >= 0.80,
        ),
        (
            f"distance modulus: mu = {mu_rec:.3f} +/- {sig_b:.3f} "
            f"(injected {MU}), slope {a:.2f} +/- {sig_a:.2f} "
            f"(injected {PL_SLOPE})",
            abs(mu_rec - MU) < 3.0 * sig_b,
        ),
        (
            f"light-curve chi2/dof vs injected model = {chi2_dof:.2f}",
            0.5 < chi2_dof < 2.0,
        ),
    ]
    return checks, (a, b, sig_a, sig_b, mu_rec)


def plot_summary(ceph, rec, times_d, mags_obs, errs_obs, mag_inj, frame0,
                 fit, mean_a_g, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("gks")
    a, b, _, _, mu_rec = fit
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # (1) P-L relation
    ax = axes[0]
    lp = np.linspace(*np.log10(P_RANGE), 50)
    ax.plot(lp, MU + PL_SLOPE * (lp - 1.0) + PL_ZP, color="#d1495b",
            lw=1.5, label="injected P-L")
    ax.plot(lp, a * (lp - 1.3) + b, color="0.4", ls="--", lw=1.2,
            label=f"fit: $\\mu$ = {mu_rec:.2f}")
    ax.errorbar(np.log10(rec["p_rec"]), rec["mean_mag"] - mean_a_g,
                yerr=rec["mean_mag_err"], fmt="o", color="#00798c", ms=4,
                lw=1, capsize=2, label="recovered (ext.-corr.)")
    ax.invert_yaxis()
    ax.set_xlabel("log P [d]")
    ax.set_ylabel("mean G [mag]")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"M101 Cepheids, WCC {SENSORFILTER}")

    # (2) example folded light curve: highest-SNR Cepheid
    j = int(np.argmax(rec["snr"]))
    ax = axes[1]
    p = float(rec["p_rec"][j])
    ph = (times_d / p) % 1.0
    ax.errorbar(ph, mags_obs[j], yerr=errs_obs[j], fmt="o",
                color="#00798c", ms=4, lw=1, capsize=2)
    fine = np.linspace(0.0, 1.0, 400)
    inj_phase = (fine - (times_d[0] / float(ceph["period_d"][j])
                         + float(ceph["phase0"][j]))) * 0.0 + fine
    model = (float(ceph["m0"][j])
             + float(ceph["amp"][j])
             * cepheid_template(fine + float(ceph["phase0"][j]) * 0.0))
    # plot the injected model on the *injected* ephemeris folded at p_rec
    tt = np.linspace(times_d.min(), times_d.max(), 2000)
    ax.plot((tt / p) % 1.0, cepheid_mag(ceph[[j]], tt)[0], ".",
            color="#d1495b", ms=1.5, alpha=0.6, label="injected")
    ax.invert_yaxis()
    ax.set_xlabel(f"phase (P = {p:.2f} d)")
    ax.set_ylabel("G [mag]")
    ax.set_title(f"Cepheid {j}: SNR = {rec['snr'][j]:.0f}")
    ax.legend(loc="upper right", fontsize=8)

    # (3) field with Cepheid positions
    ax = axes[2]
    img = frame0.image_e
    lo, hi = np.percentile(img, [5, 99.5])
    ax.imshow(img, origin="lower", cmap="gray_r",
              vmin=lo, vmax=hi)
    x, y = frame0.wcs.world_to_pixel_values(
        np.asarray(ceph["ra_field"]), np.asarray(ceph["dec_field"])
    )
    ax.scatter(x, y, s=60, facecolor="none", edgecolor="#d1495b", lw=0.8)
    ax.set_title(f"{EXPTIME:.0f} s epoch, galaxy + Cepheids")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    global N_CEPH, N_CONST, N_EPOCHS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="reduced smoke run (8 Cepheids, 8 epochs)")
    args = parser.parse_args(argv)
    if args.quick:
        N_CEPH, N_CONST, N_EPOCHS = 8, 8, 8

    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    geom = get_geometry(SENSORFILTER)
    wcs = build_wcs(RA_F, DEC_F, geom.plate_scale_mas, 0.0, SHAPE)
    times_d = np.sort(rng.uniform(0.0, BASELINE_D, N_EPOCHS))

    ceph, const = make_population(rng)
    catalog = make_catalog(wcs, ceph, const, rng, SHAPE)
    # record the field positions for the summary plot
    idx = {int(s): i for i, s in enumerate(catalog["source_id"])}
    ceph["ra_field"] = [catalog["ra"][idx[int(s)]] for s in ceph["source_id"]]
    ceph["dec_field"] = [catalog["dec"][idx[int(s)]] for s in ceph["source_id"]]
    ceph.write(os.path.join(OUT, "cepheids_injected.ecsv"),
               format="ascii.ecsv", overwrite=True)
    print(
        f"field: ({RA_F:.4f}, {DEC_F:.4f}), {len(catalog)} catalog rows "
        f"({len(catalog) - len(ceph) - len(const)} Gaia), "
        f"{N_CEPH} Cepheids P = {ceph['period_d'].min():.1f}-"
        f"{ceph['period_d'].max():.1f} d, "
        f"G = {ceph['m0'].min():.1f}-{ceph['m0'].max():.1f}"
    )

    paths = simulate_epochs(catalog, ceph, times_d)
    frames = [load_frame(p) for p in paths]

    # --- photometry: Cepheids + constant stars -----------------------------
    const_flux = np.zeros((len(const), N_EPOCHS))
    for i, sid in enumerate(const["source_id"]):
        f, _, _ = measure_star(frames, sid, times_d)
        const_flux[i] = f
        print(f"constant {i + 1:2d}/{len(const)} measured")

    flux = np.zeros((N_CEPH, N_EPOCHS))
    flux_err = np.zeros_like(flux)
    flags = np.zeros(flux.shape, dtype=int)
    for j, sid in enumerate(ceph["source_id"]):
        flux[j], flux_err[j], flags[j] = measure_star(frames, sid, times_d)
        print(f"cepheid {j + 1:2d}/{N_CEPH} measured")

    mags, errs, zp, zp_err = calibrate(
        flux, flux_err, const_flux, np.asarray(const["gmag"])
    )
    print(
        f"zero point: rms over epochs {np.std(zp):.4f} mag, "
        f"median per-epoch error {np.median(zp_err):.4f} mag"
    )

    # --- recovery -----------------------------------------------------------
    mag_inj = cepheid_mag(ceph, times_d)
    rec = Table(
        {
            "source_id": ceph["source_id"],
            "p_rec": np.zeros(N_CEPH),
            "mean_mag": np.zeros(N_CEPH),
            "mean_mag_err": np.zeros(N_CEPH),
            "snr": np.zeros(N_CEPH),
            "chi2_dof": np.zeros(N_CEPH),
        }
    )
    for j in range(N_CEPH):
        ok = (flags[j] == 0) & np.isfinite(mags[j])
        rec["p_rec"][j] = recover_period(times_d[ok], mags[j][ok],
                                         errs[j][ok])
        rec["mean_mag"][j] = flux_mean_mag(mags[j][ok])
        rec["mean_mag_err"][j] = float(np.median(errs[j][ok])
                                       / np.sqrt(ok.sum()))
        rec["snr"][j] = float(np.median(flux[j][ok] / flux_err[j][ok]))
        rec["chi2_dof"][j] = float(
            np.mean(((mags[j][ok] - mag_inj[j][ok]) / errs[j][ok]) ** 2)
        )
    rec.write(os.path.join(OUT, "cepheids_recovered.ecsv"),
              format="ascii.ecsv", overwrite=True)

    mean_a_g = float(np.mean(ceph["a_g"]))
    checks, fit = verify(ceph, rec, mean_a_g)
    plot_summary(ceph, rec, times_d, mags, errs, mag_inj, frames[0], fit,
                 mean_a_g, os.path.join(OUT, "cepheid_pl.png"))

    print()
    ok = True
    for text, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {text}")
        ok &= passed
    print(f"\noutputs in {OUT}/: frames, ECSV tables, cepheid_pl.png")
    if args.quick:
        print("(--quick: checks are informational, exiting 0)")
        sys.exit(0)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-run the reduced configuration**

Run: `python scripts/cepheids_m101.py --quick`
Expected: completes end-to-end (first run queries Gaia — network needed), prints per-epoch and per-star progress, prints three check lines, exits 0. Iterate here on any runtime errors (this is where crowding/saturation surprises surface — e.g. if the zero-point rms comes out large, check that constant stars are unsaturated and flags==0).

Also fix the plot function while iterating: delete the unused `inj_phase` / `model` locals in `plot_summary` (they are scaffolding above the `tt` plot); verify the folded light curve panel overlays sensibly.

- [ ] **Step 4: Full run**

Run: `python scripts/cepheids_m101.py`
Expected: ~15-25 min (effective-PSF cache warmup dominates the first epoch; per-epoch sim a few s; 45 photometry runs). All three checks PASS, exit 0. If check 2 fails marginally, first suspect the P-L fit errors (`fit_pl` scale) and the SNR>10 subset — do not loosen thresholds without understanding the residuals (plot `mean_mag - injected` vs `logp`).

- [ ] **Step 5: Commit**

```bash
git add scripts/cepheids_m101.py
git commit -m "feat: M101 Cepheid P-L end-to-end demo (extended sources + chromatic PSF + reddening)"
```

---

### Task 10: README section + full-suite verification

**Files:**
- Modify: `README.md` (add a short "Extended sources & chromatic PSFs" subsection near the existing feature list, plus the Cepheid demo one-liner under the examples/scripts section)

**Interfaces:** none (docs).

- [ ] **Step 1: Add README content.** Locate the feature/usage section of `README.md` and add, following the existing tone:

```markdown
### Extended sources & chromatic PSFs

`simulate_field` accepts `extended_sources=[SersicComponent(...)]` — analytic
Sérsic components (galaxy disks/bulges) rendered in e-/s/pix at native
resolution and FFT-convolved with the PSF — and `chromatic=True`, which
replaces the single central-wavelength PSF with spectrum-weighted effective
PSFs (per spectral type for stars, per component template for extended
light; in-focus only — the defocus PSF has no wavelength model).
Band-averaged F99 reddening is available via
`wcc_sim.chromatic.attenuation_factor`.

`scripts/cepheids_m101.py` is the end-to-end demo: synthetic Cepheids on the
M101 disk (Sérsic galaxy light + real Gaia foreground, per-star reddening),
recovered with `wcc-phot` PSF photometry into a period-luminosity relation
and distance modulus.
```

**Careful:** `README.md` already has unrelated uncommitted edits from the other PR-in-progress. Stage this change with `git add -p README.md` and select only the new hunk.

- [ ] **Step 2: Full test suite + demo artifacts sanity check**

Run: `python -m pytest tests/ -q && ls cepheid_out/cepheid_pl.png cepheid_out/cepheids_recovered.ecsv`
Expected: all tests pass; artifacts exist.

- [ ] **Step 3: Commit**

```bash
git add -p README.md
git commit -m "docs: extended sources, chromatic PSFs, Cepheid demo in README"
```

- [ ] **Step 4: Wrap up** — use superpowers:finishing-a-development-branch (push `feature/extended-sources`, open the PR titled "Extended sources, chromatic effective PSFs, and M101 Cepheid P-L demo"; PR body summarizes the spec and shows `cepheid_out/cepheid_pl.png`).

---

## Self-Review Notes (already applied)

- **Spec coverage:** chromatic module (Task 2-3), spt override (Task 4), extended components (5-7), pipeline + FITS (8), demo + 3 checks (9), README (10). Out-of-scope items from the spec (FITS-image input, spatially varying PSF, chromatic defocus) have no tasks — intentional.
- **Type consistency:** `effective_psf_for_spt(sim, sensorfilter, spt, ebv, focus, oversample, stamp_npix=, jitter_sigma_mas=, n_nodes=)` is called with this exact order in Tasks 2 (tests), 8 (pipeline). `render_extended(components, wcs, shape, plate_scale_mas, sensorfilter, kernels, wing=)` matches between Tasks 7 and 8. `kernels` keys are `(comp.template, float(comp.ebv))` in both.
- **Known iteration point:** Task 9 Step 3 explicitly flags the leftover scaffolding lines in `plot_summary` for cleanup during the smoke run, and Step 1 guards the `K0III` template assumption.
