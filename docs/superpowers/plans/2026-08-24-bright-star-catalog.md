# Bright-Star Catalog Supplement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill Gaia DR3's missing bright end from Hipparcos/XHIP and propagate proper motion, so a WCC field pointed at a naked-eye star actually contains it, at the position it occupies at the observation epoch.

**Architecture:** Two new leaf modules — `astrometry` (space-motion propagation, cross-match) and `brightcat` (XHIP query, normalization to Gaia-shaped rows, merge policy) — plus a vendored synthetic-colour table in `starflux` that turns a Johnson V magnitude into a Gaia G magnitude. `simulate_field` gains one step between the Gaia query and the existing rate path; nothing downstream of `rates_for_catalog` changes.

**Tech Stack:** Python 3.13, numpy, astropy (`SkyCoord.apply_space_motion`, `Table`), astroquery (`Vizier`, `Gaia`), synphot + wcc_etc (regeneration script only), pytest.

**Spec:** `docs/superpowers/specs/2026-08-24-bright-star-catalog-design.md`

## Global Constraints

- Branch: `feature/bright-star-catalog`. Commit per task.
- **No test may touch the network.** Patch `brightcat._run_query` and `catalog._run_query`, exactly as `tests/test_catalog.py` already does for Gaia.
- XHIP positions are at **epoch J1991.25** despite the `RAJ2000`/`DEJ2000` column names (verified byte-identical to `I/239/hip_main`'s `RAICRS`/`DEICRS`). Gaia DR3 positions are at **J2016.0**.
- `pmra`/`pmRA` in both catalogs is dRA/dt·cos(dec) in mas/yr — feed it to `pm_ra_cosdec`, never to `pm_ra`.
- Everything is skipped when `catalog=` is passed to `simulate_field`, exactly as the Gaia query already is, so all 207 existing tests stay green and unchanged.
- FITS keywords are ≤ 8 characters: `BRIGHTCT`, `SKYEPOCH`, `NBRIGHT`, `NBRIGHRP`.
- Run `python -m ruff check` on every file touched before committing; the repo is currently clean.

## Deviations from the spec (discovered while de-risking; spec amended in Task 0)

1. **Vendor the derived colour table, not the filter curves.** Integrating 28 Pickles templates through Johnson B, V and Gaia G costs **7.6 s** and needs the remote Johnson curves and the remote Vega spectrum. So the synthetic colours are computed once by a committed script and vendored as a 28-row CSV; runtime reads the CSV and needs no synphot at all. Validation is unchanged and stronger: the G2V template's synthetic B−V comes out **0.650**, the literature solar value.
2. **The replace threshold is in G, not V.** The rationale is Gaia's saturation regime, which is defined in G; using G also means the merged table needs no `vmag` column. G ≈ V − 0.16 for solar colours, so the practical difference is nil.
3. **VizieR's own J2000 positions are proper-motion-only.** Our full propagation (parallax + RV) therefore lands 4.6 mas from VizieR at J2000 — that is perspective acceleration, not an error. The external check is against a **pm-only** propagation, which matches VizieR to 0.000 mas; a second test then asserts the RV path moves the star by the expected 4.6 mas.
4. **Suppress one ERFA warning.** With a stand-in distance for rows with no parallax, ERFA emits `"distance overridden (Note 6)"` on every call. The result is identical to a pm-only propagation to 0.000 µas (verified), so `propagate` filters that one message narrowly.

## File Structure

| File | Responsibility |
|---|---|
| `src/wcc_sim/astrometry.py` (new) | `as_time`, `propagate`, `crossmatch`. Knows nothing about catalogs. |
| `src/wcc_sim/brightcat.py` (new) | XHIP query + cache, normalization to Gaia-shaped rows, merge policy. |
| `src/wcc_sim/data/spt_synthetic_colors.csv` (new) | 28 rows: `spt,b_v,g_v`. |
| `scripts/build_spt_colors.py` (new) | One-off regenerator for that CSV (needs network). |
| `src/wcc_sim/starflux.py` | `+ _synthetic_color_table`, `spt_from_b_v`, `g_minus_v`, `bp_rp_for_spt`. |
| `src/wcc_sim/catalog.py` | `COLUMNS` gains astrometry; `GAIA_EPOCH`; cache schema check. |
| `src/wcc_sim/pipeline.py` | `bright`, `epoch`, `bright_replace_mag`, `match_radius_arcsec`; merge step; `params`. |
| `src/wcc_sim/fitswriter.py` | Four new header cards. |
| `src/wcc_sim/psfreport.py` | `_REPRO_KEYS` gains the new knobs. |
| `scripts/example_alpha_cen.py` | Injection deleted; `--epoch`; provenance in the printout. |
| `tests/test_starflux.py` | Colour-table tests (Task 1). |
| `tests/test_astrometry.py` (new) | Propagation and cross-match (Tasks 2-3). |
| `tests/test_brightcat.py` (new) | Query, normalization, merge, end-to-end (Tasks 4-6, 8). |
| `tests/test_catalog.py` | Cache-schema test (Task 7). |

---

### Task 0: Amend the spec to match what de-risking found

**Files:**
- Modify: `docs/superpowers/specs/2026-08-24-bright-star-catalog-design.md`

- [ ] **Step 1: Apply the four deviations above to the spec**

In the "Photometry" section, replace the sentence beginning "Vendor the two SVO passband curves" with:

```markdown
Integrate every Pickles template through Johnson B, Johnson V and Gaia G
**once**, offline, and vendor the resulting colours as
`data/spt_synthetic_colors.csv` (`spt,b_v,g_v`), regenerated by
`scripts/build_spt_colors.py`. The integrations cost 7.6 s and need the
remote Johnson curves and Vega spectrum; the runtime needs neither. Use the
table twice: pick the template whose synthetic B-V is nearest the star's,
then set `G = V + (G-V)_template`.

Validation: the G2V template's synthetic B-V is 0.650, the literature solar
value. Its synthetic G-V is -0.164, against -0.14 from the published
(BP-RP) colour term — a 0.02 mag gap, which is that relation's own scatter,
and 2% in flux.
```

In the "Epoch" section, replace "reproduces `_RA.icrs` / `_DE.icrs` to **0.0 mas**" with:

```markdown
reproduces `_RA.icrs` / `_DE.icrs` to **0.000 mas** when run
proper-motion-only, which is what VizieR itself computes. Adding parallax
and radial velocity moves the result 4.6 mas at J2000 and 75 mas at
J2026.6 — perspective acceleration, absent from VizieR's own value.
```

In the "Merge" section, change `V < bright_replace_mag` to `G < bright_replace_mag` and add:

```markdown
The threshold is in G, not V: the criterion is Gaia's saturation regime,
which is defined in G, and it keeps a `vmag` column out of the merged table.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-08-24-bright-star-catalog-design.md
git commit -m "docs: correct spec on colour-table vendoring, G threshold, VizieR pm-only"
```

---

### Task 1: Synthetic colour table and the B-V lookup

**Files:**
- Create: `src/wcc_sim/data/spt_synthetic_colors.csv`
- Create: `scripts/build_spt_colors.py`
- Modify: `src/wcc_sim/starflux.py`
- Test: `tests/test_starflux.py`

**Interfaces:**
- Consumes: existing `starflux._spt_table()` → `(spts, bp_rp)`, `starflux.DATA_DIR`.
- Produces: `spt_from_b_v(b_v) -> np.ndarray[str]`, `g_minus_v(spt) -> np.ndarray[float]`, `bp_rp_for_spt(spt) -> np.ndarray[float]`, `_synthetic_color_table() -> (spts, b_v, g_v)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_starflux.py`:

```python
# --------------------------------------------------------------------------- #
# Synthetic Johnson colours (for V-magnitude catalogs like Hipparcos)          #
# --------------------------------------------------------------------------- #

def test_synthetic_b_v_of_the_solar_template_is_the_solar_value():
    """The whole V -> G path rests on this: Pickles G2V through Johnson B and
    V must give the Sun's B-V = 0.65, or the templates are being integrated
    through the wrong passbands."""
    from wcc_sim.starflux import _synthetic_color_table

    spts, b_v, _ = _synthetic_color_table()
    assert b_v[list(spts).index("G2V")] == pytest.approx(0.65, abs=0.01)


def test_synthetic_g_minus_v_is_near_the_published_colour_term():
    """-0.164 synthetic vs -0.14 from the published (BP-RP) relation; the gap
    is that relation's own scatter, so the tolerance is deliberately loose."""
    from wcc_sim.starflux import g_minus_v

    assert g_minus_v("G2V")[0] == pytest.approx(-0.15, abs=0.03)


def test_every_template_has_a_synthetic_colour():
    from wcc_sim.starflux import _spt_table, _synthetic_color_table

    assert set(_spt_table()[0]) == set(_synthetic_color_table()[0])


def test_spt_from_b_v_picks_the_nearest_template():
    from wcc_sim.starflux import spt_from_b_v

    assert spt_from_b_v(0.65)[0] == "G2V"
    assert spt_from_b_v([0.0, 1.45])[0] == "A0V"


def test_spt_from_b_v_falls_back_to_g2v_without_a_colour():
    """Same convention as spt_from_bp_rp, so a missing colour behaves the
    same whichever catalog the row came from."""
    from wcc_sim.starflux import spt_from_b_v

    assert spt_from_b_v(np.nan)[0] == "G2V"


def test_bp_rp_for_spt_round_trips_through_the_type_lookup():
    """to_gaia_like sets BP-RP from this, so spt_from_bp_rp must return the
    type it came from -- otherwise a merged row's rate uses another template."""
    from wcc_sim.starflux import bp_rp_for_spt, spt_from_bp_rp

    for spt in ("A0V", "G2V", "K5V", "M4V"):
        assert spt_from_bp_rp(bp_rp_for_spt(spt))[0] == spt
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_starflux.py -q -k "synthetic or b_v or bp_rp_for"`
Expected: FAIL, `ImportError: cannot import name '_synthetic_color_table'`.

- [ ] **Step 3: Write the generator script**

Create `scripts/build_spt_colors.py`:

```python
#!/usr/bin/env python
"""Regenerate data/spt_synthetic_colors.csv -- synthetic Johnson/Gaia colours.

Integrates every Pickles template in bp_rp_to_spt.csv through Johnson B,
Johnson V and the vendored Gaia DR3 G passband, in vegamag. Run this when the
template list or the G passband changes; it needs network for the Johnson
curves (synphot's remote filter set) and the Vega spectrum, which is why the
result is vendored instead of computed at run time (~7.6 s per process).

    python scripts/build_spt_colors.py
"""

import os

from synphot import Observation, SpectralElement
from wcc_etc.scene import get_scene

from wcc_sim.starflux import DATA_DIR, _spt_table, gaia_g_bandpass


def main():
    B = SpectralElement.from_filter("johnson_b")
    V = SpectralElement.from_filter("johnson_v")
    G = gaia_g_bandpass()

    def vegamag(spec, band):
        # force="taper": the Pickles templates are narrower than Johnson B at
        # the blue end; tapering to zero is the right boundary condition for a
        # colour, and synphot refuses to extrapolate silently.
        return Observation(spec, band, force="taper").effstim("vegamag").value

    lines = ["spt,b_v,g_v"]
    for spt in _spt_table()[0]:
        spec = get_scene(str(spt), mag=0.0).source.get_spectrum()
        b, v, g = vegamag(spec, B), vegamag(spec, V), vegamag(spec, G)
        lines.append(f"{spt},{b - v:.4f},{g - v:.4f}")

    path = os.path.join(DATA_DIR, "spt_synthetic_colors.csv")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path} ({len(lines) - 1} templates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run it and commit the generated CSV**

Run: `python scripts/build_spt_colors.py`
Expected: `wrote .../spt_synthetic_colors.csv (27 templates)`, and the G2V row reads `G2V,0.6500,-0.1638` (±0.001). If the file does not match this, stop — the passbands or the template library changed and the plan's validation numbers no longer hold.

- [ ] **Step 5: Add the lookups to `starflux.py`**

After `spt_from_bp_rp`, add:

```python
@lru_cache(maxsize=1)
def _synthetic_color_table():
    """(spts, Johnson B-V, Gaia G - Johnson V) per Pickles template.

    Vendored rather than computed: the synphot integrations cost ~8 s and
    need the remote Johnson curves. Regenerate with
    scripts/build_spt_colors.py.
    """
    path = os.path.join(DATA_DIR, "spt_synthetic_colors.csv")
    spts = np.loadtxt(path, delimiter=",", skiprows=1, usecols=0, dtype=str)
    b_v, g_v = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2),
                          unpack=True)
    return spts, b_v, g_v


def spt_from_b_v(b_v):
    """Nearest-neighbor Pickles dwarf type for Johnson B-V; NaN -> 'G2V'.

    The B-V counterpart of spt_from_bp_rp, for catalogs (Hipparcos) that
    give Johnson photometry instead of Gaia's.
    """
    spts, colors, _ = _synthetic_color_table()
    b_v = np.atleast_1d(
        np.ma.filled(np.ma.masked_invalid(np.asarray(b_v, dtype=float)), np.nan)
    )
    out = np.full(b_v.shape, "G2V", dtype=object)
    ok = np.isfinite(b_v)
    if ok.any():
        out[ok] = spts[np.abs(b_v[ok, None] - colors[None, :]).argmin(axis=1)]
    return out.astype(str)


def _by_spt(values, spt):
    spts, _, _ = _synthetic_color_table()
    index = {str(s): i for i, s in enumerate(spts)}
    return np.array([values[index[str(s)]] for s in np.atleast_1d(spt)],
                    dtype=float)


def g_minus_v(spt):
    """Synthetic Gaia G - Johnson V for a Pickles type."""
    return _by_spt(_synthetic_color_table()[2], spt)


def bp_rp_for_spt(spt):
    """The BP-RP that spt_from_bp_rp maps back to this type."""
    spts, colors = _spt_table()
    index = {str(s): i for i, s in enumerate(spts)}
    return np.array([colors[index[str(s)]] for s in np.atleast_1d(spt)],
                    dtype=float)
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_starflux.py -q`
Expected: all pass, including the pre-existing ones.

- [ ] **Step 7: Commit**

```bash
git add src/wcc_sim/starflux.py src/wcc_sim/data/spt_synthetic_colors.csv \
        scripts/build_spt_colors.py tests/test_starflux.py
git commit -m "feat(starflux): synthetic Johnson colours for V-magnitude catalogs"
```

---

### Task 2: `astrometry.propagate`

**Files:**
- Create: `src/wcc_sim/astrometry.py`
- Test: `tests/test_astrometry.py`

**Interfaces:**
- Produces: `as_time(epoch) -> Time`, `propagate(cat, from_epoch, to_epoch, ra="ra", dec="dec", pmra="pmra", pmdec="pmdec", parallax=None, rv=None) -> Table`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_astrometry.py`:

```python
"""Space-motion propagation, checked against an external reference.

The reference rows are HIP 71683 and 71681 (alpha Cen A and B) from VizieR
I/239/hip_main: positions at epoch J1991.25, and VizieR's own computed J2000
positions in the _RA.icrs / _DE.icrs columns. VizieR's computation is proper
motion only, which is what makes it a clean check on the pm path.
"""

import numpy as np
import pytest
from astropy.table import Table

HIP_EPOCH = 1991.25

#: ra, dec at J1991.25; pmra (dRA/dt cos dec), pmdec [mas/yr]; plx [mas]; rv [km/s]
HIP = Table({
    "hip": [71683, 71681],
    "ra": [219.92041034, 219.91412833],
    "dec": [-60.83514707, -60.83947139],
    "pmra": [-3678.19, -3600.35],
    "pmdec": [481.84, 952.11],
    "parallax": [742.12, 742.12],
    "radial_velocity": [-21.4, -18.6],
})

#: VizieR's own _RA.icrs / _DE.icrs for the same two rows (J2000, pm only)
VIZIER_J2000 = np.array([[219.90206584, -60.83397468],
                         [219.89617026, -60.83715604]])


def _sep_mas(table, reference):
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    got = SkyCoord(table["ra"], table["dec"], unit="deg")
    ref = SkyCoord(reference[:, 0], reference[:, 1], unit="deg")
    return got.separation(ref).to(u.mas).value


def test_proper_motion_only_reproduces_vizier_j2000():
    """The external check: pm-only propagation must land on VizieR's own
    numbers, which are also pm-only. Tolerance is 0.5 mas; the observed
    agreement is 0.000 mas."""
    from wcc_sim.astrometry import propagate

    out = propagate(HIP, HIP_EPOCH, 2000.0)
    assert _sep_mas(out, VIZIER_J2000) == pytest.approx([0.0, 0.0], abs=0.5)


def test_parallax_and_rv_add_perspective_acceleration():
    """Adding distance and radial velocity must move alpha Cen A by ~4.6 mas
    at J2000 and ~75 mas at J2026.6 -- 4.5 px at 16.869 mas/px, which is why
    the RV column is queried at all."""
    from wcc_sim.astrometry import propagate

    for epoch, expected in ((2000.0, 4.6), (2026.6, 75.3)):
        pm_only = propagate(HIP, HIP_EPOCH, epoch)
        full = propagate(HIP, HIP_EPOCH, epoch, parallax="parallax",
                         rv="radial_velocity")
        assert _sep_mas(full, np.column_stack(
            [pm_only["ra"], pm_only["dec"]]))[0] == pytest.approx(expected, rel=0.1)


def test_alpha_cen_crosses_the_field_between_catalog_epochs():
    """Why the merge propagates before matching: 24.75 years of Gaia-minus-
    Hipparcos epoch difference is 92 arcsec of motion, and the WCC field is
    162 arcsec wide."""
    from wcc_sim.astrometry import propagate

    moved = propagate(HIP, HIP_EPOCH, 2016.0)
    assert _sep_mas(moved, np.column_stack([HIP["ra"], HIP["dec"]]))[0] / 1000.0 \
        == pytest.approx(92.0, rel=0.05)


def test_missing_parallax_falls_back_to_proper_motion_only():
    """A stand-in distance is used for rows with no parallax; it must not
    perturb the answer."""
    from wcc_sim.astrometry import propagate

    no_plx = HIP.copy()
    no_plx["parallax"] = 0.0
    a = propagate(no_plx, HIP_EPOCH, 2026.6, parallax="parallax",
                  rv="radial_velocity")
    b = propagate(HIP, HIP_EPOCH, 2026.6)
    assert _sep_mas(a, np.column_stack([b["ra"], b["dec"]])) == \
        pytest.approx([0.0, 0.0], abs=1e-3)


def test_missing_parallax_emits_no_erfa_warning():
    """The stand-in distance makes ERFA report 'distance overridden'; that is
    expected and must not reach the user."""
    import warnings
    from wcc_sim.astrometry import propagate

    no_plx = HIP.copy()
    no_plx["parallax"] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        propagate(no_plx, HIP_EPOCH, 2026.6, parallax="parallax")


def test_equal_epochs_and_empty_tables_are_no_ops():
    from wcc_sim.astrometry import propagate

    same = propagate(HIP, 2000.0, 2000.0)
    assert np.array_equal(np.asarray(same["ra"]), np.asarray(HIP["ra"]))
    assert len(propagate(HIP[:0], 1991.25, 2000.0)) == 0


def test_rows_without_proper_motion_stay_put():
    from wcc_sim.astrometry import propagate

    still = HIP.copy()
    still["pmra"] = 0.0
    still["pmdec"] = 0.0
    out = propagate(still, HIP_EPOCH, 2026.6)
    assert np.allclose(np.asarray(out["ra"]), np.asarray(HIP["ra"]))
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_astrometry.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'wcc_sim.astrometry'`.

- [ ] **Step 3: Write `astrometry.py`**

```python
"""Space-motion propagation and cross-matching, for merging sky catalogs.

Catalogs disagree on epoch -- Gaia DR3 is at J2016.0, Hipparcos at J1991.25
-- and on column names, so both are parameters here. Nothing in this module
knows which catalog it is looking at.
"""

import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

#: Stand-in distance for rows with no usable parallax. ERFA ignores a
#: distance this large and propagates proper motion alone, which is exactly
#: the wanted behaviour for an unmeasured parallax.
_FAR_PC = 1e6


def as_time(epoch):
    """A Julian year (2016.0) or anything Time accepts -> Time."""
    if isinstance(epoch, Time):
        return epoch
    return Time(f"J{float(epoch)}")


def _column(cat, name, default):
    """Column `name` as float with masked and non-finite entries filled."""
    if name is None or name not in cat.colnames:
        return np.full(len(cat), default, dtype=float)
    values = np.ma.masked_invalid(np.asarray(cat[name], dtype=float))
    return np.ma.filled(values, default)


def propagate(cat, from_epoch, to_epoch, ra="ra", dec="dec", pmra="pmra",
              pmdec="pmdec", parallax=None, rv=None):
    """Copy of `cat` with positions moved from one epoch to another.

    `pmra` is dRA/dt * cos(dec) in mas/yr, as Gaia and Hipparcos both
    tabulate it. Rows without proper motion stay where they are. When
    `parallax` and `rv` name columns, they add perspective acceleration --
    75 mas, 4.5 px, for alpha Cen over 35 years -- and are ignored where the
    parallax is not positive.
    """
    out = cat.copy()
    if not len(out):
        return out
    t0, t1 = as_time(from_epoch), as_time(to_epoch)
    if abs((t1 - t0).to_value(u.yr)) < 1e-9:
        return out

    plx = _column(out, parallax, 0.0)
    known = plx > 0.0
    distance = np.where(known, 1000.0 / np.where(known, plx, 1.0), _FAR_PC)
    coords = SkyCoord(
        ra=_column(out, ra, np.nan) * u.deg,
        dec=_column(out, dec, np.nan) * u.deg,
        pm_ra_cosdec=_column(out, pmra, 0.0) * u.mas / u.yr,
        pm_dec=_column(out, pmdec, 0.0) * u.mas / u.yr,
        distance=distance * u.pc,
        radial_velocity=np.where(known, _column(out, rv, 0.0), 0.0) * u.km / u.s,
        obstime=t0,
    )
    with warnings.catch_warnings():
        # _FAR_PC rows: ERFA says it ignored the distance, which is the point
        warnings.filterwarnings("ignore", message=".*distance overridden.*")
        moved = coords.apply_space_motion(new_obstime=t1)
    out[ra] = moved.ra.deg
    out[dec] = moved.dec.deg
    return out
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_astrometry.py -q`
Expected: all pass. If `test_missing_parallax_emits_no_erfa_warning` fails, print the warning's message text and widen the `filterwarnings` message pattern to match it — do not broaden the category.

- [ ] **Step 5: Commit**

```bash
git add src/wcc_sim/astrometry.py tests/test_astrometry.py
git commit -m "feat(astrometry): epoch propagation with parallax and radial velocity"
```

---

### Task 3: `astrometry.crossmatch`

**Files:**
- Modify: `src/wcc_sim/astrometry.py`
- Test: `tests/test_astrometry.py`

**Interfaces:**
- Produces: `crossmatch(a, b, radius_arcsec, ra="ra", dec="dec") -> (np.ndarray[int], np.ndarray[int])` — index pairs into `a` and `b`, each index appearing at most once.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_astrometry.py`:

```python
# --------------------------------------------------------------------------- #
# Cross-match                                                                  #
# --------------------------------------------------------------------------- #

def _cat(*pairs):
    return Table({"ra": [p[0] for p in pairs], "dec": [p[1] for p in pairs]})


def test_crossmatch_pairs_within_the_radius_only():
    from wcc_sim.astrometry import crossmatch

    a = _cat((10.0, 0.0), (10.01, 0.0))            # second is 36" away
    b = _cat((10.0001, 0.0))                       # 0.36" from the first
    idx_a, idx_b = crossmatch(a, b, 2.0)
    assert list(idx_a) == [0] and list(idx_b) == [0]


def test_crossmatch_is_one_to_one_and_keeps_the_closer_pair():
    """Two bright rows cannot both claim one Gaia row; the closer wins and
    the other is left unmatched, so it gets added rather than dropped."""
    from wcc_sim.astrometry import crossmatch

    a = _cat((10.0002, 0.0), (10.0001, 0.0))       # 0.72" and 0.36" away
    b = _cat((10.0, 0.0))
    idx_a, idx_b = crossmatch(a, b, 5.0)
    assert list(idx_a) == [1] and list(idx_b) == [0]


def test_crossmatch_handles_empty_inputs():
    from wcc_sim.astrometry import crossmatch

    for a, b in ((_cat(), _cat((10.0, 0.0))), (_cat((10.0, 0.0)), _cat()),
                 (_cat(), _cat())):
        idx_a, idx_b = crossmatch(a, b, 2.0)
        assert len(idx_a) == 0 and len(idx_b) == 0
```

Note: `_cat()` with no arguments must produce an empty two-column table — `Table({"ra": [], "dec": []})` does.

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_astrometry.py -q -k crossmatch`
Expected: FAIL, `ImportError: cannot import name 'crossmatch'`.

- [ ] **Step 3: Implement it**

Append to `src/wcc_sim/astrometry.py`:

```python
def crossmatch(a, b, radius_arcsec, ra="ra", dec="dec"):
    """One-to-one nearest matches between two catalogs.

    Returns `(idx_a, idx_b)` such that row `idx_a[k]` of `a` and row
    `idx_b[k]` of `b` are the same star. Where several rows of `a` fall on
    one row of `b`, the closest keeps it and the rest come back unmatched --
    a merge then adds them instead of silently dropping them.
    """
    empty = (np.array([], dtype=int), np.array([], dtype=int))
    if not len(a) or not len(b):
        return empty
    coords_a = SkyCoord(np.asarray(a[ra], dtype=float),
                        np.asarray(a[dec], dtype=float), unit="deg")
    coords_b = SkyCoord(np.asarray(b[ra], dtype=float),
                        np.asarray(b[dec], dtype=float), unit="deg")
    nearest, sep, _ = coords_a.match_to_catalog_sky(coords_b)
    close = np.flatnonzero(sep.arcsec <= float(radius_arcsec))
    if not close.size:
        return empty
    # one-to-one: among rows of `a` claiming the same row of `b`, keep the
    # closest. argsort by separation, then take the first hit per b-index.
    order = close[np.argsort(sep.arcsec[close], kind="stable")]
    seen, idx_a, idx_b = set(), [], []
    for i in order:
        j = int(nearest[i])
        if j in seen:
            continue
        seen.add(j)
        idx_a.append(int(i))
        idx_b.append(j)
    keep = np.argsort(idx_a, kind="stable")
    return np.asarray(idx_a, dtype=int)[keep], np.asarray(idx_b, dtype=int)[keep]
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_astrometry.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/wcc_sim/astrometry.py tests/test_astrometry.py
git commit -m "feat(astrometry): one-to-one positional cross-match"
```

---

### Task 4: XHIP query with cache and graceful failure

**Files:**
- Create: `src/wcc_sim/brightcat.py`
- Test: `tests/test_brightcat.py`

**Interfaces:**
- Produces: `XHIP_CATALOG`, `XHIP_COLUMNS`, `XHIP_EPOCH = 1991.25`, `_run_query(ra_deg, dec_deg, radius_arcsec) -> Table`, `_empty_table() -> Table`, `query_bright(ra_deg, dec_deg, radius_arcsec, cache_dir=None) -> Table`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_brightcat.py`:

```python
"""The bright-star supplement: query, normalization, and the merge policy.

Nothing here touches the network -- brightcat._run_query is patched, the way
tests/test_catalog.py already patches the Gaia one. The fixture rows are the
real XHIP values for alpha Cen A and B.
"""

import numpy as np
import pytest
from astropy.table import Table

#: XHIP V/137D rows for HIP 71683 / 71681. Positions are epoch J1991.25
#: despite the RAJ2000 column name.
XHIP_ROWS = Table({
    "HIP": np.array([71683, 71681], dtype=np.int32),
    "RAJ2000": [219.92041034, 219.91412833],
    "DEJ2000": [-60.83514707, -60.83947139],
    "pmRA": [-3678.19, -3600.35],
    "pmDE": [481.84, 952.11],
    "Plx": [742.12, 742.12],
    "RV": [-21.4, -18.6],
    "Vmag": [-0.01, 1.35],
    "B-V": [0.710, 0.900],
    "SpType": ["G2V", "K1V"],
    "Comp": ["A", "B"],
})


@pytest.fixture
def patched_query(monkeypatch):
    """brightcat._run_query -> the alpha Cen rows, with a call counter."""
    import wcc_sim.brightcat as bc

    calls = []

    def fake(ra, dec, radius):
        calls.append((ra, dec, radius))
        return XHIP_ROWS.copy()

    monkeypatch.setattr(bc, "_run_query", fake)
    return calls


def test_query_bright_returns_the_rows(patched_query):
    from wcc_sim.brightcat import query_bright

    out = query_bright(219.9, -60.83, 110.0)
    assert len(out) == 2
    assert set(("HIP", "Vmag", "B-V", "RV")).issubset(out.colnames)
    assert len(patched_query) == 1


def test_query_bright_caches_to_disk(patched_query, tmp_path):
    """A second call with the same cone must not hit the service again, so a
    repeat run of a script works offline."""
    from wcc_sim.brightcat import query_bright

    first = query_bright(219.9, -60.83, 110.0, cache_dir=str(tmp_path))
    second = query_bright(219.9, -60.83, 110.0, cache_dir=str(tmp_path))
    assert len(patched_query) == 1
    assert list(first["HIP"]) == list(second["HIP"])
    assert len(list(tmp_path.glob("xhip_*.ecsv"))) == 1


def test_query_bright_warns_and_degrades_when_the_service_fails(monkeypatch):
    """A VizieR outage must not take the whole simulation down: warn, return
    nothing, and let the caller continue with Gaia alone."""
    import wcc_sim.brightcat as bc

    def boom(ra, dec, radius):
        raise ConnectionError("VizieR closed the connection")

    monkeypatch.setattr(bc, "_run_query", boom)
    with pytest.warns(UserWarning, match="bright-star query failed"):
        out = bc.query_bright(219.9, -60.83, 110.0)
    assert len(out) == 0
    assert set(bc.XHIP_COLUMNS).issubset(out.colnames)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_brightcat.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'wcc_sim.brightcat'`.

- [ ] **Step 3: Write the query half of `brightcat.py`**

```python
"""Bright-star supplement: the stars Gaia DR3 does not have.

Gaia's brightest source is G = 1.73 and it holds only 150 sources brighter
than G = 3, so the naked-eye stars whose scattered-light halo drives WCC
stray-light requirements are missing from it entirely -- alpha Cen A and B,
for instance, have no DR3 entry at all. This module fills that end from
XHIP (VizieR V/137D, Hipparcos plus radial velocities), normalizes the rows
into the Gaia-shaped table the rest of the pipeline expects, and merges them
by positional cross-match.
"""

import os
import warnings

import numpy as np
from astropy import units as u
from astropy.table import Table, vstack

from .astrometry import crossmatch, propagate
from .catalog import GAIA_EPOCH
from .starflux import bp_rp_for_spt, g_minus_v, spt_from_b_v

#: VizieR table: the Extended Hipparcos Compilation (Anderson & Francis 2012).
XHIP_CATALOG = "V/137D/XHIP"

XHIP_COLUMNS = [
    "HIP", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "Plx", "RV", "Vmag", "B-V",
    "SpType", "Comp",
]

#: Epoch of the XHIP positions. The columns are named RAJ2000/DEJ2000, but
#: J2000 there is the equinox, not the epoch: the values are byte-identical
#: to I/239/hip_main's RAICRS/DEICRS, which are documented as J1991.25.
#: Reading them as epoch J2000 puts alpha Cen 32 arcsec off.
XHIP_EPOCH = 1991.25

_DTYPES = {"HIP": np.int64, "SpType": "U32", "Comp": "U8"}


def _empty_table():
    return Table({c: np.array([], dtype=_DTYPES.get(c, float))
                  for c in XHIP_COLUMNS})


def _run_query(ra_deg, dec_deg, radius_arcsec):
    """Synchronous VizieR cone search (network). Thin so tests can patch it."""
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    vizier = Vizier(columns=XHIP_COLUMNS, row_limit=-1)
    found = vizier.query_region(
        SkyCoord(ra_deg, dec_deg, unit="deg"),
        radius=radius_arcsec * u.arcsec,
        catalog=XHIP_CATALOG,
    )
    return found[0] if len(found) else _empty_table()


def query_bright(ra_deg, dec_deg, radius_arcsec, cache_dir=None):
    """XHIP rows within radius_arcsec of (ra, dec), cached like query_gaia.

    A failure of the service is not fatal: it warns and returns an empty
    table so the caller can carry on with Gaia alone. Callers record that
    fact (see `merge`) rather than implying the bright end was filled.
    """
    cache_file = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        key = f"xhip_{ra_deg:.6f}_{dec_deg:+.6f}_{radius_arcsec:.1f}"
        cache_file = os.path.join(cache_dir, key + ".ecsv")
        if os.path.exists(cache_file):
            return Table.read(cache_file, format="ascii.ecsv")
    try:
        out = _run_query(ra_deg, dec_deg, radius_arcsec)
    except Exception as exc:  # network, service, or VOTable parse failure
        warnings.warn(
            f"bright-star query failed ({type(exc).__name__}: {exc}); "
            "continuing with Gaia only",
            UserWarning,
        )
        return _empty_table()
    if cache_file is not None:
        out.write(cache_file, format="ascii.ecsv", overwrite=True)
    return out
```

- [ ] **Step 4: Add `GAIA_EPOCH` to `catalog.py` so the import resolves**

In `src/wcc_sim/catalog.py`, after `COLUMNS`:

```python
#: Reference epoch of Gaia DR3 positions (Julian year).
GAIA_EPOCH = 2016.0
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_brightcat.py -q`
Expected: all four pass.

- [ ] **Step 6: Commit**

```bash
git add src/wcc_sim/brightcat.py src/wcc_sim/catalog.py tests/test_brightcat.py
git commit -m "feat(brightcat): XHIP cone search with cache and graceful failure"
```

---

### Task 5: Normalize XHIP rows into Gaia-shaped rows

**Files:**
- Modify: `src/wcc_sim/brightcat.py`
- Test: `tests/test_brightcat.py`

**Interfaces:**
- Consumes: `starflux.spt_from_b_v`, `starflux.g_minus_v`, `starflux.bp_rp_for_spt` (Task 1).
- Produces: `to_gaia_like(bright) -> Table` with columns `source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, pmra, pmdec, parallax, radial_velocity, spt, catalog`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_brightcat.py`:

```python
# --------------------------------------------------------------------------- #
# Normalization to Gaia-shaped rows                                            #
# --------------------------------------------------------------------------- #

GAIA_LIKE_COLUMNS = (
    "source_id", "ra", "dec", "phot_g_mean_mag", "phot_bp_mean_mag",
    "phot_rp_mean_mag", "pmra", "pmdec", "parallax", "radial_velocity",
    "spt", "catalog",
)


def test_to_gaia_like_produces_the_pipeline_column_set():
    from wcc_sim.brightcat import to_gaia_like

    out = to_gaia_like(XHIP_ROWS)
    assert set(GAIA_LIKE_COLUMNS).issubset(out.colnames)
    assert list(out["catalog"]) == ["hipparcos", "hipparcos"]


def test_to_gaia_like_marks_provenance_with_a_negative_source_id():
    """A negative id cannot collide with a Gaia source_id, so a merged row's
    origin is readable straight off the catalog."""
    from wcc_sim.brightcat import to_gaia_like

    assert list(to_gaia_like(XHIP_ROWS)["source_id"]) == [-71683, -71681]


def test_to_gaia_like_converts_v_to_g_through_the_template():
    """alpha Cen A: V = -0.01, B-V = 0.71 -> the template nearest that colour,
    whose synthetic G-V is about -0.166, so G is about -0.18. The published
    (BP-RP) colour term gives -0.15 for the same star; the 0.03 mag spread is
    the accuracy of this path, and 3% in rate."""
    from wcc_sim.brightcat import to_gaia_like

    out = to_gaia_like(XHIP_ROWS)
    assert out["phot_g_mean_mag"][0] == pytest.approx(-0.18, abs=0.04)
    assert out["phot_g_mean_mag"][1] == pytest.approx(1.35 - 0.2, abs=0.1)


def test_to_gaia_like_colours_round_trip_to_the_same_template():
    """The BP and RP magnitudes exist only so the untouched rate path picks
    the same template this row was built from."""
    from wcc_sim.brightcat import to_gaia_like
    from wcc_sim.starflux import spt_from_bp_rp

    out = to_gaia_like(XHIP_ROWS)
    recovered = spt_from_bp_rp(
        np.asarray(out["phot_bp_mean_mag"]) - np.asarray(out["phot_rp_mean_mag"])
    )
    assert list(recovered) == list(out["spt"])


def test_to_gaia_like_keeps_positions_at_the_catalog_epoch():
    """Normalization must not move anything: propagation is the merge's job."""
    from wcc_sim.brightcat import to_gaia_like

    out = to_gaia_like(XHIP_ROWS)
    assert out["ra"][0] == pytest.approx(219.92041034, abs=1e-8)


def test_to_gaia_like_drops_rows_without_a_v_magnitude():
    """No magnitude means no rate; the row would render as a zero-flux star."""
    from wcc_sim.brightcat import to_gaia_like

    rows = XHIP_ROWS.copy()
    rows["Vmag"] = [np.nan, 1.35]
    with pytest.warns(UserWarning, match="no V magnitude"):
        out = to_gaia_like(rows)
    assert list(out["source_id"]) == [-71681]


def test_to_gaia_like_falls_back_to_g2v_without_a_colour():
    from wcc_sim.brightcat import to_gaia_like

    rows = XHIP_ROWS.copy()
    rows["B-V"] = [np.nan, 0.9]
    assert to_gaia_like(rows)["spt"][0] == "G2V"
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_brightcat.py -q -k to_gaia_like`
Expected: FAIL, `ImportError: cannot import name 'to_gaia_like'`.

- [ ] **Step 3: Implement it**

Append to `src/wcc_sim/brightcat.py`:

```python
def _floats(cat, name, default=np.nan):
    values = np.ma.masked_invalid(np.asarray(cat[name], dtype=float))
    return np.ma.filled(values, default)


def to_gaia_like(bright):
    """XHIP rows as a Gaia-shaped catalog, positions still at XHIP_EPOCH.

    The magnitude path is the point of this function. Hipparcos gives
    Johnson V and B-V; the rate model is normalized in Gaia G. So: pick the
    Pickles template whose synthetic B-V is nearest the star's, then
    G = V + (G-V) of that template. BP and RP are set from the same
    template's tabulated BP-RP, so `spt_from_bp_rp` independently recovers
    the type and `rates_for_catalog` needs no change at all. No empirical
    colour relation is involved.
    """
    v = _floats(bright, "Vmag")
    keep = np.isfinite(v)
    dropped = int((~keep).sum())
    if dropped:
        warnings.warn(
            f"{dropped} bright-catalog row(s) have no V magnitude and were "
            "dropped: without one there is no count rate",
            UserWarning,
        )
    rows = bright[keep]
    v = v[keep]
    spt = spt_from_b_v(_floats(rows, "B-V"))
    g = v + g_minus_v(spt)
    bp_rp = bp_rp_for_spt(spt)
    return Table({
        "source_id": -np.asarray(rows["HIP"], dtype=np.int64),
        "ra": _floats(rows, "RAJ2000"),
        "dec": _floats(rows, "DEJ2000"),
        "phot_g_mean_mag": g,
        "phot_bp_mean_mag": g + 0.5 * bp_rp,
        "phot_rp_mean_mag": g - 0.5 * bp_rp,
        "pmra": _floats(rows, "pmRA", 0.0),
        "pmdec": _floats(rows, "pmDE", 0.0),
        "parallax": _floats(rows, "Plx", 0.0),
        "radial_velocity": _floats(rows, "RV", 0.0),
        "spt": spt,
        "catalog": np.full(len(rows), "hipparcos"),
    })
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_brightcat.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/wcc_sim/brightcat.py tests/test_brightcat.py
git commit -m "feat(brightcat): normalize XHIP rows to Gaia-shaped rows via synthetic colours"
```

---

### Task 6: The merge policy

**Files:**
- Modify: `src/wcc_sim/brightcat.py`
- Test: `tests/test_brightcat.py`

**Interfaces:**
- Consumes: `astrometry.propagate`, `astrometry.crossmatch`, `to_gaia_like`, `catalog.GAIA_EPOCH`.
- Produces: `merge(gaia, bright, epoch=None, replace_mag=6.0, match_radius_arcsec=2.0, gaia_epoch=GAIA_EPOCH) -> (Table, dict)`. The dict has keys `bright_catalog`, `n_bright_added`, `n_bright_replaced`, `epoch`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_brightcat.py`:

```python
# --------------------------------------------------------------------------- #
# The merge                                                                    #
# --------------------------------------------------------------------------- #

def _gaia(*rows):
    """A minimal Gaia-shaped table: (source_id, ra, dec, g)."""
    return Table({
        "source_id": np.array([r[0] for r in rows], dtype=np.int64),
        "ra": [r[1] for r in rows],
        "dec": [r[2] for r in rows],
        "phot_g_mean_mag": [r[3] for r in rows],
        "phot_bp_mean_mag": [r[3] + 0.4 for r in rows],
        "phot_rp_mean_mag": [r[3] - 0.4 for r in rows],
        "pmra": [0.0] * len(rows),
        "pmdec": [0.0] * len(rows),
        "parallax": [0.0] * len(rows),
        "radial_velocity": [0.0] * len(rows),
    })


def test_merge_adds_bright_stars_gaia_does_not_have():
    """The gap-filling case, and the whole point of the feature."""
    from wcc_sim.brightcat import merge

    gaia = _gaia((1, 219.95, -60.9, 14.3))
    merged, info = merge(gaia, XHIP_ROWS, epoch=2000.0)
    assert info["n_bright_added"] == 2
    assert info["n_bright_replaced"] == 0
    assert info["bright_catalog"] == "hipparcos"
    assert info["epoch"] == pytest.approx(2000.0)
    assert len(merged) == 3
    assert list(merged["catalog"]).count("hipparcos") == 2


def test_merge_puts_the_bright_rows_first():
    """Row 0 is the row the PSF report decomposes, so the brightest added
    star belongs there."""
    from wcc_sim.brightcat import merge

    merged, _ = merge(_gaia((1, 219.95, -60.9, 14.3)), XHIP_ROWS, epoch=2000.0)
    assert merged["source_id"][0] == -71683
    assert merged["phot_g_mean_mag"][0] < merged["phot_g_mean_mag"][1]


def test_merge_replaces_a_matched_gaia_row_when_the_star_is_bright():
    """Brighter than the threshold, Gaia's photometry is where the saturation
    systematics live, so the Hipparcos row wins."""
    from wcc_sim.brightcat import merge

    # a Gaia entry at alpha Cen A's J2000 position with a nonsense magnitude
    gaia = _gaia((1, 219.90206584, -60.83397468, 11.0))
    merged, info = merge(gaia, XHIP_ROWS[:1], epoch=2000.0)
    assert info["n_bright_replaced"] == 1
    assert info["n_bright_added"] == 0
    assert len(merged) == 1
    assert merged["source_id"][0] == -71683


def test_merge_keeps_the_gaia_row_for_a_faint_match():
    """Fainter than the threshold Gaia is the better source, so the duplicate
    is dropped rather than added twice."""
    from wcc_sim.brightcat import merge

    faint = XHIP_ROWS[:1].copy()
    faint["Vmag"] = [8.0]
    gaia = _gaia((1, 219.90206584, -60.83397468, 7.8))
    merged, info = merge(gaia, faint, epoch=2000.0)
    assert (info["n_bright_added"], info["n_bright_replaced"]) == (0, 0)
    assert list(merged["source_id"]) == [1]


def test_merge_propagates_before_matching():
    """Un-propagated, alpha Cen A sits 92 arcsec from itself between the two
    catalog epochs and would be added as a second star. With propagation the
    same star is recognised as one."""
    from wcc_sim.brightcat import merge

    gaia = _gaia((1, 219.90206584, -60.83397468, 11.0))   # J2000 position
    merged, info = merge(gaia, XHIP_ROWS[:1], epoch=2000.0)
    assert info["n_bright_added"] == 0                     # matched, not added


def test_merge_defaults_to_the_gaia_epoch():
    """epoch=None leaves Gaia positions untouched and brings the bright rows
    to them, so existing simulations do not move."""
    from wcc_sim.brightcat import merge
    from wcc_sim.catalog import GAIA_EPOCH

    gaia = _gaia((1, 219.95, -60.9, 14.3))
    merged, info = merge(gaia, XHIP_ROWS, epoch=None)
    assert info["epoch"] == pytest.approx(GAIA_EPOCH)
    assert merged["ra"][list(merged["source_id"]).index(1)] == \
        pytest.approx(219.95, abs=1e-9)


def test_merge_with_no_bright_rows_leaves_the_catalog_alone():
    """A failed or empty bright query must not reorder or drop anything, and
    must say so in the provenance."""
    from wcc_sim.brightcat import _empty_table, merge

    gaia = _gaia((1, 219.95, -60.9, 14.3), (2, 219.96, -60.91, 12.0))
    merged, info = merge(gaia, _empty_table(), epoch=None)
    assert list(merged["source_id"]) == [1, 2]
    assert info["bright_catalog"] is None
    assert (info["n_bright_added"], info["n_bright_replaced"]) == (0, 0)
    assert list(merged["spt"]) == ["", ""]      # no override for Gaia rows


def test_merge_marks_gaia_rows_with_an_empty_spt_override():
    """rates_for_catalog treats a non-empty spt as an override; Gaia rows must
    keep using their own BP-RP."""
    from wcc_sim.brightcat import merge

    merged, _ = merge(_gaia((1, 219.95, -60.9, 14.3)), XHIP_ROWS, epoch=2000.0)
    by_id = dict(zip(map(int, merged["source_id"]), merged["spt"]))
    assert by_id[1] == ""
    assert by_id[-71683] != ""
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_brightcat.py -q -k merge`
Expected: FAIL, `ImportError: cannot import name 'merge'`.

- [ ] **Step 3: Implement it**

Append to `src/wcc_sim/brightcat.py`:

```python
#: Columns the merged table carries beyond the Gaia query's own.
_PROVENANCE = {"spt": "", "catalog": "gaia"}


def _with_provenance(gaia):
    """Gaia rows, plus the two columns the merged table needs.

    `spt` is empty because `rates_for_catalog` reads a non-empty value as an
    override; Gaia rows must keep deriving their type from BP-RP.
    """
    out = gaia.copy()
    for name, value in _PROVENANCE.items():
        if name not in out.colnames:
            out[name] = np.full(len(out), value)
    return out


def merge(gaia, bright, epoch=None, replace_mag=6.0, match_radius_arcsec=2.0,
          gaia_epoch=GAIA_EPOCH):
    """Gaia plus the bright rows it is missing, all at one epoch.

    Returns `(merged, info)`. Both catalogs are propagated to a common epoch
    *before* matching: Gaia is at J2016.0 and Hipparcos at J1991.25, and a
    3.7 arcsec/yr star is 92 arcsec from itself across that gap, so an
    un-propagated match would add it twice.

    Policy, per matched pair: brighter than `replace_mag` in G the bright row
    replaces the Gaia one, which is where DR3's saturation systematics live;
    fainter, Gaia wins and the duplicate is dropped. Unmatched bright rows
    are added -- the gap-filling case. Added rows are sorted brightest-first
    and prepended, so row 0 (the row the PSF report decomposes) is the
    brightest star in the field.

    `epoch=None` means the Gaia epoch, so Gaia positions do not move and an
    empty bright table gives back the input catalog untouched.
    """
    to_epoch = float(gaia_epoch if epoch is None else
                     (epoch.jyear if hasattr(epoch, "jyear") else epoch))
    info = {"bright_catalog": None, "n_bright_added": 0,
            "n_bright_replaced": 0, "epoch": to_epoch}

    gaia_moved = propagate(gaia, gaia_epoch, to_epoch, parallax="parallax",
                           rv="radial_velocity")
    if not len(bright):
        return _with_provenance(gaia_moved), info

    rows = to_gaia_like(bright)
    if not len(rows):
        return _with_provenance(gaia_moved), info
    rows = propagate(rows, XHIP_EPOCH, to_epoch, parallax="parallax",
                     rv="radial_velocity")
    rows.sort("phot_g_mean_mag")

    idx_bright, idx_gaia = crossmatch(rows, gaia_moved, match_radius_arcsec)
    bright_g = np.asarray(rows["phot_g_mean_mag"], dtype=float)
    wins = bright_g[idx_bright] < float(replace_mag) if len(idx_bright) else \
        np.array([], dtype=bool)

    take_bright = np.ones(len(rows), dtype=bool)
    take_bright[idx_bright[~wins]] = False          # matched and faint: drop
    drop_gaia = np.zeros(len(gaia_moved), dtype=bool)
    drop_gaia[idx_gaia[wins]] = True                # matched and bright: replace

    info["bright_catalog"] = "hipparcos"
    info["n_bright_replaced"] = int(wins.sum())
    info["n_bright_added"] = int(take_bright.sum()) - int(wins.sum())
    merged = vstack(
        [rows[take_bright], _with_provenance(gaia_moved[~drop_gaia])],
        join_type="exact",
    )
    return merged, info
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_brightcat.py -q`
Expected: all pass. If `vstack` raises on column mismatch, print both column sets — the fix is in `_with_provenance`, not a looser `join_type`.

- [ ] **Step 5: Commit**

```bash
git add src/wcc_sim/brightcat.py tests/test_brightcat.py
git commit -m "feat(brightcat): merge policy with common-epoch cross-match"
```

---

### Task 7: Gaia astrometry columns and cache self-heal

**Files:**
- Modify: `src/wcc_sim/catalog.py`
- Test: `tests/test_catalog.py`

**Interfaces:**
- Produces: `COLUMNS` extended with `pmra`, `pmdec`, `parallax`, `radial_velocity`; a cache file missing any of them is re-queried.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_catalog.py` (follow the file's existing patch style for `_run_query`):

```python
def test_columns_include_the_astrometry_needed_to_propagate():
    """Without proper motion, epoch propagation would silently do nothing."""
    from wcc_sim.catalog import COLUMNS

    for name in ("pmra", "pmdec", "parallax", "radial_velocity"):
        assert name in COLUMNS


def test_gaia_epoch_is_dr3s_reference_epoch():
    from wcc_sim.catalog import GAIA_EPOCH

    assert GAIA_EPOCH == pytest.approx(2016.0)


def test_a_cache_written_before_the_new_columns_is_requeried(monkeypatch, tmp_path):
    """An old cache lacks pmra; using it would contribute zero proper motion
    for the whole field, which is worse than a re-query."""
    import wcc_sim.catalog as cat

    stale = Table({"source_id": np.array([1], dtype=np.int64), "ra": [10.0],
                   "dec": [0.0], "phot_g_mean_mag": [12.0],
                   "phot_bp_mean_mag": [12.4], "phot_rp_mean_mag": [11.6]})
    key = "gaia_10.000000_+0.000000_100.0_21.00.ecsv"
    stale.write(tmp_path / key, format="ascii.ecsv")

    calls = []

    def fake(adql):
        calls.append(adql)
        fresh = stale.copy()
        for name in ("pmra", "pmdec", "parallax", "radial_velocity"):
            fresh[name] = [1.0]
        return fresh

    monkeypatch.setattr(cat, "_run_query", fake)
    out = cat.query_gaia(10.0, 0.0, 100.0, cache_dir=str(tmp_path))
    assert len(calls) == 1
    assert "pmra" in out.colnames
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_catalog.py -q -k "astrometry or epoch or requeried"`
Expected: FAIL — `pmra` not in `COLUMNS`, no `GAIA_EPOCH`, and the stale cache is returned without a query.

- [ ] **Step 3: Implement**

In `src/wcc_sim/catalog.py`, extend `COLUMNS` and guard the cache read:

```python
COLUMNS = [
    "source_id",
    "ra",
    "dec",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "pmra",
    "pmdec",
    "parallax",
    "radial_velocity",
]

#: Reference epoch of Gaia DR3 positions (Julian year).
GAIA_EPOCH = 2016.0
```

and in `query_gaia`:

```python
        if os.path.exists(cache_file):
            cached = Table.read(cache_file, format="ascii.ecsv")
            if set(COLUMNS).issubset(cached.colnames):
                return cached
            # written before the astrometry columns existed: using it would
            # contribute zero proper motion for the whole field
```

`_empty_table` needs the new columns too — it already builds from `COLUMNS`, so confirm its dtype rule still gives float for them (`np.int64 if c == "source_id" else float`): it does.

- [ ] **Step 4: Run the whole suite**

Run: `python -m pytest -q`
Expected: 207 pre-existing + the new tests pass. The ADQL string changes, so if `tests/test_catalog.py` asserts on it, update that assertion to include the new columns.

- [ ] **Step 5: Commit**

```bash
git add src/wcc_sim/catalog.py tests/test_catalog.py
git commit -m "feat(catalog): query Gaia astrometry, re-query pre-astrometry caches"
```

---

### Task 8: Wire it into `simulate_field`

**Files:**
- Modify: `src/wcc_sim/pipeline.py`
- Modify: `src/wcc_sim/fitswriter.py:13-50` (the `_CARDS` map)
- Modify: `src/wcc_sim/psfreport.py:219-223` (`_REPRO_KEYS`)
- Test: `tests/test_brightcat.py`

**Interfaces:**
- Consumes: `brightcat.query_bright`, `brightcat.merge`, `astrometry.propagate`, `catalog.GAIA_EPOCH`.
- Produces: `simulate_field(..., bright=True, epoch=None, bright_replace_mag=6.0, match_radius_arcsec=2.0)`; `params` keys `bright`, `bright_catalog`, `n_bright_added`, `n_bright_replaced`, `epoch`, `bright_replace_mag`, `match_radius_arcsec`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_brightcat.py`:

```python
# --------------------------------------------------------------------------- #
# End to end through simulate_field                                            #
# --------------------------------------------------------------------------- #

RA0, DEC0 = 219.90206584, -60.83397468          # alpha Cen A at J2000
SHAPE = (601, 901)


@pytest.fixture
def patched_pipeline(monkeypatch):
    """Both catalog queries patched: a one-star Gaia field plus the XHIP pair."""
    import wcc_sim.pipeline as pipeline

    gaia = Table({
        "source_id": np.array([1], dtype=np.int64),
        "ra": [RA0 + 0.002], "dec": [DEC0 + 0.002],
        "phot_g_mean_mag": [14.3], "phot_bp_mean_mag": [14.9],
        "phot_rp_mean_mag": [13.7], "pmra": [-5.0], "pmdec": [-3.0],
        "parallax": [1.0], "radial_velocity": [0.0],
    })
    monkeypatch.setattr(pipeline, "query_gaia",
                        lambda *a, **k: gaia.copy())
    monkeypatch.setattr(pipeline, "query_bright",
                        lambda *a, **k: XHIP_ROWS.copy())
    return gaia


def _run(**kw):
    from wcc_sim import simulate_field

    opts = dict(sensorfilter="zwo:g", focus=0, exptime=1.0, shape=SHAPE,
                add_noise=False, seed=0, wavelength_nm=450.0, scatter=False)
    opts.update(kw)
    return simulate_field(RA0, DEC0, **opts)


def test_bright_star_lands_at_the_pointing_at_its_own_epoch(patched_pipeline):
    """alpha Cen A's J2000 position is the pointing, so at epoch 2000 it must
    sit on the array centre -- and it must be row 0, the row the report
    decomposes."""
    field = _run(bright=True, epoch=2000.0)
    ny, nx = SHAPE
    assert int(field.catalog["source_id"][0]) == -71683
    assert float(field.catalog["x"][0]) == pytest.approx((nx - 1) / 2, abs=1.5)
    assert float(field.catalog["y"][0]) == pytest.approx((ny - 1) / 2, abs=1.5)


def test_the_bright_star_dominates_the_frame(patched_pipeline):
    """A G = -0.2 star at 1 s must be orders of magnitude above a G = 14 one:
    the flux path has to have used the synthetic G, not V or a zero rate."""
    field = _run(bright=True, epoch=2000.0)
    rates = np.asarray(field.catalog["rate_e_s"], dtype=float)
    assert rates[0] > 1e5 * rates[list(field.catalog["source_id"]).index(1)]


def test_both_components_of_the_pair_are_rendered(patched_pipeline):
    """B comes from the catalog, at its catalogued offset -- no invented
    separation or position angle."""
    field = _run(bright=True, epoch=2000.0)
    ids = [int(i) for i in field.catalog["source_id"]]
    assert -71681 in ids
    i = ids.index(-71681)
    sep_px = np.hypot(float(field.catalog["x"][i]) - float(field.catalog["x"][0]),
                      float(field.catalog["y"][i]) - float(field.catalog["y"][0]))
    assert 100.0 < sep_px < 3000.0


def test_bright_false_reproduces_the_gaia_only_field(patched_pipeline):
    field = _run(bright=False)
    assert list(map(int, field.catalog["source_id"])) == [1]
    assert field.params["bright_catalog"] is None
    assert field.params["n_bright_added"] == 0


def test_params_record_the_merge_provenance(patched_pipeline):
    field = _run(bright=True, epoch=2000.0)
    p = field.params
    assert p["bright"] is True
    assert p["bright_catalog"] == "hipparcos"
    assert p["n_bright_added"] == 2
    assert p["epoch"] == pytest.approx(2000.0)
    assert p["bright_replace_mag"] == pytest.approx(6.0)


def test_a_supplied_catalog_skips_both_queries(monkeypatch):
    """Every existing test passes catalog=; none of them may acquire a
    network call or a moved star."""
    import wcc_sim.pipeline as pipeline

    def explode(*a, **k):
        raise AssertionError("query must not run when catalog= is given")

    monkeypatch.setattr(pipeline, "query_gaia", explode)
    monkeypatch.setattr(pipeline, "query_bright", explode)
    given = Table({
        "source_id": np.array([7], dtype=np.int64), "ra": [RA0], "dec": [DEC0],
        "phot_g_mean_mag": [10.0], "phot_bp_mean_mag": [10.4],
        "phot_rp_mean_mag": [9.6],
    })
    field = _run(catalog=given, bright=True, epoch=2026.6)
    assert list(map(int, field.catalog["source_id"])) == [7]


def test_the_new_knobs_are_in_the_reproduction_snippet(patched_pipeline):
    """The report's TO REPRODUCE block has to reproduce the field, and the
    sky epoch changes the field."""
    from wcc_sim.psfreport import reproduction_call

    src = reproduction_call(_run(bright=True, epoch=2000.0))
    assert "bright=True" in src
    assert "epoch=2000.0" in src


def test_the_fits_header_records_the_bright_merge(patched_pipeline, tmp_path):
    from astropy.io import fits

    out = tmp_path / "merged.fits"
    _run(bright=True, epoch=2000.0, output=str(out))
    header = fits.getheader(str(out))
    assert header["BRIGHTCT"] == "hipparcos"
    assert header["SKYEPOCH"] == pytest.approx(2000.0)
    assert header["NBRIGHT"] == 2
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_brightcat.py -q -k "bright_star or dominates or components or params_record or supplied or snippet or fits_header"`
Expected: FAIL, `TypeError: simulate_field() got an unexpected keyword argument 'bright'`.

- [ ] **Step 3: Add the parameters and the merge step to `pipeline.py`**

Add the imports:

```python
from .astrometry import propagate
from .brightcat import merge as merge_bright
from .brightcat import query_bright
from .catalog import GAIA_EPOCH, query_gaia
```

Add to the signature, after `mag_limit=21.0`:

```python
    bright=True,
    epoch=None,
    bright_replace_mag=6.0,
    match_radius_arcsec=2.0,
```

Document them in the docstring, after the `catalog=` paragraph:

```
    `bright=True` (default) supplements the Gaia query with Hipparcos/XHIP
    (see wcc_sim.brightcat): Gaia DR3's brightest source is G = 1.73, so the
    naked-eye stars that drive stray-light requirements are missing from it
    entirely. `epoch=` (a Julian year) propagates both catalogs' proper
    motions to that epoch; the default leaves them at Gaia's J2016.0. Both
    are ignored when `catalog=` is given.
```

Replace the query block:

```python
    bright_info = {"bright_catalog": None, "n_bright_added": 0,
                   "n_bright_replaced": 0, "epoch": None}
    if catalog is None:
        catalog = query_gaia(
            ra, dec, radius_arcsec, mag_limit=mag_limit, cache_dir=cache_dir
        )
        if bright:
            catalog, bright_info = merge_bright(
                catalog,
                query_bright(ra, dec, radius_arcsec, cache_dir=cache_dir),
                epoch=epoch,
                replace_mag=bright_replace_mag,
                match_radius_arcsec=match_radius_arcsec,
            )
        elif epoch is not None:
            catalog = propagate(catalog, GAIA_EPOCH, epoch,
                                parallax="parallax", rv="radial_velocity")
            bright_info["epoch"] = float(
                epoch.jyear if hasattr(epoch, "jyear") else epoch
            )
    catalog = catalog.copy()
```

Add to the `params` dict, next to `"mag_limit"`:

```python
        "bright": bool(bright),
        "bright_replace_mag": float(bright_replace_mag),
        "match_radius_arcsec": float(match_radius_arcsec),
        **bright_info,
```

- [ ] **Step 4: Add the header cards to `fitswriter._CARDS`**

```python
    "bright_catalog": ("BRIGHTCT", "bright-star catalog merged in (empty=none)"),
    "epoch": ("SKYEPOCH", "[yr] Julian epoch of catalog positions"),
    "n_bright_added": ("NBRIGHT", "bright rows added to the Gaia catalog"),
    "n_bright_replaced": ("NBRIGHRP", "Gaia rows replaced by bright-catalog rows"),
```

- [ ] **Step 5: Add the knobs to `psfreport._REPRO_KEYS`**

```python
_REPRO_KEYS = (
    "ra", "dec", "sensorfilter", "focus", "exptime", "n_reads", "pa",
    "shape", "stamp_npix", "oversample", "wavelength_nm", "jitter_sigma_mas",
    "scatter", "scatter_fraction", "wings", "chromatic", "add_noise", "seed",
    "bright", "epoch",
)
```

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest -q`
Expected: everything passes. `test_reproduction_settings_cover_every_psf_knob` and `tests/test_fitswriter.py` are the two most likely to need their key lists extended — extend them, do not weaken them.

- [ ] **Step 7: Commit**

```bash
git add src/wcc_sim/pipeline.py src/wcc_sim/fitswriter.py \
        src/wcc_sim/psfreport.py tests/test_brightcat.py tests/test_fitswriter.py \
        tests/test_psfreport.py
git commit -m "feat(pipeline): merge the bright catalog and propagate to an epoch"
```

---

### Task 9: Rewrite the alpha Cen script on the real sky

**Files:**
- Modify: `scripts/example_alpha_cen.py`

**Interfaces:**
- Consumes: `simulate_field(bright=..., epoch=...)`, `field.params` provenance keys, the `catalog` column.

- [ ] **Step 1: Delete the injection and let the pipeline build the catalog**

Remove `star_pair`-style construction entirely: `ALPHA_CEN_A_G`, `ALPHA_CEN_A_BP_RP`, `build_catalog`, `brightest_on_chip_first`, `field_radius_arcsec`, the `--gaia-only` and `--mag` flags, and the `catalog=` argument to `simulate_field`. The pointing constant `TARGET` stays. Replace the module docstring with:

```python
"""The scattered-light halo on a real field: alpha Centauri.

    python scripts/example_alpha_cen.py [--outdir alpha_cen_out]

A real pointing, 14 39 36.50 -60 50 02.3, with no synthetic sources at all:
Gaia DR3 for the field and Hipparcos/XHIP for the bright end, which is where
alpha Cen A and B come from -- Gaia has neither, its brightest source being
G = 1.73.

The coordinate is alpha Cen A's J2000 position, so `--epoch` defaults to
2000.0 and the star sits at the array centre with B at its catalogued
offset. The pair moves 3.71 arcsec/yr, so `--epoch 2026.6` walks it 98
arcsec west, off a 162-arcsec-wide array -- worth seeing once.
"""
```

- [ ] **Step 2: Add `--epoch` and pass the new knobs**

```python
    p.add_argument("--epoch", type=float, default=2000.0,
                   help="Julian year to propagate catalog positions to; the "
                        "default matches the pointing, which is alpha Cen A's "
                        "J2000 position")
    p.add_argument("--no-bright", action="store_true",
                   help="Gaia only, for comparison: the field without its "
                        "brightest stars")
```

```python
    field = simulate_field(
        TARGET.ra.deg, TARGET.dec.deg,
        sensorfilter=args.sensorfilter,
        focus=0,
        exptime=args.exptime,
        shape=shape,
        wavelength_nm=args.wavelength,
        scatter=True,
        bright=not args.no_bright,
        epoch=args.epoch,
        mag_limit=args.mag_limit,
        seed=0,
        cache_dir=cache_dir,
        output=os.path.join(args.outdir, "alpha_cen.fits"),
        report=os.path.join(args.outdir, "alpha_cen_psf.pdf"),
    )
```

- [ ] **Step 3: Print the provenance instead of the injection note**

Replace the `row 0` line and add a catalog-provenance line:

```python
    p_ = field.params
    print(f"catalogs           Gaia DR3 + "
          f"{p_['bright_catalog'] or 'none'}: "
          f"{p_['n_bright_added']} bright rows added, "
          f"{p_['n_bright_replaced']} replaced, epoch J{p_['epoch']:.2f}")
    print(f"row 0              {cat['catalog'][0]}, "
          f"G = {float(cat['phot_g_mean_mag'][0]):.2f}, {cat['spt'][0] or '-'} at "
          f"({float(cat['x'][0]):,.0f}, {float(cat['y'][0]):,.0f}) px, "
          f"{dec['flux_e']:.4g} e-")
```

- [ ] **Step 4: Run it for real and read the output**

Run: `python scripts/example_alpha_cen.py --outdir /tmp/ac --exptime 10`
Expected, and each of these is a check on a different part of the chain:
- `catalogs Gaia DR3 + hipparcos: 2 bright rows added, 0 replaced, epoch J2000.00`
- `row 0 hipparcos, G = -0.18, G5V at (4,784, 3,189) px` — within a pixel or two of frame centre
- alpha Cen B present in the neighbour table, a few hundred pixels away
- the halo above background across the whole array, as before

Then run `python scripts/example_alpha_cen.py --outdir /tmp/ac26 --exptime 10 --epoch 2026.6` and confirm the pair has left the array (row 0 becomes a Gaia star, and the bright rows report as off-chip). If alpha Cen A is *not* at the centre at epoch 2000, stop: either `XHIP_EPOCH` or the `pm_ra_cosdec` convention is wrong, and the propagation tests in Task 2 should have caught it.

- [ ] **Step 5: Look at the report PDF**

Open `/tmp/ac/alpha_cen_psf.png`. The frame should show the bright star at centre with its halo, the saturation mask directly below it at the same size, and the summary block reporting `2 star` plus the Gaia field count.

- [ ] **Step 6: Commit**

```bash
git add scripts/example_alpha_cen.py
git commit -m "feat(scripts): alpha Cen from the real catalogs, no injected star"
```

---

### Task 10: Final verification and PR

**Files:** none

- [ ] **Step 1: Full suite and lint**

```bash
python -m pytest -q
python -m ruff check src/wcc_sim scripts tests
```
Expected: all tests pass (207 pre-existing plus roughly 35 new), ruff clean.

- [ ] **Step 2: Confirm no test reaches the network**

```bash
python -m pytest -q -p no:cacheprovider
```
Then check by inspection that every new test either patches `_run_query`/`query_bright`/`query_gaia` or passes `catalog=`. A test that hangs for seconds is a test that is talking to VizieR.

- [ ] **Step 3: Open the PR**

```bash
git push -u origin feature/bright-star-catalog
gh pr create --title "Bright-star catalog supplement and epoch propagation" --body "$(cat <<'EOF'
## Summary

Gaia DR3's brightest source is G = 1.73 and it holds only 150 sources
brighter than G = 3, so pointing WCC at a naked-eye star simulated a frame
of faint field stars and no halo. This fills the bright end from
Hipparcos/XHIP and propagates proper motion to a requested epoch.

- `wcc_sim.brightcat` — XHIP cone search, normalization to Gaia-shaped rows
  via synthetic Johnson colours, and the merge policy
- `wcc_sim.astrometry` — space-motion propagation and one-to-one cross-match
- `simulate_field(bright=True, epoch=None, bright_replace_mag=6.0)`
- `scripts/example_alpha_cen.py` — the alpha Cen field with no injected star

## Why the epoch matters

alpha Cen moves 3.71"/yr and the WCC field is 162" x 108". Gaia is at
J2016.0 and Hipparcos at J1991.25, so an un-propagated merge would place the
same star twice, 92" apart. Neglecting radial velocity costs 4.5 px at
epoch 2026.6.

## Validation

- pm-only propagation reproduces VizieR's own J2000 positions for HIP 71683
  to 0.000 mas
- the G2V template's synthetic B-V is 0.650, the literature solar value
- alpha Cen A lands within a pixel of the pointing at epoch J2000

Spec: `docs/superpowers/specs/2026-08-24-bright-star-catalog-design.md`
Plan: `docs/superpowers/plans/2026-08-24-bright-star-catalog.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**Spec coverage:** XHIP-one-query → Task 4. Epoch trap → Task 4 Step 3 (`XHIP_EPOCH` comment) and Task 2's tests. Propagation with parallax/RV → Task 2. Common epoch before matching → Task 6 (`test_merge_propagates_before_matching`). Synthetic photometry → Task 1. Merge policy, all three branches → Task 6. `astrometry.py` / `brightcat.py` / `starflux` / `catalog.py` / `simulate_field` / script components → Tasks 2-3 / 4-6 / 1 / 7 / 8 / 9. Error handling: service failure → Task 4; no `Vmag` → Task 5; blank `B-V` → Tasks 1 and 5; no proper motion → Task 2; Gaia empty + bright non-empty → covered by `merge`'s `vstack` on an empty Gaia table, exercised indirectly by `test_merge_adds_bright_stars_gaia_does_not_have`. Provenance in `params`/FITS/report → Task 8.

**Placeholders:** none — every code step carries the code and every run step carries the command and the expected output.

**Type consistency:** `spt_from_b_v`/`g_minus_v`/`bp_rp_for_spt` (Task 1) are consumed with those exact names in Task 5. `propagate(..., parallax=, rv=)` keyword names (Task 2) match every call site in Tasks 6 and 8. `crossmatch(a, b, radius_arcsec)` returns `(idx_a, idx_b)` in that order, and Task 6 calls it as `crossmatch(rows, gaia_moved, ...)` and unpacks `(idx_bright, idx_gaia)` accordingly. `merge` returns `(Table, dict)` with the four `info` keys that Task 8 splats into `params` and Task 9 prints.
