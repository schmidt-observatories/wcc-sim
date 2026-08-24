# Bright-star catalog supplement and proper-motion epoch propagation

**Date:** 2026-08-24
**Deliverables:**
- `wcc_sim.brightcat` — Hipparcos/XHIP cone search, normalized into Gaia-like
  rows, merged into the Gaia catalog by positional cross-match.
- `wcc_sim.astrometry` — space-motion propagation to a requested epoch and
  the cross-match used by the merge.
- `wcc_sim.starflux` — vendored Johnson B and V passbands and per-template
  synthetic colors, so a V magnitude becomes a G magnitude without an
  empirical color relation.
- `simulate_field(bright=True, epoch=None, bright_replace_mag=6.0)`.
- `scripts/example_alpha_cen.py` — rewritten with no injected star: the
  alpha Cen field as the catalogs actually describe it.

## Goal

A WCC field containing a naked-eye star currently simulates without it.
Gaia DR3 has no entry for the brightest few hundred stars in the sky, so
pointing at one produces a frame of faint field stars and no halo — the
opposite of the stray-light case the simulator exists to study. Fill Gaia's
bright end from Hipparcos, and propagate proper motion so a high-motion star
lands where it actually is at the observation epoch.

## Why: Gaia's bright end, measured

Queried against `gaiadr3.gaia_source` on 2026-08-24:

| Quantity | Value |
|---|---|
| Brightest source in DR3 | G = 1.73 |
| Sources with G < 3 | 150 |
| Sources with G < 6 | 6,764 |
| Brightest DR3 source within 3' of 14 39 36.5 -60 50 02 | G = 12.72 |

The nominal DR3 range is G ~ 3-21. Brighter than G ~ 6 the catalog is
incomplete and its photometry carries saturation systematics; brighter than
G = 1.73 it is empty. The sky has roughly 30 stars brighter than V = 1.7,
and they are exactly the stars whose scattered-light halo drives WCC
requirements.

Hipparcos has the two this field is about (`I/239/hip_main`, and identical
astrometry in XHIP):

| HIP | V | B-V | SpType | pm | Plx | RV |
|---|---|---|---|---|---|---|
| 71683 (alpha Cen A) | -0.01 | 0.71 | G2V | 3.710"/yr | 742.12 mas | -21.4 km/s |
| 71681 (alpha Cen B) | 1.35 | 0.90 | K1V | 3.724"/yr | 742.12 mas | -18.6 km/s |

Both are absent from DR3. Both come with the proper motion that makes the
second half of this spec necessary: the WCC field is 162" x 108", and this
pair crosses 3.71" of it per year.

## Design decisions

### Bright catalog: XHIP, one query

`V/137D/XHIP` (Anderson & Francis 2012) is Hipparcos with a radial velocity
column, and returns everything the merge needs in one cone search: `HIP`,
position, `pmRA`, `pmDE`, `Plx`, `RV`, `Vmag`, `B-V`, `SpType`. Hipparcos is
complete to V ~ 7.3 and largely complete to V ~ 9, which covers Gaia's gap
with room to spare.

Rejected:
- **`I/239/hip_main` alone** — no radial velocity, and RV is worth 4.5 px of
  position for this field (below).
- **`I/239/hip_main` + XHIP for RV** — two queries, two failure modes, for
  data XHIP already carries.
- **Tycho-2 (`I/259/tyc2`)** — covers V ~ 7-11.5, where Gaia is already
  complete, so nearly every row is a duplicate; no spectral types; BT/VT
  would need its own color transformation. Its VizieR endpoint also dropped
  the connection during design queries.
- **Yale BSC (`V/50`)** — V < 6.5 only, a subset of what Hipparcos gives.

**Trap to document in code:** XHIP's position columns are named `RAJ2000` /
`DEJ2000`, but the values are byte-identical to `hip_main`'s `RAICRS` /
`DEICRS` — the J2000 in the name is the equinox, not the epoch. The
positions are at **epoch J1991.25**. Treating them as epoch J2000 puts
alpha Cen 32" from the truth.

### Epoch: propagate with parallax and radial velocity, to a common epoch before matching

`astrometry.propagate` wraps `SkyCoord.apply_space_motion`, passing parallax
as distance and RV when the catalog has it. Validated against VizieR's own
computed J2000 positions for HIP 71683: propagating from J1991.25 reproduces
`_RA.icrs` / `_DE.icrs` to **0.0 mas**.

Radial velocity is not optional at this precision. Neglecting it for
HIP 71683 (mu = 3.71"/yr, Plx = 742 mas, RV = -21.4 km/s) costs:

| Target epoch | Error without RV | In pixels (16.869 mas/px) |
|---|---|---|
| J2000.0 | 4.6 mas | 0.3 px |
| J2016.0 | 36.9 mas | 2.2 px |
| J2026.6 | 75.3 mas | 4.5 px |

Rows with masked `Plx` or `RV` are propagated with whatever they do have;
`apply_space_motion` treats a missing distance as infinite, which is correct
for a distant star and the only sane fallback for an unmeasured one.

**Both catalogs are propagated to a single epoch before the cross-match.**
Gaia positions are at J2016.0 and Hipparcos at J1991.25, 24.75 years apart:
un-propagated, alpha Cen A sits 92" from itself between the two catalogs and
would never match. `epoch=None` propagates the bright rows to Gaia's
J2016.0 and leaves Gaia rows untouched, so Gaia positions in an existing
simulation are bit-unchanged (its *contents* change only by the bright rows
the merge adds, which is the point of `bright=True`); an explicit `epoch`
propagates both. `params["epoch"]` records the effective epoch, J2016.0
included, so the output always states which epoch its sky is at.

### Photometry: synthetic colors from the same spectral library

Hipparcos gives Johnson V and B-V; the flux path is normalized in Gaia G.
Vendor the two SVO passband curves next to the existing `gaia_dr3_g.dat`,
integrate every Pickles template through B, V and G once (memoized), and use
the resulting table twice: pick the template whose synthetic B-V is nearest
the star's, then set `G = V + (G-V)_template`. The row's BP and RP
magnitudes are set from the same template's synthetic colors, so
`spt_from_bp_rp` independently recovers the same template and
`rates_for_catalog` needs no change at all.

Rejected:
- **A published (B-V) -> (G-V) polynomial** — an empirical relation, fit to
  a different sample, inconsistent with the library we actually integrate,
  and one more set of coefficients to cite and maintain.
- **Parsing Hipparcos SpType strings** — the catalog's types are
  heterogeneous (`A0Vn`, `F5IV-V`, `M2Iab`), and the Pickles set wired up
  here is dwarfs-only, so most non-dwarf strings would fall back anyway.
  The strings are carried through to the output catalog as provenance, not
  used for the rate.

### Merge: cross-match, Hipparcos wins brighter than V = 6

Hipparcos is complete to V ~ 7.3 and Gaia is sound for G > 6, so most
Hipparcos rows in a field duplicate a Gaia row. After propagating both to
the common epoch:

- **No Gaia counterpart within `match_radius_arcsec` (default 2.0)** — add
  the Hipparcos row. This is the gap-filling case.
- **Counterpart, and V < `bright_replace_mag` (default 6.0)** — replace the
  Gaia row, which is where DR3's saturation systematics live.
- **Counterpart, and V >= 6** — keep the Gaia row and drop the Hipparcos one.

Matching is nearest-neighbour with a one-to-one constraint: if two Hipparcos
rows fall on one Gaia row, the closer wins and the other is treated as
unmatched. The 2" radius is far below the alpha Cen AB separation at any
epoch of interest, so A cannot be matched to B's counterpart.

Rejected: **Gaia always wins** (a bright star with a spurious faint G keeps
it), and **a pure magnitude split with no matching** (a star straddling the
cut is either duplicated or dropped).

## Components

### `wcc_sim/astrometry.py`

```python
propagate(cat, from_epoch, to_epoch, ra="ra", dec="dec",
          pmra="pmra", pmdec="pmdec", plx=None, rv=None)  -> Table (copy)
crossmatch(a, b, radius_arcsec)                            -> (idx_a, idx_b)
```

Epochs are Julian years (float) or `astropy.time.Time`. `propagate` returns
a copy with `ra`/`dec` replaced and is a no-op when the epochs are equal.
Column names are parameters because the two catalogs disagree on them.

### `wcc_sim/brightcat.py`

```python
BRIGHT_COLUMNS = [...]                  # HIP, position, pm, Plx, RV, Vmag, B-V, SpType
query_bright(ra, dec, radius_arcsec, cache_dir=None)  -> Table
to_gaia_like(bright)                                  -> Table
merge(gaia, bright, epoch, replace_mag=6.0, match_radius_arcsec=2.0)
    -> (Table, dict)                                  # merged, provenance counts
_run_query(...)                                       # thin VizieR seam, patched in tests
```

Cached as ECSV in `cache_dir` under a key mirroring `query_gaia`'s
(`xhip_{ra}_{dec}_{radius}.ecsv`), so a repeat run is offline.
`to_gaia_like` emits `source_id = -HIP` (negative, cannot collide with a
Gaia id), `ra`/`dec` at J1991.25, the synthetic G/BP/RP, `spt`, the
astrometry columns, and `catalog = "hipparcos"`.

### `wcc_sim/starflux.py` additions

`johnson_b_bandpass()`, `johnson_v_bandpass()` — same `Empirical1D` loader
as `gaia_g_bandpass`, reading two new vendored `data/johnson_[bv].dat`
curves from SVO FPS. `template_colors()` — memoized table of synthetic
`(B-V, G-V, BP-RP)` per Pickles type. `spt_from_b_v(b_v)` — nearest
template by synthetic B-V, mirroring `spt_from_bp_rp`, NaN -> G2V.

### `wcc_sim/catalog.py` changes

`COLUMNS` gains `pmra`, `pmdec`, `parallax`, `radial_velocity`. A cached
ECSV missing any of them is re-queried and overwritten rather than used, so
existing `notebooks/gaia_cache/*.ecsv` self-heal instead of silently
contributing zero proper motion. Masked values are filled with 0 (pm) or
left masked (parallax, RV), which `propagate` handles.

### `simulate_field` changes (`wcc_sim/pipeline.py`)

Three new parameters — `bright=True`, `epoch=None`, `bright_replace_mag=6.0`
— and one new step, between the Gaia query and `rates_for_catalog`:

```
query_gaia -> query_bright -> propagate both to common epoch
           -> merge -> (existing rate/render path, unchanged)
```

All three are ignored when `catalog=` is supplied, exactly as the Gaia query
already is, so every existing test and hand-built catalog is unaffected.
`params` gains `bright`, `bright_catalog` (`"hipparcos"` or `None`),
`n_bright_added`, `n_bright_replaced`, `epoch`, `bright_replace_mag`,
`match_radius_arcsec`; `reproduction_settings()` gains the same keys so the
report's TO REPRODUCE snippet still reproduces the field, and
`fitswriter._CARDS` gains `BRIGHTCAT`, `EPOCH`, `NBRIGHT`, `NBRIGHREP`
cards (the writer already renders `None` as an empty card).

### `scripts/example_alpha_cen.py`

The injected star is deleted. `--epoch` defaults to **2000.0**, matching the
pointing: `14 39 36.50 -60 50 02.3` is alpha Cen A's J2000 position, so at
that epoch the star sits at frame centre and B appears at its real
catalogued offset — no invented separation or position angle. `--epoch
2026.6` then demonstrates the pair drifting 98" west, off a 162"-wide array,
which is the point worth showing. The printout gains the merge provenance
(added/replaced counts, epoch, which rows came from which catalog) and keeps
the existing halo-versus-background table.

## Error handling

- **VizieR unreachable, timing out, or returning no table** — `UserWarning`
  naming the failure, empty bright table, simulation continues Gaia-only
  with `bright_catalog=None` recorded, so the report shows the bright end
  was not filled rather than implying it was. Mirrors `query_gaia`'s
  existing warn-and-return-empty behaviour.
- **Row with no `Vmag`** — dropped, counted, one warning for the batch: no
  magnitude means no rate.
- **Row with blank `B-V`** — G2V, the same fallback `spt_from_bp_rp` uses
  for a missing color.
- **`epoch` given but a row has no proper motion** — position used as-is.
- **Gaia query empty and bright query non-empty** — valid; the field is the
  bright rows alone.

## Testing

No test touches the network: `brightcat._run_query` is patched the way
`catalog._run_query` already is, with a fixture table carrying the real
HIP 71681/71683 values from this spec.

- `propagate` on HIP 71683 from J1991.25 to J2000.0 lands within 5 mas of
  VizieR's `_RA.icrs`/`_DE.icrs` — an external check, not self-consistency.
- Dropping RV from that same propagation moves the star by 4-5 mas at
  J2000 and ~75 mas at J2026.6, confirming the RV path is wired in.
- `crossmatch` respects the radius and the one-to-one constraint.
- The three merge branches: unmatched added, matched V < 6 replaced,
  matched V >= 6 kept, with the provenance counts asserted.
- Synthetic G-V for the G2V template is ~ -0.14, and alpha Cen A's row comes
  out at G ~ -0.15, both against published values.
- A `_run_query` that raises produces a warning, a Gaia-only catalog, and
  `bright_catalog=None`.
- End-to-end `simulate_field(bright=True, epoch=2000.0)` with the patched
  query puts alpha Cen A within a pixel of frame centre, B at its
  catalogued offset, and both saturating.
- A cached Gaia ECSV lacking `pmra` is re-queried rather than trusted.

## Out of scope (later PRs)

- Tycho-2 or any V ~ 7-11 supplement: Gaia already covers it.
- Giants and supergiants: the Pickles set wired up here is dwarfs-only, an
  existing limitation of the Gaia path too, not introduced by this change.
- Gaia's own bright-star photometric corrections for 3 < G < 6.
- Resolved binaries as anything other than what the catalog says: alpha Cen
  A and B are separate HIP rows and so come out as two stars for free.
- Variability, and epoch-dependent binary orbits.
