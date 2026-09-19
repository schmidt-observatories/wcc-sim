"""Builder for notebooks_scrap/halpha_hci_feasibility.ipynb (nbformat).

Run from notebooks_scrap/:  python build_halpha_hci_feasibility.py
then  jupyter nbconvert --to notebook --execute --inplace halpha_hci_feasibility.ipynb
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "halpha_hci_feasibility.ipynb")

cells = []
md = lambda s: cells.append(new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(new_code_cell(s.strip("\n")))

md(r"""
# WCC high-contrast imaging of accreting protoplanets in H-alpha: feasibility

**Question.** Can the WCC detect the H-alpha emission of young, accreting planets (the PDS 70 b/c class)
next to their host star, and which of the three H-alpha filters in the WCC ETC (2, 6, 20 nm) does it best?

**Approach.** Everything runs through the WCC ETC instrument model (`wcc_etc`, via `wcc_sim`). The host
star is a Pickles template normalised in Gaia $G$; the planet is a pure emission line of fixed integrated
flux (the ETC's absolute-flux `emission` source). The stellar PSF is the ETC's jitter-blurred Airy pattern,
optionally degraded to a Strehl ratio $S<1$ with the core + halo model of
`jitter_strehl_sensitivity.ipynb`, plus the FRED scattered-light halo (`wcc_sim.scatter`). The host is kept
below saturation by stacking short frames. Two contrast metrics per separation:

1. **photon-noise 5σ contrast** after an ideal PSF subtraction: planet-to-star flux ratio at which the planet's
   aperture signal is 5x the photon + detector noise of the stellar light in that aperture;
2. **raw contrast**: planet flux equal to the stellar light in the same aperture, i.e. what it takes to see the
   planet as a bump on the wing with no subtraction at all.

Baseline PSF: jitter 10 mas (1σ per axis, the Lazuli config), $S = 1$. The requirement corner ($S = 0.8$ at r,
615 nm, which is $S = 0.82$ at 656 nm by Maréchal) is shown where it matters; the full jitter/Strehl
sensitivity is in `halpha_hci_jitter_strehl.ipynb`. Helpers live in `hci_psf.py` next to this notebook.
""")

code(r"""
import os, time, warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

plt.style.use('gks')
warnings.filterwarnings('ignore')

from hci_psf import (Instrument, contrast_curve, aperture_fractions, strehl_at, HALO_K, OVERSAMPLE,
                     airy_fwhm_mas)

FILTERS = ['zwo:halpha2', 'zwo:halpha6', 'zwo:halpha20']
FLAB = {'zwo:halpha2': 'H-alpha 2 nm', 'zwo:halpha6': 'H-alpha 6 nm', 'zwo:halpha20': 'H-alpha 20 nm'}
TEAL, RED, AMBER, BLUE, GREEN, DBLUE = '#00798c', '#d1495b', '#edae49', '#30638e', '#66a182', '#003d5b'
FCOL = dict(zip(FILTERS, [TEAL, AMBER, RED]))
REQ_JIT, REQ_S_R = 10.0, 0.8
OUT = 'halpha_hci_out'
os.makedirs(OUT, exist_ok=True)

# ---- PDS 70 (the worked example; edit here) ---------------------------------------------------
HOST_SPT, HOST_G = 'K7V', 11.7          # PDS 70: K7, Gaia G ~ 11.7, d = 112 pc
HOST_HA_EW_A = 0.0                      # host's own H-alpha emission EW [A]; 0 = continuum only (see caveats)
# line fluxes [erg/s/cm^2] and projected separations [mas]; see the PDS 70 section for the sources
PLANETS = {
    'b': dict(sep_mas=180.0, flux=8.1e-16, flux_lo=2.3e-16, flux_hi=1.6e-15),
    'c': dict(sep_mas=220.0, flux=3.1e-16, flux_lo=1.9e-16, flux_hi=4.8e-16),
}

def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=200)
    fig.savefig(os.path.join(OUT, name + '.pdf'))

t0 = time.time()
INST = {f: Instrument(f, HOST_SPT, HOST_G) for f in FILTERS}
for inst in INST.values():
    inst.star_rate *= 1 + HOST_HA_EW_A / (10 * inst.band_eqw_nm)
S_REQ = float(strehl_at(REQ_S_R, INST['zwo:halpha2'].wavelength_m * 1e9))
i2 = INST['zwo:halpha2']
print(f'lambda_eff = {i2.wavelength_m * 1e9:.1f} nm, D = {i2.diameter_m:.3f} m, Airy FWHM = {i2.fwhm_mas:.1f} mas, '
      f'plate scale {i2.plate_mas:.2f} mas/px ({i2.fwhm_mas / i2.plate_mas:.2f} px/FWHM)')
print(f'S = {REQ_S_R} at 615 nm  ->  S = {S_REQ:.3f} at H-alpha;  setup {time.time() - t0:.1f} s')
""")

md(r"""
## 1. The three H-alpha filters

The planet is a line: its detected rate is the same in all three filters (the line sits inside each band).
The star is continuum: its rate scales with the filter's equivalent width. So the planet/star contrast
improves as 1/bandwidth, and the narrowest filter gives the biggest gain, provided its peak throughput
holds up. A second, less obvious win: the host saturates the 16 ke- IMX455 well quickly, so the frame time
is set by the star's peak pixel and the narrow filter allows longer frames, fewer reads, less read noise.
""")

code(r"""
fig, ax = plt.subplots(figsize=(7, 3.8))
for f in FILTERS:
    bp = INST[f].sim.sensor.bandpass
    w = bp.waveset.value / 10
    m = (w > 640) & (w < 672)
    ax.plot(w[m], bp(bp.waveset)[m].value, color=FCOL[f], label=f'{FLAB[f]} (EW {INST[f].band_eqw_nm:.1f} nm)')
ax.axvline(656.3, color='0.4', ls='--', lw=1)
ax.set_xlabel('wavelength [nm]'); ax.set_ylabel('total throughput (EOL)'); ax.legend(fontsize=8)
ax.set_title('WCC H-alpha filters as in the ETC (IMX455 channel)')
fig.tight_layout(); savefig(fig, 'fig01_filters')

rows = []
for f in FILTERS:
    inst = INST[f]
    lr = inst.line_rate(1e-16)
    tf = inst.frame_time(REQ_JIT, 1.0)
    rows.append(dict(filter=FLAB[f], eqw_nm=inst.band_eqw_nm, star_e_s=inst.star_rate, line_1e16_e_s=lr,
                     contrast_1e16=lr / inst.star_rate, sky_e_s_px=inst.sky_rate, t_frame_s=tf,
                     frames_per_hour=3600 / tf))
FT = Table(rows)
for c, fmt in [('eqw_nm', '%.2f'), ('star_e_s', '%.4g'), ('line_1e16_e_s', '%.3f'), ('contrast_1e16', '%.2e'),
               ('sky_e_s_px', '%.2e'), ('t_frame_s', '%.2f'), ('frames_per_hour', '%.0f')]:
    FT[c].format = fmt
FT.write(os.path.join(OUT, 'filters.ecsv'), format='ascii.ecsv', overwrite=True)
print(f'host {HOST_SPT} G = {HOST_G}; planet line 1e-16 erg/s/cm^2; frame time = half well at 10 mas, S = 1')
FT.pprint(max_width=200)
g = FT['contrast_1e16'][0] / FT['contrast_1e16'][2]
print(f'\ncheck: contrast gain 2 nm / 20 nm = {g:.2f}  vs  EW ratio {FT["eqw_nm"][2] / FT["eqw_nm"][0]:.2f}')
assert abs(g / (FT['eqw_nm'][2] / FT['eqw_nm'][0]) - 1) < 0.1
""")

md(r"""
## 2. The host star's PSF at 656 nm

Radial profile of the PDS 70 host in the 2 nm filter, in e-/s per detector pixel, for the baseline PSF, the
requirement-corner Strehl, and the FRED scattered-light halo. Two things to notice:

* At 656 nm the Airy core is 45 mas FWHM (2.7 px). PDS 70 b at 180 mas sits 4 Airy FWHM out, where the
  Airy wing is ~$10^{-4}$ of the total flux per pixel: bright, but photon-noise-countable.
* The FRED halo is tabulated on 0.4 mm cells (106 px) and is held flat inside 85 px (1.4"); its level there
  is $2\times10^{-10}$ per pixel, five to six orders of magnitude below the Airy wing at 0.1-0.5". Whether
  it is achromatic or scales as $\lambda^{-2}$ makes no difference in this zone, so the two are plotted but
  the rest of the notebook uses the as-is (conservative) halo. What the FRED map cannot tell us is whether
  the *near-core* scatter (micro-roughness, particulates) rises above this floor inside 1"; that is a
  stray-light question for the engineering team (see caveats).
""")

code(r"""
def radial_profile_px(psf_fine, plate_mas, dr_px=0.5):
    n = psf_fine.shape[0]; c = (n - 1) / 2
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - c, yy - c) / OVERSAMPLE            # detector px
    edges = np.arange(0, r.max(), dr_px)
    idx = np.digitize(r.ravel(), edges) - 1
    p = psf_fine.ravel() * OVERSAMPLE ** 2               # per detector-pixel fraction, sampled finely
    prof = np.bincount(idx, weights=p, minlength=edges.size) / np.maximum(np.bincount(idx, minlength=edges.size), 1)
    return (edges[:-1] + dr_px / 2) * plate_mas, prof[:-1]

inst = i2
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for j, s, col, lab in [(0.0, 1.0, DBLUE, 'no jitter, S = 1'), (REQ_JIT, 1.0, TEAL, f'{REQ_JIT:.0f} mas, S = 1 (baseline)'),
                       (REQ_JIT, S_REQ, RED, f'{REQ_JIT:.0f} mas, S = {S_REQ:.2f} (req. corner, k = {HALO_K:.0f})')]:
    r, p = radial_profile_px(inst.fine_psf(j, s), inst.plate_mas)
    ax.plot(r, p * inst.star_rate, color=col, label=lab)
rr = np.linspace(1, 600, 300)
halo = inst.halo.profile(rr / inst.plate_mas) * inst.star_rate
ax.plot(rr, halo, color='0.3', ls='--', label='FRED scatter halo, as-is (450 nm)')
ax.plot(rr, halo * (450 / 656.3) ** 2, color='0.3', ls=':', label=r'FRED halo $\times(450/656)^2$')
for name, pl in PLANETS.items():
    ax.axvline(pl['sep_mas'], color=AMBER, lw=1)
    ax.text(pl['sep_mas'] + 4, 3e-5, f'PDS 70 {name}', color=AMBER, fontsize=8, rotation=90, va='bottom')
ax.axhline(inst.sky_rate + inst.dark_rate, color=GREEN, ls='-.', lw=1, label='sky + dark per pixel')
ax.set_yscale('log'); ax.set_ylim(1e-7, 3e3); ax.set_xlim(0, 600)
ax.set_xlabel('separation [mas]'); ax.set_ylabel('host star light [e$^-$ s$^{-1}$ px$^{-1}$]')
ax.set_title(f'{HOST_SPT} G = {HOST_G} host in {FLAB["zwo:halpha2"]}: azimuthally averaged profile')
ax.legend(fontsize=8, loc='upper right'); fig.tight_layout(); savefig(fig, 'fig02_host_profile')

r, p = radial_profile_px(inst.fine_psf(REQ_JIT, 1.0), inst.plate_mas)
for name, pl in PLANETS.items():
    k = np.argmin(np.abs(r - pl['sep_mas']))
    print(f"PDS 70 {name} at {pl['sep_mas']:.0f} mas: Airy wing {p[k]:.2e} of total per px = {p[k] * inst.star_rate:.2f} e-/s/px; "
          f"FRED halo {inst.halo.profile(pl['sep_mas'] / inst.plate_mas):.1e} per px")
""")

md(r"""
## 3. Contrast curves

For each separation and filter the aperture radius (1-3 px) is re-optimised for the best photon-limited
contrast, as an observer would. The stellar light in the aperture is azimuthally averaged over 12 position
angles. Noise terms: stellar photons in the aperture, sky + dark, and read noise from every frame in the
stack (frame time = half the well on the star's peak pixel). The reference PSF is assumed noise-free
(an ideal model or a much longer reference stack); a reference of equal depth would add the stellar term
once more.
""")

code(r"""
SEPS = np.arange(50.0, 501.0, 10.0)
TIMES = [3600.0, 5 * 3600.0]
CC = {}
t0 = time.time()
for f in FILTERS:
    for T in TIMES:
        CC[f, T, 1.0] = contrast_curve(INST[f], REQ_JIT, 1.0, SEPS, T)
    CC[f, 3600.0, S_REQ] = contrast_curve(INST[f], REQ_JIT, S_REQ, SEPS, 3600.0)
print(f'{len(CC)} contrast curves in {time.time() - t0:.0f} s')

rows = []
for (f, T, s), cc in CC.items():
    for k in range(SEPS.size):
        rows.append(dict(filter=FLAB[f], total_time_s=T, strehl=s, sep_mas=SEPS[k], contrast_5sig=cc['contrast'][k],
                         raw_contrast=cc['raw'][k], r_ap_px=cc['r_ap_px'][k], t_frame_s=cc['t_frame_s'],
                         n_frames=cc['n_frames']))
Table(rows).write(os.path.join(OUT, 'contrast_curves.ecsv'), format='ascii.ecsv', overwrite=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
# filter comparison in absolute line flux: "planet/star ratio" is not comparable between bands
for f in FILTERS:
    inst = INST[f]; cc = CC[f, 3600.0, 1.0]
    per_1e16 = inst.line_rate(1e-16) / inst.star_rate
    ax.plot(SEPS, cc['contrast'] / per_1e16 * 1e-16, color=FCOL[f], label=f'{FLAB[f]}: 5σ photon limit, 1 h')
    ax.plot(SEPS, cc['raw'] / per_1e16 * 1e-16, color=FCOL[f], ls=':', lw=1.2)
ax.plot([], [], color='0.4', ls=':', label='dotted: raw (planet = star light in aperture)')
for name, pl in PLANETS.items():
    ax.errorbar(pl['sep_mas'], pl['flux'], yerr=[[pl['flux'] - pl['flux_lo']], [pl['flux_hi'] - pl['flux']]],
                fmt='o', color='k', ms=6, capsize=3)
    ax.text(pl['sep_mas'] + 6, pl['flux'] * 1.3, f'PDS 70 {name}', fontsize=8)
ax.set_yscale('log'); ax.set_xlim(40, 510); ax.set_ylim(1e-17, 3e-13)
ax.set_xlabel('separation [mas]'); ax.set_ylabel('H-alpha line flux [erg s$^{-1}$ cm$^{-2}$]')
ax.set_title(f'{HOST_SPT} G = {HOST_G} host, jitter 10 mas, S = 1'); ax.legend(fontsize=7.5, loc='upper right')

ax = axes[1]
f = 'zwo:halpha2'
ax.plot(SEPS, CC[f, 3600.0, 1.0]['contrast'], color=TEAL, label='1 h, S = 1')
ax.plot(SEPS, CC[f, 5 * 3600.0, 1.0]['contrast'], color=DBLUE, label='5 h, S = 1')
ax.plot(SEPS, CC[f, 3600.0, S_REQ]['contrast'], color=RED, ls='--', label=f'1 h, S = {S_REQ:.2f} (req. corner)')
ax.plot(SEPS, CC[f, 3600.0, 1.0]['raw'], color=TEAL, ls=':', lw=1.2, label='raw, S = 1')
ax.plot(SEPS, CC[f, 3600.0, S_REQ]['raw'], color=RED, ls=':', lw=1.2, label=f'raw, S = {S_REQ:.2f}')
for name, pl in PLANETS.items():
    c = INST[f].line_rate(pl['flux']) / INST[f].star_rate
    ax.errorbar(pl['sep_mas'], c, yerr=[[c - INST[f].line_rate(pl['flux_lo']) / INST[f].star_rate],
                                        [INST[f].line_rate(pl['flux_hi']) / INST[f].star_rate - c]],
                fmt='o', color=AMBER, ms=6, mec='k', mew=0.5, capsize=3)
    ax.text(pl['sep_mas'] + 6, c * 1.4, f'PDS 70 {name}', fontsize=8)
ax.set_yscale('log'); ax.set_xlim(40, 510); ax.set_ylim(1e-6, 3e-2)
ax.set_xlabel('separation [mas]'); ax.set_ylabel('planet / star flux ratio in band')
ax.set_title(f'{FLAB[f]}: time and Strehl; bars = literature flux range'); ax.legend(fontsize=7.5, loc='upper right')
fig.tight_layout(); savefig(fig, 'fig03_contrast_curves')
""")

code(r"""
# noise budget at the PDS 70 b separation, 1 h, baseline PSF
print('variance budget in the optimal aperture at 180 mas, 1 h, jitter 10 mas, S = 1:')
for f in FILTERS:
    inst = INST[f]; cc = CC[f, 3600.0, 1.0]
    k = np.argmin(np.abs(SEPS - 180)); r_ap = cc['r_ap_px'][k]
    sfrac, pfrac = aperture_fractions(inst, inst.fine_psf(REQ_JIT, 1.0), [180.0], r_ap)
    T = 3600.0; n_pix = np.pi * r_ap ** 2
    star = inst.star_rate * sfrac[0] * T; sky = (inst.sky_rate + inst.dark_rate) * n_pix * T
    rn = cc['n_frames'] * inst.read_noise ** 2 * n_pix; tot = star + sky + rn
    print(f"  {FLAB[f]:14s}: r_ap {r_ap:.1f} px, EE {pfrac:.2f}, frame {cc['t_frame_s']:.2f} s x {cc['n_frames']} frames; "
          f"variance: star {star / tot:.2f}, sky+dark {sky / tot:.2f}, read {rn / tot:.2f}; "
          f"5σ contrast {cc['contrast'][k]:.2e}, raw {cc['raw'][k]:.2e}")
""")

md(r"""
## 4. PDS 70 b and c

Literature H-alpha line fluxes (erg s$^{-1}$ cm$^{-2}$); the line is variable at the factor-of-a-few level,
so the range is carried, not just a nominal value:

| planet | separation | nominal | range | sources |
|---|---|---|---|---|
| b | ~180 mas | $8.1\times10^{-16}$ | $2.3\times10^{-16}$ to $1.6\times10^{-15}$ | Hashimoto et al. 2020 (MUSE re-analysis); Haffert et al. 2019 range; Zhou et al. 2021 HST mean |
| c | ~220 mas | $3.1\times10^{-16}$ | $1.9$ to $4.8\times10^{-16}$ | Hashimoto et al. 2020; Haffert et al. 2019 range |

Host: K7, Gaia $G \approx 11.7$, 112 pc; a weak-line T Tauri star whose own H-alpha emission is small
(`HOST_HA_EW_A` above; set it to include the host line in the narrow bands). Below: SNR of each planet
versus total integration for each filter, at the photon limit with ideal PSF subtraction.
""")

code(r"""
T_GRID = np.logspace(np.log10(60), np.log10(10 * 3600), 25)
rows = []
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, (name, pl) in zip(axes, PLANETS.items()):
    for f in FILTERS:
        inst = INST[f]
        for key, ls in [('flux', '-'), ('flux_lo', ':'), ('flux_hi', ':')]:
            snr = np.array([contrast_curve(inst, REQ_JIT, 1.0, [pl['sep_mas']], T, planet_rate=inst.line_rate(pl[key]))['snr'][0]
                            for T in T_GRID])
            ax.plot(T_GRID / 3600, snr, ls, color=FCOL[f], lw=2 if ls == '-' else 1, label=FLAB[f] if ls == '-' else None)
            if key == 'flux':
                for T in [600.0, 3600.0, 5 * 3600.0]:
                    cc = contrast_curve(inst, REQ_JIT, 1.0, [pl['sep_mas']], T, planet_rate=inst.line_rate(pl['flux']))
                    ccS = contrast_curve(inst, REQ_JIT, S_REQ, [pl['sep_mas']], T, planet_rate=inst.line_rate(pl['flux']))
                    rows.append(dict(planet=name, filter=FLAB[f], total_time_s=T, snr_S1=cc['snr'][0], snr_Sreq=ccS['snr'][0],
                                     raw_ratio_S1=(inst.line_rate(pl['flux']) / inst.star_rate) / cc['raw'][0],
                                     r_ap_px=cc['r_ap_px'][0], n_frames=cc['n_frames']))
    ax.axhline(5, color='0.4', ls='--', lw=1); ax.text(0.02, 5.5, '5σ', color='0.4', fontsize=8)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('total integration [h]')
    ax.set_title(f"PDS 70 {name}: {pl['sep_mas']:.0f} mas, F = {pl['flux']:.1e} (dotted: {pl['flux_lo']:.1e}, {pl['flux_hi']:.1e})",
                 fontsize=10)
axes[0].set_ylabel('SNR (photon limit, ideal PSF subtraction)'); axes[0].legend(fontsize=8, loc='lower right')
fig.tight_layout(); savefig(fig, 'fig04_pds70_snr_vs_time')
PDS = Table(rows)
for c in ['snr_S1', 'snr_Sreq', 'raw_ratio_S1']: PDS[c].format = '%.1f'
PDS.write(os.path.join(OUT, 'pds70_snr.ecsv'), format='ascii.ecsv', overwrite=True)
print('raw_ratio = planet flux / stellar light in the same aperture (>1: visible without subtraction)')
PDS.pprint(max_lines=40, max_width=200)
""")

md(r"""
## 5. Generic sensitivity: what line flux is detectable, for which hosts

Young stars in the nearby star-forming regions (Taurus, Lupus, Upper Sco, 100-160 pc) span $G \approx 9$-13.
The 5σ line-flux limit in 1 h (photon limit, ideal subtraction, baseline PSF) as a function of separation,
for three host brightnesses and the three filters. Brighter hosts are worse in two ways: more stellar light
in the aperture and shorter frames, hence more read noise.
""")

code(r"""
HOST_MAGS = [9.0, 11.0, 13.0]
SEPS_G = np.arange(50.0, 501.0, 25.0)
t0 = time.time()
rows = []
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for ax, f in zip(axes, FILTERS):
    for m, col in zip(HOST_MAGS, [RED, AMBER, TEAL]):
        inst = Instrument(f, HOST_SPT, m)
        per_1e16 = inst.line_rate(1e-16) / inst.star_rate
        cc = contrast_curve(inst, REQ_JIT, 1.0, SEPS_G, 3600.0)
        flim = cc['contrast'] / per_1e16 * 1e-16
        ax.plot(SEPS_G, flim, color=col, label=f'host G = {m:.0f} (frame {cc["t_frame_s"]:.1f} s)')
        for s, fl, rap in zip(SEPS_G, flim, cc['r_ap_px']):
            rows.append(dict(filter=FLAB[f], host_G=m, sep_mas=s, flux_lim_5sig_1h=fl, r_ap_px=rap, t_frame_s=cc['t_frame_s']))
    for name, pl in PLANETS.items():
        ax.plot(pl['sep_mas'], pl['flux'], 'o', color='k', ms=5)
        ax.text(pl['sep_mas'] + 6, pl['flux'] * 1.2, f'PDS 70 {name}', fontsize=8)
    ax.set_yscale('log'); ax.set_xlim(40, 510); ax.set_xlabel('separation [mas]'); ax.set_title(FLAB[f])
    ax.legend(fontsize=8, loc='upper right')
axes[0].set_ylabel('5σ line-flux limit in 1 h [erg s$^{-1}$ cm$^{-2}$]')
fig.suptitle(f'{HOST_SPT} host, jitter 10 mas, S = 1, photon limit with ideal PSF subtraction', y=1.0)
fig.tight_layout(); savefig(fig, 'fig05_fluxlimit_grid')
GRID = Table(rows); GRID['flux_lim_5sig_1h'].format = '%.2e'
GRID.write(os.path.join(OUT, 'fluxlimit_grid.ecsv'), format='ascii.ecsv', overwrite=True)
print(f'grid in {time.time() - t0:.0f} s')
GRID[np.isin(GRID['sep_mas'], [100.0, 200.0, 400.0])].pprint(max_lines=40)
""")

md(r"""
## 6. Summary and caveats

**Filter.** The 2 nm filter is the right one: the planet is a line, so its signal is the same in every filter,
while the stellar continuum drops with bandwidth. Two regimes:
* *Photon-limited* (ideal subtraction): the noise is the square root of the stellar light, and the frame time
  shrinks with the star's brightness so the read-noise term scales the same way. The line-flux limit therefore
  improves only as $\sqrt{\rm bandwidth}$: 2 nm beats 20 nm by 3.1x and 6 nm by 1.7x.
* *Systematics-limited* (PSF-subtraction residuals, which scale with the stellar light itself): the gain is the
  full bandwidth ratio, ~10x over 20 nm, ~3x over 6 nm. This is the regime a real observation will be in
  (next notebook), so the narrow filter's advantage is the large one.
Peak throughput is nearly identical for the three filters in the ETC's EOL curves, so nothing is lost in the
core.

**PDS 70.** At the photon limit with an ideal PSF subtraction, both planets are detectable in the 2 nm filter
within an hour (Table above): the Airy wing at 180-220 mas is ~$10^{-4}$ of the stellar flux per pixel, which
is bright but countable. Planet b even approaches the raw (no-subtraction) threshold in the 2 nm band. The
real limit will therefore be **PSF-subtraction systematics**, not photons; that is what
`halpha_hci_jitter_strehl.ipynb` quantifies.

**What this does not include.**
* Any PSF-subtraction residual (jitter or Strehl changes between science and reference, polarisation,
  colour mismatch between host and reference star); noise from the reference itself.
* The host's own H-alpha emission: an accreting classical T Tauri host with EW of tens of Å would more
  than double the stellar light in the 2 nm band. PDS 70 is a weak-line T Tauri star; `HOST_HA_EW_A` is a
  knob for other targets.
* Near-core scattered light: the FRED map is flat inside 1.4" by construction and puts the halo five orders
  of magnitude below the Airy wing here, but it has no information on that scale. Micro-roughness scattering
  at 0.1-0.5" is the one instrument property this analysis cannot bound.
* The PSF is monochromatic at the effective wavelength (fine for a 2 nm band), unobscured Airy (the ETC's
  model), and static; the planet is unresolved with a 2 Å line width.
* Saturation is treated with a half-well rule on the peak pixel; no bleeding or persistence. Frame times below
  a few tenths of a second (bright hosts, wide filters) would exceed the IMX455 full-frame readout rate and need
  a sub-array readout; the duty-cycle loss is not modelled.
""")

nb = new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
with open(OUT, "w") as f:
    nbf.write(nb, f)
print("wrote", len(cells), "cells to", OUT)
