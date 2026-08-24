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
