#!/usr/bin/env python
"""Regenerate data/spt_synthetic_colors.csv -- synthetic Johnson/Gaia colours.

Integrates every Pickles dwarf template in TEMPLATES through Johnson B,
Johnson V and the vendored Gaia DR3 G, BP and RP passbands, in vegamag. Run
this when the template list or a passband changes; it needs network for the
Johnson curves (synphot's remote filter set) and the Vega spectrum, which is
why the result is vendored instead of computed at run time (~8 s per
process).

    python scripts/build_spt_colors.py
"""

import os

from synphot import Observation, SpectralElement
from wcc_etc.scene import get_scene

from wcc_sim.starflux import DATA_DIR, gaia_bandpass

#: The wcc_etc Pickles dwarf templates the colour lookup chooses from.
TEMPLATES = (
    "O5V", "O9V", "B0V", "B1V", "B3V", "B5-7V", "B8V",
    "A0V", "A2V", "A3V", "A5V", "F0V", "F2V", "F5V", "F8V",
    "G0V", "G2V", "G5V", "G8V", "K0V", "K2V", "K5V", "K7V",
    "M0V", "M2V", "M4V", "M5V",
)


def main():
    B = SpectralElement.from_filter("johnson_b")
    V = SpectralElement.from_filter("johnson_v")
    G, BP, RP = gaia_bandpass("g"), gaia_bandpass("bp"), gaia_bandpass("rp")

    def vegamag(spec, band):
        # force="taper": the Pickles templates are narrower than Johnson B at
        # the blue end; tapering to zero is the right boundary condition for a
        # colour, and synphot refuses to extrapolate silently.
        return Observation(spec, band, force="taper").effstim("vegamag").value

    lines = ["spt,b_v,g_v,bp_rp"]
    for spt in TEMPLATES:
        spec = get_scene(spt, mag=0.0).source.get_spectrum()
        b, v, g = vegamag(spec, B), vegamag(spec, V), vegamag(spec, G)
        bp_rp = vegamag(spec, BP) - vegamag(spec, RP)
        lines.append(f"{spt},{b - v:.4f},{g - v:.4f},{bp_rp:.4f}")

    path = os.path.join(DATA_DIR, "spt_synthetic_colors.csv")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path} ({len(lines) - 1} templates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
