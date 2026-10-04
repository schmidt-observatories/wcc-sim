#!/usr/bin/env python
"""Rebuild the packaged scattered-light radial profile from the FRED map.

The FRED ``.fgd`` (24 MB) ships with wcc_etc, and the combined core+scatter
PSF FITS (75 MB) is a diagnostic product -- neither belongs in this repo, and
neither is needed at runtime. This script reduces the map to the ~30 kB
azimuthal profile that ``wcc_sim.scatter`` actually loads.

    python scripts/build_scatter_profile.py [--fgd PATH] [--n-bins 300]

Run it again if the stray-light model is re-run in FRED.
"""

import argparse

from wcc_sim.scatter import PROFILE_PATH, build_scatter_table


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fgd", default=None,
                   help="FRED .fgd map (default: the one wcc_etc ships)")
    p.add_argument("--n-bins", type=int, default=300,
                   help="log-spaced radial bins (default: 300)")
    p.add_argument("-o", "--output", default=PROFILE_PATH,
                   help="output ECSV path")
    args = p.parse_args(argv)

    table = build_scatter_table(fgd_path=args.fgd, n_bins=args.n_bins)
    table.write(args.output, format="ascii.ecsv", overwrite=True)

    m = table.meta
    print(f"source        : {m['SCATFILE']}")
    print(f"integrated P  : {m['P_FULL']:.6e} W per W in  ({m['WAVELEN']:.0f} nm)")
    print(f"peak irrad.   : {m['IRRPEAK']:.6e} W/mm^2 per W")
    print(f"radial reach  : {m['RMAXMM']:.1f} mm = {m['RMAXMM'] / 0.00376:,.0f} "
          f"IMX455 px  ({m['NBINS']} bins)")
    print(f"disc fraction : {m['FDISC']:.4f} exact 2D, "
          f"{m['FDISCPRF']:.4f} from the profile "
          f"({abs(m['FDISCPRF'] / m['FDISC'] - 1) * 100:.3f}% apart)")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
