"""Command-line interface: wcc-sim --ra ... --dec ... -o out.fits"""

import argparse

from .pipeline import simulate_field


def build_parser():
    p = argparse.ArgumentParser(
        prog="wcc-sim",
        description="Simulate a WCC detector image of the Gaia field at (RA, Dec).",
    )
    p.add_argument("--ra", type=float, required=True, help="pointing RA [deg, ICRS]")
    p.add_argument("--dec", type=float, required=True, help="pointing Dec [deg, ICRS]")
    p.add_argument("--sensorfilter", default="zwo:r",
                   help="wcc_etc sensorfilter, e.g. zwo:r, zwo:r+1, qcmos:bb")
    p.add_argument("--focus", type=int, choices=[0, 1, 2], default=None,
                   help="waves of defocus (default: sensorfilter's focus level)")
    p.add_argument("--exptime", type=float, default=90.0, help="total exposure [s]")
    p.add_argument("--n-reads", type=int, default=1, help="coadded frames")
    p.add_argument("--pa", type=float, default=0.0, help="position angle [deg E of N]")
    p.add_argument("--jitter", type=float, default=None,
                   dest="jitter_sigma_mas", help="jitter sigma [mas]")
    p.add_argument("--mag-limit", type=float, default=21.0, help="Gaia G faint limit")
    p.add_argument("--no-noise", action="store_true", help="skip noise realization")
    p.add_argument("--no-clean", action="store_true", help="omit CLEAN extension")
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument("--shape", type=int, nargs=2, metavar=("NY", "NX"), default=None,
                   help="subarray shape (default: full array)")
    p.add_argument("--stamp-npix", type=int, default=None,
                   help="PSF stamp size [detector px, odd]")
    p.add_argument("--no-wings", action="store_true",
                   help="skip the analytic PSF wing extension beyond the stamp")
    p.add_argument("--cache-dir", default=None, help="Gaia query cache directory")
    p.add_argument("-o", "--output", required=True, help="output FITS path")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    field = simulate_field(
        ra=args.ra,
        dec=args.dec,
        sensorfilter=args.sensorfilter,
        focus=args.focus,
        exptime=args.exptime,
        n_reads=args.n_reads,
        pa=args.pa,
        jitter_sigma_mas=args.jitter_sigma_mas,
        mag_limit=args.mag_limit,
        add_noise=not args.no_noise,
        seed=args.seed,
        output=args.output,
        shape=tuple(args.shape) if args.shape else None,
        stamp_npix=args.stamp_npix,
        wings=not args.no_wings,
        cache_dir=args.cache_dir,
        write_clean=not args.no_clean,
    )
    n_sat = int(field.saturation_mask.sum())
    print(
        f"Wrote {args.output}: {field.image_adu.shape[1]}x{field.image_adu.shape[0]} px, "
        f"{field.params['n_sources']} sources, focus={field.params['focus']}w, "
        f"{n_sat} saturated px"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
