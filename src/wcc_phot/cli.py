"""Command-line interface: wcc-phot frame1.fits frame2.fits ... -o phot.fits"""

import argparse

import numpy as np

from .pipeline import run_photometry


def build_parser():
    p = argparse.ArgumentParser(
        prog="wcc-phot",
        description=(
            "Differential aperture/PSF photometry of a target and its best "
            "reference stars across a series of wcc-sim FITS frames."
        ),
    )
    p.add_argument("frames", nargs="+", help="wcc-sim FITS frames (same field)")
    tgt = p.add_argument_group("target (source id or coordinates)")
    tgt.add_argument("--source-id", type=int, default=None,
                     help="Gaia source_id of the target")
    tgt.add_argument("--ra", type=float, default=None, help="target RA [deg]")
    tgt.add_argument("--dec", type=float, default=None, help="target Dec [deg]")
    p.add_argument("--method", choices=["aperture", "psf"], default="aperture")
    p.add_argument("--n-ref", type=int, default=10,
                   help="number of reference stars")
    p.add_argument("--r-ap", type=float, default=None,
                   help="aperture radius [px] (default: model-PSF EE radius)")
    p.add_argument("--r-in", type=float, default=None,
                   help="annulus inner radius [px] (default: 1.5 r_ap)")
    p.add_argument("--r-out", type=float, default=None,
                   help="annulus outer radius [px] (default: 2.5 r_ap)")
    p.add_argument("--centroid-box", type=int, default=None,
                   help="centroid box side [px, odd]")
    p.add_argument("--fit-shape", type=int, default=None,
                   help="PSF fit region side [px, odd]")
    p.add_argument("--ee", type=float, default=0.95,
                   help="encircled-energy fraction for the default r_ap")
    p.add_argument("--iso-dmag", type=float, default=1.0,
                   help="reject refs with a neighbor brighter than G+iso_dmag")
    p.add_argument("--lc-csv", default=None,
                   help="also write the light curve as ECSV")
    p.add_argument("--report", default=None, metavar="PATH",
                   help="write a one-page PDF + PNG diagnostic report "
                        "(field + apertures, light curve, centroids, "
                        "noise metrics)")
    p.add_argument("--live", action="store_true",
                   help="live matplotlib window: frame + apertures + light "
                        "curve updated as each image is analyzed")
    p.add_argument("--live-pause", type=float, default=0.2,
                   help="seconds to pause on each frame in live mode")
    p.add_argument("--live-zoom", type=int, default=None,
                   help="crop the live image panel to +-N px around the target")
    p.add_argument("-o", "--output", required=True,
                   help="output FITS path (STARS/PHOT/LC)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.source_id is not None:
        target = args.source_id
    elif args.ra is not None and args.dec is not None:
        target = (args.ra, args.dec)
    else:
        build_parser().error("give --source-id or both --ra and --dec")

    viewer = None
    if args.live:
        from .live import LiveViewer

        viewer = LiveViewer(pause=args.live_pause, zoom=args.live_zoom)

    result = run_photometry(
        args.frames,
        target,
        method=args.method,
        n_ref=args.n_ref,
        r_ap=args.r_ap,
        r_in=args.r_in,
        r_out=args.r_out,
        centroid_box=args.centroid_box,
        fit_shape=args.fit_shape,
        ee=args.ee,
        iso_dmag=args.iso_dmag,
        output=args.output,
        on_frame=viewer,
        report=args.report,
    )
    if args.lc_csv is not None:
        result.lightcurve.write(args.lc_csv, format="ascii.ecsv", overwrite=True)

    lc = result.lightcurve
    rms_ppm = 1e6 * float(np.std(lc["rel_flux_norm"]))
    err_ppm = 1e6 * float(np.median(lc["rel_flux_norm_err"]))
    print(
        f"Wrote {args.output}: {result.params['n_frames']} frames, "
        f"target {result.params['target_source_id']} "
        f"(G={result.stars['gmag'][0]:.2f}), "
        f"{result.params['n_ref']} refs, {args.method} r_ap="
        f"{result.params['r_ap']:.1f} px; rel-flux rms {rms_ppm:.0f} ppm "
        f"(median error {err_ppm:.0f} ppm)"
    )
    if viewer is not None:
        viewer.hold()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
