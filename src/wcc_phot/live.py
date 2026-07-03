"""Live matplotlib view of the photometry as each frame is analyzed.

Pass a LiveViewer as `run_photometry(..., on_frame=viewer)` (or use the
CLI `--live` flag): the left panel shows the current frame with the
target (red) and reference (teal) apertures at their re-centroided
positions, the right panel grows the relative light curve point by
point. Any callable accepting the same event dict works as `on_frame`,
so custom displays (GUIs, DS9, web) can hook in the same way.
"""

import warnings

import numpy as np

TARGET_COLOR = "#d1495b"
REF_COLOR = "#00798c"


class LiveViewer:
    """Interactive two-panel display updated once per analyzed frame."""

    def __init__(self, pause=0.2, percent=99.5, cmap="gray_r", zoom=None):
        """`zoom` (px, optional) crops the image panel around the target."""
        self.pause = pause
        self.percent = percent
        self.cmap = cmap
        self.zoom = zoom
        self._fig = None

    # -- pipeline callback ------------------------------------------------
    def __call__(self, event):
        import matplotlib.pyplot as plt

        if self._fig is None:
            self._setup(event)
        self._update(event)
        self._fig.canvas.draw_idle()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # non-GUI backend chatter
            plt.pause(self.pause)

    def hold(self):
        """Keep the window open after the run (no-op without a figure)."""
        import matplotlib.pyplot as plt

        if self._fig is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.ioff()
                plt.show()

    def close(self):
        import matplotlib.pyplot as plt

        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None

    # -- internals ---------------------------------------------------------
    def _setup(self, event):
        import matplotlib.pyplot as plt
        from astropy.visualization import simple_norm

        plt.ion()
        self._fig, (self._ax_img, self._ax_lc) = plt.subplots(
            1, 2, figsize=(12, 5.5), width_ratios=[1.15, 1]
        )
        self._norm = simple_norm(event["image_e"], "asinh", percent=self.percent)
        self._im = self._ax_img.imshow(
            event["image_e"], origin="lower", cmap=self.cmap, norm=self._norm
        )
        if self.zoom is not None:
            x0, y0 = event["x"][0], event["y"][0]
            self._ax_img.set_xlim(x0 - self.zoom, x0 + self.zoom)
            self._ax_img.set_ylim(y0 - self.zoom, y0 + self.zoom)
        self._ax_img.set_xlabel("x [px]")
        self._ax_img.set_ylabel("y [px]")

        from matplotlib.patches import Circle

        geom = event["geom"]
        self._circles = []
        for j, role in enumerate(event["roles"]):
            color = TARGET_COLOR if role == "target" else REF_COLOR
            circle = Circle(
                (event["x"][j], event["y"][j]),
                geom.r_ap,
                fill=False,
                color=color,
                lw=1.4,
            )
            self._ax_img.add_patch(circle)
            self._circles.append(circle)
        self._annuli = []
        for radius in (geom.r_in, geom.r_out):
            ring = Circle(
                (event["x"][0], event["y"][0]),
                radius,
                fill=False,
                color=TARGET_COLOR,
                lw=0.8,
                ls="--",
                alpha=0.7,
            )
            self._ax_img.add_patch(ring)
            self._annuli.append(ring)

        self._times, self._rel, self._rel_err = [], [], []
        self._ax_lc.set_xlabel("time")
        self._ax_lc.set_ylabel("target / ref ensemble")
        self._fig.tight_layout()

    def _update(self, event):
        self._im.set_data(event["image_e"])
        for circle, x, y in zip(self._circles, event["x"], event["y"]):
            circle.center = (x, y)
        for ring in self._annuli:
            ring.center = (event["x"][0], event["y"][0])
        self._ax_img.set_title(
            f"frame {event['frame'] + 1}/{event['n_frames']}"
        )

        self._times.append(event["time"])
        self._rel.append(event["rel_flux"])
        self._rel_err.append(event["rel_flux_err"])
        self._ax_lc.clear()
        self._ax_lc.errorbar(
            self._times,
            self._rel,
            yerr=self._rel_err,
            fmt="o",
            color=TARGET_COLOR,
            ecolor=REF_COLOR,
            capsize=2,
        )
        median = float(np.median(self._rel))
        self._ax_lc.axhline(median, color=REF_COLOR, lw=0.8, ls=":")
        self._ax_lc.set_xlabel("time")
        self._ax_lc.set_ylabel("target / ref ensemble")
        self._ax_lc.set_title(
            f"rel flux = {event['rel_flux']:.5f} "
            f"± {event['rel_flux_err']:.5f}"
        )
