PSFs and defocus
================

The three focus modes
---------------------

The WCC observes both in focus and deliberately defocused (to average over
pixel-level systematics for bright stars). ``wcc-sim`` supports three modes,
selected with ``focus=``:

.. list-table::
   :header-rows: 1
   :widths: 12 30 58

   * - ``focus``
     - PSF
     - Source
   * - 0
     - In-focus Airy pattern
     - Analytic (``wcc_etc.AiryPSF``)
   * - 1
     - +1 wave of defocus
     - Zemax Huygens PSF data bundled with ``wcc_etc``
   * - 2
     - +2 waves of defocus
     - Zemax Huygens PSF data bundled with ``wcc_etc``

If ``focus=None`` (the default), the focus level implied by the sensorfilter
key is used — e.g. ``zwo:r+1`` defaults to 1 wave, ``zwo:r`` to in-focus.

The defocused PSFs are *annular*: the core is centrally depressed and most of
the energy lands in a ring (radius ~29 px for 2 waves on the IMX455). This
has consequences for saturation flagging (see :doc:`noise`).

Oversampled rendering
---------------------

Stars land at arbitrary sub-pixel positions, so the PSF is rendered once on a
fine grid and shifted per star:

1. :func:`~wcc_sim.psf.render_oversampled_psf` asks ``wcc_etc`` to render the
   PSF on a grid **oversampled 11×** relative to the detector (by describing
   a detector with 11× smaller pixels to the PSF context). Jitter blur is
   applied by ``wcc_etc`` at the fine plate scale.
2. :func:`~wcc_sim.render.add_star` shifts the fine-grid stamp by the star's
   sub-pixel offset (an ``np.roll``, exact to 1/22 px) and bins 11×11 fine
   pixels into each detector pixel.

Stamp sizes
-----------

Stamps are square with an odd number of detector pixels, set by
:data:`wcc_sim.psf.DEFAULT_STAMP`:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - ``focus``
     - Stamp [px]
     - Rationale
   * - 0
     - 129
     - Holds ≥ 99.5 % of the jitter-blurred Airy energy (65 px would
       truncate ~1.3 % in the wings).
   * - 1, 2
     - 257
     - The Zemax Huygens defocus data spans ~272 IMX455 px.

Override with ``stamp_npix=`` (odd) to trade wing fidelity for speed, and
``oversample=`` to change the sub-pixel resolution.

Jitter
------

Pointing jitter is modeled as a Gaussian blur of the PSF. By default the
telescope's configured jitter (from the ``wcc_etc`` telescope model) is used;
override per-simulation with ``jitter_sigma_mas=``. The value actually used
is recorded in ``field.params["jitter_sigma_mas"]`` and the ``JITTER`` header
card.

.. code-block:: python

   from wcc_sim.detectors import make_base_simulation
   from wcc_sim.psf import render_oversampled_psf

   sim = make_base_simulation("zwo:r")
   psf0 = render_oversampled_psf(sim, focus=0)                      # Airy
   psf2 = render_oversampled_psf(sim, focus=2, jitter_sigma_mas=50) # defocus + jitter

Both return a normalized 2-D array on the fine grid (sum = 1), ready for
:func:`~wcc_sim.render.render_scene`.
