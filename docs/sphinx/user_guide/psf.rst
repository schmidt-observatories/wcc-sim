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
key is used: ``zwo:r+1`` defaults to 1 wave, ``zwo:r`` to in-focus.

The defocused PSFs are *annular*: the core is centrally depressed and most of
the energy lands in a ring (radius ~29 px for 2 waves on the IMX455). This
has consequences for saturation flagging (see :doc:`noise`). "N waves" is a
Fringe Zernike Z4 coefficient of N waves (2N waves peak to valley), the
convention of the Zemax runs. The Huygens images are recentred on their flux
centroid before use, so the catalog position is the quantity a centroider
recovers (the ring is slightly asymmetric, by 0.1 to 0.2 px).

Which wavelength
----------------

In focus, the Airy core of each star is built at the effective wavelength of
its spectral template through the band, the same choice ``wcc_etc`` makes
for that star, rather than at the filter pivot. The pivot is ~130 nm too
blue for an M dwarf in the broad band, which would give a 23 % too high peak
pixel. The per-star value is recorded in the ``CAT`` column
``psf_wavelength_nm``; the field-level PSF (wing fit, extended-source
kernels, the PSF report) uses the pivot, recorded as ``PSFWAVE``.

``wavelength_nm=`` pins every star to one wavelength instead. It changes
the PSF geometry only: the synthetic photometry still integrates the full
bandpass, so count rates do not move. Use it to put the core at the 450 nm
of the stray-light ray trace, for instance. Defocus (``focus=1|2``) ignores
it, since the Huygens images carry no wavelength model.

``chromatic=True`` goes one step further and replaces each template's
single-wavelength Airy core with a spectrum-weighted effective PSF (seven
nodes across the band, reddened per component for extended sources). It is
a documented no-op for ``focus != 0``.

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

Beyond the stamp
----------------

A stamp is finite, and at a hard stretch its square edge sits ~100 sigma
above the sky around a bright star. By default (``wings=True``) each star's
PSF is continued beyond the stamp with an analytic power-law wing,
``profile(r) = c r^alpha``, fitted to the outer annulus of the stamp itself
(:func:`~wcc_sim.wings.fit_wing_model`; alpha = -3 in focus, the Airy
envelope, steeper for the defocused PSFs). The stamp's outer 20 % is
crossfaded into the model and the wing is drawn on a circular footprint out
to the radius where ``flux x profile(r)`` falls to ``wing_floor_sigma``
(default 0.1) of the per-pixel background noise. Every star is renormalised
by the same factor so photometry stays magnitude independent; the fitted
parameters and the floor are recorded as ``WINGA``, ``WINGC`` and
``WINGFLR``. The measured scattered-light halo is a second additive term on
top of this wing; see :doc:`scatter`. ``wings=False`` turns both off.

The 0.1 sigma floor hides the truncation in any single pixel, but it is a
systematic cut: a statistic averaging N pixels sees it at sqrt(N) times that
level, so a scattered-light background measured over ~1e5 pixels needs
``wing_floor_sigma`` of 0.01 or less (and a longer render).

Looking at it
-------------

``field.plot_psf_profile()`` draws the azimuthally averaged profile of a
star split into its Airy core, scattered-light halo and total against what
landed in the noiseless image; ``field.psf_report(path)`` (CLI
``--report``) writes the one-page report built around it; and
:func:`wcc_sim.psfreport.psf_decomposition` returns the numbers. See
:doc:`plotting` and the tutorials :doc:`../notebooks/03_psf_focus_modes` and
:doc:`../notebooks/08_psf_wings`.
