wcc-sim: Image Simulator for the Lazuli WCC
===========================================

.. image:: _static/logo_schmidt_observatory_system.png
   :alt: Schmidt Observatory System
   :width: 320px

**wcc-sim** makes synthetic images of what the **Wide-field Context Camera
(WCC)** on *Lazuli* would see. You give it a pointing on the sky. It looks up
the real stars there in Gaia DR3, works out how many electrons each one
produces in the detector, draws every star with the camera's point-spread
function, adds the noise a real exposure would have, and writes a FITS image
you can analyse like data.

The instrument numbers (throughput, detector noise, saturation) come from the
`wcc-etc <https://github.com/schmidt-observatories/wcc-etc>`_ exposure time
calculator, so an image from wcc-sim and a signal-to-noise estimate from the
ETC agree by construction. The companion package ``wcc_phot`` measures
differential photometry on a series of those images.

In more detail: the Gaia photometry of each star is converted to a detector
count rate through the wcc-etc instrument model; the star is rendered with the
in-focus (Airy) or +1/+2-wave defocus (Zemax Huygens) PSF; photon, sky, dark
and read noise are added with full-well and ADC saturation; and the result is
written as a multi-extension FITS file (``SCI``, ``SATMASK``, ``CAT``,
``CLEAN``) with a TAN WCS.

.. note::

   New here? Read :doc:`installation` and then work through
   :doc:`quickstart`. The :doc:`tutorials` are runnable Jupyter notebooks
   that cover every major feature end to end, and :doc:`cli` documents the
   ``wcc-sim`` command line. Terms such as ADU, full well, ``n_reads`` or
   TAN WCS are defined in the :doc:`glossary`.

Highlights
----------

- **One call, one image.** :func:`~wcc_sim.simulate_field` goes from
  (RA, Dec) to a calibratable FITS image in ~11 s for the full 9568×6380 px
  Sony IMX455 array.
- **Real sky.** Gaia DR3 cone search with an on-disk cache; each star's
  BP−RP color is mapped to a Pickles spectral type and normalized to its
  G magnitude before integration through the instrument throughput.
- **Faithful instrument model.** Count rates, sky background, dark current,
  read noise, full-well and ADC saturation all come from the ``wcc_etc``
  exposure time calculator, so the simulator and the ETC agree by
  construction.
- **Three PSF modes.** In-focus Airy, +1-wave and +2-wave defocus (Zemax
  Huygens data), rendered on an oversampled grid with sub-pixel placement and
  optional Gaussian jitter blur.
- **Two detectors.** ``zwo:*`` (Sony IMX455, 9568×6380 px, 16.87 mas/px)
  and ``qcmos:*`` (Hamamatsu HWK4123, 4096×2304 px, 20.64 mas/px).
- **Analysis-ready output.** Multi-extension FITS with a TAN WCS, per-pixel
  saturation mask, the injected Gaia catalog (with pixel positions, spectral
  types, rates, and saturation flags), and an optional noiseless ``CLEAN``
  image for photometric closure tests.
- **Plots built in.** ``field.plot()``, ``field.plot_cutout()``,
  ``field.plot_psf_profile()`` and the ``result.plot_*`` methods of
  ``wcc_phot`` draw the standard views with one call, each taking ``ax=``
  (see :doc:`user_guide/plotting`).

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials

.. toctree::
   :maxdepth: 2
   :caption: Reference

   cli
   api/index
   glossary
   changelog

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
