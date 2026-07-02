wcc-sim: Image Simulator for the Lazuli WCC
===========================================

.. image:: _static/logo_schmidt_sciences.png
   :alt: Schmidt Sciences Astrophysics Institute
   :width: 320px

**wcc-sim** is the end-to-end photometric image simulator for the **Wide-field
Context Camera (WCC)** on *Lazuli*. Given a pointing (RA, Dec) it queries
**Gaia DR3**, converts each star's Gaia photometry to detector count rates
through the `wcc-etc <https://github.com/schmidt-observatories/wcc-etc>`_
instrument model, renders every star with the in-focus (Airy) or +1/+2-wave
defocus (Zemax Huygens) PSF, adds photon/sky/dark/read noise with full-well +
ADC saturation, and writes a multi-extension FITS image (``SCI`` + ``SATMASK``
+ ``CAT`` + ``CLEAN``) with a TAN WCS.

.. note::

   New here? Read :doc:`installation` and then work through
   :doc:`quickstart`. The :doc:`tutorials` are runnable Jupyter notebooks
   that cover every major feature end to end, and :doc:`cli` documents the
   ``wcc-sim`` command line.

Highlights
----------

- **One call, one image** — :func:`~wcc_sim.simulate_field` goes from
  (RA, Dec) to a calibratable FITS image in ~11 s for the full 9568×6380 px
  Sony IMX455 array.
- **Real sky** — Gaia DR3 cone search with an on-disk cache; each star's
  BP−RP color is mapped to a Pickles spectral type and normalized to its
  G magnitude before integration through the instrument throughput.
- **Faithful instrument model** — count rates, sky background, dark current,
  read noise, full-well and ADC saturation all come from the ``wcc_etc``
  exposure time calculator, so the simulator and the ETC agree by
  construction.
- **Three PSF modes** — in-focus Airy, +1-wave and +2-wave defocus (Zemax
  Huygens data), rendered on an oversampled grid with sub-pixel placement and
  optional Gaussian jitter blur.
- **Two detectors** — ``zwo:*`` (Sony IMX455, 9568×6380 px, 16.87 mas/px)
  and ``qcmos:*`` (Hamamatsu HWK4123, 4096×2304 px, 20.64 mas/px).
- **Analysis-ready output** — multi-extension FITS with a TAN WCS, per-pixel
  saturation mask, the injected Gaia catalog (with pixel positions, spectral
  types, rates, and saturation flags), and an optional noiseless ``CLEAN``
  image for photometric closure tests.

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
   changelog

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
