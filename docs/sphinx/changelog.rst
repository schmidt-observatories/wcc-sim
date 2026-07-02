Changelog
=========

0.1.0 (2026-07-02)
------------------

First release.

- End-to-end pipeline :func:`~wcc_sim.simulate_field`: Gaia DR3 query →
  ``wcc_etc`` count rates → oversampled PSF rendering (in-focus Airy,
  +1/+2-wave Zemax defocus) → ETC noise model with full-well + ADC
  saturation → multi-extension FITS (``SCI``/``SATMASK``/``CAT``/``CLEAN``)
  with a TAN WCS.
- ``wcc-sim`` command-line interface.
- Sony IMX455 (``zwo:*``) and Hamamatsu HWK4123 qCMOS (``qcmos:*``)
  detector geometries.
- Gaia query cache, per-star saturation flags (windowed for the defocus
  ring), reproducible seeding, and a network-free test suite.
- Sphinx documentation with tutorial notebooks.
