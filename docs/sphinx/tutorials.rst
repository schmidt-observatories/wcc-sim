Tutorials
=========

These tutorials are runnable Jupyter notebooks from the ``notebooks/``
directory of the repository, rendered here with their saved outputs. Each one
is self-contained — download it and run it interactively to experiment. The
Gaia queries they perform are cached in ``notebooks/gaia_cache/``, so they
run offline.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Notebook
     - What it covers
   * - :doc:`notebooks/01_wcc_sim_example`
     - The full end-to-end example with verification: simulate a field,
       overlay the Gaia catalog, compare the three PSF modes, and check
       astrometry, photometric closure, noise statistics, and saturation.
   * - :doc:`notebooks/02_psf_focus_modes`
     - The three PSF modes (in-focus Airy, +1/+2-wave defocus), stamp
       sizes, oversampling, and the effect of jitter.
   * - :doc:`notebooks/03_detectors_and_filters`
     - The two detectors (IMX455 vs qCMOS), sensorfilter keys, and how
       Gaia photometry maps to spectral types and count rates.
   * - :doc:`notebooks/04_noise_and_saturation`
     - The ETC noise model step by step, ``n_reads`` semantics, per-pixel
       saturation masks, and the windowed per-star ``saturated`` flag.
   * - :doc:`notebooks/05_astrometry_and_catalogs`
     - The TAN WCS, position angles, injecting your own catalog, and
       astrometric round-trip checks.
   * - :doc:`notebooks/06_cli_and_fits`
     - The ``wcc-sim`` command line and the multi-extension FITS output,
       read back with astropy.
   * - :doc:`notebooks/07_psf_wings`
     - The analytic PSF wing extension: before/after images at hard
       stretch, the fitted power-law wing models, seam smoothness at the
       old stamp edge, and how the wings change the encircled energy.
   * - :doc:`notebooks/08_wcc_phot_photometry`
     - The ``wcc_phot`` differential photometry pipeline: target +
       reference selection, per-frame centroiding, aperture vs PSF
       fluxes, the relative light curve, and the live viewer hook.

.. toctree::
   :maxdepth: 1
   :hidden:

   notebooks/01_wcc_sim_example
   notebooks/02_psf_focus_modes
   notebooks/03_detectors_and_filters
   notebooks/04_noise_and_saturation
   notebooks/05_astrometry_and_catalogs
   notebooks/06_cli_and_fits
   notebooks/07_psf_wings
   notebooks/08_wcc_phot_photometry
