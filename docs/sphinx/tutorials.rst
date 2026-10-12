Tutorials
=========

These tutorials are runnable Jupyter notebooks from the ``notebooks/``
directory of the repository, rendered here with their saved outputs. Each one
is self-contained. Download it and run it interactively to experiment. The
Gaia queries they perform are cached in ``notebooks/gaia_cache/``, so they
run offline.

Start with the first two: notebook 01 simulates a field and shows the standard
views in a few minutes, and notebook 02 checks that the images behave like
data. The rest each cover one topic.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Notebook
     - What it covers
   * - :doc:`notebooks/01_getting_started`
     - Simulate a full frame, overlay the Gaia catalog, compare the three
       PSF modes on one star, and write the FITS file.
   * - :doc:`notebooks/02_verification`
     - The checks: PSF peak ordering, astrometric residuals, photometric
       closure on the ETC rates, noise statistics, saturation flags, and the
       FITS round trip. Every section ends in an assertion.
   * - :doc:`notebooks/03_psf_focus_modes`
     - The three PSF modes (in-focus Airy, +1/+2-wave defocus), stamp
       sizes, oversampling, and the effect of jitter.
   * - :doc:`notebooks/04_detectors_and_filters`
     - The two detectors (IMX455 vs qCMOS), sensorfilter keys, and how
       Gaia photometry maps to spectral types and count rates.
   * - :doc:`notebooks/05_noise_and_saturation`
     - The ETC noise model step by step, ``n_reads`` semantics, per-pixel
       saturation masks, and the windowed per-star ``saturated`` flag.
   * - :doc:`notebooks/06_astrometry_and_catalogs`
     - The TAN WCS, position angles, injecting your own catalog, and
       astrometric round-trip checks.
   * - :doc:`notebooks/07_cli_and_fits`
     - The ``wcc-sim`` command line and the multi-extension FITS output,
       read back with astropy.
   * - :doc:`notebooks/08_psf_wings`
     - The analytic PSF wing extension: before/after images at hard
       stretch, the fitted power-law wing models, seam smoothness at the
       old stamp edge, and how the wings change the encircled energy.
   * - :doc:`notebooks/09_wcc_phot_photometry`
     - The ``wcc_phot`` differential photometry pipeline: target +
       reference selection, per-frame centroiding, aperture vs PSF
       fluxes, the relative light curve, and the live viewer hook.
   * - :doc:`notebooks/10_plotting`
     - Every ``.plot_*`` method once: images with catalog and saturation
       overlays, cutouts, the PSF radial profile, stamps and encircled
       energy, and the photometry panels, all with ``ax=``.

.. toctree::
   :maxdepth: 1
   :hidden:

   notebooks/01_getting_started
   notebooks/02_verification
   notebooks/03_psf_focus_modes
   notebooks/04_detectors_and_filters
   notebooks/05_noise_and_saturation
   notebooks/06_astrometry_and_catalogs
   notebooks/07_cli_and_fits
   notebooks/08_psf_wings
   notebooks/09_wcc_phot_photometry
   notebooks/10_plotting
