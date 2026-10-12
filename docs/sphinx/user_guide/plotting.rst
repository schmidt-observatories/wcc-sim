Plotting
========

Every result object can draw itself. :class:`~wcc_sim.SimulatedField` and
:class:`~wcc_phot.pipeline.PhotometryResult` carry ``.plot_*`` methods, each
taking ``ax=None`` and returning the axes it drew on:

.. code-block:: python

   field = simulate_field(150.1, 2.2, shape=(1024, 1024), seed=1)
   field.plot(catalog=True, satmask=True)      # a new figure in the house style
   field.plot_cutout(star=0, half_width=120)   # log stretch around catalog row 0
   field.plot_psf_profile()                    # Airy core, scatter halo, render

   result = run_photometry(frames, target=source_id)
   result.plot()                               # the relative light curve
   result.plot_field(frames[0])                # the apertures on the first frame

Pass an axes to draw into a figure of your own; the returned axes can then
be titled or re-limited like any other:

.. code-block:: python

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
   field.plot(ax=ax1, which="clean")
   field.plot_satmask(ax=ax2)
   ax1.set_title("noiseless expectation")

What is available
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 36 42

   * - Object
     - Method
     - Shows
   * - ``SimulatedField``
     - ``plot(which="adu", catalog=False, satmask=False)``
     - The image (``"adu"``, ``"e"`` or ``"clean"``) on an asinh stretch;
       full frames are block-max reduced for display with the axes kept in
       native pixels.
   * -
     - ``plot_cutout(star=0, half_width=300, which="clean")``
     - A log-stretched cutout around one catalog row, floor at the sky level.
   * -
     - ``plot_psf_profile(star=0)``
     - The radial profile split into Airy core, scattered-light halo and
       total, against the rendered image (see :doc:`scatter`).
   * -
     - ``plot_satmask()``
     - The saturation mask.
   * -
     - ``psf_report(path)``
     - The one-page PSF report as PDF and PNG
       (:func:`wcc_sim.psfreport.make_psf_report`).
   * - ``PhotometryResult``
     - ``plot()`` / ``plot_lightcurve(**errorbar_kw)``
     - The normalised relative light curve; extra keywords go to
       ``errorbar`` so two results can share one axes.
   * -
     - ``plot_field(frame)``
     - The frame cropped to the selected stars, apertures at the median
       centroids.
   * -
     - ``plot_binned_rms()``, ``plot_raw_fluxes()``, ``plot_rms_vs_mag()``
     - The noise diagnostics of the report.
   * -
     - ``plot_centroids(coord="x")``, ``plot_background()``
     - Centroid drift about each star's median; annulus background per
       frame.
   * -
     - ``report(path, frame=None)``
     - The one-page photometry report as PDF and PNG
       (:func:`wcc_phot.report.make_report`).

The functions behind the methods are public too, and the ones in
:mod:`wcc_sim.plotting` take plain arrays, so an image read back from a
FITS file can be shown the same way:

.. code-block:: python

   from wcc_sim.plotting import plot_image, plot_stamp, plot_radial_profile

   plot_image(hdul["SCI"].data, satmask=hdul["SATMASK"].data.astype(bool))
   plot_stamp(field.models["stamp"], half_width=40, plate_scale_mas=16.87)
   plot_radial_profile(field.image_clean, xc, yc, bg=sky_level, log=True)

:func:`~wcc_sim.plotting.radial_profile` and
:func:`~wcc_sim.plotting.encircled_energy` return the numbers the last two
draw.

The house style
---------------

Figures the methods create for themselves use the packaged style
``src/wcc_sim/data/plot_default.mplstyle`` (STIX serif, inward ticks, a
light grid, and the teal/red/amber palette exposed as
``wcc_sim.plotting.TEAL`` etc.). An axes you pass in keeps your own
rcParams. To apply the style to a whole session:

.. code-block:: python

   from wcc_sim.plotting import use_style
   use_style()

The report writers (``make_psf_report``, ``make_report``) draw on a
pyplot-free figure under the same style, so they run headless.

The tutorial :doc:`../notebooks/10_plotting` shows every method once.
