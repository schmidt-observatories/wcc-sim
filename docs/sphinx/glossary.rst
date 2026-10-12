Glossary
========

The terms these pages use without stopping to define them. Each entry links
to the page that goes into detail.

.. glossary::
   :sorted:

   ADU
      Analog-to-digital unit, the integer a camera reports for each pixel. It
      is the electron count divided by the :term:`gain`, plus the bias level,
      clipped to the range of the ADC. ``image_adu`` is in these units; see
      :doc:`user_guide/noise`.

   gain
      The conversion between electrons and ADU. In wcc-sim the ``gain``
      parameter and the ``GAIN`` header card are in electrons per ADU, so a
      delivered frame converts back to electrons as ``(ADU - bias) * gain``.

   bias
      A constant offset added to every pixel before digitization, in ADU.
      Both WCC sensors carry a bias of zero in the ETC configuration, so the
      clip at zero ADU removes negative read-noise excursions.

   full well
      The largest number of electrons a pixel can hold before it saturates.
      The Sony IMX455 well is about 16 ke-. Together with the
      :term:`ADC ceiling` it sets the per-frame saturation level.

   ADC ceiling
      The largest value the analog-to-digital converter can report
      (``adc_max``, 65535 for a 16-bit sensor), expressed in electrons as
      ``adc_max * gain``. A pixel saturates at the smaller of the full well
      and the ADC ceiling; that per-frame level is written to the ``SATLEVEL``
      header card and the per-pixel result to the ``SATMASK`` extension.

   read noise
      The Gaussian noise added each time a frame is read out, in electrons.
      With :term:`n_reads` coadded frames the read-noise variance grows as
      ``n_reads``.

   dark current
      Electrons generated in a pixel without any light, at a rate in e-/s per
      pixel that comes from the ETC configuration.

   sky background
      The zodiacal-light surface brightness through the filter, converted by
      the ETC to a rate in e-/s per pixel and added uniformly to the image.

   n_reads
      The number of equal frames a total exposure is split into and summed.
      Saturation is evaluated per frame, so more reads keep bright stars
      unsaturated at the cost of more read noise. See :doc:`user_guide/noise`.

   count rate
      The number of electrons per second a star produces in the detector
      through the full telescope, filter and detector throughput. It comes
      from the ETC, once per spectral type at G = 15, and is scaled per star
      by its G magnitude. See :doc:`user_guide/fluxes`.

   sensorfilter
      The ``"kind:band"`` string that names a detector and filter in wcc-etc,
      such as ``zwo:r`` or ``qcmos:bb``. A suffix such as ``zwo:r+1`` or
      ``zwo:bb2`` names a defocused configuration. See
      :doc:`user_guide/concepts`.

   focus
      The amount of defocus in waves: 0 is the in-focus :term:`Airy` PSF, 1
      and 2 use the Zemax :term:`Huygens PSF` data. The WCC defocuses on
      purpose to spread bright stars over more pixels before they saturate.
      See :doc:`user_guide/psf`.

   PSF
      Point-spread function, the image a point source such as a star leaves
      on the detector. wcc-sim renders each star by placing a PSF at its
      position.

   Airy
      The diffraction pattern of a circular aperture: a bright core
      surrounded by faint rings. It is the in-focus PSF model, computed at
      each star's effective wavelength through the band.

   Huygens PSF
      A physical-optics PSF computed by Zemax and bundled with wcc-etc, used
      for the +1 and +2 wave defocused modes. Both are annular, with a
      centrally depressed core.

   jitter
      Pointing motion during the exposure, modelled as a Gaussian blur of the
      PSF with a width ``jitter_sigma_mas`` in milliarcseconds.

   mas
      Milliarcsecond, one thousandth of an arcsecond. Plate scales and
      jitter are quoted in mas.

   plate scale
      The angle on the sky covered by one pixel: 16.87 mas per pixel for the
      Sony IMX455 and 20.64 mas per pixel for the Hamamatsu qCMOS.

   oversampling
      The PSF is rendered on a grid finer than the detector pixels (11 times
      finer by default), shifted to the star's sub-pixel position, and then
      binned down to detector pixels. See :doc:`user_guide/psf`.

   stamp
      The finite square image of the PSF drawn for each star, ``stamp_npix``
      detector pixels on a side. Light outside the stamp is handled by the
      :term:`wings` and the :term:`scattered-light halo`.

   wings
      The analytic power-law extension of the PSF beyond the stamp edge,
      fitted to the stamp's outer annulus and drawn out to where it falls
      below a fraction of the background noise. See :doc:`user_guide/psf`.

   scattered-light halo
      The measured stray-light profile of the telescope, added around every
      star as a term that reaches across the whole array. Its normalisation
      is recorded in the ``SCATNORM`` header card. See
      :doc:`user_guide/scatter`.

   WCS
      World coordinate system, the mapping between pixel coordinates and sky
      coordinates stored in the FITS header. wcc-sim writes a TAN WCS.

   TAN
      The gnomonic (tangent-plane) projection used by the WCS, the standard
      choice for a flat detector imaging a small patch of sky.

   Gaia DR3
      The third data release of the Gaia mission, the source catalogue wcc-sim
      queries. Each star carries a G magnitude and the BP and RP magnitudes
      whose difference, the BP-RP colour, selects its spectral template. See
      :doc:`user_guide/catalog`.

   Pickles template
      A library of observed stellar spectra sorted by spectral type. wcc-sim
      uses the dwarf templates to turn a Gaia colour into a spectrum. See
      :doc:`user_guide/fluxes`.

   encircled energy
      The fraction of a star's total light that falls within a given radius
      of its centre, as a function of that radius. The default photometry
      aperture is the 95 per cent encircled-energy radius.

   asinh stretch
      A display scaling that is linear for faint pixels and logarithmic for
      bright ones, so a sky-level star and a saturated one can be seen in the
      same image. ``field.plot()`` uses it. See :doc:`user_guide/plotting`.

   SNR
      Signal-to-noise ratio of a measurement, here the flux of a star in an
      aperture divided by the noise in that aperture.

   CLEAN image
      The noiseless expectation image in electrons (``image_clean``), the
      sources plus sky and dark with no noise drawn, kept for photometric
      closure tests and written as the optional ``CLEAN`` FITS extension.
