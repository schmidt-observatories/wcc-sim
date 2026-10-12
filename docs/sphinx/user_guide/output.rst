FITS output and WCS
===================

The ``SimulatedField`` object
-----------------------------

:func:`~wcc_sim.simulate_field` returns a :class:`~wcc_sim.SimulatedField`
dataclass:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - Contents
   * - ``image_adu``
     - Digitized image [ADU], float32, what a real frame looks like.
   * - ``image_e``
     - The same realization in electrons, before gain/bias/clipping. A
       diagnostic, not what photometry reads: ``wcc_phot`` loads the
       delivered product ``(image_adu - bias) * gain`` from an object,
       exactly as it loads ``(SCI - BIAS) * GAIN`` from a file.
   * - ``image_clean``
     - Noiseless expectation image [e-] (sources + sky + dark).
   * - ``saturation_mask``
     - Boolean per pixel: full-well or ADC saturated.
   * - ``wcs``
     - astropy TAN WCS of the frame.
   * - ``catalog``
     - The injected catalog with ``x, y, spt, spt_fallback, rate_e_s,
       psf_wavelength_nm, in_image, saturated`` added (see :doc:`catalog`).
   * - ``params``
     - Every input parameter plus derived quantities (plate scale, gain,
       read noise, sky/dark rates, saturation ceiling, wing and halo
       settings, number of sources, ...); the header cards below are
       written from it.
   * - ``models``
     - The PSF ingredients the render used (stamp, wing model, halo), kept
       for :mod:`wcc_sim.psfreport` and the ``plot_*`` methods; not
       written to FITS.

``field.write(path)`` (or ``output=`` on ``simulate_field``) writes the FITS
file; ``field.to_hdulist()`` returns the ``HDUList`` without touching disk.
``field.plot()`` and its siblings draw the images (see :doc:`plotting`).

FITS layout
-----------

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - EXT
     - Type
     - Contents
   * - ``SCI``
     - Primary image, float32
     - The ADU image, with the full WCS + parameter header. ``BUNIT='adu'``.
   * - ``SATMASK``
     - Image, uint8
     - 1 where the pixel saturated (full well or ADC).
   * - ``CAT``
     - Binary table
     - The injected catalog.
   * - ``CLEAN``
     - Image, float32
     - Noiseless expectation [e-]; omitted with ``write_clean=False`` /
       ``--no-clean``. ``BUNIT='electron'``.

Header cards
------------

Beyond the WCS keywords, the ``SCI`` header records the full provenance.
Cards whose value is ``None`` for a run (no halo, single-mode detector) are
left out.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Group
     - Cards
   * - Pointing
     - ``RA_PNT``, ``DEC_PNT`` [deg, ICRS, array center], ``PA`` [deg E of N].
   * - Inputs
     - ``SENSORF``, ``FOCUS`` [waves], ``EXPTIME`` [s], ``NREADS``,
       ``JITTER`` [mas], ``MAGLIM`` (absent when ``catalog=`` was given),
       ``SEED`` (drawn and recorded when none was given), ``GAIARAD``
       [arcsec], ``ADDNOISE``.
   * - Geometry
     - ``NSRC``, ``NPIX``, ``PLTSCL`` [mas/pix], ``PIXSZUM`` [micron],
       ``STAMPPIX``, ``PSFOVER``.
   * - Detector
     - ``GAIN`` [e-/ADU], ``BIAS`` [ADU], ``GAINMODE``, ``SATLEVEL``
       [e-, the per-frame ceiling min(well, ADC x gain)], ``WELLDEP``
       [e-], ``RDNOISE`` [e-], ``DARK`` and ``SKYRATE`` [e-/s/pix].
   * - PSF
     - ``PSFWAVE`` [nm, the field-level wavelength; per-star values are in
       ``CAT``], ``CHROMPSF``, ``WINGS``, ``WINGA`` (power-law slope),
       ``WINGC`` (amplitude), ``WINGFLR`` [e-/pix, the drawn floor].
   * - Scattered light
     - ``SCATTER``, ``SCATFRAC``, ``SCATNORM`` (``F_FOCAL``, ``P_FULL`` or
       ``USER``: which reference the fraction is quoted against),
       ``SCATFILE`` (the FRED map), ``SCATRCH`` [pix, modelled reach].
   * - Extended sources
     - ``NEXTSRC``.
   * - Software
     - ``WCCSIMV``, ``WCCETCV`` (versions), ``DATE`` (creation time, UTC),
       ``BUNIT``.

The ``CLEAN`` extension carries ``BUNIT = 'electron'``; the ``CAT``
extension is the output catalog of :doc:`catalog`.

The WCS
-------

:func:`~wcc_sim.wcsutil.build_wcs` constructs a gnomonic (``RA---TAN`` /
``DEC--TAN``) WCS centered on the pointing, with no distortion terms:

- ``CRPIX`` is the array center (FITS 1-based convention);
- the CD matrix encodes the plate scale and the position angle ``pa``
  (degrees E of N), with the conventional east-left parity
  (RA increases to the left at ``pa=0``);
- ``pa`` rotates the field on the detector.

Round-trip accuracy of catalog positions through the WCS is at the
floating-point level. The tutorials include an astrometric closure check.

Reading it back
---------------

.. code-block:: python

   from astropy.io import fits
   from astropy.table import Table
   from astropy.wcs import WCS

   with fits.open("field_1wave.fits") as hdul:
       sci = hdul["SCI"].data
       hdr = hdul["SCI"].header
       wcs = WCS(hdr)
       sat = hdul["SATMASK"].data.astype(bool)
       cat = Table.read(hdul["CAT"])
       clean = hdul["CLEAN"].data   # if written

   print(hdr["SENSORF"], hdr["EXPTIME"], hdr["WCCSIMV"])
