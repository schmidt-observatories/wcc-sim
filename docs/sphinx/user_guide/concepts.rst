Concepts: the pipeline
======================

``wcc-sim`` has a single public entry point, :func:`~wcc_sim.simulate_field`,
that runs a fixed pipeline. Each stage lives in its own module and can also be
used stand-alone (all are documented in the :doc:`../api/index`).

.. code-block:: text

   (RA, Dec, sensorfilter, focus, exptime, ...)
        │
        ▼
   1. catalog.query_gaia        Gaia DR3 cone search, sized to the detector
        │                       footprint (+10" margin), with an on-disk cache;
        │                       or your own table via catalog=
        ▼
   2. starflux.rates_for_catalog
        │                       BP-RP → Pickles dwarf template, count rate
        │                       interpolated in colour between the bracketing
        │                       templates through the wcc_etc instrument model,
        │                       scaled by G mag
        ▼
   3. psf.render_oversampled_psf
        │                       in-focus Airy at each template's effective
        │                       wavelength (or spectrum-weighted when
        │                       chromatic=True), or the Zemax defocus PSF, on an
        │                       11x-oversampled grid, jitter-blurred
        │
        ├─ wings.fit_wing_model power-law diffraction wing fitted to the stamp's
        │                       outer annulus, drawn beyond the stamp down to
        │                       wing_floor_sigma of the background noise
        │
        ├─ scatter.halo_for_sensorfilter
        │                       the measured scattered-light halo, an additive
        │                       term reaching past the chip diagonal
        ▼
   4. render.render_scene       every star sub-pixel placed and binned onto
        │                       the detector grid [e-]; extended.render_extended
        │                       adds Sersic components by FFT convolution
        ▼
   5. render.add_noise_and_digitize
        │                       + sky & dark, per-frame full-well/ADC ceiling,
        │                       Poisson + read noise, gain + bias → ADU
        ▼
   6. SimulatedField            image_adu / image_e / image_clean / satmask /
                                wcs / catalog / params → FITS via fitswriter;
                                optional PSF report via psfreport

Design principles
-----------------

**The ETC is the instrument model.** All count rates, the sky background,
dark current, read noise, gain, bias, full well and ADC limits come from
``wcc_etc`` (the WCC exposure time calculator). The simulator adds only what
an ETC does not have: a real star field, detector *geometry* (array
dimensions), a WCS, and per-pixel rendering. An SNR measured on a ``wcc-sim``
image therefore agrees with the ETC prediction by construction.

**Rates are memoized per spectral type.** A field can contain tens of
thousands of stars, but only ~45 Pickles dwarf types. The rate for each
(type, sensorfilter) pair is computed once at a reference magnitude
(G = 15) and scaled analytically per star with
:math:`10^{-0.4\,(G - 15)}` — this is what makes the full array fast.

**Everything is reproducible.** Pass ``seed=`` to pin the noise realization;
the Gaia query is cached on disk if you pass ``cache_dir=``; every input
parameter (plus derived quantities like the sky rate and plate scale) is
stored in ``field.params`` and written to the FITS header.

The two detectors
-----------------

Sensors are addressed by ``wcc_etc`` *sensorfilter* strings, ``"kind:band"``:

.. list-table::
   :header-rows: 1
   :widths: 18 28 18 18 18

   * - Kind
     - Detector
     - Array [px]
     - Plate scale
     - Bit depth
   * - ``zwo:*``
     - Sony IMX455 (ZWO ASI6200MM)
     - 9568 × 6380
     - 16.87 mas/px
     - 16-bit
   * - ``qcmos:*``
     - Hamamatsu HWK4123 qCMOS
     - 4096 × 2304
     - 20.64 mas/px
     - 12-bit

Array dimensions live in :data:`wcc_sim.detectors.ARRAY_DIMS` (the ``wcc_etc``
``Sensor`` only stores pixel area); everything else is read from the ETC
configuration. Some sensorfilter keys carry an explicit focus suffix (e.g.
``zwo:r+1``); if you do not pass ``focus=``, the sensorfilter's default focus
level is used.

Key parameters at a glance
--------------------------

Every argument of :func:`~wcc_sim.simulate_field`; the docstring has the
details and the CLI flag names are on the :doc:`../cli` page.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Meaning
   * - ``ra, dec``
     - Pointing of the array center [deg, ICRS].
   * - ``sensorfilter``
     - ``wcc_etc`` sensor:filter key, e.g. ``zwo:r``, ``qcmos:bb``.
   * - ``focus``
     - Waves of defocus: 0 (Airy), 1, or 2 (Zemax Huygens PSFs). ``None``
       takes the sensorfilter's default.
   * - ``exptime, n_reads``
     - Total exposure [s] and number of coadded frames (ETC semantics).
   * - ``pa``
     - Position angle of the detector [deg E of N].
   * - ``jitter_sigma_mas``
     - Gaussian jitter blur; default is the telescope's configured jitter.
   * - ``mag_limit``
     - Gaia G faint limit of injected sources (default 21).
   * - ``catalog``
     - Bypass the Gaia query with your own astropy Table (see
       :doc:`catalog`).
   * - ``cache_dir``
     - Directory for the on-disk Gaia cache; repeat runs are offline.
   * - ``shape``
     - ``(ny, nx)`` subarray override — use for quick looks and tests.
   * - ``stamp_npix, oversample``
     - PSF stamp size [detector px, odd] and sub-pixel oversampling
       (odd; default 11). See :doc:`psf`.
   * - ``wavelength_nm``
     - One wavelength for every star's in-focus Airy core (PSF geometry
       only). Default: each star's effective wavelength through the band.
   * - ``chromatic``
     - Spectrum-weighted effective PSFs per template (in focus only).
   * - ``wings, wing_floor_sigma``
     - Draw the analytic diffraction wing beyond the stamp (default on),
       down to this many sigma of the background noise per pixel
       (default 0.1).
   * - ``scatter, scatter_fraction``
     - Add the measured scattered-light halo (default on; off when
       ``wings=False``) and override the instrument-wide scattered
       fraction. See :doc:`scatter`.
   * - ``extended_sources``
     - List of :class:`~wcc_sim.extended.SersicComponent` for analytic
       galaxy light, convolved with the PSF.
   * - ``gain_mode``
     - Readout gain for multi-mode detectors (qCMOS ``"high"`` 32x or
       ``"low"`` 1x); sets which ceiling saturates first. See :doc:`noise`.
   * - ``add_noise, seed``
     - Toggle the noise realization / pin the RNG. An unseeded run draws a
       seed and records it, so it can still be repeated.
   * - ``output, write_clean``
     - Write FITS, optionally with the noiseless ``CLEAN`` extension.
   * - ``report``
     - Also write the one-page PSF radial-profile report (PDF + PNG).
