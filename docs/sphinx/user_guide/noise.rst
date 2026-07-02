Noise and saturation
====================

:func:`~wcc_sim.render.add_noise_and_digitize` applies the ``wcc_etc`` noise
model to the rendered source image, following the ETC's semantics exactly so
that measured SNRs match ETC predictions.

The model
---------

Starting from the source-only image in electrons:

1. **Sky + dark.** Uniform per-pixel rates (zodiacal background and dark
   current from the ETC) times ``exptime`` are added, giving the *clean*
   expectation image ``image_clean``.
2. **Saturation cap.** The per-frame expectation ``image_clean / n_reads``
   is compared against the full-well depth and the ADC ceiling
   (``adc_max × gain``); the expectation is capped at the smaller of the two
   (times ``n_reads``), and the per-pixel boolean ``satmask`` records which
   pixels hit the limit.
3. **Poisson noise** on the (capped) expectation.
4. **Read noise**, Gaussian with variance scaling as ``n_reads`` (each
   coadded frame contributes one read).
5. **Digitization.** Electrons are divided by the gain, the bias level is
   added, and the result is clipped to ``[0, n_reads × adc_max]`` ADU.

With ``add_noise=False``, steps 3–4 are skipped and ``image_e`` equals
``image_clean`` — useful for photometric closure tests.

``n_reads`` semantics
---------------------

``n_reads`` follows the ETC convention of *coadded frames*: the total
exposure time is split into ``n_reads`` equal frames that are summed.
Consequences:

- saturation is evaluated **per frame** — splitting a long exposure into
  more reads is the way to keep bright stars unsaturated;
- read-noise variance grows linearly with ``n_reads``;
- the ADU ceiling of the summed image is ``n_reads × adc_max``.

Per-star saturation flags
-------------------------

The FITS/output catalog carries a per-star ``saturated`` column computed by
:func:`~wcc_sim.render.star_saturated`. It flags a star if **any** saturated
pixel lies within 32 px of its center — a window, not just the central pixel,
because the defocused PSFs are centrally depressed: a bright star can
saturate its ring while its central pixel stays below full well (the 2-wave
ring extends to ~29 px).

Quick example
-------------

.. code-block:: python

   from wcc_sim import simulate_field

   field = simulate_field(291.0, 44.5, sensorfilter="zwo:r", focus=2,
                          exptime=90, n_reads=1, seed=42, shape=(1024, 1024))
   n_sat_px = field.saturation_mask.sum()
   sat_stars = field.catalog[field.catalog["saturated"]]

   # Same field, 9 reads: per-frame exposure drops from 90 s to 10 s
   field9 = simulate_field(291.0, 44.5, sensorfilter="zwo:r", focus=2,
                           exptime=90, n_reads=9, seed=42, shape=(1024, 1024))

Every noise-model quantity used is recorded in ``field.params`` (and the FITS
header): ``gain``, ``read_noise``, ``dark_e_s``, ``sky_e_s``, ``well_depth``.
