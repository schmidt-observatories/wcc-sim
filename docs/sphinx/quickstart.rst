Quick start
===========

This page walks through the core workflow in a few minutes. Everything below
uses the single public entry point, :func:`~wcc_sim.simulate_field`, or the
equivalent :doc:`command line <cli>`.

The mental model
----------------

One call runs the whole pipeline:

.. code-block:: text

   (RA, Dec)  ──►  Gaia DR3 query        (catalog.py)
              ──►  counts/s per star     (starflux.py, via wcc_etc)
              ──►  oversampled PSF       (psf.py: Airy or Zemax defocus)
              ──►  scene rendering       (render.py: sub-pixel placement)
              ──►  noise + digitization  (render.py: Poisson/sky/dark/read, ADU)
              ──►  SimulatedField        (image + satmask + catalog + WCS)
              ──►  FITS                  (fitswriter.py: SCI/SATMASK/CAT/CLEAN)

1. Simulate a field
-------------------

.. code-block:: python

   from wcc_sim import simulate_field

   field = simulate_field(
       291.0, 44.5,              # pointing RA, Dec [deg, ICRS]
       sensorfilter="zwo:r",     # Sony IMX455 + r filter
       focus=1,                  # +1 wave of defocus (0 = in focus, 2 = +2 waves)
       exptime=90,               # seconds
       seed=42,                  # reproducible noise
       output="field_1wave.fits" # write FITS (optional)
   )

The first run performs a Gaia DR3 cone search sized to the detector footprint
(pass ``cache_dir="gaia_cache"`` to reuse it offline afterwards). The full
9568×6380 px array takes ~11 s; pass ``shape=(1024, 1024)`` for a quick look.

2. Inspect the result
---------------------

:func:`~wcc_sim.simulate_field` returns a :class:`~wcc_sim.SimulatedField`:

.. code-block:: python

   field.image_adu        # digitized image [ADU], what a real frame looks like
   field.image_e          # same image in electrons, pre-ADC
   field.image_clean      # noiseless source-only image [e-]
   field.saturation_mask  # bool per pixel: full-well or ADC saturated
   field.wcs              # astropy TAN WCS
   field.catalog          # Gaia table + x, y, spt, rate_e_s, in_image, saturated
   field.params           # every input + derived quantity (gain, sky rate, ...)

.. code-block:: python

   import matplotlib.pyplot as plt
   from astropy.visualization import simple_norm

   plt.imshow(field.image_adu, origin="lower",
              norm=simple_norm(field.image_adu, "asinh", percent=99.5),
              cmap="gray")

3. Or use the command line
--------------------------

The same simulation from the shell:

.. code-block:: bash

   wcc-sim --ra 291.0 --dec 44.5 --sensorfilter zwo:r --focus 1 \
           --exptime 90 --seed 42 -o field_1wave.fits

See :doc:`cli` for every option and a cookbook of common invocations.

4. Read the FITS output back
----------------------------

.. code-block:: python

   from astropy.io import fits
   from astropy.table import Table

   with fits.open("field_1wave.fits") as hdul:
       hdul.info()                      # SCI, SATMASK, CAT, CLEAN
       sci = hdul["SCI"].data           # ADU image with WCS in the header
       sat = hdul["SATMASK"].data       # uint8 saturation mask
       cat = Table.read(hdul["CAT"])    # injected catalog

Where to go next
----------------

- :doc:`user_guide/index` — how each pipeline stage works and which knobs it
  exposes (PSF stamps, jitter, noise model, saturation, FITS layout).
- :doc:`tutorials` — runnable notebooks, from a first quick look to
  photometric closure and astrometric verification.
- :doc:`api/index` — the full API reference.
