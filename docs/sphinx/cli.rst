Command-line interface
======================

Installing the package puts a ``wcc-sim`` executable on your path (entry
point: :func:`wcc_sim.cli.main`). It exposes the full
:func:`~wcc_sim.simulate_field` pipeline:

.. code-block:: bash

   wcc-sim --ra 291.0 --dec 44.5 --sensorfilter zwo:r --focus 1 \
           --exptime 90 --seed 42 -o field_1wave.fits

On success it prints a one-line summary and exits 0:

.. code-block:: text

   Wrote field_1wave.fits: 9568x6380 px, 52840 sources, focus=1w, 3121 saturated px

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Meaning
   * - ``--ra`` *(required)*
     - —
     - Pointing RA [deg, ICRS].
   * - ``--dec`` *(required)*
     - —
     - Pointing Dec [deg, ICRS].
   * - ``-o, --output`` *(required)*
     - —
     - Output FITS path.
   * - ``--sensorfilter``
     - ``zwo:r``
     - ``wcc_etc`` sensorfilter key, e.g. ``zwo:r``, ``zwo:r+1``,
       ``qcmos:bb``.
   * - ``--focus``
     - sensorfilter's
     - Waves of defocus: ``0``, ``1``, or ``2``.
   * - ``--exptime``
     - ``90``
     - Total exposure time [s].
   * - ``--n-reads``
     - ``1``
     - Coadded frames (ETC semantics; saturation is per frame).
   * - ``--pa``
     - ``0``
     - Position angle [deg E of N].
   * - ``--jitter``
     - telescope's
     - Jitter Gaussian sigma [mas].
   * - ``--mag-limit``
     - ``21``
     - Gaia G faint limit of injected sources.
   * - ``--no-noise``
     - off
     - Skip the noise realization (expectation image only).
   * - ``--no-clean``
     - off
     - Omit the noiseless ``CLEAN`` extension (halves file size).
   * - ``--seed``
     - none
     - RNG seed for a reproducible noise realization.
   * - ``--shape NY NX``
     - full array
     - Subarray shape — quick looks and tests.
   * - ``--stamp-npix``
     - per focus
     - PSF stamp size [detector px, odd]; see :doc:`user_guide/psf`.
   * - ``--cache-dir``
     - none
     - Gaia query cache directory (offline reruns; see
       :doc:`user_guide/catalog`).

Cookbook
--------

Quick look — 1024×1024 subarray, cached Gaia query:

.. code-block:: bash

   wcc-sim --ra 291.0 --dec 44.5 --shape 1024 1024 \
           --cache-dir gaia_cache --seed 42 -o quicklook.fits

Focus sweep — the same field at all three focus levels:

.. code-block:: bash

   for f in 0 1 2; do
     wcc-sim --ra 291.0 --dec 44.5 --focus $f --exptime 90 --seed 42 \
             --shape 1024 1024 --cache-dir gaia_cache -o field_focus$f.fits
   done

Full-frame qCMOS image, broadband filter:

.. code-block:: bash

   wcc-sim --ra 150.1 --dec 2.2 --sensorfilter qcmos:bb \
           --exptime 60 --seed 7 -o qcmos_bb.fits

Noiseless truth image for photometric closure (no noise, keep CLEAN):

.. code-block:: bash

   wcc-sim --ra 291.0 --dec 44.5 --no-noise --shape 2048 2048 \
           --cache-dir gaia_cache -o truth.fits

Bright-star handling — split 900 s into 10 reads so per-frame exposure
stays short, and extend the jitter:

.. code-block:: bash

   wcc-sim --ra 291.0 --dec 44.5 --exptime 900 --n-reads 10 \
           --jitter 50 --seed 42 -o deep_10reads.fits

Example scripts
---------------

The repository's ``scripts/`` directory contains ready-to-run examples:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Script
     - What it does
   * - ``scripts/smoke_run.py``
     - Python end-to-end smoke run: a Kepler-field pointing at all three
       focus levels, 1024×1024, FITS to ``smoke_out/``. Network required
       once (queries are cached).
   * - ``scripts/example_quicklook.sh``
     - Single fast subarray image with a cached Gaia query.
   * - ``scripts/example_focus_sweep.sh``
     - The CLI focus sweep above, one FITS per focus level.
   * - ``scripts/example_detectors.sh``
     - The same pointing on both detectors (IMX455 r-band, qCMOS
       broadband) for a side-by-side comparison.
