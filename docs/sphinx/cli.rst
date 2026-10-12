Command-line interface
======================

Installing the package puts two executables on your path: ``wcc-sim``
(entry point :func:`wcc_sim.cli.main`), which exposes the full
:func:`~wcc_sim.simulate_field` pipeline, and ``wcc-phot``
(:func:`wcc_phot.cli.main`), which runs :func:`~wcc_phot.run_photometry` on a
series of the frames it writes. ``wcc-sim`` first:

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
   * - ``--no-wings``
     - off
     - Skip the analytic PSF wing beyond the stamp. Also switches the
       scattered-light halo off, since the halo is drawn by the wing
       machinery.
   * - ``--wavelength NM``
     - per star
     - Wavelength of the in-focus Airy core [nm], for every star (PSF
       geometry only; the bandpass still sets the flux). Default: each
       star's effective wavelength through the band. Ignored for
       ``--focus 1|2``.
   * - ``--no-scatter``
     - off
     - Skip the measured scattered-light halo (see
       :doc:`user_guide/scatter`).
   * - ``--scatter-fraction F``
     - ``F_FOCAL``
     - Instrument-wide scattered fraction; default is the ray trace's
       image-plane fraction (7.13e-3). Recorded as ``SCATFRAC`` /
       ``SCATNORM``.
   * - ``--gain-mode MODE``
     - detector's
     - Readout gain for multi-mode detectors. qCMOS: ``high`` (32x, the
       12-bit ADC clips at 552 e-) or ``low`` (1x, the well clips first).
       Recorded as ``GAINMODE`` / ``SATLEVEL``.
   * - ``--report PATH``
     - none
     - Also write the one-page PSF radial-profile report (PDF + PNG, Airy
       core vs scattered-light halo; :func:`wcc_sim.psfreport.make_psf_report`).
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

Scattered light: the same field with the halo off, and with the PSF report
written next to the image:

.. code-block:: bash

   wcc-sim --ra 291.0 --dec 44.5 --no-scatter --seed 42 -o no_halo.fits
   wcc-sim --ra 291.0 --dec 44.5 --seed 42 --report psf_report.pdf -o with_halo.fits

qCMOS in its low-gain (1x) mode, so the 7500 e- well rather than the 12-bit
ADC sets the saturation ceiling:

.. code-block:: bash

   wcc-sim --ra 150.1 --dec 2.2 --sensorfilter qcmos:bb --gain-mode low \
           --exptime 60 --seed 7 -o qcmos_lowgain.fits

``wcc-phot``
------------

``wcc-phot`` measures a differential light curve of one target against the
best reference stars across a series of ``wcc-sim`` frames of the same field
(same sensorfilter and focus). The target is a Gaia ``source_id`` or an
(RA, Dec) pair; everything else has a default derived from the frames'
headers and the model PSF:

.. code-block:: bash

   wcc-phot series*.fits --source-id 2130453233564838016 -o phot.fits
   wcc-phot series*.fits --ra 291.0 --dec 44.5 --method psf \
            --report phot_report.pdf -o phot.fits

It prints a one-line summary (frames, target, references, aperture, rel-flux
RMS against the median predicted error) and writes a FITS file with
``STARS`` (the selection), ``PHOT`` (one row per star per frame) and ``LC``
(one row per frame) extensions; see :class:`wcc_phot.pipeline.PhotometryResult`.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Meaning
   * - ``frames`` *(positional)*
     - —
     - One or more ``wcc-sim`` FITS frames of the same field.
   * - ``--source-id ID``
     - —
     - Gaia ``source_id`` of the target (this or ``--ra``/``--dec``).
   * - ``--ra DEG``, ``--dec DEG``
     - —
     - Target coordinates; must match a catalog star within 2 arcsec.
   * - ``-o, --output`` *(required)*
     - —
     - Output FITS path (``STARS`` / ``PHOT`` / ``LC``).
   * - ``--method``
     - ``aperture``
     - ``aperture`` or ``psf`` (fit the wcc-sim PSF model as a photutils
       ``ImagePSF``, initialised from the aperture pass).
   * - ``--n-ref``
     - ``10``
     - Number of reference stars (fewer are used, with a warning, when the
       field cannot supply them).
   * - ``--r-ap``
     - model EE radius
     - Aperture radius [px]; default encloses ``--ee`` of the model PSF.
   * - ``--r-in``, ``--r-out``
     - ``1.5``, ``2.5`` r_ap
     - Background annulus radii [px].
   * - ``--centroid-box``
     - ``2 r_ap + 1``
     - Centre-of-mass centroid box side [px, odd].
   * - ``--fit-shape``
     - ``2 r_ap + 1``
     - PSF fit region side [px, odd] (``--method psf``).
   * - ``--ee``
     - ``0.95``
     - Encircled-energy fraction that sets the default ``r_ap``.
   * - ``--iso-dmag``
     - ``1.0``
     - Reject a reference with a neighbour brighter than G + ``iso_dmag``
       inside the annulus radius.
   * - ``--lc-csv PATH``
     - none
     - Also write the light curve as ECSV.
   * - ``--report PATH``
     - none
     - One-page PDF + PNG diagnostic report (field with apertures, light
       curve, centroid drift, noise metrics;
       :func:`wcc_phot.report.make_report`).
   * - ``--live``
     - off
     - Matplotlib window updated as each frame is measured
       (:class:`wcc_phot.live.LiveViewer`); ``--live-pause S`` sets the
       cadence and ``--live-zoom N`` crops to +/- N px around the target.

Mixed sensorfilters or focus levels in one series are rejected up front.
Frames are streamed one at a time, so a long series does not need to fit in
memory. The tutorial :doc:`notebooks/08_wcc_phot_photometry` walks through
the same pipeline from Python.

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
   * - ``scripts/example_photometry.sh``
     - Five dithered 4096x4096 frames of the Kepler-field pointing, then
       ``wcc-phot`` on them (add ``--live`` to watch).
   * - ``scripts/transit_trappist1b.py``
     - End-to-end validation: injects the TRAPPIST-1b transit into the
       Gaia magnitude of TRAPPIST-1, simulates 36 x 300 s defocused frames,
       runs ``wcc_phot`` and checks the recovered depth.
   * - ``scripts/cepheids_m101.py``
     - End-to-end validation: synthetic Cepheids and calibration stars on
       the M101 disk (Sersic galaxy light plus the real Gaia foreground),
       with a recovered period-luminosity relation. ``--quick`` for a
       smoke run.
   * - ``scripts/build_scatter_profile.py``
     - Regenerates the packaged azimuthal scattered-light profile from the
       FRED stray-light map (see :doc:`user_guide/scatter`).
   * - ``scripts/build_spt_colors.py``
     - Regenerates the synthetic BP-RP / B-V colour table of the Pickles
       templates (network: synphot's Johnson curves and Vega).
