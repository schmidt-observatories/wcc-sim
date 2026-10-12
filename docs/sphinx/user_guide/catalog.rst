Gaia catalogs
=============

Cone search
-----------

:func:`~wcc_sim.catalog.query_gaia` runs a Gaia DR3 TAP query for all
sources with :math:`G \le` ``mag_limit`` inside a circle:

.. code-block:: python

   from wcc_sim.catalog import query_gaia

   cat = query_gaia(291.0, 44.5, radius_arcsec=110.0, mag_limit=21.0,
                    cache_dir="gaia_cache")

The returned astropy Table has the photometry the simulator needs,
``source_id``, ``ra``, ``dec``, ``phot_g_mean_mag``, ``phot_bp_mean_mag``,
``phot_rp_mean_mag``, plus the astrometry columns ``pmra``, ``pmdec``,
``parallax`` and ``radial_velocity``. Positions are Gaia's, at epoch
J2016.0; :func:`~wcc_sim.simulate_field` places stars where Gaia saw them
and does not propagate them to an observation date. The astrometry columns
are there for :func:`wcc_sim.astrometry.propagate`, which the bright-star
supplement below uses to bring two catalogs to one epoch.

When called from :func:`~wcc_sim.simulate_field`, the search radius is sized
automatically to the detector footprint: half the array diagonal plus a
10-arcsecond margin, so stars just off the edge still contribute their PSF
wings.

The query runs as an asynchronous TAP job, because the archive caps
synchronous jobs at 2000 rows and returns an arbitrary subset without
saying so. A result of exactly that length is still treated as possibly
truncated: it raises a ``UserWarning`` and is not cached. An empty result
(a blank patch, or a too-bright ``mag_limit``) also warns and produces a
sky-only image rather than an error.

On-disk cache
-------------

Pass ``cache_dir=`` to cache query results as ECSV files keyed by
``(ra, dec, radius, mag_limit)``:

.. code-block:: text

   gaia_cache/gaia_291.000000_+44.500000_107.0_21.00.ecsv

The pointing in the key is rounded to 1 arcsec, so the frames of a dithered
series (sub-pixel to a few arcsec apart) share one query. Repeat calls with
identical parameters read the file instead of hitting the Gaia archive, so
simulations become fully offline and reproducible. A cache file written
before the astrometry columns existed is ignored with a warning and the
archive is queried again. The repository's ``notebooks/gaia_cache/`` ships
with the cached queries used by the tutorials.

Bright stars Gaia does not have
-------------------------------

Gaia's brightest source is G = 1.7, so the naked-eye stars whose halos drive
the stray-light requirements are missing from a DR3 query. The optional
:mod:`wcc_sim.brightcat` module fills that end from XHIP (Hipparcos plus
radial velocities, VizieR V/137D): :func:`~wcc_sim.brightcat.query_bright`
fetches and caches the rows, :func:`~wcc_sim.brightcat.to_gaia_like`
converts them to the Gaia-shaped table (G from V and B-V, a template from
the spectral type), and :func:`~wcc_sim.brightcat.merge` propagates both
catalogs to one epoch, cross-matches them, and lets the Hipparcos row win
for stars brighter than G = 6 (where DR3's saturation systematics live).
The merged table is sorted brightest first, so row 0, the row the PSF report
decomposes, is the brightest star in the field. It is not part of the
default pipeline; pass the merged table as ``catalog=``.

Bringing your own catalog
-------------------------

Any astropy Table with the six photometric Gaia columns (the astrometry
columns are optional) can be passed straight to
:func:`~wcc_sim.simulate_field` via ``catalog=`` — the Gaia query is skipped
entirely. This is how the test suite runs without network, and it is the
easiest way to inject synthetic grids of stars:

.. code-block:: python

   import numpy as np
   from astropy.table import Table
   from wcc_sim import simulate_field

   n = 25
   cat = Table({
       "source_id": np.arange(n),
       "ra": 150.1 + np.random.uniform(-0.02, 0.02, n),
       "dec": 2.2 + np.random.uniform(-0.02, 0.02, n),
       "phot_g_mean_mag": np.random.uniform(10, 18, n),
       "phot_bp_mean_mag": np.full(n, np.nan),   # NaN color -> G2V
       "phot_rp_mean_mag": np.full(n, np.nan),
   })
   field = simulate_field(150.1, 2.2, catalog=cat, shape=(2048, 2048), seed=1)

A ``spt`` column, when present, pins the template of a star instead of
deriving it from BP-RP. Rows without a finite G magnitude, or that do not
project onto the tangent plane, are dropped with a warning.

The output catalog
------------------

:func:`~wcc_sim.simulate_field` returns the catalog with eight added columns
(also written to the FITS ``CAT`` extension):

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Column
     - Meaning
   * - ``x, y``
     - 0-based pixel position from the WCS (may be off-array).
   * - ``spt``
     - Assigned Pickles dwarf spectral type (see :doc:`fluxes`).
   * - ``spt_fallback``
     - True when the star had no usable BP-RP and got the G2V fallback.
   * - ``rate_e_s``
     - Total point-source count rate [e-/s].
   * - ``psf_wavelength_nm``
     - Wavelength of the star's in-focus Airy core (its template's
       effective wavelength through the band, or the ``wavelength_nm``
       override; see :doc:`psf`).
   * - ``in_image``
     - True if the star center falls on the array.
   * - ``saturated``
     - True if any saturated pixel lies within 32 px of the star (window
       sized to cover the 2-wave defocus ring; see :doc:`noise`).
