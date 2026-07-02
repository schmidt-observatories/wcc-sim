Gaia catalogs
=============

Cone search
-----------

:func:`~wcc_sim.catalog.query_gaia` performs a synchronous Gaia DR3 TAP query
for all sources with :math:`G \le` ``mag_limit`` inside a circle:

.. code-block:: python

   from wcc_sim.catalog import query_gaia

   cat = query_gaia(291.0, 44.5, radius_arcsec=110.0, mag_limit=21.0,
                    cache_dir="gaia_cache")

The returned astropy Table has the columns ``source_id``, ``ra``, ``dec``,
``phot_g_mean_mag``, ``phot_bp_mean_mag``, ``phot_rp_mean_mag``.

When called from :func:`~wcc_sim.simulate_field`, the search radius is sized
automatically to the detector footprint: half the array diagonal plus a
10-arcsecond margin, so stars just off the edge still contribute their PSF
wings.

On-disk cache
-------------

Pass ``cache_dir=`` to cache query results as ECSV files keyed by
``(ra, dec, radius, mag_limit)``:

.. code-block:: text

   gaia_cache/gaia_291.000000_+44.500000_107.0_21.00.ecsv

Repeat calls with identical parameters read the file instead of hitting the
Gaia archive — simulations become fully offline and reproducible. The
repository's ``notebooks/gaia_cache/`` ships with the cached queries used by
the tutorials.

An empty result (e.g. a pointing at a blank patch or a too-bright
``mag_limit``) produces a ``UserWarning`` and a sky-only image rather than an
error.

Bringing your own catalog
-------------------------

Any astropy Table with the six Gaia columns can be passed straight to
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

The output catalog
------------------

:func:`~wcc_sim.simulate_field` returns the catalog with six added columns
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
   * - ``rate_e_s``
     - Total point-source count rate [e-/s].
   * - ``in_image``
     - True if the star center falls on the array.
   * - ``saturated``
     - True if any saturated pixel lies within 32 px of the star (window
       sized to cover the 2-wave defocus ring; see :doc:`noise`).
