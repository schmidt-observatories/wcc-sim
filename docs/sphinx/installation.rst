Installation
============

Requirements
------------

- **Python 3.11 or newer.**
- ``wcc_etc`` **0.6.0 or newer** — the WCC exposure time calculator. It is
  *not* on PyPI; install it from source first (see below).
- The scientific stack pulled in automatically as dependencies: ``numpy``,
  ``scipy``, ``astropy``, ``astroquery``, and ``synphot``.

Install from source
-------------------

``wcc-sim`` depends on ``wcc_etc`` for the instrument model, so install that
first, then ``wcc-sim`` itself, both in *editable* mode so local changes to
``src/`` are picked up immediately:

.. code-block:: bash

   # 1. the instrument model
   git clone git@github.com:schmidt-observatories/wcc-etc.git
   pip install -e wcc-etc

   # 2. the simulator
   git clone git@github.com:schmidt-observatories/wcc-sim.git
   cd wcc-sim
   pip install -e .

Using a conda environment
-------------------------

The synphot / astropy / astroquery stack is happiest in a dedicated
environment:

.. code-block:: bash

   conda create -n wcc-sim python=3.13
   conda activate wcc-sim
   pip install -e <path-to-wcc-etc>
   pip install -e .

Verify the installation
-----------------------

The quickest check runs a tiny noiseless simulation with a hand-made catalog
(no network needed):

.. code-block:: python

   import numpy as np
   from astropy.table import Table
   from wcc_sim import simulate_field

   cat = Table({
       "source_id": [1], "ra": [150.1], "dec": [2.2],
       "phot_g_mean_mag": [12.0], "phot_bp_mean_mag": [12.3],
       "phot_rp_mean_mag": [11.6],
   })
   field = simulate_field(150.1, 2.2, catalog=cat, shape=(512, 512),
                          add_noise=False)
   print(field.image_e.sum())  # total source electrons

If that prints a large positive number, you are ready to go. Continue with
:doc:`quickstart`. (The first *real* simulation also needs network access for
the Gaia DR3 cone search — see :doc:`user_guide/catalog` for the on-disk
cache.)

Running the test suite
----------------------

The package ships with a ``pytest`` suite (no network required — Gaia queries
are mocked):

.. code-block:: bash

   pip install -e ".[test]"
   pytest

Building the documentation locally
-----------------------------------

The documentation you are reading is built with `Sphinx
<https://www.sphinx-doc.org>`_ and the *Read the Docs* theme. The tutorial
pages are rendered from the Jupyter notebooks via `nbsphinx
<https://nbsphinx.readthedocs.io>`_, which requires `pandoc
<https://pandoc.org>`_ to be installed.

.. code-block:: bash

   # from the repository root
   pip install -e .                              # so autodoc can import the package
   pip install -r docs/sphinx/requirements.txt   # Sphinx + theme + nbsphinx
   # plus a pandoc binary, e.g.  conda install pandoc   or   brew install pandoc

   cd docs/sphinx
   make html

Open ``docs/sphinx/_build/html/index.html`` in a browser. To force a clean
rebuild, run ``make clean html``.
