.. container::

--------------

README
======

Heat is a distributed tensor framework for high performance data
analytics.

Project Status
--------------

|CPU/CUDA/ROCm tests| |Documentation Status| |coverage| |license: MIT|
|PyPI Version| |Downloads| |Anaconda-Server Badge| |fair-software.eu|
|OpenSSF Scorecard| |OpenSSF Best Practices| |DOI| |Benchmarks| |Code
style: black| |JuRSE Code Pick of the Month|

Table of Contents
-----------------

-  `What is Heat for? <#what-is-heat-for>`__
-  `Features <#features>`__
-  `Support Channels <#support-channels>`__
-  `Citing Heat <#citing-heat>`__
-  `FAQ <#faq>`__
-  `Acknowledgements <#acknowledgements>`__
-  `Getting Started <#getting-started>`__
-  `Installation <#installation>`__

   -  `Requirements <#requirements>`__
   -  `pip <#pip>`__
   -  `conda <#conda>`__

-  `Contribution guidelines <#contribution-guidelines>`__

   -  `Resources <#resources>`__


What is Heat for?
-----------------

Heat builds on `PyTorch <https://pytorch.org/>`__ and
`mpi4py <https://mpi4py.readthedocs.io>`__ to provide high-performance
computing infrastructure for memory-intensive applications within the
NumPy/SciPy ecosystem.

With Heat you can: - port existing NumPy/SciPy code from single-CPU to
multi-node clusters with minimal coding effort; - exploit the entire,
cumulative RAM of your many nodes for memory-intensive operations and
algorithms; - run your NumPy/SciPy code on GPUs (CUDA, ROCm, coming up:
Apple MPS).

For a example that highlights the benefits of multi-node parallelism,
hardware acceleration, and how easy this can be done with the help of
Heat, see, e.g., our `blog post on trucated SVD of a 200GB data
set <https://helmholtz-analytics.github.io/heat/2023/06/16/new-feature-hsvd.html>`__.

Check out our `coverage tables <coverage_tables.md>`__ to see which
NumPy, SciPy, scikit-learn functions are already supported.

If you need a functionality that is not yet supported: - `search
existing issues <https://github.com/helmholtz-analytics/heat/issues>`__
and make sure to leave a comment if someone else already requested it; -
`open a new
issue <https://github.com/helmholtz-analytics/heat/issues/new/choose>`__.

Check out our `features <#features>`__ and the `Heat API
Reference <https://heat.readthedocs.io/en/latest/autoapi/index.html>`__
for a complete list of functionalities.

Features
--------

-  High-performance n-dimensional arrays
-  CPU, GPU, and distributed computation using MPI
-  Powerful data analytics and machine learning methods
-  Seamless integration with the NumPy/SciPy ecosystem
-  Python array API (work in progress)

Support Channels
----------------

Go ahead and ask questions on `GitHub
Discussions <https://github.com/helmholtz-analytics/heat/discussions>`__.
If you found a bug or are missing a feature, then please file a new
`issue <https://github.com/helmholtz-analytics/heat/issues/new/choose>`__.
You can also get in touch with us on
`Mattermost <https://mattermost.hzdr.de/signup_user_complete/?id=3sixwk9okpbzpjyfrhen5jpqfo>`__
(sign up with your GitHub credentials). Once you log in, you can
introduce yourself on the ``Town Square`` channel.

Citing Heat
-----------

.. raw:: html

   <!-- If you find Heat helpful for your research, please mention it in your publications. You can cite: -->

Please do mention Heat in your publications if it helped your research.
You can cite:

-  Götz, M., Debus, C., Coquelin, D., Krajsek, K., Comito, C.,
   Knechtges, P., Hagemeier, B., Tarnawa, M., Hanselmann, S., Siggel,
   S., Basermann, A. & Streit, A. (2020). HeAT - a Distributed and
   GPU-accelerated Tensor Framework for Data Analytics. In 2020 IEEE
   International Conference on Big Data (Big Data) (pp. 276-287). IEEE,
   DOI: 10.1109/BigData50022.2020.9378050.

::

   @inproceedings{heat2020,
       title={{HeAT -- a Distributed and GPU-accelerated Tensor Framework for Data Analytics}},
       author={
         Markus Götz and
         Charlotte Debus and
         Daniel Coquelin and
         Kai Krajsek and
         Claudia Comito and
         Philipp Knechtges and
         Björn Hagemeier and
         Michael Tarnawa and
         Simon Hanselmann and
         Martin Siggel and
         Achim Basermann and
         Achim Streit
       },
       booktitle={2020 IEEE International Conference on Big Data (Big Data)},
       year={2020},
       pages={276-287},
       month={December},
       publisher={IEEE},
       doi={10.1109/BigData50022.2020.9378050}
   }

FAQ
---

Work in progress…

Acknowledgements
----------------

*This work is supported by the* `Helmholtz Association Initiative and
Networking
Fund <https://www.helmholtz.de/en/about_us/the_association/initiating_and_networking/>`__\ *under
project number ZT-I-0003 and the Helmholtz AI platform grant.*

*This project has received funding from Google Summer of Code (GSoC) in
2022.*

*This work is partially carried out under a*
`programme <https://activities.esa.int/index.php/4000144045>`__ *of, and
funded by, the European Space Agency. Any view expressed in this
repository or related publications can in no way be taken to reflect the
official opinion of the European Space Agency.*

--------------

.. raw:: html

   <div align="center">

.. |CPU/CUDA/ROCm tests| image:: https://codebase.helmholtz.cloud/helmholtz-analytics/ci/badges/heat/base/pipeline.svg
   :target: https://codebase.helmholtz.cloud/helmholtz-analytics/ci/-/commits/heat/base
.. |Documentation Status| image:: https://readthedocs.org/projects/heat/badge/?version=latest
   :target: https://heat.readthedocs.io/en/latest/?badge=latest
.. |coverage| image:: https://codecov.io/gh/helmholtz-analytics/heat/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/helmholtz-analytics/heat
.. |license: MIT| image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
.. |PyPI Version| image:: https://img.shields.io/pypi/v/heat
   :target: https://pypi.org/project/heat/
.. |Downloads| image:: https://pepy.tech/badge/heat
   :target: https://pepy.tech/project/heat
.. |Anaconda-Server Badge| image:: https://anaconda.org/conda-forge/heat/badges/version.svg
   :target: https://anaconda.org/conda-forge/heat
.. |fair-software.eu| image:: https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green
   :target: https://fair-software.eu
.. |OpenSSF Scorecard| image:: https://api.securityscorecards.dev/projects/github.com/helmholtz-analytics/heat/badge
   :target: https://securityscorecards.dev/viewer/?uri=github.com/helmholtz-analytics/heat
.. |OpenSSF Best Practices| image:: https://bestpractices.coreinfrastructure.org/projects/7688/badge
   :target: https://bestpractices.coreinfrastructure.org/projects/7688
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2531472.svg
   :target: https://doi.org/10.5281/zenodo.2531472
.. |Benchmarks| image:: https://img.shields.io/badge/Grafana-Benchmarks-2ea44f
   :target: https://57bc8d92-72f2-4869-accd-435ec06365cb.ka.bw-cloud-instance.org:3000/d/adjpqduq9r7k0a/heat-cb?orgId=1
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. |JuRSE Code Pick of the Month| image:: https://img.shields.io/badge/JuRSE_Code_Pick-August_2024-blue
   :target: https://www.fz-juelich.de/en/rse/jurse-community/jurse-code-of-the-month/august-2024


Getting Started
===============

Go to `Quick Start <quick_start.md>`__ for a quick overview. For more
details, see `Installation <#installation>`__.

**You can test your setup** by running the
```heat_test.py`` <https://github.com/helmholtz-analytics/heat/blob/main/scripts/heat_test.py>`__
script:

.. code:: shell

   mpirun -n 2 python heat_test.py

It should print something like this:

.. code:: shell

   x is distributed:  True
   Global DNDarray x:  DNDarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=ht.int32, device=cpu:0, split=0)
   Global DNDarray x:
   Local torch tensor on rank  0 :  tensor([0, 1, 2, 3, 4], dtype=torch.int32)
   Local torch tensor on rank  1 :  tensor([5, 6, 7, 8, 9], dtype=torch.int32)

Check out our Jupyter Notebook
`Tutorials <https://github.com/helmholtz-analytics/heat/blob/main/tutorials/>`__,
choose ``local`` to try things out on your machine, or ``hpc`` if you
have access to an HPC system.

The complete documentation of the latest version is always deployed on
`Read the Docs <https://heat.readthedocs.io/>`__.

.. raw:: html

   <!-- # Goals

   Heat is a flexible and seamless open-source software for high performance data
   analytics and machine learning. It provides highly optimized algorithms and data structures for tensor computations using CPUs, GPUs, and distributed cluster systems on top of MPI. The goal of Heat is to fill the gap between single-node data analytics and machine learning libraries, and  high-performance computing (HPC). Heat's interface integrates seamlessly with the existing data science ecosystem and makes  writing scalable
   scientific and data science applications as effortless as using NumPy.

   Heat allows you to tackle your actual Big Data challenges that go beyond the
   computational and memory needs of your laptop and desktop.
    -->

Installation
------------

Requirements
~~~~~~~~~~~~

Basics
^^^^^^

-  python >= 3.9
-  MPI (OpenMPI, MPICH, Intel MPI, etc.)
-  mpi4py >= 3.0.0
-  pytorch >= 2.0.0

Parallel I/O
^^^^^^^^^^^^

-  h5py
-  netCDF4

GPU support
^^^^^^^^^^^

In order to do computations on your GPU(s): - your CUDA or ROCm
installation must match your hardware and its drivers; - your `PyTorch
installation <https://pytorch.org/get-started/locally/>`__ must be
compiled with CUDA/ROCm support.

HPC systems
^^^^^^^^^^^

On most HPC-systems you will not be able to install/compile MPI or
CUDA/ROCm yourself. Instead, you will most likely need to load a
pre-installed MPI and/or CUDA/ROCm module from the module system. Maybe,
you will even find PyTorch, h5py, or mpi4py as (part of) such a module.
Note that for optimal performance on GPU, you need to usa an MPI library
that has been compiled with CUDA/ROCm support (e.g., so-called
“CUDA-aware MPI”).

pip
~~~

Install the latest version with

.. code:: bash

   pip install heat[hdf5,netcdf]

where the part in brackets is a list of optional dependencies. You can
omit it, if you do not need HDF5 or NetCDF support.

**conda**
~~~~~~~~~

The conda build includes all dependencies **including OpenMPI**.

.. code:: bash

   conda install -c conda-forge heat


Contribution guidelines
=======================

**We welcome contributions from the community, if you want to contribute
to Heat, be sure to review the** `Contribution
Guidelines <contributing.md>`__ **and** `Resources <#resources>`__
**before getting started!**

We use `GitHub
issues <https://github.com/helmholtz-analytics/heat/issues>`__ for
tracking requests and bugs, please see
`Discussions <https://github.com/helmholtz-analytics/heat/discussions>`__
for general questions and discussion. You can also get in touch with us
on
`Mattermost <https://mattermost.hzdr.de/signup_user_complete/?id=3sixwk9okpbzpjyfrhen5jpqfo>`__
(sign up with your GitHub credentials). Once you log in, you can
introduce yourself on the ``Town Square`` channel.

If you’re unsure where to start or how your skills fit in, reach out!
You can ask us here on GitHub, by leaving a comment on a relevant issue
that is already open.

**If you are new to contributing to open source,** `this
guide <https://opensource.guide/how-to-contribute/>`__ **helps explain
why, what, and how to get involved.**

Resources
---------

-  `Heat
   Tutorials <https://github.com/helmholtz-analytics/heat/tree/main/tutorials>`__
-  `Heat API
   Reference <https://heat.readthedocs.io/en/latest/autoapi/index.html>`__

Parallel Computing and MPI:
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  David Henty’s
   `course <https://www.archer2.ac.uk/training/courses/200514-mpi/>`__
-  Wes Kendall’s `Tutorials <https://mpitutorial.com/tutorials/>`__
-  Rolf Rabenseifner’s `MPI course
   material <https://www.hlrs.de/training/self-study-materials/mpi-course-material>`__
   (including C, Fortran **and** Python via ``mpi4py``)

mpi4py
~~~~~~

-  `mpi4py
   docs <https://mpi4py.readthedocs.io/en/stable/tutorial.html>`__
-  `Tutorial <https://www.kth.se/blogs/pdc/2019/08/parallel-programming-in-python-mpi4py-part-1/>`__
   # License

Heat is distributed under the MIT license, see our `LICENSE <LICENSE>`__
file.
