HeAT - Helmholtz Analytics Toolkit
==================================

![HeAT Logo](https://raw.githubusercontent.com/helmholtz-analytics/heat/master/doc/images/logo_HeAT.png)

HeAT is a distributed tensor framework for high performance data analytics.

Project Status
--------------

[![Build Status](https://travis-ci.com/helmholtz-analytics/heat.svg?branch=master)](https://travis-ci.com/helmholtz-analytics/heat)
[![Documentation Status](https://readthedocs.org/projects/heat/badge/?version=latest)](https://heat.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/helmholtz-analytics/heat/branch/master/graph/badge.svg)](https://codecov.io/gh/helmholtz-analytics/heat)

Goals
-----

HeAT is a flexible and seamless open-source software for high performance data
analytics and machine learning. It provides highly optimized algorithms and data
structures for tensor computations using CPUs, GPUs and distributed cluster
systems on top of MPI. The goal of HeAT is to fill the gap between data
analytics and machine learning libraries with a strong focus on on single-node
performance, and traditional high-performance computing (HPC). HeAT's generic
Python-first programming interface integrates seamlessly with the existing data
science ecosystem and makes it as effortless as using numpy to write scalable
scientific and data science applications.

HeAT allows you to tackle your actual Big Data challenges that go beyond the
computational and memory needs of your laptop and desktop.

Features
--------

* High-performance n-dimensional tensors
* CPU, GPU and distributed computation using MPI
* Powerful data analytics and machine learning methods
* Abstracted communication via split tensors
* Python API

Getting Started
---------------

Check out our Jupyter Notebook [tutorial](https://github.com/helmholtz-analytics/heat/blob/master/scripts/tutorial.ipynb)
right here on Github or in the /scripts directory.

The complete documentation of the latest version is always deployed on
[Read the Docs](https://heat.readthedocs.io/).

Requirements
------------

HeAT is based on [PyTorch](https://pytorch.org/). Specifially, we are exploiting
PyTorch's support for GPUs *and* MPI parallelism. For MPI support we utilize
[mpi4py](https://mpi4py.readthedocs.io). Both packages can be installed via pip
or automatically using the setup.py.


Installation
------------

Tagged releases are made available on the
[Python Package Index (PyPI)](https://pypi.org/project/heat/). You can typically
install the latest version with

> $ pip install heat[hdf5, netcdf]

where the part in brackets is a list of optional dependencies. You can omit
it, if you do not need HDF5 or NetCDF support.

Hacking
-------

If you want to work with the development version, you can checkout the sources using

> $ git clone https://github.com/helmholtz-analytics/heat.git

The installation can then be done from the checked out sources with

> $ pip install .[hdf5, netcdf, dev]

The extra `dev` dependency pulls in additional tools to support the enforcement
of coding conventions ([Black](https://github.com/psf/black)) and to support a
pre-commit hook to do the same. In order to fully use this framework, please
also install the pre-commit hook with

> $ pre-commit install

In order to check compliance of your changes before even trying to commit anything,
you can run

> $ pre-commit run --all-files

License
-------

HeAT is distributed under the MIT license, see our
[LICENSE](LICENSE) file.

Acknowledgements
----------------

*This work is supported by the [Helmholtz Association Initiative and
Networking](https://www.helmholtz.de/en/about_us/the_association/initiating_and_networking/)
Fund under project number ZT-I-0003.*
