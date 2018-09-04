HeAT - Helmholtz Analytics Toolkit
==================================

![HeAT Logo](doc/images/logo_HeAT.png)

HeAT is a distributed tensor framework for high performance data analytics.

Project Status
--------------

[![Build Status](https://travis-ci.com/helmholtz-analytics/heat.svg?branch=master)](https://travis-ci.com/helmholtz-analytics/heat)
[![Documentation Status](https://readthedocs.org/projects/heat/badge/?version=latest)](https://heat.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/helmholtz-analytics/heat/branch/master/graph/badge.svg)](https://codecov.io/gh/helmholtz-analytics/heat)

Goals
-----

The goal of HeAT is to fill the gap between machine learning libraries that have
a strong focus on exploiting GPUs for performance, and traditional, distributed
high-performance computing (HPC). The basic idea is to provide a generic,
distributed tensor library with machine learning methods based on it.

Among other things, the implementation will allow us to tackle use cases that
would otherwise exceed memory limits of a single node.

Features
--------

  * high-performance n-dimensional tensors
  * CPU, GPU and distributed computation using MPI
  * powerful machine learning methods using above mentioned tensors

Requirements
------------

HeAT is based on [PyTorch](https://pytorch.org/). Specifially, we are exploiting
PyTorch's support for GPUs *and* MPI parallelism. Therefore, PyTorch must be
compiled with MPI support when using HeAT. The instructions to install PyTorch
in that way are contained in the script
[install-torch.sh](install-torch.sh),
which we're also using to install PyTorch in Travis CI.

Installation
------------

Tagged releases are made available on the
[Python Package Index (PyPI)](https://pypi.org/project/heat/). You can typically
install the latest version with

> $ pip install heat

If you want to work with the development version, you can checkout the sources using

> $ git clone https://github.com/helmholtz-analytics/heat.git

License
-------

HeAT is distributed under the MIT license, see our
[LICENSE](LICENSE) file.

Acknowledgements
----------------

This work is supported by the [Helmholtz Association Initiative and
Networking](https://www.helmholtz.de/en/about_us/the_association/initiating_and_networking/)
Fund under project number ZT-I-0003.
