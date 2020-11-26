<div align="center">
  <img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/master/doc/images/logo.png">
</div>

---

HeAT is a distributed tensor framework for high performance data analytics.

Project Status
--------------

[![Build Status](https://travis-ci.com/helmholtz-analytics/heat.svg?branch=master)](https://travis-ci.com/helmholtz-analytics/heat)
[![Documentation Status](https://readthedocs.org/projects/heat/badge/?version=latest)](https://heat.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/helmholtz-analytics/heat/branch/master/graph/badge.svg)](https://codecov.io/gh/helmholtz-analytics/heat)
[![license: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/heat)](https://pepy.tech/project/heat)
[![Mattermost Chat](https://img.shields.io/badge/chat-on%20mattermost-blue.svg)](https://mattermost-haf.fz-juelich.de/signup_user_complete/?id=iqrr6pmxb38fzqffa51qqhcu8w)

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

Support Channels
----------------

We use [StackOverflow](https://stackoverflow.com/tags/pyheat/) as a forum for questions about Heat.
If you do not find an answer to your question, then please ask a new question there and be sure to
tag it with "pyheat".

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

> $ pip install heat[hdf5,netcdf]

where the part in brackets is a list of optional dependencies. You can omit
it, if you do not need HDF5 or NetCDF support.

Hacking
-------

If you want to work with the development version, you can check out the sources using

> $ git clone https://github.com/helmholtz-analytics/heat.git

The installation can then be done from the checked-out sources with

> $ pip install .[hdf5,netcdf,dev]

We welcome contributions from the community, please check out our [Contribution Guidelines](contributing.md) before getting started!

License
-------

HeAT is distributed under the MIT license, see our
[LICENSE](LICENSE) file.

Citing HeAT
-----------

If you find HeAT helpful for your research, please mention it in your academic publications. You can cite:

- Götz, M., Debus, C., Coquelin, D., et al., "HeAT - a Distributed and GPU-accelerated Tensor Framework for Data Analytics." 2020 IEEE International Conference on Big Data (Big Data). IEEE, 2020 (accepted). [[Download](https://arxiv.org/abs/2007.13552)]

Acknowledgements
----------------

*This work is supported by the [Helmholtz Association Initiative and
Networking Fund](https://www.helmholtz.de/en/about_us/the_association/initiating_and_networking/)
under project number ZT-I-0003.*

*This work is supported by the [Helmholtz Association Initiative and
Networking Fund](https://www.helmholtz.de/en/about_us/the_association/initiating_and_networking/)
under the Helmholtz AI platform grant.*

---

<div align="center">
  <a href="https://www.dlr.de/EN/Home/home_node.html"><img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/master/doc/images/dlr_logo.svg" height="50px" hspace="3%" vspace="20px"></a><a href="https://www.fz-juelich.de/portal/EN/Home/home_node.html"><img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/master/doc/images/fzj_logo.svg" height="50px" hspace="3%" vspace="20px"></a><a href="http://www.kit.edu/english/index.php"><img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/master/doc/images/kit_logo.svg" height="50px" hspace="3%" vspace="20px"></a><a href="https://www.helmholtz.de/en/"><img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/master/doc/images/helmholtz_logo.svg" height="50px" hspace="3%" vspace="20px"></a>
</div>
