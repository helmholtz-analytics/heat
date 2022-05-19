<div align="center">
  <img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/main/doc/images/logo.png">
</div>

---

Heat is a distributed tensor framework for high performance data analytics.

Project Status
--------------

[![Jenkins](https://img.shields.io/jenkins/build?jobUrl=https%3A%2F%2Fheat-ci.fz-juelich.de%2Fjob%2Fheat%2Fjob%2Fheat%2Fjob%2Fmain%2F&label=CPU)](https://heat-ci.fz-juelich.de/blue/organizations/jenkins/heat%2Fheat/activity?branch=main)
[![Jenkins](https://img.shields.io/jenkins/build?jobUrl=https%3A%2F%2Fheat-ci.fz-juelich.de%2Fjob%2FGPU%2520Cluster%2Fjob%2Fmain%2F&label=GPU)](https://heat-ci.fz-juelich.de/blue/organizations/jenkins/GPU%20Cluster%2Fmain/activity)
[![Documentation Status](https://readthedocs.org/projects/heat/badge/?version=latest)](https://heat.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/helmholtz-analytics/heat/branch/main/graph/badge.svg)](https://codecov.io/gh/helmholtz-analytics/heat)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/heat)](https://pepy.tech/project/heat)

Goals
-----

Heat is a flexible and seamless open-source software for high performance data
analytics and machine learning. It provides highly optimized algorithms and data
structures for tensor computations using CPUs, GPUs and distributed cluster
systems on top of MPI. The goal of Heat is to fill the gap between data
analytics and machine learning libraries with a strong focus on single-node
performance, and traditional high-performance computing (HPC). Heat's generic
Python-first programming interface integrates seamlessly with the existing data
science ecosystem and makes it as effortless as using numpy to write scalable
scientific and data science applications.

Heat allows you to tackle your actual Big Data challenges that go beyond the
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

Check out our Jupyter Notebook [tutorial](https://github.com/helmholtz-analytics/heat/blob/main/scripts/tutorial.ipynb)
right here on Github or in the /scripts directory.

The complete documentation of the latest version is always deployed on
[Read the Docs](https://heat.readthedocs.io/).

Support Channels
----------------

We use [StackOverflow](https://stackoverflow.com/tags/pyheat/) as a forum for questions about Heat.
If you do not find an answer to your question, then please ask a new question there and be sure to
tag it with "pyheat".

You can also reach us on [GitHub Discussions](https://github.com/helmholtz-analytics/heat/discussions).

Requirements
------------

Heat requires Python 3.7 or newer.
Heat is based on [PyTorch](https://pytorch.org/). Specifically, we are exploiting
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

**It is recommended to use the most recent supported version of PyTorch!**

It is also very important to ensure that the PyTorch version is compatible with the local CUDA installation.
More information can be found [here](https://pytorch.org/get-started/locally/).

Hacking
-------

If you want to work with the development version, you can check out the sources using

> $ git clone https://github.com/helmholtz-analytics/heat.git

The installation can then be done from the checked-out sources with

> $ pip install .[hdf5,netcdf,dev]

We welcome contributions from the community, please check out our [Contribution Guidelines](contributing.md) before getting started!

License
-------

Heat is distributed under the MIT license, see our
[LICENSE](LICENSE) file.

Citing Heat
-----------

If you find Heat helpful for your research, please mention it in your publications. You can cite:

- Götz, M., Debus, C., Coquelin, D., Krajsek, K., Comito, C., Knechtges, P., Hagemeier, B., Tarnawa, M., Hanselmann, S., Siggel, S., Basermann, A. & Streit, A. (2020). HeAT - a Distributed and GPU-accelerated Tensor Framework for Data Analytics. In 2020 IEEE International Conference on Big Data (Big Data) (pp. 276-287). IEEE, DOI: 10.1109/BigData50022.2020.9378050.

```
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
```

Acknowledgements
----------------

*This work is supported by the [Helmholtz Association Initiative and
Networking Fund](https://www.helmholtz.de/en/about_us/the_association/initiating_and_networking/)
under project number ZT-I-0003 and the Helmholtz AI platform grant.*

---

<div align="center">
  <a href="https://www.dlr.de/EN/Home/home_node.html"><img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/master/doc/images/dlr_logo.svg" height="50px" hspace="3%" vspace="20px"></a><a href="https://www.fz-juelich.de/portal/EN/Home/home_node.html"><img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/master/doc/images/fzj_logo.svg" height="50px" hspace="3%" vspace="20px"></a><a href="http://www.kit.edu/english/index.php"><img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/master/doc/images/kit_logo.svg" height="50px" hspace="3%" vspace="20px"></a><a href="https://www.helmholtz.de/en/"><img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/master/doc/images/helmholtz_logo.svg" height="50px" hspace="3%" vspace="20px"></a>
</div>
