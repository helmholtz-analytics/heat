<div align="center">
  <img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/main/doc/images/logo.png">
</div>

---

Heat is a distributed tensor framework for high performance data analytics.

# Project Status

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2531472.svg)](https://doi.org/10.5281/zenodo.2531472)
[![CI workflow](https://github.com/helmholtz-analytics/heat/actions/workflows/ci_cb.yml/badge.svg)](https://github.com/helmholtz-analytics/heat/actions/workflows/ci_cb.yml)
[![Documentation Status](https://readthedocs.org/projects/heat/badge/?version=latest)](https://heat.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/helmholtz-analytics/heat/branch/main/graph/badge.svg)](https://codecov.io/gh/helmholtz-analytics/heat)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/heat)](https://pepy.tech/project/heat)

# New
- **NO GSoC WITH US THIS YEAR**. On Feb 21st we learned that our Organisation has not been selected this year. Congratulations to the [GSoC 2023 Mentoring Orgs](https://summerofcode.withgoogle.com/programs/2023/organizations)!

  Thanks everybody for your interest and contributions! 🙏  We will still review every PR. Remember to add your full name to your GitHub profile, so you'll appear as recognizable co-author with the next release.

  We wish everybody good luck with the program! Let us know how it went...

<!-- - SAVE THE DATE 📆 On Feb 23rd, Google Summer of Code Community Talk with Stephanie Taylor, Google Summer of Code Program Manager - [sign up here](https://docs.google.com/forms/d/e/1FAIpQLSdsXkVhHY54cYCnYfhMe0p42R3WB0uqIptnqdduhsP_zXBp8Q/viewform) by Feb 22nd if you want to attend -->
<!-- - We've submitted our application to be part of Google Summer of Code 2023 🙏
   - [List of Ideas](https://docs.google.com/document/d/1OKUf_n_whN0M20c__jj8R5jyGABPYYxmdBfyzto4OZk) (subject to change)
   - [Related Issues](https://github.com/helmholtz-analytics/heat/issues?q=is%3Aissue+is%3Aopen+label%3AGSoC2023) (subject to change)
   - [Good First Issues](https://github.com/helmholtz-analytics/heat/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) (updated frequently)
   - [Quick Start](quick_start.md) for new users and contributors -->

# Goals

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

# Features

* High-performance n-dimensional tensors
* CPU, GPU and distributed computation using MPI
* Powerful data analytics and machine learning methods
* Abstracted communication via split tensors
* Python API

# Support Channels

We use [StackOverflow](https://stackoverflow.com/tags/pyheat/) as a forum for questions about Heat.
If you do not find an answer to your question, then please ask a new question there and be sure to
tag it with "pyheat".

You can also reach us on [GitHub Discussions](https://github.com/helmholtz-analytics/heat/discussions).

# Requirements

Heat requires Python 3.7 or newer.
Heat is based on [PyTorch](https://pytorch.org/). Specifically, we are exploiting
PyTorch's support for GPUs *and* MPI parallelism. For MPI support we utilize
[mpi4py](https://mpi4py.readthedocs.io). Both packages can be installed via pip
or automatically using the setup.py.

# Installation

Tagged releases are made available on the
[Python Package Index (PyPI)](https://pypi.org/project/heat/). You can typically
install the latest version with

```
$ pip install heat[hdf5,netcdf]
```

where the part in brackets is a list of optional dependencies. You can omit
it, if you do not need HDF5 or NetCDF support.

**It is recommended to use the most recent supported version of PyTorch!**

**It is also very important to ensure that the PyTorch version is compatible with the local CUDA installation.**
More information can be found [here](https://pytorch.org/get-started/locally/).

# Hacking

If you want to work with the development version, you can check out the sources using

```
$ git clone <https://github.com/helmholtz-analytics/heat.git>
```

The installation can then be done from the checked-out sources with

```
$ pip install heat[hdf5,netcdf,dev]
```

# Getting Started

TL;DR: [Quick Start](quick_start.md) (Read this to get a quick overview of Heat).

Check out our Jupyter Notebook [**Tutorial**](https://github.com/helmholtz-analytics/heat/blob/main/scripts/)
right here on Github or in the /scripts directory, to learn and understand about the basics and working of Heat.

The complete documentation of the latest version is always deployed on
[Read the Docs](https://heat.readthedocs.io/).

***Try your first Heat program***

```shell
$ python
```

```python
>>> import heat as ht
>>> x = ht.arange(10,split=0)
>>> print(x)
DNDarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=ht.int32, device=cpu:0, split=0)
>>> y = ht.ones(10,split=0)
>>> print(y)
DNDarray([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=ht.float32, device=cpu:0, split=0)
>>> print(x + y)
DNDarray([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.], dtype=ht.float32, device=cpu:0, split=0)
```

### Also, you can test your setup by running the [`heat_test.py`](https://github.com/helmholtz-analytics/heat/blob/main/scripts/heat_test.py) script:

```shell
mpirun -n 2 python heat_test.py
```

### It should print something like this:

```shell
x is distributed:  True
Global DNDarray x:  DNDarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=ht.int32, device=cpu:0, split=0)
Global DNDarray x:
Local torch tensor on rank  0 :  tensor([0, 1, 2, 3, 4], dtype=torch.int32)
Local torch tensor on rank  1 :  tensor([5, 6, 7, 8, 9], dtype=torch.int32)
```

## Resources:

* [Heat Tutorials](https://heat.readthedocs.io/en/latest/tutorials.html)
* [Heat API Reference](https://heat.readthedocs.io/en/latest/autoapi/index.html)

### Parallel Computing and MPI:

* @davidhenty's [course](https://www.archer2.ac.uk/training/courses/200514-mpi/)
* Wes Kendall's [Tutorials](https://mpitutorial.com/tutorials/)

### mpi4py

* [mpi4py docs](https://mpi4py.readthedocs.io/en/stable/tutorial.html)
* [Tutorial](https://www.kth.se/blogs/pdc/2019/08/parallel-programming-in-python-mpi4py-part-1/)

# Contribution guidelines

**We welcome contributions from the community, if you want to contribute to Heat, be sure to review the [Contribution Guidelines](contributing.md) before getting started!**

We use [GitHub issues](https://github.com/helmholtz-analytics/heat/issues) for tracking requests and bugs, please see [Discussions](https://github.com/helmholtz-analytics/heat/discussions) for general questions and discussion, and You can also get in touch with us on [Mattermost](https://mattermost.hzdr.de/signup_user_complete/?id=3sixwk9okpbzpjyfrhen5jpqfo). You can sign up with your GitHub credentials. Once you log in, you can introduce yourself on the `Town Square` channel.

Small improvements or fixes are always appreciated; issues labeled as **"good first issue"** may be a good starting point.

If you’re unsure where to start or how your skills fit in, reach out! You can ask us here on GitHub, by leaving a comment on a relevant issue that is already open.

**If you are new to contributing to open source, [this guide](https://opensource.guide/how-to-contribute/) helps explain why, what, and how to get involved.**

# License

Heat is distributed under the MIT license, see our
[LICENSE](LICENSE) file.

# Citing Heat

If you find Heat helpful for your research, please mention it in your publications. You can cite:

* Götz, M., Debus, C., Coquelin, D., Krajsek, K., Comito, C., Knechtges, P., Hagemeier, B., Tarnawa, M., Hanselmann, S., Siggel, S., Basermann, A. & Streit, A. (2020). HeAT - a Distributed and GPU-accelerated Tensor Framework for Data Analytics. In 2020 IEEE International Conference on Big Data (Big Data) (pp. 276-287). IEEE, DOI: 10.1109/BigData50022.2020.9378050.

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

## Acknowledgements

*This work is supported by the [Helmholtz Association Initiative and
Networking Fund](https://www.helmholtz.de/en/about_us/the_association/initiating_and_networking/)
under project number ZT-I-0003 and the Helmholtz AI platform grant.*

---

<div align="center">
  <a href="https://www.dlr.de/EN/Home/home_node.html"><img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/master/doc/images/dlr_logo.svg" height="50px" hspace="3%" vspace="20px"></a><a href="https://www.fz-juelich.de/portal/EN/Home/home_node.html"><img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/master/doc/images/fzj_logo.svg" height="50px" hspace="3%" vspace="20px"></a><a href="http://www.kit.edu/english/index.php"><img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/master/doc/images/kit_logo.svg" height="50px" hspace="3%" vspace="20px"></a><a href="https://www.helmholtz.de/en/"><img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/master/doc/images/helmholtz_logo.svg" height="50px" hspace="3%" vspace="20px"></a>
</div>
