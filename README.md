<div align="center">
  <img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/main/doc/images/logo.png">
</div>

---

Heat is a distributed tensor framework for high performance data analytics.

# Project Status

[![CPU/CUDA/ROCm tests](https://codebase.helmholtz.cloud/helmholtz-analytics/ci/badges/heat/base/pipeline.svg)](https://codebase.helmholtz.cloud/helmholtz-analytics/ci/-/commits/heat/base)
[![Documentation Status](https://readthedocs.org/projects/heat/badge/?version=latest)](https://heat.readthedocs.io/en/latest/?badge=latest)
[![coverage](https://codecov.io/gh/helmholtz-analytics/heat/branch/main/graph/badge.svg)](https://codecov.io/gh/helmholtz-analytics/heat)
[![license: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI Version](https://img.shields.io/pypi/v/heat)](https://pypi.org/project/heat/)
[![Downloads](https://pepy.tech/badge/heat)](https://pepy.tech/project/heat)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu)
[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/7688/badge)](https://bestpractices.coreinfrastructure.org/projects/7688)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2531472.svg)](https://doi.org/10.5281/zenodo.2531472)
[![Benchmarks](https://img.shields.io/badge/Github--Pages-Benchmarks-2ea44f)](https://helmholtz-analytics.github.io/heat/dev/bench)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Table of contents:
 COPILOT WILL FILL OUT AT THE END

# What is Heat for?

Heat builds on [PyTorch](https://pytorch.org/) and [mpi4py](https://mpi4py.readthedocs.io) to provide high-performance computing infrastructure for memory-intensive applications within the NumPy/SciPy ecosystem.


With Heat you can:
- port existing NumPy/SciPy code from single-CPU to multi-node clusters with minimal coding effort;
- exploit the entire, cumulative RAM of your many nodes for memory-intensive operations and algorithms;
- run your NumPy/SciPy code on GPUs (CUDA, ROCm, coming up: Apple MPS).

Check out our [coverage tables](coverage_tables.md) to see which NumPy, SciPy, scikit-learn functions are already supported.

 If you need a functionality that is not yet supported:
  - [search existing issues](https://github.com/helmholtz-analytics/heat/issues) and make sure to comment if someone else already requested it;
  - [open a new issue](https://github.com/helmholtz-analytics/heat/issues/new/choose).


Check out our [features](#features) and the [Heat API Reference](https://heat.readthedocs.io/en/latest/autoapi/index.html) for a complete list of functionalities.

# Features

* High-performance n-dimensional arrays
* CPU, GPU, and distributed computation using MPI
* Powerful data analytics and machine learning methods
* Seamless integration with the NumPy/SciPy ecosystem
* Python array API (work in progress)

# Installation

## Requirements

### Basics
- python >= 3.8
- MPI (OpenMPI, MPICH, Intel MPI, etc.)
- mpi4py >= 3.0.0
- pytorch >= 1.8.0

### Parallel I/O
- h5py
- netCDF4

### pip
Install the latest version with

```bash
pip install heat[hdf5,netcdf]
```
where the part in brackets is a list of optional dependencies. You can omit
it, if you do not need HDF5 or NetCDF support.

### **conda**

The conda build includes all dependencies **including OpenMPI**.
```bash
 conda install -c conda-forge heat
 ```

# Getting Started

Go to [Quick Start](quick_start.md) for a quick overview.

Check out our Jupyter Notebook [**Tutorial**](https://github.com/helmholtz-analytics/heat/blob/main/scripts/)
right here on GitHub or in the `./scripts` directory, to learn and understand Heat's basics.

The complete documentation of the latest version is always deployed on
[Read the Docs](https://heat.readthedocs.io/).

### You can test your setup by running the [`heat_test.py`](https://github.com/helmholtz-analytics/heat/blob/main/scripts/heat_test.py) script:

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


# FAQ
In progress...

  - Users
  - Developers
  - Students
  - system administrators

<!-- # Goals

Heat is a flexible and seamless open-source software for high performance data
analytics and machine learning. It provides highly optimized algorithms and data structures for tensor computations using CPUs, GPUs, and distributed cluster systems on top of MPI. The goal of Heat is to fill the gap between single-node data analytics and machine learning libraries, and  high-performance computing (HPC). Heat's interface integrates seamlessly with the existing data science ecosystem and makes  writing scalable
scientific and data science applications as effortless as using NumPy.

Heat allows you to tackle your actual Big Data challenges that go beyond the
computational and memory needs of your laptop and desktop.
 -->

# Support Channels

We use [GitHub Discussions](https://github.com/helmholtz-analytics/heat/discussions) as a forum for questions about Heat.
If you found a bug or miss a feature, then please file a new [issue](https://github.com/helmholtz-analytics/heat/issues/new/choose).


# Contribution guidelines

**We welcome contributions from the community, if you want to contribute to Heat, be sure to review the [Contribution Guidelines](contributing.md) and [Resources for MPI programming](#resources-for-mpi-programming)  before getting started!**

We use [GitHub issues](https://github.com/helmholtz-analytics/heat/issues) for tracking requests and bugs, please see [Discussions](https://github.com/helmholtz-analytics/heat/discussions) for general questions and discussion. You can also get in touch with us on [Mattermost](https://mattermost.hzdr.de/signup_user_complete/?id=3sixwk9okpbzpjyfrhen5jpqfo) (sign up with your GitHub credentials). Once you log in, you can introduce yourself on the `Town Square` channel.

If you’re unsure where to start or how your skills fit in, reach out! You can ask us here on GitHub, by leaving a comment on a relevant issue that is already open.

**If you are new to contributing to open source, [this guide](https://opensource.guide/how-to-contribute/) helps explain why, what, and how to get involved.**


## Resources for MPI programming

* [Heat Tutorials](https://heat.readthedocs.io/en/latest/tutorials.html)
* [Heat API Reference](https://heat.readthedocs.io/en/latest/autoapi/index.html)

### Parallel Computing and MPI:

* @davidhenty's [course](https://www.archer2.ac.uk/training/courses/200514-mpi/)
* Wes Kendall's [Tutorials](https://mpitutorial.com/tutorials/)

### mpi4py

* [mpi4py docs](https://mpi4py.readthedocs.io/en/stable/tutorial.html)
* [Tutorial](https://www.kth.se/blogs/pdc/2019/08/parallel-programming-in-python-mpi4py-part-1/)
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

*This project has received funding from Google Summer of Code (GSoC) in 2022.*


---

<div align="center">
  <a href="https://www.dlr.de/EN/Home/home_node.html"><img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/master/doc/images/dlr_logo.svg" height="50px" hspace="3%" vspace="20px"></a><a href="https://www.fz-juelich.de/portal/EN/Home/home_node.html"><img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/master/doc/images/fzj_logo.svg" height="50px" hspace="3%" vspace="20px"></a><a href="http://www.kit.edu/english/index.php"><img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/master/doc/images/kit_logo.svg" height="50px" hspace="3%" vspace="20px"></a><a href="https://www.helmholtz.de/en/"><img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/master/doc/images/helmholtz_logo.svg" height="50px" hspace="3%" vspace="20px"></a>
</div>
