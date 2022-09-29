<div align="center">
  <img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/main/doc/images/logo.png">
</div>


## About

Heat is a flexible and seamless open-source software for high performance data analytics and machine learning. It provides highly optimized algorithms and data structures for tensor computations using CPUs, GPUs and distributed cluster systems on top of MPI. The goal of Heat is to fill the gap between data analytics and machine learning libraries with a strong focus on single-node performance, and traditional high-performance computing (HPC). Heat's generic Python-first programming interface integrates seamlessly with the existing data science ecosystem and makes it as effortless as using numpy to write scalable scientific and data science applications.

Heat allows you to tackle your actual Big Data challenges that go beyond the computational and memory needs of your laptop and desktop.

## Features

* High-performance n-dimensional tensors
* CPU, GPU and distributed computation using MPI
* Powerful data analytics and machine learning methods
* Abstracted communication via split tensors
* Python API


## Installation

Tagged releases are made available on the
[Python Package Index (PyPI)](https://pypi.org/project/heat/). You can typically
install the latest version with

> $ pip install heat[hdf5,netcdf]

where the part in brackets is a list of optional dependencies. You can omit
it, if you do not need HDF5 or NetCDF support.

**It is recommended to use the most recent supported version of PyTorch!**

It is also very important to ensure that the PyTorch version is compatible with the local CUDA installation.
More information can be found [here](https://pytorch.org/get-started/locally/).


## Getting Started

Check out our Jupyter Notebook [tutorial](https://github.com/helmholtz-analytics/heat/blob/main/scripts/tutorial.ipynb)
right here on Github or in the /scripts directory.

The complete documentation of the latest version is always deployed on
[Read the Docs](https://heat.readthedocs.io/).
