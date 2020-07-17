Getting Started
===============

HeAT is a Python package for accelerated and distributed tensor computations. Internally, it is based on `PyTorch <https://pytorch.org/>`_. Consequently, all operating systems that support Python and PyTorch also support a HeAT installation. Currently, this list contains at least Linux, MacOS and Windows. However, most of our development is done under Linux and interoperability should therefore be optimal.

Prerequisites
-------------

Python
^^^^^^

HeAT requires Python 3.6 or greater, which is pre-installed by default on most Linux distributions. You can check your Python by running:

.. code-block:: bash

    python3 --version

If you do not have a recent installation on you system, you may want to upgrade it.

`Ubuntu <https://ubuntu.com/>`_/`Debian <https://www.debian.org/>`_/`Mint <https://www.linuxmint.com/>`_

.. code-block:: bash

    sudo apt-get update && sudo apt-get install python3

`Fedora <https://getfedora.org/>`_/`CentOS <https://www.centos.org/>`_/`RHEL <https://www.redhat.com/de/technologies/linux-platforms/enterprise-linux>`_

.. code-block:: bash

    sudo dnf update python3

If you have new administrator privileges on your system, because you are working on a cluster for example, make sure to check its *user guide*, the module system (``module spider python``) or get in touch with the administrators.

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

You can accelerate computations with HeAT in different ways. For GPU acceleration ensure that you have a `CUDA <https://developer.nvidia.com/cuda-zone>`_ installation on your system. Distributed computations require an MPI stack on you computer. We recommend `MVAPICH <https://mvapich.cse.ohio-state.edu/>`_ or `OpenMPI <https://www.open-mpi.org/>`_. Finally, for parallel data I/O, HeAT offers interface to `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ and `NetCDF <https://www.unidata.ucar.edu/software/netcdf/>`_. You can obtain these packages using your operating system's package manager.

Installation
------------

Virtual Environments
^^^^^^^^^^^^^^^^^^^^

We highly recommend to use `virtual environments (venv) <https://docs.python.org/3/tutorial/venv.html>`_ for managing your Python packages. A virtual environment is a self-contained directory tree for a particular Python version and its packages. It allows you not only to install packages without administrator privileges, install `pip <https://pypi.org/project/pip/>`_, Python's package manager, but also to have multiple package environments with different package versions in parallel.

You can find the complete manual for venv in the `Python documentation <https://docs.python.org/3/tutorial/venv.html>`_. Below is a small code snippet that creates a new virtual environment in your home directory (``~/.virtualenvs/heat``). The subsequent command enables the environment. You can access the Python interpreter by typing ``python`` and PIP with ``pip``.

.. code-block:: bash

    python3 -m venv ~/.virtualenvs/heatenv
    source ~/.virtualenvs/heatenv/bin/activate

You can deactivate a virtual environment by executing:

.. code-block:: bash

    deactivate

pip
^^^

Official HeAT releases are made available on the `Python Package Index (PyPI) <https://pypi.org/>`_. You obtain the latest version by running:

.. code-block:: bash

    pip install heat

Optionally, you can enable and install HDF5 and/or NetCDF support by adding the respective extra requirements as follows.

.. code-block:: bash

    pip install 'heat[hdf5, netcdf]'

Verification
^^^^^^^^^^^^

To ensure that HeAT was installed correctly, you can run this tiny code snippet that creates a vector with 42 entries.

.. code-block:: bash

    python -c "import heat as ht; print(ht.arange(10))"

You should see the following output

.. code-block:: bash

    DNDarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=ht.int32, device=cpu:0, split=None)

Building From Source
--------------------

For most users a HeAT installation from pip will be the most simple. However, if you want to test out the latest features or even want to contribute to HeAT, you will need to build from source. At first, clone our repository by running:

.. code-block:: bash

    git clone https://github.com/helmholtz-analytics/heat.git

Afterwards, change to the cloned source code directory and run the setup scripts.

.. code-block:: bash

  $ cd heat
  $ pip install -e '.[hdf5, netcdf]'
