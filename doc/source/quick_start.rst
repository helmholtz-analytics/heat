Heat Quick Start
================

No-frills instructions for `new
users <#new-users-condaconda-pippip-hpchpc-dockerdocker>`__ and `new
contributors <#new-contributors>`__.

New Users (`conda <#conda>`__, `pip <#pip>`__, `HPC <#hpc>`__, `Docker <#docker>`__)
------------------------------------------------------------------------------------

``conda``
~~~~~~~~~

The Heat conda build includes all dependencies including OpenMPI.

.. code:: shell

   conda create --name heat_env
   conda activate heat_env
   conda install -c conda-forge heat

`Test <#test>`__ your installation.

``pip``
~~~~~~~

Pre-requisite: MPI installation. We test with
`OpenMPI <https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/index.html>`__

Virtual environment and installation:

::

   python -m venv heat_env
   source heat_env/bin/activate
   pip install heat[hdf5,netcdf]

`Test <#test>`__ your installation.

HPC
~~~

Work in progress.

Docker
~~~~~~

Get the docker image from our package repository

::

   docker pull ghcr.io/helmholtz-analytics/heat:<version-tag>

or build it from our Dockerfile

::

   git clone https://github.com/helmholtz-analytics/heat.git
   cd heat/docker
   docker build --build-arg HEAT_VERSION=X.Y.Z --build-arg PYTORCH_IMG=<nvcr-tag> -t heat:X.Y.Z .

``<nvcr-tag>`` should be replaced with an existing version of the
official Nvidia pytorch container image. Information and existing tags
can be found on the
`here <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>`__

See `our docker
README <https://github.com/helmholtz-analytics/heat/tree/main/docker/README.md>`__
for other details.

Test
~~~~

In your terminal, test your setup with the
```heat_test.py`` <https://github.com/helmholtz-analytics/heat/blob/main/scripts/heat_test.py>`__
script:

::

   mpirun -n 2 python heat_test.py

It should print something like this:

::

   x is distributed:  True
   Global DNDarray x:  DNDarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=ht.int32, device=cpu:0, split=0)
   Global DNDarray x:
   Local torch tensor on rank  0 :  tensor([0, 1, 2, 3, 4], dtype=torch.int32)
   Local torch tensor on rank  1 :  tensor([5, 6, 7, 8, 9], dtype=torch.int32)

New Contributors
----------------

1. Pick an Issue you’d like to work on. Check out `Good First
   Issues <https://github.com/helmholtz-analytics/heat/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22>`__,
   start from the most recent. Get in touch and ask to be assigned to
   the issue.

2. **IMPORTANT:** As soon as an issue is assigned, a new branch will be
   created (a comment will be posted under the relevant issue). Do use
   this branch to make your changes, it has been checked out from the
   correct source branch (i.e. ``main`` for new features, ``release/*``
   for bug fixes).

3. `Fork <https://docs.github.com/en/get-started/quickstart/contributing-to-projects>`__
   or, if you have write access, clone the `Heat
   repository <https://github.com/helmholtz-analytics/heat>`__.

4. **Setting up a dev-environment with CONDA:** Create a virtual
   environment ``heat_dev`` with all dependencies via
   `scripts/heat_dev.yml <https://github.com/helmholtz-analytics/heat/blob/main/scripts/heat_dev.yml>`__.
   Note that ``scripts/heat_dev.yml`` does not install Heat.

   ::

      conda env create -f scripts/heat_dev.yml
      conda activate heat_dev

   Note that in case you want to use a GPU while developing on your
   local machine, you need to set up a CUDA environment by using
   ``scripts/heat_dev_cuda11.yml``\ for CUDA 11 or
   ``scripts/heat_dev_cuda12.yml``\ for CUDA 12, respectively, instead
   of ``scripts/heat_dev.yml``.

   **Setting up a dev-environment with PIP:** Create a virtual
   environment ``heatenv`` with
   ``python -m venv <path_to_store_venvs>/heatenv``, and activate it by
   ``source <path_to_store_venvs>/heatenv/bin/activate``. Then clone the
   Heat-repo from GitHub by

   ::

      git clone https://github.com/helmholtz-analytics/heat.git

   go to the Heat-folder (``cd heat``), and install (in editable fashion
   “``-e``”) by

   ::

      pip install -e '.[hdf5, netcdf]'

5. In the ``/heat`` directory of your local repo, install the
   `pre-commit hooks <https://pre-commit.com/>`__:

   ::

      cd $MY_REPO_DIR/heat/
      pre-commit install

6. Write and run (locally) `unit
   tests <https://docs.python.org/3/library/unittest.html>`__ for any
   change you introduce. Here’s a sample of our `test
   modules <https://github.com/helmholtz-analytics/heat/tree/main/heat/core/tests>`__.

   Running all unit tests locally, e.g. on 3 processes:

   ::

      mpirun -n 3 python -m unittest

   Testing one module only, e.g. ``manipulations``:

   ::

      mpirun -n 3 python -m unittest heat/core/tests/test_manipulations.py

   Testing one function within a module,
   e.g. ``manipulations.concatenate``:

   ::

      mpirun -n 3 python -m unittest heat.core.tests.test_manipulations.TestManipulations.test_concatenate

   Testing with CUDA (if available):

   ::

      export HEAT_TEST_USE_DEVICE=gpu
      mpirun -n 3 python -m unittest

   Helpful options for debugging:

   ::

      mpirun --tag-output -n 3 python -m unittest -vf

7. After `making and
   pushing <https://docs.github.com/en/get-started/quickstart/contributing-to-projects#making-and-pushing-changes>`__
   your changes, go ahead and `create a Pull
   Request <https://docs.github.com/en/get-started/quickstart/contributing-to-projects#making-a-pull-request>`__.
   Make sure you go through the Due Diligence checklist (part of our PR
   template). Consider `allowing us to edit your
   branch <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/allowing-changes-to-a-pull-request-branch-created-from-a-fork#enabling-repository-maintainer-permissions-on-existing-pull-requests>`__
   for a smoother review process.

   ## Thank you so much for your time!
