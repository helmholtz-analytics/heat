# Heat Quick Start

No-frills instructions for [new users](#new-users-condaconda-pippip-hpchpc-dockerdocker) and [new contributors](#new-contributors).

## New Users ([conda](#conda), [pip](#pip), [HPC](#hpc), [Docker](#docker))

### `conda`
A Heat conda build is [in progress](https://github.com/helmholtz-analytics/heat/issues/1050).
The script [heat_env.yml](https://github.com/helmholtz-analytics/heat/blob/main/scripts/heat_env.yml):
- creates a virtual environment `heat_env`
- installs all dependencies including OpenMPI using [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)
- installs Heat via `pip`

```
conda env create -f heat_env.yml
conda activate heat_env
```

[Test](#test) your installation.

### `pip`

Pre-requisite: MPI installation. We test with [OpenMPI](https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/index.html)

Virtual environment and installation:
```
python -m venv heat_env
source heat_env/bin/activate
pip install heat[hdf5,netcdf]
```
[Test](#test) your installation.

### HPC
Work in progress...

### Docker
Work in progress ([PR 970](https://github.com/helmholtz-analytics/heat/pull/970))

### Test
In your terminal, test your setup with the [`heat_test.py`](https://github.com/helmholtz-analytics/heat/blob/main/scripts/heat_test.py) script:

```
mpirun -n 2 python heat_test.py
```

It should print something like this:
```
x is distributed:  True
Global DNDarray x:  DNDarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=ht.int32, device=cpu:0, split=0)
Global DNDarray x:
Local torch tensor on rank  0 :  tensor([0, 1, 2, 3, 4], dtype=torch.int32)
Local torch tensor on rank  1 :  tensor([5, 6, 7, 8, 9], dtype=torch.int32)
```

## New Contributors

1. [Fork](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) or, if you have write access, clone the [Heat repository](https://github.com/helmholtz-analytics/heat).

2. Create a virtual environment `heat_dev` with all dependencies via [heat_dev.yml](https://github.com/helmholtz-analytics/heat/blob/main/scripts/heat_dev.yml). Note that `heat_dev.yml` does not install Heat via `pip` (as opposed to [`heat_env.yml`](#conda) for users).

    ```
    conda env create -f heat_dev.yml
    conda activate heat_dev
    ```


3. In the `/heat` directory of your local repo, install the [pre-commit hooks]( https://pre-commit.com/):

    ```
    cd $MY_REPO_DIR/heat/
    pre-commit install
    ```

4. Pick an Issue you'd like to work on. Check out [Good First Issues](https://github.com/helmholtz-analytics/heat/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22), start from most recent.

5. [New branches](https://docs.github.com/en/get-started/quickstart/contributing-to-projects#creating-a-branch-to-work-on) should be named according to the following scheme:
   - New feature: `features/ISSUE_NUMBER-my-new-feature`
   - Bug fix: `bugs/ISSUE_NUMBER-my-bug-fix`
   - Documentation: `docs/ISSUE_NUMBER-my-better-docs`
   - Automation (CI, GitHub Actions etc.): `workflows/ISSUE_NUMBER-my-fancy-workflow`

6. Write and run (locally) [unit tests](https://docs.python.org/3/library/unittest.html) for any change you introduce. Here's a sample of our [test modules](https://github.com/helmholtz-analytics/heat/tree/main/heat/core/tests).

    Running all unit tests locally, e.g. on 3 processes:

    ```
    mpirun -n 3 python -m unittest
    ```
    Testing one module only, e.g. `manipulations`:

    ```
    mpirun -n 3 python -m unittest heat/core/tests/test_manipulations.py
    ```

    Testing one function within a module, e.g. `manipulations.concatenate`:

    ```
    mpirun -n 3 python -m unittest heat.core.tests.test_manipulations.TestManipulations.test_concatenate
    ```

    Testing with CUDA (if available):

    ```
    export HEAT_TEST_USE_DEVICE=gpu
    mpirun -n 3 python -m unittest
    ```

    Helpful options for debugging:

    ```
    mpirun --tag-output -n 3 python -m unittest -vf
    ```


7. After [making and pushing](https://docs.github.com/en/get-started/quickstart/contributing-to-projects#making-and-pushing-changes) your changes, go ahead and [create a Pull Request](https://docs.github.com/en/get-started/quickstart/contributing-to-projects#making-a-pull-request). Make sure you go through the Due Diligence checklist (part of our PR template).


    ## Thank you so much for your time!
