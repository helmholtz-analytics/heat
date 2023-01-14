## Heat Quick Start

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

1. Clone the [Heat repository](https://github.com/helmholtz-analytics/heat).
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

5. New branches should be named according to the following scheme:
   - New feature: `features/ISSUE_NUMBER-my-new-feature`
   - Bug fix: `bugs/ISSUE_NUMBER-my-bug-fix`
   - Documentation: `docs/ISSUE_NUMBER-my-better-docs`
   - Automation (CI, GitHub Actions etc.): `workflows/ISSUE_NUMBER-my-fancy-workflow`

6. After making your changes, go ahead create a Pull Request so we can review them. Thank you so much!
