# Quick Start

This guide provides instructions for setting up Heat as a user or a contributor.

## New users

### Package managers (conda, mamba, pixi, etc.)
The Heat conda-forge build includes all necessary dependencies, including OpenMPI.

```shell
conda create --name heat_env
conda activate heat_env
conda install -c conda-forge heat
```

### Standard Python (pip)

The PyPI build includes all Python dependencies. It requires an MPI implementation to be installed on your system. Heat is typically tested against OpenMPI.

```bash
python -m venv heat_env
source heat_env/bin/activate
pip install heat[hdf5,netcdf,zarr]
```

### HPC environments

Heat is a native HPC library designed to run on large-scale clusters. For production environments, it is recommended to use HPC-native package managers, however `pip` environments and container runtimes are also an option.

#### EasyBuild & Spack

Heat provides configurations for common HPC software stack managers. Refer to the specific repository for the most recent recipes:

- EasyBuild: Search for the `heat` software block.

- Spack: Use `spack install py-heat`.

- On JSC systems, Heat is available as a module:

```bash
module load GCC OpenMPI heat
```

#### Docker

See our [docker README](https://github.com/helmholtz-analytics/heat/blob/main/docker/README.md) on GitHub.

#### Verification

Confirm the installation by running a distributed smoke test to verify tensor splitting across processes:

```bash
mpirun -n 2 python -c "import heat as ht; x = ht.arange(10, split=0); print(f'Rank {ht.communication.MPI_SELF.rank} local shape: {x.lshape}')"
```
This should output something like:

```
Rank 0 local shape: (5,)
Rank 1 local shape: (5,)
```

indicating that the tensor was successfully split across the two processes.

----

## New contributors

### Development environment setup

Contributors should install Heat in editable mode to ensure code changes are reflected immediately. Use one of the following methods to establish a development environment.

#### Method A: automated dependency management (conda / mamba / pixi)

For automated handling of MPI and CUDA/ROCm toolkits.

1. Fork the Heat repository and clone it locally (or, if you have write access, clone the main repository).

2. Go to the root of the cloned repository and create a new conda environment:

```bash
cd heat
conda env create -f scripts/heat_dev.yml
conda activate heat_dev
```

3. Install Heat in editable mode:

```bash
pip install -e '.[hdf5, netcdf, zarr, dev]'
```

#### Method B: manual dependency management (pip)

Prerequisite: a functional MPI installation already exists on your system.

1. Fork the Heat repository and clone it locally (or, if you have write access, clone the main repository).

2. Create a virtual environment and activate it:

```bash
python -m venv heat_dev
source heat_dev/bin/activate
```

3. Install Heat in editable mode:

```bash
pip install -e '.[hdf5, netcdf, zarr, dev]'
```

### Repository syncing
If you cloned a fork, add the main repository as remote for synchronizing in the future using:

```bash
git remote add upstream https://github.com/helmholtz-analytics/heat.git
```

### Quality Control

Install pre-commit hooks to enforce coding standards locally before pushing changes:

```bash
pre-commit install
```


## All set!

For more details on our workflow, see our [contributing guidelines](./CONTRIBUTING.md). We look forward to your contributions!
