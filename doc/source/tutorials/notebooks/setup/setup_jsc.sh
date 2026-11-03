#!/bin/bash

module --force purge
module load Stages/2026
module load GCC/14.3.0

module load OpenMPI/5.0.8 mpi4py/4.1.0 PyTorch/2.9.1 torchvision/0.24.1 #MPI
module --show-hidden load IPython ipyparallel # parallel usage on notebook
module load h5py zarr netcdf4-python # I/O
module load xarray scikit-learn matplotlib # interoperability

# Create the environment if needed
if [[ ! -d .venv ]]; then
            python -m venv .venv
            echo "Created new venv: $(which python)"
fi

# Activate it in this shell
source .venv/bin/activate

module load git

pip install --upgrade pip
pip install --upgrade packaging

# Install/update latest version of Heat from GitHub
python -m pip install --no-deps --upgrade "git+https://github.com/helmholtz-analytics/heat.git"

echo "Environment ready: $(which python)"
