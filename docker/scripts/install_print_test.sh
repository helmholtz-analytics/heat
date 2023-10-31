#!/bin/bash
# Scripts to quickly obtain all relevant information out of a new nvidia pytorch container.

# Container setup
apt update && DEBIAN_FRONTEND=noninteractive apt install -y build-essential openssh-client python3-dev git && apt clean && rm -rf /var/lib/apt/lists/*

# Setup heat dependencies
git clone https://github.com/helmholtz-analytics/heat.git
cd heat
pip install --upgrade pip
pip install mpi4py --no-binary :all:
pip install .[netcdf,hdf5,dev]

# Print environment
pip list | grep heat
pip list | grep torch
python --version
nvcc --version
mpirun --version

# Run tests
HEAT_TEST_USE_DEVICE=gpu mpirun -n 1 pytest heat/
