#!/bin/bash
# Scripts to quickly obtain all relevant information out of a new nvidia pytorch container. Run it inside a pytorch container from nvidia and it will first print the software stack (cuda version, torch version, ...), install heat from source, and run the heat unit tests. Usefull to quickly check if a container is compatible with heat.

# Container setup
apt update && DEBIAN_FRONTEND=noninteractive apt install -y build-essential openssh-client python3-dev git && apt clean && rm -rf /var/lib/apt/lists/*

# Print environment
pip list | grep torch
python --version
nvcc --version
mpirun --version

# Install heat from source.
git clone https://github.com/helmholtz-analytics/heat.git
cd heat
pip install --upgrade pip
pip install mpi4py --no-binary :all:
pip install .[netcdf,hdf5,dev]

# Run tests
HEAT_TEST_USE_DEVICE=gpu mpirun -n 1 pytest heat/
