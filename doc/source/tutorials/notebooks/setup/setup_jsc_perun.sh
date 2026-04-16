#!/bin/bash

module --force purge
module load Stages/2025
module load GCC
module load Python
module load jupyter-server  # provides ipykernel

module load OpenMPI mpi4py #MPI
module load IPython  # parallel usage on notebook
module load heat
module load h5py zarr netcdf4-python # I/O
module load xarray scikit-learn matplotlib # interoperability
module load SciPy-bundle Python-bundle-PyPI py-cpuinfo # perun deps

# shellcheck disable=SC1091
source /p/project1/training2546/comito1/jupyter/kernels/perun_notebook/bin/activate
