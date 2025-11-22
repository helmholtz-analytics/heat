#!/bin/bash

module purge
module load Stages/2025
module load GCC
module load Python
module load jupyter-server  # provides ipykernel

module load ParaStationMPI mpi4py #MPI
module load IPython ipyparallel/.9.0.0 # parallel usage on notebook
module load PyTorch torchvision
module load h5py zarr netCDF # I/O
module load xarray scikit-learn matplotlib # interoperability
module load SciPy-bundle Python-bundle-PyPI py-cpuinfo # perun deps
module load heat



# module -- purge
# module load Stages/2025

# ml GCC ParaStationMPI heat h5py

# ml IPython ipyparallel/.9.0.0

# # needed for interoperability examples
# ml xarray scikit-learn matplotlib
