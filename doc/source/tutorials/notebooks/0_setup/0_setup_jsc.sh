#!/bin/bash

module --force purge
module load Stages/2025
module load GCC

module load ParaStationMPI mpi4py heat #MPI
module load IPython ipyparallel/.9.0.0 # parallel usage on notebook
module load h5py zarr netcdf4-python # I/O
module load xarray scikit-learn matplotlib # interoperability



# module -- purge
# module load Stages/2025

# ml GCC ParaStationMPI heat h5py

# ml IPython ipyparallel/.9.0.0

# # needed for interoperability examples
# ml xarray scikit-learn matplotlib
