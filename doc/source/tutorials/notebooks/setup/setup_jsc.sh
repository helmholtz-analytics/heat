#!/bin/bash

module --force purge
module load Stages/2026
module load GCC/14.3.0

module load OpenMPI/5.0.8 mpi4py heat #MPI
module --show-hidden load IPython ipyparallel # parallel usage on notebook
module load h5py zarr netcdf4-python # I/O
module load xarray scikit-learn matplotlib # interoperability
