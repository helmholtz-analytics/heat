#!/bin/bash

module --force purge
module load Stages/2026
module load GCC

module load ParaStationMPI mpi4py heat #MPI
module load IPython ipyparallel # parallel usage on notebook
module load h5py zarr netcdf4-python # I/O
module load xarray scikit-learn matplotlib # interoperability
