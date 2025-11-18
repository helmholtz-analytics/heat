#!/bin/bash

module --force purge
module load Stages/2025

ml GCC ParaStationMPI heat h5py

ml IPython ipyparallel/.9.0.0

# needed for interoperability examples
ml xarray scikit-learn matplotlib
