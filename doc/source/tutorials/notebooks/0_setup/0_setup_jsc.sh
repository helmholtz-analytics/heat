#!/bin/bash

module --force purge
module load Stages/2025

ml GCC ParaStationMPI heat

ml IPython ipyparallel/.9.0.0


