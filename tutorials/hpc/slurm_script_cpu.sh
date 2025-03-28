#!/bin/bash

#SBATCH --partition=<partition_name>
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time="00:01:00"

export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python ~/heat/tutorials/hpc/01_basics/01_basics_dndarrays.py
