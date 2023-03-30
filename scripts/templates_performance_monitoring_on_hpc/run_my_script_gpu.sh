#!/bin/bash
#SBATCH --account=haf
#SBATCH -o scaling_test_gpu_%j.txt

#export NUM_CORES = 1
#export OMP_NUM_THREADS = $NUM_CORES
#export MKL_NUM_THREADS = $NUM_CORES

echo "job" $SLURM_JOB_ID "nodes:" $SLURM_JOB_NUM_NODES "MPI-procs:" $SLURM_NTASKS

srun python my_script.py
srun max_mem.sh
