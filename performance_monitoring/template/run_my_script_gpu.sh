#!/bin/bash
#SBATCH --account=haf
#SBATCH -e scaling_test_errors_gpu.txt
#SBATCH -o scaling_test_output_gpu.txt
#SBATCH --open-mode=append

#export NUM_CORES = 1
#export OMP_NUM_THREADS = $NUM_CORES
#export MKL_NUM_THREADS = $NUM_CORES

echo "job" $SLURM_JOB_ID "nodes:" $SLURM_JOB_NUM_NODES "MPI-procs:" $SLURM_NTASKS

srun python my_script.py
