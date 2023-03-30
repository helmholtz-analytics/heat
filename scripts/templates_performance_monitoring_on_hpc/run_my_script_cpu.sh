#!/bin/bash

#SBATCH --account=haf
#SBATCH -o scaling_test_cpu_%j.txt

echo "job" $SLURM_JOB_ID "nodes:" $SLURM_JOB_NUM_NODES "MPI-procs:" $SLURM_NTASKS "CPUs per MPI-proc:" $SLURM_CPUS_PER_TASK "OMP/MKL-threads per MPI-proc:" $OMP_NUM_THREADS "/" $MKL_NUM_THREADS

srun --cpu-bind=threads python my_script.py
srun max_mem.sh
