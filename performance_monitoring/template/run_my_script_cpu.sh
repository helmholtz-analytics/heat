#!/bin/bash
#SBATCH --account=haf
#SBATCH -e scaling_test_errors_cpu.txt
#SBATCH -o scaling_test_output_cpu.txt
#SBATCH --open-mode=appends

echo "job" $SLURM_JOB_ID "nodes:" $SLURM_JOB_NUM_NODES "MPI-procs:" $SLURM_NTASKS "CPUs per MPI-proc:" $SLURM_CPUS_PER_TASK "OMP/MKL-threads per MPI-proc:" $OMP_NUM_THREADS "/" $MKL_NUM_THREADS

srun --cpu-bind=threads python my_script.py
