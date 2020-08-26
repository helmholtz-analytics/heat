#!/bin/bash

#SBATCH --exclusive
#SBATCH -A haf
#SBATCH --cpus-per-task 6
#SBATCH -t 04:00:00
#SBATCH -J Heat_DPNN_Benchmark
#SBATCH -o bench_heat_%j.out

set -eu; set -o pipefail
module restore Heat
source /p/project/haf/users/vonderlehr/heat/heat-venv/bin/activate
export CUDA_VISIBLE_DEVICES=""


function runbenchmark()
{
        echo $(date)
        echo "Benchmark Parameters:"
        echo "Total Number of Tasks: ${SLURM_NTASKS}"
        echo "Number of Nodes: ${SLURM_JOB_NUM_NODES}"
        echo "Tasks per Node: ${SLURM_NTASKS_PER_NODE}"

	if [ "$1" == "b" ]
  		then srun python -u ./grad_perf_eval.py --blocking --batch-size $2 --nn-id $3
  		else srun python -u ./grad_perf_eval.py --batch-size $2 --nn-id $3
	fi
}


for bs in 32 64 128
do 
	for nn in 1 2 3
	do
		runbenchmark "nb" $bs $nn
		runbenchmark "b" $bs $nn
	done
done
