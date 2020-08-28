!/bin/bash

for t in 1 2 4
do  
	sbatch --nodes 1 --ntasks-per-node $t grad_perf_eval.sh
done

for n in 1 2 4 8 15
do
	sbatch --nodes $n --tasks-per-node 4 grad_perf_eval.sh
done
