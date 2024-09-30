There are two example scripts in this directory, `slurm_script_cpu.sh` and `slurm_script_gpu.sh`, that demonstrate how to run a Heat application on an HPC-system with SLURM as resource manager.

1. `slurm_script_cpu.sh` is an example script that runs a Heat application on a CPU node. You must specify the name of the respective partition of your cluster. Moreover, the
    numer of CPU cores available at a node of your system must be greater or equal to the product of the tasks-per-node- and the cpus-per-task-argument (=8x16=128 in the example).

2. `slurm_script_gpu.sh` is an example script that runs a Heat application on a GPU node. You must specify the name of the respective partition of your cluster. Moreover, the
    numer of GPU devices available at a node of your system must be greater or equal to the number of GPUs requested in the script (=4 in the example).

## Remarks

* Please have a look into the documentation of your HPC-system for its detailed configuration and properties. Maybe, you have to adjust the script to your system.
* You need to load the required modules (e.g., for MPI, CUDA etc.) from the module system of your HPC-system before running the script. Moreover, you need to install Heat in a virtual environment (and activate it). Alternatively, you may use spack (if available on your system) for installing Heat and its dependencies.
