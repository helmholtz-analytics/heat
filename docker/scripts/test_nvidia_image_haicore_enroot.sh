#!/bin/bash
# Example SLURM/ENROOT script. It will mount the container using enroot, and then run the test script to test the compatibility of the image with the source version of heat.

# Clear environment, else mpi4py will fail to install.
ml purge

SBATCH_PARAMS=(
	--partition 	   normal
	--time      	   00:10:00
	--nodes     	   1
	--tasks-per-node   1
	--gres		   gpu:1
	--container-image  ~/containers/nvidia+pytorch+23.05-py3.sqsh
	--container-writable
	--container-mounts /etc/slurm/task_prolog.hk:/etc/slurm/task_prolog.hk,/scratch:/scratch
	--container-mount-home
)

sbatch "${SBATCH_PARAMS[@]}" ./install_print_test.sh
