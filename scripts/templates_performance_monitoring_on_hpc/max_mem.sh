#!/bin/bash

echo -n "$(hostname): "
cat /sys/fs/cgroup/memory/slurm/uid_${SLURM_JOB_UID}/job_${SLURM_JOB_ID}/memory.max_usage_in_bytes

###########################################################################################
# Important note: make sure that max_mem.sh is executable (e.g., chmod +x max_mem.sh)
###########################################################################################
