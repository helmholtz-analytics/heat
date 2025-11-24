# heat_dl_00_env_info.py
"""
====================================================================================================
SLURM ENVIRONMENT TRAINING
====================================================================================================

----------------------------------------------------------------------------------------------------
Task 1 — One Process on One Node
----------------------------------------------------------------------------------------------------
Run:  
    srun --ntasks=1 \
         --nodes=1 \
         --ntasks-per-node=1 \
         python python heat_dl_00_env_info.py

Questions:
    Q1: How many processes are created?
    Q2: What are RANK, NPROCS, LOCALID?
    Q3: Why is LOCALID always 0?

----------------------------------------------------------------------------------------------------
Task 2 — Two Processes on One Node
----------------------------------------------------------------------------------------------------
Run:
    srun --ntasks=2 --nodes=1 python heat_dl_00_env_info.py

Questions:
    Q1: What RANK values appear?
    Q2: Why is NPROCS the same in both?
    Q3: What does LOCALID represent?

----------------------------------------------------------------------------------------------------
Task 3 — Four Processes on Four GPUs
----------------------------------------------------------------------------------------------------
Run:
    srun --ntasks=4 --nodes=1 --ntasks-per-node=4  python heat_dl_00_env_info.py

Questions:
    Q1: What LOCALID values do you expect?
    Q2: How should these map to GPUs?
    Q3: What does CUDA_VISIBLE_DEVICES look like?
    
----------------------------------------------------------------------------------------------------
Task 4 — Reflection Summary
----------------------------------------------------------------------------------------------------

Questions:
    Q1: What do RANK, NPROCS, LOCALID mean on a single node?
    Q2: How should the GPU be selected within PyTorch?
    Q3: What happens if two processes use the same GPU?

====================================================================================================
End of Tasks
====================================================================================================
"""

import os
import socket

# check SLURM
if "SLURM_JOB_ID" not in os.environ:
    print("Warning: not running inside a SLURM environment.")
      
hostname = socket.gethostname()
pid = os.getpid()
rank = os.environ.get("SLURM_PROCID")
nprocs = os.environ.get("SLURM_NPROCS")
localid = os.environ.get("SLURM_LOCALID")
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

print(
    f"[{hostname}] "
    f"PID={pid} | RANK={rank} | NPROCS={nprocs} | "
    f"LOCALID={localid} | CUDA_VISIBLE_DEVICES={cuda_visible}"
)
