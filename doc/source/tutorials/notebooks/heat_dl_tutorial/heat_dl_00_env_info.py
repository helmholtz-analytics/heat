# heat_dl_00_env_info.py
"""
Run on JURECA (1 node, 4 GPUs):

    srun --ntasks=4 \
         --nodes=1 \
         --ntasks-per-node=4 \
         --gres=gpu:4 \
         python step_00_env_info_single_node.py

===========================================================
Single-Node, 4-GPU Learning Tasks
===========================================================

------------------------------------------------------------
Task 1 — One Process on One Node
------------------------------------------------------------
Run:
    srun --ntasks=1 --nodes=1 --gres=gpu:4 python step_00_env_info_single_node.py

Questions:
    Q1: How many processes are created?
    Q2: What are RANK, NPROCS, LOCALID?
    Q3: Why is LOCALID always 0?


------------------------------------------------------------
Task 2 — Two Processes on One Node
------------------------------------------------------------
Run:
    srun --ntasks=2 --nodes=1 --gres=gpu:4 python step_00_env_info_single_node.py

Questions:
    Q1: What RANK values appear?
    Q2: Why is NPROCS the same in both?
    Q3: What does LOCALID represent?



------------------------------------------------------------
Task 3 — Four Processes on Four GPUs
------------------------------------------------------------
Run:
    srun --ntasks=4 --nodes=1 --ntasks-per-node=4 --gres=gpu:4 python step_00_env_info_single_node.py

Questions:
    Q1: What LOCALID values do you expect?
    Q2: How should these map to GPUs?
    Q3: What does CUDA_VISIBLE_DEVICES look like?



------------------------------------------------------------
Task 4 — Understanding GPU Affinity
------------------------------------------------------------
Run:
    srun --ntasks=4 --gres=gpu:4 python step_00_env_info_single_node.py

Questions:
    Q1: How does SLURM ensure unique GPU assignment?
    Q2: Why do processes often only see GPU "0"?
    Q3: Why does DDP rely on LOCALID?



------------------------------------------------------------
Task 5 — Predict Mapping With 3 Processes
------------------------------------------------------------
Run:
    srun --ntasks=3 --nodes=1 --gres=gpu:4 python step_00_env_info_single_node.py

Questions:
    Q1: What RANK values appear?
    Q2: What LOCALID values appear?
    Q3: How many GPUs remain unused?


------------------------------------------------------------
Task 6 — Reflection Summary
------------------------------------------------------------

Questions:
    Q1: What do RANK, NPROCS, LOCALID mean on a single node?
    Q2: Why should GPU selection be based on LOCALID?
    Q3: What happens if two processes use the same GPU?



===========================================================
End of Tasks
===========================================================
"""

import os
import socket

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
