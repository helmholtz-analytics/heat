# heat_dl__03_process_group.py
"""
===========================================================================
DDP Initialization Script — Questions
===========================================================================

This script shows the minimal steps to initialize a PyTorch Distributed
Data Parallel (DDP) process group under SLURM using the NCCL backend.

It uses the environment variables set by SLURM:
    • SLURM_PROCID  → global rank
    • SLURM_NPROCS  → world size
    • MASTER_ADDR   → address of rank 0 node
    • MASTER_PORT   → port for communication

---------------------------------------------------------------------------
Task 1 — Understanding DDP Initialization
---------------------------------------------------------------------------
Questions:
    Q1: What does dist.init_process_group() do?
    Q2: Why must every process call this function?
    Q3: What information must each process know?
        How is this information retrived?

---------------------------------------------------------------------------
Task 2 — SLURM Rank and World Size
---------------------------------------------------------------------------
Questions:
    Q1: What is SLURM_PROCID?
    Q2: What is SLURM_NPROCS?
    Q3: Why is rank 0 special?

---------------------------------------------------------------------------
Task 3 — MASTER_ADDR and MASTER_PORT
---------------------------------------------------------------------------
Questions:
    Q1: Why do we need MASTER_ADDR?
    Q2: Why do we add "i" to the hostname in SLURM?
    Q3: Why do we run nslookup on the hostname?

---------------------------------------------------------------------------
Task 4 — NCCL Backend
---------------------------------------------------------------------------
Questions:
    Q1: Why is NCCL recommended for multi-GPU training?
    Q2: When should we use "gloo" instead?
    Q3: Does NCCL support CPU-only training?

---------------------------------------------------------------------------
Task 5 — Destroying the Process Group
---------------------------------------------------------------------------
Questions:
    Q1: Why call dist.destroy_process_group()?
    Q2: What happens if we omit it?

---------------------------------------------------------------------------
Task 6 — Launching With SLURM
---------------------------------------------------------------------------
Questions:
    Q1: How do you launch this script with 4 processes?
    Q2: How many GPUs does each process use?
    Q3: What happens if you launch with --ntasks=1?

===========================================================================
End of Q/A Section
===========================================================================
"""
import os
import torch.distributed as dist

"""
MASTER_ADDR=$(scontrol show hostnames | head -n 1)
MASTER_ADDR="${MASTER_ADDR}i"
export MASTER_ADDR=$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')
export MASTER_PORT=6000
"""


def main():
    # Check if we're inside a SLURM environment
    if "SLURM_PROCID" in os.environ and "SLURM_NPROCS" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NPROCS"])
    else:
        # Fallback for local execution
        rank = 0
        world_size = 1
        print("Running locally: setting rank=0, world_size=1")

    # Only initialize distributed if world_size > 1
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        print(f"Rank {rank}/{world_size} initialized")
        dist.destroy_process_group()
    else:
        print("Distributed not initialized (single-process mode).")


if __name__ == "__main__":
    main()
