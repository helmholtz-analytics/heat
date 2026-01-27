"""
====================================================================================================
CUDA + PyTorch GPU Inspection
====================================================================================================

This script prints detailed information about:
  - SLURM environment (RANK, LOCALID, task count)
  - CUDA visibility as exposed by SLURM
  - PyTorch CUDA availability
  - CUDA + cuDNN versions
  - GPU count / properties / memory / compute capability

Use it to verify correct GPU mapping under SLURM.


For other accelerators, PyTorch provides the method
torch.accelerator.current_accelerator() (available since version 2.9.0).

----------------------------------------------------------------------------------------------------
Task 1 — Single GPU Test
----------------------------------------------------------------------------------------------------
Run:
    srun --ntasks=1 --gres=gpu:1 python heat_dl_01_gpu_info.py

Questions:
    Q1: What should CUDA_VISIBLE_DEVICES show?
    Q2: How many GPUs should PyTorch detect?
    Q3: What is the expected value of `current device index`?

----------------------------------------------------------------------------------------------------
Task 2 — Multi-GPU Visibility (4 tasks / 4 GPUs)
----------------------------------------------------------------------------------------------------
Run:
    srun --ntasks=4 --gres=gpu:4 python heat_dl_01_gpu_info.py

Expected:
    Each process sees exactly **one GPU**, even though the node has 4.

Questions:
    Q1: What values do LOCALID and CUDA_VISIBLE_DEVICES show?
    Q2: Does each process see a unique GPU?
    Q3: Why does PyTorch still report GPU count = 1?

----------------------------------------------------------------------------------------------------
Task 4 — Understanding Device Properties
----------------------------------------------------------------------------------------------------
Run:
    srun --ntasks=1 --gres=gpu:1 heat_dl_01_gpu_info.py

Questions:
    Q1: What device properties does PyTorch report?
    Q2: Why is total memory shown in gigabytes?
    Q3: What does "compute capability" mean?

----------------------------------------------------------------------------------------------------
Task 5 — Compare SLURM and PyTorch Device Counts
----------------------------------------------------------------------------------------------------
Run:
    srun --ntasks=1 --gres=gpu:4 python heat_dl_01_gpu_info.py

Questions:
    Q1: How many GPUs does SLURM make visible to this single task?
    Q2: How many GPUs does PyTorch report?
    Q3: Why might they differ from the number of physical GPUs?

----------------------------------------------------------------------------------------------------
Task 6 — Debugging Mismatched GPU Usage
----------------------------------------------------------------------------------------------------
Run in a misconfigured job (example):
    srun --ntasks=2 --gres=gpu:1 python heat_dl_01_gpu_info.py

Questions:
    Q1: What happens if two tasks request 1 GPU each but only 1 GPU exists?
    Q2: How does CUDA_VISIBLE_DEVICES help diagnose the issue?
    Q3: What failure modes might appear in PyTorch?

====================================================================================================
End of Tasks
====================================================================================================
"""

import os
import socket
import torch


def cuda_info():

    host = socket.gethostname()
    pid = os.getpid()

    # SLURM metadata (falls back to ? if not available)
    rank = os.environ.get("SLURM_PROCID", "?")
    localid = os.environ.get("SLURM_LOCALID", "?")
    nprocs = os.environ.get("SLURM_NPROCS", "?")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "?")

    print(
        f"\n[HOST={host} | PID={pid} | RANK={rank} | LOCALID={localid} | "
        f"NPROCS={nprocs} | CUDA_VISIBLE={visible}]"
    )

    print("=== CUDA & GPU Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        return  # stop here if no GPU

    # Global CUDA + cuDNN
    print(f"CUDA version (PyTorch build): {torch.version.cuda}")
    print(f"cuDNN version               : {torch.backends.cudnn.version()}")

    # GPU enumeration
    gpu_count = torch.cuda.device_count()
    print(f"GPU count                   : {gpu_count}")

    # Current device
    current_dev = torch.cuda.current_device()
    print(f"PyTorch device index        : {current_dev}")
    print(f"Current device name         : {torch.cuda.get_device_name(current_dev)}")

    # Per-device details
    print("\n--- Per-Device Properties ---")
    for dev_id in range(gpu_count):
        props = torch.cuda.get_device_properties(dev_id)
        mem_gb = round(props.total_memory / (1024**3), 2)
        print(
            f"[GPU {dev_id}] {props.name} | {mem_gb} GB | "
            f"Compute Capability {props.major}.{props.minor} | "
            f"MPs {props.multi_processor_count}"
        )

    print()  # spacer


if __name__ == "__main__":
    cuda_info()
