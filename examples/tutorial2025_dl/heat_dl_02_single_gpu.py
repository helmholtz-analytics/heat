# heat_dl_02_single_gpu.py
"""
====================================================================================================
Single-GPU Timing + FLOPs + Throughput + Memory
====================================================================================================

This script trains a small Linear model on a single GPU and reports:
    • Per-step time + mean + standard error
    • FLOPs per step (approx for Linear layer)
    • GFLOPs/sec (compute throughput)
    • Samples/sec (data throughput)
    • GPU memory usage (allocated / reserved)
    • Total runtime

Use this to understand GPU utilization, timing stability, and model cost.

----------------------------------------------------------------------------------------------------
Task 1 — GPU Verification
----------------------------------------------------------------------------------------------------
Questions:
    Q1: How do we know the script runs on the GPU?
    Q2: What happens if no GPU exists?
    Q3: Why is torch.cuda.synchronize() used before timing?


----------------------------------------------------------------------------------------------------
Task 2 — FLOPs Computation
----------------------------------------------------------------------------------------------------
Questions:
    Q1: What FLOPs formula is used for nn.Linear?
    Q2: Why is the FLOP count approximate?
    Q3: Why are FLOPs important?


----------------------------------------------------------------------------------------------------
Task 3 — Understanding Throughput
----------------------------------------------------------------------------------------------------
Questions:
    Q1: How is samples/sec computed?
    Q2: What does FLOPs/sec measure?
    Q3: Why is this useful?


----------------------------------------------------------------------------------------------------
Task 4 — GPU Memory Monitoring
----------------------------------------------------------------------------------------------------
Questions:
    Q1: What is "allocated" vs "reserved" GPU memory?
    Q2: Why is reserved memory often larger?
    Q3: Why does memory usage remain almost constant?


----------------------------------------------------------------------------------------------------
Task 5 — Timing Stability
----------------------------------------------------------------------------------------------------
Questions:
    Q1: Why can the first steps be slower?
    Q2: Why compute standard error?
    Q3: What factors influence step-time variability?


----------------------------------------------------------------------------------------------------
Task 6 — Model Size & Performance Scaling
----------------------------------------------------------------------------------------------------
Questions:
    Q1: If we increase model dimensions, what happens to FLOPs?
    Q2: What happens to step time?
    Q3: What happens to throughput (samples/sec)?



----------------------------------------------------------------------------------------------------
Task 7 — FLOPs Computation, the 2nd
----------------------------------------------------------------------------------------------------
Questions:
    Q1: What is the exact FLOPS a a forward pass?
    Q2: What is the FLOPS of the backward pass, exact and estimate
    Q3: How does this approx. generalize to multiple layers


====================================================================================================
====================================================================================================
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import math


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------


def format_seconds(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def compute_time_stats(times):
    if len(times) == 0:
        return 0.0, 0.0
    mean_val = sum(times) / len(times)
    if len(times) == 1:
        return mean_val, 0.0
    variance = sum((t - mean_val) ** 2 for t in times) / (len(times) - 1)
    std_error = math.sqrt(variance) / math.sqrt(len(times))
    return mean_val, std_error


def compute_linear_flops(batch, in_dim, out_dim):
    """
    FLOPs for Linear = 2 * batch * in_dim * out_dim
    (1 multiply + 1 add per weight)
    """
    return 2 * batch * in_dim * out_dim


# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

in_dim, out_dim = 10, 1
batch_size = 16
nepochs = 1000
print_interval = 100

model = nn.Linear(in_dim, out_dim).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001)

x = torch.randn(batch_size, in_dim).cuda()
y = torch.randn(batch_size, out_dim).cuda()

step_times = []
flops_per_step = compute_linear_flops(batch_size, in_dim, out_dim)

# ---------------------------------------------------------
# Training loop with timing and GPU monitoring
# ---------------------------------------------------------

train_start = time.perf_counter()
for step in range(nepochs):
    torch.cuda.synchronize()
    step_start = time.perf_counter()

    pred = model(x)
    loss = ((pred - y) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    step_end = time.perf_counter()
    step_times.append(step_end - step_start)

    if step % print_interval == 0 or step == nepochs - 1:
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(
            f"[{step:04d}/{nepochs}] Loss={loss.item():.6f} "
            f"| Mem alloc={allocated:.1f}MB reserved={reserved:.1f}MB"
        )

train_end = time.perf_counter()
total_time = train_end - train_start

# ---------------------------------------------------------
# Statistics & Reporting
# ---------------------------------------------------------

mean_step, std_error = compute_time_stats(step_times)
samples_per_sec = batch_size / mean_step
flops = flops_per_step / mean_step

print("\n===== Training Summary =====")
print(f"Batch size:             {batch_size}")
print(f"Epochs:                 {nepochs}")
print(f"Total runtime:          {format_seconds(total_time)}")
print(f"Avg time/step:          {mean_step*1000:.4f} +/- {std_error*1000:.4f} ms")
print(f"Throughput:             {samples_per_sec:.2f} samples/sec")
print(f"Approx FLOPs/step:      {flops_per_step:d} FLOPs")
print(f"Compute throughput:     {flops:.3f} FLOPs/sec")
print(f"GPU Memory Allocated:   {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
print(f"GPU Memory Reserved:    {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
print("============================\n")
