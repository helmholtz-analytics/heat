# step_02_init_ddp.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


"""
==============================================================
Distributed PyTorch Example (rank % 4 GPU selection)
==============================================================

This example assumes:
- SLURM launches *one process per GPU*
- There are exactly 4 GPUs per node
- The local GPU index can be computed as: local_rank = rank % 4

==============================================================
Teaching Questions (and Answers)
==============================================================

Q1: Why do we select the GPU as rank % 4?


Q2: Why do all ranks initialize the process group?


Q3: Why does each rank create its own model?

Q4: Why do we call loss.backward() on every rank?

Q5: Why is destroy_process_group() needed?


Q6: What exactly does DDP replicate across processes?


Q7: How does DDP synchronize gradients?


Q8: What is a gradient bucket and why does DDP use them?


Q9: How does DDP achieve overlap between communication and computation?


Q10: How does DDP ensure model parameters are identical across ranks?


Q11: Why must the forward pass be identical across ranks?


Q12: Why is DistributedDataParallel faster than torch.nn.DataParallel?


Q13: What happens if one rank crashes during training?


Q14: Why can’t we call optimizer.step() before gradients synchronize?


Q15: What is the difference between distributed data parallel and
     model parallel?


Q16: Why does DDP require the same batch size on every rank?

Q17: Why does DDP require each rank to enter backward() exactly once?


Q18: What is "gradient bucketing order" and why does it matter?



==============================================================
"""


# step_02_init_ddp.py (modified with training summary)
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time

# ---- Utility: format seconds ---------------------------------------------------
def format_seconds(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


# ---- A tiny neural network -----------------------------------------------------
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


def main():
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NPROCS"])

    #local_rank = rank % 4
    torch.cuda.set_device(0)
    #torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    print(f"[Rank {rank}] initialized on GPU {0}")

    model = TinyNet().cuda(0)
    #model = DDP(model, device_ids=[local_rank])
    model = DDP(model, device_ids=[0])

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(64, 32).cuda(0)
    y = torch.randint(0, 10, (64,), device=f"cuda:{0}")

    batch_size = x.size(0)
    nepochs = 1

    # --- FLOPs (forward + backward) ---
    # Forward FLOPs
    flops_fc1 = 32 * 64 * 2
    flops_relu = 64
    flops_fc2 = 64 * 10 * 2
    flops_forward = flops_fc1 + flops_relu + flops_fc2

    # Backward FLOPs (rough rule: ~2x forward for linear layers)
    flops_backward = 2 * (flops_fc1 + flops_fc2)
    flops_per_step = flops_forward + flops_backward

    # --- Timing variables ---
    step_times = []
    torch.cuda.synchronize()
    start_time = time.time()

    from torch.profiler import profile, record_function, ProfilerActivity
    nepochs = 100
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for step in range(nepochs):
            t0 = time.time()

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            step_times.append(time.time() - t0)

            if rank == 0:
                print(f"Step {step}: loss = {loss.item():.4f}")

    torch.cuda.synchronize()
    total_time = time.time() - start_time

    # Compute summary stats on rank 0
    if rank == 0:
        import statistics
        mean_step = statistics.mean(step_times)
        std_error = statistics.pstdev(step_times)

        samples_per_sec = batch_size / mean_step

        # Dummy FLOP estimates
        # Compute FLOPs for TinyNet
        # Linear(32->64): 32*64*2 FLOPs (mul+add)
        # ReLU: 64 FLOPs
        # Linear(64->10): 64*10*2 FLOPs
        flops_fc1 = 32 * 64 * 2
        flops_relu = 64
        flops_fc2 = 64 * 10 * 2
        flops_per_step = flops_fc1 + flops_relu + flops_fc2
        flops = flops_per_step / mean_step / 1e9

        print("\n===== Training Summary =====")
        print(f"Batch size:             {batch_size}")
        print(f"Epochs:                 {nepochs}")
        print(f"Total runtime:          {format_seconds(total_time)}")
        print(f"Avg time/step:          {mean_step*1000:.4f} +/- {std_error*1000:.4f} ms")
        print(f"Throughput:             {samples_per_sec:.2f} samples/sec")
        print(f"Forward FLOPs/step:     {flops_forward:.3f} FLOPs")
        print(f"Backward FLOPs/step:    {flops_backward:.3f} FLOPs")
        print(f"Total FLOPs/step:       {flops_per_step:.3f} FLOPs")
        print(f"Compute throughput:     {flops:.3f} GFLOPs/sec")
        print(f"GPU Memory Allocated:   {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"GPU Memory Reserved:    {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        print("============================\n")

    dist.destroy_process_group()
    if rank == 0:
        print("Training complete.")


if __name__ == "__main__":
    main()
