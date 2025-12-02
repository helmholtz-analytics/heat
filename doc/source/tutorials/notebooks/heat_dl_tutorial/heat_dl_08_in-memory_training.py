"""
Q1: What is the purpose of this script?


Q2: How is CIFAR-10 loaded?


Q3: How does the ring-based shuffle work?


Q4: Why does shuffling happen on CPU?


Q5: How is distributed training initialized?


Q5: What environment is required?
"""

import argparse
import os
import sys
import time
import warnings
import mpi4py
from mpi4py import MPI
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import heat as ht
warnings.filterwarnings("ignore", category=ResourceWarning)

"""
export CUDA_VISIBLE_DEVICES=0,1,2,3

MASTER_ADDR=$(scontrol show hostnames | head -n 1)
MASTER_ADDR="${MASTER_ADDR}i"
export MASTER_ADDR=$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')
export MASTER_PORT=6000
"""

# --------------------------------------------
# Simple CNN for CIFAR-10
# --------------------------------------------
class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),       # 64 x 16 x 16

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),       # 128 x 8 x 8

            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------
# Load CIFAR-10 shard into CPU memory per MPI rank
# --------------------------------------------
def load_cifar10_shard_cpu(rank, world_size):
    """
    Each MPI rank loads only its shard of CIFAR-10,
    and keeps it as CPU tensors.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,   # only rank 0 actually downloads
        transform=transform,
    )
    # Make sure all ranks see the files
    ht.comm.barrier()

    total = len(dataset)  # typically 50,000
    shard = total // world_size

    start = rank * shard
    end = (rank + 1) * shard if rank != world_size - 1 else total

    images = []
    labels = []

    for i in range(start, end):
        img, lbl = dataset[i]         # img: Tensor [3,32,32] on CPU
        images.append(img)
        labels.append(lbl)

    x_local = torch.stack(images, dim=0)           # [N_local, 3,32,32] on CPU
    y_local = torch.tensor(labels, dtype=torch.long)  # [N_local] on CPU

    # Ensure even number of samples for 1/2 splitting
    n = x_local.shape[0]
    if n % 2 != 0:
        x_local = x_local[:-1]
        y_local = y_local[:-1]

    return x_local, y_local




def heat_ring_shuffle_cpu(local_x, local_y):
    """
    CPU-side ring shuffle using HeAT's non-blocking Isend/Irecv
    to avoid deadlocks.

    local_x, local_y: CPU tensors of shape [N, ...] and [N]
    """
    rank = ht.comm.rank
    world_size = ht.comm.size

    if world_size == 1:
        return local_x, local_y  # nothing to shuffle

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size

    n = local_x.shape[0]
    # ensure even number of samples
    if n % 2 != 0:
        n -= 1
        local_x = local_x[:n]
        local_y = local_y[:n]

    half = n // 2

    # split into keep/send halves (still on CPU)
    keep_x = local_x[:half].clone()
    send_x = local_x[half:].contiguous()
    keep_y = local_y[:half].clone()
    send_y = local_y[half:].contiguous()

    # allocate recv buffers (same size as send half)
    recv_x = torch.empty_like(send_x)
    recv_y = torch.empty_like(send_y)

    # ---- non-blocking MPI to avoid deadlock ----
    reqs = []

    # X part
    reqs.append(ht.comm.Isend(send_x, dest=next_rank, tag=200))
    reqs.append(ht.comm.Irecv(recv_x, source=prev_rank, tag=200))

    # Y part
    reqs.append(ht.comm.Isend(send_y, dest=next_rank, tag=201))
    reqs.append(ht.comm.Irecv(recv_y, source=prev_rank, tag=201))

    # Wait for all four operations to complete
    for r in reqs:
        r.Wait()

    # build new local tensors
    new_x = torch.cat([keep_x, recv_x], dim=0)
    new_y = torch.cat([keep_y, recv_y], dim=0)

    return new_x, new_y



# --------------------------------------------
# Round-robin shuffle on CPU using HeAT comm
# --------------------------------------------
def heat_ring_shuffle_cpu_21(local_x, local_y):
    """
    local_x, local_y: CPU tensors, shape [N, ...] and [N]
    On each MPI rank:
      - Keep first half of samples locally
      - Send second half to next rank
      - Receive second half from previous rank
      - Concatenate: new_local = keep_half + received_half

    Uses ht.comm.send / ht.comm.recv on CPU tensors.
    """
    rank = ht.comm.rank
    world_size = ht.comm.size

    if world_size == 1:
        return local_x, local_y  # no shuffle needed

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size

    n = local_x.shape[0]
    # Just in case, enforce even length
    if n % 2 != 0:
        n -= 1
        local_x = local_x[:n]
        local_y = local_y[:n]

    half = n // 2

    # Split into keep / send halves (still CPU tensors)
    keep_x = local_x[:half].clone()
    send_x = local_x[half:].clone()
    keep_y = local_y[:half].clone()
    send_y = local_y[half:].clone()

    # Send and receive using HeAT communicator (MPI under the hood)
    ht.comm.send(send_x, dest=next_rank, tag=200)
    recv_x = ht.comm.recv(source=prev_rank, tag=200)

    ht.comm.send(send_y, dest=next_rank, tag=201)
    recv_y = ht.comm.recv(source=prev_rank, tag=201)

    # recv_x and recv_y are CPU tensors from the neighbors
    new_x = torch.cat([keep_x, recv_x], dim=0)
    new_y = torch.cat([keep_y, recv_y], dim=0)

    return new_x, new_y


# --------------------------------------------
# Training loop for one MPI rank
# --------------------------------------------
def train(rank, world_size, local_rank):
    # Initialize PyTorch distributed with MPI backend


    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    assert dist.get_rank() == rank
    assert dist.get_world_size() == world_size
    assert ht.comm.rank == rank
    assert ht.comm.size == world_size


    # Load CIFAR-10 shard into CPU memory
    local_x_cpu, local_y_cpu = load_cifar10_shard_cpu(rank, world_size)

    slice_size=1000
    # Ensure slice_size is even (required by 1/2 splitting)
    if slice_size % 2 != 0:
        slice_size -= 1

    local_x_cpu = local_x_cpu[:slice_size].clone()
    local_y_cpu = local_y_cpu[:slice_size].clone()
    # Build model & DDP


    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    model = CIFAR10Model().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    batch_size = 128


    for epoch in range(epochs):
        # Build DataLoader over CPU tensors
        dataset = TensorDataset(local_x_cpu, local_y_cpu)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # no local shuffle; shuffling is done by ring,
        )
        print("epoch: ", epoch)
        model.train()



        for xb_cpu, yb_cpu in loader:
            # Move batch CPU -> GPU
            xb = xb_cpu.to(device, non_blocking=True)
            yb = yb_cpu.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()



        # -------------------------------
        # CPU-side round-robin shuffle
        # -------------------------------
        print('mean_before_shuffle: ', epoch, local_rank,local_x_cpu.mean())
        local_x_cpu, local_y_cpu = heat_ring_shuffle_cpu(local_x_cpu, local_y_cpu)
        print('mean_after_shuffle: ', epoch, local_rank, local_x_cpu.mean())
        # Ensure all ranks finish shuffle before next epoch
        dist.barrier()

    dist.destroy_process_group()


# --------------------------------------------
# Main entry (MPI)
# --------------------------------------------
def main():
    world_size = int(os.environ["SLURM_NPROCS"])
    rank       = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    train(rank, world_size, local_rank)


if __name__ == "__main__":
    main()
