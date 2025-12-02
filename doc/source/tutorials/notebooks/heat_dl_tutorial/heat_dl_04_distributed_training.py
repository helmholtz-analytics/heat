"""
====================================================================================================
Distributed PyTorch Example on JURECA (4 GPUS per Node)
====================================================================================================

This example assumes:
- SLURM launches *one process per GPU*
- There are exactly 4 GPUs per node
- The local GPU index can be computed as: local_rank = rank % 4 or retrieved
  from SLURM_LOCALID
- However, SLURm might be configurated such that each process sees only one GPU

====================================================================================================

====================================================================================================

Q1: Why do we select the GPU as rank % 4/SLURM_LOCALID or '0'?


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


Q12: What happens if one rank crashes during training?

====================================================================================================
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
import torchvision.transforms as transforms
import time
import argparse
import statistics
import math




# ---- Utility: format seconds ---------------------------------------------------------------------
def format_seconds(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


# --------------------------------------------------------------------------------------------------
# 1. Random CIFAR-shaped Dataset
# --------------------------------------------------------------------------------------------------

class RandomData(Dataset):
    def __init__(self, length=50000, num_classes=10):
        self.length = length
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = torch.randn(3, 32, 32)                   # random image
        label = torch.randint(0, self.num_classes, ()) # random label
        return img, label



class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 32×16×16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 64×8×8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x


# --------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_cifar",
        action="store_true",
        help="Train on CIFAR-10 instead of fake data.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    # CPU info
    cpus = os.sched_getaffinity(0)   # 0 = current process
    print("CPU affinity:", cpus)
    print("As list:", list(cpus))

    # --- SLURM sanity check -----------------------------------------------------------------------
    if "SLURM_JOB_ID" not in os.environ:
        print("Error: not running inside a SLURM environment.", file=sys.stderr)
        sys.exit(1)

    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NPROCS"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    # --- GPU visibility check ---------------------------------------------------------------------
    num_visible = torch.cuda.device_count()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET")

    if rank == 0:
        print(f"[Rank 0] CUDA_VISIBLE_DEVICES = {cuda_visible_devices}")
        print(f"[Rank 0] PyTorch sees {num_visible} CUDA device(s)")

    if num_visible == 1:
        selected_device = 0
    else:
        selected_device = local_rank

    torch.cuda.set_device(selected_device)
    device = torch.device(f"cuda:{selected_device}")

    if rank == 1:
        print(f"Second process (RANK=1) sees {num_visible} GPUs using cuda:{selected_device}")

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    model = TinyNet().to(device)
    model = DDP(model, device_ids=[selected_device])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)



    # ----------------------------------------------------------------------------------------------
    # DATA SOURCE: CIFAR-10 or Fake
    # ----------------------------------------------------------------------------------------------
    if args.use_cifar:
        if rank == 0:
            print("[Info] Using CIFAR-10 dataset.")

        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
        ])

        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform)


    else:
        if rank == 0:
            print("[Info] Using fake random data.")

        trainset = RandomData(length=50000)


    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)

    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
    #
    dist.barrier()
    # ----------------------------------------------------------------------------------------------
    # Training Loop
    # ----------------------------------------------------------------------------------------------
    nepochs = 5
    epoch_times = []

    torch.cuda.synchronize()
    start_time = time.time()
    t0 = time.time()


    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        # ------------------------------------------------------------------------------------------
        # 4. Training Loop
        # ------------------------------------------------------------------------------------------

        for epoch in range(nepochs):
            t0 = time.time()

            model.train()
            running_loss = 0.0
            for images, labels in trainloader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()


            epoch_times.append(time.time() - t0)

            print(
                  f"[Rank {local_rank}] Epoch {epoch+1}: "
                  f"loss = {running_loss / len(trainloader):.4f}"
            )


    dist.barrier()
    # ----------------------------------------------------------------------------------------------
    if rank == 0:


        vals = epoch_times[1:]
        mean_epoch = statistics.mean(vals)
        # sample standard deviation
        sd = statistics.stdev(vals)
        # standard error
        se = sd / math.sqrt(len(vals))
        print(f"\nEpoch time: {mean_epoch:.3f} ± {se:.3f}s")

    dist.destroy_process_group()

    if rank == 0:
        print("Training complete.")

# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
