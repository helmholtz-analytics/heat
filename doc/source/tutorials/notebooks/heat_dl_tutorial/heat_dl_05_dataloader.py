"""
srun --ntasks 1 --nodes=1 --ntasks-per-node=1   python heat_dl_05_dataloader.py

====================================================================================================
ImageNet DataLoader Throughput Benchmark (Single GPU)
====================================================================================================

This script measures:
    - Total images loaded per second
    - Per-worker throughput (images/sec/worker)
    - Loader efficiency vs batch size / number of workers

It loads ImageNet from disk using torchvision.datasets.ImageFolder
and measures how fast batches are produced by the DataLoader.

====================================================================================================
Teaching Questions (and Answers)
====================================================================================================

Q1: Why measure throughput per worker?


Q2: What limits DataLoader performance?


Q3: Why do larger batch sizes increase throughput?


Q4: Why use pin_memory=True?


Q5: Why do we drop the GPU computations entirely in this benchmark?


Q6: Is more workers always better?


Q7: Why do we use non_blocking=True when loading to GPU?


====================================================================================================
"""
import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import warnings
warnings.filterwarnings("ignore")

def benchmark_loader(root, batch_size=256, num_workers=8, warmup=1, measure_batches=3):
    print("=" * 100)
    print(f"Benchmarking: batch_size={batch_size}, num_workers={num_workers}")
    print("=" * 100)

    # 1. Standard ImageNet transforms (random crop + flip)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # 2. ImageFolder dataset
    dataset = datasets.ImageFolder(os.path.join(root, "train"), transform=transform)

    # 3. DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # 4. Warmup (avoid first-iteration overhead effects)
    it = iter(loader)
    for _ in range(warmup):
        next(it)

    # 5. Measure throughput
    start = time.time()
    total_images = 0
    for _ in range(measure_batches):
        batch = next(it)
        batch_size_actual = batch[0].shape[0]
        total_images += batch_size_actual



    duration = time.time() - start

    images_per_sec = total_images / duration
    images_per_sec_per_worker = images_per_sec / num_workers

    print(f"Total images processed:  {total_images}")
    print(f"Total time:              {duration:.3f} sec")
    print(f"Overall throughput:      {images_per_sec:.2f} images/sec")
    print(f"Per-worker throughput:   {images_per_sec_per_worker:.2f} images/sec/worker")
    print()

    return images_per_sec, images_per_sec_per_worker


def main():
    imagenet_root = "/p/scratch/training2546/datasets"

    # Try various worker counts
    for workers in [1, 4, 8, 12, 16, 20]:
        benchmark_loader(
            root=imagenet_root,
            batch_size=256,
            num_workers=workers,
            warmup=1,
            measure_batches=5,
        )


if __name__ == "__main__":
    main()
