#!/usr/bin/env python

import argparse
import h5py
import time
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch statistical moments gpu benchmark")
    parser.add_argument("--file", type=str, help="file to benchmark")
    parser.add_argument("--dataset", type=str, help="dataset within file to benchmark")
    parser.add_argument("--trials", type=int, help="number of benchmark trials")
    args = parser.parse_args()

    print(f"Loading data... {args.file}[{args.dataset}]", end="")
    with h5py.File(args.file, "r") as handle:
        data = torch.Tensor(handle[args.dataset], device="cuda")
    print("\t[OK]")

    for function in [torch.mean, torch.std]:
        for axis in [None, 0, 1]:
            print(f"{function} axis={axis}")
            for trial in range(args.trials):
                print(f"Trial {trial}...", end="")
                start = time.perf_counter()
                function(data, axis=axis)
                end = time.perf_counter()
                print(f"\t{end - start}s")
