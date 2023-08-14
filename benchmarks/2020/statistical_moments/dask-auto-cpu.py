#!/usr/bin/env python

import argparse
import os
import time

import dask.array as da
import h5py
from dask.distributed import Client

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dask auto statistical moments cpu benchmark")
    parser.add_argument("--file", type=str, help="file to benchmark")
    parser.add_argument("--dataset", type=str, help="dataset within file to benchmark")
    parser.add_argument("--trials", type=int, help="number of benchmark trials")
    args = parser.parse_args()

    client = Client(scheduler_file=os.path.join(os.getcwd(), "scheduler.json"))

    print(f"Loading data... {args.file}[{args.dataset}]", end="")
    with h5py.File(args.file, "r") as handle:
        data = da.from_array(handle[args.dataset], chunks="auto").persist()
    print("\t[OK]")

    for function in [da.mean, da.std]:
        for axis in [None, 0, 1]:
            print(f"{function} axis={axis}")
            for trial in range(args.trials):
                print(f"Trial {trial}...", end="")
                start = time.perf_counter()
                function(data, axis=axis).compute()
                end = time.perf_counter()
                print(f"\t{end - start}s")
