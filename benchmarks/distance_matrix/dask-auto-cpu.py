# !/usr/bin/env python

import argparse
import dask
import dask.array as da
import dask_ml.metrics as dmm
import h5py
import os
import time

from dask.distributed import Client

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dask auto distance matrix cpu benchmark")
    parser.add_argument("--file", type=str, help="file to benchmark")
    parser.add_argument("--dataset", type=str, help="dataset within file to benchmark")
    parser.add_argument("--trials", type=int, help="number of benchmark trials")
    args = parser.parse_args()

    client = Client(scheduler_file=os.path.join(os.getcwd(), "scheduler.json"))

    print("Loading data... {}[{}]".format(args.file, args.dataset), end="")
    with h5py.File(args.file, "r") as handle:
        data = da.from_array(handle[args.dataset], chunks=("auto", -1)).persist()
    print("\t[OK]")

    for trial in range(args.trials):
        print("Trial {}...".format(trial), end="")
        start = time.perf_counter()
        dist = dmm.euclidean_distances(data, data).compute()
        end = time.perf_counter()
        print("\t{}s".format(end - start))
