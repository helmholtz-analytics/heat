#!/usr/bin/env python

import argparse
import dask
import dask.array as da
import dask_ml.cluster as dmc
import h5py
import os
import time

from dask.distributed import Client
from mpi4py import MPI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dask kmeans cpu chunks benchmark")
    parser.add_argument("--file", type=str, help="file to benchmark")
    parser.add_argument("--dataset", type=str, help="dataset within file to benchmark")
    parser.add_argument("--trials", type=int, help="number of benchmark trials")
    parser.add_argument("--clusters", type=int, help="number of cluster centers")
    parser.add_argument("--iterations", type=int, help="kmeans iterations")
    args = parser.parse_args()

    client = Client(scheduler_file=os.path.join(os.getcwd(), "scheduler.json"))

    print(f"Loading data... {args.file}[{args.dataset}]", end="")
    with h5py.File(args.file, "r") as handle:
        chunk_size = handle[args.dataset].shape[0] // (MPI.COMM_WORLD.size - 1)
        data = da.from_array(handle[args.dataset], chunks=(chunk_size, -1)).persist()
    print("\t[OK]")

    for trial in range(args.trials):
        print(f"Trial {trial}...", end="")
        kmeans = dmc.KMeans(
            n_clusters=args.clusters, max_iter=args.iterations, init="random", tol=-1.0
        )
        start = time.perf_counter()
        kmeans.fit(data)
        end = time.perf_counter()
        print(f"\t{end - start}s")
