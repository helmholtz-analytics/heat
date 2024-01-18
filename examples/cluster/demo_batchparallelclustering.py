import heat as ht
import argparse
import time
import numpy as np
import h5py

"""
This programm allows to compare the performance of batch-parallel K-means clustering and classical (parallel) K-means clustering (both in Heat)
Note that batch-parallel K-means on 1 MPI-process and GPU is just a PyTorch/GPU-implementation of K-means.
"""

# Create a parser object
parser = argparse.ArgumentParser(
    prog="Demo and Comparison for Batch-Parallel K-Means Clustering in Heat",
    description="Demo and Comparison for Batch-Parallel K-Means Clustering in Heat",
)

# Add the arguments
parser.add_argument(
    "--method",
    type=str,
    help="The method to use for clustering: batch-parallel k-means (bpkm) or k-means (km)",
    required=True,
)
parser.add_argument(
    "--init",
    type=str,
    help="The method to use for initialization: random, ++, or batch-parallel",
    required=True,
)
parser.add_argument("--n_clusters", type=int, help="The number of clusters to find", required=True)
parser.add_argument(
    "--n_merges",
    type=int,
    default=0,
    help="Number of merges to perform in batch-parallel clustering (0 = None)",
    required=False,
)
parser.add_argument("--filepath", type=str, help="path to the data set", required=True)
parser.add_argument(
    "--results_path", type=str, default="./", help="path for the results", required=False
)
parser.add_argument(
    "--device", type=str, help="device to use for computation: cpu or gpu", required=True
)
parser.add_argument(
    "--preprocessing",
    type=bool,
    default=True,
    help="whether preprocessing (normalization) needs to be performed or not",
    required=False,
)
parser.add_argument(
    "--n_samples",
    type=int,
    help="Number of runs to perform (for benchmarking purposes)",
    required=True,
)

# Parse the arguments
args = parser.parse_args()

method = args.method
init = args.init
n_clusters = args.n_clusters
n_merges = args.n_merges
filepath = args.filepath
results_path = args.results_path
device = args.device
preprocessing_flag = args.preprocessing
n_samples = args.n_samples
n_procs = ht.MPI_WORLD.size


# cath possible errors
# batch-parallel clustering only allows k-means++ initialization
if method == "bpkm":
    if init != "++":
        raise ValueError("Batch-parallel clustering only allows k-means++ initialization")
    else:
        init = "k-means++"
    if n_merges <= 1:
        n_merges = None
elif method == "km":
    if init == "++":
        init = "kmeans++"
else:
    raise ValueError("Invalid method specified")

# get data name of the data set
with h5py.File(filepath, "r") as f:
    data_name = list(f.keys())[0]

# load the data
X = ht.load_hdf5(filepath, dataset=data_name, split=0, device=device)

# normalize the data if required
if preprocessing_flag:
    scaler = ht.preprocessing.StandardScaler(copy=False)
    scaler.fit_transform(X)

# prepare arrays for measured times and functional values
times = np.zeros(n_samples)
values = np.zeros(n_samples)

# perform clustering
for i in range(n_samples):
    if method == "bpkm":
        if ht.MPI_WORLD.rank == 0:
            print(
                f"Run {i+1} of {n_samples}: Batch-parallel K-means with {n_clusters} clusters, {n_merges} merges, and {init} initialization on {n_procs} MPI-processes ({device})"
            )
        clusterer = ht.cluster.BatchParallelKMeans(
            n_clusters=n_clusters, init=init, n_procs_to_merge=n_merges
        )
    else:
        if ht.MPI_WORLD.rank == 0:
            print(
                f"Run {i+1} of {n_samples}: K-means with {n_clusters} clusters and {init} initializationon on {n_procs} MPI-processes ({device})"
            )
        clusterer = ht.cluster.KMeans(n_clusters=n_clusters, init=init)
    ht.MPI_WORLD.Barrier()
    t0 = time.perf_counter()
    labels = clusterer.fit_predict(X)
    ht.MPI_WORLD.Barrier()
    t1 = time.perf_counter()
    times[i] = t1 - t0
    if method == "bpkm" or method == "km":
        values[i] = clusterer.functional_value_
    del clusterer

# save results
if ht.MPI_WORLD.rank == 0:
    np.savetxt(
        f"{results_path}times_{method}_{n_clusters}_{n_merges}_{init}_{n_samples}_{n_procs}_{device}.txt",
        times,
    )
    np.savetxt(
        f"{results_path}values_{method}_{n_clusters}_{n_merges}_{init}_{n_samples}_{n_procs}_{device}.txt",
        values,
    )
