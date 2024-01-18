import argparse
import time
import numpy as np
import sklearn.cluster as sklc
import sklearn.preprocessing as sklp
import h5py

"""
Comparison with scikit-learn
"""

# Create a parser object
parser = argparse.ArgumentParser(
    prog="Comparison with scikit-learn Kmeans implementations",
    description="Comparison with scikit-learn Kmeans implementations",
)

# Add the arguments
parser.add_argument(
    "--method",
    type=str,
    help="The method to use for clustering: k-means (km) or minibatch k-means (mbkm)",
    required=True,
)
parser.add_argument(
    "--init",
    type=str,
    help="The method to use for initialization: random or ++",
    required=True,
)
parser.add_argument("--n_clusters", type=int, help="The number of clusters to find", required=True)
parser.add_argument(
    "--batch_size",
    type=int,
    default=0,
    help="Batch size for minibatch k-means",
    required=False,
)
parser.add_argument("--filepath", type=str, help="path to the data set", required=True)
parser.add_argument(
    "--results_path", type=str, default="./", help="path for the results", required=False
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
batch_size = args.batch_size
filepath = args.filepath
results_path = args.results_path
preprocessing_flag = args.preprocessing
n_samples = args.n_samples


# cath possible errors
# batch-parallel clustering only allows k-means++ initialization
if init == "++":
    init = "k-means++"
if method == "mbkm":
    if batch_size <= 1:
        raise ValueError("Batch size must be larger than 1 for minibatch k-means")
else:
    batch_size = None
    if not method == "km":
        raise ValueError("Method must be either km or mbkm")


# load the data
with h5py.File(filepath, "r") as f:
    data_name = list(f.keys())[0]
    X = f[data_name][:, :]

# normalize the data if required
if preprocessing_flag:
    scaler = sklp.StandardScaler(copy=False)
    scaler.fit_transform(X)

# prepare arrays for measured times and functional values
times = np.zeros(n_samples)
values = np.zeros(n_samples)

# perform clustering
for i in range(n_samples):
    if method == "km":
        print(
            f"Run {i+1} of {n_samples}: sci-kit learn K-means with {n_clusters} clusters and {init} initialization"
        )
        clusterer = sklc.KMeans(n_clusters=n_clusters, init=init)
    else:
        print(
            f"Run {i+1} of {n_samples}: scikit-learn Mini-batch K-means with {n_clusters} clusters, {init} initializationon and batch size {batch_size}"
        )
        clusterer = sklc.MiniBatchKMeans(
            n_clusters=n_clusters, init=init, batch_size=batch_size, compute_labels=True
        )
    t0 = time.perf_counter()
    clusterer.fit(X)
    t1 = time.perf_counter()
    times[i] = t1 - t0
    if method == "km" or method == "mbkm":
        values[i] = clusterer.inertia_
    del clusterer

# save results
np.savetxt(
    f"{results_path}times_skl{method}_{n_clusters}_{batch_size}_{init}_{n_samples}_1_cpu.txt", times
)
np.savetxt(
    f"{results_path}values_skl{method}_{n_clusters}_{batch_size}_{init}_{n_samples}_1_cpu.txt",
    values,
)
