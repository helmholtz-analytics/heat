import heat as ht
import argparse
import time
import numpy as np

# Create a parser object
parser = argparse.ArgumentParser(
    prog="Demo for Batch-Parallel K-Means Clustering in Heat",
    description="Demo for Batch-Parallel K-Means Clustering in Heat",
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
device = args.device
preprocessing_flag = args.preprocessing
n_samples = args.n_samples
n_procs = ht.MPI_WORLD.size


# cath possible errors
# batch-parallel clustering only allows k-means++ initialization
if method == "bpkm" and init != "++":
    raise ValueError("Batch-parallel clustering only allows k-means++ initialization")
if method == "bpkm" and n_merges == 1:
    raise ValueError("n_merges must be 0 (i.e. None) or > 1 for batch-parallel clustering")

# load the data
X = ht.load(filepath, split=0, device=device)

# normalize the data if required
if preprocessing_flag:
    scaler = ht.preprocessing.StandardScaler(copy=False)
    scaler.fit_transform(X)

# perform clustering
times = np.zeros(n_samples)
values = np.zeros(n_samples)
