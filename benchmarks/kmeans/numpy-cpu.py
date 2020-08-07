#!/usr/bin/env python

import argparse
import h5py
import numpy as np
import time

from pypapi import papi_high
from pypapi import events as papi_events
from sklearn.metrics import pairwise_distances


class KMeans:
    def __init__(self, n_clusters=8, init="random", max_iter=300, tol=-1.0):
        self.init = init
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.tol = tol

        self._inertia = float("nan")
        self._cluster_centers = None

    def _initialize_centroids(self, x):
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        indices = indices[: self.n_clusters]
        self._cluster_centers = x[indices]

    def _fit_to_cluster(self, x):
        distances = pairwise_distances(x, self._cluster_centers)
        matching_centroids = distances.argmin(axis=1)

        return matching_centroids

    def fit(self, x):
        self._initialize_centroids(x)
        new_cluster_centers = self._cluster_centers.copy()

        # iteratively fit the points to the centroids
        for _ in range(self.max_iter):
            # determine the centroids
            matching_centroids = self._fit_to_cluster(x)

            # update the centroids
            for i in range(self.n_clusters):
                # points in current cluster
                selection = (matching_centroids == i).astype(np.int64).reshape(-1, 1)

                # accumulate points and total number of points in cluster
                assigned_points = (x * selection).sum(axis=0)
                points_in_cluster = selection.sum(axis=0).clip(1.0, np.iinfo(np.int64).max)

                # compute the new centroids
                new_cluster_centers[i : i + 1, :] = assigned_points / points_in_cluster

            # check whether centroid movement has converged
            self._inertia = ((self._cluster_centers - new_cluster_centers) ** 2).sum()
            self._cluster_centers = new_cluster_centers.copy()
            if self.tol is not None and self._inertia <= self.tol:
                break

        return self


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NumPy kmeans cpu benchmark")
    parser.add_argument("--file", type=str, help="file to benchmark")
    parser.add_argument("--dataset", type=str, help="dataset within file to benchmark")
    parser.add_argument("--trials", type=int, help="number of benchmark trials")
    parser.add_argument("--clusters", type=int, help="number of cluster centers")
    parser.add_argument("--iterations", type=int, help="kmeans iterations")
    args = parser.parse_args()

    print("Loading data... {}[{}]".format(args.file, args.dataset), end="")
    with h5py.File(args.file, "r") as handle:
        data = np.array(handle[args.dataset])
    print("\t[OK]")

    for trial in range(args.trials):
        print("Trial {}...".format(trial), end="")
        kmeans = KMeans(n_clusters=args.clusters, max_iter=args.iterations)
        papi_high.start_counters([papi_events.PAPI_DP_OPS, papi_events.PAPI_TOT_INS])
        start = time.perf_counter()
        kmeans.fit(data)
        end = time.perf_counter()
        result = papi_high.stop_counters()
        print("\t{}s {} flops64 {} ops".format(end - start, result[0], result[1]))
