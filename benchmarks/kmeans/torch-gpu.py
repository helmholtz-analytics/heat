#!/usr/bin/env python

import argparse
import h5py
import time
import torch


class KMeans:
    def __init__(self, n_clusters=8, init="random", max_iter=300, tol=-1.0):
        self.init = init
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.tol = tol

        self._inertia = float("nan")
        self._cluster_centers = None

    def _initialize_centroids(self, x):
        indices = torch.randperm(x.shape[0])[: self.n_clusters]
        self._cluster_centers = x[indices]

    def _fit_to_cluster(self, x):
        distances = torch.cdist(x, self._cluster_centers)
        matching_centroids = distances.argmin(axis=1, keepdim=True)

        return matching_centroids

    def fit(self, x):
        self._initialize_centroids(x)
        new_cluster_centers = self._cluster_centers.clone()

        # iteratively fit the points to the centroids
        for _ in range(self.max_iter):
            # determine the centroids
            matching_centroids = self._fit_to_cluster(x)

            # update the centroids
            for i in range(self.n_clusters):
                # points in current cluster
                selection = (matching_centroids == i).type(torch.int64)

                # accumulate points and total number of points in cluster
                assigned_points = (x * selection).sum(axis=0, keepdim=True)
                points_in_cluster = selection.sum(axis=0, keepdim=True).clamp(
                    1.0, torch.iinfo(torch.int64).max
                )

                # compute the new centroids
                new_cluster_centers[i : i + 1, :] = assigned_points / points_in_cluster

            # check whether centroid movement has converged
            self._inertia = ((self._cluster_centers - new_cluster_centers) ** 2).sum()
            self._cluster_centers = new_cluster_centers.clone()
            if self.tol is not None and self._inertia <= self.tol:
                break

        return self


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch kmeans gpu benchmark")
    parser.add_argument("--file", type=str, help="file to benchmark")
    parser.add_argument("--dataset", type=str, help="dataset within file to benchmark")
    parser.add_argument("--trials", type=int, help="number of benchmark trials")
    parser.add_argument("--clusters", type=int, help="number of cluster centers")
    parser.add_argument("--iterations", type=int, help="kmeans iterations")
    args = parser.parse_args()

    print("Loading data... {}[{}]".format(args.file, args.dataset), end="")
    with h5py.File(args.file, "r") as handle:
        data = torch.tensor(handle[args.dataset], device="cuda")
    print("\t[OK]")

    for trial in range(args.trials):
        print("Trial {}...".format(trial), end="")
        kmeans = KMeans(n_clusters=args.clusters, max_iter=args.iterations)
        start = time.perf_counter()
        kmeans.fit(data)
        end = time.perf_counter()
        print("\t{}s".format(end - start))
