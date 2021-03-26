#!/usr/bin/env python

import argparse
import heat as ht
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HeAT kmeans gpu benchmark")
    parser.add_argument("--file", type=str, help="file to benchmark")
    parser.add_argument("--dataset", type=str, help="dataset within file to benchmark")
    parser.add_argument("--trials", type=int, help="number of benchmark trials")
    parser.add_argument("--clusters", type=int, help="number of cluster centers")
    parser.add_argument("--iterations", type=int, help="kmeans iterations")
    args = parser.parse_args()

    ht.use_device("gpu")

    print("Loading data... {}[{}]".format(args.file, args.dataset), end="")
    data = ht.load(args.file, dataset=args.dataset, split=0)
    print("\t[OK]")

    for trial in range(args.trials):
        print("Trial {}...".format(trial), end="")
        kmeans = ht.cluster.KMeans(n_clusters=args.clusters, max_iter=args.iterations)
        start = time.perf_counter()
        kmeans.fit(data)
        end = time.perf_counter()
        print("\t{}s".format(end - start))
