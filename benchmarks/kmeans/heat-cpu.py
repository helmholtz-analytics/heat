#!/usr/bin/env python

import argparse
import heat as ht
import time

from pypapi import papi_high
from pypapi import events as papi_events

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HeAT kmeans cpu benchmark")
    parser.add_argument("--file", type=str, help="file to benchmark")
    parser.add_argument("--dataset", type=str, help="dataset within file to benchmark")
    parser.add_argument("--trials", type=int, help="number of benchmark trials")
    parser.add_argument("--clusters", type=int, help="number of cluster centers")
    parser.add_argument("--iterations", type=int, help="kmeans iterations")
    args = parser.parse_args()

    print("Loading data... {}[{}]".format(args.file, args.dataset), end="")
    data = ht.load(args.file, dataset=args.dataset, split=0)
    print("\t[OK]")

    for trial in range(args.trials):
        print("Trial {}...".format(trial), end="")
        kmeans = ht.cluster.KMeans(n_clusters=args.clusters, max_iter=args.iterations)
        papi_high.start_counters([papi_events.PAPI_SP_OPS, papi_events.PAPI_TOT_INS])
        start = time.perf_counter()
        kmeans.fit(data)
        end = time.perf_counter()
        result = papi_high.stop_counters()
        print("\t{}s {} flops32 {} ops".format(end - start, result[0], result[1]))
