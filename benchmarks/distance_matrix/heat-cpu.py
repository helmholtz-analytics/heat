#!/usr/bin/env python

import argparse
import h5py
import heat as ht
import time

from pypapi import papi_high
from pypapi import events as papi_events

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HeAT distance matrix cpu benchmark")
    parser.add_argument("--file", type=str, help="file to benchmark")
    parser.add_argument("--dataset", type=str, help="dataset within file to benchmark")
    parser.add_argument("--trials", type=int, help="number of benchmark trials")
    args = parser.parse_args()

    ht.use_device("gpu")

    print("Loading data... {}[{}]".format(args.file, args.dataset), end="")
    data = ht.load(args.file, dataset=args.dataset, split=0)
    print("\t[OK]")

    print("quadratic_expansion=False")
    for trial in range(args.trials):
        print("Trial {}...".format(trial), end="")
        start = time.perf_counter()
        dist = ht.spatial.distance.cdist(data, data, quadratic_expansion=False)
        end = time.perf_counter()
        result = papi_high.stop_counters()
        print("\t{}s {} flops64 {} ops".format(end - start, result[0], result[1]))

    print("quadratic_expansion=True")
    for trial in range(args.trials):
        print("Trial {}...".format(trial), end="")
        start = time.perf_counter()
        dist = ht.spatial.distance.cdist(data, data, quadratic_expansion=True)
        end = time.perf_counter()
        result = papi_high.stop_counters()
        print("\t{}s {} flops64 {} ops".format(end - start, result[0], result[1]))
