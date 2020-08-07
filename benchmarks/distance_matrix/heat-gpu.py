#!/usr/bin/env python

import argparse
import heat as ht
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HeAT distance matrix gpu benchmark")
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
        print("\t{}s".format(end - start))

    print("quadratic_expansion=True")
    for trial in range(args.trials):
        print("Trial {}...".format(trial), end="")
        start = time.perf_counter()
        dist = ht.spatial.distance.cdist(data, data, quadratic_expansion=True)
        end = time.perf_counter()
        print("\t{}s".format(end - start))
