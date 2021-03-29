#!/usr/bin/env python

import argparse
import h5py
import numpy as np
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NumPy statistical moments cpu benchmark")
    parser.add_argument("--file", type=str, help="file to benchmark")
    parser.add_argument("--dataset", type=str, help="dataset within file to benchmark")
    parser.add_argument("--trials", type=int, help="number of benchmark trials")
    args = parser.parse_args()

    print("Loading data... {}[{}]".format(args.file, args.dataset), end="")
    with h5py.File(args.file, "r") as handle:
        data = np.array(handle[args.dataset])
    print("\t[OK]")

    for function in [np.mean, np.std]:
        for axis in [None, 0, 1]:
            print("{} axis={}".format(function, axis))
            for trial in range(args.trials):
                print("Trial {}...".format(trial), end="")
                start = time.perf_counter()
                function(data, axis=axis)
                end = time.perf_counter()
                print("\t{}s".format(end - start))
