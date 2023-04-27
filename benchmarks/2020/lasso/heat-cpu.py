#!/usr/bin/env python

import argparse
import heat as ht
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HeAT lasso cpu benchmark")
    parser.add_argument("--file", type=str, help="file to benchmark")
    parser.add_argument("--dataset", type=str, help="dataset within file to benchmark")
    parser.add_argument("--labels", type=str, help="dataset within file pointing to the labels")
    parser.add_argument("--trials", type=int, help="number of benchmark trials")
    parser.add_argument("--iterations", type=int, help="iterations")
    args = parser.parse_args()

    print("Loading data... {}[{}]".format(args.file, args.dataset), end="")
    data = ht.load(args.file, args.dataset, split=0)
    labels = ht.load(args.file, args.labels, split=0)
    print("\t[OK]")

    for trial in range(args.trials):
        print("Trial {}...".format(trial), end="")
        lasso = ht.regression.Lasso(max_iter=args.iterations, tol=-1.0)
        start = time.perf_counter()
        lasso.fit(data, labels)
        end = time.perf_counter()
        print("\t{}s".format(end - start))
