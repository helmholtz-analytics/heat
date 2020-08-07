#!/usr/bin/env python

import argparse
import heat as ht
import time

from pypapi import papi_high
from pypapi import events as papi_events

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
        papi_high.start_counters([papi_events.PAPI_SP_OPS, papi_events.PAPI_TOT_INS])
        start = time.perf_counter()
        lasso.fit(data, labels)
        end = time.perf_counter()
        result = papi_high.stop_counters()
        print("\t{}s {} flops32 {} ops".format(end - start, result[0], result[1]))
