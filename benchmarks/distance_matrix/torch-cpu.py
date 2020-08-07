#!/usr/bin/env python

import argparse
import h5py
import torch
import time

from pypapi import papi_high
from pypapi import events as papi_events

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch distance matrix cpu benchmark")
    parser.add_argument("--file", type=str, help="file to benchmark")
    parser.add_argument("--dataset", type=str, help="dataset within file to benchmark")
    parser.add_argument("--trials", type=int, help="number of benchmark trials")
    args = parser.parse_args()

    print("Loading data... {}[{}]".format(args.file, args.dataset), end="")
    with h5py.File(args.file, "r") as handle:
        data = torch.tensor(handle[args.dataset], device="cpu")
    print("\t[OK]")

    for trial in range(args.trials):
        print("Trial {}...".format(trial), end="")
        papi_high.start_counters([papi_events.PAPI_DP_OPS, papi_events.PAPI_TOT_INS])
        start = time.perf_counter()
        dist = torch.cdist(data, data)
        end = time.perf_counter()
        result = papi_high.stop_counters()
        print("\t{}s {} flops64 {} ops".format(end - start, result[0], result[1]))
