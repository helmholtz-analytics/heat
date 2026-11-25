# flake8: noqa
import os

import heat as ht
from perun import monitor


@monitor()
def isin_both_split(elements, test_elements):
    ht.isin(elements, test_elements)


@monitor()
def isin_both_split_invert(elements, test_elements):
    ht.isin(elements, test_elements, invert=True)


@monitor()
def isin_test_replicated(elements, test_elements):
    ht.isin(elements, test_elements.resplit_(None))


@monitor()
def in1d(elements, test_elements):
    ht.in1d(elements, test_elements)


def run_logical_benchmarks():
    """
    Benchmarks for ht.isin() and ht.in1d() with distributed test values.

    Sizes are controlled via environment variables; the defaults already span
    several gigabytes of distributed input:
      HEAT_BENCHMARK_SIZE      number of element values      (default 1e8,  0.8 GB int64)
      HEAT_BENCHMARK_TEST_SIZE number of test values         (default 1e7,  0.08 GB int64)
    For the design point of 1e9+ elements (8 GB per operand) set
    HEAT_BENCHMARK_SIZE=1000000000 and run on a cluster; per-rank memory of
    ht.isin scales with the local shards of both inputs, roughly
    ``7 * (size + test_size) * 8 / comm.size`` bytes at the peak.
    """
    n = int(os.environ.get("HEAT_BENCHMARK_SIZE", 100_000_000))
    m = int(os.environ.get("HEAT_BENCHMARK_TEST_SIZE", 10_000_000))

    # deterministic pseudo-random element values in [0, 10 * n)
    elements = (ht.arange(n, split=0, dtype=ht.int64) * 1103515245) % (10 * n)
    test_elements = ht.random.randint(0, 10 * n, (m,), split=0, dtype=ht.int64)

    isin_both_split(elements, test_elements)
    isin_both_split_invert(elements, test_elements)
    in1d(elements, test_elements)
    # the replicated path only makes sense for moderate test sets
    if m <= 20_000_000:
        isin_test_replicated(elements, test_elements)
