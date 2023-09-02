# flake8: noqa
import os

import heat as ht

ht.use_device(os.environ["HEAT_DEVICE"] if os.environ["HEAT_DEVICE"] else "cpu")
ht.random.seed(12345)

from cluster import run_cluster_benchmarks
from linalg import run_linalg_benchmarks
from manipulations import run_manipulation_benchmarks

run_linalg_benchmarks()
run_cluster_benchmarks()
run_manipulation_benchmarks()
