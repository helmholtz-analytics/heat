# flake8: noqa
import os

import heat as ht

ht.use_device(os.environ["HEAT_DEVICE"] if os.environ["HEAT_DEVICE"] else "cpu")
ht.random.seed(12345)

world_size = ht.MPI_WORLD.size
rank = ht.MPI_WORLD.rank
print(f"{rank}/{world_size}: Working on {ht.get_device()}")

from linalg import run_linalg_benchmarks
from cluster import run_cluster_benchmarks
from manipulations import run_manipulation_benchmarks
from preprocessing import run_preprocessing_benchmarks
from decomposition import run_decomposition_benchmarks
from heat_signal import run_signal_benchmarks

run_linalg_benchmarks()
run_cluster_benchmarks()
run_manipulation_benchmarks()
run_preprocessing_benchmarks()
run_decomposition_benchmarks()
run_signal_benchmarks()
