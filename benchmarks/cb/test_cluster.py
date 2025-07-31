# flake8: noqa
import os

import heat as ht

ht.use_device("gpu")
ht.random.seed(12345)

world_size = ht.MPI_WORLD.size
rank = ht.MPI_WORLD.rank
print(f"{rank}/{world_size}: Working on {ht.get_device()}")

from cluster import run_cluster_benchmarks

run_cluster_benchmarks()

print("Done")
