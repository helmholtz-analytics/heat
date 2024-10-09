# flake8: noqa
import os

import heat as ht

ht.use_device(os.environ["HEAT_DEVICE"] if os.environ["HEAT_DEVICE"] else "cpu")
ht.random.seed(12345)

from linalg import run_linalg_benchmarks
from cluster import run_cluster_benchmarks
from manipulations import run_manipulation_benchmarks
from preprocessing import run_preprocessing_benchmarks
from sizes import *

if ht.MPI_WORLD.rank == 0:
    print("elements per process:", N_ELEMENTS_PER_PROC)
    print("GB per process:", N_ELEMENTS_PER_PROC * 4 / 1e9)
    print("elements total:", N_ELEMENTS_TOTAL)
    print("GB total:", N_ELEMENTS_TOTAL * 4 / 1e9)

    print(f"square shape: {GSIZE_SQ} x {GSIZE_SQ}")
    print(f"cube shape: {GSIZE_CB} x {GSIZE_CB} x {GSIZE_CB}")
    print(f"tall-skinny shape: {GSIZE_TS_L} x {GSIZE_TS_S}")
    print(f"very tall-skinny shape: {GSIZE_vTS_L} x {GSIZE_vTS_S}")
    print(f"Lanczos size: {LANCZOS_SIZE} x {LANCZOS_SIZE}")

run_linalg_benchmarks()
print("linalg benchmarks done")

run_cluster_benchmarks()
print("cluster benchmarks done")

run_manipulation_benchmarks()
print("manipulation benchmarks done")

run_preprocessing_benchmarks()
print("preprocessing benchmarks done")
