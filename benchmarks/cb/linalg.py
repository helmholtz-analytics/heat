# flake8: noqa
import itertools

import torchvision.datasets as datasets
from mpi4py import MPI
from perun import monitor

import heat as ht


@monitor()
def matmul_split_0(a, b):
    a @ b


@monitor()
def matmul_split_1(a, b):
    a @ b


@monitor()
def qr_split_0(a):
    for t in range(1, 3):
        qr = a.qr(tiles_per_proc=t)


@monitor()
def qr_split_1(a):
    for t in range(1, 3):
        qr = a.qr(tiles_per_proc=t)


@monitor()
def hierachical_svd_rank(data, r):
    approx_svd = ht.linalg.hsvd_rank(data, maxrank=r, compute_sv=True, silent=True)


@monitor()
def hierachical_svd_tol(data, tol):
    approx_svd = ht.linalg.hsvd_rtol(data, rtol=tol, compute_sv=True, silent=True)


@monitor()
def lanczos(B):
    V, T = ht.lanczos(B, m=B.shape[0])


def run_linalg_benchmarks():
    n = 3000
    a = ht.random.random((n, n), split=0)
    b = ht.random.random((n, n), split=0)
    matmul_split_0(a, b)
    del a, b

    a = ht.random.random((n, n), split=1)
    b = ht.random.random((n, n), split=1)
    matmul_split_1(a, b)
    del a, b

    n = 2000
    a_0 = ht.random.random((n, n), split=0)
    a_1 = ht.random.random((n, n), split=1)
    qr_split_0(a_0)
    qr_split_1(a_1)
    del a_0, a_1

    n = 50
    A = ht.random.random((n, n), dtype=ht.float64, split=0)
    B = A @ A.T
    lanczos(B)
    del A, B

    data = ht.utils.data.matrixgallery.random_known_rank(
        1000, 500 * MPI.COMM_WORLD.Get_size(), 10, split=1, dtype=ht.float32
    )[0]
    hierachical_svd_rank(data, 10)
    hierachical_svd_tol(data, 1e-2)
    del data
