# flake8: noqa
import heat as ht
from mpi4py import MPI
from perun import monitor
from sizes import GSIZE_TS_L, GSIZE_TS_S, GSIZE_SQ, LANCZOS_SIZE

"""
Benchmarks in this file:
- matrix-matrix multiplication of square matrices with different splits (00,11,01,10)
- QR decomposition of
    - tall-skinny matrix with split=0
    - square matrix with split=1
- Lanczos algorithm on a square matrix of size 1000 x 1000
- Hierarchical SVD with fixed rank or fixed tolerance for a short-fat matrix of rank 10 and split=1
- Full SVD with tall-skinny matrix and split=0
"""


@monitor()
def matmul_split_0(a, b):
    a @ b


@monitor()
def matmul_split_1(a, b):
    a @ b


@monitor()
def matmul_split_01(a, b):
    a @ b


@monitor()
def matmul_split_10(a, b):
    a @ b


@monitor()
def qr_split_0(a):
    qr = ht.linalg.qr(a)


@monitor()
def qr_split_1(a):
    qr = ht.linalg.qr(a)


@monitor()
def hierachical_svd_rank(data, r):
    approx_svd = ht.linalg.hsvd_rank(data, maxrank=r, compute_sv=True, silent=True)


@monitor()
def hierachical_svd_tol(data, tol):
    approx_svd = ht.linalg.hsvd_rtol(data, rtol=tol, compute_sv=True, silent=True)


@monitor()
def svd_full_ts(data):
    svd = ht.linalg.svd(data)


@monitor()
def lanczos(B):
    V, T = ht.lanczos(B, m=B.shape[0])


def run_linalg_benchmarks():
    n = GSIZE_SQ
    a = ht.random.random((n, n), split=0)
    b = ht.random.random((n, n), split=0)
    matmul_split_0(a, b)
    del a, b

    a = ht.random.random((n, n), split=1)
    b = ht.random.random((n, n), split=1)
    matmul_split_1(a, b)
    del a, b

    a = ht.random.random((n, n), split=0)
    b = ht.random.random((n, n), split=1)
    matmul_split_01(a, b)
    del a, b

    a = ht.random.random((n, n), split=1)
    b = ht.random.random((n, n), split=0)
    matmul_split_10(a, b)
    del a, b

    n = GSIZE_TS_S
    m = GSIZE_TS_L
    a_0 = ht.random.random((m, n), split=0)
    qr_split_0(a_0)
    del a_0

    n = GSIZE_SQ
    a_1 = ht.random.random((n, n), split=1)
    qr_split_1(a_1)
    del a_1

    A = ht.random.random((LANCZOS_SIZE, LANCZOS_SIZE), dtype=ht.float64, split=0)
    B = A @ A.T
    lanczos(B)
    del A, B

    data = ht.utils.data.matrixgallery.random_known_rank(
        GSIZE_TS_S, GSIZE_TS_L, 10, split=1, dtype=ht.float32
    )[0]
    hierachical_svd_rank(data, 10)
    hierachical_svd_tol(data, 1e-2)
    del data

    data = ht.random.random((GSIZE_TS_L, GSIZE_TS_S), split=0)
    svd_full_ts(data)
    del data
