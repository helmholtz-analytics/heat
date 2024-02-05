# flake8: noqa
import heat as ht
from mpi4py import MPI
from perun import monitor


@monitor()
def matmul_split_0(a, b):
    a @ b


@monitor()
def matmul_split_1(a, b):
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

    n = int((4000000 // MPI.COMM_WORLD.size) ** 0.5)
    m = MPI.COMM_WORLD.size * n
    a_1 = ht.random.random((m, n), split=0)
    qr_split_0(a_0)
    del a_0

    n = 2000
    a_1 = ht.random.random((n, n), split=1)
    qr_split_1(a_1)
    del a_1

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
