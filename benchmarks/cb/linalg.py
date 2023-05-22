# flake8: noqa
import heat as ht
import torchvision.datasets as datasets
from perun.decorator import monitor


@monitor()
def matmul_cpu_split_0(n: int = 3000):
    a = ht.random.random((n, n), split=0, device="cpu")
    b = ht.random.random((n, n), split=0, device="cpu")
    a @ b


@monitor()
def matmul_cpu_split_1(n: int = 3000):
    a = ht.random.random((n, n), split=1, device="cpu")
    b = ht.random.random((n, n), split=1, device="cpu")
    a @ b


# @monitor()
# def qr_cpu(n: int = 2000):
#     for t in range(1, 3):
#         for sp in range(2):
#             a = ht.random.random((n, n), split=sp)
#             qr = a.qr(tiles_per_proc=t)


@monitor()
def lanczos_cpu(n: int = 50):
    A = ht.random.random((n, n), dtype=ht.float64, split=0)
    B = A @ A.T
    V, T = ht.lanczos(B, m=n)


@monitor()
def hierachical_svd_rank(data, r):
    approx_svd = ht.linalg.hsvd_rank(data, maxrank=r, compute_sv=True, silent=True)


@monitor()
def hierachical_svd_tol(data, tol):
    approx_svd = ht.linalg.hsvd_rtol(data, rtol=tol, compute_sv=True, silent=True)


matmul_cpu_split_0()
matmul_cpu_split_1()
# qr_cpu()
lanczos_cpu()

mnist = (
    datasets.MNIST(root="./data", train=True, download=True, transform=None)
    .data.reshape((60000, 28 * 28))
    .T
)
mnist = ht.array(mnist, dtype=ht.float32, split=None, device="cpu").resplit_(1)

hierachical_svd_rank(mnist, 10)
hierachical_svd_tol(mnist, 1e-2)
