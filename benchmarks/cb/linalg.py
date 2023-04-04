# flake8: noqa
import heat as ht
from perun.decorator import monitor


@monitor()
def matmul_cpu_split_0(n: int = 5000):
    a = ht.random.random((n, n), split=0, device="cpu")
    b = ht.random.random((n, n), split=0, device="cpu")
    a @ b


@monitor()
def matmul_cpu_split_1(n: int = 5000):
    a = ht.random.random((n, n), split=1, device="cpu")
    b = ht.random.random((n, n), split=1, device="cpu")
    a @ b


@monitor()
def qr_cpu(n: int = 3000):
    for t in range(1, 3):
        for sp in range(2):
            a = ht.random.random((n, n), split=sp)
            qr = a.qr(tiles_per_proc=t)


@monitor()
def lanczos_cpu(n: int = 500):
    A = ht.random.random((n, n), dtype=ht.float64, split=0)
    B = A @ A.T
    V, T = ht.lanczos(B, m=n)


if __name__ == "__main__":
    matmul_cpu_split_0()
    matmul_cpu_split_1()
    qr_cpu()
    lanczos_cpu()
