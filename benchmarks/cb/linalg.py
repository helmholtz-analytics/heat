# flake8: noqa
import heat as ht
from perun.decorator import monitor


@monitor()
def matmul_cpu_split_0(a, b):
    a @ b


@monitor()
def matmul_cpu_split_1(a, b):
    a @ b


@monitor()
def qr_cpu_split_0(a):
    for t in range(1, 3):
        qr = a.qr(tiles_per_proc=t)


@monitor()
def qr_cpu_split_1(a):
    for t in range(1, 3):
        qr = a.qr(tiles_per_proc=t)


@monitor()
def lanczos_cpu(B):
    V, T = ht.lanczos(B, m=B.shape[0])


n = 3000
a = ht.random.random((n, n), split=0, device="cpu")
b = ht.random.random((n, n), split=0, device="cpu")
matmul_cpu_split_0(a, b)
del a, b

a = ht.random.random((n, n), split=1, device="cpu")
b = ht.random.random((n, n), split=1, device="cpu")
matmul_cpu_split_1(a, b)
del a, b

n = 2000
a_0 = ht.random.random((n, n), split=0)
a_1 = ht.random.random((n, n), split=1)
qr_cpu_split_0(a_0)
qr_cpu_split_1(a_1)
del a_0, a_1

n = 50
A = ht.random.random((n, n), dtype=ht.float64, split=0)
B = A @ A.T
lanczos_cpu(B)
del B
