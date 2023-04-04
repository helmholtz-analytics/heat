import heat as ht
from perun.decorator import monitor


@monitor()
def matmul_cpu_split_0(n: int = 1000) -> ht.DNDarray:
    a = ht.random.random((n, n), split=0, device="cpu")
    b = ht.random.random((n, n), split=0, device="cpu")
    return a @ b


@monitor()
def matmul_cpu_split_1(n: int = 1000) -> ht.DNDarray:
    a = ht.random.random((n, n), split=1, device="cpu")
    b = ht.random.random((n, n), split=1, device="cpu")
    return a @ b


if __name__ == "__main__":
    matmul_cpu_split_0()
    matmul_cpu_split_1()
