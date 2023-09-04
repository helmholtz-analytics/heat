# flake8: noqa
import heat as ht
from perun import monitor


@monitor()
def reshape(arrays):
    for array in arrays:
        a = ht.reshape(array, (10000000, -1), new_split=1)


def run_manipulation_benchmarks():
    sizes = [10000, 20000, 40000]
    arrays = []
    for size in sizes:
        arrays.append(ht.zeros((1000, size), split=1))

    reshape(arrays)
