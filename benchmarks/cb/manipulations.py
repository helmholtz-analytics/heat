# flake8: noqa
import heat as ht
from perun import monitor


@monitor()
def concatenate(arrays):
    # benchmark concatenation of 3 arrays with split 1, None, 1 respectively
    a = ht.concatenate(arrays, axis=1)


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

    arrays = []
    for i, size in enumerate(sizes):
        if i == 1:
            split = None
        else:
            split = 1
        arrays.append(ht.zeros((1000, size), split=split))
    concatenate(arrays)
