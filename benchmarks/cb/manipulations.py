# flake8: noqa
import heat as ht
from typing import List
from perun import monitor


@monitor()
def concatenate(arrays):
    # benchmark concatenation of 3 arrays with split 1, None, 1 respectively
    a = ht.concatenate(arrays, axis=1)


@monitor()
def reshape(arrays):
    for array in arrays:
        a = ht.reshape(array, (10000000, -1), new_split=1)


@monitor()
def resplit(array, new_split: List[int | None]):
    for new_split in new_split:
        a = ht.resplit(array, axis=new_split)
        del a


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

    if ht.comm.size > 1:
        shape = [100, 50, 50, 20, 86]
        n_elements = ht.array(shape).prod().item()
        mem = n_elements * 4 / 1e9
        array = ht.reshape(ht.arange(0, n_elements, split=0, dtype=ht.float32), shape) * (
            ht.comm.rank + 1
        )

        resplit(array, [None, 2, 4])
