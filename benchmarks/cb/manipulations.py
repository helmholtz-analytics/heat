# flake8: noqa
import heat as ht
from perun.decorator import monitor


@monitor()
def reshape_cpu(arrays):
    for array in arrays:
        a = ht.reshape(array, (10000000, -1), new_split=1)


sizes = [10000, 20000, 40000]
arrays = []
for size in sizes:
    arrays.append(ht.zeros((1000, size), split=1))

reshape_cpu(arrays)
