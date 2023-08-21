# flake8: noqa
import heat as ht
from perun.decorator import monitor


@monitor()
def reshape_cpu():
    sizes = [10000, 20000, 40000]
    for size in sizes:
        st = ht.zeros((1000, size), split=1)
        a = ht.reshape(st, (10000000, -1), new_split=1)


reshape_cpu()
