# flake8: noqa
import heat as ht
from perun import monitor


@monitor()
def reshape():
    sizes = [10000, 20000, 40000]
    for size in sizes:
        st = ht.zeros((1000, size), split=1)
        a = ht.reshape(st, (10000000, -1), new_split=1)


def run_manipulation_benchmarks():
    reshape()
