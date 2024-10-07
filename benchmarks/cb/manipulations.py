# flake8: noqa
import heat as ht
from typing import List
from perun import monitor
from sizes import GSIZE_SQ, GSIZE_CB

"""
Bencharks so far:
- concatenation along split axis
- reshaping along split axis with new_split
- resplitting (of a split array)
- unsplit a split array
"""


@monitor()
def concatenate(arrays):
    a = ht.concatenate(arrays, axis=1)


@monitor()
def concatenate_nosplit(arrays):
    a = ht.concatenate(arrays, axis=1)


@monitor()
def reshape(array):
    a = ht.reshape(array, (array.shape[0] * array.shape[1], -1), new_split=1)


@monitor()
def reshape_nosplit(array):
    a = ht.reshape(array, (array.shape[0] * array.shape[1], -1), new_split=1)


@monitor()
def resplit(array):
    a = ht.resplit(array, axis=1)


@monitor()
def unsplit(array):
    a = ht.resplit(array, axis=None)


def run_manipulation_benchmarks():
    arrays = [
        ht.zeros((GSIZE_SQ // 2, GSIZE_SQ), split=1),
        ht.zeros((GSIZE_SQ // 2, GSIZE_SQ), split=1),
    ]
    concatenate(arrays)
    del arrays

    arrays = [
        ht.zeros((GSIZE_SQ // 2, GSIZE_SQ), split=0),
        ht.zeros((GSIZE_SQ // 2, GSIZE_SQ), split=0),
    ]
    concatenate_nosplit(arrays)
    del arrays

    array = ht.zeros((GSIZE_CB, GSIZE_CB, GSIZE_CB), split=0)
    reshape(arrays)
    del array

    array = ht.zeros((GSIZE_CB, GSIZE_CB, GSIZE_CB), split=2)
    reshape_nosplit(arrays)
    del array

    array = ht.ones((GSIZE_SQ, GSIZE_SQ), split=0)
    resplit(array)

    array = ht.ones((GSIZE_SQ, GSIZE_SQ), split=0)
    unsplit(array)
