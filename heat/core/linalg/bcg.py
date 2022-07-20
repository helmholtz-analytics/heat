"""
Bidiagonalization of input DNDarray.
"""
import itertools
from operator import imod
from turtle import left
import numpy as np
import torch
import warnings
import heat as ht

from mpi4py import MPI

from utils import (
    apply_house,
    apply_house_left,
    apply_house_right,
    gen_house_mat,
    gen_house_vec,
)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

from typing import List, Callable, Union, Optional, Tuple, TypeVar

from torch._C import Value
from inspect import stack
from mpi4py import MPI
from pathlib import Path

from heat.core.devices import cpu

from heat.core import communication
from heat.core import arithmetics
from heat.core import complex_math
from heat.core import constants
from heat.core import exponential
from heat.core import dndarray
from heat.core import factories
from heat.core import manipulations
from heat.core.manipulations import *
from heat.core import rounding
from heat.core import sanitation
from heat.core import statistics
from heat.core import stride_tricks
from heat.core import types
from svd import *
from qr import *

__all__ = ["bi_diagonalize"]


def bi_diagonalize(A, overwrite_arr=True):
    """
    Updating the given input A matrix into a bidiagonal matrix,
    this is done using the house holder reflectors.

    If the matrix A is full then the usual procedure for bidiagonal
    reduction is to apply "Full length => Householder transformations
    alternatively from left and right end.

    The reduction algorithm used here is based on the Householder transformations
    If the transformation matrices U1 & V1 are also needed, then this algo is
    very useful.

    Our matrix A denotes a matrix of size m X n

    Let k = min(m,n);
    A is reduced to bi-diagonal form. So, The reduction proceeds in
    k = steps, each of which produces one row of the
    resulting bi-diagonal matrix B.

    bi_diaonalize(A) -> returns a matrix (B), which is a bidiagonal matrix

    With the use of 3 functions gen_house_vec(x), apply_house_left(), apply_house_right() we change the input matrix into a bidiagonal matrix
    We are not returning U1,Vt1 now.
    But U1 & vt1 might be useful at the end to calculate the U,V.Transpose() matrices in the equation
    svd(arr) = U,sigma,V.Transpose()  for the final svd calculation.

    As of now the algorithm is working fine, but algorithm can be further optimized.
    Using the fact that we will apply this algorithm to a band matrix which we get after using the function ht.block_diagonalize(arr)


    Parameters
    ----------
    A : ht.DNDarray
        2D input matrix (m x n)

    overwrite_arr : bool, Optional
        Default: True
        if True, will overwrite the input matrix A into a bi-diagonal matrix.
        if False, will return a new bi-diagonal matrix,


    Returns
    -------
    result : DNDarray

        B : ht.DNDarray

    """
    if overwrite_arr:
        arr = A
        # The input matrix is overwritten with the result, i.e it will be changed to a bidiagonal matrix.
    else:
        arr = A.copy()
        # The input matrix is not overwritten, i.e a new matrix "arr" which is a copy of input matrix, will be changed to a bidiagonal matrix.

    m, n = arr.shape
    k = min(m, n)
    # k is the minimum of m and n

    U1, vt1 = ht.eye(m, dtype=ht.float64), ht.eye(n, dtype=ht.float64)
    # U1 is an identity matrix of size m x m, vt1 is an identity matrix of size n x n

    for i in range(k):
        v_left, tau_left = gen_house_vec(arr[i:, i])
        # All the elements in the ith column below arr[i][i] including itself, are send to the "gen_house_vec" function.
        apply_house_left(arr[i:, i:], v_left, tau_left, U1, m, i)

        if i <= n - 2:
            v_right, tau_right = gen_house_vec(torch.t(arr[i, i + 1 :]))
            # All the elements in the ith row to the right of arr[i][i] including itself, are send to the "gen_house_vec" function.
            apply_house_right(arr[i:, i + 1 :], v_right, tau_right, vt1, n, i)

    return arr


# arr = ht.zeros([15,12], dtype=ht.float64)
a = ht.random.rand(30, dtype=ht.float64)
a = a.reshape(5, 6)
# print("Input matrix:", a, sep = "\n")
# a = a.larray
print("Input matrix:", a, sep="\n")
# U1,B1,Vt1 = bi_diagonalize(a)
bi_diagonalize(a.larray)
# print("Matrix U1 is: ", U1)
# print("Matrix B1 is: ", B1)
# print("Matrix Vt1 is: ", Vt1)
# k = (U1 @ B1 @ Vt1)
# print(k)
print(a)
# print(B1)
