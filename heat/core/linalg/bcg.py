"""
Bidiagonalization of input DNDarray.
"""
import itertools
from operator import imod
from turtle import left
import torch
import math
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
    this is done using the householder reflectors.

    If the matrix A is complete (Full) then the usual procedure for bidiagonal
    reduction is to apply "Full-length => Householder transformations
    alternatively from the left and right end.

    The reduction algorithm used here is based on the Householder transformations
    If the transformation matrices U1 & V1 are also needed, then this algorithm is
    very useful.

    Our matrix A denotes a matrix of size m X n

    Let k = min(m,n);
    A is reduced to a bi-diagonal form. So, The reduction proceeds in
    k = steps, each of which produces one row of the
    resulting in bi-diagonal matrix B.

    bi_diaonalize(A) -> returns a matrix (B), which is a bidiagonal matrix

    With the use of 3 functions gen_house_vec(x), apply_house_left(), apply_house_right() we change the input matrix into a bidiagonal matrix
    We are not returning U1, Vt1 now.
    But U1 & vt1 might be useful at the end to calculate the U,V.Transpose() matrices in the equation
    svd(arr) = U,sigma,V.Transpose()  for the final svd calculation.

    Currently, the algorithm is working fine, but the algorithm can be further optimized.
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

    To change the serial algorithm into an Efficient parallel program, we do 3 steps:
    1. We should distribute the data and the computational work to multiple processors.
    2. Then, a 'local view' of the block decomposition will lead to a pipelined parallel algorithm.
    3.



    The band form is further reduced to the final condensed form using the bulge chasing technique.
    This procedure annihilates the extra off-diagonal elements by chasing the created
    fill-in elements down to the bottom right side of the matrix using successive
    orthogonal transformations at each sweep.



    Three Kernels are used to implement this algorithm:

    xGBLER kernel: This kernel triggers the beginning of each sweep by successive element-wise
                   annihilations of the extra non-zero entries within a single column, It then
                   applies all the left updates creating single bulges, which have to be immediately
                   annihilated and then followed by the right updates on the corresponding
                   data block loaded into the cache memory.


    xGBRCE kernel: This kernel successively applies all the right updates coming from the
                   previous kernels, either xGBELR or xGBLRX (described below). This subsequently
                   generates single bulges, which have to be immediately annihilated by appropriate
                   left transformations in order to eventually avoid an expansion of the
                   fill-in structure (Figure 3(b)) by subsequent orthogonal transformations.

    xGBLRX kernel: This kernel successively applies all the left updates coming from the
                   xGBRCE kernel and create single bulge out of the diagonal, then similar
                   to xGBELR, it eliminate the bulge and apply the corresponding right updates.


    Note: We will do the bulge chasing using Halos.


    """
    if overwrite_arr:
        arr = A
        # The input matrix is overwritten with the result, i.e it will be changed to a bidiagonal matrix.
    else:
        arr = A.copy()
        # The input matrix is not overwritten, i.e a new matrix "arr" which is a copy of the input matrix, will be changed to a bidiagonal matrix.

    m, n = arr.shape
    k = min(m, n)
    # k is the minimum of m and n

    # Find the width of the diagonal of the input matrix.

    diag_width = 1
    row_0 = arr[0, 0:]
    for i in range(1, len(row_0)):
        if row_0[i] != 0:
            diag_width += 1

    print("diag_width: ", diag_width)

    U1, vt1 = ht.eye(m, dtype=ht.float64), ht.eye(n, dtype=ht.float64)
    # U1 is an identity matrix of size m x m, vt1 is an identity matrix of size n x n

    for i in range(k):
        # A = arr._DNDarray__prephalo(i)
        v_left, tau_left = gen_house_vec(arr[i : i + diag_width, i])
        # All the elements in the ith column upto index = width of the diagonal in the band matrix, below arr[i][i] including itself, are send to the "gen_house_vec" function.
        apply_house_left(
            arr[i : i + diag_width, i : (i + diag_width + 1)], v_left, tau_left, U1, m, i
        )

        if i <= n - 2:
            v_right, tau_right = gen_house_vec(torch.t(arr[i, i + 1 : i + diag_width]))
            # All the elements in the ith row upto index = width of the diagonal in the band matrix, the right of arr[i][i] including itself, are send to the "gen_house_vec" function.
            apply_house_right(
                arr[i : (i + diag_width + 1), i + 1 : (i + 1 + diag_width)],
                v_right,
                tau_right,
                vt1,
                n,
                i,
            )

    return arr


# mpiexec -np 3 python c:/Users/DELL/heat/heat/core/linalg/bcg.py
# arr = ht.zeros([15,12], dtype=ht.float64)
# print("Hello world from rank", str(rank), "of", str(size))

# a = ht.array([[0.2677, 0.7491, 0.5088, 0.4953, 0.0959, 0.1744],
#         [0.0341, 0.3601, 0.0869, 0.2640, 0.2803, 0.1916],
#         [0.1342, 0.5625, 0.1345, 0.8248, 0.9556, 0.9317],
#         [0.7166, 0.1113, 0.9824, 0.4516, 0.0804, 0.8889],
#         [0.7074, 0.1604, 0.6801, 0.2890, 0.8342, 0.7405]], dtype=ht.float64,split=None)
# print("Input matrix:", a, sep = "\n")
# a = a.larray

# U1,B1,Vt1 = bi_diagonalize(a)

# print("Matrix U1 is: ", U1)
# print("Matrix B1 is: ", B1)
# print("Matrix Vt1 is: ", Vt1)
# k = (U1 @ B1 @ Vt1)
# print(k)
# ht.local_printing()
# array_with_halos
# print(a.get_halo(1))
# print(a.halo_next)
# print(a.halo_prev)

a = ht.random.rand(150, dtype=ht.float64, split=0)
a = a.reshape(10, 15)
m, n = a.shape
b = a
c = a

print("Input matrix:", a, sep="\n")
# print(a.get_halo(5))
# print(a.halo_prev)
# print(a.halo_next)
a = a._DNDarray__prephalo(0, math.floor(m / 3))
b = b._DNDarray__prephalo(math.floor(m / 3), math.floor(2 * m / 3))
c = c._DNDarray__prephalo(math.floor(2 * m / 3), m)
print(a)
# print(b)
# print(c)

bi_diagonalize(a)
print(a)
