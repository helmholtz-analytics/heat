"""
Bidiagonalization of input DNDarray.
"""
from asyncio import wait_for
import itertools
from multiprocessing.connection import wait
from operator import imod
import re
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


from qr import __split0_r_calc, __split0_q_loop, __split1_qr_loop
from heat.core import tiling


from heat.core.devices import cpu

from heat.core import communication
from heat.core import arithmetics
from heat.core import complex_math
from heat.core import constants
from heat.core import exponential
from heat.core import dndarray
from heat.core import factories
from heat.core import manipulations
from heat.core.manipulations import resplit
from heat.core import rounding
from heat.core import sanitation
from heat.core import statistics
from heat.core import stride_tricks
from heat.core import types
from svd import block_diagonalize


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
    svd(arr) = U,sigma,V.Transpose()  for the final svd function.

    Currently, the algorithm is working fine, but the algorithm can be further optimized, Using the fact that we will apply this algorithm
    to a band matrix, which we get after using the function ht.block_diagonalize(arr)


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


    bl -> The lower bandwidth of the matrix.
    bu -> The upper bandwidth of the matrix.
    b = bl + bu => Is the number of off diagonals present in the matrix.

    The reduction proceeds in k = min(m,n) steps each of which produces one row of the resulting bidiagonal matrix B.
    The band is partitioned into block columns Dj,Ej and transformations are applied in each sweep.

    After the kth sweep, the leading k X (k + 1) block (Bk) of A contains the first k rows of the resulting bidiagonal matrix B.
    There is a slight difference in the way band is partitioned into block columns Dj,Ej for the cases where m>=n & n>m.



    To change the serial algorithm into an Efficient parallel program, We should distribute the data and the computational work to multiple processors.

    The band form is further reduced to the final condensed form using the bulge chasing technique.
    This procedure annihilates the extra off-diagonal elements by chasing the created
    fill-in elements down to the bottom right side of the matrix using successive
    orthogonal transformations at each sweep.



    Note: We will do the bulge chasing using Halos.

    Second Stage of reduction: Band matrix to a real bidiagonal matrix. This uses a bulge chasing algorithm.


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

    b = 0
    bl = 0
    bu = 0
    row_0 = arr[0, 0:]
    col_0 = arr[0:, 0]

    for i in range(1, len(col_0)):
        if col_0[i] != 0:
            bl += 1

    for i in range(1, len(row_0)):
        if row_0[i] != 0:
            bu += 1

    b = bl + bu

    # print("b: ", b)

    U1, vt1 = ht.eye(m, dtype=ht.float64), ht.eye(n, dtype=ht.float64)
    # U1 is an identity matrix of size m x m, vt1 is an identity matrix of size n x n

    # print(arr)

    for i in range(k):

        if m >= n:
            if rank == 0 and i < k - 1:
                # print(f"at {i} arr = ", arr)
                D1 = arr[i : i + 1 + bl, i]
                v_left, tau_left = gen_house_vec(D1)
                Uj = apply_house_left(D1, v_left, tau_left, U1, D1.shape[0], i)
                arr[i : i + 1 + bl, i] = D1

                comm.send(arr, dest=1, tag=1)
                comm.send(Uj, dest=1, tag=2)

                req2 = comm.irecv(source=1, tag=3)
                arr = req2.wait()

            else:
                # print(f"at {i} arr = ", arr)
                D1 = arr[i : i + 1 + bl, i]
                v_left, tau_left = gen_house_vec(D1)
                Uj = apply_house_left(D1, v_left, tau_left, U1, D1.shape[0], i)
                arr[i : i + 1 + bl, i] = D1

            if i < k - 1:
                for j in range(2, k):
                    # print("j is = ", j)

                    if j == 2:
                        if rank == 1:
                            # print("here: ", arr)
                            req3 = comm.irecv(source=0, tag=1)
                            arr = req3.wait()
                            # print("now here,", arr)
                            Ej = arr[i : i + 1 + bl, i + 1 : i + b + 1]
                            # print(Ej, end="    ")
                            if Ej.size(0) > 0 and Ej.size(1) > 0:

                                Uj = comm.recv(source=0, tag=2)
                                Ej = torch.matmul(Uj.float(), Ej.float())
                                # print("Ej is: ", Ej, end="  ")
                                v_right, tau_right = gen_house_vec(Ej[0, :])
                                vj = apply_house_right(Ej[0, :], v_right, tau_right, vt1, n, j)
                                # arr[i,1:] = Ej[0,:]
                                # print("Ej is: ", Ej)
                                # print(i)
                                arr[i : i + 1 + bl, i + 1 : i + b + 1] = Ej[:]
                                # print("arr is: ", arr)
                                # print(Ej[0, :])
                                # print(arr[i,:])
                                Aj = arr[i : i + bl + b + 1, i + 1 : i + b + 1]
                                Aj = torch.matmul(Aj.float(), vj)
                                # arr[i : i + bl + b + 1, i + 1 : i + b + 1] = Aj

                            Dj = arr[i + 1 + bl : i + 1 + bl + b, i + 1 : i + 1 + b]
                            if Dj.size(0) > 0 and Dj.size(1) > 0:
                                v_left, tau_left = gen_house_vec(Dj[:, 0])
                                Uj = apply_house_left(Dj, v_left, tau_left, U1, m, j)
                                arr[i + 1 + bl : i + 1 + bl + b, i + 1 : i + 1 + b] = Dj
                                # comm.send(Uj, dest=2)

                            p_left, p_right = i + 1 + bl, i + 1 + bl + b
                            # req = comm.isend(arr, dest=0, tag=3)
                            # req.wait()
                            # print(f"ok {i} and {j}", arr)

                    else:
                        # if rank == j - 1:
                        # Uj = comm.recv(source=j - 1)
                        if rank == 1:
                            # print("This one:", arr)
                            # req4 = comm.irecv(source=0,tag=1)
                            # arr = req4.wait()
                            Ej = arr[p_left:p_right, i + (j - 2) * b + 1 : i + 1 + (j - 1) * b]

                            if Ej.size(0) > 0 and Ej.size(1) > 0:
                                Ej = torch.matmul(Uj.float(), Ej.float())
                                v_right, tau_right = gen_house_vec(Ej[0, :])
                                vj = apply_house_right(Ej[0, :], v_right, tau_right, vt1, n, j)
                                arr[p_left:p_right, i + (j - 2) * b + 1 : i + 1 + (j - 1) * b] = Ej

                                Aj = arr[
                                    p_left : p_right + b, i + (j - 2) * b + 1 : i + 1 + (j - 1) * b
                                ]
                                Aj = torch.matmul(Aj.float(), vj)
                                arr[
                                    p_left : p_right + b, i + (j - 2) * b + 1 : i + 1 + (j - 1) * b
                                ] = Aj
                                # print("came here:")

                            Dj = arr[
                                p_right : p_right + b, i + (j - 2) * b + 1 : i + 1 + (j - 1) * b
                            ]

                            if Dj.size(0) > 0 and Dj.size(1) > 0:
                                v_left, tau_left = gen_house_vec(Dj[:, 0])
                                Uj = apply_house_left(Dj, v_left, tau_left, U1, m, j)
                                arr[
                                    p_right : p_right + b, i + (j - 2) * b + 1 : i + 1 + (j - 1) * b
                                ] = Dj
                                # comm.send(Uj, dest=j + 1)

                            p_left, p_right = p_right, p_right + b
                            # print(f"ok {i} and {j}")

                if rank == 1:
                    req = comm.isend(arr, dest=0, tag=3)
                    req.wait()

        else:
            if rank == 0 and i < k - 1:

                E1 = arr[i, i : i + 1 + bu]
                D1 = arr[i + 1 : i + 1 + b, i : i + 1 + bu]

                if E1.size(0) > 0:
                    v_right, tau_right = gen_house_vec(E1)
                    vj1 = apply_house_right(E1, v_right, tau_right, vt1, n, i)

                    Aj = arr[i : i + 1 + b, i : i + 1 + bu]
                    torch.matmul(Aj.float(), vj1)
                    arr[i, i : i + 1 + bu] = E1

                if D1.size(0) > 0 and D1.size(1) > 0:
                    v_left, tau_left = gen_house_vec(D1[:, 0])
                    Uj = apply_house_left(D1, v_left, tau_left, U1, m, i)
                    arr[i + 1 : i + 1 + b, i : i + 1 + bu] = D1

                p_left, p_right = i + 1, i + 1 + b

                comm.send(arr, dest=1, tag=1)
                comm.send(Uj, dest=1, tag=2)

                req2 = comm.irecv(source=1, tag=3)
                arr = req2.wait()

            else:
                E1 = arr[i, i : i + 1 + bu]
                D1 = arr[i + 1 : i + 1 + b, i : i + 1 + bu]

                if E1.size(0) > 0:
                    v_right, tau_right = gen_house_vec(E1)
                    vj1 = apply_house_right(E1, v_right, tau_right, vt1, n, i)

                    Aj = arr[i : i + 1 + b, i : i + 1 + bu]
                    torch.matmul(Aj.float(), vj1)
                    arr[i, i : i + 1 + bu] = E1

                if D1.size(0) > 0 and D1.size(1) > 0:
                    v_left, tau_left = gen_house_vec(D1[:, 0])
                    Uj = apply_house_left(D1, v_left, tau_left, U1, m, i)
                    arr[i + 1 : i + 1 + b, i : i + 1 + bu] = D1

                p_left, p_right = i + 1, i + 1 + b

            if i < k - 1:
                # print("i is: ", i)
                for j in range(2, k):
                    # print("j is = ", j)

                    if j == 2 and rank == 1:
                        req3 = comm.irecv(source=0, tag=1)
                        arr = req3.wait()

                        Ej = arr[
                            p_left:p_right, i + 1 + bu + (j - 2) * b : i + 1 + bu + (j - 1) * b
                        ]
                        # print("Ej is: ", Ej)
                        if Ej.size(0) > 0 and Ej.size(1) > 0:

                            Uj = comm.recv(source=0, tag=2)
                            Ej = torch.matmul(Uj.float(), Ej.float())
                            # print("Ej is: ", Ej)

                            v_right, tau_right = gen_house_vec(Ej[0, :])
                            vj = apply_house_right(Ej[0, :], v_right, tau_right, vt1, n, j)

                            arr[
                                p_left:p_right, i + 1 + bu + (j - 2) * b : i + 1 + bu + (j - 1) * b
                            ] = Ej

                            Aj = arr[
                                p_left : p_right + b,
                                i + 1 + bu + (j - 2) * b : i + 1 + bu + (j - 1) * b,
                            ]
                            Aj = torch.matmul(Aj.float(), vj)
                            print("came here:")

                        Dj = arr[
                            p_right : p_right + b,
                            i + 1 + bu + (j - 2) * b : i + 1 + bu + (j - 1) * b,
                        ]

                        if Dj.size(0) > 0 and Dj.size(1) > 0:
                            v_left, tau_left = gen_house_vec(Dj[:, 0])
                            Uj = apply_house_left(Dj, v_left, tau_left, U1, m, j)
                            arr[
                                p_right : p_right + b,
                                i + 1 + bu + (j - 2) * b : i + 1 + bu + (j - 1) * b,
                            ] = Dj

                        p_left, p_right = p_right, p_right + b

                    elif rank == 1:

                        Ej = arr[
                            p_left:p_right, i + 1 + bu + (j - 2) * b : i + 1 + bu + (j - 1) * b
                        ]
                        # print("Ej is: ", Ej)
                        if Ej.size(0) > 0 and Ej.size(1) > 0:

                            Ej = torch.matmul(Uj.float(), Ej.float())
                            # print("Ej is: ", Ej)

                            v_right, tau_right = gen_house_vec(Ej[0, :])
                            vj = apply_house_right(Ej[0, :], v_right, tau_right, vt1, n, j)

                            arr[
                                p_left:p_right, i + 1 + bu + (j - 2) * b : i + 1 + bu + (j - 1) * b
                            ] = Ej

                            Aj = arr[
                                p_left : p_right + b,
                                i + 1 + bu + (j - 2) * b : i + 1 + bu + (j - 1) * b,
                            ]
                            Aj = torch.matmul(Aj.float(), vj)
                            # print("came here:")

                        Dj = arr[
                            p_right : p_right + b,
                            i + 1 + bu + (j - 2) * b : i + 1 + bu + (j - 1) * b,
                        ]

                        if Dj.size(0) > 0 and Dj.size(1) > 0:
                            v_left, tau_left = gen_house_vec(Dj[:, 0])
                            Uj = apply_house_left(Dj, v_left, tau_left, U1, m, j)
                            arr[
                                p_right : p_right + b,
                                i + 1 + bu + (j - 2) * b : i + 1 + bu + (j - 1) * b,
                            ] = Dj

                        p_left, p_right = p_right, p_right + b

                if rank == 1:
                    req = comm.isend(arr, dest=0, tag=3)
                    req.wait()

    return arr


# U1,B1,Vt1 = bi_diagonalize(a)

# ht.local_printing()
# array_with_halos
# print(a.get_halo(1))

# a = ht.random.rand(200, dtype=ht.float64, split=0)
# a = a.reshape(17, 15)
# print(a)


# U, a, V = block_diagonalize(a)

# final = torch.tensor
# m, n = a.shape
# a = resplit(a, None)

# a = a._DNDarray__cat_halo()


# a = bi_diagonalize(a)
# print("Tensor a final: ", a)
