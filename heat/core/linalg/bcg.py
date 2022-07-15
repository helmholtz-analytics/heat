import itertools
from operator import imod
from turtle import left
import numpy as np
import torch
import warnings
import heat as ht

from mpi4py import MPI

from heat.core.linalg.utils import apply_house, apply_house_left, apply_house_right, gen_house_mat, gen_house_vec
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

from typing import List, Callable, Union, Optional, Tuple

from torch._C import Value
from inspect import stack
from mpi4py import MPI
from pathlib import Path
from typing import List, Union, Tuple, TypeVar

from heat.core.devices import cpu

from ..communication import MPI
from .. import arithmetics
from .. import complex_math
from .. import constants
from .. import exponential
from heat.core import dndarray 
from .. import factories
from .. import manipulations
from heat.core.manipulations import *
from .. import rounding
from .. import sanitation
from .. import statistics
from .. import stride_tricks
from .. import types
from .svd import *
from .qr import *

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
        k = steps down the matrix, each of which produces one row of the
        resulting bi-diagonal matrix B. 



        bidiaonalize(A) -> returns a matrix (B), which is a bidiagonal matrix

        Parameters
        ----------
        arr : ht.DNDarray
            2D input matrix (m x n)
        overwrite_arr : bool, Optional
            Default: True
            if True, will overwrite the input matrix A into a bi-diagonal matrix.
            if False, will return a bi-diagonal matrix, 

         

        Returns
        -------
        result : tuple
            U1 : ht.DNDarray

            B : ht.DNDarray

            V1 : ht.DNDarray 



        """

        if(overwrite_arr== True):
            arr = A
            # The input matrix is overwritten with the result, i.e it will be changed to a bidiagonal matrix.
        else:
            arr = A.copy()
            # The input matrix is not overwritten, i.e a new matrix "arr" which is a copy of input matrix, will be changed to a bidiagonal matrix.

        m,n = arr.shape

        if(m>=n):
            U1,vt1 = ht.eye(m,dtype=ht.float64), ht.eye(n,dtype=ht.float64)
            # U1 is an identity matrix of size m x m, vt1 is an identity matrix of size n x n
            for i in range(n):
                v_left,tau_left = gen_house_vec(arr[i:,i])
                # All the elements in the ith column below arr[i][i] including itself, are send to the "gen_house_vec" function.
                apply_house_left(arr[i:,i:], v_left, tau_left, U1, m, i)


                if i<=n-2:
                    v_right,tau_right = gen_house_vec(arr[i,i+1:].T)
                    # All the elements in the ith row to the right of arr[i][i] including itself, are send to the "gen_house_vec" function.
                    apply_house_right(arr[i:,i+1:], v_right, tau_right, vt1, n, i)                    
            
            return arr
        
        else:
            U1,vt1 = ht.eye(m,dtype=ht.float64), ht.eye(n,dtype=ht.float64)
            # U1 is an identity matrix of size m x m, vt1 is an identity matrix of size n x n
            for i in range(m):
            
                v_left,tau_left = gen_house_vec(arr[i:,i])
                apply_house_left(arr[i:,i:], v_left, tau_left, U1, m, i)

                if i<=n-2:
                    v_right,tau_right = gen_house_vec(arr[i,i+1:].T)
                    apply_house_right(arr[i:,i+1:], v_right, tau_right, vt1, n, i)

            return arr

#arr = ht.zeros([15,12], dtype=ht.float64)
#a = ht.arange(20,dtype=ht.float64)
#a = a.reshape(4,5)

#U1,B1,Vt1 = bi_diagonalize(a)
#bi_diagonalize(a)
#print("Matrix U1 is: ", U1)
#print("Matrix B1 is: ", B1)
#print("Matrix Vt1 is: ", Vt1)
#k = (U1 @ B1 @ Vt1)
#print(k)
#print(a)
#print(B1)
