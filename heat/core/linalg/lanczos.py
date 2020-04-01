import math
import torch

from ..communication import MPI
from .. import arithmetics
from .. import dndarray
from .. import factories
from .. import random
from .basics import matmul
from .basics import dot

__all__ = ["cg", "lanczos", "norm", "projection"]


def cg(A, b, x0, out=None):
    """
    Conjugate gradients method for solving a system of linear equations Ax = b

    Parameters
    ----------
    A : ht.DNDarray
        2D symmetric, positive definite Matrix
    b : ht.DNDarray
        1D vector
    x0 : ht.DNDarray
        Arbitrary 1D starting vector

    Returns
    -------
    ht.DNDarray
        Returns the solution x of the system of linear equations. If out is given, it is returned
    """

    if (
        not isinstance(A, dndarray.DNDarray)
        or not isinstance(b, dndarray.DNDarray)
        or not isinstance(x0, dndarray.DNDarray)
    ):
        raise TypeError(
            "A, b and x0 need to be of type ht.dndarra, but were {}, {}, {}".format(
                type(A), type(b), type(x0)
            )
        )

    if not (A.numdims == 2):
        raise RuntimeError("A needs to be a 2D matrix")
    if not (b.numdims == 1):
        raise RuntimeError("b needs to be a 1D vector")
    if not (x0.numdims == 1):
        raise RuntimeError("c needs to be a 1D vector")

    r = b - matmul(A, x0)
    p = r
    print(A.shape, p.shape)
    rsold = matmul(r, r)
    x = x0

    for i in range(len(b)):
        Ap = matmul(A, p)
        alpha = rsold / matmul(p, Ap)
        x = x + (alpha * p)
        r = r - (alpha * Ap)
        rsnew = matmul(r, r)
        if math.sqrt(rsnew) < 1e-10:
            print("Residual r = {} reaches tolerance in it = {}".format(math.sqrt(rsnew), i))
            if out is not None:
                out = x
                return out
            return x

        p = r + ((rsnew / rsold) * p)
        rsold = rsnew

    if out is not None:
        out = x
        return out
    return x


def lanczos(A, m, v0=None, V_out=None, T_out=None):
    """
    Lanczos algorithm for iterative approximation of the solution to the eigenvalue problem,  an adaptation of power methods to find the m "most useful" (tending towards extreme highest/lowest) eigenvalues and eigenvectors of an n×n Hermitian matrix, where often m<<n
    Parameters
    ----------
    A : ht.DNDarray
        2D symmetric, positive definite Matrix
    m : int
        number of Lanczos iterations
    v0 : ht.DNDarray (optional)
        1D starting vector of euclidian norm 1

    Returns
    -------
    V ht.DNDarray
        Matrix of size nxm, with orthonormal columns, that span the Krylow subspace. If out is given, it is returned
    T ht.DNDarray
        Tridiagonal matrix of size mxm, with coefficients alpha_1,...alpha_n on the diagonal and coefficients beta_1,...,beta_n-1 on the side-diagonals. If out is given, it is returned

    """
    if not isinstance(A, dndarray.DNDarray):
        raise TypeError("A needs to be of type ht.dndarra, but was {}".format(type(A)))

    if not (A.numdims == 2):
        raise RuntimeError("A needs to be a 2D matrix")
    if not isinstance(m, (int, float)):
        raise TypeError("m must be eiter int or float, but was {}".format(type(m)))

    n, column = A.shape
    if n != column:
        raise TypeError("Input Matrix A needs to be symmetric.")
    T = factories.zeros((m, m))
    if A.split == 0:
        # This is done for better memory access in the reorthogonalization Gram-Schmidt algorithm
        V = factories.ones((m, n), split=1, dtype=A.dtype, device=A.device)
    else:
        V = factories.ones((m, n), split=None, dtype=A.dtype, device=A.device)

    if v0 is None:
        vr = random.rand(n)
        v0 = vr / norm(vr)

    # # 0th iteration
    # # vector v0 has euklidian norm = 1
    w = matmul(A, v0)
    alpha = dot(w, v0)
    w = w - alpha * v0
    T[0, 0] = alpha
    V[0, :] = v0
    for i in range(1, int(m)):
        beta = norm(w)
        if abs(beta) < 1e-10:
            print("Lanczos breakdown in iteration {}".format(i))
            # Lanczos Breakdown, pick a random vector to continue
            vr = random.rand(n, dtype=A.dtype)
            # orthogonalize v_r with respect to all vectors v[i]
            for j in range(i):
                vi_loc = V._DNDarray__array[j, :]
                a = torch.dot(vr._DNDarray__array, vi_loc)
                b = torch.dot(vi_loc, vi_loc)
                A.comm.Allreduce(MPI.IN_PLACE, a, MPI.SUM)
                A.comm.Allreduce(MPI.IN_PLACE, b, MPI.SUM)
                vr._DNDarray__array = vr._DNDarray__array - a / b * vi_loc
            # normalize v_r to Euclidian norm 1 and set as ith vector v
            vi = vr / norm(vr)
        else:
            vr = w

            # Reorthogonalization
            # ToDo: Rethink this; mask torch calls, See issue #494
            # This is the fast solution, using item access on the ht.dndarray level is way slower
            for j in range(i):
                vi_loc = V._DNDarray__array[j, :]
                a = torch.dot(vr._DNDarray__array, vi_loc)
                b = torch.dot(vi_loc, vi_loc)
                A.comm.Allreduce(MPI.IN_PLACE, a, MPI.SUM)
                A.comm.Allreduce(MPI.IN_PLACE, b, MPI.SUM)
                vr._DNDarray__array = vr._DNDarray__array - a / b * vi_loc

            vi = vr / norm(vr)

        w = matmul(A, vi)
        alpha = dot(w, vi)
        w = w - alpha * vi - beta * V[i - 1, :]

        T[i - 1, i] = beta
        T[i, i - 1] = beta
        T[i, i] = alpha
        V[i, :] = vi

    if V.split is not None:
        V.resplit_(axis=None)
    V = V.transpose()
    if T_out is not None:
        T_out = T
        if V_out is not None:
            V_out = V
            return V_out, T_out
        return V, T_out
    elif V_out is not None:
        V_out = V
        return V_out, T

    return V, T


def norm(a):
    """
    Frobenius norm of vector a

    Parameters
    ----------
    a : ht.DNDarray

    Returns
    -------
    float
        Returns the vector norm (lenght) of a
    """
    if not isinstance(a, dndarray.DNDarray):
        raise TypeError("a must be of type ht.DNDarray, but was {}".format(type(a)))

    d = a ** 2

    for i in range(len(a.shape) - 1, -1, -1):
        d = arithmetics.sum(d, axis=i)

    return math.sqrt(d)


def projection(a, b):
    """
    Projection of vector a onto vector b

    Parameters
    ----------
    a : ht.DNDarray (1D)
    b : ht.DNDarray (1D)

    Returns
    -------
    ht.DNDarray
        Returns the vector projection of b in the direction of a
    """
    if not isinstance(a, dndarray.DNDarray) or not isinstance(b, dndarray.DNDarray):
        raise TypeError(
            "a, b must be of type ht.DNDarray, but were {}, {}".format(type(a), type(b))
        )

    if len(a.shape) != 1 or len(b.shape) != 1:
        raise RuntimeError(
            "a, b must be vectors of length 1, but were {}, {}".format(len(a.shape), len(b.shape))
        )

    return (dot(a, b) / dot(b, b)) * b
