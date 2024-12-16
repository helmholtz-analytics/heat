"""
Implements polar decomposition (PD)
"""

import numpy as np
import collections
import torch
from typing import Type, Callable, Dict, Any, TypeVar, Union, Tuple

from ..communication import MPICommunication
from ..dndarray import DNDarray
from .. import factories
from .. import types
from ..linalg import matrix_norm, vector_norm, matmul, qr, solve_triangular
from .basics import _estimate_largest_singularvalue, condest
from ..indexing import where
from ..random import randn
from ..devices import Device
from ..manipulations import vstack, hstack, concatenate, diag, balance
from ..exponential import sqrt
from .. import statistics
from mpi4py import MPI

from scipy.special import ellipj
from scipy.special import ellipkm1

__all__ = ["pd"]


def _zolopd_n_iterations(r: int, kappa: float) -> int:
    """
    Returns the number of iterations required in the Zolotarev-PD algorithm.
    See the Table 3.1 in: Nakatsukasa, Y., & Freund, R. W. (2016). Computing the polar decomposition with applications. SIAM Review, 58(3), DOI: https://doi.org/10.1137/140990334

    Inputs are `r` and `kappa` (named as in the paper), output is the number of iterations.
    """
    if kappa <= 1e2:
        its = [4, 3, 2, 2, 2, 2, 2, 2]
    elif kappa <= 1e3:
        its = [3, 3, 2, 2, 2, 2, 2, 2]
    elif kappa <= 1e5:
        its = [5, 3, 3, 3, 2, 2, 2, 2]
    elif kappa <= 1e7:
        its = [5, 4, 3, 3, 3, 2, 2, 2]
    else:
        its = [6, 4, 3, 3, 3, 3, 3, 2]
    return its[r - 1]


def _compute_zolotarev_coefficients(
    r: int, ell: float, device: str, dtype: types.datatype = types.float64
) -> Tuple[DNDarray, DNDarray, types.datatype]:
    """
    Computes c=(c_i)_i defined in equation (3.4), as well as a=(a_j)_j and Mhat defined in formulas (4.2)/(4.3) of the paper Nakatsukasa, Y., & Freund, R. W. (2016). Computing the polar decomposition with applications. SIAM Review, 58(3), DOI: https://doi.org/10.1137/140990334.
    Evaluations of the respective complete elliptic integral of the first kind and the Jacobi elliptic functions are imported from SciPy.

    Inputs are `r` and `ell` (named as in the paper), as well as the Heat data type `dtype` of the output (required for reasons of consistency).
    Output is a tupe containing the vectors `a` and `c` as DNDarrays and `Mhat`.
    """
    uu = np.arange(1, 2 * r + 1) * ellipkm1(ell**2) / (2 * r + 1)
    ellipfcts = np.asarray(ellipj(uu, 1 - ell**2)[:2])
    cc = ell**2 * ellipfcts[0, :] ** 2 / ellipfcts[1, :] ** 2
    aa = np.zeros(r)
    Mhat = 1
    for j in range(1, r + 1):
        p1 = 1
        p2 = 1
        for k in range(1, r + 1):
            p1 *= cc[2 * j - 2] - cc[2 * k - 1]
            if k != j:
                p2 *= cc[2 * j - 2] - cc[2 * k - 2]
        aa[j - 1] = -p1 / p2
        Mhat *= (1 + cc[2 * j - 2]) / (1 + cc[2 * j - 1])
    return (
        factories.array(cc, dtype=dtype, split=None, device=device),
        factories.array(aa, dtype=dtype, split=None, device=device),
        factories.array(Mhat, dtype=dtype, split=None, device=device),
    )


def pd(
    A: DNDarray,
    r: int = 8,
    calcH: bool = True,
    condition_estimate: float = 0.0,
    silent: bool = True,
) -> Tuple[DNDarray, DNDarray]:
    """
    Computes the so-called polar decomposition of the input 2D DNDarray ``A``, i.e., it returns the orthogonal matrix ``U`` and the symmetric, positive definite
    matrix ``H`` such that ``A = U @ H``.

    Input
    -----
    A : ht.DNDarray,
        The input matrix for which the polar decomposition is computed;
        must be two-dimensional, of data type float32 or float64, and must have at least as many rows as columns.
    r : int, optional, default: 8
        The parameter r used in the Zolotarev-PD algorithm; must be an integer between 1 and 8.
        Higher values of r lead to faster convergence, but memory consumption is proportional to r.
    calcH : bool, optional, default: True
        If True, the function returns the symmetric, positive definite matrix H. If False, only the orthogonal matrix U is returned.
    condition_estimate : float, optional, default: 0.
        This argument allows to provide an estimate for the condition number of the input matrix ``A``, if such estimate is already known.
        If a positive number greater than 1., this value is used as an estimate for the condition number of A.
        If smaller or equal than 1., the condition number is estimated internally (default).
    silent : bool, optional, default: True
        If True, the function does not print any output. If False, some information is printed during the computation.

    Notes
    -----
    The implementation follows Algorithm 5.1 in Reference [1]; however, instead of switching from QR to Cholesky decomposition depending on the condition number,
    we stick to QR decomposition in all iterations.

    References
    ----------
    [1] Nakatsukasa, Y., & Freund, R. W. (2016). Computing the polar decomposition with applications. SIAM Review, 58(3), DOI: https://doi.org/10.1137/140990334.
    """
    # check whether input is DNDarray of correct shape
    if not isinstance(A, DNDarray):
        raise TypeError(f"Input ``A`` needs to be a DNDarray but is {type(A)}.")
    if not A.ndim == 2:
        raise ValueError(f"Input ``A`` needs to be a 2D DNDarray, but its dimension is {A.ndim}.")
    if A.shape[0] < A.shape[1]:
        raise ValueError(
            f"Input ``A`` must have at least as many rows as columns, but has shape {A.shape}."
        )
    # check if A is a real floating point matrix and choose tolerances tol accordingly
    if A.dtype == types.float32:
        tol = 1.19e-7
    elif A.dtype == types.float64:
        tol = 2.22e-16
    else:
        raise TypeError(
            f"Input ``A`` must be of data type float32 or float64 but has data type {A.dtype}"
        )

    # check if input for r is reasonable
    if not isinstance(r, int) or r < 1 or r > 8:
        raise ValueError(
            f"If specified, input ``r`` must be an integer between 1 and 8, but is {r} of data type {type(r)}."
        )

    # check if input for condition_estimate is reasonable
    if not isinstance(condition_estimate, float):
        raise TypeError(
            f"If specified, input ``condition_estimate`` must be a float but is {type(condition_estimate)}."
        )

    alpha = _estimate_largest_singularvalue(A).item()

    if condition_estimate <= 1.0:
        kappa = condest(A).item()
    else:
        kappa = condition_estimate

    if A.comm.rank == 0 and not silent:
        print(
            f"Condition number estimate: {kappa:2.2e} / Estimate for largest singular value: {alpha:2.2e}."
        )

    # initialize X for the iteration: input ``A``, normalized by largest singular value
    X = A / alpha

    # iteration counter and maximum number of iterations
    it = 0
    itmax = _zolopd_n_iterations(r, kappa)

    # parameters and coefficients, see Ref. [1] for their meaning
    ell = 1.0 / kappa
    c, a, Mhat = _compute_zolotarev_coefficients(r, ell, A.device, dtype=A.dtype)

    while it < itmax:
        it += 1
        if not silent:
            if A.comm.rank == 0:
                print(f"Starting Zolotarev-PD iteration no. {it}...")
        # remember current X for later convergence check
        X_old = X.copy()

        # repeat X r-times and create (repeated) identity matrix
        # this allows to compute the r-many QR decomposition and matrix multiplications in batch-parallel manor
        X = factories.array(
            X.larray.repeat(r, 1, 1),
            is_split=X.split + 1 if X.split is not None else None,
            comm=A.comm,
        )
        cId = factories.eye(A.shape[1], dtype=A.dtype, comm=A.comm, split=A.split, device=A.device)
        cId = factories.array(
            cId.larray.repeat(r, 1, 1),
            is_split=cId.split + 1 if cId.split is not None else None,
            comm=A.comm,
        )
        cId *= c[0::2].reshape(-1, 1, 1) ** 0.5
        X = concatenate([X, cId], axis=1)
        Q, _ = qr(X)
        Q1 = Q[:, : A.shape[0], : A.shape[1]].balance()
        Q2 = Q[:, A.shape[0] :, : A.shape[1]].transpose([0, 2, 1]).balance()
        del Q
        X = Mhat * (
            X[:, : A.shape[0], :].balance() / r
            + a.reshape(-1, 1, 1)
            / c[0::2].reshape(-1, 1, 1) ** 0.5
            * matmul(Q1, Q2).resplit_(X.split)
        )
        del (Q1, Q2)
        # finally, sum over the batch-dimension to get back the result of the iteration
        X = X.sum(axis=0)

        # check for convergence and break if tolerance is reached
        if it > 1 and matrix_norm(X - X_old, ord="fro") / matrix_norm(X, ord="fro") <= tol ** (
            1 / (2 * r + 1)
        ):
            if not silent:
                if A.comm.rank == 0:
                    print(f"Zolotarev-PD iteration converged after {it} iterations.")
            break
        elif it < itmax:
            # if another iteration is necessary, update coefficients and parameters for next iteration
            ellold = ell
            ell = 1
            for j in range(r):
                ell *= (ellold**2 + c[2 * j + 1]) / (ellold**2 + c[2 * j])
            ell *= Mhat * ellold
            c, a, Mhat = _compute_zolotarev_coefficients(r, ell, A.device, dtype=A.dtype)
        else:
            if not silent:
                if A.comm.rank == 0:
                    print(
                        f"Zolotarev-PD iteration did not reach the convergence criterion after {itmax} iterations, which is most likely due to limited numerical accuracy and/or poor estimation of the condition number. The result may still be useful, but should be handeled with care!"
                    )
    # postprocessing: compute H if requested
    if calcH:
        H = matmul(X.T, A)
        H = 0.5 * (H + H.T.resplit(H.split))
        return X, H.resplit(A.split)
    else:
        return X
