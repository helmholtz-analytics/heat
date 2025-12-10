"""
Randomized linear algebra algorithms, including randomized SVD (rSVD) and randomized EIGH (reigh).
"""

import torch
from typing import Union, Tuple

from ..dndarray import DNDarray
from ..linalg import matmul, qr, svd
from ..random import randn
from ..sanitation import sanitize_in_nd_realfloating

__all__ = ["rsvd", "reigh"]


##############################################################################################
# Randomized SVD "rSVD"
##############################################################################################


def _randomized_range_finder(
    A: DNDarray,
    rank: int,
    n_oversamples: int = 10,
    power_iter: int = 0,
    qr_procs_to_merge: int = 2,
) -> DNDarray:
    """
    Randomized range finder with q power iterations, stabilized with QR decompositions.
    Computes an orthonormal matrix Q with r columns whose range approximates the range of A.
    n_oversamples is the number of additional samples used to improve the quality of the approximation.
    For qr_procs_to_merge see the corresponding remarks for :func:`heat.linalg.qr() <heat.core.linalg.qr.qr()>`.
    """
    if not isinstance(rank, int):
        raise TypeError(f"rank must be an integer, but is {type(rank)}.")
    if rank < 1:
        raise ValueError(f"rank must be positive, but is {rank}.")
    if not isinstance(n_oversamples, int):
        raise TypeError(
            f"if provided, n_oversamples must be an integer, but is {type(n_oversamples)}."
        )
    if n_oversamples < 0:
        raise ValueError(f"n_oversamples must be non-negative, but is {n_oversamples}.")
    if not isinstance(power_iter, int):
        raise TypeError(f"if provided, power_iter must be an integer, but is {type(power_iter)}.")
    if power_iter < 0:
        raise ValueError(f"power_iter must be non-negative, but is {power_iter}.")

    sanitize_in_nd_realfloating(A, "A", [2])

    ell = rank + n_oversamples
    q = power_iter

    # random matrix
    splitOmega = 1 if A.split == 0 else 0
    Omega = randn(A.shape[1], ell, dtype=A.dtype, device=A.device, split=splitOmega)

    # compute the range of A
    Y = matmul(A, Omega)
    Q, _ = qr(Y, procs_to_merge=qr_procs_to_merge)

    # power iterations
    for _ in range(q):
        if Q.split is not None and Q.shape[Q.split] < Q.comm.size:
            Q.resplit_(None)
        Y = matmul(A.T, Q)
        Q, _ = qr(Y, procs_to_merge=qr_procs_to_merge)
        if Q.split is not None and Q.shape[Q.split] < Q.comm.size:
            Q.resplit_(None)
        Y = matmul(A, Q)
        Q, _ = qr(Y, procs_to_merge=qr_procs_to_merge)

    return Q


def rsvd(
    A: DNDarray,
    rank: int,
    n_oversamples: int = 10,
    power_iter: int = 0,
    qr_procs_to_merge: int = 2,
) -> Union[Tuple[DNDarray, DNDarray, DNDarray], Tuple[DNDarray, DNDarray]]:
    r"""
    Randomized SVD (rSVD) with prescribed truncation rank `rank`.
    If :math:`A = U \operatorname{diag}(S) V^T` is the true SVD of A, this routine computes an approximation for U[:,:rank] (and S[:rank], V[:,:rank]).

    The accuracy of this approximation depends on the structure of A ("low-rank" is best) and appropriate choice of parameters.

    Parameters
    ----------
    A : DNDarray
        2D-array (float32/64) of which the rSVD has to be computed.
    rank : int
        truncation rank. (This parameter corresponds to `n_components` in scikit-learn's TruncatedSVD.)
    n_oversamples : int, optional
        number of oversamples. The default is 10.
    power_iter : int, optional
        number of power iterations. The default is 0.
        Choosing `power_iter > 0` can improve the accuracy of the SVD approximation in the case of slowly decaying singular values, but increases the computational cost.
    qr_procs_to_merge : int, optional
        number of processes to merge at each step of QR decomposition in the power iteration (if power_iter > 0). The default is 2. See the corresponding remarks for :func:`heat.linalg.qr() <heat.core.linalg.qr.qr()>` for more details.


    Notes
    -----
    Memory requirements: the SVD computation of a matrix of size (rank + n_oversamples) x (rank + n_oversamples) must fit into the memory of a single process.
    The implementation follows Algorithm 4.4 (randomized range finder) and Algorithm 5.1 (direct SVD) in [1].

    References
    ----------
    [1] Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM review, 53(2), 217-288.
    """
    Q = _randomized_range_finder(
        A,
        rank,
        n_oversamples=n_oversamples,
        power_iter=power_iter,
        qr_procs_to_merge=qr_procs_to_merge,
    )

    # compute the SVD of the projected matrix
    if Q.split is not None and Q.shape[Q.split] < Q.comm.size:
        Q.resplit_(None)
    B = matmul(Q.T, A)
    B.resplit_(
        None
    )  # B will be of size ell x n and thus small enough to fit into memory of a single process
    U, sigma, V = svd(B)  # actually just torch svd as input is not split anymore
    U = matmul(Q, U)[:, :rank]
    U.balance_()
    S = sigma[:rank]
    V = V[:, :rank]
    V.balance_()
    return U, S, V


def reigh(
    A: DNDarray,
    rank: int,
    n_oversamples: int = 10,
    power_iter: int = 0,
    qr_procs_to_merge: int = 2,
) -> Tuple[DNDarray, DNDarray]:
    r"""
    Randomized eigenvalue decomposition (rEIGH) with prescribed truncation rank `rank`.

    Parameters
    ----------
    A : DNDarray
        2D-array (float32/64) of which the rEIGH has to be computed. Must be symmetric.
    rank : int
        truncation rank. (This parameter corresponds to `n_components` in scikit-learn's PCA.)
    n_oversamples : int, optional
        number of oversamples. The default is 10.
    power_iter : int, optional
        number of power iterations. The default is 0.
        Choosing `power_iter > 0` can improve the accuracy of the eigenvalue approximation in the case of slowly decaying eigenvalues, but increases the computational cost.
    qr_procs_to_merge : int, optional
        number of processes to merge at each step of QR decomposition in the power iteration (if power_iter > 0). The default is 2. See the corresponding remarks for :func:`heat.linalg.qr() <heat.core.linalg.qr.qr()>` for more details.


    Notes
    -----
    Memory requirements: the symmetric eigenvalue decomposition of a matrix of size (rank + n_oversamples) x (rank + n_oversamples) must fit into the memory of a single process.
    The implementation follows Algorithm 4.4 (randomized range finder) and Algorithm 5.3 (eigenvalue decomposition from an SVD) in [1].

    References
    ----------
    [1] Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions.
    SIAM review, 53(2), 217-288.
    """
    Q = _randomized_range_finder(
        A,
        rank,
        n_oversamples=n_oversamples,
        power_iter=power_iter,
        qr_procs_to_merge=qr_procs_to_merge,
    )
    # compute the eigenvalue decomposition of the projected matrix
    if Q.split is not None and Q.shape[Q.split] < Q.comm.size:
        Q.resplit_(None)
    B = matmul(Q.T, matmul(A, Q))
    B.resplit_(
        None
    )  # B will be of size ell x ell and thus small enough to fit into memory of a single process
    eigvals, eigvecs = torch.linalg.eigh(
        B.larray
    )  # actually just torch eigh as input is not split anymore
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx][:rank]
    eigvecs = eigvecs[:, idx][:, :rank]
    eigvecs = DNDarray(
        eigvecs,
        tuple(eigvecs.shape),
        dtype=A.dtype,
        split=None,
        device=A.device,
        comm=A.comm,
        balanced=A.balanced,
    )
    eigvals = DNDarray(
        eigvals,
        tuple(eigvals.shape),
        dtype=A.dtype,
        split=None,
        device=A.device,
        comm=A.comm,
        balanced=A.balanced,
    )
    eigvecs = matmul(Q, eigvecs)
    return eigvals, eigvecs
