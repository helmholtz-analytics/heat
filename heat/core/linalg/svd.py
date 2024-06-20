"""
file for future "full" SVD implementation
"""

from typing import Tuple
from ..dndarray import DNDarray
from .qr import qr
from ..types import float32, float64
import torch

__all__ = ["svd"]


def svd(
    A: DNDarray,
    full_matrices: bool = False,
    compute_uv: bool = True,
    qr_procs_to_merge: int = 2,
) -> Tuple[DNDarray, DNDarray, DNDarray]:
    """
    Computes the singular value decomposition of a matrix (the input array ``A``).
    For an input DNDarray ``A`` of shape ``(M, N)``, the function returns DNDarrays ``U``, ``S``, and ``V`` such that ``A = U @ ht.diag(S) @ V.T``
    with shapes ``(M, min(M,N))``, ``(min(M, N),)``, and ``(min(M,N),N)``, respectively, in the case that ``compute_uv=True``, or
    only the vector containing the singular values ``S`` of shape ``(min(M, N),)`` in the case that ``compute_uv=False``. By definition of the singular value decomposition,
    the matrix ``U`` is orthogonal, the matrix ``V`` is orthogonal, and the entries of the vector ``S``are non-negative real numbers.

    We refer to, e.g., wikipedia (https://en.wikipedia.org/wiki/Singular_value_decomposition) or to Gene H. Golub and Charles F. Van Loan, Matrix Computations (3rd Ed., 1996),
    for more detailed information on the singular value decomposition.

    Parameters
    ----------
    A : ht.DNDarray
        The input array (2D, float32 or float64) for which the singular value decomposition is computed.
        Must be tall skinny (``M >> N``) or short fat (``M << n``) for the current implementation; an implementation that covers the remaining cases is planned.
    full_matrices : bool, optional
        currently, only the default value ``False`` is supported. This argument is included for compatibility with NumPy.
    compute_uv : bool, optional
        if ``True``, the matrices ``U`` and ``V`` are computed and returned together with the singular values ``S``.
        If ``False``, only the vector ``S`` containing the singular values is returned.
    qr_procs_to_merge : int, optional
        the number of processes to merge in the tall skinny QR decomposition that is applied if the input array is tall skinny (``M > N``) or short fat (``M < N``).
        See the corresponding remarks for ``heat.linalg.qr`` for more details.


    Remarks
    ----------
    Unlike in NumPy, we currently do not support the option ``full_matrices=True``, since this can result in heavy memory consumption (in particular for tall skinny
    and short fat matrices) that should be avoided in the context Heat is designed for. If you nevertheless require this feature, please open an issue on GitHub.

    The algorithm used for the computation of the singular value depens on the shape of the input array ``A``.
    For tall and skinny matrices (``M > N``), the algorithm is based on the tall-skinny QR decomposition; currently this is the only supported algorithm.
    """
    if full_matrices:
        raise NotImplementedError(
            "SVD with 'full_matrices=True' is not supported by Heat yet as it might result in heavy memory usage. \n Please open an issue on GitHub if you nevertheless require this feature."
        )

    if not isinstance(A, DNDarray):
        raise TypeError(f"'A' must be a DNDarray, but is {type(A)}")
    if not isinstance(qr_procs_to_merge, int):
        raise TypeError(
            f"procs_to_merge must be an int, but is currently {type(qr_procs_to_merge)}"
        )
    if qr_procs_to_merge < 0 or qr_procs_to_merge == 1:
        raise ValueError(
            f"procs_to_merge must be 0 (for merging all processes at once) or at least 2, but is currently {qr_procs_to_merge}"
        )
    if qr_procs_to_merge == 0:
        qr_procs_to_merge = A.comm.size
    if A.ndim != 2:
        raise ValueError(
            f"Array ``A`` must be 2 dimensional, buts has {A.ndim} dimensions. \n Please open an issue on GitHub if you require SVD for batches of matrices similar to PyTorch."
        )
    if A.dtype not in [float32, float64]:
        raise TypeError(
            f"Array ``A`` must have a datatype of float32 or float64, but has {A.dtype}"
        )

    if A.is_distributed() and A.split == 0:
        if A.lshape_map[:, 0].max().item() < A.shape[1]:
            raise ValueError(
                "Input ``A`` is split along the rows and the local chunks of data are rectangular with more columns than rows. \n This case is not supported by the current implementation of SVD in Heat."
            )
        else:
            # this is the distributed, tall skinny case
            # compute SVD via tall skinny QR
            if compute_uv:
                # compute full SVD: first full QR, then SVD of R
                Q, R = qr(A, mode="reduced", procs_to_merge=qr_procs_to_merge)
                Utilde_loc, S_loc, Vt_loc = torch.linalg.svd(R.larray, full_matrices=False)
                Utilde = DNDarray(
                    Utilde_loc,
                    tuple(Utilde_loc.shape),
                    dtype=A.dtype,
                    split=None,
                    device=A.device,
                    comm=A.comm,
                    balanced=A.balanced,
                )
                S = DNDarray(
                    S_loc,
                    tuple(S_loc.shape),
                    dtype=A.dtype,
                    split=None,
                    device=A.device,
                    comm=A.comm,
                    balanced=A.balanced,
                )
                V = DNDarray(
                    Vt_loc.T,
                    tuple(Vt_loc.T.shape),
                    dtype=A.dtype,
                    split=None,
                    device=A.device,
                    comm=A.comm,
                    balanced=A.balanced,
                )
                U = (Utilde.T @ Q.T).T
                return U, S, V
            else:
                # compute only singular values: first only R of QR, then singular values only of R
                _, R = qr(A, mode="r", procs_to_merge=qr_procs_to_merge)
                S_loc = torch.linalg.svdvals(R.larray)
                S = DNDarray(
                    S_loc,
                    tuple(S_loc.shape),
                    dtype=A.dtype,
                    split=None,
                    device=A.device,
                    comm=A.comm,
                    balanced=A.balanced,
                )
                return S
    if A.is_distributed() and A.split == 1:

        if A.lshape_map[:, 1].max().item() < A.shape[0]:
            raise ValueError(
                "Input ``A`` is split along the columns and the local chunks of data are rectangular with more rows than columns. \n This case is not supported by the current implementation of SVD in Heat."
            )
        else:
            # this is the distributed, short fat case
            # apply the tall skinny SVD to the transpose of A
            if compute_uv:
                V, S, U = svd(
                    A.T,
                    full_matrices=full_matrices,
                    compute_uv=True,
                    qr_procs_to_merge=qr_procs_to_merge,
                )
                return U, S, V
            else:
                S = svd(
                    A.T,
                    full_matrices=full_matrices,
                    compute_uv=False,
                    qr_procs_to_merge=qr_procs_to_merge,
                )
                return S

    else:
        # this is the non-distributed case
        if compute_uv:
            U_loc, S_loc, Vt_loc = torch.linalg.svd(A.larray, full_matrices=full_matrices)
            U = DNDarray(
                U_loc,
                tuple(U_loc.shape),
                dtype=A.dtype,
                split=None,
                device=A.device,
                comm=A.comm,
                balanced=A.balanced,
            )
            S = DNDarray(
                S_loc,
                tuple(S_loc.shape),
                dtype=A.dtype,
                split=None,
                device=A.device,
                comm=A.comm,
                balanced=A.balanced,
            )
            V = DNDarray(
                Vt_loc.T,
                tuple(Vt_loc.T.shape),
                dtype=A.dtype,
                split=None,
                device=A.device,
                comm=A.comm,
                balanced=A.balanced,
            )
            return U, S, V
        else:
            S_loc = torch.linalg.svdvals(A.larray)
            S = DNDarray(
                S_loc,
                tuple(S_loc.shape),
                dtype=A.dtype,
                split=None,
                device=A.device,
                comm=A.comm,
                balanced=A.balanced,
            )
            return S
