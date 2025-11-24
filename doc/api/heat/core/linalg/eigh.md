Module heat.core.linalg.eigh
============================
Implements Symmetric Eigenvalue Decomposition

Functions
---------

`eigh(A: heat.core.dndarray.DNDarray, r_max_zolopd: int = 8, silent: bool = True) ‑> Tuple[heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray]`
:   Computes the symmetric eigenvalue decomposition of a symmetric n x n - matrix A, provided as a DNDarray.

    The function returns DNDarrays Lambda (shape (n,) with split = 0) and V (shape (n,n)) such that
    A = V @ diag(Lambda) @ V^T, where Lambda contains the eigenvalues of A and V is an orthonormal matrix
    containing the corresponding eigenvectors as columns.

    Parameters
    ----------
    A : DNDarray
        The input matrix. Must be symmetric.
    r_max_zolopd : int, optional
        This is a hyperparameter for the computation of the polar decomposition via :func:`heat.linalg.polar` which is
        applied multiple times in this function. See the documentation of :func:`heat.linalg.polar` for more details on its
        meaning and the respective default value.
    silent : bool, optional
        If True (default), suppresses output messages; otherwise, some information on the recursion is printed to the console.

    Notes
    -----
    Unlike the :func:`torch.linalg.eigh` function, the eigenvalues are returned in descending order.
    Note that no check of symmetry is performed on the input matrix A; thus, applying this function to a non-symmetric matrix may
    result in unpredictable behaviour without a specific error message pointing to this issue.

    The algorithm used for the computation of the symmetric eigenvalue decomposition is based on the Zolotarev polar decomposition;
    see Algorithm 5.2 in:

        Nakatsukasa, Y., & Freund, R. W. (2016). Computing fundamental matrix decompositions accurately via the
        matrix sign function in two iterations: The power of Zolotarev's functions. SIAM Review, 58(3).

    See Also
    --------
    :func:`heat.linalg.polar`
