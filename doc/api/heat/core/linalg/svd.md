Module heat.core.linalg.svd
===========================
file for future "full" SVD implementation

Functions
---------

`svd(A: heat.core.dndarray.DNDarray, full_matrices: bool = False, compute_uv: bool = True, qr_procs_to_merge: int = 2, r_max_zolopd: int = 8) ‑> Tuple[heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray]`
:   Computes the singular value decomposition of a matrix (the input array ``A``).
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
        See the corresponding remarks for :func:``heat.linalg.qr`` for more details.
    r_max_zolopd : int, optional
        an internal parameter only relevant for the case that the input matrix is neither tall-skinny nor short-fat.
        This parameter is passed to the Zolotarev-Polar Decomposition and the symmetric eigenvalue decomposition that is applied in this case.
        See the documentation of :func:``heat.linalg.polar`` as well as of :func:``heat.linalg.eigh`` for more details.

    Notes
    -----
    Unlike in NumPy, we currently do not support the option ``full_matrices=True``, since this can result in heavy memory consumption (in particular for tall skinny
    and short fat matrices) that should be avoided in the context Heat is designed for. If you nevertheless require this feature, please open an issue on GitHub.

    The algorithm used for the computation of the singular value depens on the shape of the input array ``A``.
    For tall and skinny matrices (``M > N``), the algorithm is based on the tall-skinny QR decomposition. For the remaining cases we use the approach based on
    Zolotarev-Polar Decomposition and a symmetric eigenvalue decomposition based on Zolotarev-Polar Decomposition; see Algorithm 5.3 in:

        Nakatsukasa, Y., & Freund, R. W. (2016). Computing fundamental matrix decompositions accurately via the
        matrix sign function in two iterations: The power of Zolotarev's functions. SIAM Review, 58(3).

    See Also
    --------
    :func:`heat.linalg.qr`
    :func:`heat.linalg.polar`
    :func:`heat.linalg.eigh`
