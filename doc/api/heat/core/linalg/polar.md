Module heat.core.linalg.polar
=============================
Implements polar decomposition (PD)

Functions
---------

`polar(A: heat.core.dndarray.DNDarray, r: int = None, calcH: bool = True, condition_estimate: float = 1e+16, silent: bool = True, r_max: int = 8) ‑> Tuple[heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray]`
:   Computes the so-called polar decomposition of the input 2D DNDarray ``A``, i.e., it returns the orthogonal matrix ``U`` and the symmetric, positive definite
    matrix ``H`` such that ``A = U @ H``.

    Input
    -----
    A : ht.DNDarray,
        The input matrix for which the polar decomposition is computed;
        must be two-dimensional, of data type float32 or float64, and must have at least as many rows as columns.
    r : int, optional, default: None
        The parameter r used in the Zolotarev-PD algorithm; if provided, must be an integer between 1 and 8 that divides the number of MPI processes.
        Higher values of r lead to faster convergence, but memory consumption is proportional to r.
        If not provided, the largest 1 <= r <= r_max that divides the number of MPI processes is chosen.
    calcH : bool, optional, default: True
        If True, the function returns the symmetric, positive definite matrix H. If False, only the orthogonal matrix U is returned.
    condition_estimate : float, optional, default: 1.e16.
        This argument allows to provide an estimate for the condition number of the input matrix ``A``, if such estimate is already known.
        If a positive number greater than 1., this value is used as an estimate for the condition number of A.
        If smaller or equal than 1., the condition number is estimated internally.
        The default value of 1.e16 is the worst case scenario considered in [1].
    silent : bool, optional, default: True
        If True, the function does not print any output. If False, some information is printed during the computation.
    r_max : int, optional, default: 8
        See the description of r for the meaning; r_max is only taken into account if r is not provided.


    Notes
    -----
    The implementation follows Algorithm 5.1 in Reference [1]; however, instead of switching from QR to Cholesky decomposition depending on the condition number,
    we stick to QR decomposition in all iterations.

    References
    ----------
    [1] Nakatsukasa, Y., & Freund, R. W. (2016). Computing Fundamental Matrix Decompositions Accurately via the Matrix Sign Function in Two Iterations: The Power of Zolotarev's Functions. SIAM Review, 58(3), DOI: https://doi.org/10.1137/140990334.
