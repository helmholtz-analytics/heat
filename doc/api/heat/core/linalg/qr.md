Module heat.core.linalg.qr
==========================
QR decomposition of ``DNDarray``s.

Functions
---------

`qr(A: heat.core.dndarray.DNDarray, mode: str = 'reduced', procs_to_merge: int = 2) ‑> Tuple[heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray]`
:   Calculates the QR decomposition of a 2D ``DNDarray``.
    Factor the matrix ``A`` as *QR*, where ``Q`` is orthonormal and ``R`` is upper-triangular.
    If ``mode = "reduced``, function returns ``QR(Q=Q, R=R)``, if ``mode = "r"`` function returns ``QR(Q=None, R=R)``

    This function also works for batches of matrices; in this case, the last two dimensions of the input array are considered as the matrix dimensions.
    The output arrays have the same leading batch dimensions as the input array.

    Parameters
    ----------
    A : DNDarray of shape (M, N), of shape (...,M,N) in the batched case
        Array which will be decomposed. So far only arrays with datatype float32 or float64 are supported
    mode : str, optional
        default "reduced" returns Q and R with dimensions (M, min(M,N)) and (min(M,N), N). Potential batch dimensions are not modified.
        "r" returns only R, with dimensions (min(M,N), N).
    procs_to_merge : int, optional
        This parameter is only relevant for split=0 (-2, in the batched case) and determines the number of processes to be merged at one step during the so-called TS-QR algorithm.
        The default is 2. Higher choices might be faster, but will probably result in higher memory consumption. 0 corresponds to merging all processes at once.
        We only recommend to modify this parameter if you are familiar with the TS-QR algorithm (see the references below).

    Notes
    -----
    The distribution schemes of ``Q`` and ``R`` depend on that of the input ``A``.

        - If ``A`` is distributed along the columns (A.split = 1), so will be ``Q`` and ``R``.

        - If ``A`` is distributed along the rows (A.split = 0), ``Q`` too will have  `split=0`. ``R`` won't be distributed, i.e. `R. split = None`, if ``A`` is tall-skinny, i.e., if
          the largest local chunk of data of ``A`` has at least as many rows as columns. Otherwise, ``R`` will be distributed along the rows as well, i.e., `R.split = 0`.

    Note that the argument `calc_q` allowed in earlier Heat versions is no longer supported; `calc_q = False` is equivalent to `mode = "r"`.
    Unlike ``numpy.linalg.qr()``, `ht.linalg.qr` only supports ``mode="reduced"`` or ``mode="r"`` for the moment, since "complete" may result in heavy memory usage.

    Heats QR function is built on top of PyTorchs QR function, ``torch.linalg.qr()``, using LAPACK (CPU) and MAGMA (CUDA) on
    the backend. Both cases split=0 and split=1 build on a column-block-wise version of stabilized Gram-Schmidt orthogonalization.
    For split=1 (-1, in the batched case), this is directly applied to the local arrays of the input array.
    For split=0, a tall-skinny QR (TS-QR) is implemented for the case of tall-skinny matrices (i.e., the largest local chunk of data has at least as many rows as columns),
    and extended to non tall-skinny matrices by applying a block-wise version of stabilized Gram-Schmidt orthogonalization.

    References
    ----------
    Basic information about QR factorization/decomposition can be found at, e.g.:

        - https://en.wikipedia.org/wiki/QR_factorization,

        - Gene H. Golub and Charles F. Van Loan. 1996. Matrix Computations (3rd Ed.).

    For an extensive overview on TS-QR and its variants we refer to, e.g.,

        - Demmel, James, et al. “Communication-Optimal Parallel and Sequential QR and LU Factorizations.” SIAM Journal on Scientific Computing, vol. 34, no. 1, 2 Feb. 2012, pp. A206–A239., doi:10.1137/080731992.
