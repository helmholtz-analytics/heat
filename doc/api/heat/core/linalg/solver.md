Module heat.core.linalg.solver
==============================
Collection of solvers for systems of linear equations.

Functions
---------

`cg(A: heat.core.dndarray.DNDarray, b: heat.core.dndarray.DNDarray, x0: heat.core.dndarray.DNDarray, out: heat.core.dndarray.DNDarray | None = None) ‑> heat.core.dndarray.DNDarray`
:   Conjugate gradients method for solving a system of linear equations :math: `Ax = b`

    Parameters
    ----------
    A : DNDarray
        2D symmetric, positive definite Matrix
    b : DNDarray
        1D vector
    x0 : DNDarray
        Arbitrary 1D starting vector
    out : DNDarray, optional
        Output Vector

`lanczos(A: heat.core.dndarray.DNDarray, m: int, v0: heat.core.dndarray.DNDarray | None = None, V_out: heat.core.dndarray.DNDarray | None = None, T_out: heat.core.dndarray.DNDarray | None = None) ‑> Tuple[heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray]`
:   The Lanczos algorithm is an iterative approximation of the solution to the eigenvalue problem, as an adaptation of
    power methods to find the m "most useful" (tending towards extreme highest/lowest) eigenvalues and eigenvectors of
    an :math:`n \times n` Hermitian matrix, where often :math:`m<<n`.
    It returns two matrices :math:`V` and :math:`T`, where:

        - :math:`V` is a Matrix of size :math:`n\times m`, with orthonormal columns, that span the Krylow subspace \n
        - :math:`T` is a Tridiagonal matrix of size :math:`m\times m`, with coefficients :math:`\alpha_1,..., \alpha_n`
          on the diagonal and coefficients :math:`\beta_1,...,\beta_{n-1}` on the side-diagonals\n

    Parameters
    ----------
    A : DNDarray
        2D Hermitian (if complex) or symmetric positive-definite matrix.
        Only distribution along axis 0 is supported, i.e. `A.split` must be `0` or `None`.
    m : int
        Number of Lanczos iterations
    v0 : DNDarray, optional
        1D starting vector of Euclidean norm 1. If not provided, a random vector will be used to start the algorithm
    V_out : DNDarray, optional
        Output Matrix for the Krylow vectors, Shape = (n, m), dtype=A.dtype, must be initialized to zero
    T_out : DNDarray, optional
        Output Matrix for the Tridiagonal matrix, Shape = (m, m), must be initialized to zero

`solve_triangular(A: heat.core.dndarray.DNDarray, b: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Solver for (possibly batched) upper triangular systems of linear equations: it returns `x` in `Ax = b`, where `A` is a (possibly batched) upper triangular matrix and
    `b` a (possibly batched) vector or matrix of suitable shape, both provided as input to the function.
    The implementation builts on the corresponding solver in PyTorch and implements an memory-distributed, MPI-parallel block-wise version thereof.

    Parameters
    ----------
    A : DNDarray
        An upper triangular invertible square (n x n) matrix or a batch thereof,  i.e. a ``DNDarray`` of shape `(..., n, n)`.
    b : DNDarray
        a (possibly batched) n x k matrix, i.e. an DNDarray of shape (..., n, k), where the batch-dimensions denoted by ... need to coincide with those of A.
        (Batched) Vectors have to be provided as ... x n x 1 matrices and the split dimension of b must the second last dimension if not None.
    Note
    ---------
    Since such a check might be computationally expensive, we do not check whether A is indeed upper triangular.
    If you require such a check, please open an issue on our GitHub page and request this feature.
