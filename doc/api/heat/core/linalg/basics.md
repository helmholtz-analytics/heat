Module heat.core.linalg.basics
==============================
Basic linear algebra operations on distributed ``DNDarray``

Functions
---------

`condest(A: heat.core.dndarray.DNDarray, p: int | str = None, algorithm: str = 'randomized', params: list = None) ‑> heat.core.dndarray.DNDarray`
:   Computes a (possibly randomized) upper estimate of the l2-condition number of the input 2D DNDarray.

    Parameters
    ----------
    A : DNDarray
        The matrix, i.e., a 2D DNDarray, for which the condition number shall be estimated.
    p : int or str (optional)
        The norm to use for the condition number computation. If None, the l2-norm (default, p=2) is used.
        So far, only p=2 is implemented.
    algorithm : str
        The algorithm to use for the estimation. Currently, only "randomized" (default) is implemented.
    params : dict (optional)
        A list of parameters required for the chosen algorithm; if not provided, default values for the respective algorithm are chosen.
        If `algorithm="randomized"` the number of random samples to use can be specified under the key "nsamples"; default is 10.

    Notes
    -----
    The "randomized" algorithm follows the approach described in [1]; note that in the paper actually the condition number w.r.t. the Frobenius norm is estimated.
    However, this yields an upper bound for the condition number w.r.t. the l2-norm as well.

    References
    ----------
    [1] T. Gudmundsson, C. S. Kenney, and A. J. Laub. Small-Sample Statistical Estimates for Matrix Norms. SIAM Journal on Matrix Analysis and Applications 1995 16:3, 776-792.

`cross(a: heat.core.dndarray.DNDarray, b: heat.core.dndarray.DNDarray, axisa: int = -1, axisb: int = -1, axisc: int = -1, axis: int = -1) ‑> heat.core.dndarray.DNDarray`
:   Returns the cross product. 2D vectors will we converted to 3D.

    Parameters
    ----------
    a : DNDarray
        First input array.
    b : DNDarray
        Second input array. Must have the same shape as 'a'.
    axisa: int
        Axis of `a` that defines the vector(s). By default, the last axis.
    axisb: int
        Axis of `b` that defines the vector(s). By default, the last axis.
    axisc: int
        Axis of the output containing the cross product vector(s). By default, the last axis.
    axis : int
        Axis that defines the vectors for which to compute the cross product. Overrides `axisa`, `axisb` and `axisc`. Default: -1

    Raises
    ------
    ValueError
        If the two input arrays don't match in shape, split, device, or comm. If the vectors are along the split axis.
    TypeError
        If 'axis' is not an integer.

    Examples
    --------
    >>> a = ht.eye(3)
    >>> b = ht.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    >>> cross = ht.cross(a, b)
    DNDarray([[0., 0., 1.],
              [1., 0., 0.],
              [0., 1., 0.]], dtype=ht.float32, device=cpu:0, split=None)

`det(a: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Returns the determinant of a square matrix.

    Parameters
    ----------
    a : DNDarray
        A square matrix or a stack of matrices. Shape = (...,M,M)

    Raises
    ------
    RuntimeError
        If the dtype of 'a' is not floating-point.
    RuntimeError
        If `a.ndim < 2` or if the length of the last two dimensions is not the same.

    Examples
    --------
    >>> a = ht.array([[-2, -1, 2], [2, 1, 4], [-3, 3, -1]])
    >>> ht.linalg.det(a)
    DNDarray(54., dtype=ht.float64, device=cpu:0, split=None)

`dot(a: heat.core.dndarray.DNDarray, b: heat.core.dndarray.DNDarray, out: heat.core.dndarray.DNDarray | None = None) ‑> heat.core.dndarray.DNDarray | float`
:   Returns the dot product of two ``DNDarrays``.
    Specifically,

        1. If both a and b are 1-D arrays, it is inner product of vectors.

        2. If both a and b are 2-D arrays, it is matrix multiplication, but using matmul or ``a@b`` is preferred.

        3. If either a or b is 0-D (scalar), it is equivalent to multiply and using ``multiply(a, b)`` or ``a*b`` is preferred.

    Parameters
    ----------
    a : DNDarray
        First input DNDarray
    b : DNDarray
        Second input DNDarray
    out : DNDarray, optional
        Output buffer.

    See Also
    --------
    vecdot
        Supports (vector) dot along an axis.

`inv(a: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Computes the multiplicative inverse of a square matrix.

    Parameters
    ----------
    a : DNDarray
        Square matrix of floating-point data type or a stack of square matrices. Shape = (...,M,M)

    Raises
    ------
    RuntimeError
        If the inverse does not exist.
    RuntimeError
        If the dtype is not floating-point
    RuntimeError
        If a is not at least two-dimensional or if the lengths of the last two dimensions are not the same.

    Examples
    --------
    >>> a = ht.array([[1.0, 2], [2, 3]])
    >>> ht.linalg.inv(a)
    DNDarray([[-3.,  2.],
              [ 2., -1.]], dtype=ht.float32, device=cpu:0, split=None)

`matmul(a: heat.core.dndarray.DNDarray, b: heat.core.dndarray.DNDarray, allow_resplit: bool = False) ‑> heat.core.dndarray.DNDarray`
:   Matrix multiplication of two ``DNDarrays``: ``a@b=c`` or ``A@B=c``.
    Returns a tensor with the result of ``a@b``. The split dimension of the returned array is
    typically the split dimension of a. If both are ``None`` and if ``allow_resplit=False`` then ``c.split`` is also ``None``.

    Batched inputs (with batch dimensions being leading dimensions) are allowed; see also the Notes below.

    Parameters
    ----------
    a : DNDarray
        matrix :math:`L \times P` or vector :math:`P` or batch of matrices: :math:`B_1 \times ... \times B_k \times L \times P`
    b : DNDarray
        matrix :math:`P \times Q` or vector :math:`P` or batch of matrices: :math:`B_1 \times ... \times B_k \times P \times Q`
    allow_resplit : bool, optional
        Whether to distribute ``a`` in the case that both ``a.split is None`` and ``b.split is None``.
        Default is ``False``. If ``True``, if both are not split then ``a`` will be distributed in-place along axis 0.

    Notes
    -----
    - For batched inputs, batch dimensions must coincide and if one matrix is split along a batch axis the other must be split along the same axis.
    - If ``a`` or ``b`` is a vector the result will also be a vector.
    - We recommend to avoid the particular split combinations ``1``-``0``, ``None``-``0``, and ``1``-``None`` (for ``a.split``-``b.split``) due to their comparably high memory consumption, if possible. Applying ``DNDarray.resplit_`` or ``heat.resplit`` on one of the two factors before calling ``matmul`` in these situations might improve performance of your code / might avoid memory bottlenecks.

    References
    ----------
    [1] R. Gu, et al., "Improving Execution Concurrency of Large-scale Matrix Multiplication on
    Distributed Data-parallel Platforms," IEEE Transactions on Parallel and Distributed Systems,
    vol 28, no. 9. 2017.

    [2] S. Ryu and D. Kim, "Parallel Huge Matrix Multiplication on a Cluster with GPGPU
    Accelerators," 2018 IEEE International Parallel and Distributed Processing Symposium
    Workshops (IPDPSW), Vancouver, BC, 2018, pp. 877-882.

    Examples
    --------
    >>> a = ht.ones((n, m), split=1)
    >>> a[0] = ht.arange(1, m + 1)
    >>> a[:, -1] = ht.arange(1, n + 1).larray
    [0/1] tensor([[1., 2.],
                  [1., 1.],
                  [1., 1.],
                  [1., 1.],
                  [1., 1.]])
    [1/1] tensor([[3., 1.],
                  [1., 2.],
                  [1., 3.],
                  [1., 4.],
                  [1., 5.]])
    >>> b = ht.ones((j, k), split=0)
    >>> b[0] = ht.arange(1, k + 1)
    >>> b[:, 0] = ht.arange(1, j + 1).larray
    [0/1] tensor([[1., 2., 3., 4., 5., 6., 7.],
                  [2., 1., 1., 1., 1., 1., 1.]])
    [1/1] tensor([[3., 1., 1., 1., 1., 1., 1.],
                  [4., 1., 1., 1., 1., 1., 1.]])
    >>> linalg.matmul(a, b).larray
    [0/1] tensor([[18.,  8.,  9., 10.],
                  [14.,  6.,  7.,  8.],
                  [18.,  7.,  8.,  9.],
                  [22.,  8.,  9., 10.],
                  [26.,  9., 10., 11.]])
    [1/1] tensor([[11., 12., 13.],
                  [ 9., 10., 11.],
                  [10., 11., 12.],
                  [11., 12., 13.],
                  [12., 13., 14.]])

`matrix_norm(x: heat.core.dndarray.DNDarray, axis: Tuple[int, int] | None = None, keepdims: bool = False, ord: int | str | None = None) ‑> heat.core.dndarray.DNDarray`
:   Computes the matrix norm of an array.

    Parameters
    ----------
    x : DNDarray
        Input array
    axis : tuple, optional
        Both axes of the matrix. If `None` 'x' must be a matrix. Default: `None`
    keepdims : bool, optional
        Retains the reduced dimension when `True`. Default: `False`
    ord : int, 'fro', 'nuc', optional
        The matrix norm order to compute. If `None` the Frobenius norm (`'fro'`) is used. Default: `None`

    See Also
    --------
    norm
        Computes the vector or matrix norm of an array.
    vector_norm
        Computes the vector norm of an array.

    Notes
    -----
    The following norms are supported:

    =====  ============================
    ord    norm for matrices
    =====  ============================
    None   Frobenius norm
    'fro'  Frobenius norm
    'nuc'  nuclear norm
    inf    max(sum(abs(x), axis=1))
    -inf   min(sum(abs(x), axis=1))
    1      max(sum(abs(x), axis=0))
    -1     min(sum(abs(x), axis=0))
    =====  ============================

    The following matrix norms are currently **not** supported:

    =====  ============================
    ord    norm for matrices
    =====  ============================
    2      largest singular value
    -2     smallest singular value
    =====  ============================

    Raises
    ------
    TypeError
        If axis is not a 2-tuple
    ValueError
        If an invalid matrix norm is given or 'x' is a vector.

    Examples
    --------
    >>> ht.matrix_norm(ht.array([[1, 2], [3, 4]]))
    DNDarray([[5.4772]], dtype=ht.float64, device=cpu:0, split=None)
    >>> ht.matrix_norm(ht.array([[1, 2], [3, 4]]), keepdims=True, ord=-1)
    DNDarray([[4.]], dtype=ht.float64, device=cpu:0, split=None)

`norm(x: heat.core.dndarray.DNDarray, axis: int | Tuple[int, int] | None = None, keepdims: bool = False, ord: int | float | str | None = None) ‑> heat.core.dndarray.DNDarray`
:   Return the vector or matrix norm of an array.

    Parameters
    ----------
    x : DNDarray
        Input vector
    axis : int, tuple, optional
        Axes along which to compute the norm. If an integer, vector norm is used. If a 2-tuple, matrix norm is used.
        If `None`, it is inferred from the dimension of the array. Default: `None`
    keepdims : bool, optional
        Retains the reduced dimension when `True`. Default: `False`
    ord : int, float, inf, -inf, 'fro', 'nuc'
        The norm order to compute. See Notes

    See Also
    --------
    vector_norm
        Computes the vector norm of an array.
    matrix_norm
        Computes the matrix norm of an array.

    Notes
    -----
    The following norms are supported:

    =====  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                L2-norm (Euclidean)
    'fro'  Frobenius norm                --
    'nuc'  nuclear norm                  --
    inf    max(sum(abs(x), axis=1))      max(abs(x))
    -inf   min(sum(abs(x), axis=1))      min(abs(x))
    0      --                            sum(x != 0)
    1      max(sum(abs(x), axis=0))      L1-norm (Manhattan)
    -1     min(sum(abs(x), axis=0))      1./sum(1./abs(a))
    2      --                            L2-norm (Euclidean)
    -2     --                            1./sqrt(sum(1./abs(a)**2))
    other  --                            sum(abs(x)**ord)**(1./ord)
    =====  ============================  ==========================

    The following matrix norms are currently **not** supported:

    =====  ============================
    ord    norm for matrices
    =====  ============================
    2      largest singular value
    -2     smallest singular value
    =====  ============================

    Raises
    ------
    ValueError
        If 'axis' has more than 2 elements

    Examples
    --------
    >>> from heat import linalg as LA
    >>> a = ht.arange(9, dtype=ht.float) - 4
    >>> a
    DNDarray([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.], dtype=ht.float32, device=cpu:0, split=None)
    >>> b = a.reshape((3, 3))
    >>> b
    DNDarray([[-4., -3., -2.],
          [-1.,  0.,  1.],
          [ 2.,  3.,  4.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> LA.norm(a)
    DNDarray(7.7460, dtype=ht.float32, device=cpu:0, split=None)
    >>> LA.norm(b)
    DNDarray(7.7460, dtype=ht.float32, device=cpu:0, split=None)
    >>> LA.norm(b, ord="fro")
    DNDarray(7.7460, dtype=ht.float32, device=cpu:0, split=None)
    >>> LA.norm(a, float("inf"))
    DNDarray([4.], dtype=ht.float32, device=cpu:0, split=None)
    >>> LA.norm(b, ht.inf)
    DNDarray([9.], dtype=ht.float32, device=cpu:0, split=None)
    >>> LA.norm(a, -ht.inf))
    DNDarray([0.], dtype=ht.float32, device=cpu:0, split=None)
    >>> LA.norm(b, -ht.inf)
    DNDarray([2.], dtype=ht.float32, device=cpu:0, split=None)
    >>> LA.norm(a, 1)
    DNDarray([20.], dtype=ht.float32, device=cpu:0, split=None)
    >>> LA.norm(b, 1)
    DNDarray([7.], dtype=ht.float32, device=cpu:0, split=None)
    >>> LA.norm(a, -1)
    DNDarray([0.], dtype=ht.float32, device=cpu:0, split=None)
    >>> LA.norm(b, -1)
    DNDarray([6.], dtype=ht.float32, device=cpu:0, split=None)
    >>> LA.norm(a, 2)
    DNDarray(7.7460, dtype=ht.float32, device=cpu:0, split=None)
    >>> LA.norm(a, -2)
    DNDarray([0.], dtype=ht.float32, device=cpu:0, split=None)
    >>> LA.norm(a, 3)
    DNDarray([5.8480], dtype=ht.float32, device=cpu:0, split=None)
    >>> LA.norm(a, -3)
    DNDarray([0.], dtype=ht.float32, device=cpu:0, split=None)
    c = ht.array([[ 1, 2, 3],
                  [-1, 1, 4]])
    >>> LA.norm(c, axis=0)
    DNDarray([1.4142, 2.2361, 5.0000], dtype=ht.float64, device=cpu:0, split=None)
    >>> LA.norm(c, axis=1)
    DNDarray([3.7417, 4.2426], dtype=ht.float64, device=cpu:0, split=None)
    >>> LA.norm(c, axis=1, ord=1)
    DNDarray([6., 6.], dtype=ht.float64, device=cpu:0, split=None)
    >>> m = ht.arange(8).reshape(2, 2, 2)
    >>> LA.norm(m, axis=(1, 2))
    DNDarray([ 3.7417, 11.2250], dtype=ht.float32, device=cpu:0, split=None)
    >>> LA.norm(m[0, :, :]), LA.norm(m[1, :, :])
    (DNDarray(3.7417, dtype=ht.float32, device=cpu:0, split=None), DNDarray(11.2250, dtype=ht.float32, device=cpu:0, split=None))

`outer(a: heat.core.dndarray.DNDarray, b: heat.core.dndarray.DNDarray, out: heat.core.dndarray.DNDarray | None = None, split: int | None = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the outer product of two 1-D DNDarrays: :math:`out(i, j) = a(i) \times b(j)`.
    Given two vectors, :math:`a = (a_0, a_1, ..., a_N)` and :math:`b = (b_0, b_1, ..., b_M)`, the outer product is:

    .. math::
        :nowrap:

        \begin{pmatrix}
           a_0 \cdot b_0  & a_0 \cdot b_1 & . & . &  a_0 \cdot b_M \\
           a_1 \cdot b_0 & a_1 \cdot b_1 & . & . & a_1 \cdot b_M \\
           . & . & . & . & .   \\
           a_N \cdot b_0 & a_N \cdot b_1 & . & . & a_N \cdot b_M
        \end{pmatrix}

    Parameters
    ----------
    a : DNDarray
        1-dimensional: :math:`N`
        Will be flattened by default if more than 1-D.
    b : DNDarray
        1-dimensional: :math:`M`
        Will be flattened by default if more than 1-D.
    out : DNDarray, optional
          2-dimensional: :math:`N \times M`
          A location where the result is stored
    split : int, optional
            Split dimension of the resulting DNDarray. Can be 0, 1, or None.
            This is only relevant if the calculations are memory-distributed.
            Default is ``split=0`` (see Notes).

    Notes
    -----
    Parallel implementation of outer product, assumes arrays are dense.
    In the classical (dense) case, one of the two arrays needs to be communicated around the processes in
    a ring.

    * Sending ``b`` around in a ring results in ``outer`` being split along the rows (``outer.split = 0``).


    * Sending ``a`` around in a ring results in ``outer`` being split along the columns (``outer.split = 1``).


    So, if specified, ``split`` defines which ``DNDarray`` stays put and which one is passed around.
    If ``split`` is ``None`` or unspecified, the result will be distributed along axis ``0``, i.e. by default ``b`` is
    passed around, ``a`` stays put.

    Examples
    --------
    >>> a = ht.arange(4)
    >>> b = ht.arange(3)
    >>> ht.outer(a, b).larray
    (3 processes)
    [0/2]   tensor([[0, 0, 0],
                    [0, 1, 2],
                    [0, 2, 4],
                    [0, 3, 6]], dtype=torch.int32)
    [1/2]   tensor([[0, 0, 0],
                    [0, 1, 2],
                    [0, 2, 4],
                    [0, 3, 6]], dtype=torch.int32)
    [2/2]   tensor([[0, 0, 0],
                    [0, 1, 2],
                    [0, 2, 4],
                    [0, 3, 6]], dtype=torch.int32)
    >>> a = ht.arange(4, split=0)
    >>> b = ht.arange(3, split=0)
    >>> ht.outer(a, b).larray
    [0/2]   tensor([[0, 0, 0],
                    [0, 1, 2]], dtype=torch.int32)
    [1/2]   tensor([[0, 2, 4]], dtype=torch.int32)
    [2/2]   tensor([[0, 3, 6]], dtype=torch.int32)
    >>> ht.outer(a, b, split=1).larray
    [0/2]   tensor([[0],
                    [0],
                    [0],
                    [0]], dtype=torch.int32)
    [1/2]   tensor([[0],
                    [1],
                    [2],
                    [3]], dtype=torch.int32)
    [2/2]   tensor([[0],
                    [2],
                    [4],
                    [6]], dtype=torch.int32)
    >>> a = ht.arange(5, dtype=ht.float32, split=0)
    >>> b = ht.arange(4, dtype=ht.float64, split=0)
    >>> out = ht.empty((5,4), dtype=ht.float64, split=1)
    >>> ht.outer(a, b, split=1, out=out)
    >>> out.larray
    [0/2]   tensor([[0., 0.],
                    [0., 1.],
                    [0., 2.],
                    [0., 3.],
                    [0., 4.]], dtype=torch.float64)
    [1/2]   tensor([[0.],
                    [2.],
                    [4.],
                    [6.],
                    [8.]], dtype=torch.float64)
    [2/2]   tensor([[ 0.],
                    [ 3.],
                    [ 6.],
                    [ 9.],
                    [12.]], dtype=torch.float64)

`projection(a: heat.core.dndarray.DNDarray, b: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Projection of vector ``a`` onto vector ``b``

    Parameters
    ----------
    a : DNDarray
        The vector to be projected. Must be a 1D ``DNDarray``
    b : DNDarray
        The vector to project onto. Must be a 1D ``DNDarray``

`trace(a: heat.core.dndarray.DNDarray, offset: int | None = 0, axis1: int | None = 0, axis2: int | None = 1, dtype: heat.core.types.datatype | None = None, out: heat.core.dndarray.DNDarray | None = None) ‑> heat.core.dndarray.DNDarray | float`
:   Return the sum along diagonals of the array

    If `a` is 2D, the sum along its diagonal with the given offset is returned, i.e. the sum of
    elements a[i, i+offset] for all i.

    If `a` has more than two dimensions, then the axes specified by `axis1` and `axis2` are used
    to determine the 2D-sub-DNDarrays whose traces are returned.
    The shape of the resulting array is the same as that of `a` with `axis1` and `axis2` removed.

    Parameters
    ----------
    a : array_like
        Input array, from which the diagonals are taken
    offset : int, optional
        Offsets of the diagonal from the main diagonal. Can be both positive and negative. Defaults to 0.
    axis1: int, optional
        Axis to be used as the first axis of the 2D-sub-arrays from which the diagonals
        should be taken. Default is the first axis of `a`
    axis2 : int, optional
        Axis to be used as the second axis of the 2D-sub-arrays from which the diagonals
        should be taken. Default is the second two axis of `a`
    dtype : dtype, optional
        Determines the data-type of the returned array and of the accumulator where the elements are
        summed. If `dtype` has value None than the dtype is the same as that of `a`
    out: ht.DNDarray, optional
        Array into which the output is placed. Its type is preserved and it must be of the right shape
        to hold the output
        Only applicable if `a` has more than 2 dimensions, thus the result is not a scalar.
        If distributed, its split axis might change eventually.

    Returns
    -------
    sum_along_diagonals : number (of defined dtype) or ht.DNDarray
        If `a` is 2D, the sum along the diagonal is returned as a scalar
        If `a` has more than 2 dimensions, then a DNDarray of sums along diagonals is returned

    Examples
    --------
    2D-case
    >>> x = ht.arange(24).reshape((4, 6))
    >>> x
        DNDarray([[ 0,  1,  2,  3,  4,  5],
                  [ 6,  7,  8,  9, 10, 11],
                  [12, 13, 14, 15, 16, 17],
                  [18, 19, 20, 21, 22, 23]], dtype=ht.int32, device=cpu:0, split=None)
    >>> ht.trace(x)
        42
    >>> ht.trace(x, 1)
        46
    >>> ht.trace(x, -2)
        31

    > 2D-case
    >>> x = x.reshape((2, 3, 4))
    >>> x
        DNDarray([[[ 0,  1,  2,  3],
                   [ 4,  5,  6,  7],
                   [ 8,  9, 10, 11]],

                  [[12, 13, 14, 15],
                   [16, 17, 18, 19],
                   [20, 21, 22, 23]]], dtype=ht.int32, device=cpu:0, split=None)
    >>> ht.trace(x)
        DNDarray([16, 18, 20, 22], dtype=ht.int32, device=cpu:0, split=None)
    >>> ht.trace(x, 1)
        DNDarray([24, 26, 28, 30], dtype=ht.int32, device=cpu:0, split=None)
    >>> ht.trace(x, axis1=0, axis2=2)
        DNDarray([13, 21, 29], dtype=ht.int32, device=cpu:0, split=None)

`transpose(a: heat.core.dndarray.DNDarray, axes: List[int] | None = None) ‑> heat.core.dndarray.DNDarray`
:   Permute the dimensions of an array.

    Parameters
    ----------
    a : DNDarray
        Input array.
    axes : None or List[int,...], optional
        By default, reverse the dimensions, otherwise permute the axes according to the values given.

`tril(m: heat.core.dndarray.DNDarray, k: int = 0) ‑> heat.core.dndarray.DNDarray`
:   Returns the lower triangular part of the ``DNDarray``.
    The lower triangular part of the array is defined as the elements on and below the diagonal, the other elements of
    the result array are set to 0.
    The argument ``k`` controls which diagonal to consider. If ``k=0``, all elements on and below the main diagonal are
    retained. A positive value includes just as many diagonals above the main diagonal, and similarly a negative
    value excludes just as many diagonals below the main diagonal.

    Parameters
    ----------
    m : DNDarray
        Input array for which to compute the lower triangle.
    k : int, optional
        Diagonal above which to zero elements. ``k=0`` (default) is the main diagonal, ``k<0`` is below and ``k>0`` is above.

`triu(m: heat.core.dndarray.DNDarray, k: int = 0) ‑> heat.core.dndarray.DNDarray`
:   Returns the upper triangular part of the ``DNDarray``.
    The upper triangular part of the array is defined as the elements on and below the diagonal, the other elements of the result array are set to 0.
    The argument ``k`` controls which diagonal to consider. If ``k=0``, all elements on and below the main diagonal are
    retained. A positive value includes just as many diagonals above the main diagonal, and similarly a negative
    value excludes just as many diagonals below the main diagonal.

    Parameters
    ----------
    m : DNDarray
        Input array for which to compute the upper triangle.
    k : int, optional
        Diagonal above which to zero elements. ``k=0`` (default) is the main diagonal, ``k<0`` is below and ``k>0`` is above.

`vdot(x1: heat.core.dndarray.DNDarray, x2: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Computes the dot product of two vectors. Higher-dimensional arrays will be flattened.

    Parameters
    ----------
    x1 : DNDarray
        first input array. If it's complex, it's complex conjugate will be used.
    x2 : DNDarray
        second input array.

    Raises
    ------
    ValueError
        If the number of elements is inconsistent.

    See Also
    --------
    dot
        Return the dot product without using the complex conjugate.

    Examples
    --------
    >>> a = ht.array([1 + 1j, 2 + 2j])
    >>> b = ht.array([1 + 2j, 3 + 4j])
    >>> ht.vdot(a, b)
    DNDarray([(17+3j)], dtype=ht.complex64, device=cpu:0, split=None)
    >>> ht.vdot(b, a)
    DNDarray([(17-3j)], dtype=ht.complex64, device=cpu:0, split=None)

`vecdot(x1: heat.core.dndarray.DNDarray, x2: heat.core.dndarray.DNDarray, axis: int | None = None, keepdims: bool | None = None) ‑> heat.core.dndarray.DNDarray`
:   Computes the (vector) dot product of two DNDarrays.

    Parameters
    ----------
    x1 : DNDarray
        first input array.
    x2 : DNDarray
        second input array. Must be compatible with x1.
    axis : int, optional
        axis over which to compute the dot product. The last dimension is used if 'None'.
    keepdims : bool, optional
        If this is set to 'True', the axes which are reduced are left in the result as dimensions with size one.

    See Also
    --------
    dot
        NumPy-like dot function.

    Examples
    --------
    >>> ht.vecdot(ht.full((3, 3, 3), 3), ht.ones((3, 3)), axis=0)
    DNDarray([[9., 9., 9.],
              [9., 9., 9.],
              [9., 9., 9.]], dtype=ht.float32, device=cpu:0, split=None)

`vector_norm(x: heat.core.dndarray.DNDarray, axis: int | Tuple[int] | None = None, keepdims=False, ord: int | float | None = None) ‑> heat.core.dndarray.DNDarray`
:   Computes the vector norm of an array.

    Parameters
    ----------
    x : DNDarray
        Input array
    axis : int, tuple, optional
        Axis along which to compute the vector norm. If `None` 'x' must be a vector. Default: `None`
    keepdims : bool, optional
        Retains the reduced dimension when `True`. Default: `False`
    ord : int, float, optional
        The norm order to compute. If `None` the euclidean norm (`2`) is used. Default: `None`

    See Also
    --------
    norm
        Computes the vector norm or matrix norm of an array.
    matrix_norm
        Computes the matrix norm of an array.

    Notes
    -----
    The following norms are suported:

    =====  ==========================
    ord    norm for vectors
    =====  ==========================
    None   L2-norm (Euclidean)
    inf    max(abs(x))
    -inf   min(abs(x))
    0      sum(x != 0)
    1      L1-norm (Manhattan)
    -1     1./sum(1./abs(a))
    2      L2-norm (Euclidean)
    -2     1./sqrt(sum(1./abs(a)**2))
    other  sum(abs(x)**ord)**(1./ord)
    =====  ==========================

    Raises
    ------
    TypeError
        If axis is not an integer or a 1-tuple
    ValueError
        If an invalid vector norm is given.

    Examples
    --------
    >>> ht.vector_norm(ht.array([1, 2, 3, 4]))
    DNDarray([5.4772], dtype=ht.float64, device=cpu:0, split=None)
    >>> ht.vector_norm(ht.array([[1, 2], [3, 4]]), axis=0, ord=1)
    DNDarray([[4., 6.]], dtype=ht.float64, device=cpu:0, split=None)
