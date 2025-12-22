Module heat.sparse.manipulations
================================
Manipulation operations for (potentially distributed) `DCSR_matrix`.

Functions
---------

`to_dense(sparse_matrix: __DCSX_matrix, order='C', out: DNDarray = None) ‑> heat.core.dndarray.DNDarray`
:   Convert :class:`~heat.sparse.DCSX_matrix` to a dense :class:`~heat.core.DNDarray`.
    Output follows the same distribution among processes as the input

    Parameters
    ----------
    sparse_matrix : :class:`~heat.sparse.DCSR_matrix`
        The sparse csr matrix which is to be converted to a dense array
    order: str, optional
        Options: ``'C'`` or ``'F'``. Specifies the memory layout of the newly created `DNDarray`. Default is ``order='C'``,
        meaning the array will be stored in row-major order (C-like). If ``order=‘F’``, the array will be stored in
        column-major order (Fortran-like).
    out : DNDarray
        Output buffer in which the values of the dense format is stored.
        If not specified, a new DNDarray is created.

    Raises
    ------
    ValueError
        If shape of output buffer does not match that of the input.
    ValueError
        If split axis of output buffer does not match that of the input.

    Examples
    --------
    >>> indptr = torch.tensor([0, 2, 3, 6])
    >>> indices = torch.tensor([0, 2, 2, 0, 1, 2])
    >>> data = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float)
    >>> torch_sparse_csr = torch.sparse_csr_tensor(indptr, indices, data)
    >>> heat_sparse_csr = ht.sparse.sparse_csr_matrix(torch_sparse_csr, split=0)
    >>> heat_sparse_csr
    (indptr: tensor([0, 2, 3, 6]), indices: tensor([0, 2, 2, 0, 1, 2]), data: tensor([1., 2., 3., 4., 5., 6.]), dtype=ht.float32, device=cpu:0, split=0)
    >>> heat_sparse_csr.todense()
    DNDarray([[1., 0., 2.],
              [0., 0., 3.],
              [4., 5., 6.]], dtype=ht.float32, device=cpu:0, split=0)

`to_sparse_csc(array: DNDarray) ‑> heat.sparse.dcsx_matrix.DCSC_matrix`
:   Convert the distributed array to a sparse DCSC_matrix representation.

    Parameters
    ----------
    array : DNDarray
        The distributed array to be converted to a sparse DCSC_matrix.

    Returns
    -------
    DCSC_matrix
        A sparse DCSC_matrix representation of the input DNDarray.

    Examples
    --------
    >>> dense_array = ht.array([[1, 0, 0], [0, 0, 2], [0, 3, 0]])
    >>> dense_array.to_sparse_csc()
    (indptr: tensor([0, 1, 2, 3]), indices: tensor([0, 2, 1]), data: tensor([1, 3, 2]), dtype=ht.int64, device=cpu:0, split=None)

`to_sparse_csr(array: DNDarray) ‑> heat.sparse.dcsx_matrix.DCSR_matrix`
:   Convert the distributed array to a sparse DCSR_matrix representation.

    Parameters
    ----------
    array : DNDarray
        The distributed array to be converted to a sparse DCSR_matrix.

    Returns
    -------
    DCSR_matrix
        A sparse DCSR_matrix representation of the input DNDarray.

    Examples
    --------
    >>> dense_array = ht.array([[1, 0, 0], [0, 0, 2], [0, 3, 0]])
    >>> dense_array.to_sparse_csr()
    (indptr: tensor([0, 1, 2, 3]), indices: tensor([0, 2, 1]), data: tensor([1, 2, 3]), dtype=ht.int64, device=cpu:0, split=None)
