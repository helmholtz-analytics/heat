Module heat.sparse.factories
============================
Provides high-level DCSR_matrix initialization functions

Functions
---------

`sparse_csc_matrix(obj: Iterable, dtype: Type[heat.core.types.datatype] | None = None, split: int | None = None, is_split: int | None = None, device: heat.core.devices.Device | None = None, comm: heat.core.communication.Communication | None = None) ‑> heat.sparse.dcsx_matrix.DCSC_matrix`
:   Create a :class:`~heat.sparse.DCSC_matrix`.

    Parameters
    ----------
    obj : array_like
        A tensor or array, any object exposing the array interface, an object whose ``__array__`` method returns an
        array, or any (nested) sequence. Sparse tensor that needs to be distributed.
    dtype : datatype, optional
        The desired data-type for the sparse matrix. If not given, then the type will be determined as the minimum type required
        to hold the objects in the sequence. This argument can only be used to ‘upcast’ the array. For downcasting, use
        the :func:`~heat.sparse.DCSC_matrix.astype` method.
    split : int or None, optional
        The axis along which the passed array content ``obj`` is split and distributed in memory. DCSC_matrix only supports
        distribution along axis 1. Mutually exclusive with ``is_split``.
    is_split : int or None, optional
        Specifies the axis along which the local data portions, passed in obj, are split across all machines. DCSC_matrix only
        supports distribution along axis 1. Useful for interfacing with other distributed-memory code. The shape of the global
        array is automatically inferred. Mutually exclusive with ``split``.
    device : str or Device, optional
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on (i.e. globally set default
        device).
    comm : Communication, optional
        Handle to the nodes holding distributed array chunks.

    Raises
    ------
    ValueError
        If split and is_split parameters are not one of 1 or None.

    Examples
    --------
    Create a :class:`~heat.sparse.DCSC_matrix` from :class:`torch.Tensor` (layout ==> torch.sparse_csc)
    >>> indptr = torch.tensor([0, 2, 3, 6])
    >>> indices = torch.tensor([0, 2, 2, 0, 1, 2])
    >>> data = torch.tensor([1.0, 4.0, 5.0, 2.0, 3.0, 6.0], dtype=torch.float)
    >>> torch_sparse_csc = torch.sparse_csc_tensor(indptr, indices, data)
    >>> heat_sparse_csc = ht.sparse.sparse_csc_matrix(torch_sparse_csc, split=1)
    >>> heat_sparse_csc
    (indptr: tensor([0, 2, 3, 6]), indices: tensor([0, 2, 2, 0, 1, 2]), data: tensor([1., 4., 5., 2., 3., 6.]), dtype=ht.float32, device=cpu:0, split=1)

    Create a :class:`~heat.sparse.DCSC_matrix` from :class:`scipy.sparse.csc_matrix`
    >>> scipy_sparse_csc = scipy.sparse.csc_matrix((data, indices, indptr))
    >>> heat_sparse_csc = ht.sparse.sparse_csc_matrix(scipy_sparse_csc, split=1)
    >>> heat_sparse_csc
    (indptr: tensor([0, 2, 3, 6], dtype=torch.int32), indices: tensor([0, 2, 2, 0, 1, 2], dtype=torch.int32), data: tensor([1., 4., 5., 2., 3., 6.]), dtype=ht.float32, device=cpu:0, split=1)

    Create a :class:`~heat.sparse.DCSC_matrix` using data that is already distributed (with `is_split`)
    >>> indptrs = [torch.tensor([0, 2, 3]), torch.tensor([0, 3])]
    >>> indices = [torch.tensor([0, 2, 2]), torch.tensor([0, 1, 2])]
    >>> data = [torch.tensor([1, 2, 3], dtype=torch.float),
                torch.tensor([4, 5, 6], dtype=torch.float)]
    >>> rank = ht.MPI_WORLD.rank
    >>> local_indptr = indptrs[rank]
    >>> local_indices = indices[rank]
    >>> local_data = data[rank]
    >>> local_torch_sparse_csr = torch.sparse_csr_tensor(local_indptr, local_indices, local_data)
    >>> heat_sparse_csr = ht.sparse.sparse_csr_matrix(local_torch_sparse_csr, is_split=0)
    >>> heat_sparse_csr
    (indptr: tensor([0, 2, 3, 6]), indices: tensor([0, 2, 2, 0, 1, 2]), data: tensor([1., 2., 3., 4., 5., 6.]), dtype=ht.float32, device=cpu:0, split=1)

    Create a :class:`~heat.sparse.DCSC_matrix` from List
    >>> ht.sparse.sparse_csc_matrix([[0, 0, 1], [1, 0, 2], [0, 0, 3]])
    (indptr: tensor([0, 1, 1, 4]), indices: tensor([1, 0, 1, 2]), data: tensor([1, 1, 2, 3]), dtype=ht.int64, device=cpu:0, split=None)

`sparse_csr_matrix(obj: Iterable, dtype: Type[heat.core.types.datatype] | None = None, split: int | None = None, is_split: int | None = None, device: heat.core.devices.Device | None = None, comm: heat.core.communication.Communication | None = None) ‑> heat.sparse.dcsx_matrix.DCSR_matrix`
:   Create a :class:`~heat.sparse.DCSR_matrix`.

    Parameters
    ----------
    obj : array_like
        A tensor or array, any object exposing the array interface, an object whose ``__array__`` method returns an
        array, or any (nested) sequence. Sparse tensor that needs to be distributed.
    dtype : datatype, optional
        The desired data-type for the sparse matrix. If not given, then the type will be determined as the minimum type required
        to hold the objects in the sequence. This argument can only be used to ‘upcast’ the array. For downcasting, use
        the :func:`~heat.sparse.DCSR_matrix.astype` method.
    split : int or None, optional
        The axis along which the passed array content ``obj`` is split and distributed in memory. DCSR_matrix only supports
        distribution along axis 0. Mutually exclusive with ``is_split``.
    is_split : int or None, optional
        Specifies the axis along which the local data portions, passed in obj, are split across all machines. DCSR_matrix only
        supports distribution along axis 0. Useful for interfacing with other distributed-memory code. The shape of the global
        array is automatically inferred. Mutually exclusive with ``split``.
    device : str or Device, optional
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on (i.e. globally set default
        device).
    comm : Communication, optional
        Handle to the nodes holding distributed array chunks.

    Raises
    ------
    ValueError
        If split and is_split parameters are not one of 0 or None.

    Examples
    --------
    Create a :class:`~heat.sparse.DCSR_matrix` from :class:`torch.Tensor` (layout ==> torch.sparse_csr)
    >>> indptr = torch.tensor([0, 2, 3, 6])
    >>> indices = torch.tensor([0, 2, 2, 0, 1, 2])
    >>> data = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float)
    >>> torch_sparse_csr = torch.sparse_csr_tensor(indptr, indices, data)
    >>> heat_sparse_csr = ht.sparse.sparse_csr_matrix(torch_sparse_csr, split=0)
    >>> heat_sparse_csr
    (indptr: tensor([0, 2, 3, 6]), indices: tensor([0, 2, 2, 0, 1, 2]), data: tensor([1., 2., 3., 4., 5., 6.]), dtype=ht.float32, device=cpu:0, split=0)

    Create a :class:`~heat.sparse.DCSR_matrix` from :class:`scipy.sparse.csr_matrix`
    >>> scipy_sparse_csr = scipy.sparse.csr_matrix((data, indices, indptr))
    >>> heat_sparse_csr = ht.sparse.sparse_csr_matrix(scipy_sparse_csr, split=0)
    >>> heat_sparse_csr
    (indptr: tensor([0, 2, 3, 6], dtype=torch.int32), indices: tensor([0, 2, 2, 0, 1, 2], dtype=torch.int32), data: tensor([1., 2., 3., 4., 5., 6.]), dtype=ht.float32, device=cpu:0, split=0)

    Create a :class:`~heat.sparse.DCSR_matrix` using data that is already distributed (with `is_split`)
    >>> indptrs = [torch.tensor([0, 2, 3]), torch.tensor([0, 3])]
    >>> indices = [torch.tensor([0, 2, 2]), torch.tensor([0, 1, 2])]
    >>> data = [torch.tensor([1, 2, 3], dtype=torch.float),
                torch.tensor([4, 5, 6], dtype=torch.float)]
    >>> rank = ht.MPI_WORLD.rank
    >>> local_indptr = indptrs[rank]
    >>> local_indices = indices[rank]
    >>> local_data = data[rank]
    >>> local_torch_sparse_csr = torch.sparse_csr_tensor(local_indptr, local_indices, local_data)
    >>> heat_sparse_csr = ht.sparse.sparse_csr_matrix(local_torch_sparse_csr, is_split=0)
    >>> heat_sparse_csr
    (indptr: tensor([0, 2, 3, 6]), indices: tensor([0, 2, 2, 0, 1, 2]), data: tensor([1., 2., 3., 4., 5., 6.]), dtype=ht.float32, device=cpu:0, split=0)

    Create a :class:`~heat.sparse.DCSR_matrix` from List
    >>> ht.sparse.sparse_csr_matrix([[0, 0, 1], [1, 0, 2], [0, 0, 3]])
    (indptr: tensor([0, 1, 3, 4]), indices: tensor([2, 0, 2, 2]), data: tensor([1, 1, 2, 3]), dtype=ht.int64, device=cpu:0, split=None)
