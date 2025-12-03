"""Manipulation operations for (potentially distributed) `DCSR_matrix`."""

from __future__ import annotations

from heat.sparse.dcsx_matrix import DCSC_matrix, DCSR_matrix, __DCSX_matrix
from heat.sparse.factories import sparse_csc_matrix, sparse_csr_matrix
from ..core.memory import sanitize_memory_layout
from ..core.dndarray import DNDarray
from ..core.factories import empty

__all__ = [
    "to_dense",
    "to_sparse_csr",
    "to_sparse_csc",
]


def __to_sparse(array: DNDarray, orientation="row") -> __DCSX_matrix:
    """
    Convert the distributed array to a sparse DCSX_matrix representation.
    This is a common method for converting a distributed array to a sparse matrix representation.

    Parameters
    ----------
    array : DNDarray
        The distributed array to be converted to a sparse matrix.

    orientation : str
        The orientation of the sparse matrix. Options: ``'row'`` or ``'col'``. Default is ``'row'``.

    Returns
    -------
    DCSX_matrix

    Raises
    ------
    ValueError
        If the orientation is not ``'row'`` or ``'col'``.
    """
    if orientation not in ["row", "col"]:
        raise ValueError(f"Invalid orientation: {orientation}. Options: 'row' or 'col'")

    array.balance_()
    method = sparse_csr_matrix if orientation == "row" else sparse_csc_matrix
    result = method(
        array.larray,
        dtype=array.dtype,
        is_split=array.split,
        device=array.device,
        comm=array.comm,
    )
    return result


def to_sparse_csr(array: DNDarray) -> DCSR_matrix:
    """
    Convert the distributed array to a sparse DCSR_matrix representation.

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
    """
    return __to_sparse(array, orientation="row")


DNDarray.to_sparse_csr = to_sparse_csr
DNDarray.to_sparse_csr.__doc__ = to_sparse_csr.__doc__


def to_sparse_csc(array: DNDarray) -> DCSC_matrix:
    """
    Convert the distributed array to a sparse DCSC_matrix representation.

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
    """
    return __to_sparse(array, orientation="col")


DNDarray.to_sparse_csc = to_sparse_csc
DNDarray.to_sparse_csc.__doc__ = to_sparse_csc.__doc__


def to_dense(sparse_matrix: __DCSX_matrix, order="C", out: DNDarray = None) -> DNDarray:
    """
    Convert :class:`~heat.sparse.DCSX_matrix` to a dense :class:`~heat.core.DNDarray`.
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
    """
    if out is not None:
        if out.shape != sparse_matrix.shape:
            raise ValueError(
                f"Expected output buffer with shape {sparse_matrix.shape} but was {out.shape}"
            )

        if out.split != sparse_matrix.split:
            raise ValueError(
                f"Expected output buffer with split axis {sparse_matrix.split} but was {out.split}"
            )

    if out is None:
        out = empty(
            shape=sparse_matrix.shape,
            split=sparse_matrix.split,
            dtype=sparse_matrix.dtype,
            device=sparse_matrix.device,
            comm=sparse_matrix.comm,
            order=order,
        )

    out.larray = sanitize_memory_layout(sparse_matrix.larray.to_dense(), order=order)
    return out


__DCSX_matrix.todense = lambda self, order="C", out=None: to_dense(self, order, out)
__DCSX_matrix.to_dense = lambda self, order="C", out=None: to_dense(self, order, out)
