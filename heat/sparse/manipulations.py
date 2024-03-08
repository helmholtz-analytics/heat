"""Manipulation operations for (potentially distributed) `DCSR_matrix`."""

from __future__ import annotations

from heat.sparse.dcsr_matrix import DCSR_matrix
from heat.sparse.factories import sparse_csr_matrix
from ..core.memory import sanitize_memory_layout
from ..core.dndarray import DNDarray
from ..core.factories import empty

__all__ = [
    "to_dense",
    "to_sparse",
]


def to_sparse(array: DNDarray) -> DCSR_matrix:
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

    Notes
    -----
    This method allows for the conversion of a DNDarray into a sparse DCSR_matrix representation,
    which is useful for handling large and sparse datasets efficiently.

    Examples
    --------
    >>> dense_array = ht.array([[1, 0, 0], [0, 0, 2], [0, 3, 0]])
    >>> sparse_matrix = dense_array.to_sparse()

    """
    array.balance_()
    result = sparse_csr_matrix(
        array.larray, dtype=array.dtype, device=array.device, comm=array.comm, is_split=array.split
    )
    return result


DNDarray.to_sparse = to_sparse
DNDarray.to_sparse.__doc__ = to_sparse.__doc__


def to_dense(sparse_matrix: DCSR_matrix, order="C", out: DNDarray = None) -> DNDarray:
    """
    Convert :class:`~heat.sparse.DCSR_matrix` to a dense :class:`~heat.core.DNDarray`.
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


DCSR_matrix.todense = lambda self, order="C", out=None: to_dense(self, order, out)
DCSR_matrix.to_dense = lambda self, order="C", out=None: to_dense(self, order, out)
