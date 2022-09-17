"""
Manipulation operations for (potentially distributed) `Dcsr_matrix`.
"""
from __future__ import annotations

from heat.sparse.dcsr_matrix import Dcsr_matrix

from ..core.dndarray import DNDarray
from ..core.factories import empty

__all__ = [
    "todense",
]


def todense(sparse_matrix: Dcsr_matrix, order=None, out: DNDarray = None):
    """
    Convert :class:`~heat.sparse.Dcsr_matrix` to a dense :class:`~heat.core.DNDarray`.
    Output follows the same distribution among processes as the input

    Parameters
    ----------
    sparse_matrix : :class:`~heat.sparse.Dcsr_matrix`
        The sparse csr matrix which is to be converted to a dense array
    order: str, optional
        TODO
        Options: ``'C'`` or ``'F'``. Specifies the memory layout of the newly created array. Default is ``order='C'``,
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
            raise ValueError("Shape of output buffer does not match")

        if out.split != sparse_matrix.split:
            raise ValueError("Split axis of output buffer does not match")

    if out is None:
        out = empty(
            shape=sparse_matrix.shape,
            split=sparse_matrix.split,
            dtype=sparse_matrix.dtype,
            device=sparse_matrix.device,
            comm=sparse_matrix.comm,
        )

    out.larray = sparse_matrix.larray.to_dense()
    return out


Dcsr_matrix.todense = lambda self, order=None, out=None: todense(self, order, out)
