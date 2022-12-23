"""Generalized operations for DCSR_matrix"""
import torch
import numpy as np

from heat.sparse.dcsr_matrix import DCSR_matrix

from . import factories
from ..core.communication import MPI
from ..core.dndarray import DNDarray
from ..core import types

from typing import Callable, Optional, Dict

__all__ = []


def __binary_op_csr(
    operation: Callable,
    t1: DCSR_matrix,
    t2: DCSR_matrix,
    out: Optional[DCSR_matrix] = None,
    fn_kwargs: Optional[Dict] = {},
) -> DCSR_matrix:
    """
    Generic wrapper for element-wise binary operations of two operands.
    Takes the operation function and the two operands involved in the operation as arguments.

    Parameters
    ----------
    operation : PyTorch function
        The operation to be performed. Function that performs operation elements-wise on the involved tensors,
        e.g. add values from other to self
    t1: DCSR_matrix
        The first operand involved in the operation.
    t2: DCSR_matrix
        The second operand involved in the operation.
    out: DCSR_matrix, optional
        Output buffer in which the result is placed. If not provided, a freshly allocated matrix is returned.
    fn_kwargs: Dict, optional
        keyword arguments used for the given operation
        Default: {} (empty dictionary)

    Returns
    -------
    result: ht.sparse.DCSR_matrix
        A DCSR_matrix containing the results of element-wise operation.
    """
    if not np.isscalar(t1) and not isinstance(t1, DCSR_matrix):
        raise TypeError(
            f"Only Dcsr_matrices and numeric scalars are supported, but input was {type(t1)}"
        )
    if not np.isscalar(t2) and not isinstance(t2, DCSR_matrix):
        raise TypeError(
            f"Only Dcsr_matrices and numeric scalars are supported, but input was {type(t2)}"
        )

    if not isinstance(t1, DCSR_matrix) and not isinstance(t2, DCSR_matrix):
        raise TypeError(
            f"Operator only to be used with Dcsr_matrices, but input types were {type(t1)} and {type(t2)}"
        )

    promoted_type = types.result_type(t1, t2).torch_type()

    # If one of the inputs is a scalar
    # just perform the operation on the data tensor
    # and create a new sparse matrix
    if np.isscalar(t1) or np.isscalar(t2):
        matrix = t1
        scalar = t2

        if np.isscalar(t1):
            matrix = t2
            scalar = t1

        res_values = operation(matrix.larray.values().to(promoted_type), scalar, **fn_kwargs)
        res_torch_sparse_csr = torch.sparse_csr_tensor(
            matrix.lindptr,
            matrix.lindices,
            res_values,
            size=matrix.lshape,
            device=matrix.device.torch_device,
        )
        return factories.sparse_csr_matrix(
            res_torch_sparse_csr, is_split=matrix.split, comm=matrix.comm, device=matrix.device
        )

    if t1.shape != t2.shape:
        raise ValueError(
            f"Dcsr_matrices of different shapes are not supported, but input shapes were {t1.shape} and {t2.shape}"
        )
    output_shape = t1.shape

    if t1.split is not None or t2.split is not None:
        if t1.split is None:
            t1 = factories.sparse_csr_matrix(t1.larray, split=0)

        if t2.split is None:
            t2 = factories.sparse_csr_matrix(t2.larray, split=0)

    output_split = t1.split
    output_device = t1.device
    output_comm = t1.comm
    output_balanced = t1.balanced
    output_lshape = t1.lshape

    # sanitize out buffer
    if out is not None:
        if out.shape != output_shape:
            raise ValueError(
                f"Output buffer shape is not compatible with the result. Expected {output_shape}, received {out.shape}"
            )

        if out.split != output_split:
            if out.split is None:
                out = factories.sparse_csr_matrix(out.larray, split=0)
            else:
                out = factories.sparse_csr_matrix(
                    torch.sparse_csr_tensor(
                        torch.tensor(out.indptr, dtype=torch.int64),
                        torch.tensor(out.indices, dtype=torch.int64),
                        torch.tensor(out.data),
                    )
                )

        out.device = output_device
        out.balanced = (
            output_balanced  # At this point, inputs and out buffer assumed to be balanced
        )

    # If there are no non-zero elements in either tensors, skip torch operation to
    #   1. Avoid unnecessary computation
    #   2. Avoid undefined behaviour when no data in process
    if t1.lnnz == 0 and t2.lnnz == 0:
        result = t1.larray
    else:
        result = operation(t1.larray.to(promoted_type), t2.larray.to(promoted_type), **fn_kwargs)

    if output_split is not None:
        output_gnnz = torch.tensor(result._nnz())
        output_comm.Allreduce(MPI.IN_PLACE, output_gnnz, MPI.SUM)
        output_gnnz = output_gnnz.item()
    else:
        output_gnnz = torch.tensor(result._nnz())

    output_type = types.canonical_heat_type(result.dtype)

    if out is None:
        return DCSR_matrix(
            array=torch.sparse_csr_tensor(
                result.crow_indices().to(torch.int64),
                result.col_indices().to(torch.int64),
                result.values(),
                size=output_lshape,
            ),
            gnnz=output_gnnz,
            gshape=output_shape,
            dtype=output_type,
            split=output_split,
            device=output_device,
            comm=output_comm,
            balanced=output_balanced,
        )

    out.larray.copy_(result)
    out.gnnz = output_gnnz
    out.dtype = output_type
    out.comm = output_comm
    return out
