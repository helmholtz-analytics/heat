"""
Generalized MPI operations. i.e. element-wise binary operations
"""
import torch

from heat.sparse.dcsr_matrix import Dcsr_matrix

from . import factories
from ..core.communication import MPI
from ..core.dndarray import DNDarray
from ..core import types

from typing import Callable, Optional, Type, Union, Dict

__all__ = []


def __binary_op_sparse(
    operation: Callable,
    t1: Union[Dcsr_matrix, int, float],
    t2: Union[Dcsr_matrix, int, float],
    out: Optional[Dcsr_matrix] = None,
    where: Optional[DNDarray] = None,
    fn_kwargs: Optional[Dict] = {},
) -> Dcsr_matrix:
    # TODO: where argument

    # Check inputs --> for now, only `Dcsr_matrix` accepted
    # TODO: Might have to include scalars and `DNDarray`
    if not isinstance(t1, Dcsr_matrix):
        raise TypeError(
            f"Only Dcsr_matrices and numeric scalars are supported, but input was {type(t1)}"
        )
    if not isinstance(t2, Dcsr_matrix):
        raise TypeError(
            f"Only Dcsr_matrices and numeric scalars are supported, but input was {type(t2)}"
        )
    promoted_type = types.result_type(t1, t2).torch_type()

    # For now restrict input shapes to be the same
    # TODO: allow broadcasting?
    if t1.shape != t2.shape:
        raise ValueError(
            f"Dcsr_matrices of different shapes are not supported, but input shapes were {t1.shape} and {t2.shape}"
        )
    output_shape = t1.shape
    output_lshape = t1.lshape

    def __get_out_params(target, other=None, map=None):
        """
        Getter for the output parameters of a binary operation with target distribution.
        If `other` is provided, its distribution will be matched to `target` or, if provided,
        redistributed according to `map`.

        Parameters
        ----------
        target : Dcsr_matrix
            Dcsr_matrix determining the parameters
        other : Dcsr_matrix
            Dcsr_matrix to be adapted
        map : Tensor
            lshape_map `other` should be matched to. Defaults to `target.lshape_map`

        Returns
        -------
        Tuple
            split, device, comm, balanced, [other]
        """
        # For now, assume that distributions of both the inputs are the same
        # TODO: allow unbalanced inputs that require resplit?
        # if other is not None:
        #     if out is None:
        #         other = sanitation.sanitize_distribution(other, target=target, diff_map=map)
        #     return target.split, target.device, target.comm, target.balanced, other
        return target.split, target.device, target.comm, target.balanced

    if t1.split is not None or t2.split is not None:
        if t1.split is None:
            t1 = factories.sparse_csr_matrix(t1.larray, split=0)

        if t2.split is None:
            t2 = factories.sparse_csr_matrix(t2.larray, split=0)
    output_split, output_device, output_comm, output_balanced = __get_out_params(t1)

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
                # Gather out to single process
                # TODO: Since out is going to be rewritten anyways,
                # can we do this more efficiently
                # without having to copy all values?
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
    # TODO: torch arithmetic operations not implemented for Integers
    result = operation(t1.larray.to(promoted_type), t2.larray.to(promoted_type), **fn_kwargs)

    output_gnnz = torch.tensor(result._nnz())
    output_comm.Allreduce(MPI.IN_PLACE, output_gnnz, MPI.SUM)
    output_gnnz = output_gnnz.item()

    output_lnnz = result._nnz()

    output_type = types.canonical_heat_type(result.dtype)

    if out is None and where is None:
        return Dcsr_matrix(
            array=result,
            gnnz=output_gnnz,
            lnnz=output_lnnz,
            gshape=output_shape,
            lshape=output_lshape,
            dtype=output_type,
            split=output_split,
            device=output_device,
            comm=output_comm,
            balanced=output_balanced,
        )

    # if where is not None:
    #     if out is None:
    #         out = factories.empty(
    #             output_shape,
    #             dtype=promoted_type,
    #             split=output_split,
    #             device=output_device,
    #             comm=output_comm,
    #         )
    #     if where.split != out.split:
    #         where = sanitation.sanitize_distribution(where, target=out)
    #     result = torch.where(where.larray, result, out.larray)

    # TODO: Any better way to do this?
    out.larray.copy_(result)
    out.gnnz = output_gnnz
    out.lnnz = output_lnnz
    out.dtype = output_type
    out.comm = output_comm
    return out
