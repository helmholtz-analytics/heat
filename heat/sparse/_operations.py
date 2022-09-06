"""
Generalized MPI operations. i.e. element-wise binary operations
"""

import numpy as np
import torch

from heat.sparse.dcsr_matrix import Dcsr_matrix

from . import factories
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

    # Check inputs
    if not np.isscalar(t1) and not isinstance(t1, Dcsr_matrix):
        raise TypeError(
            "Only Dcsr_matrices and numeric scalars are supported, but input was {}".format(
                type(t1)
            )
        )
    if not np.isscalar(t2) and not isinstance(t2, Dcsr_matrix):
        raise TypeError(
            "Only Dcsr_matrices and numeric scalars are supported, but input was {}".format(
                type(t2)
            )
        )

    def __get_out_params(target, other=None, map=None):
        """
        Getter for the output parameters of a binary operation with target distribution.
        If `other` is provided, its distribution will be matched to `target` or, if provided,
        redistributed according to `map`.

        Parameters
        ----------
        target : DNDarray
            DNDarray determining the parameters
        other : DNDarray
            DNDarray to be adapted
        map : Tensor
            lshape_map `other` should be matched to. Defaults to `target.lshape_map`

        Returns
        -------
        Tuple
            split, device, comm, balanced, [other]
        """
        if other is not None:
            # if out is None:
            # other = sanitation.sanitize_distribution(other, target=target, diff_map=map)
            return target.split, target.device, target.comm, target.balanced, other
        return target.split, target.device, target.comm, target.balanced

    if (isinstance(t1, Dcsr_matrix) and t1.split) or (isinstance(t2, Dcsr_matrix) and t2.split):
        if isinstance(t1, Dcsr_matrix) and t1.split is None:
            t1 = factories.sparse_csr_matrix(t1.larray, split=0)

        if isinstance(t2, Dcsr_matrix) and t2.split is None:
            t2 = factories.sparse_csr_matrix(t2.larray, split=0)
        output_split, output_device, output_comm, output_balanced, t1 = __get_out_params(t2, t1)
    else:  # both are not split
        output_split, output_device, output_comm, output_balanced = __get_out_params(t1)

    # TODO: does not really work with other data types
    result = operation(t1.larray.to(torch.float32), t2.larray.to(torch.float32), **fn_kwargs)

    lnnz = result.values().shape[0]
    if out is None and where is None:
        return Dcsr_matrix(
            array=result,
            gnnz=lnnz,
            lnnz=lnnz,
            gshape=t1.shape,
            lshape=t1.lshape,
            dtype=result.dtype,
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

    out.larray.copy_(result)
    return out
