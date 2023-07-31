"""
Basic linear algebra operations on distributed ``DNDarray``
"""
import itertools
import numpy as np
import torch
import warnings

from typing import List, Callable, Union, Optional, Tuple

from torch._C import Value

from ..communication import MPI
from .. import arithmetics
from .. import complex_math
from .. import constants
from .. import exponential
from ..dndarray import DNDarray
from .. import factories
from .. import manipulations
from .. import rounding
from .. import sanitation
from .. import statistics
from .. import stride_tricks
from .. import types

__all__ = [
    "cross",
    "det",
    "dot",
    "inv",
    "matmul",
    "matrix_norm",
    "norm",
    "outer",
    "projection",
    "trace",
    "transpose",
    "tril",
    "triu",
    "vdot",
    "vecdot",
    "vector_norm",
]


def cross(
    a: DNDarray, b: DNDarray, axisa: int = -1, axisb: int = -1, axisc: int = -1, axis: int = -1
) -> DNDarray:
    """
    Returns the cross product. 2D vectors will we converted to 3D.

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
    """
    sanitation.sanitize_in(a)
    sanitation.sanitize_in(b)

    if a.device != b.device:
        raise ValueError(f"'a' and 'b' must have the same device type, {a.device} != {b.device}")
    if a.comm != b.comm:  # pragma: no cover
        raise ValueError(f"'a' and 'b' must have the same comm, {a.comm} != {b.comm}")

    a_2d, b_2d = False, False
    a_shape, b_shape = list(a.shape), list(b.shape)

    if axis != -1 or torch.unique(torch.tensor([axisa, axisb, axisc, axis])).numel() == 1:
        axis = stride_tricks.sanitize_axis(a.shape, axis)
        axisa, axisb, axisc = (axis,) * 3
    else:
        axisa = stride_tricks.sanitize_axis(a.shape, axisa)
        axisb = stride_tricks.sanitize_axis(b.shape, axisb)
        axisc = stride_tricks.sanitize_axis(a.shape, axisc)

    if a.split == axisa or b.split == axisb:
        raise ValueError(
            "The computation of the cross product with vectors along the split axis is not supported."
        )

    # all dimensions except axisa, axisb must be broadcastable
    del a_shape[axisa], b_shape[axisb]
    output_shape = stride_tricks.broadcast_shape(a_shape, b_shape)

    # 2d -> 3d vector
    if a.shape[axisa] == 2:
        a_2d = True
        shape = tuple(1 if i == axisa else j for i, j in enumerate(a.shape))
        a = manipulations.concatenate(
            [a, factories.zeros(shape, dtype=a.dtype, device=a.device, comm=a.comm)], axis=axisa
        )
    if b.shape[axisb] == 2:
        b_2d = True
        shape = tuple(1 if i == axisb else j for i, j in enumerate(b.shape))
        b = manipulations.concatenate(
            [b, factories.zeros(shape, dtype=b.dtype, device=b.device)], axis=axisb
        )

    if axisc != axisa:
        a = manipulations.moveaxis(a, axisa, axisc)

    if axisc != axisb:
        b = manipulations.moveaxis(b, axisb, axisc)

    axis = axisc

    # by now split axes must be aligned
    if a.split != b.split:
        raise ValueError(f"'a' and 'b' must have the same split, {a.split} != {b.split}")

    if not (a.is_balanced and b.is_balanced):
        # TODO: replace with sanitize_redistribute after #888 is merged
        b = manipulations.redistribute(b, b.lshape_map, a.lshape_map)

    promoted = torch.promote_types(a.larray.dtype, b.larray.dtype)

    ret = torch.cross(a.larray.type(promoted), b.larray.type(promoted), dim=axis)

    # if both vector axes have dimension 2, return the z-component of the cross product
    if a_2d and b_2d:
        z_slice = [slice(None, None, None)] * ret.ndim
        z_slice[axisc] = -1
        ret = ret[z_slice]
    else:
        output_shape = output_shape[:axis] + (3,) + output_shape[axis:]

    ret = DNDarray(ret, output_shape, types.heat_type_of(ret), a.split, a.device, a.comm, True)
    return ret


def det(a: DNDarray) -> DNDarray:
    """
    Returns the determinant of a square matrix.

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
    >>> a = ht.array([[-2,-1,2],[2,1,4],[-3,3,-1]])
    >>> ht.linalg.det(a)
    DNDarray(54., dtype=ht.float64, device=cpu:0, split=None)
    """
    sanitation.sanitize_in(a)  # pragma: no cover

    if a.ndim < 2:
        raise RuntimeError("DNDarray must be at least two-dimensional.")

    m, n = a.shape[-2:]
    if m != n:
        raise RuntimeError("Last two dimensions of the DNDarray must be square.")

    if types.heat_type_is_exact(a.dtype):
        raise RuntimeError("dtype of DNDarray must be floating-point.")

    # no split in the square matrices
    if not a.is_distributed() or a.split < a.ndim - 2:
        data = torch.linalg.det(a.larray)
        sp = a.split if a.is_distributed() else None
        return DNDarray(
            data,
            a.shape[:-2],
            types.heat_type_of(data),
            split=sp,
            device=a.device,
            comm=a.comm,
            balanced=a.balanced,
        )

    acopy = a.copy()
    acopy = manipulations.reshape(acopy, (-1, m, m), new_split=a.split - a.ndim + 3)
    adet = factories.ones(acopy.shape[0], dtype=a.dtype, device=a.device, comm=a.comm)

    for k in range(adet.shape[0]):
        m = 0
        for i in range(n):
            # partial pivoting
            if np.isclose(acopy[k, i, i].item(), 0):
                abord = True
                for j in range(i + 1, n):
                    if not np.isclose(acopy[k, j, i].item(), 0):
                        if a.split == a.ndim - 2:  # split=0 on square matrix
                            acopy[k, i, :], acopy[k, j, :] = acopy[k, j, :], acopy[k, i, :].copy()
                        else:  # split=1
                            acopy.larray[k, i, :], acopy.larray[k, j, :] = (
                                acopy.larray[k, j, :],
                                acopy.larray[k, i, :].clone(),
                            )
                        abord = False
                        m += 1
                        break
                if abord:
                    adet[k] = 0
                    break

            adet[k] *= acopy[k, i, i]
            z = acopy[k, i + 1 :, i, None].larray / acopy[k, i, i].item()
            acopy[k, i + 1 :, :].larray -= z * acopy[k, i, :].larray

        if m % 2 != 0:
            adet[k] = -adet[k]

    adet = manipulations.reshape(adet, a.shape[:-2])

    return adet


def dot(a: DNDarray, b: DNDarray, out: Optional[DNDarray] = None) -> Union[DNDarray, float]:
    """
    Returns the dot product of two ``DNDarrays``.
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
    """
    if isinstance(a, (float, int)) or isinstance(b, (float, int)) or a.ndim == 0 or b.ndim == 0:
        # 3. If either a or b is 0-D (scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a * b is preferred.
        if out is not None:
            out = a * b
            return out
        return a * b
    elif a.ndim == 1 and b.ndim == 1:
        # 1. If both a and b are 1-D arrays, it is inner product of vectors.
        if a.split is None and b.split is None:
            sl = slice(None)
            asl = bsl = sl
            # st = 0
        else:  # at least one of them is split
            # todo: scale this by the starting index of the vector and do a lloc getitem
            st, _, sl = a.comm.chunk(a.shape, a.split if a.split is not None else b.split)
            asl = sl if a.split is None else slice(sl[0].start - st, sl[0].stop - st)
            bsl = sl if b.split is None else slice(sl[0].start - st, sl[0].stop - st)

        ret = torch.dot(a.lloc[asl], b.lloc[bsl])
        if a.is_distributed() or b.is_distributed():
            a.comm.Allreduce(MPI.IN_PLACE, ret, MPI.SUM)

        if out is not None:
            out = DNDarray(ret, (), types.heat_type_of(ret), None, a.device, a.comm, True)
            return out
        return DNDarray(ret, (), types.heat_type_of(ret), None, a.device, a.comm, True)
    elif a.ndim <= 2 and b.ndim <= 2:
        # 2. If both a and b are 2-D arrays, it is matrix multiplication, but using matmul or a @ b is preferred.
        ret = matmul(a, b)
        if out is not None:
            out.larray = ret.larray
            out._DNDarray__dtype = ret.dtype
            out._DNDarray__split = ret.split
            out._DNDarray__device = ret.device
            out._DNDarray__comm = ret.comm
            return out
        return ret
    else:
        raise NotImplementedError("ht.dot not implemented for N-D dot M-D arrays")


def inv(a: DNDarray) -> DNDarray:
    """
    Computes the multiplicative inverse of a square matrix.

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
    >>> a = ht.array([[1., 2], [2, 3]])
    >>> ht.linalg.inv(a)
    DNDarray([[-3.,  2.],
              [ 2., -1.]], dtype=ht.float32, device=cpu:0, split=None)
    """
    sanitation.sanitize_in(a)  # pragma: no cover

    if a.ndim < 2:
        raise RuntimeError("DNDarray must be at least two-dimensional.")

    m, n = a.shape[-2:]
    if m != n:
        raise RuntimeError("Last two dimensions of the DNDarray must be square.")

    if types.heat_type_is_exact(a.dtype):
        raise RuntimeError("dtype of DNDarray must be floating-point.")

    # no split in the square matrices
    if not a.is_distributed() or a.split < a.ndim - 2:
        data = torch.inverse(a.larray)
        return DNDarray(
            data,
            a.shape,
            types.heat_type_of(data),
            split=a.split,
            device=a.device,
            comm=a.comm,
            balanced=a.balanced,
        )

    acopy = a.copy()
    acopy = manipulations.reshape(acopy, (-1, m, m), new_split=a.split - a.ndim + 3)
    ainv = factories.zeros_like(acopy)
    for i in range(m):
        ainv[:, i, i] = 1

    _, displs = acopy.counts_displs()

    for k in range(ainv.shape[0]):
        rank = 0
        for i in range(n):
            # partial pivoting
            if np.isclose(acopy[k, i, i].item(), 0):
                abord = True
                for j in range(i + 1, n):
                    if not np.isclose(acopy[k, j, i].item(), 0):
                        if a.split == a.ndim - 2:  # split=0 on square matrix
                            ainv[k, i, :], ainv[k, j, :] = ainv[k, j, :], ainv[k, i, :].copy()
                            acopy[k, i, :], acopy[k, j, :] = acopy[k, j, :], acopy[k, i, :].copy()
                        else:  # split=1
                            acopy.larray[k, i, :], acopy.larray[k, j, :] = (
                                acopy.larray[k, j, :],
                                acopy.larray[k, i, :].clone(),
                            )
                            ainv.larray[k, i, :], ainv.larray[k, j, :] = (
                                ainv.larray[k, j, :],
                                ainv.larray[k, i, :].clone(),
                            )
                        abord = False
                        break
                if abord:
                    raise RuntimeError("Inverse does not exist")

            scale = acopy[k, i, i].item()

            # Circumvent an issue with DNDarray setter and getter that caused precision errors
            if a.split == a.ndim - 2:
                if rank < acopy.comm.size - 1 and i >= displs[rank + 1]:
                    rank += 1
                if acopy.comm.rank == rank:
                    ainv.larray[k, i - displs[rank], :] /= scale
                    acopy.larray[k, i - displs[rank], :] /= scale
            else:
                ainv[k, i, :].larray /= scale
                acopy[k, i, :].larray /= scale

            factor = acopy[k, i + 1 :, i, None].larray
            ainv[k, i + 1 :, :].larray -= factor * ainv[k, i, :].larray
            acopy[k, i + 1 :, :].larray -= factor * acopy[k, i, :].larray

        # backwards
        for i in range(n - 1, 0, -1):
            factor = acopy[k, :i, i, None].larray
            ainv[k, :i, :].larray -= factor * ainv[k, i, :].larray
            acopy[k, :i, :].larray -= factor * acopy[k, i, :].larray

    ainv = manipulations.reshape(ainv, a.shape, new_split=a.split)

    return ainv


def matmul(a: DNDarray, b: DNDarray, allow_resplit: bool = False) -> DNDarray:
    """
    Matrix multiplication of two ``DNDarrays``: ``a@b=c`` or ``A@B=c``.
    Returns a tensor with the result of ``a@b``. The split dimension of the returned array is
    typically the split dimension of a. However, if ``a.split=None`` then the the ``c.split`` will be
    set as the split dimension of ``b``. If both are ``None`` then ``c.split`` is also ``None``.

    Parameters
    ----------
    a : DNDarray
        2 dimensional: :math:`L \\times P`
    b : DNDarray
        2 dimensional: :math:`P \\times Q`
    allow_resplit : bool, optional
        Whether to distribute ``a`` in the case that both ``a.split is None`` and ``b.split is None``.
        Default is ``False``. If ``True``, if both are not split then ``a`` will be distributed in-place along axis 0.

    Notes
    -----
    - If ``a`` is a split vector then the returned vector will be of shape (:math:`1xQ`) and will be split in the 1st dimension
    - If ``b`` is a vector and either ``a`` or ``b`` is split, then the returned vector will be of shape (:math:`Lx1`) and will be split in the 0th dimension

    References
    ----------
    [1] R. Gu, et al., "Improving Execution Concurrency of Large-scale Matrix Multiplication on
    Distributed Data-parallel Platforms," IEEE Transactions on Parallel and Distributed Systems,
    vol 28, no. 9. 2017. \n
    [2] S. Ryu and D. Kim, "Parallel Huge Matrix Multiplication on a Cluster with GPGPU
    Accelerators," 2018 IEEE International Parallel and Distributed Processing Symposium
    Workshops (IPDPSW), Vancouver, BC, 2018, pp. 877-882.

    Example
    -------
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
    """
    if a.gshape[-1] != b.gshape[0]:
        raise ValueError(
            f"If the last dimension of a ({a.gshape[-1]}) is not the same size as the second-to-last dimension of b. ({b.gshape[-2]})"
        )

    # determine if a larger type is needed for c
    c_type = types.promote_types(a.dtype, b.dtype)
    gpu_int_flag = False
    if str(a.device)[:3] == "gpu":
        og_type = c_type
        if c_type in [types.uint8, types.int8, types.int16, types.int32]:
            c_type = types.float32
            gpu_int_flag = True
        elif c_type == types.int64:
            c_type = types.float64
            gpu_int_flag = True

    if a.dtype != c_type:
        a = c_type(a, device=a.device)
    if b.dtype != c_type:
        b = c_type(b, device=b.device)

    # early out for single-process setup, torch matmul
    if a.comm.size == 1:
        ret = factories.array(torch.matmul(a.larray, b.larray), device=a.device)
        if gpu_int_flag:
            ret = og_type(ret, device=a.device)
        return ret

    if a.split is None and b.split is None:  # matmul from torch
        if len(a.gshape) < 2 or len(b.gshape) < 2 or not allow_resplit:
            # if either of A or B is a vector
            ret = factories.array(torch.matmul(a.larray, b.larray), device=a.device, comm=a.comm)
            if gpu_int_flag:
                ret = og_type(ret, device=a.device)
            return ret

        a.resplit_(0)
        slice_0 = a.comm.chunk(a.shape, a.split)[2][0]
        hold = a.larray @ b.larray

        c = factories.zeros((a.gshape[-2], b.gshape[1]), dtype=c_type, device=a.device, comm=a.comm)
        c.larray[slice_0.start : slice_0.stop, :] += hold
        c.comm.Allreduce(MPI.IN_PLACE, c, MPI.SUM)
        if gpu_int_flag:
            c = og_type(c, device=a.device)
        return c

    # if they are vectors they need to be expanded to be the proper dimensions
    vector_flag = False  # flag to run squeeze at the end of the function
    if len(a.gshape) < 2 and len(b.gshape) < 2:
        # make both split 0, do a local mm then a sum
        a.resplit_(0)
        b.resplit_(0)
        res = a.larray @ b.larray
        a.comm.Allreduce(MPI.IN_PLACE, res, MPI.SUM)
        ret = factories.array(res, split=None, device=a.device, comm=a.comm)
        if gpu_int_flag:
            ret = og_type(ret, device=a.device)
        return ret
    elif len(a.gshape) < 2:
        a = manipulations.expand_dims(a, axis=0)
        vector_flag = True
    elif len(b.gshape) < 2:
        b = manipulations.expand_dims(b, axis=1)
        vector_flag = True

    split_0_flag = False
    split_1_flag = False
    split_01_flag = False
    split_10_flag = False

    tdev = a.device.torch_device

    if (
        (a.split == 0 and b.split is None) or (a.split is None and b.split == 1)
    ) and not vector_flag:
        split = a.split if a.split is not None else b.split
        split = split if not vector_flag else 0
        c = factories.zeros(
            (a.gshape[-2], b.gshape[1]), split=split, dtype=c_type, device=a.device, comm=a.comm
        )
        c.larray += a.larray @ b.larray

        ret = c if not vector_flag else c.squeeze()
        if gpu_int_flag:
            ret = og_type(ret, device=a.device)
        return ret

    elif a.split == 1 and b.split is None:
        c = torch.zeros((a.gshape[-2], b.gshape[1]), dtype=c_type.torch_type(), device=tdev)

        a_idx = a.comm.chunk(a.shape, a.split)[2]
        c += a.larray @ b.larray[a_idx[1].start : a_idx[1].start + a.lshape[-1], :]
        a.comm.Allreduce(MPI.IN_PLACE, c, MPI.SUM)
        c = c if not vector_flag else c.squeeze()
        ret = factories.array(
            c, split=a.split if b.gshape[1] > 1 else 0, device=a.device, comm=a.comm
        )
        if gpu_int_flag:
            ret = og_type(ret, device=a.device)
        return ret

    elif a.split is None and b.split == 0:
        c = torch.zeros((a.gshape[-2], b.gshape[1]), dtype=c_type.torch_type(), device=tdev)
        b_idx = b.comm.chunk(b.shape, b.split)[2]
        c += a.larray[:, b_idx[0].start : b_idx[0].start + b.lshape[0]] @ b.larray
        b.comm.Allreduce(MPI.IN_PLACE, c, MPI.SUM)
        c = c if not vector_flag else c.squeeze()
        ret = factories.array(
            c, split=b.split if a.gshape[-2] > 1 else 0, device=a.device, comm=a.comm
        )
        if gpu_int_flag:
            ret = og_type(ret, device=a.device)
        return ret

    elif (
        a.split == 0 and b.split is None
    ):  # this case and the one below will only be reaching if one of them is a vector
        c = torch.zeros((a.gshape[-2], b.lshape[1]), dtype=c_type.torch_type(), device=tdev)
        a_idx = a.comm.chunk(a.shape, a.split)[2]
        c[a_idx[0]] += a.larray @ b.larray
        a.comm.Allreduce(MPI.IN_PLACE, c, MPI.SUM)
        c = c if not vector_flag else c.squeeze()
        split = a.split if b.gshape[1] > 1 else 0
        split = split if not vector_flag else 0
        ret = factories.array(c, split=split, device=a.device, comm=a.comm)
        if gpu_int_flag:
            ret = og_type(ret, device=a.device)
        return ret

    elif a.split is None and b.split == 1:
        c = torch.zeros((a.gshape[-2], b.lshape[1]), dtype=c_type.torch_type(), device=tdev)
        c += a.larray @ b.larray
        c = c if not vector_flag else c.squeeze()
        split = b.split if a.gshape[1] > 1 else 0
        split = split if not vector_flag else 0
        ret = factories.array(c, is_split=split, device=a.device, comm=a.comm)
        if gpu_int_flag:
            ret = og_type(ret, device=a.device)
        return ret

    elif a.split == 0 and b.split == 0:
        split_0_flag = True
    elif a.split == 1 and b.split == 1:
        split_1_flag = True
    elif a.split == 0 and b.split == 1:
        split_01_flag = True
    elif a.split == 1 and b.split == 0:
        split_10_flag = True
    else:
        raise NotImplementedError("splits > 1 not implemented")

    # block sizes dont need to be the same. thy just need the same inner dimension (kB)
    kB = 0
    rem_a, rem_b = [0] * 2
    if a.split == len(a.gshape) - 1 and b.split == len(a.gshape) - 2:
        # if the split direction is the last dim in a and the first dim in b
        # the max inner dim (kB) is the min value from the result of the integer division
        # of the last dim of a/world size and the first dim of b/world size
        kB = min([a.gshape[-1] // a.comm.size, b.gshape[0] // b.comm.size])
    elif a.split == len(a.gshape) - 2 and b.split == len(a.gshape) - 1:
        kB = a.gshape[-1]
    elif a.split == len(a.gshape) - 1:
        kB = a.gshape[-1] // a.comm.size
    elif b.split == len(a.gshape) - 2:
        kB = b.gshape[0] // b.comm.size
        kB = min(kB, a.gshape[-1])

    if a.lshape[-1] % kB != 0 or (kB == 1 and a.lshape[-1] != 1):
        rem_a = 1
    if b.lshape[0] % kB != 0 or (kB == 1 and b.lshape[-2] != 1):
        rem_b = 1

    # get the lshape map to determine what needs to be sent where as well as M and N
    # lshape map dims -> {node, a=0, b=1, lshape}
    lshape_map = torch.zeros((a.comm.size, 2, len(a.gshape)), dtype=int, device=tdev)
    lshape_map[a.comm.rank, 0, :] = torch.tensor(a.lshape, device=tdev)
    lshape_map[b.comm.rank, 1, :] = torch.tensor(b.lshape, device=tdev)
    a.comm.Allreduce(MPI.IN_PLACE, lshape_map, MPI.SUM)

    # find mB (first blocking dim for a) and nB (2nd blocking dim for b)
    mB = lshape_map[:, 0, -2].min().item()
    nB = lshape_map[:, 1, -1].min().item()

    # check for remaining dims in the outside dimensions
    rem_a_out, rem_b_out = 0, 0
    if a.lshape[-2] % mB != 0 or (kB == 1 and a.lshape[-2] != 1):
        rem_a_out = 1
    if b.lshape[-1] % nB != 0 or (kB == 1 and b.lshape[-1] != 1):
        rem_b_out = 1

    # get the flags from all processes
    # rem_map dims guide -> {process number, a/b (0/1), True/False (1/0)
    #   if there is a remainder in this dimension
    rem_map = torch.zeros((a.comm.size, 2, 2))
    rem_map[a.comm.rank, 0, :] = torch.tensor((rem_a_out, rem_a), device=tdev)
    rem_map[a.comm.rank, 1, :] = torch.tensor((rem_b, rem_b_out), device=tdev)
    rem_map_comm = a.comm.Iallreduce(MPI.IN_PLACE, rem_map, MPI.SUM)

    # index_map dims guide -> {process number, a=0/b=1, relevent 1st index, 2nd index}
    index_map = torch.zeros((a.comm.size, 2, 2, 2), dtype=int, device=tdev)
    a_idx = a.comm.chunk(a.shape, a.split)[2]
    index_map[a.comm.rank, 0, 0] = torch.tensor((a_idx[0].start, a_idx[0].stop), device=tdev)
    index_map[a.comm.rank, 0, 1] = torch.tensor((a_idx[1].start, a_idx[1].stop), device=tdev)
    b_idx = b.comm.chunk(b.shape, b.split)[2]
    index_map[b.comm.rank, 1, 0] = torch.tensor((b_idx[0].start, b_idx[0].stop), device=tdev)
    index_map[b.comm.rank, 1, 1] = torch.tensor((b_idx[1].start, b_idx[1].stop), device=tdev)

    index_map_comm = a.comm.Iallreduce(MPI.IN_PLACE, index_map, MPI.SUM)

    # for the communication scheme, the output array needs to be created
    c_shape = (a.gshape[-2], b.gshape[1])
    c = factories.zeros(c_shape, split=a.split, dtype=c_type, device=a.device, comm=a.comm)

    # get the index map for c
    c_index_map = factories.zeros((c.comm.size, 2, 2), device=a.device, comm=a.comm)
    c_idx = c.comm.chunk(c.shape, c.split)[2]
    c_index_map[c.comm.rank, 0, :] = (c_idx[0].start, c_idx[0].stop)
    c_index_map[c.comm.rank, 1, :] = (c_idx[1].start, c_idx[1].stop)
    c_wait = c.comm.Iallreduce(MPI.IN_PLACE, c_index_map, MPI.SUM)

    if a.split == 0:
        a_block_map = torch.zeros(
            (a.comm.size, a.shape[-2] // mB // a.comm.size, a.shape[-1] // kB, 2),
            dtype=torch.int,
            device=tdev,
        )
    elif a.split == 1:
        a_block_map = torch.zeros(
            (a.comm.size, a.shape[-2] // mB, a.shape[-1] // kB // a.comm.size, 2),
            dtype=torch.int,
            device=tdev,
        )
    # units-> [process, dim0 block number, dim1 block number, start coord] **indices are local

    # below is to handle the edge case where there is only one element in one dimension of a
    a_d0_1s_flag, a_d1_1s_flag = False, False
    if any(lshape_map[:, 0, :][:, 0] == 1):
        a_d0_1s_flag = True
    if any(lshape_map[:, 0, :][:, 1] == 1):
        a_d1_1s_flag = True

    index_map_comm.Wait()
    for pr in range(a.comm.size):
        start0 = index_map[pr, 0, 0, 0].item()
        stop0 = index_map[pr, 0, 0, 1].item()
        start1 = index_map[pr, 0, 1, 0].item()
        stop1 = index_map[pr, 0, 1, 1].item()

        for dim0 in range(
            (stop0 - start0) // mB // a.comm.size if a_d0_1s_flag else (stop0 - start0) // mB
        ):
            # loop over the number of blocks in the 0th dimension
            for dim1 in range(
                (stop1 - start1) // kB // a.comm.size if a_d1_1s_flag else (stop1 - start1) // kB
            ):
                # loop over the number of blocks in the 1st dimension
                a_block_map[pr, dim0, dim1] = torch.tensor(
                    (dim0 * mB, dim1 * kB), dtype=torch.int, device=tdev
                )
    rem_map_comm.Wait()
    if b.split == 0:
        # the blocks are shifted in the 2nd dimension of A for as many remainders
        # there are between the blocks in the first dim of B
        cnt = 0
        for r in rem_map[:, 1, 0]:
            if r.item():
                cnt += 1
                a_block_map[:, :, cnt:, 1] += 1

    if b.split == 0:
        b_block_map = torch.zeros(
            (b.comm.size, b.shape[-2] // kB // b.comm.size, b.shape[-1] // nB, 2),
            dtype=torch.int,
            device=tdev,
        )
    elif b.split == 1:
        b_block_map = torch.zeros(
            (b.comm.size, b.shape[-2] // kB, b.shape[-1] // nB // b.comm.size, 2),
            dtype=torch.int,
            device=tdev,
        )
    # units-> [process, dim0 block number, dim1 block number, start coord] **indices are local

    # below is to handle the edge case where there is only one element in one dimension of b
    b_d0_1s_flag, b_d1_1s_flag = False, False
    if any(lshape_map[:, 1, :][:, 0] == 1):
        b_d0_1s_flag = True
    if any(lshape_map[:, 1, :][:, 1] == 1):
        b_d1_1s_flag = True

    for pr in range(b.comm.size):
        start0 = index_map[pr, 1, 0, 0].item()
        stop0 = index_map[pr, 1, 0, 1].item()
        start1 = index_map[pr, 1, 1, 0].item()
        stop1 = index_map[pr, 1, 1, 1].item()

        # loop over the number of blocks in the 0th dimension
        for dim0 in range(
            (stop0 - start0) // kB // b.comm.size if b_d0_1s_flag else (stop0 - start0) // kB
        ):
            # loop over the number of blocks in the 1st dimension
            for dim1 in range(
                (stop1 - start1) // nB // b.comm.size if b_d1_1s_flag else (stop1 - start1) // nB
            ):
                b_block_map[pr, dim0, dim1] = torch.tensor(
                    (dim0 * kB, dim1 * nB), dtype=torch.int, device=tdev
                )

    if a.split == 1:
        cnt = 0
        # this loop will push the blocks in B to adjust for the remainders in A
        for r in rem_map[:, 0, 1]:
            if r.item():
                cnt += 1
                b_block_map[:, cnt:, :, 0] += 1

    # work loop: loop over all processes (also will incorporate the remainder calculations)
    c_wait.Wait()

    if split_0_flag:
        # need to send b here and not a
        #   the rows on 'a' are complete, and the columns of 'b' are split
        # locations of the remainders in b
        b_rem_locs0 = torch.nonzero(rem_map[:, 1, 0] == 1, as_tuple=False)
        a_rem_locs0 = torch.nonzero(rem_map[:, 0, 0] == 1, as_tuple=False)
        # remainders for a in the
        a_node_rem_s0 = a.larray[:mB, kB : (kB + 1) * b_rem_locs0.numel() : kB + 1]
        b_rem = torch.empty(
            b_rem_locs0.numel(), b.lshape[-1], dtype=a.dtype.torch_type(), device=tdev
        )

        # this if/elif/else loop is for the handling of
        if a.comm.rank in a_rem_locs0:
            # if A is split in dim0 and the rank has a remainder in this direction
            r = a.larray[-1]
            r_loc = index_map[a.comm.rank, 0, 0, 1] - index_map[a.comm.rank, 0, 0, 0] - 1
        else:
            r = None
            r_loc = None

        req = {}
        b_lp_data = {}
        for pr in range(b.comm.size):
            # ibcast data on node first
            if b.comm.rank == pr:
                b_lp_data[pr] = b.larray.clone()
            else:
                b_lp_data[pr] = torch.zeros(
                    (lshape_map[pr, 1, 0].item(), lshape_map[pr, 1, 1].item()),
                    dtype=b.dtype.torch_type(),
                    device=tdev,
                )

            # sending a to all nodes for b to operate with
            req[pr] = b.comm.Ibcast(b_lp_data[pr], root=pr)

            # receive the data from the last loop and do the calculation with that
            if pr != 0:
                req[pr - 1].Wait()
                # after receiving the last loop's bcast
                __mm_c_block_setter(
                    b_proc=pr - 1,
                    a_proc=a.comm.rank,
                    a_data=a.larray,
                    b_data=b_lp_data[pr - 1],
                    b_block_map=b_block_map,
                    a_block_map=a_block_map,
                    b_split=b.split,
                    a_split=a.split,
                    mB=mB,
                    kB=kB,
                    nB=nB,
                    c=c.larray,
                )

                # check if there is a remainder on b in the previous node
                # this loop is intended to get the remainders of b since it is the one being passed
                if pr - 1 in b_rem_locs0:
                    # takes care of the remainders in b as well as dim0 of a
                    b_rem[pr - 1] = b_lp_data[pr - 1][-1]

                # this loop is to take care of the remainders in dim0 of A
                if a_rem_locs0.nelement() != 0 and r_loc is not None:
                    st = index_map[pr - 1, 1, 0, 0].item()
                    sp = index_map[pr - 1, 1, 0, 1].item()
                    c.larray[r_loc.item(), :] += r[st:sp] @ b_lp_data[pr - 1]
                del b_lp_data[pr - 1]

            # need to wait if its the last loop, also need to collect the remainders
            if pr == b.comm.size - 1:
                req[pr].Wait()
                __mm_c_block_setter(
                    b_proc=pr,
                    a_proc=a.comm.rank,
                    a_data=a.larray,
                    b_data=b_lp_data[pr],
                    b_block_map=b_block_map,
                    a_block_map=a_block_map,
                    b_split=b.split,
                    a_split=a.split,
                    mB=mB,
                    kB=kB,
                    nB=nB,
                    c=c.larray,
                )
                # check if there is a remainder on b on the last node (there shouldnt be)
                if pr in b_rem_locs0:
                    # this is to save the data from B required by the remainders from dim1 of A
                    b_rem[pr] = b_lp_data[pr][-1]

                # this loop is to take care of the remainders in the 0th dimension of A
                if a_rem_locs0.nelement() != 0 and r_loc is not None:
                    st = index_map[pr, 1, 0, 0].item()
                    sp = index_map[pr, 1, 0, 1].item()

                    if split_01_flag:
                        st1 = index_map[pr, 1, 1, 0].item()
                        sp1 = index_map[pr, 1, 1, 1].item()
                        c.larray[r_loc.item(), st1:sp1] += r[st:sp] @ b_lp_data[pr]
                    else:
                        c.larray[r_loc.item(), :] += r[st:sp] @ b_lp_data[pr]

                # set the final blocks on the last loop, then adjust for the
                # the remainders which were collected in b_rem
                if b_rem_locs0.numel():
                    c.larray[: a_node_rem_s0.shape[0]] += a_node_rem_s0 @ b_rem
                del b_lp_data[pr]

        if vector_flag:
            c_loc = c.larray.squeeze()
            if c_loc.nelement() == 1:
                c_loc = torch.tensor(c_loc, device=tdev)

            c = factories.array(c_loc, is_split=0, device=a.device, comm=a.comm)
        if gpu_int_flag:
            c = og_type(c, device=a.device)
        return c

    elif split_1_flag:
        # for this case, a is sent to b
        #   this is because 'b' has complete columns and the rows of 'a' are split
        # locations of the remainders in b
        b_rem_locs1 = torch.nonzero(rem_map[:, 1, 1] == 1, as_tuple=False)
        a_rem_locs1 = torch.nonzero(rem_map[:, 0, 1] == 1, as_tuple=False)
        b_node_rem_s1 = b.larray[kB : (kB + 1) * a_rem_locs1.numel() : kB + 1, :nB]
        # b_node_rem_s1 -> remainders for a in the

        a_rem = torch.empty(
            a.lshape[-2], a_rem_locs1.numel(), dtype=b.dtype.torch_type(), device=tdev
        )
        # this if/elif/else loop is for the handling of
        if b.comm.rank in b_rem_locs1:
            # if b is split in dim1 and the rank has a remainder in this direction
            r = b.larray[:, -1]
            r_loc = index_map[a.comm.rank, 1, 1, 1] - index_map[a.comm.rank, 1, 1, 0] - 1
        else:
            r = None
            r_loc = None
        req = {}
        a_lp_data = {}
        for pr in range(a.comm.size):
            # ibcast data on node first
            if a.comm.rank == pr:
                a_lp_data[pr] = a.larray.clone()
            else:
                a_lp_data[pr] = torch.zeros(
                    (lshape_map[pr, 0, 0].item(), lshape_map[pr, 0, 1].item()),
                    dtype=a.dtype.torch_type(),
                    device=tdev,
                )
            # sending a to all nodes for b to operate with
            req[pr] = a.comm.Ibcast(a_lp_data[pr], root=pr)
            # receive the data from the last loop and do the calculation with that
            if pr != 0:
                # after receiving the last loop's bcast
                req[pr - 1].Wait()
                __mm_c_block_setter(
                    a_proc=pr - 1,
                    b_proc=b.comm.rank,
                    a_data=a_lp_data[pr - 1],
                    b_data=b.larray,
                    b_block_map=b_block_map,
                    a_block_map=a_block_map,
                    b_split=b.split,
                    a_split=a.split,
                    mB=mB,
                    kB=kB,
                    nB=nB,
                    c=c.larray,
                )
                # check if there is a remainder on b in the previous node
                # this loop is intended to get the remainders of b since it is the one being passed
                if pr - 1 in a_rem_locs1:
                    # takes care of the remainders in b as well as dim0 of a
                    a_rem[:, pr - 1] = a_lp_data[pr - 1][:, -1]
                # this loop is to take care of the remainders in dim1 of B
                if b_rem_locs1.nelement() != 0 and r_loc is not None:
                    st = index_map[pr - 1, 0, 1, 0].item()
                    sp = index_map[pr - 1, 0, 1, 1].item()

                    c.larray[:, r_loc.item()] += (a_lp_data[pr - 1] @ r[st:sp, None]).flatten()

                del a_lp_data[pr - 1]

            # need to wait if its the last loop, also need to collect the remainders
            if pr == b.comm.size - 1:
                req[pr].Wait()
                __mm_c_block_setter(
                    a_proc=pr,
                    b_proc=a.comm.rank,
                    a_data=a_lp_data[pr],
                    b_data=b.larray,
                    b_block_map=b_block_map,
                    a_block_map=a_block_map,
                    b_split=b.split,
                    a_split=a.split,
                    mB=mB,
                    kB=kB,
                    nB=nB,
                    c=c.larray,
                )
                # check if there is a remainder on b on the last node (there shouldnt be)
                if pr in a_rem_locs1:
                    # this is to save the data from B required by the remainders from dim1 of A
                    a_rem[:, pr] = a_lp_data[pr][:, -1]
                # this loop is to take care of the remainders in the 0th dimension of A
                if b_rem_locs1.nelement() != 0 and r_loc is not None:
                    st = index_map[pr, 0, 1, 0].item()
                    sp = index_map[pr, 0, 1, 1].item()
                    c.larray[:, r_loc.item()] += (a_lp_data[pr] @ r[st:sp, None]).flatten()
                # set the final blocks on the last loop, then adjust for the the remainders which were collected in b_rem
                if a_rem_locs1.numel():
                    c.larray[:, : b_node_rem_s1.shape[1]] += a_rem @ b_node_rem_s1
                del a_lp_data[pr]
        if vector_flag:
            c = factories.array(c.larray.squeeze(), is_split=0, device=a.device, comm=a.comm)
        if gpu_int_flag:
            c = og_type(c, device=a.device)
        return c

    elif split_01_flag:
        # for this case there are no remainders which need to be taken care of
        req = {}
        b_lp_data = {}
        for pr in range(a.comm.size):
            # ibcast data on node first
            if b.comm.rank == pr:
                b_lp_data[pr] = b.larray.clone()
            else:
                b_lp_data[pr] = torch.empty(
                    (lshape_map[pr, 1, 0].item(), lshape_map[pr, 1, 1].item()),
                    dtype=b.dtype.torch_type(),
                    device=tdev,
                )
            # sending a to all nodes for b to operate with
            req[pr] = b.comm.Ibcast(b_lp_data[pr], root=pr)

            # receive the data from the last loop and do the calculation with that
            if pr != 0:
                req[pr - 1].Wait()
                # after receiving the last loop's bcast
                st0 = index_map[pr - 1, 0, 0, 0].item()
                sp0 = index_map[pr - 1, 0, 0, 1].item() + 1
                st1 = index_map[pr - 1, 1, 1, 0].item()
                sp1 = index_map[pr - 1, 1, 1, 1].item()

                c.larray[: sp0 - st0, st1:sp1] += a.larray @ b_lp_data[pr - 1]

                del b_lp_data[pr - 1]
            if pr == b.comm.size - 1:
                req[pr].Wait()
                st0 = index_map[pr, 0, 0, 0].item()
                sp0 = index_map[pr, 0, 0, 1].item() + 1
                st1 = index_map[pr, 1, 1, 0].item()
                sp1 = index_map[pr, 1, 1, 1].item()
                c.larray[: sp0 - st0, st1:sp1] += a.larray @ b_lp_data[pr]
                del b_lp_data[pr]
        if vector_flag:
            c = factories.array(c.larray.squeeze(), is_split=0, device=a.device, comm=a.comm)
        if gpu_int_flag:
            c = og_type(c, device=a.device)

        return c

    elif split_10_flag:
        # todo: this may create the full matrix on evey process, issue #360
        # for this case, only a sum is needed at the end
        a_rem_locs1 = torch.nonzero(rem_map[:, 0, 1] == 1, as_tuple=False)
        # locations of the remainders in b
        b_rem_locs0 = torch.nonzero(rem_map[:, 1, 0] == 1, as_tuple=False)
        res = torch.zeros((a.gshape[-2], b.gshape[1]), dtype=c_type.torch_type(), device=tdev)
        for i in range(a.lshape[-1] // kB):
            res += a.larray[:mB, i * kB : i * kB + kB] @ b.larray[i * kB : i * kB + kB, :nB]
        if a.comm.rank in a_rem_locs1 and b.comm.rank in b_rem_locs0 and kB > 1:
            # these Nones are used to change the dims if the full process is not covered
            res += a.larray[:, -1, None] @ b.larray[None, -1, :]

        a.comm.Allreduce(MPI.IN_PLACE, res, MPI.SUM)
        split = a.split if b.gshape[1] > 1 else 0
        if vector_flag:
            split = 0
            res = res.squeeze()
        c = factories.array(res, split=split, device=a.device, comm=a.comm)
        if gpu_int_flag:
            c = og_type(c, device=a.device)
        return c


DNDarray.__matmul__ = lambda self, other: matmul(self, other)


def matrix_norm(
    x: DNDarray,
    axis: Optional[Tuple[int, int]] = None,
    keepdims: bool = False,
    ord: Optional[Union[int, str]] = None,
) -> DNDarray:
    """
    Computes the matrix norm of an array.

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
    >>> ht.matrix_norm(ht.array([[1,2],[3,4]]))
    DNDarray([[5.4772]], dtype=ht.float64, device=cpu:0, split=None)
    >>> ht.matrix_norm(ht.array([[1,2],[3,4]]), keepdims=True, ord=-1)
    DNDarray([[4.]], dtype=ht.float64, device=cpu:0, split=None)
    """
    sanitation.sanitize_in(x)

    if x.ndim < 2:
        raise ValueError("Cannot compute a matrix norm of a vector.")

    if axis is None:
        if x.ndim > 2:
            raise ValueError("Cannot infer axis on arrays with more than two dimensions.")
        else:
            axis = (0, 1)

    if (not isinstance(axis, tuple)) or len(axis) != 2:
        raise TypeError("'axis' must be a 2-tuple.")

    row_axis, col_axis = axis

    if ord == 1:
        if col_axis > row_axis and not keepdims:
            col_axis -= 1
        return statistics.max(
            arithmetics.sum(rounding.abs(x), axis=row_axis, keepdims=keepdims),
            axis=col_axis,
            keepdims=keepdims,
        )
    elif ord == -1:
        if col_axis > row_axis and not keepdims:
            col_axis -= 1
        return statistics.min(
            arithmetics.sum(rounding.abs(x), axis=row_axis, keepdims=keepdims),
            axis=col_axis,
            keepdims=keepdims,
        )
    elif ord == 2:
        raise NotImplementedError("The largest singular value can't be computed yet.")
    elif ord == -2:
        raise NotImplementedError("The smallest singular value can't be computed yet.")
    elif ord == constants.inf:
        if row_axis > col_axis and not keepdims:
            row_axis -= 1
        return statistics.max(
            arithmetics.sum(rounding.abs(x), axis=col_axis, keepdims=keepdims),
            axis=row_axis,
            keepdims=keepdims,
        )
    elif ord == -constants.inf:
        if row_axis > col_axis and not keepdims:
            row_axis -= 1
        return statistics.min(
            arithmetics.sum(rounding.abs(x), axis=col_axis, keepdims=keepdims),
            axis=row_axis,
            keepdims=keepdims,
        )
    elif ord in [None, "fro"]:
        return exponential.sqrt(
            arithmetics.sum((complex_math.conj(x) * x).real, axis=axis, keepdims=keepdims)
        )
    elif ord == "nuc":
        raise NotImplementedError("The nuclear norm can't be computed yet.")
    else:
        raise ValueError("Invalid norm order for matrices.")


def norm(
    x: DNDarray,
    axis: Optional[Union[int, Tuple[int, int]]] = None,
    keepdims: bool = False,
    ord: Optional[Union[int, float, str]] = None,
) -> DNDarray:
    """
    Return the vector or matrix norm of an array.

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
    >>> LA.norm(b, ord='fro')
    DNDarray(7.7460, dtype=ht.float32, device=cpu:0, split=None)
    >>> LA.norm(a, float('inf'))
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
    >>> m = ht.arange(8).reshape(2,2,2)
    >>> LA.norm(m, axis=(1,2))
    DNDarray([ 3.7417, 11.2250], dtype=ht.float32, device=cpu:0, split=None)
    >>> LA.norm(m[0, :, :]), LA.norm(m[1, :, :])
    (DNDarray(3.7417, dtype=ht.float32, device=cpu:0, split=None), DNDarray(11.2250, dtype=ht.float32, device=cpu:0, split=None))
    """
    sanitation.sanitize_in(x)

    ndim = x.ndim

    if axis is None:
        if ord is None or (ord == 2 and ndim == 1) or (ord == "fro" and ndim == 2):
            x = x.flatten()
            if types.issubdtype(x.dtype, types.complex):
                sqnorm = dot(x.real, x.real) + dot(x.imag, x.imag)
            else:
                sqnorm = dot(x, x)
            ret = exponential.sqrt(sqnorm)
            if keepdims:
                ret = ret.reshape(ndim * [1])
            return ret
        elif ndim == 2:
            return matrix_norm(x, axis, keepdims, ord)
        else:
            return vector_norm(x, axis, keepdims, ord)

    if isinstance(axis, int) or len(axis) == 1:
        return vector_norm(x, axis, keepdims, ord)
    elif len(axis) == 2:
        return matrix_norm(x, axis, keepdims, ord)
    else:
        raise ValueError("Improper number of dimensions to norm.")


DNDarray.norm: Callable[[DNDarray], float] = lambda self: norm(self)
DNDarray.norm.__doc__ = norm.__doc__


def outer(
    a: DNDarray, b: DNDarray, out: Optional[DNDarray] = None, split: Optional[int] = None
) -> DNDarray:
    """
    Compute the outer product of two 1-D DNDarrays: :math:`out(i, j) = a(i) \\times b(j)`.
    Given two vectors, :math:`a = (a_0, a_1, ..., a_N)` and :math:`b = (b_0, b_1, ..., b_M)`, the outer product is:

    .. math::
        :nowrap:

        \\begin{pmatrix}
           a_0 \\cdot b_0  & a_0 \\cdot b_1 & . & . &  a_0 \\cdot b_M \\\\
           a_1 \\cdot b_0 & a_1 \\cdot b_1 & . & . & a_1 \\cdot b_M \\\\
           . & . & . & . & .   \\\\
           a_N \\cdot b_0 & a_N \\cdot b_1 & . & . & a_N \\cdot b_M
        \\end{pmatrix}

    Parameters
    ----------
    a : DNDarray
        1-dimensional: :math:`N`
        Will be flattened by default if more than 1-D.
    b : DNDarray
        1-dimensional: :math:`M`
        Will be flattened by default if more than 1-D.
    out : DNDarray, optional
          2-dimensional: :math:`N \\times M`
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

    * Sending ``b`` around in a ring results in ``outer`` being split along the rows (``outer.split = 0``).\n

    * Sending ``a`` around in a ring results in ``outer`` being split along the columns (``outer.split = 1``).\n

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
    """
    # sanitize input
    devices = []
    for array in [a, b]:
        sanitation.sanitize_in(array)
        devices.append(array.device)
    if devices.count(devices[0]) == 2:
        device = devices[0]
    else:
        raise RuntimeError(
            f"input arrays on different devices: input 0 on {devices[0]}, input 1 on {devices[1]}"
        )

    # sanitize dimensions
    # TODO implement is_1D in sanitation module #468
    if a.ndim > 1:
        a = manipulations.flatten(a)
    if b.ndim > 1:
        b = manipulations.flatten(b)
    if a.ndim == 0 or b.ndim == 0:
        raise RuntimeError(f"a, b must be 1-D DNDarrays, but were {a.ndim}-D and {b.ndim}-D")

    outer_gshape = (a.gshape[0], b.gshape[0])
    t_a = a.larray
    t_b = b.larray
    t_outer_dtype = torch.promote_types(t_a.dtype, t_b.dtype)
    t_a, t_b = t_a.type(t_outer_dtype), t_b.type(t_outer_dtype)
    outer_dtype = types.canonical_heat_type(t_outer_dtype)

    if out is not None:
        sanitation.sanitize_out(out, outer_gshape, split, device)
        t_out_dtype = out.larray.dtype

    # distributed outer product, dense arrays (TODO: sparse, #384)
    if a.comm.is_distributed() and split is not None or a.split is not None or b.split is not None:
        # MPI coordinates
        rank = a.comm.rank
        size = a.comm.size
        t_outer_slice = 2 * [slice(None, None, None)]

        if a.split is None:
            a.resplit_(axis=0)
            t_a = a.larray.type(t_outer_dtype)
        if b.split is None:
            b.resplit_(axis=0)
            t_b = b.larray.type(t_outer_dtype)
        if split is None:
            # Split semantics: default out.split = a.split
            split = a.split
            if out is not None and out.split is None:
                out.resplit_(axis=split)

        # calculate local slice of outer product
        if split == 0:
            lshape_map = b.create_lshape_map()
            t_outer_shape = (a.lshape[0], b.gshape[0])
            _, _, local_slice = b.comm.chunk(b.gshape, b.split)
            t_outer_slice[1] = local_slice[0]
        elif split == 1:
            lshape_map = a.create_lshape_map()
            t_outer_shape = (a.gshape[0], b.lshape[0])
            _, _, local_slice = a.comm.chunk(a.gshape, a.split)
            t_outer_slice[0] = local_slice[0]
        t_outer = torch.zeros(t_outer_shape, dtype=t_outer_dtype, device=t_a.device)
        if lshape_map[rank] != 0:
            t_outer[t_outer_slice] = torch.einsum("i,j->ij", t_a, t_b)

        # Ring: fill in missing slices of outer product
        # allocate memory for traveling data
        if split == 0:
            t_b_run = torch.empty(lshape_map[0], dtype=t_outer_dtype, device=t_a.device)
        elif split == 1:
            t_a_run = torch.empty(lshape_map[0], dtype=t_outer_dtype, device=t_b.device)

        for p in range(size - 1):
            # prepare for sending
            dest_rank = rank + 1 if rank != size - 1 else 0
            # prepare for receiving
            origin_rank = rank - 1 if rank != 0 else size - 1
            actual_origin = origin_rank - p
            if origin_rank < p:
                actual_origin += size
            # blocking send and recv
            if split == 0:
                b.comm.Send(t_b, dest_rank)
                b.comm.Recv(t_b_run, origin_rank)
                # buffer from actual_origin could be smaller than allocated buffer
                t_b = t_b_run[: lshape_map[actual_origin]]
                _, _, remote_slice = b.comm.chunk(
                    b.gshape, b.split, rank=actual_origin, w_size=size
                )
                t_outer_slice[1] = remote_slice[0]
            elif split == 1:
                a.comm.Send(t_a, dest_rank)
                a.comm.Recv(t_a_run, origin_rank)
                # buffer from actual_origin could be smaller than allocated buffer
                t_a = t_a_run[: lshape_map[actual_origin]]
                _, _, remote_slice = a.comm.chunk(
                    a.gshape, a.split, rank=actual_origin, w_size=size
                )
                t_outer_slice[0] = remote_slice[0]
            t_outer[t_outer_slice] = torch.einsum("i,j->ij", t_a, t_b)
    else:
        # outer product, all local
        t_outer = torch.einsum("i,j->ij", t_a, t_b)
        split = None

    outer = DNDarray(
        t_outer,
        gshape=outer_gshape,
        dtype=outer_dtype,
        split=split,
        device=a.device,
        comm=a.comm,
        balanced=True,
    )

    if out is not None:
        out.larray = outer.larray.type(t_out_dtype)
        return out

    return outer


def projection(a: DNDarray, b: DNDarray) -> DNDarray:
    """
    Projection of vector ``a`` onto vector ``b``

    Parameters
    ----------
    a : DNDarray
        The vector to be projected. Must be a 1D ``DNDarray``
    b : DNDarray
        The vector to project onto. Must be a 1D ``DNDarray``
    """
    if not isinstance(a, DNDarray) or not isinstance(b, DNDarray):
        raise TypeError(f"a, b must be of type ht.DNDarray, but were {type(a)}, {type(b)}")

    if len(a.shape) != 1 or len(b.shape) != 1:
        raise RuntimeError(
            f"a, b must be vectors of length 1, but were {len(a.shape)}, {len(b.shape)}"
        )

    return (dot(a, b) / dot(b, b)) * b


def trace(
    a: DNDarray,
    offset: Optional[int] = 0,
    axis1: Optional[int] = 0,
    axis2: Optional[int] = 1,
    dtype: Optional[types.datatype] = None,
    out: Optional[DNDarray] = None,
) -> Union[DNDarray, float]:
    """

    Return the sum along diagonals of the array

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
    """
    # ----------------------------------------------------------------------------
    # SANITATION
    # ----------------------------------------------------------------------------
    if not isinstance(a, (DNDarray, torch.Tensor, np.ndarray, list, tuple)):
        raise TypeError(
            f"`a` must be a DNDarray, torch.Tensor, np.ndarray, list or tuple, is {type(a)}"
        )
    # cast input `a` to DNDarray
    elif not isinstance(a, DNDarray):
        a = factories.array(a)

    # assure correct dimensionality of input
    if len(a.lshape) < 2:
        raise ValueError(f"`a` must contain at least 2 dimensions, not {len(a.lshape)}")

    # sanitize axis1, axis2
    if not isinstance(axis1, int):
        raise TypeError(f"`axis1` must be integer, not {type(axis1)}")
    if not isinstance(axis2, int):
        raise TypeError(f"`axis2` must be integer, not {type(axis2)}")

    # translate negative to positive indexing (trace axes)
    if axis1 < 0:
        axis1 = axis1 % a.ndim
    if axis2 < 0:
        axis2 = axis2 % a.ndim

    if axis1 == axis2:
        raise ValueError(f"axis1 ({axis1}) and axis2 ({axis2}) cannot be the same.")
    if axis1 >= a.ndim:
        raise ValueError(f"`axis1` ({axis1}) out of bounds for {a.ndim}-dimensional array.")
    if axis2 >= a.ndim:
        raise ValueError(f"`axis2` ({axis2}) out of bounds for {a.ndim}-dimensional array.")

    # sanitize offset
    if not isinstance(offset, int):
        raise TypeError(f"`offset` must be an integer, not {type(offset)}")

    # sanitize dtype
    try:
        if dtype is None:
            dtype = a.dtype
        else:
            dtype = types.canonical_heat_type(dtype)
    except TypeError:  # type cannot be converted to ht.type
        raise ValueError(f"`dtype` must be a datatype or None, not {type(dtype)}")

    # sanitize out
    if out is not None:
        if not isinstance(out, DNDarray):
            raise TypeError(f"`out` must be a ht.DNDarray or None not {type(out)}")
        elif a.ndim == 2:
            raise ValueError(
                "`out` is not applicable if result is a scalar / input `a` is 2-dimensional"
            )

    # ----------------------------------------------------------------------------
    # ALGORITHM
    # ----------------------------------------------------------------------------
    # ---------------------------------------------
    # CASE 2D input (ignore axis1, axis) => scalar
    # ---------------------------------------------
    if a.ndim == 2:
        # CASE 1.1: offset results into an empty array
        if offset <= -a.gshape[0] or offset >= a.gshape[1]:
            sum_along_diagonals_t = torch.tensor(
                0, dtype=dtype.torch_type(), device=a.device.torch_device
            )

        # CASE 1.2: non-zero array, call torch.trace on concerned sub-DNDarray
        else:
            # determine the additional offset created by distribution of `a`
            a_sub = a
            if a.is_distributed():
                offset_split, _, _ = a.comm.chunk(a.gshape, a.split)
                if a.split == 0:
                    offset += offset_split
                # a.split == 1
                else:
                    offset -= offset_split

            # Calculate resulting/concerned sub-array `a_sub`
            if offset > 0:
                offset = min(offset, a_sub.lshape[1])
                a_sub = factories.array(
                    a_sub.larray[:, offset:], device=a_sub.device, comm=a_sub.comm
                )
            elif offset < 0:
                offset = min(-offset, a_sub.lshape[0])
                a_sub = factories.array(
                    a_sub.larray[offset:, :], device=a_sub.device, comm=a_sub.comm
                )

            # calculate trace /partial sum on that sub-array
            if 0 not in a_sub.lshape:
                sum_along_diagonals_t = torch.trace(a_sub.larray)

                # make sure result is of correct dtype
                sum_along_diagonals_t = sum_along_diagonals_t.type(dtype.torch_type())

            # empty array => result = 0
            else:
                sum_along_diagonals_t = torch.tensor(
                    0, dtype=dtype.torch_type(), device=a_sub.device.torch_device
                )

        # sum up all partial sums
        if a.is_distributed():
            a.comm.Allreduce(MPI.IN_PLACE, sum_along_diagonals_t, MPI.SUM)

        # convert resulting 0-d tensor to (python) scalar
        return sum_along_diagonals_t.item()

    # -------------------------------
    # CASE > 2D => DNDArray
    # -------------------------------

    # sanitize axis1, axis2 (make sure axis1 < axis2)
    if axis1 > axis2:
        axis1, axis2 = axis2, axis1
    # ----------------------------------
    # CASE split axis NOT IN trace axes
    # ----------------------------------
    # compute each diagonal sum
    if not (a.is_distributed() and a.split in (axis1, axis2)):
        # extract diagonals
        diag_t = torch.diagonal(a.larray, offset=offset, dim1=axis1, dim2=axis2)

        # sum them up along the last axis (and convert to given dtype)
        last_axis = diag_t.ndim - 1
        sum_along_diagonals_t = torch.sum(diag_t, last_axis, dtype=dtype.torch_type())
    # -----------------------------
    # CASE split axis IN trace axes
    # -----------------------------
    else:
        # combination that would NOT result into array of zeros
        if -offset < a.gshape[axis1] or offset < a.gshape[axis2]:
            # adapt the offset to distribution
            # (to result into required diagonal elements on each process)
            offset_split, _, _ = a.comm.chunk(a.gshape, a.split)

            if a.split == axis1:
                offset += offset_split
            else:  # a.split == axis2
                offset -= offset_split

        diag_t = torch.diagonal(a.larray, offset=offset, dim1=axis1, dim2=axis2)

        # empty diagonal => create an array of zeros for following summation
        if 0 in diag_t.shape:
            res_shape = [1 if i == 0 else i for i in diag_t.shape]
            diag_t = torch.zeros(res_shape, device=a.device.torch_device)

        # create recvbuffer (with correct resulting shape)
        sum_along_diagonals_t = torch.clone(diag_t)
        res_shape = list(sum_along_diagonals_t.shape)
        del res_shape[-1]  # as summed up along the last axis
        sum_along_diagonals_t = torch.reshape(sum_along_diagonals_t, res_shape)

        # Sum up all partial sums (and gather them)
        # in out
        if out is not None:
            result_array = out
        # in a
        else:
            result_array = a

        result_array.comm.Allreduce(MPI.IN_PLACE, sum_along_diagonals_t, MPI.SUM)

        if result_array.split is None:
            split_axis = None
        else:
            last_axis = sum_along_diagonals_t.ndim - 1
            split_axis = result_array.split if result_array.split <= last_axis else last_axis

        sum_along_diagonals = factories.array(
            sum_along_diagonals_t,
            dtype=dtype,
            split=split_axis,
            comm=result_array.comm,
            device=result_array.device,
        )

        if out is not None:
            sanitation.sanitize_out(out, tuple(res_shape), out.split, out.device)
            out.larray = sum_along_diagonals.larray

        return sum_along_diagonals

    if a.is_distributed():
        # (...and a.split not in (axis1, axis2))
        gather_axis = a.split if a.split < axis2 else a.split - 2
        # check if gather_axis is in range of result
        if gather_axis >= sum_along_diagonals_t.ndim:
            gather_axis = sum_along_diagonals_t.ndim - 1

        # Stack all partial results back together along the correct axis
        sum_along_diagonals = factories.array(
            sum_along_diagonals_t, dtype=dtype, is_split=gather_axis, comm=a.comm, device=a.device
        )
    # input not distributed
    else:
        # check if split axis is in range of result
        if a.split is not None and a.split >= sum_along_diagonals_t.ndim:
            gather_axis = sum_along_diagonals_t.ndim - 1
        else:
            gather_axis = a.split

        # convert torch result back to DNDarray
        sum_along_diagonals = factories.array(
            sum_along_diagonals_t, dtype=dtype, split=gather_axis, comm=a.comm, device=a.device
        )

    if out is not None:
        # resplit to guarantee correct results
        if out.split != gather_axis:
            warnings.warn(
                f"Split axis of `out` will be changed from {out.split} to {gather_axis} to "
                f"guarantee correct results."
            )
            out.resplit_(gather_axis)
        # sanitize out
        output_gshape = list(a.gshape)
        del output_gshape[axis1], output_gshape[axis2 - 1]
        sanitation.sanitize_out(out, tuple(output_gshape), gather_axis, out.device)

        # store result
        out.larray = sum_along_diagonals_t
        return out

    return sum_along_diagonals


# inline function
DNDarray.trace: Callable[
    [
        DNDarray,
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[types.datatype],
        Optional[DNDarray],
    ],
    Union[DNDarray, float],
] = lambda self, offset=0, axis1=0, axis2=1, dtype=None, out=None: trace(
    self, offset, axis1, axis2, dtype, out
)
DNDarray.trace.__doc__ = trace.__doc__


@torch.jit.script
def __mm_c_block_setter(
    b_proc: int,
    a_proc: int,
    a_data: torch.Tensor,
    b_data: torch.Tensor,
    b_block_map: torch.Tensor,
    a_block_map: torch.Tensor,
    b_split: int,
    a_split: int,
    mB: int,
    kB: int,
    nB: int,
    c: torch.Tensor,
) -> None:
    """
    Helper function for multiplying elements of A and B (see :func:'matmul <matmul>') and putting the results into the
    correct place in C.

    Parameters
    ----------
    b_proc : int
        process with the data for the data for element b
    a_proc : int
        process with the data for the data for element a
    a_data : torch.Tensor
        data from A
    b_data : torch.Tensor
        data from B
    b_block_map : torch.Tensor
        block map for B
    a_block_map : torch.Tensor
        block map for A
    b_split : int
        split of B
    a_split : int
        split of A
    mB : int
        block size of m
    kB : int
        block size of K
    nB : int
        block size of n
    c : torch.Tensor
        the local data for C
    """
    # # (int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int, int, torch.Tensor) -> None
    shp_b = b_block_map.shape
    offset_a = b_proc * shp_b[1] if b_proc != 0 else 0
    shp_a = a_block_map.shape
    offset_b = a_proc * shp_a[2] if a_proc != 0 else 0
    # offsets are the number of blocks in the multiplication direction on previous nodes
    # print(a_block_map[a_proc].shape[0])
    for bl_1_a in (
        torch.arange(offset_a, offset_a + shp_b[1], dtype=torch.long, device=c.device)
        if b_split == 0
        else torch.arange(a_block_map[a_proc].shape[0], dtype=torch.long, device=c.device)
    ):
        # offset is the number of blocks on the previous node in the direction of multiplication
        for bl_0_a in torch.arange(
            a_block_map[a_proc].shape[0], dtype=torch.long, device=c.device
        ):  # dim0
            for bl_1_b in torch.arange(
                b_block_map[b_proc].shape[1], dtype=torch.long, device=c.device
            ):
                for bl_0_b in (
                    torch.arange(offset_b, offset_b + shp_a[1], dtype=torch.long, device=c.device)
                    if a_split == 1
                    else torch.arange(
                        b_block_map[b_proc].shape[0], dtype=torch.long, device=c.device
                    )
                ):
                    # this offset is the same as before but for b
                    a_start1 = int(a_block_map[a_proc, bl_0_a, bl_1_a, 1].item())
                    a_start0 = int(a_block_map[a_proc, bl_0_a, bl_1_a, 0].item())
                    a_block = a_data[a_start0 : a_start0 + mB, a_start1 : a_start1 + kB]

                    b_start0 = int(b_block_map[b_proc, bl_0_b, bl_1_b, 0].item())
                    b_start1 = int(b_block_map[b_proc, bl_0_b, bl_1_b, 1].item())
                    b_block = b_data[b_start0 : b_start0 + kB, b_start1 : b_start1 + nB]

                    c_start0 = a_start0
                    c_start1 = b_start1
                    c[c_start0 : c_start0 + mB, c_start1 : c_start1 + nB] += a_block @ b_block


def transpose(a: DNDarray, axes: Optional[List[int]] = None) -> DNDarray:
    """
    Permute the dimensions of an array.

    Parameters
    ----------
    a : DNDarray
        Input array.
    axes : None or List[int,...], optional
        By default, reverse the dimensions, otherwise permute the axes according to the values given.
    """
    # type check the input tensor
    sanitation.sanitize_in(a)

    # set default value for axes permutations
    dimensions = len(a.shape)
    if axes is None:
        axes = tuple(reversed(range(dimensions)))
    # if given, sanitize the input
    else:
        try:
            # convert to a list to allow index access
            axes = list(axes)
        except TypeError:
            raise ValueError("axes must be an iterable containing ints")

        if len(axes) != dimensions:
            raise ValueError("axes do not match tensor shape")
        for index, axis in enumerate(axes):
            if not isinstance(axis, int):
                raise TypeError(f"axis must be an integer, but was {type(axis)}")
            elif axis < 0:
                axes[index] = axis + dimensions

    # infer the new split axis, it is the position of the split axis within the new axes permutation
    try:
        transposed_split = axes.index(a.split) if a.split is not None else None
    except ValueError:
        raise ValueError("axes do not match tensor shape")

    # try to rearrange the tensor and return a new transposed variant
    try:
        transposed_data = a.larray.permute(*axes)
        transposed_shape = tuple(a.shape[axis] for axis in axes)

        return DNDarray(
            transposed_data,
            transposed_shape,
            a.dtype,
            transposed_split,
            a.device,
            a.comm,
            a.balanced,
        )
    # if not possible re- raise any torch exception as ValueError
    except (RuntimeError, IndexError) as exception:
        raise ValueError(str(exception))


DNDarray.transpose: Callable[[DNDarray, List[int]], DNDarray] = lambda self, axes=None: transpose(
    self, axes
)
DNDarray.transpose.__doc__ = transpose.__doc__

DNDarray.T = property(transpose)

# statically allocated index slices for non-iterable dimensions in triangular operations
__index_base = (slice(None), slice(None))


def __tri_op(m: DNDarray, k: int, op: Callable) -> DNDarray:
    """
    Generic implementation of triangle operations on a ``DNDarray``. It takes care of input sanitation and non-standard
    broadcast behavior of the 2D triangle-operators.

    Parameters
    ----------
    m : DNDarray
        Input array for which to compute the triangle operator.
    k : int, optional
        Diagonal above which to apply the triangle operator, ``k<0`` is below and ``k>0`` is above.
    op : callable
        Implementation of the triangle operator.

    Raises
    ------
    TypeError
        If the input is not a tensor or the diagonal offset cannot be converted to an integral value.
    """
    sanitation.sanitize_in(m)

    try:
        k = int(k)
    except ValueError:
        raise TypeError(f"Expected k to be integral, but was {type(k)}")

    # chunk the global shape of the tensor to obtain the offset compared to the other ranks
    offset, _, _ = m.comm.chunk(m.shape, m.split)
    dimensions = len(m.shape)

    # manually repeat the input for vectors
    if dimensions == 1:
        triangle = m.larray.expand(m.shape[0], -1)
        if torch.numel(triangle > 0):
            triangle = op(triangle, k - offset)

        return DNDarray(
            triangle,
            (m.shape[0], m.shape[0]),
            m.dtype,
            None if m.split is None else 1,
            m.device,
            m.comm,
            m.balanced,
        )

    original = m.larray
    output = original.clone()

    # modify k to account for tensor splits
    if m.split is not None:
        if m.split + 1 == dimensions - 1:
            k += offset
        elif m.split == dimensions - 1:
            k -= offset

    # in case of two dimensions we can just forward the call to the callable
    if dimensions == 2:
        if torch.numel(original) > 0:
            op(original, k, out=output)
    # more than two dimensions: iterate over all but the last two to realize 2D broadcasting
    else:
        ranges = [range(elements) for elements in m.lshape[:-2]]
        for partial_index in itertools.product(*ranges):
            index = partial_index + __index_base
            op(original[index], k, out=output[index])

    return DNDarray(output, m.shape, m.dtype, m.split, m.device, m.comm, m.balanced)


def tril(m: DNDarray, k: int = 0) -> DNDarray:
    """
    Returns the lower triangular part of the ``DNDarray``.
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
    """
    return __tri_op(m, k, torch.tril)


DNDarray.tril: Callable[[DNDarray, int], DNDarray] = lambda self, k=0: tril(self, k)
DNDarray.tril.__doc__ = tril.__doc__


def triu(m: DNDarray, k: int = 0) -> DNDarray:
    """
    Returns the upper triangular part of the ``DNDarray``.
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
    """
    return __tri_op(m, k, torch.triu)


DNDarray.triu: Callable[[DNDarray, int], DNDarray] = lambda self, k=0: triu(self, k)
DNDarray.triu.__doc__ = triu.__doc__


def vdot(x1: DNDarray, x2: DNDarray) -> DNDarray:
    """
    Computes the dot product of two vectors. Higher-dimensional arrays will be flattened.

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
    >>> a = ht.array([1+1j, 2+2j])
    >>> b = ht.array([1+2j, 3+4j])
    >>> ht.vdot(a,b)
    DNDarray([(17+3j)], dtype=ht.complex64, device=cpu:0, split=None)
    >>> ht.vdot(b,a)
    DNDarray([(17-3j)], dtype=ht.complex64, device=cpu:0, split=None)
    """
    x1 = manipulations.flatten(x1)
    x2 = manipulations.flatten(x2)

    return arithmetics.sum(arithmetics.multiply(complex_math.conjugate(x1), x2))


def vecdot(
    x1: DNDarray, x2: DNDarray, axis: Optional[int] = None, keepdims: Optional[bool] = None
) -> DNDarray:
    """
    Computes the (vector) dot product of two DNDarrays.

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
    >>> ht.vecdot(ht.full((3,3,3),3), ht.ones((3,3)), axis=0)
    DNDarray([[9., 9., 9.],
              [9., 9., 9.],
              [9., 9., 9.]], dtype=ht.float32, device=cpu:0, split=None)
    """
    m = arithmetics.mul(x1, x2)

    if axis is None:
        axis = m.ndim - 1

    return arithmetics.sum(m, axis=axis, keepdims=keepdims)


def vector_norm(
    x: DNDarray,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims=False,
    ord: Optional[Union[int, float]] = None,
) -> DNDarray:
    """
    Computes the vector norm of an array.

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
    >>> ht.vector_norm(ht.array([1,2,3,4]))
    DNDarray([5.4772], dtype=ht.float64, device=cpu:0, split=None)
    >>> ht.vector_norm(ht.array([[1,2],[3,4]]), axis=0, ord=1)
    DNDarray([[4., 6.]], dtype=ht.float64, device=cpu:0, split=None)
    """
    sanitation.sanitize_in(x)

    if axis is None:
        pass
    elif isinstance(axis, tuple):
        if len(axis) > 1:
            raise TypeError("'axis' must be an integer or 1-tuple for vectors.")
    else:
        try:
            axis = int(axis)
        except Exception:
            raise TypeError("'axis' must be an integer or 1-tuple for vectors.")

    if ord == constants.INF:
        return statistics.max(rounding.abs(x), axis=axis, keepdims=keepdims)
    elif ord == -constants.INF:
        return statistics.min(rounding.abs(x), axis=axis, keepdims=keepdims)
    elif ord == 0:
        return arithmetics.sum(x != 0, axis=axis, keepdims=keepdims).astype(types.float)
    elif ord == 1:
        return arithmetics.sum(rounding.abs(x), axis=axis, keepdims=keepdims)
    elif ord is None or ord == 2:
        s = (complex_math.conj(x) * x).real
        return exponential.sqrt(arithmetics.sum(s, axis=axis, keepdims=keepdims))
    elif isinstance(ord, str):
        raise ValueError(f"Norm order {ord} is invalid for vectors")
    else:
        ret = arithmetics.pow(rounding.abs(x), ord)
        ret = arithmetics.sum(ret, axis=axis, keepdims=keepdims)
        ret = arithmetics.pow(ret, 1.0 / ord)
        return ret
