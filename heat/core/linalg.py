import itertools
import torch

from .communication import MPI
from . import dndarray
from . import factories
from . import manipulations
from . import tiling
from . import types

__all__ = ["dot", "matmul", "qr", "qr_old", "transpose", "tril", "triu"]


def larft(v, tau):
    """
    forms the triangular factor T of a real block reflector H of order n, which is defined as a product of k elementary reflectors.
    This is a special case of the LAPACK subroutine of the same name for use with QR. (this function is not implemented in PyTorch

    Parameters
    ----------
    v : PyTorch Tensor, shape: m x nb
        array of the transform vectors, lower triangular, can be calculated by torch.geqrf
    tau : PyTorch Tensor
          array of scalar factors of the elementary reflector H_i, can be calculated by torch.geqrf
    t : optional, PyTorch Tensor
        output

    Returns
    -------
    t : the nb x nb triangular factor of the block reflector v

    References
    ----------
    [0] http://www.netlib.org/lapack/lapack-3.1.1/html/dlarft.f.html
    """
    # V is of size: m, nb
    # T is of size: nb, nb -> k is nb
    # todo: v must be 2D
    m, nb = v.shape
    t = torch.eye(nb)
    # tau = torch.round(tau * 10**8) * 10**-8
    if m == 0:
        return t
    t[0, 0] = tau[0]
    # mask = tau == 0
    for i in range(1, tau.shape[0]):
        if tau[i] == 0:
            t[i:, i] = 0
        else:
            t[0:i, i] = -1 * tau[i] * v[i:m, 0:i].t() @ v[i:m, i]

            t[0:i, i] = t[0:i, 0:i] @ t[0:i, i]
        t[i, i] = tau[i]

    return t


def dot(a, b, out=None):
    """
    Dot product of two arrays. Specifically,

    1. If both a and b are 1-D arrays, it is inner product of vectors.
    2. If both a and b are 2-D arrays, it is matrix multiplication, but using matmul or `a @ b` is preferred.
    3. If either a or b is 0-D (scalar), it is equivalent to multiply and using `ht.multiply(a, b)` or `a * b` is preferred.

    Parameters
    ----------
    a : ht.DNDarray
    b : ht.DNDarray

    Returns
    -------
    ht.DNDarray or single value (float or int)
        Returns the dot product of a and b. If a and b are both scalars or both 1-D arrays then a scalar is returned;
        otherwise an array is returned. If out is given, then it is returned.
    """
    if (
        isinstance(a, (float, int))
        or isinstance(b, (float, int))
        or a.numdims == 0
        or b.numdims == 0
    ):
        # 3. If either a or b is 0-D (scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a * b is preferred.
        if out is not None:
            out = a * b
            return out
        return a * b
    elif a.numdims == 1 and b.numdims == 1:
        # 1. If both a and b are 1-D arrays, it is inner product of vectors.
        if a.split is not None or b.split is not None:
            sl = a.comm.chunk(a.shape, a.split if a.split is not None else b.split)[2]
        ret = torch.dot(a[sl]._DNDarray__array, b[sl]._DNDarray__array)
        if a.is_distributed() or b.is_distributed():
            a.comm.Allreduce(MPI.IN_PLACE, ret, MPI.SUM)

        if out is not None:
            out = ret.item()
            return out
        return ret.item()
    elif a.numdims == 2 and b.numdims == 2:
        # 2. If both a and b are 2-D arrays, it is matrix multiplication, but using matmul or a @ b is preferred.
        ret = matmul(a, b)
        if out is not None:
            if out is not None:
                out._DNDarray__array = ret._DNDarray__array
                out._DNDarray__dtype = ret.dtype
                out._DNDarray__split = ret.split
                out._DNDarray__device = ret.device
                out._DNDarray__comm = ret.comm
            return out
        return ret
    else:
        raise NotImplementedError("ht.dot not implemented for N-D dot M-D arrays")


def matmul(a, b):
    """
    Matrix multiplication of two DNDarrays

    for comment context -> a @ b = c or A @ B = c

    Parameters
    ----------
    a : ht.DNDarray
        2 dimensional: L x P
    b : ht.DNDarray
        2 dimensional: P x Q

    Returns
    -------
    ht.DNDarray
        returns a tensor with the result of a @ b. The split dimension of the returned array is typically the split dimension of a.
        However, if a.split = None then c.split will be set as the split dimension of b. If both are None then c.split is also None.
        ** NOTE ** if a is a split vector then the returned vector will be of shape (1xQ) and will be split in the 1st dimension
        ** NOTE ** if b is a vector and either a or b is split, then the returned vector will be of shape (Lx1) and will be split in the 0th dimension

    References
    ----------
    [1] R. Gu, et al., "Improving Execution Concurrency of Large-scale Matrix Multiplication on Distributed Data-parallel Platforms,"
        IEEE Transactions on Parallel and Distributed Systems, vol 28, no. 9. 2017.
    [2] S. Ryu and D. Kim, "Parallel Huge Matrix Multiplication on a Cluster with GPGPU Accelerators,"
        2018 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), Vancouver, BC, 2018, pp. 877-882.

    Example
    -------
    >>> a = ht.ones((n, m), split=1)
    >>> a[0] = ht.arange(1, m + 1)
    >>> a[:, -1] = ht.arange(1, n + 1)
    (0/1) tensor([[1., 2.],
                  [1., 1.],
                  [1., 1.],
                  [1., 1.],
                  [1., 1.]])
    (1/1) tensor([[3., 1.],
                  [1., 2.],
                  [1., 3.],
                  [1., 4.],
                  [1., 5.]])
    >>> b = ht.ones((j, k), split=0)
    >>> b[0] = ht.arange(1, k + 1)
    >>> b[:, 0] = ht.arange(1, j + 1)
    (0/1) tensor([[1., 2., 3., 4., 5., 6., 7.],
                  [2., 1., 1., 1., 1., 1., 1.]])
    (1/1) tensor([[3., 1., 1., 1., 1., 1., 1.],
                  [4., 1., 1., 1., 1., 1., 1.]])
    >>> linalg.matmul(a, b)
    (0/1) tensor([[18.,  8.,  9., 10.],
                  [14.,  6.,  7.,  8.],
                  [18.,  7.,  8.,  9.],
                  [22.,  8.,  9., 10.],
                  [26.,  9., 10., 11.]])
    (1/1) tensor([[11., 12., 13.],
                  [ 9., 10., 11.],
                  [10., 11., 12.],
                  [11., 12., 13.],
                  [12., 13., 14.]])
    """
    if a.gshape[-1] != b.gshape[0]:
        raise ValueError(
            "If the last dimension of a ({}) is not the same size as the second-to-last dimension of b. ({})".format(
                a.gshape[-1], b.gshape[-2]
            )
        )

    # determine if a larger type is needed for c
    c_type = types.promote_types(a.dtype, b.dtype)
    if a.dtype != c_type:
        a = c_type(a)
    if b.dtype != c_type:
        b = c_type(b)

    if a.split is None and b.split is None:  # matmul from torch
        if len(a.gshape) < 2 or len(b.gshape) < 2:
            # if either of A or B is a vector
            return factories.array(torch.matmul(a._DNDarray__array, b._DNDarray__array))
        else:
            a = a.resplit_(0)
            slice_0 = a.comm.chunk(a.shape, a.split)[2][0]
            hold = a._DNDarray__array @ b._DNDarray__array

            c = factories.zeros((a.gshape[-2], b.gshape[1]), dtype=c_type)
            c._DNDarray__array[slice_0.start : slice_0.stop, :] += hold
            c.comm.Allreduce(MPI.IN_PLACE, c, MPI.SUM)
            return c
    else:
        # if they are vectors they need to be expanded to be the proper dimensions
        vector_flag = False  # flag to run squeeze at the end of the function
        both_vec = 0
        if len(a.gshape) < 2:
            a = manipulations.expand_dims(a, axis=0)
            vector_flag = True
            both_vec += 1
        if len(b.gshape) < 2:
            b = manipulations.expand_dims(b, axis=1)
            vector_flag = True
            both_vec += 1
        both_vec = True if both_vec == 2 else False

        split_0_flag = False
        split_1_flag = False
        split_01_flag = False
        split_10_flag = False

        if (
            (a.split == 0 and b.split is None) or (a.split is None and b.split == 1)
        ) and not vector_flag:
            split = a.split if a.split is not None else b.split
            split = split if not vector_flag else 0
            c = factories.zeros((a.gshape[-2], b.gshape[1]), split=split, dtype=c_type)
            c._DNDarray__array += a._DNDarray__array @ b._DNDarray__array

            return c if not vector_flag else c.squeeze()

        elif b.split is None and a.split == 1:
            c = torch.zeros((a.gshape[-2], b.gshape[1]), dtype=c_type.torch_type())
            a_idx = a.comm.chunk(a.shape, a.split)[2]
            c += (
                a._DNDarray__array
                @ b._DNDarray__array[a_idx[1].start : a_idx[1].start + a.lshape[-1], :]
            )
            a.comm.Allreduce(MPI.IN_PLACE, c, MPI.SUM)
            c = c if not vector_flag else c.squeeze()
            c = factories.array(c, split=a.split if b.gshape[1] > 1 else 0)
            return c

        elif a.split is None and b.split == 0:
            c = torch.zeros((a.gshape[-2], b.gshape[1]), dtype=c_type.torch_type())
            b_idx = b.comm.chunk(b.shape, b.split)[2]
            c += (
                a._DNDarray__array[:, b_idx[0].start : b_idx[0].start + b.lshape[0]]
                @ b._DNDarray__array
            )
            b.comm.Allreduce(MPI.IN_PLACE, c, MPI.SUM)
            c = c if not vector_flag else c.squeeze()
            c = factories.array(c, split=b.split if a.gshape[-2] > 1 else 0)
            return c

        elif (
            a.split == 0 and b.split is None
        ):  # this case and the one below will only be reaching if one of them is a vector
            c = torch.zeros((a.gshape[-2], b.lshape[1]), dtype=c_type.torch_type())
            a_idx = a.comm.chunk(a.shape, a.split)[2]
            c[a_idx[0]] += a._DNDarray__array @ b._DNDarray__array
            a.comm.Allreduce(MPI.IN_PLACE, c, MPI.SUM)
            c = c if not vector_flag else c.squeeze()
            split = a.split if b.gshape[1] > 1 else 0
            split = split if not vector_flag else 0
            c = factories.array(c, split=split)
            return c

        elif a.split is None and b.split == 1:
            c = torch.zeros((a.gshape[-2], b.lshape[1]), dtype=c_type.torch_type())
            c += a._DNDarray__array @ b._DNDarray__array
            c = c if not vector_flag else c.squeeze()
            split = b.split if a.gshape[1] > 1 else 0
            split = split if not vector_flag else 0
            c = factories.array(c, is_split=split)
            return c

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

        # block sizes dont need to be the same. thy just need the same inner dimmension (kB)
        kB = 0
        rem_a, rem_b = [0] * 2
        if (
            a.split == len(a.gshape) - 1 and b.split == len(a.gshape) - 2
        ):  # if the split direction is the last dim in a and the first dim in b
            # the max inner dim (kB) is the min value from the result of the integer division of the last dim of a/world size and the first dim of b/world size
            kB = min([a.gshape[-1] // a.comm.size, b.gshape[0] // b.comm.size])
        elif a.split == len(a.gshape) - 2 and b.split == len(a.gshape) - 1:
            kB = a.gshape[-1]
        elif a.split == len(a.gshape) - 1:
            kB = a.gshape[-1] // a.comm.size
        elif b.split == len(a.gshape) - 2:
            kB = b.gshape[0] // b.comm.size
            kB = kB if kB < a.gshape[-1] else a.gshape[-1]

        if a.lshape[-1] % kB != 0 or (kB == 1 and a.lshape[-1] != 1):
            rem_a = 1
        if b.lshape[0] % kB != 0 or (kB == 1 and b.lshape[-2] != 1):
            rem_b = 1

        # get the lshape map to determine what needs to be sent where as well as M and N
        # lshape map dims -> {node, a=0, b=1, lshape}
        lshape_map = torch.zeros((a.comm.size, 2, len(a.gshape)), dtype=int)
        lshape_map[a.comm.rank, 0, :] = torch.Tensor(a.lshape)
        lshape_map[b.comm.rank, 1, :] = torch.Tensor(b.lshape)
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
        # rem_map dims guide -> {process number, a/b (0/1), True/False (1/0) if there is a remainder in this dimension
        rem_map = torch.zeros((a.comm.size, 2, 2))
        rem_map[a.comm.rank, 0, :] = torch.Tensor((rem_a_out, rem_a))
        rem_map[a.comm.rank, 1, :] = torch.Tensor((rem_b, rem_b_out))
        rem_map_comm = a.comm.Iallreduce(MPI.IN_PLACE, rem_map, MPI.SUM)

        # index_map dims guide -> {process number, a=0/b=1, relevent 1st index, 2nd index}
        index_map = torch.zeros((a.comm.size, 2, 2, 2), dtype=int)
        a_idx = a.comm.chunk(a.shape, a.split)[2]
        index_map[a.comm.rank, 0, 0] = torch.Tensor((a_idx[0].start, a_idx[0].stop))
        index_map[a.comm.rank, 0, 1] = torch.Tensor((a_idx[1].start, a_idx[1].stop))
        b_idx = b.comm.chunk(b.shape, b.split)[2]
        index_map[b.comm.rank, 1, 0] = torch.Tensor((b_idx[0].start, b_idx[0].stop))
        index_map[b.comm.rank, 1, 1] = torch.Tensor((b_idx[1].start, b_idx[1].stop))
        index_map_comm = a.comm.Iallreduce(MPI.IN_PLACE, index_map, MPI.SUM)

        # for the communication scheme, the output array needs to be created
        c_shape = (a.gshape[-2], b.gshape[1])
        c = factories.zeros(c_shape, split=a.split, dtype=c_type)

        # get the index map for c
        c_index_map = factories.zeros((c.comm.size, 2, 2))
        c_idx = c.comm.chunk(c.shape, c.split)[2]
        c_index_map[c.comm.rank, 0, :] = (c_idx[0].start, c_idx[0].stop)
        c_index_map[c.comm.rank, 1, :] = (c_idx[1].start, c_idx[1].stop)
        c_wait = c.comm.Iallreduce(MPI.IN_PLACE, c_index_map, MPI.SUM)

        if a.split == 0:
            a_block_map = torch.zeros(
                (a.comm.size, a.shape[-2] // mB // a.comm.size, a.shape[-1] // kB, 2)
            )
        elif a.split == 1:
            a_block_map = torch.zeros(
                (a.comm.size, a.shape[-2] // mB, a.shape[-1] // kB // a.comm.size, 2)
            )
        # units-> [process, dim0 block number, dim1 block number, start coord] **indices are local

        # below is to handle the edge case where there is only one element in one dimension of a
        a_d0_1s_flag, a_d1_1s_flag = False, False
        if any(lshape_map[:, 0, :][:, 0] == 1):
            a_d0_1s_flag = True
        if any(lshape_map[:, 0, :][:, 1] == 1):
            a_d1_1s_flag = True

        index_map_comm.wait()
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
                    (stop1 - start1) // kB // a.comm.size
                    if a_d1_1s_flag
                    else (stop1 - start1) // kB
                ):
                    # loop over the number of blocks in the 1st dimension
                    a_block_map[pr, dim0, dim1] = torch.tensor(
                        (dim0 * mB, dim1 * kB), dtype=torch.int
                    )
        rem_map_comm.wait()
        if b.split == 0:
            # the blocks are shifted in the 2nd dimension of A for as many remainders there are between the blocks in the first dim of B
            cnt = 0
            for r in rem_map[:, 1, 0]:
                if r.item():
                    cnt += 1
                    a_block_map[:, :, cnt:, 1] += 1

        if b.split == 0:
            b_block_map = torch.zeros(
                (b.comm.size, b.shape[-2] // kB // b.comm.size, b.shape[-1] // nB, 2)
            )
        if b.split == 1:
            b_block_map = torch.zeros(
                (b.comm.size, b.shape[-2] // kB, b.shape[-1] // nB // b.comm.size, 2)
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
                    (stop1 - start1) // nB // b.comm.size
                    if b_d1_1s_flag
                    else (stop1 - start1) // nB
                ):
                    b_block_map[pr, dim0, dim1] = torch.tensor(
                        (dim0 * kB, dim1 * nB), dtype=torch.int
                    )

        if a.split == 1:
            cnt = 0
            # this loop will push the blocks in B to adjust for the remainders in A
            for r in rem_map[:, 0, 1]:
                if r.item():
                    cnt += 1
                    b_block_map[:, cnt:, :, 0] += 1

        # @torch.jit.script
        def c_block_setter(
            b_proc,
            a_proc,
            a_data,
            b_data,
            b_block_map=b_block_map,
            a_block_map=a_block_map,
            b_split=b.split,
            a_split=a.split,
            mB=mB,
            kB=kB,
            nB=nB,
            c=c._DNDarray__array,
        ):
            # type: (int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int, int, torch.Tensor) -> None
            shp_b = b_block_map.shape
            offset_a = b_proc * shp_b[1] if b_proc != 0 else 0
            shp_a = a_block_map.shape
            offset_b = a_proc * shp_a[2] if a_proc != 0 else 0
            # offsets are the number of blocks in the multiplication direction on previous nodes
            # print(a_block_map[a_proc].shape[0])
            for bl_1_a in (
                torch.arange(offset_a, offset_a + shp_b[1], dtype=torch.long)
                if b_split == 0
                else torch.arange(a_block_map[a_proc].shape[0], dtype=torch.long)
            ):
                # this offset is the number of blocks on the previous node in the direction of multiplication
                for bl_0_a in torch.arange(a_block_map[a_proc].shape[0], dtype=torch.long):  # dim0
                    for bl_1_b in torch.arange(b_block_map[b_proc].shape[1], dtype=torch.long):
                        for bl_0_b in (
                            torch.arange(offset_b, offset_b + shp_a[1], dtype=torch.long)
                            if a_split == 1
                            else torch.arange(b_block_map[b_proc].shape[0], dtype=torch.long)
                        ):
                            # this offset is the same as before but for b
                            # print('h', a_block_map[a_proc, bl_0_a, bl_1_a, 1])
                            # print('h', a_block_map[a_proc, bl_0_a, 0, 1])
                            a_start1 = int(a_block_map[a_proc, bl_0_a, bl_1_a, 1].item())
                            a_start0 = int(a_block_map[a_proc, bl_0_a, bl_1_a, 0].item())
                            a_block = a_data[a_start0 : a_start0 + mB, a_start1 : a_start1 + kB]

                            b_start0 = int(b_block_map[b_proc, bl_0_b, bl_1_b, 0].item())
                            b_start1 = int(b_block_map[b_proc, bl_0_b, bl_1_b, 1].item())
                            b_block = b_data[b_start0 : b_start0 + kB, b_start1 : b_start1 + nB]

                            c_start0 = a_start0
                            c_start1 = b_start1
                            # print(c_start0, c_start1, mB, nB)
                            c[c_start0 : c_start0 + mB, c_start1 : c_start1 + nB] += (
                                a_block @ b_block
                            )

        # work loop: loop over all processes (also will incorporate the remainder calculations)
        c_wait.wait()

        if split_0_flag:
            # need to send b here and not a
            # locations of the remainders in b
            b_rem_locs0 = (rem_map[:, 1, 0] == 1).nonzero()
            a_rem_locs0 = (rem_map[:, 0, 0] == 1).nonzero()
            # remainders for a in the
            a_node_rem_s0 = a._DNDarray__array[:mB, kB : (kB + 1) * b_rem_locs0.numel() : kB + 1]
            b_rem = torch.empty(b_rem_locs0.numel(), b.lshape[-1], dtype=a.dtype.torch_type())

            # this if/elif/else loop is for the handling of
            if a.comm.rank in a_rem_locs0:
                # if A is split in dim0 and the rank has a remainder in this direction
                r = a._DNDarray__array[-1]
                r_loc = index_map[a.comm.rank, 0, 0, 1] - index_map[a.comm.rank, 0, 0, 0] - 1
            else:
                r = None
                r_loc = None

            req = {}
            b_lp_data = {}
            for pr in range(b.comm.size):
                # ibcast data on node first
                if b.comm.rank == pr:
                    b_lp_data[pr] = b._DNDarray__array
                else:
                    b_lp_data[pr] = torch.zeros(
                        (lshape_map[pr, 1, 0].item(), lshape_map[pr, 1, 1].item()),
                        dtype=b.dtype.torch_type(),
                    )

                # sending a to all nodes for b to operate with
                req[pr] = b.comm.Ibcast(b_lp_data[pr], root=pr)

                # receive the data from the last loop and do the calculation with that
                if pr != 0:
                    req[pr - 1].wait()
                    # after receiving the last loop's bcast
                    c_block_setter(
                        b_proc=pr - 1,
                        a_proc=a.comm.rank,
                        a_data=a._DNDarray__array,
                        b_data=b_lp_data[pr - 1],
                    )

                    # check if there is a remainder on b in the previous node
                    # this loop is intended to get the remainders of b since it is the one being passed
                    if pr - 1 in b_rem_locs0:
                        # takes care of the remainders in b as well as dim0 of a
                        b_rem[pr - 1] = b_lp_data[pr - 1][-1]

                    # this loop is to take care of the remainders in dim0 of A
                    if a_rem_locs0.nelement() != 0:
                        if r_loc is not None:
                            st = index_map[pr - 1, 1, 0, 0].item()
                            sp = index_map[pr - 1, 1, 0, 1].item()
                            c._DNDarray__array[r_loc.item(), :] += r[st:sp] @ b_lp_data[pr - 1]

                    del b_lp_data[pr - 1]

                # need to wait if its the last loop, also need to collect the remainders
                if pr == b.comm.size - 1:
                    req[pr].wait()
                    c_block_setter(
                        b_proc=pr,
                        a_proc=a.comm.rank,
                        a_data=a._DNDarray__array,
                        b_data=b_lp_data[pr],
                    )
                    # check if there is a remainder on b on the last node (there shouldnt be)
                    if pr in b_rem_locs0:
                        # this is to save the data from B required by the remainders from dim1 of A
                        b_rem[pr] = b_lp_data[pr][-1]

                    # this loop is to take care of the remainders in the 0th dimension of A
                    if a_rem_locs0.nelement() != 0:
                        if r_loc is not None:
                            st = index_map[pr, 1, 0, 0].item()
                            sp = index_map[pr, 1, 0, 1].item()

                            if split_01_flag:
                                st1 = index_map[pr, 1, 1, 0].item()
                                sp1 = index_map[pr, 1, 1, 1].item()
                                c._DNDarray__array[r_loc.item(), st1:sp1] += (
                                    r[st:sp] @ b_lp_data[pr]
                                )
                            else:
                                c._DNDarray__array[r_loc.item(), :] += r[st:sp] @ b_lp_data[pr]

                    # set the final blocks on the last loop, then adjust for the the remainders which were collected in b_rem
                    if b_rem_locs0.numel():
                        c._DNDarray__array[: a_node_rem_s0.shape[0]] += a_node_rem_s0 @ b_rem

                    del b_lp_data[pr]

            if vector_flag:
                c_loc = c._DNDarray__array.squeeze()
                if c_loc.nelement() == 1:
                    c = torch.tensor(c_loc)
                c = factories.array(c_loc, is_split=0)
            return c

        elif split_1_flag:
            # for this case, a is sent to b
            # locations of the remainders in b
            b_rem_locs1 = (rem_map[:, 1, 1] == 1).nonzero()
            a_rem_locs1 = (rem_map[:, 0, 1] == 1).nonzero()
            # remainders for a in the
            b_node_rem_s1 = b._DNDarray__array[kB : (kB + 1) * a_rem_locs1.numel() : kB + 1, :nB]
            a_rem = torch.empty(a.lshape[-2], a_rem_locs1.numel(), dtype=b.dtype.torch_type())

            # this if/elif/else loop is for the handling of
            if b.comm.rank in b_rem_locs1:
                # if b is split in dim1 and the rank has a remainder in this direction
                r = b._DNDarray__array[:, -1]
                r_loc = index_map[a.comm.rank, 1, 1, 1] - index_map[a.comm.rank, 1, 1, 0] - 1
            else:
                r = None
                r_loc = None

            req = {}
            a_lp_data = {}
            for pr in range(a.comm.size):
                # ibcast data on node first
                if a.comm.rank == pr:
                    a_lp_data[pr] = a._DNDarray__array
                else:
                    a_lp_data[pr] = torch.zeros(
                        (lshape_map[pr, 0, 0].item(), lshape_map[pr, 0, 1].item()),
                        dtype=a.dtype.torch_type(),
                    )

                # sending a to all nodes for b to operate with
                req[pr] = a.comm.Ibcast(a_lp_data[pr], root=pr)

                # receive the data from the last loop and do the calculation with that
                if pr != 0:
                    # after receiving the last loop's bcast
                    req[pr - 1].wait()
                    c_block_setter(
                        a_proc=pr - 1,
                        b_proc=b.comm.rank,
                        a_data=a_lp_data[pr - 1],
                        b_data=b._DNDarray__array,
                    )

                    # check if there is a remainder on b in the previous node
                    # this loop is intended to get the remainders of b since it is the one being passed
                    if pr - 1 in a_rem_locs1:
                        # takes care of the remainders in b as well as dim0 of a
                        a_rem[:, pr - 1] = a_lp_data[pr - 1][:, -1]

                    # this loop is to take care of the remainders in dim1 of B
                    if b_rem_locs1.nelement() != 0:
                        if r_loc is not None:
                            st = index_map[pr - 1, 0, 1, 0].item()
                            sp = index_map[pr - 1, 0, 1, 1].item()
                            c._DNDarray__array[:, r_loc.item()] += (
                                a_lp_data[pr - 1] @ r[st:sp, None]
                            ).flatten()

                    del a_lp_data[pr - 1]

                # need to wait if its the last loop, also need to collect the remainders
                if pr == b.comm.size - 1:
                    req[pr].wait()
                    c_block_setter(
                        a_proc=pr,
                        b_proc=a.comm.rank,
                        a_data=a_lp_data[pr],
                        b_data=b._DNDarray__array,
                    )
                    # check if there is a remainder on b on the last node (there shouldnt be)
                    if pr in a_rem_locs1:
                        # this is to save the data from B required by the remainders from dim1 of A
                        a_rem[:, pr] = a_lp_data[pr][:, -1]

                    # this loop is to take care of the remainders in the 0th dimension of A
                    if b_rem_locs1.nelement() != 0:
                        if r_loc is not None:
                            st = index_map[pr, 0, 1, 0].item()
                            sp = index_map[pr, 0, 1, 1].item()
                            c._DNDarray__array[:, r_loc.item()] += (
                                a_lp_data[pr] @ r[st:sp, None]
                            ).flatten()

                    # set the final blocks on the last loop, then adjust for the the remainders which were collected in b_rem
                    if a_rem_locs1.numel():
                        c._DNDarray__array[:, : b_node_rem_s1.shape[1]] += a_rem @ b_node_rem_s1

                    del a_lp_data[pr]
            c = c if not vector_flag else factories.array(c._DNDarray__array.squeeze(), is_split=0)
            return c

        elif split_01_flag:
            # for this case there are no remainders which need to be taken care of
            req = {}
            b_lp_data = {}
            for pr in range(a.comm.size):
                # ibcast data on node first
                if b.comm.rank == pr:
                    b_lp_data[pr] = b._DNDarray__array
                else:
                    b_lp_data[pr] = torch.empty(
                        (lshape_map[pr, 1, 0].item(), lshape_map[pr, 1, 1].item()),
                        dtype=b.dtype.torch_type(),
                    )

                # sending a to all nodes for b to operate with
                req[pr] = b.comm.Ibcast(b_lp_data[pr], root=pr)

                # receive the data from the last loop and do the calculation with that
                if pr != 0:
                    req[pr - 1].wait()
                    # after receiving the last loop's bcast
                    st0 = index_map[pr - 1, 0, 0, 0].item()
                    sp0 = index_map[pr - 1, 0, 0, 1].item() + 1
                    st1 = index_map[pr - 1, 1, 1, 0].item()
                    sp1 = index_map[pr - 1, 1, 1, 1].item()
                    c._DNDarray__array[: sp0 - st0, st1:sp1] += (
                        a._DNDarray__array @ b_lp_data[pr - 1]
                    )

                    del b_lp_data[pr - 1]

                if pr == b.comm.size - 1:
                    req[pr].wait()
                    st0 = index_map[pr, 0, 0, 0].item()
                    sp0 = index_map[pr, 0, 0, 1].item() + 1
                    st1 = index_map[pr, 1, 1, 0].item()
                    sp1 = index_map[pr, 1, 1, 1].item()
                    c._DNDarray__array[: sp0 - st0, st1:sp1] += a._DNDarray__array @ b_lp_data[pr]

                    del b_lp_data[pr]
            c = c if not vector_flag else factories.array(c._DNDarray__array.squeeze(), is_split=0)
            return c

        elif split_10_flag:
            # for this case, only a sum is needed at the end
            a_rem_locs1 = (rem_map[:, 0, 1] == 1).nonzero()
            # locations of the remainders in b
            b_rem_locs0 = (rem_map[:, 1, 0] == 1).nonzero()
            res = torch.zeros((a.gshape[-2], b.gshape[1]), dtype=c_type.torch_type())
            for i in range(a.lshape[-1] // kB):
                res += (
                    a._DNDarray__array[:mB, i * kB : i * kB + kB]
                    @ b._DNDarray__array[i * kB : i * kB + kB, :nB]
                )
            if a.comm.rank in a_rem_locs1 and b.comm.rank in b_rem_locs0:
                # these Nones are used to change the dims
                res += a._DNDarray__array[:, -1, None] @ b._DNDarray__array[None, -1, :]

            a.comm.Allreduce(MPI.IN_PLACE, res, MPI.SUM)
            split = a.split if b.gshape[1] > 1 else 0
            split = split if not vector_flag else 0
            res = res if not vector_flag else res.squeeze()
            c = factories.array(res, split=split if not both_vec else None)
            return c


def qr(a, tiles_per_proc=1, calc_q=True, overwrite_a=False):
    if a.split == 0:
        return __qr_split0(
            a=a, tiles_per_proc=tiles_per_proc, calc_q=calc_q, overwrite_a=overwrite_a
        )
    if a.split == 1:
        return __qr_split1(
            a=a, tiles_per_proc=tiles_per_proc, calc_q=calc_q, overwrite_a=overwrite_a
        )


def __qr_split1(a, tiles_per_proc=1, calc_q=True, overwrite_a=False):
    if not isinstance(a, dndarray.DNDarray):
        raise TypeError("'a' must be a DNDarray")
    if not overwrite_a:
        a = a.copy()
    # do comm when applying the Q to the trailing matrix
    a_tiles = tiling.SquareDiagTiles(
        a, tile_per_proc=tiles_per_proc
    )  # type: tiling.SquareDiagTiles
    tile_columns = a_tiles.tile_columns
    tile_rows = a_tiles.tile_rows
    print(a_tiles.col_indices)

    q0 = factories.eye((a.gshape[0], a.gshape[0]), split=0, dtype=a.dtype, comm=a.comm)
    q0_tiles = tiling.SquareDiagTiles(
        q0, tile_per_proc=tiles_per_proc
    )  # type: tiling.SquareDiagTiles
    q0_tiles.match_tiles(a_tiles)

    # loop over the tile columns
    rank = a.comm.rank
    # q_dict = {}
    cols_on_proc = torch.cumsum(torch.tensor(a_tiles.tile_columns_per_process), dim=0)
    # ==================================== R Calculation ===========================================
    # todo: change tile columns to be the correct number here
    lp_cols = tile_columns if a.gshape[0] > a.gshape[1] else tile_rows
    for dcol in range(lp_cols):  # dcol is the diagonal column
        # loop over each column, need to do the QR for each tile in the column(should be rows)
        # need to get the diagonal process
        not_completed_processes = torch.nonzero(dcol < cols_on_proc).flatten()
        diag_process = not_completed_processes[0].item()
        # get the diagonal tile and do qr on it
        # send q to the other processes
        # 1st qr: only on diagonal tile + apply to the row
        # q_dict[dcol] = {}
        if rank == diag_process:
            q1, r1 = a_tiles[dcol, dcol].qr(some=False)
            a.comm.Bcast(q1.clone(), root=diag_process)
            a_tiles[dcol, dcol] = r1
            # apply q1 to the trailing matrix

            # need to convert dcol to a local index
            loc_col = dcol - sum(a_tiles.tile_columns_per_process[:rank])
            hold = a_tiles.local_get(key=(dcol, slice(loc_col + 1, None)))
            if hold is not None:
                # print(dcol, hold.shape)
                a_tiles.local_set(
                    key=(dcol, slice(loc_col + 1, None)), data=torch.matmul(q1.T, hold)
                )
        elif rank > diag_process:
            # update the trailing matrix and then do q calc
            sz = a_tiles.get_tile_size(key=(dcol, dcol))
            q1 = torch.empty((sz[0], sz[0]))
            loc_col = 0
            a.comm.Bcast(q1, root=diag_process)
            hold = a_tiles.local_get(key=(dcol, slice(0, None)))
            a_tiles.local_set(key=(dcol, slice(0, None)), data=torch.matmul(q1.T, hold))
        else:
            # these processes are already done calculating R, only need to calc Q, but need q1
            sz = a_tiles.get_tile_size(key=(dcol, dcol))
            q1 = torch.empty((sz[0], sz[0]))
            a.comm.Bcast(q1, root=diag_process)

        # ======================== begin q calc for single tile QR ========================
        if calc_q:
            for row in range(q0_tiles.tile_rows_per_process[rank]):
                # q1 is applied to each tile of the column dcol of q0 then written there
                q0_tiles.local_set(
                    key=(row, dcol), data=torch.matmul(q0_tiles.local_get(key=(row, dcol)), q1)
                )
        del q1
        # ======================== end q calc for single tile QR ==========================
        # loop over the rest of the rows, combine the tiles, then apply the result to the rest
        # 2nd step: merged QR on the rows
        diag_tile = a_tiles[dcol, dcol]
        diag_sz = a_tiles.get_tile_size(key=(dcol, dcol))
        # (Q) need to get the start stop of diag tial
        diag_st_sp = a_tiles.get_start_stop(key=(dcol, dcol))
        for row in range(dcol + 1, tile_rows):
            if rank == diag_process:
                # cat diag tile and loop tile
                loop_tile = a_tiles[row, dcol]
                loop_cat = torch.cat((diag_tile, loop_tile), dim=0)
                # qr
                ql, rl = loop_cat.qr(some=False)
                # send ql to all
                a.comm.Bcast(ql.clone(), root=diag_process)
                # save ql
                # q_dict[dcol][row] = [ql, diag_sz]
                # set rs
                a_tiles[dcol, dcol] = rl[: diag_sz[0]]
                a_tiles[row, dcol] = rl[diag_sz[0] :]
                # apply q to rest
                if loc_col + 1 < a_tiles.tile_columns_per_process[rank]:
                    upp = a_tiles.local_get(key=(dcol, slice(loc_col + 1, None)))
                    low = a_tiles.local_get(key=(row, slice(loc_col + 1, None)))
                    hold = torch.matmul(ql.T, torch.cat((upp, low), dim=0))
                    # set upper
                    a_tiles.local_set(key=(dcol, slice(loc_col + 1, None)), data=hold[: diag_sz[0]])
                    # set lower
                    a_tiles.local_set(key=(row, slice(loc_col + 1, None)), data=hold[diag_sz[0] :])
            elif rank > diag_process:
                lp_sz = a_tiles.get_tile_size(key=(row, dcol))
                ql = torch.empty([lp_sz[0] + diag_sz[0]] * 2)
                a.comm.Bcast(ql, root=diag_process)
                upp = a_tiles.local_get(key=(dcol, slice(0, None)))
                low = a_tiles.local_get(key=(row, slice(0, None)))
                hold = torch.matmul(ql.T, torch.cat((upp, low), dim=0))
                # set upper
                a_tiles.local_set(key=(dcol, slice(0, None)), data=hold[: diag_sz[0]])
                # set lower
                a_tiles.local_set(key=(row, slice(0, None)), data=hold[diag_sz[0] :])
            else:
                lp_sz = a_tiles.get_tile_size(key=(row, dcol))
                ql = torch.empty([lp_sz[0] + diag_sz[0]] * 2)
                a.comm.Bcast(ql, root=diag_process)
            # ======================== begin q calc for merged tile QR ==========================
            if calc_q:
                top_left = ql[: diag_sz[0], : diag_sz[0]]
                top_right = ql[: diag_sz[0], diag_sz[0] :]
                bottom_left = ql[diag_sz[0] :, : diag_sz[0]]
                bottom_right = ql[diag_sz[0] :, diag_sz[0] :]
                # two multiplications: one for the left tiles and one for the right
                # left tiles --------------------------------------------------------------------
                # create a column of the same size as the tile row of q0
                qloop_col_left = torch.zeros(a_tiles.get_tile_size(key=(slice(dcol, None), dcol)))
                # top left starts at 0 and goes until diag_sz[1]
                # print(qloop_col_left.shape, top_left.shape, diag_sz)
                qloop_col_left[: diag_sz[0]] = top_left
                # bottom left starts at ? and goes until ? (only care about 0th dim)
                st, sp, _, _ = a_tiles.get_start_stop(key=(row, 0))
                st -= diag_st_sp[0]  # adjust these by subtracting the start index of the diag tile
                sp -= diag_st_sp[0]
                qloop_col_left[st:sp] = bottom_left
                # right tiles --------------------------------------------------------------------
                # create a columns tensor of the size of the tile column of index 'row'
                # print(row, dcol)
                sz = q0_tiles.get_tile_size(key=(row, slice(dcol, None)))
                qloop_col_right = torch.zeros(sz[1], sz[0])
                # top left starts at 0 and goes until diag_sz[1]
                qloop_col_right[: diag_sz[0]] = top_right
                # bottom left starts at ? and goes until ? (only care about 0th dim)
                st, sp, _, _ = a_tiles.get_start_stop(key=(row, 0))
                st -= diag_st_sp[0]  # adjust these by subtracting the start index of the diag tile
                sp -= diag_st_sp[0]
                qloop_col_right[st:sp] = bottom_right

                for qrow in range(q0_tiles.tile_rows_per_process[rank]):
                    # q1 is applied to each tile of the column dcol of q0 then written there
                    q0_row = q0_tiles.local_get(key=(qrow, slice(dcol, None))).clone()
                    q0_tiles.local_set(key=(qrow, dcol), data=torch.matmul(q0_row, qloop_col_left))
                    q0_tiles.local_set(key=(qrow, row), data=torch.matmul(q0_row, qloop_col_right))
            del ql
            # ======================== end q calc for merged tile QR ============================
    if not calc_q:
        return None, a_tiles.arr
    # print('before', q0.lshape)
    q0.balance_()  # q0 might be purposely unbalanced during the tile matching
    # print(q0.lshape)
    return q0, a


def __qr_split0(a, tiles_per_proc=1, calc_q=True, overwrite_a=False):
    """

    :param a:
    tile_rows : tiles per process

    :return:
    """
    # TODO: determine the following:
    # D (number of domains to use) -> must be found with some testing on the HPC machines
    if not isinstance(a, dndarray.DNDarray):
        raise TypeError("'a' must be a DNDarray")
    if not overwrite_a:
        a = a.copy()
    a_tiles = tiling.SquareDiagTiles(
        a, tile_per_proc=tiles_per_proc
    )  # type: tiling.SquareDiagTiles
    tile_columns = a_tiles.tile_columns
    tile_rows_proc = a_tiles.tile_rows_per_process

    q0 = factories.eye((a.gshape[0], a.gshape[0]), split=0, dtype=a.dtype, comm=a.comm)
    q0_tiles = tiling.SquareDiagTiles(
        q0, tile_per_proc=tiles_per_proc
    )  # type: tiling.SquareDiagTiles
    q0_tiles.match_tiles(a_tiles)

    # loop over the tile columns
    rank = a.comm.rank
    q_dict = {}
    q_dict_waits = {}
    proc_tile_start = torch.cumsum(torch.tensor(a_tiles.tile_rows_per_process), dim=0)
    # ==================================== R Calculation ===========================================
    for col in range(tile_columns):  # for each tile column (need to do the last rank separately)
        # for each process need to do local qr
        not_completed_processes = torch.nonzero(col < proc_tile_start).flatten()
        if rank not in not_completed_processes:
            # if the process is done calculating R the break the loop
            break
        diag_process = not_completed_processes[0]
        local_tile_row = 0 if rank != diag_process else col - sum(tile_rows_proc[:rank])

        # only work on the processes which have not computed the final result
        q_dict[col] = {}
        q_dict_waits[col] = {}
        __qr_local_qr_calc(
            col=col, rank=rank, a_tiles=a_tiles, local_tile_row=local_tile_row, q_dict=q_dict
        )

        # todo: modify the merge so that all of the q's dont have to go to the diagonal process,
        #  this will also require a change in the merge function, and thus the q_merge function
        __qr_global_qr_calc(
            a_tiles=a_tiles,
            rank=rank,
            q_dict=q_dict,
            q_dict_waits=q_dict_waits,
            col=col,
            diag_process=diag_process,
            comm=a.comm,
            not_completed_processes=not_completed_processes,
        )

    if not calc_q:
        # return statement if not calculating q
        return None, a
    # ===================================== Q Calculation ==========================================
    for col in range(tile_columns):
        diag_process = (
            torch.nonzero(proc_tile_start > col)[0].item()
            if col != tile_columns
            else a.comm.size - 1
        )

        # local q merge
        __qr_q_waits(q_dict_waits, q_dict, col)
        __qr_apply_local_q_to_q0(
            rank=rank,
            q_dict=q_dict,
            col=col,
            a_tiles=a_tiles,
            q_tiles=q0_tiles,
            comm=a.comm,
            diag=diag_process,
        )

        # global q merge
        global_merge_dict = (
            __qr_global_q_dict_set(
                q_dict_col=q_dict[col], col=col, a_tiles=a_tiles, q_tiles=q0_tiles
            )
            if rank == diag_process
            else {}
        )
        print(global_merge_dict.keys())
        __qr_apply_global_q_to_q0(
            global_merge_dict=global_merge_dict,
            col=col,
            q_tiles=q0_tiles,
            rank=rank,
            comm=a.comm,
            diag=diag_process,
        )
        if col in q_dict.keys():
            del q_dict[col]
    return q0, a


def __qr_q_waits(q_dict_waits, q_dict, col):
    """
    In the q_dict_waits is all of the q values from the global merges

    Parameters
    ----------
    q_dict_waits : Dict
        Dictionary will all of the global merge Qs which were sent to the diagonal process
        these have the form:
        q_dict_waits[col][key] - the waits for one column, key is of the form
            'p0' + str(pr0) + 'p1' + str(pr1)
        q_dict_waits[col][key][0] -> Q
        q_dict_waits[col][key][1] -> comm.Wait
        q_dict_waits[col][key][2] -> upper shape
        q_dict_waits[col][key][3] -> lower shape
        q_dict_waits[col][key][4] -> loop number in the while loop
    q_dict : Dict
        Dictionary of the Q tensors used for the QR of the column 'col'
    col : int, 1 element torch.Tensor
        current column (current iteration of QR)

    Returns
    -------
    None
    Entries are added to q_dict[col] with a new key given by 'loop number' + p0 + # + pr #
    """
    if col not in q_dict_waits.keys():
        return
    for key in q_dict_waits[col].keys():
        new_key = q_dict_waits[col][key][3].wait() + key + "e"
        q_dict_waits[col][key][0][1].wait()
        q_dict[col][new_key] = [
            q_dict_waits[col][key][0][0],
            q_dict_waits[col][key][1].wait(),
            q_dict_waits[col][key][2].wait(),
        ]
    del q_dict_waits[col]


def __qr_apply_local_q_to_q0(rank, q_dict, col, a_tiles, q_tiles, comm, diag):
    """
    Apply the Qs calculated for the local data (split=0) to the current Q0 tensor (Q0 @ q_loc)
    This sends the local merge q to all processes with Ibcast then does the multiplication

    Parameters
    ----------
    rank : int
        processes rank
    q_dict : Dict
        dictionary of the Q tensors used in the R calculation of the column 'col'
    col : int, single element torch.Tensor
        column for which to calculate Q
    a_tiles : tiling.SquareDiagTiles
        tiling object for the tiles of A
    q_tiles : tiling.SquareDiagTiles
        tiling object for the tiles of Q0
    comm : ht.comm object
        communication object to use
    diag : int, single element torch.Tensor
        the rank of the process which contains the diagonal tile

    Returns
    -------
    None
    The function updates elements of q_tiles
    """
    # ----------------------------------------------------------------------------------------------
    # create the local Q then send it to all processes
    if col in q_dict.keys():
        local_merge_q = {
            rank: [
                __qr_local_q_merge(q_dict_local=q_dict[col], col=col, a_tiles=a_tiles, rank=rank),
                None,
            ]
        }
    else:
        local_merge_q = {}

    for r in range(diag, comm.size):
        if r != rank:
            hld = torch.zeros([q_tiles.lshape_map[r][q_tiles.arr.split]] * 2)
        else:
            hld = local_merge_q[r][0].clone()
        wait = comm.Ibcast(hld, root=r)
        local_merge_q[r] = [hld, wait]
    # ----------------------------------------------------------------------------------------------
    # multiply Q0 with local Qs
    for r in range(diag, comm.size):
        if local_merge_q[r][1] is not None:
            # receive q from the other processes
            local_merge_q[r][1].wait()

        sum_row = sum(q_tiles.tile_rows_per_process[:r])
        end_row = q_tiles.tile_rows_per_process[r] + sum_row
        # slice of q_tiles -> [0 -> end local, 1 -> start -> stop]
        # todo: change this for split 1
        q_t_hold = q_tiles.local_get(key=(slice(None), slice(sum_row, end_row)))
        # apply the local merge to q0 then update q0
        q_t_hold = q_t_hold @ local_merge_q[r][0]
        q_tiles.local_set(key=(slice(None), slice(sum_row, end_row)), data=q_t_hold)
        del local_merge_q[r]


def __qr_apply_global_q_to_q0(global_merge_dict, col, q_tiles, rank, comm, diag):
    """

    Parameters
    ----------
    global_merge_dict
    col
    q_tiles
    rank
    comm
    diag

    Returns
    -------

    """

    if rank == diag:
        merge_dict_keys = set(global_merge_dict.keys())
    else:
        merge_dict_keys = None
    merge_dict_keys = comm.bcast(merge_dict_keys, root=diag)

    # send the caqr dictionary to all processes
    for k in merge_dict_keys:
        if rank == diag:
            snd = global_merge_dict[k].clone()
            snd_shape = snd.shape
            comm.bcast(snd_shape, root=diag)
        else:
            snd_shape = None
            snd_shape = comm.bcast(snd_shape, root=diag)
            snd = torch.empty(snd_shape)

        wait = comm.Ibcast(snd, root=diag)
        global_merge_dict[k] = [snd, wait]

    qi_mult = {}
    for c in range(q_tiles.tile_columns):
        # this loop is to slice the merge_dict keys along each column
        qi_mult_set = set([(i, c) for i in range(col, q_tiles.tile_columns)])
        if len(qi_mult_set & merge_dict_keys) != 0:
            qi_mult[c] = list(qi_mult_set & merge_dict_keys)

    # have all the q_merge in one place, now just do the mm with q0
    # get all the keys which are in a column -> this is in qi_mult[column]
    row_inds = q_tiles.row_indices + [q_tiles.arr.gshape[0]]
    for q0_row in range(q_tiles.tile_rows):
        q_copy = q_tiles[q0_row, :].clone() if q_tiles[q0_row, :] is not None else None
        for qi_col in qi_mult.keys():
            # the keys of qi_mult are the columns of qi
            # need to now get the tiles of q0 which they correspond to
            out_sz = q_tiles.get_tile_size(key=(q0_row, qi_col))
            # print(out_sz, q_tiles.row_indices)
            pr = q_tiles.get_tile_proc(key=(q0_row, qi_col))
            if rank == pr and qi_col in qi_mult.keys():
                mult_qi_col = torch.zeros((q_copy.shape[1], out_sz[1]))
                # inner dim must be the same, -> get q0[outer, qi_mult[outer][k][0]]
                # ind is the index of the merge_dict
                # next, get the qi tile
                print(q0_row, qi_col, q_tiles[q0_row, qi_col].shape)
                for ind in qi_mult[qi_col]:
                    if global_merge_dict[ind][1] is not None:
                        global_merge_dict[ind][1].wait()
                    print(ind)
                    mult_qi_col[row_inds[ind[0]] : row_inds[ind[0] + 1], :] = global_merge_dict[
                        ind
                    ][0]
                hold = torch.matmul(q_copy, mult_qi_col)
                q_tiles[q0_row, qi_col] = hold


def __qr_local_qr_calc(col, rank, a_tiles, local_tile_row, q_dict):
    """

    Parameters
    ----------
    col
    rank
    a_tiles
    local_tile_row
    q_dict

    Returns
    -------

    """
    # todo: jit this? cant in current form -> dictionary is use. possible chance for list instead
    # todo: binary merge
    # 1: operate on either 0, column (not diag_pr) or row, column (row is adjusted with column)
    # need to determine which process is operating on a partial -> local_tile_row_index_pr

    # first is the QR of tiles which lay on the same column as the diagonal
    # first qr is on the tile corresponding with local_tile_row, col
    base_tile = a_tiles.local_get((local_tile_row, col))
    # print('begin local tqsr', col, len(torch.where(torch.isnan(base_tile))[0]))
    # print(rank, (local_tile_row, col))
    q1, r1 = base_tile.qr(some=False)

    if len(torch.where(torch.isnan(q1))[0]) != 0:
        print(
            col,
            len(torch.where(torch.isnan(base_tile))[0]),
            len(torch.where(torch.isnan(q1))[0]),
            len(torch.where(torch.isnan(r1))[0]),
        )
        raise ValueError("q has nans!")
    q_dict[col]["l0"] = [q1, base_tile.shape]

    # set the r1 to the tile selected with the key
    a_tiles.local_set(key=(local_tile_row, col), data=r1)
    if col != a_tiles.tile_columns - 1:
        base_rest = a_tiles.local_get((local_tile_row, slice(col + 1, None)))
        loc_rest = torch.matmul(q1.T, base_rest.clone())
        a_tiles.local_set(key=(local_tile_row, slice(col + 1, None)), data=loc_rest)

    for d in range(local_tile_row + 1, a_tiles.tile_rows_per_process[rank]):
        # local merge
        # d is the row it is being merged with
        loop_tile = a_tiles.local_get(key=(d, col))
        q_lp, r_lp = torch.cat((base_tile, loop_tile), dim=0).qr(some=False)
        q_dict[col]["l" + str(d)] = [q_lp, base_tile.shape, loop_tile.shape]

        # set r in both
        a_tiles.local_set(key=(local_tile_row, col), data=r_lp[: base_tile.shape[0]])
        a_tiles.local_set(key=(d, col), data=r_lp[base_tile.shape[0] :])

        # replace the base/loop a_tiles with r, then multiply q_lp by the rest of them
        if col != a_tiles.tile_columns - 1:
            loop_rest = a_tiles.local_get((d, slice(col + 1, None)))
            loop_rest_q = torch.matmul(q_lp.t(), torch.cat((base_rest, loop_rest), dim=0))
            # set rest in both
            a_tiles.local_set(
                key=(local_tile_row, slice(col + 1, None)), data=loop_rest_q[: base_tile.shape[0]]
            )
            a_tiles.local_set(key=(d, slice(col + 1, None)), data=loop_rest_q[base_tile.shape[0] :])


def __qr_local_q_merge(q_dict_local, col, a_tiles, rank):
    """
    function to merge the Qs calculated by the local_tsqr function FOR ONE COLUMN!
    this function is only to be used for merging the TSQR Qs for one column
    these are the single number entries of the q_dict[col]

    :param q_dict:
    :param col:
    :param rank:
    :return:
    """
    lcl_col_shape = a_tiles.local_get(key=(slice(None), col)).shape
    # need to get the start and stop of all tiles on the process
    # for this need to get the rows_per_process[rank] and the row_indices
    row_ind = a_tiles.row_indices
    prev_rows_per_pr = sum(a_tiles.tile_rows_per_process[:rank])
    rows_per_pr = a_tiles.tile_rows_per_process[rank]
    if rows_per_pr == 1:
        # if there is only one tile on the process: return q_dict[col]['0']
        out = q_dict_local["l0"][0].clone().detach()
        del q_dict_local["l0"]
        return out

    # 0. get the local indices of the tiles for the column
    loc_rows = (
        torch.tensor(row_ind[prev_rows_per_pr : prev_rows_per_pr + rows_per_pr])
        - row_ind[prev_rows_per_pr]
    )
    offset = (
        torch.tensor(row_ind[col].item() - row_ind[prev_rows_per_pr].item())
        if row_ind[col].item() > row_ind[prev_rows_per_pr].item()
        else torch.tensor(0)
    )
    # 1: create an eye matrix of the row's zero'th dim^2
    q0 = q_dict_local["l0"]  # [0] -> q, [1] -> shape of a use in q calc (q is square)
    del q_dict_local["l0"]
    base_q = torch.eye(lcl_col_shape[a_tiles.arr.split], dtype=q0[0].dtype)

    # 2: set the top corner of the eye as '0'
    base_q[offset : offset + q0[1][0], offset : offset + q0[1][0]] = q0[0]

    # 3: loop over the single digit keys
    lcl_single_keys = [i for i in q_dict_local.keys() if i[0] == "l"]
    lcl_single_keys.sort(reverse=False)  # this will loop over the keys backwards
    # todo: reversed or not?
    # todo: binary merge loop
    for key in lcl_single_keys:
        # 3a. create an eye matrix and set the 'a_tiles' of that accordingly (split Q into 4)
        lp_q_out = torch.eye(lcl_col_shape[0], dtype=base_q.dtype)
        lp_q = q_dict_local[key]

        # lp_q_sp0 = lp_q[0].shape[0] - lp_q[1][0]
        # 3b. set lp_q_out in the right areas
        lp_row = loc_rows[int(key[1:])]
        # top left
        lp_q_out[offset : offset + lp_q[1][0], offset : offset + lp_q[1][0]] = lp_q[0][
            : lp_q[1][0], : lp_q[1][0]
        ]
        # bottom left
        lp_q_out[lp_row : lp_row + lp_q[2][0], offset : offset + lp_q[1][0]] = lp_q[0][
            lp_q[1][0] :, : lp_q[1][0]
        ]
        # top right
        lp_q_out[offset : offset + lp_q[1][0], lp_row : lp_row + lp_q[2][0]] = lp_q[0][
            : lp_q[1][0], lp_q[1][0] :
        ]
        # bottom right
        lp_q_out[lp_row : lp_row + lp_q[2][0], lp_row : lp_row + lp_q[2][0]] = lp_q[0][
            lp_q[1][0] :, lp_q[1][0] :
        ]
        # # 3c. either add to a dictionary or do the mm with the previous q
        base_q = base_q @ lp_q_out
        del q_dict_local[key]
    return base_q


def __qr_global_q_dict_set(q_dict_col, col, a_tiles, q_tiles, global_merge_dict=None):
    """
    The function takes the orginial Q tensors from the global QR calculation and sets them to
    the keys which corresponds with their tile coordinates in Q. this returns a separate dictionary,
    it does NOT set the values of Q

    Parameters
    ----------
    q_dict_local : Dictionary
    col : int
    a_tiles : tiling.SquarDiagTiles
    q_tiles : tiling.SquarDiagTiles
    rank : int
    split1 : Bool

    :return: none
        q_dict_col will look like [cords] = tensor
    """
    # todo: get they keys from the diag_proc on all processes
    # q is already created, the job of this function is to create the group the merging q's together
    # it takes the merge qs, splits them, then puts them into a new dictionary
    # steps
    proc_tile_start = torch.cumsum(torch.tensor(a_tiles.tile_rows_per_process), dim=0)
    diag_proc = torch.nonzero(proc_tile_start > col)[0].item()
    proc_tile_start = torch.cat((torch.tensor([0]), proc_tile_start[:-1]), dim=0)

    # 1: create caqr dictionary
    # need to have empty lists for all tiles in q
    global_merge_dict = {} if global_merge_dict is None else global_merge_dict

    # intended to be used as [row][column] -> data
    # 2: loop over keys in the dictionary
    merge_list = list(q_dict_col.keys())
    merge_list.sort()
    # todo: possible improvement -> make the keys have the process they are on as well,
    #  then can async get them if they are not on the diagonal process
    for key in merge_list:
        # print(col, key)
        # this loops over all of the Qs for col and creates the dictionary for the pr Q merges
        p0 = key.find("p0")
        p1 = key.find("p1")
        end = key.find("e")
        r0 = int(key[p0 + 2 : p1])
        r1 = int(key[p1 + 2 : end])
        lp_q = q_dict_col[key][0].clone()
        base_size = q_dict_col[key][1]
        # cut the q into 4 bits (end of base array)
        # todo: modify this so that it will get what is needed from the process,
        #  instead of gathering all the qs
        top_left = lp_q[: base_size[0], : base_size[0]]
        top_right = lp_q[: base_size[0], base_size[0] :]
        bottom_left = lp_q[base_size[0] :, : base_size[0]]
        bottom_right = lp_q[base_size[0] :, base_size[0] :]
        # need to adjust the keys to be the global row
        if diag_proc == r0:
            col0 = col
        else:
            col0 = proc_tile_start[r0].item()
        col1 = proc_tile_start[r1].item()
        # col0 and col1 are the columns numbers
        # r0 and r1 are the ranks

        # if there are no elements on that location than set it as the tile
        # 1. get keys of what already has data
        curr_keys = set(global_merge_dict.keys())
        # 2. determine which tiles need to be touched/created
        # these are the keys which are to be multiplied by the q in the current loop
        # for matrix of form: | J  K |
        #                     | L  M |
        mult_keys_00 = [(i, col0) for i in range(q_tiles.tile_columns)]  # (J)
        # (J) -> inds: (i, col0)(col0, col0) -> set at (i, col0)
        mult_keys_01 = [(i, col0) for i in range(q_tiles.tile_columns)]  # (K)
        # (K) -> inds: (i, col0)(col0, col1) -> set at (i, col1)
        mult_keys_10 = [(i, col1) for i in range(q_tiles.tile_columns)]  # (L)
        # (L) -> inds: (i, col1)(col1, col0) -> set at (i, col0)
        mult_keys_11 = [(i, col1) for i in range(q_tiles.tile_columns)]  # (M)
        # (M) -> inds: (i, col1)(col1, col1) -> set at (i, col1)

        # if there are no elements in the mult_keys then set the element to the same place
        s00 = set(mult_keys_00) & curr_keys
        s01 = set(mult_keys_01) & curr_keys
        s10 = set(mult_keys_10) & curr_keys
        s11 = set(mult_keys_11) & curr_keys
        hold_dict = global_merge_dict.copy()

        # (J)
        if not len(s00):
            global_merge_dict[col0, col0] = top_left
        else:  # -> do the mm for all of the mult keys
            for k in s00:
                global_merge_dict[k[0], col0] = hold_dict[k] @ top_left
        # (K)
        if not len(s01):
            # check that we are not overwriting here
            global_merge_dict[col0, col1] = top_right
        else:  # -> do the mm for all of the mult keys
            for k in s01:
                global_merge_dict[k[0], col1] = hold_dict[k] @ top_right
        # (L)
        if not len(s10):
            # check that we are not overwriting here
            global_merge_dict[col1, col0] = bottom_left
        else:  # -> do the mm for all of the mult keys
            for k in s10:
                global_merge_dict[k[0], col0] = hold_dict[k] @ bottom_left
        # (M)
        if not len(s11):
            # check that we are not overwriting here
            global_merge_dict[col1, col1] = bottom_right
        else:  # -> do the mm for all of the mult keys
            for k in s11:
                global_merge_dict[k[0], col1] = hold_dict[k] @ bottom_right
    return global_merge_dict


def __qr_global_qr_calc(
    a_tiles, rank, q_dict, q_dict_waits, col, diag_process, comm, not_completed_processes
):
    """
    Binary tree used to merge the QR's calculated for each process

    Parameters
    ----------
    a_tiles : tiles.SquareDiagTiles

    rank : int
        rank of the process
    q_dict : Dict
        dictionary to save the calculated q matrices to
    q_dict_waits : Dict
        dictionary to save the calculated q matrices to which are
        not calculated on the diagonal process
    col : int
        the current column of the the QR loop
    diag_process : int
        rank of the process which has the tile which lies along the diagonal
    comm : MPICommunication (ht.DNDarray.comm)
        The communicator used. (Intended as the communication of the DNDarray 'a' given to qr)
    not_completed_processes : list
        list of the processes which are still not completed

    Returns
    -------
    None
    """
    rem1 = None
    rem2 = None
    offset = not_completed_processes[0]
    loop_size_remaining = not_completed_processes.clone()
    completed = False if loop_size_remaining.size()[0] > 1 else True
    procs_remaining = loop_size_remaining.size()[0]
    loop = 0
    while not completed:
        if procs_remaining % 2 == 1:
            # if the number of processes active is odd need to save the remainders
            # max possible == 2
            if rem1 is None:
                rem1 = loop_size_remaining[-1]
                loop_size_remaining = loop_size_remaining[:-1]
            elif rem2 is None:
                rem2 = loop_size_remaining[-1]
                loop_size_remaining = loop_size_remaining[:-1]
        # send the data to the corresponding processes
        if rank in loop_size_remaining:
            zipped = zip(
                loop_size_remaining.flatten()[: procs_remaining // 2],
                loop_size_remaining.flatten()[procs_remaining // 2 :],
            )
            for pr in zipped:
                pr0, pr1 = int(pr[0].item()), int(pr[1].item())
                if rank in [pr0, pr1]:
                    q, upper_shape, lower_shape = __qr_merge_tile_rows_qr(
                        pr0=pr0,
                        pr1=pr1,
                        column=col,
                        rank=rank,
                        a_tiles=a_tiles,
                        diag_process=diag_process,
                    )
                    q_dict[col][str(loop) + "p0" + str(pr0) + "p1" + str(pr1) + "e"] = [
                        q,
                        upper_shape,
                        lower_shape,
                    ]

                # send [q, upper_shape, lower_shape] if pr0 / pr1 are not diag_proc
                if diag_process not in (pr0, pr1):
                    # todo: set q_dict on the diag process to have the shapes at least
                    __qr_send_q_to_diag_pr(
                        col=col,
                        pr0=pr0,
                        pr1=pr1,
                        diag_process=diag_process,
                        comm=comm,
                        q=q if rank in [pr0, pr1] else None,
                        upper_shape=upper_shape if rank in [pr0, pr1] else None,
                        lower_shape=lower_shape if rank in [pr0, pr1] else None,
                        q_dict_waits=q_dict_waits,
                        key=str(loop),
                    )

        loop_size_remaining = loop_size_remaining[: -1 * (procs_remaining // 2)]
        procs_remaining = loop_size_remaining.size()[0]

        if rem1 is not None and rem2 is not None:
            # combine rem1 and rem2 in the same way as the other nodes,
            # then save the results in rem1 to be used later
            if rank in [rem1, rem2]:
                q, upper_shape, lower_shape = __qr_merge_tile_rows_qr(
                    pr0=rem1,
                    pr1=rem2,
                    column=col,
                    rank=rank,
                    a_tiles=a_tiles,
                    diag_process=diag_process,
                )
                q_dict[col][str(loop) + "p0" + str(int(rem1)) + "p1" + str(int(rem2)) + "e"] = [
                    q,
                    upper_shape,
                    lower_shape,
                ]
            if diag_process not in (rem1, rem2):
                rem1, rem2 = int(rem1), int(rem2)
                __qr_send_q_to_diag_pr(
                    col=col,
                    pr0=rem1,
                    pr1=rem2,
                    diag_process=diag_process,
                    comm=comm,
                    q=q if rank in [rem1, rem2] else None,
                    upper_shape=upper_shape if rank in [rem1, rem2] else None,
                    lower_shape=lower_shape if rank in [rem1, rem2] else None,
                    q_dict_waits=q_dict_waits,
                    key=str(loop),
                )
            rem1 = rem2
            rem2 = None

        loop += 1
        if rem1 is not None and rem2 is None and procs_remaining == 1:
            # combine rem1 with process 0 (offset) and set completed to True
            # this should be the last thing that happens
            if rank in [offset, rem1]:
                q, upper_shape, lower_shape = __qr_merge_tile_rows_qr(
                    pr0=offset,
                    pr1=rem1,
                    column=col,
                    rank=rank,
                    a_tiles=a_tiles,
                    diag_process=diag_process,
                )
                q_dict[col][str(loop) + "p0" + str(int(offset)) + "p1" + str(int(rem1)) + "e"] = [
                    q,
                    upper_shape,
                    lower_shape,
                ]
            if diag_process not in (offset, rem1):
                offset, rem1 = int(offset), int(rem1)
                __qr_send_q_to_diag_pr(
                    col=col,
                    pr0=offset,
                    pr1=rem1,
                    diag_process=diag_process,
                    comm=comm,
                    q=q if rank in [offset, rem1] else None,
                    upper_shape=upper_shape if rank in [offset, rem1] else None,
                    lower_shape=lower_shape if rank in [offset, rem1] else None,
                    q_dict_waits=q_dict_waits,
                    key=str(loop),
                )
            rem1 = None

        completed = True if procs_remaining == 1 and rem1 is None and rem2 is None else False


def __qr_send_q_to_diag_pr(
    col, pr0, pr1, diag_process, comm, q, upper_shape, lower_shape, q_dict_waits, key
):
    """
    This function is to send the merged Q to the diagonal process.
    This is needed for the Q calculation when two processes are merged and neither is the diagonal
    process

    Parameters
    ----------
    col : int
        The current column used in the parent QR loop
    pr0, pr1 : int, int
        Rank of processes 0 and 1. These are the processes used in the calculation of q
    diag_process : int
        The rank of the process which has the tile along the diagonal for the given column
    comm : MPICommunication (ht.DNDarray.comm)
        The communicator used. (Intended as the communication of the DNDarray 'a' given to qr)
    q : torch.Tensor
        The q as calculed for the merge of the a_tiles
    upper_shape : torch.Shape
        The shape of the upper tile used in the merge qr operation (from process pr0)
    lower_shape : torch.Shape
        The shape of the lower tile used in the merge qr operation (from process pr1)
    q_dict_waits : Dict
        Dictionary used in the collection of the Qs which are sent to the diagonal process

    Returns
    -------
    None
    """
    # this is to send the merged q to the diagonal process for the forming of q
    base_tag = "1" + str(pr1.item() if isinstance(pr1, torch.Tensor) else pr1)
    if comm.rank == pr1:
        comm.send(tuple(q.shape), dest=diag_process, tag=int(base_tag + "1"))
        comm.Isend(q, dest=diag_process, tag=int(base_tag + "12"))
        comm.isend(upper_shape, dest=diag_process, tag=int(base_tag + "123"))
        comm.isend(lower_shape, dest=diag_process, tag=int(base_tag + "1234"))
        comm.isend(key, dest=diag_process, tag=int(base_tag + "12345"))
    if comm.rank == diag_process:
        # q_dict_waits now looks like a
        q_sh = comm.recv(source=pr1, tag=int(base_tag + "1"))
        q_wait = torch.zeros(q_sh)
        q_dict_waits[col]["p0" + str(pr0) + "p1" + str(pr1)] = []
        q_dict_waits[col]["p0" + str(pr0) + "p1" + str(pr1)].append(
            [q_wait, comm.Irecv(q_wait, source=pr1, tag=int(base_tag + "12"))]
        )
        q_dict_waits[col]["p0" + str(pr0) + "p1" + str(pr1)].append(
            comm.irecv(source=pr1, tag=int(base_tag + "123"))
        )
        q_dict_waits[col]["p0" + str(pr0) + "p1" + str(pr1)].append(
            comm.irecv(source=pr1, tag=int(base_tag + "1234"))
        )
        q_dict_waits[col]["p0" + str(pr0) + "p1" + str(pr1)].append(
            comm.irecv(source=pr1, tag=int(base_tag + "12345"))
        )


def __qr_merge_tile_rows_qr(pr0, pr1, column, rank, a_tiles, diag_process):
    """
    Merge two tile rows, take their QR, and apply it to the trailing process
    this will modify A
    This fuction should only be run on processes pr0 and pr1

    Parameters
    ----------
    pr0, pr1 : int, int
        Process ranks of the processes to be used
    column : int
        the current process of the QR calculation
    rank : int
        the rank of the process
    a_tiles : ht.tiles.SquareDiagTiles
        tiling object used for getting/setting the tiles required
    diag_process : int
        The rank of the process which has the tile along the diagonal for the given column

    Returns
    -------
    list : [q of the merged tiles, shape of the tile on pr0, shape of the tile on pr1]
    """
    pr0 = pr0.item() if isinstance(pr0, torch.Tensor) else pr0
    pr1 = pr1.item() if isinstance(pr1, torch.Tensor) else pr1
    comm = a_tiles.arr.comm
    upper_row = sum(a_tiles.tile_rows_per_process[:pr0]) if pr0 != diag_process else column
    lower_row = sum(a_tiles.tile_rows_per_process[:pr1]) if pr1 != diag_process else column

    upper_size = a_tiles.get_tile_size((upper_row, column))
    lower_size = a_tiles.get_tile_size((lower_row, column))

    if rank == pr0:
        upper = a_tiles[upper_row, column]
        upper_rest = a_tiles[upper_row, column + 1 :]
        comm.Send(upper.clone(), dest=pr1, tag=986)
        lower = torch.empty(lower_size)
        comm.Recv(lower, source=pr1, tag=4363)
    if rank == pr1:
        lower = a_tiles[lower_row, column]
        upper = torch.empty(upper_size)
        comm.Recv(upper, source=pr0, tag=986)
        comm.Send(lower.clone(), dest=pr0, tag=4363)

    q_merge, r = torch.cat((upper, lower), dim=0).qr(some=False)

    a_tiles[upper_row, column] = r[: upper.shape[0]]
    a_tiles[lower_row, column] = r[upper.shape[0] :]

    if column < a_tiles.tile_columns - 1:
        upper_rest_size = a_tiles.get_tile_size((upper_row, slice(column + 1, None)))
        lower_rest_size = a_tiles.get_tile_size((lower_row, slice(column + 1, None)))

        if rank == pr0:
            upper_rest = a_tiles[upper_row, column + 1 :]
            lower_rest = torch.empty(lower_rest_size)
            comm.Send(upper_rest.clone(), dest=pr1, tag=98654)
            comm.Recv(lower_rest, source=pr1, tag=436364)

        if rank == pr1:
            upper_rest = torch.empty(upper_rest_size)
            lower_rest = a_tiles[lower_row, column + 1 :]
            comm.Recv(upper_rest, source=pr0, tag=98654)
            comm.Send(lower_rest.clone(), dest=pr0, tag=436364)

        cat_tensor = torch.cat((upper_rest, lower_rest), dim=0)
        new_rest = torch.matmul(q_merge.t(), cat_tensor)
        # the data for upper rest is a slice of the new_rest, need to slice only the 0th dim
        a_tiles[upper_row, column + 1 :] = new_rest[: upper_rest.shape[0]].clone()
        # set the lower rest
        a_tiles[lower_row, column + 1 :] = new_rest[upper_rest.shape[0] :].clone()
    return q_merge, upper.shape, lower.shape


def qr_old(a, copy=True, return_q=True, output=None):
    """
    Compute the qr factorization of a matrix.
    Factor the matrix a as qr, where q is orthonormal and r is upper-triangular.

    :param a:
    :param output:
    :return:
    NOTE : to get it so that the input is in the proper shape (split=1), take the transpose if q=0
           this means that A.T = QR and A = Q"R" where Q" = Q (orthogonal) and R" = Q.T @ R.T @ Q.T
           ...it is unlikely that this works.... need to verify and change if necessary

    References
    ----------
    [0] Jaeyoung Choi, Jack J. Dongarra, L. Susan Ostrouchov, Antoine P. Petitet, David W. Walker,
        and R. Clint Whaley, “Design and Implementation of the ScaLAPACK LU, QR, and Cholesky
        Factorization Routines,” Scientific Programming, vol. 5, no. 3, pp. 173-184, 1996.
        https://doi.org/10.1155/1996/483083.
    [1] Gene H. Golub and Charles F. Van Loan. 1996. Matrix Computations (3rd Ed.).
    Johns Hopkins University Press, Baltimore, MD, USA.
    """
    if not isinstance(a, dndarray.DNDarray):
        raise TypeError("'a' must be a DNDarray")

    if copy:
        a = a.copy()

    if not a.is_distributed():
        # local op
        q, r = a._DNDarray__array.qr()
        return (
            factories.array(q, dtype=a.dtype, split=a.split, comm=a.comm, device=a.device),
            factories.array(r, dtype=a.dtype, split=a.split, comm=a.comm, device=a.device),
        )

    lshape_map = factories.zeros((a.comm.size, 2), dtype=int)
    lshape_map[a.comm.rank, :] = torch.Tensor(a.lshape)
    lshap_wait = a.comm.Iallreduce(MPI.IN_PLACE, lshape_map, MPI.SUM)

    if a.split == 0:
        # generate the 'chunk map' if a.split = 0
        # this is used to find start and stop points while gathering the data before running geqrf()
        chunk_map = torch.zeros((a.comm.size, 2), dtype=int)
        block_indexes = a.comm.chunk(a.gshape, 1)[1]
        chunk_map[a.comm.rank, :] = torch.Tensor(block_indexes)
        a.comm.Allreduce(MPI.IN_PLACE, chunk_map, MPI.SUM)
        chunk_map[..., 1] = chunk_map[..., 1].cumsum(dim=0)

    rank = a.comm.rank
    # r_out is only used if only q is required
    r_out = torch.zeros(a.lshape, dtype=a.dtype.torch_type())
    # need to keep track of the previous columns to adjust for the amount to slice off the top in the 0th dimension
    prev_cols = 0
    # initialize q as the unity matrix with the shape of a
    q_old = factories.eye(a.gshape[0], split=a.split, comm=a.comm, device=a.device, dtype=a.dtype)
    for pr in range(a.comm.size):
        equal = True if rank == pr else False
        greater = True if rank > pr else False

        # a_block is what will be operated on by geqrf()
        if a.split == 0:
            # gather data from all of the other processes
            st_dim1 = chunk_map[pr - 1, 1].item() if pr != 0 else 0
            sp_dim1 = chunk_map[pr, 1].item()
            a_block = a.comm.gather(a._DNDarray__array[:, st_dim1:sp_dim1], pr)
            if a_block is not None:
                a_block = torch.cat([i for i in a_block], dim=0)[prev_cols:]
        else:
            lshap_wait.wait()
            # split is 1, nothing needs to change, only need to select prev_cols to the end of the 0th dimension
            st_dim1, sp_dim1 = 0, lshape_map[pr, 1].item()
            a_block = a._DNDarray__array[prev_cols:]

        if not return_q and pr == a.comm.size - 1:
            # exit loop only setting r
            if equal:
                a_geqrf, tau = a_block.geqrf()
                # set the r_out from the geqrf function
                r_out[prev_cols:, :] = a_geqrf.triu()
            return factories.array(
                r_out,
                is_split=a.split,
                device=a.device,
                comm=a.comm,
                dtype=types.canonical_heat_type(r_out.dtype),
            )

        # run geqrf on the A1 block ( A = [A1 A2] )
        if equal:
            a_geqrf, tau = a_block.geqrf()
            # get v from the lower triangular portion of a
            v = a_geqrf.tril()
            # create a mask to set the diagonal values of v to 1
            v_mask = torch.eye(min(v.shape), dtype=a.dtype.torch_type())
            dims_to_pad = [i - j for i, j in zip(v.shape, v_mask.shape)]
            dims_to_pad = [0, dims_to_pad[1], 0, dims_to_pad[0]]
            v_mask = torch.nn.functional.pad(v_mask, dims_to_pad, "constant", 0)
            v.masked_fill_(v_mask.bool(), 1)
        else:
            # todo: if differing chunk size need to change the second item here
            # create v and t on the other processes
            v = torch.empty(
                (a.gshape[0] - prev_cols, sp_dim1 - st_dim1), dtype=a.dtype.torch_type()
            )
            t = torch.empty((sp_dim1 - st_dim1, sp_dim1 - st_dim1), dtype=a.dtype.torch_type())

        req_v = a.comm.Ibcast(v, root=pr)
        if equal:
            # calculate t while v is being sent
            t = larft(v, tau)
        req_t = a.comm.Ibcast(t, root=pr)

        # if only R is needed then the below can be used
        if not return_q:
            # todo: update this to work with split=0
            if greater:
                req_v.wait()
                # todo: replace the rest of a with v.t @ a
                # print('vshape', v.shape)
                w = v.t() @ a_block
                req_t.wait()
                # print(a)
                a_block -= v @ t.t() @ w
                r_out = a._DNDarray__array[:, : lshape_map[pr, 1].item() + 1]

        if return_q:
            req_v.wait()
            # pad v on the top with zeros to avoid changing the already calculated portions of R
            v = torch.nn.functional.pad(v, [0, 0, prev_cols, 0], "constant", 0)
            # the split semantics are to get q and a to have the same splits
            i_m = factories.eye(
                a.gshape[0],
                split=0 if a.split == 1 else 1,
                comm=a.comm,
                device=a.device,
                dtype=a.dtype,
            )
            req_t.wait()
            t = t @ v.t()
            v = factories.array(
                v,
                dtype=types.canonical_heat_type(v.dtype),
                split=0 if a.split == 1 else 1,
                comm=a.comm,
                device=a.device,
            )
            t = factories.array(
                t,
                dtype=types.canonical_heat_type(t.dtype),
                split=None,
                comm=a.comm,
                device=a.device,
            )
            # Q = I - V @ T @ V.T
            q = i_m - v @ t
            # a is partially transformed into R
            a = q.T @ a
            # q is stored in q_old to be used in the next round (Q = Q1 @ Q2 @ Q3 ...)
            q_old = q_old @ q
        prev_cols += sp_dim1 - st_dim1  # todo: change if diff block size
    return q_old, a


def transpose(a, axes=None):
    """
    Permute the dimensions of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axes : None or list of ints, optional
        By default, reverse the dimensions, otherwise permute the axes according to the values given.

    Returns
    -------
    p : ht.DNDarray
        a with its axes permuted.
    """
    # type check the input tensor
    if not isinstance(a, dndarray.DNDarray):
        raise TypeError("a must be of type ht.DNDarray, but was {}".format(type(a)))

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
                raise TypeError("axis must be an integer, but was {}".format(type(axis)))
            elif axis < 0:
                axes[index] = axis + dimensions

    # infer the new split axis, it is the position of the split axis within the new axes permutation
    try:
        transposed_split = axes.index(a.split) if a.split is not None else None
    except ValueError:
        raise ValueError("axes do not match tensor shape")

    # try to rearrange the tensor and return a new transposed variant
    try:
        transposed_data = a._DNDarray__array.permute(*axes)
        transposed_shape = tuple(a.shape[axis] for axis in axes)

        return dndarray.DNDarray(
            transposed_data, transposed_shape, a.dtype, transposed_split, a.device, a.comm
        )
    # if not possible re- raise any torch exception as ValueError
    except (RuntimeError, IndexError) as exception:
        raise ValueError(str(exception))


# statically allocated index slices for non-iterable dimensions in triangular operations
__index_base = (slice(None), slice(None))


def __tri_op(m, k, op):
    """
    Generic implementation of triangle operations on tensors. It takes care of input sanitation and non-standard
    broadcast behavior of the 2D triangle-operators.

    Parameters
    ----------
    m : ht.DNDarray
        Input tensor for which to compute the triangle operator.
    k : int, optional
        Diagonal above which to apply the triangle operator, k<0 is below and k>0 is above.
    op : callable
        Implementation of the triangle operator.

    Returns
    -------
    triangle_tensor : ht.DNDarray
        DNDarray with the applied triangle operation

    Raises
    ------
    TypeError
        If the input is not a tensor or the diagonal offset cannot be converted to an integral value.
    """
    if not isinstance(m, dndarray.DNDarray):
        raise TypeError("Expected m to be a tensor but was {}".format(type(m)))

    try:
        k = int(k)
    except ValueError:
        raise TypeError("Expected k to be integral, but was {}".format(type(k)))

    # chunk the global shape of the tensor to obtain the offset compared to the other ranks
    offset, _, _ = m.comm.chunk(m.shape, m.split)
    dimensions = len(m.shape)

    # manually repeat the input for vectors
    if dimensions == 1:
        triangle = m._DNDarray__array.expand(m.shape[0], -1)
        if torch.numel(triangle > 0):
            triangle = op(triangle, k - offset)

        return dndarray.DNDarray(
            triangle,
            (m.shape[0], m.shape[0]),
            m.dtype,
            None if m.split is None else 1,
            m.device,
            m.comm,
        )

    original = m._DNDarray__array
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

    return dndarray.DNDarray(output, m.shape, m.dtype, m.split, m.device, m.comm)


def tril(m, k=0):
    """
    Returns the lower triangular part of the tensor, the other elements of the result tensor are set to 0.

    The lower triangular part of the tensor is defined as the elements on and below the diagonal.

    The argument k controls which diagonal to consider. If k=0, all elements on and below the main diagonal are
    retained. A positive value includes just as many diagonals above the main diagonal, and similarly a negative
    value excludes just as many diagonals below the main diagonal.

    Parameters
    ----------
    m : ht.DNDarray
        Input tensor for which to compute the lower triangle.
    k : int, optional
        Diagonal above which to zero elements. k=0 (default) is the main diagonal, k<0 is below and k>0 is above.

    Returns
    -------
    lower_triangle : ht.DNDarray
        Lower triangle of the input tensor.
    """
    return __tri_op(m, k, torch.tril)


def triu(m, k=0):
    """
    Returns the upper triangular part of the tensor, the other elements of the result tensor are set to 0.

    The upper triangular part of the tensor is defined as the elements on and below the diagonal.

    The argument k controls which diagonal to consider. If k=0, all elements on and below the main diagonal are
    retained. A positive value includes just as many diagonals above the main diagonal, and similarly a negative
    value excludes just as many diagonals below the main diagonal.

    Parameters
    ----------
    m : ht.DNDarray
        Input tensor for which to compute the upper triangle.
    k : int, optional
        Diagonal above which to zero elements. k=0 (default) is the main diagonal, k<0 is below and k>0 is above.

    Returns
    -------
    upper_triangle : ht.DNDarray
        Upper triangle of the input tensor.
    """
    return __tri_op(m, k, torch.triu)
