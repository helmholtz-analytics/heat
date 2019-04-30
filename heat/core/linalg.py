import itertools
import torch

from .communication import MPI, MPI_WORLD
from . import dndarray
from . import factories

__all__ = [
    'matmul',
    'transpose',
    'tril',
    'triu'
]


def matmul(a, b, out=None, out_split=None):
    """
    Matrix multiplication function based of the CRMM method as described in Reference 1. Communication scheme based upon reference [2]

    Parameters
    ----------
    a : ht.tensor
        2 dimensions: L x P

    b : ht.tensor
        2 dimensions: P x Q

    out : ht.tensor
        Optional
        output tensor

    out_split : int
        Optional
        split direction of the output tensor if out is None
        if out is None and out_split is None then the split direction of a is chosen

    Returns
    -------
    ht.tensor
        returns a tensor with the result of a @ b

    References
    ----------
    [1] R. Gu, et al., "Improving Execution Concurrency of Large-scale Matrix Multiplication on Distributed Data-parallel Platforms,"
        IEEE Transactions on Parallel and Distributed Systems, vol 28, no. 9. 2017.
    [2] S. Ryu and D. Kim, "Parallel Huge Matrix Multiplication on a Cluster with GPGPU Accelerators,"
        2018 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), Vancouver, BC, 2018, pp. 877-882.

    Process:
    --------
    1. split a and b into blocks of the respective sizes (M x kB and kB x N)
        kB must be the same in both
        if a common kB cannot be found then x rows/columns can be sliced off and saved for later
            this cut must occur in the 'P' dimension of both matricies
            best way to do this is to either equate all of the

    detailed process:
    -----------------
    1. get the lshape of the data on all procs

    possible problems:
    -> a and b not the same splits
    """
    if a.gshape[-1] != b.gshape[-2]:
        raise ValueError("If the last dimension of a ({}) is not the same size as the second-to-last dimension of b. ({})".format(a.gshape[-1], b.gshape[-2]))

    if a.is_distributed() and b.is_distributed():  # else its simple matmul from torch
        # block sizes dont need to be the same. thy just need the same inner dimmension (kB)
        kB = 0
        rem_a, rem_b = [0] * 2
        if a.split == len(a.gshape)-1 and b.split == len(a.gshape)-2:  # if the split direction is the last dim in a and the first dim in b
            # the max inner dim (kB) is the min value from the result of the integer division of the last dim of a/world size and the first dim of b/world size
            kB = min([a.gshape[-1] // a.comm.size, b.gshape[0] // b.comm.size])
        elif a.split == len(a.gshape)-1:
            kB = a.gshape[-1] // a.comm.size
        elif b.split == len(a.gshape)-2:
            kB = b.gshape[-2] // b.comm.size
            kB = kB if kB < a.gshape[-1] else a.gshape[-1]
        else:  # if the split is not in either of these directions then kB can be anything, it just needs to be tuned
            # what to do here?
            raise NotImplementedError("need to implement case of split 0 and split 1 for a and b respectively")
            pass

        if a.lshape[-1] % kB != 0:
            rem_a = 1
        if b.lshape[-2] % kB != 0:
            rem_b = 1

        # print(kB, rem_a, rem_b)

        # get the lshape map to determine what needs to be sent where as well as M and N
        # lshape map dims -> {node, a=0, b=1, lshape}
        lshape_map = factories.zeros((a.comm.size, 2, len(a.gshape)))
        lshape_map[a.comm.rank, 0, :] = torch.Tensor(a.lshape)
        lshape_map[b.comm.rank, 1, :] = torch.Tensor(b.lshape)
        a.comm.Allreduce(MPI.IN_PLACE, lshape_map, MPI.SUM)

        # find mB (first blocking dim for a) and nB (2nd blocking dim for b)
        mB = min(lshape_map[:, 0, -2]).item()
        nB = min(lshape_map[:, 1, -1]).item()
        # todo: handle the outside dimensional remainders
        print('mb', mB, 'kB', kB, 'nb', nB)

        # check for remaining dims in the outside dimensions
        rem_a_out, rem_b_out = 0, 0
        if a.lshape[-2] % mB != 0:
            rem_a_out = 1
        if b.lshape[-1] % nB != 0:
            rem_b_out = 1
        print('rems:', rem_a_out, rem_a, rem_b, rem_b_out)
        # print('block sizes:', '\n a:', mB, kB, '\n b:', kB, nB)
        # if there is any rem_a then there are remainders on all processes

        # get the flags from all processes
        rem_map = factories.zeros((a.comm.size, len(a.gshape) + len(b.gshape)))
        rem_map_hold = factories.zeros((a.comm.size, len(a.gshape) + len(b.gshape)))
        rem_map_hold[a.comm.rank, :] = torch.Tensor([rem_a_out, rem_a, rem_b, rem_b_out])
        a.comm.Allreduce(rem_map_hold, rem_map, MPI.SUM)
        print(rem_map)

        # TODO: make index map of where all the data is / tie the indecies of the data to the data (dictionary)
        # index_map dims guide -> {process number, a=0/b=1, relevent indices (only need the last two)}
        index_map = factories.zeros((a.comm.size, 2, 1))
        a_idx = a.comm.chunk(a.shape, a.split)[2]
        index_map[a.comm.rank, 0, :] = tuple()
        # with the index map then the communication can be determined
        # this index map will also be used as a meta data / dictionary when the data is passed around
########################################################################################################################################################


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
        raise TypeError('a must be of type ht.DNDarray, but was {}'.format(type(a)))

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
            raise ValueError('axes must be an iterable containing ints')

        if len(axes) != dimensions:
            raise ValueError('axes do not match tensor shape')
        for index, axis in enumerate(axes):
            if not isinstance(axis, int):
                raise TypeError('axis must be an integer, but was {}'.format(type(axis)))
            elif axis < 0:
                axes[index] = axis + dimensions

    # infer the new split axis, it is the position of the split axis within the new axes permutation
    try:
        transposed_split = axes.index(a.split) if a.split is not None else None
    except ValueError:
        raise ValueError('axes do not match tensor shape')

    # try to rearrange the tensor and return a new transposed variant
    try:
        transposed_data = a._DNDarray__array.permute(*axes)
        transposed_shape = tuple(a.shape[axis] for axis in axes)

        return dndarray.DNDarray(transposed_data, transposed_shape, a.dtype, transposed_split, a.device, a.comm)
    # if not possible re- raise any torch exception as ValueError
    except RuntimeError as exception:
        raise ValueError(str(exception))


# statically allocated index slices for non-iterable dimensions in triangular operations
__index_base = (slice(None), slice(None),)


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
        raise TypeError('Expected m to be a tensor but was {}'.format(type(m)))

    try:
        k = int(k)
    except ValueError:
        raise TypeError('Expected k to be integral, but was {}'.format(type(k)))

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
            (m.shape[0], m.shape[0],),
            m.dtype,
            None if m.split is None else 1,
            m.device,
            m.comm
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
