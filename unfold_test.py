"""
Test module for DNDarray.unfold
"""

import heat as ht
from mpi4py import MPI
import torch

from heat import factories
from heat import DNDarray


def unfold(a: DNDarray, dimension: int, size: int, step: int = 1):
    """
    Returns a view of the original tensor which contains all slices of size size from self tensor in the dimension dimension.

    Behaves like torch.Tensor.unfold for DNDarrays. [torch.Tensor.unfold](https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html)
    """
    comm = a.comm
    dev = a.device
    tdev = dev.torch_device

    if a.split is None or comm.size == 1 or a.split != dimension:  # early out
        ret = factories.array(
            a.larray.unfold(dimension, size, step), is_split=a.split, device=dev, comm=comm
        )

        return ret
    else:  # comm.size > 1 and split axis == unfold axis
        # initialize the array
        # a_shape = a.shape
        # index range [0:sizedim-1-(size-1)] = [0:sizedim-size]
        # --> size of axis: ceil((sizedim-size+1) / step) = floor(sizedim-size) / step)) + 1
        # ret_shape = (*a_shape[:dimension], int((a_shape[dimension]-size)/step) + 1, a_shape[dimension+1:], size)

        # ret = ht.zeros(ret_shape, device=dev, split=a.split)

        # send the needed entries in the unfold dimension from node n to n+1 or n-1
        a.get_halo(size - 1)
        a_lshapes_cum = torch.hstack(
            [
                torch.zeros(1, dtype=torch.int32, device=tdev),
                torch.cumsum(a.lshape_map[:, dimension], 0),
            ]
        )
        if comm.rank == 0:
            print(a_lshapes_cum)
        min_index = ((a_lshapes_cum[comm.rank] - 1) // step + 1) * step - a_lshapes_cum[
            comm.rank
        ]  # min local index in unfold dimension
        print(f"min_index on rank {comm.rank}: {min_index}")
        unfold_loc = a.larray[
            dimension * (slice(None, None, None),) + (slice(min_index, None, None), Ellipsis)
        ].unfold(dimension, size, step)
        ret_larray = unfold_loc
        if comm.rank < comm.size - 1:  # possibly unfold with halo from next rank
            max_index = a.lshape[dimension] - min_index - 1
            max_index = max_index // step * step + min_index  # max local index in unfold dimension
            rem = max_index + size - a.lshape[dimension]
            if rem > 0:  # need data from halo
                unfold_halo = torch.cat(
                    (
                        a.larray[
                            dimension * (slice(None, None, None),)
                            + (slice(max_index, None, None), Ellipsis)
                        ],
                        a.halo_next[
                            dimension * (slice(None, None, None),)
                            + (slice(None, rem, None), Ellipsis)
                        ],
                    ),
                    dimension,
                ).unfold(dimension, size, step)
                ret_larray = torch.cat((unfold_loc, unfold_halo), dimension)
        ret = factories.array(ret_larray, is_split=dimension, device=dev, comm=comm)

        return ret


# tests
n = 100

# x = torch.arange(0.0, n)
# y = 2 * x.unsqueeze(1) + x.unsqueeze(0)
# y = factories.array(y)
# y.resplit_(1)

# print(y)
# print(unfold(y, 0, 3, 1))

x = torch.arange(0, n)
y = factories.array(x)
y.resplit_(0)

u = x.unfold(0, 5, 10)
u = factories.array(u)
v = unfold(y, 0, 5, 10)

comm = u.comm
# print(v)
equal = ht.equal(u, v)

u_shape = u.shape
v_shape = v.shape

if comm.rank == 0:
    print(f"u.shape: {u_shape}")
    print(f"v.shape: {v_shape}")
    print(f"torch and heat unfold equal: {equal}")
    print(f"u: {u}")

print(f"v: {v}")
