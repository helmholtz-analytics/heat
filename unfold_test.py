"""
Test module for DNDarray.unfold
"""

import heat as ht
from mpi4py import MPI
import torch

from heat import factories
from heat import DNDarray


def unfold(a: DNDarray, dimension: int, size: int, step: int):
    """
    Returns a view of the original tensor which contains all slices of size size from self tensor in the dimension dimension.

    Behaves like torch.Tensor.unfold for DNDarrays. [torch.Tensor.unfold](https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html)
    """
    comm = a.comm
    dev = a.device

    if a.split is None or comm.size == 1 or a.split != dimension:  # early out
        ret = factories.array(
            a.larray.unfold(dimension, size, step), is_split=a.split, device=dev, comm=comm
        )

        return ret


# tests
n = 8

x = torch.arange(0.0, n)
y = 2 * x.unsqueeze(1) + x.unsqueeze(0)
y = factories.array(y)
y.resplit_(1)

print(y)
print(unfold(y, 0, 3, 1))
