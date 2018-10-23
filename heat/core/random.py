import torch

from .communication import MPI_SELF
from . import tensor
from . import types


def set_gseed(seed):
    # TODO: think about proper random number generation
    # TODO: comment me
    # TODO: test me
    torch.manual_seed(seed)


def uniform(low=0.0, high=1.0, size=None):
    # TODO: comment me
    # TODO: test me
    if size is None:
        size = (1,)

    return tensor(torch.Tensor(*size).uniform_(low, high), size, types.float32, None, MPI_SELF)
