import numpy as np
import torch
import warnings

from .communication import MPI

from . import dndarray
from . import factories
from . import stride_tricks
from . import types


__all__ = ["sanitize_input", "sanitize_sequence", "scalar_to_1d"]


def sanitize_input(x):
    """
    Raise TypeError if input is not DNDarray
    """
    if not isinstance(x, dndarray.DNDarray):
        raise TypeError("input must be a DNDarray, is {}".format(type(x)))


def sanitize_sequence(seq):
    """
    Check if sequence is valid, return list.

    Parameters
    ----------
    seq : Union[Sequence[ints, ...], Sequence[floats, ...], DNDarray, torch.tensor]

    Returns
    -------
    seq : List
    """
    if isinstance(seq, list):
        return seq
    elif isinstance(seq, tuple):
        return list(seq)
    elif isinstance(seq, dndarray.DNDarray):
        if seq.split is None:
            return seq._DNDarray__array.tolist()
        else:
            raise ValueError(
                "seq is a distributed DNDarray, expected a list, a tuple, or a process-local array."
            )
    elif isinstance(seq, torch.Tensor):
        return seq.tolist()
    else:
        raise TypeError(
            "seq must be a list, a tuple, or a process-local array, got {}".format(type(seq))
        )


def scalar_to_1d(x):
    """
    Turn a scalar DNDarray into a 1-D DNDarray with 1 element.

    Parameters
    ----------
    x : DNDarray
        with `x.ndim = 0`

    Returns
    -------
    x : DNDarray
        where `x.ndim = 1` and `x.shape = (1,)`
    """
    return factories.array(
        x._DNDarray__array.unsqueeze(0), dtype=x.dtype, split=x.split, comm=x.comm
    )
