import numpy as np
import torch
import warnings

from .communication import MPI

from . import dndarray
from . import factories
from . import stride_tricks
from . import types


__all__ = [
    "sanitize_in",
    "sanitize_infinity",
    "sanitize_in_tensor",
    "sanitize_lshape",
    "sanitize_out",
    "sanitize_sequence",
    "scalar_to_1d",
]


def sanitize_in(x):
    """
    Raise TypeError if input is not DNDarray

    Parameters
    ----------
    x : Object
    """
    if not isinstance(x, dndarray.DNDarray):
        raise TypeError("input must be a DNDarray, is {}".format(type(x)))


def sanitize_infinity(x):
    """
    Returns largest possible value for the dtype of the input array.

    Parameters:
    -----------
    x: `DNDarray` or `torch.Tensor`

    Returns:
    --------
    largest: largest possible value for the given dtype
    """
    dtype = x.dtype if isinstance(x, torch.Tensor) else x.larray.dtype
    try:
        largest = torch.finfo(dtype).max
    except TypeError:
        largest = torch.iinfo(dtype).max

    return largest


def sanitize_in_tensor(x):
    """
    Raise TypeError if input is not torch.tensor

    Parameters
    ----------
    x : Object
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("input must be a torch.tensor, is {}".format(type(x)))


def sanitize_lshape(array, tensor):
    """
    Verify shape consistency when manipulating process-local `larray`s (torch tensors).

    Parameters
    ----------
    array : DNDarray
        the original, potentially distributed `DNDarray`
    tensor : torch.tensor
        process-local data meant to replace `array.larray`
    """
    # no input sanitation here, must be taken care of in parent function
    tshape = tuple(tensor.shape)
    if tshape == array.lshape:
        return tshape
    else:
        gshape = array.gshape
        split = array.split
        if split is None:
            raise ValueError(
                "Shape of local tensor is inconsistent with global DNDarray: tensor.shape is {}, should be {}".format(
                    tshape, gshape
                )
            )

        # size of non-split dimensions must match global shape
        reduced_gshape = gshape[:split] + gshape[split + 1 :]
        reduced_tshape = tshape[:split] + tshape[split + 1 :]
        if reduced_tshape == reduced_gshape:
            return tshape
        else:
            raise ValueError(
                "Shape of local tensor along non-split axes is inconsistent with global DNDarray: tensor.shape is {}, DNDarray is {}".format(
                    tuple(tshape), gshape
                )
            )


def sanitize_out(out, output_shape, output_split, output_device):
    """
    Validate out buffer

    Parameters
    ----------
    out : Object
          the `out` buffer where the result of some operation will be stored

    output_shape : Tuple
                   the calculated shape returned by the operation

    output_split : Int
                   the calculated split axis returned by the operation

    output_device : Str
                    "cpu" or "gpu" as per location of data
    """

    if not isinstance(out, dndarray.DNDarray):
        raise TypeError("expected `out` to be None or a DNDarray, but was {}".format(type(out)))

    if out.gshape != output_shape:
        raise ValueError(
            "Expecting output buffer of shape {}, got {}".format(output_shape, out.shape)
        )
    if out.split is not output_split:
        raise ValueError(
            "Split axis of output buffer is inconsistent with split semantics (see documentation)."
        )
    if out.device is not output_device:
        raise ValueError(
            "Device mismatch: out is on {}, should be on {}".format(out.device, output_device)
        )


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
    else:
        raise TypeError("seq must be a list or a tuple, got {}".format(type(seq)))


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
    return factories.array(x.larray.unsqueeze(0), dtype=x.dtype, split=x.split, comm=x.comm)
