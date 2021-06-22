"""
Collection of validation/sanitation routines.
"""
from __future__ import annotations

import numpy as np
import torch
import warnings
from typing import Any, Union, Sequence, List, Tuple

from .communication import MPI
from .dndarray import DNDarray

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


def sanitize_in(x: Any):
    """
    Verify that input object is ``DNDarray``.

    Parameters
    ----------
    x : Any
        Input object

    Raises
    ------
    TypeError
        When ``x`` is not a ``DNDarray``.
    """
    if not isinstance(x, DNDarray):
        raise TypeError("Input must be a DNDarray, is {}".format(type(x)))


def sanitize_infinity(x: Union[DNDarray, torch.Tensor]) -> Union[int, float]:
    """
    Returns largest possible value for the ``dtype`` of the input array.

    Parameters
    -----------
    x: Union[DNDarray, torch.Tensor]
        Input object.
    """
    dtype = x.dtype if isinstance(x, torch.Tensor) else x.larray.dtype
    try:
        largest = torch.finfo(dtype).max
    except TypeError:
        largest = torch.iinfo(dtype).max

    return largest


def sanitize_in_tensor(x: Any):
    """
    Verify that input object is ``torch.Tensor``.

    Parameters
    ----------
    x : Any
        Input object.

    Raises
    ------
    TypeError
        When ``x`` is not a ``torch.Tensor``.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor, is {}".format(type(x)))


def sanitize_lshape(array: DNDarray, tensor: torch.Tensor):
    """
    Verify shape consistency when manipulating process-local arrays.

    Parameters
    ----------
    array : DNDarray
        the original, potentially distributed ``DNDarray``
    tensor : torch.Tensor
        process-local data meant to replace ``array.larray``

    Raises
    ------
    ValueError
        if shape of local ``torch.Tensor`` is inconsistent with global ``DNDarray``.
    """
    # input sanitation is done in parent function
    tshape = tuple(tensor.shape)
    if tshape == array.lshape:
        return
    gshape = array.gshape
    split = array.split
    if split is None:
        # allow for axes with size 0, other axes must match
        non_zero = list(i for i in range(len(tshape)) if tshape[i] != 0)
        cond = list(tshape[i] == gshape[i] for i in non_zero).count(True) == len(non_zero)
        if cond:
            return
        else:
            raise ValueError(
                "Shape of local tensor is inconsistent with global DNDarray: tensor.shape is {}, should be {}".format(
                    tshape, gshape
                )
            )
    # size of non-split dimensions must match global shape
    reduced_gshape = gshape[:split] + gshape[split + 1 :]
    reduced_tshape = tshape[:split] + tshape[split + 1 :]
    if reduced_tshape == reduced_gshape:
        return
    raise ValueError(
        "Shape of local tensor along non-split axes is inconsistent with global DNDarray: tensor.shape is {}, DNDarray is {}".format(
            tshape, gshape
        )
    )


def sanitize_out(out: Any, output_shape: Tuple, output_split: int, output_device: str):
    """
    Validate output buffer ``out``.

    Parameters
    ----------
    out : Any
          the `out` buffer where the result of some operation will be stored

    output_shape : Tuple
                   the calculated shape returned by the operation

    output_split : Int
                   the calculated split axis returned by the operation

    output_device : Str
                    "cpu" or "gpu" as per location of data

    Raises
    ------
    TypeError
        if ``out`` is not a ``DNDarray``.
    ValueError
        if shape, split direction, or device of the output buffer ``out`` do not match the operation result.
    """
    if not isinstance(out, DNDarray):
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


def sanitize_sequence(
    seq: Union[Sequence[int, ...], Sequence[float, ...], DNDarray, torch.Tensor]
) -> List:
    """
    Check if sequence is valid, return list.

    Parameters
    ----------
    seq : Union[Sequence[int, ...], Sequence[float, ...], DNDarray, torch.Tensor]
        Input sequence.

    Raises
    ------
    TypeError
        if ``seq`` is neither a list nor a tuple
    """
    if isinstance(seq, list):
        return seq
    elif isinstance(seq, tuple):
        return list(seq)
    else:
        raise TypeError("seq must be a list or a tuple, got {}".format(type(seq)))


def scalar_to_1d(x: DNDarray) -> DNDarray:
    """
    Turn a scalar ``DNDarray`` into a 1-D ``DNDarray`` with 1 element.

    Parameters
    ----------
    x : DNDarray
        with `x.ndim = 0`
    """
    return factories.array(
        x.larray.unsqueeze(0), dtype=x.dtype, split=x.split, comm=x.comm, device=x.device
    )
