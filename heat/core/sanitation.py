"""
Collection of validation/sanitation routines.
"""
from __future__ import annotations

import numpy as np
import torch
import warnings
from typing import Any, Union, Sequence, List, Tuple

from .communication import MPI, Communication
from .dndarray import DNDarray

from . import factories
from . import stride_tricks
from . import types


__all__ = [
    "sanitize_distribution",
    "sanitize_in",
    "sanitize_infinity",
    "sanitize_in_tensor",
    "sanitize_lshape",
    "sanitize_out",
    "sanitize_sequence",
    "scalar_to_1d",
]


def sanitize_distribution(
    *args: DNDarray, target: DNDarray, diff_map: torch.Tensor = None
) -> Union[DNDarray, Tuple(DNDarray)]:
    """
    Distribute every `arg` according to `target.lshape_map` or, if provided, `diff_map`.
    After this sanitation, the lshapes are compatible along the split dimension.
    `Args` can contain non-distributed DNDarrays, they will be split afterwards, if `target` is split.

    Parameters
    ----------
    args : DNDarray
        Dndarrays to be distributed

    target : DNDarray
        Dndarray used to sanitize the metadata and to, if diff_map is not given, determine the resulting distribution.

    diff_map : torch.Tensor (optional)
        Different lshape_map. Overwrites the distribution of the target array.
        Used in cases when the target array does not correspond to the actually wanted distribution,
        e.g. because it only contains a single element along the split axis and gets broadcast.

    Raises
    ------
    TypeError
        When an argument is not a ``DNDarray`` or ``None``.
    ValueError
        When the split-axes or sizes along the split-axis do not match.

    See Also
    ---------
    :func:`~heat.core.dndarray.create_lshape_map`
        Function to create the lshape_map.
    """
    out = []
    sanitize_in(target)
    # early out on 1 process
    if target.comm.size == 1:
        for arg in args:
            sanitize_in(arg)
            out.append(arg)
        return tuple(out) if len(out) > 1 else out[0]

    target_split = target.split
    if diff_map is not None:
        sanitize_in_tensor(diff_map)
        target_map = diff_map
        if target_split is not None:
            tmap_split = target_map[:, target_split]
            target_size = tmap_split.sum().item()
            # Check if the diff_map is balanced
            w_size = target_map.shape[0]
            tmap_balanced = torch.full_like(tmap_split, fill_value=target_size // w_size)
            remainder = target_size % w_size
            tmap_balanced[:remainder] += 1
            target_balanced = torch.equal(tmap_balanced, tmap_split)
    elif target_split is not None:
        target_map = target.lshape_map
        target_size = target.shape[target_split]
        target_balanced = target.is_balanced(force_check=False)

    for arg in args:
        sanitize_in(arg)
        if target.comm != arg.comm:
            try:
                raise NotImplementedError(
                    f"Not implemented for other comms, found {target.comm.name} and {arg.comm.name}"
                )
            except Exception:
                raise NotImplementedError("Not implemented for other comms")
        elif target_split is None:
            if arg.split is not None:
                raise NotImplementedError(
                    f"DNDarrays must have the same split axes, found {target_split} and {arg.split}"
                )
            else:
                out.append(arg)
        elif arg.shape[target_split] == 1 and target_size > 1:  # broadcasting in split-dimension
            out.append(arg.resplit(None))
        elif arg.shape[target_split] != target_size:
            raise ValueError(
                f"Cannot distribute to match in split dimension, shapes are {target.shape} and {arg.shape}"
            )
        elif arg.split is None:  # undistributed case
            if target_balanced:
                out.append(
                    factories.array(
                        arg, split=target_split, copy=False, comm=arg.comm, device=arg.device
                    )
                )
            else:
                idx = [slice(None)] * arg.ndim
                idx[target_split] = slice(
                    target_map[: arg.comm.rank, target_split].sum(),
                    target_map[: arg.comm.rank + 1, target_split].sum(),
                )
                out.append(
                    factories.array(
                        arg.larray[tuple(idx)],
                        is_split=target_split,
                        copy=False,
                        comm=arg.comm,
                        device=arg.device,
                    )
                )
        elif arg.split != target_split:
            raise NotImplementedError(
                f"DNDarrays must have the same split axes, found {target_split} and {arg.split}"
            )
        elif not (
            # False
            target_balanced
            and arg.is_balanced(force_check=False)
        ):  # Split axes are the same and atleast one is not balanced
            current_map = arg.lshape_map
            out_map = current_map.clone()
            out_map[:, target_split] = target_map[:, target_split]
            if not (current_map[:, target_split] == target_map[:, target_split]).all():
                out.append(arg.redistribute(lshape_map=current_map, target_map=out_map))
            else:
                out.append(arg)
        else:  # both are balanced
            out.append(arg)
    if len(out) == 1:
        return out[0]
    return tuple(out)


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
        raise TypeError(f"Input must be a DNDarray, is {type(x)}")


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
        raise TypeError(f"Input must be a torch.Tensor, is {type(x)}")


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
        non_zero = [i for i in range(len(tshape)) if tshape[i] != 0]
        cond = [tshape[i] == gshape[i] for i in non_zero].count(True) == len(non_zero)
        if cond:
            return
        else:
            raise ValueError(
                f"Shape of local tensor is inconsistent with global DNDarray: tensor.shape is {tshape}, should be {gshape}"
            )
    # size of non-split dimensions must match global shape
    reduced_gshape = gshape[:split] + gshape[split + 1 :]
    reduced_tshape = tshape[:split] + tshape[split + 1 :]
    if reduced_tshape == reduced_gshape:
        return
    raise ValueError(
        f"Shape of local tensor along non-split axes is inconsistent with global DNDarray: tensor.shape is {tshape}, DNDarray is {gshape}"
    )


def sanitize_out(
    out: Any,
    output_shape: Tuple,
    output_split: int,
    output_device: str,
    output_comm: Communication = None,
):
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

    output_comm : Communication
                    Communication object of the result of the operation

    Raises
    ------
    TypeError
        if ``out`` is not a ``DNDarray``.
    ValueError
        if shape, split direction, or device of the output buffer ``out`` do not match the operation result.
    """
    if not isinstance(out, DNDarray):
        raise TypeError(f"expected `out` to be None or a DNDarray, but was {type(out)}")

    out_proxy = out.__torch_proxy__()
    out_proxy.names = [
        "split" if (out.split is not None and i == out.split) else f"_{i}"
        for i in range(out_proxy.ndim)
    ]
    out_proxy = out_proxy.squeeze()

    check_proxy = torch.ones(1).expand(output_shape)
    check_proxy.names = [
        "split" if (output_split is not None and i == output_split) else f"_{i}"
        for i in range(check_proxy.ndim)
    ]
    check_proxy = check_proxy.squeeze()

    if out_proxy.shape != check_proxy.shape:
        raise ValueError(f"Expecting output buffer of shape {output_shape}, got {out.shape}")
    count_split = int(out.split is not None) + int(output_split is not None)
    if count_split == 1:
        raise ValueError(
            "Split axis of output buffer is inconsistent with split semantics for this operation."
        )
    elif count_split == 2:
        if out.shape[out.split] > 1:  # split axis is not squeezed out
            if out_proxy.names.index("split") != check_proxy.names.index("split"):
                raise ValueError(
                    "Split axis of output buffer is inconsistent with split semantics for this operation."
                )
        else:  # split axis is squeezed out
            num_dim_before_split = len(
                [name for name in out_proxy.names if int(name[1:]) < out.split]
            )
            check_num_dim_before_split = len(
                [name for name in check_proxy.names if int(name[1:]) < output_split]
            )
            if num_dim_before_split != check_num_dim_before_split:
                raise ValueError(
                    "Split axis of output buffer is inconsistent with split semantics for this operation."
                )
    if out.device != output_device:
        raise ValueError(f"Device mismatch: out is on {out.device}, should be on {output_device}")
    if output_comm is not None and out.comm != output_comm:
        try:
            raise NotImplementedError(
                f"Not implemented for other comms, found {out.comm.name} and {output_comm.name}"
            )
        except Exception:
            raise NotImplementedError("Not implemented for other comms")


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
        raise TypeError(f"seq must be a list or a tuple, got {type(seq)}")


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
