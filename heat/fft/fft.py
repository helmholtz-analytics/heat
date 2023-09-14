"""Provides a collection of Discrete Fast Fourier Transforms (DFFT) and their inverses."""

import torch

from ..core.communication import MPI
from ..core.dndarray import DNDarray
from ..core.stride_tricks import sanitize_axis
from ..core.types import promote_types, heat_type_of
from ..core.factories import array, zeros

from typing import Type, Union, Tuple, Any, Iterable, Optional

__all__ = ["fft2", "fftn"]


# TODO: implement __fft_op, __fftn_op to deal with the different operations
def __fftn_op(x: DNDarray, fftn_op: callable, **kwargs) -> DNDarray:
    """
    Helper function for fftn
    """
    try:
        local_x = x.larray
    except AttributeError:
        raise TypeError("x must be a DNDarray, is {}".format(type(x)))
    original_split = x.split

    # sanitize kwargs
    axes = kwargs.get("axes", None)
    axes = sanitize_axis(x.gshape, axes)
    s = kwargs.get("s", None)
    norm = kwargs.get("norm", None)

    # non-distributed DNDarray
    if not x.is_distributed():
        result = fftn_op(local_x, s=s, dim=axes, norm=norm)
        return array(result, split=original_split, device=x.device, comm=x.comm)

    # distributed DNDarray:
    # calculate output shape
    output_shape = list(x.shape)
    if s is not None:
        if axes is None:
            axes = tuple(range(x.ndim)[-len(s) :])
        for i, axis in enumerate(axes):
            output_shape[axis] = s[i]
    else:
        s = tuple(output_shape[axis] for axis in axes)
    output_shape = tuple(output_shape)

    fft_along_split = original_split in axes

    # FFT along non-split axes only
    if not fft_along_split:
        result = fftn_op(local_x, s=s, dim=axes, norm=norm)
        return DNDarray(
            result,
            gshape=output_shape,
            dtype=heat_type_of(result),
            split=original_split,
            device=x.device,
            comm=x.comm,
            balanced=x.balanced,
        )

    # FFT along split axis
    if original_split != 0:
        # transpose x so redistribution starts from axis 0
        transpose_axes = list(range(x.ndim))
        transpose_axes[0], transpose_axes[original_split] = (
            transpose_axes[original_split],
            transpose_axes[0],
        )
        x = x.transpose(transpose_axes)

    # redistribute x from axis 0 to 1
    _ = x.resplit(axis=1)
    # FFT along axis 0 (now non-split)
    split_index = axes.index(original_split)
    partial_result = __fftn_op(_, fftn_op, s=(s[split_index],), axes=(0,), norm=norm)
    del _
    # redistribute partial result from axis 1 to 0
    partial_result.resplit_(axis=0)
    if original_split != 0:
        # transpose x, partial_result back to original shape
        x = x.transpose(transpose_axes)
        partial_result = partial_result.transpose(transpose_axes)

    # now apply FFT along leftover (non-split) axes
    axes = list(axes)
    axes.remove(original_split)
    axes = tuple(axes)
    s = list(s)
    s = s[:split_index] + s[split_index + 1 :]
    s = tuple(s)
    result = __fftn_op(partial_result, fftn_op, s=s, axes=axes, norm=norm)
    del partial_result
    return array(result.larray, is_split=original_split, device=x.device, comm=x.comm)


def fft2(
    x: DNDarray, s: Tuple[int, int] = None, axes: Tuple[int, int] = (-2, -1), norm: str = None
) -> DNDarray:
    """
    Compute the 2-dimensional discrete Fourier Transform.

    This function computes the 2-dimensional discrete Fourier Transform over the specified axes in an M-dimensional
    array by means of the Fast Fourier Transform (FFT). By default, the last two axes are transformed, while the
    remaining axes are left unchanged.

    Parameters
    ----------
    x : DNDarray
        Input array, can be complex
    s : Tuple[int, int], optional
        Shape of the output along the transformed axes. (default is x.shape)
    axes : Tuple[int, int], optional
        Axes over which to compute the FFT. If not given, the last `len(s)` axes are used, or all axes if `s` is also
        not specified. Repeated indices in `axes` means that the transform over that axis is performed multiple times.
        (default is (-2, -1))
    norm : str, optional
        Normalization mode: 'forward', 'backward', or 'ortho' (see `numpy.fft` for details). Default is "backward".

    Notes
    -----
    This function requires MPI communication if the input array is distributed and the split axis is transformed.
    """
    return __fftn_op(x, torch.fft.fft2, s=s, axes=axes, norm=norm)


def fftn(
    x: DNDarray, s: Tuple[int, ...] = None, axes: Tuple[int, ...] = None, norm: str = None
) -> DNDarray:
    """
    Compute the N-dimensional discrete Fourier Transform.

    This function computes the N-dimensional discrete Fourier Transform over any number of axes in an M-dimensional
    array by means of the Fast Fourier Transform (FFT). By default, all axes are transformed, with the real transform
    performed over the last axis, while the remaining transforms are complex.

    Parameters
    ----------
    x : DNDarray
        Input array, can be complex
    s : Tuple[int, ...], optional
        Shape of the output along the transformed axes. (default is x.shape)
    axes : Tuple[int, ...], optional
        Axes over which to compute the FFT. If not given, the last `len(s)` axes are used, or all axes if `s` is also
        not specified. Repeated indices in `axes` means that the transform over that axis is performed multiple times.
        (default is None)
    norm : str, optional
        Normalization mode: 'forward', 'backward', or 'ortho' (see `numpy.fft` for details). Default is "backward".

    Notes
    -----
    This function requires MPI communication if the input array is distributed and the split axis is transformed.
    """
    return __fftn_op(x, torch.fft.fftn, s=s, axes=axes, norm=norm)
