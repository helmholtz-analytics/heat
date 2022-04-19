"""Provides a collection of signal-processing operations"""

import torch
from typing import Union, Tuple, Sequence

from .communication import MPI
from .dndarray import DNDarray
from .types import promote_types
from .manipulations import pad
import torch.nn.functional as fc

__all__ = ["convolve"]


def convolve(a: DNDarray, v: DNDarray, mode: str = "full") -> DNDarray:
    """
    Returns the discrete, linear convolution of two one-dimensional `DNDarray`s.

    Parameters
    ----------
    a : DNDarray
        One-dimensional signal `DNDarray` of shape (N,)
    v : DNDarray
        One-dimensional filter weight `DNDarray` of shape (M,).
    mode : str
        Can be 'full', 'valid', or 'same'. Default is 'full'.
        'full':
          Returns the convolution at
          each point of overlap, with an output shape of (N+M-1,). At
          the end-points of the convolution, the signals do not overlap
          completely, and boundary effects may be seen.
        'same':
          Mode 'same' returns output  of length 'N'. Boundary
          effects are still visible. This mode is not supported for
          even-sized filter weights
        'valid':
          Mode 'valid' returns output of length 'N-M+1'. The
          convolution product is only given for points where the signals
          overlap completely. Values outside the signal boundary have no
          effect.

    Notes
    -----
        TODO: fix this note and underlying API inconsistencies
        There is  differences to the numpy convolve function:
        The inputs are not swapped if v is larger than a. The reason is that v needs to be
        non-splitted. This should not influence performance.


    Examples
    --------
    Note how the convolution operator flips the second array
    before "sliding" the two across one another:
    >>> a = ht.ones(10)
    >>> v = ht.arange(3).astype(ht.float)
    >>> ht.convolve(a, v, mode='full')
    DNDarray([0., 1., 3., 3., 3., 3., 2.])

    Only return the middle values of the convolution.
    Contains boundary effects, where zeros are taken
    into account:
    >>> ht.convolve(a, v, mode='same')
    DNDarray([1., 3., 3., 3., 3.])

    Compute only positions where signal and filter weights
    completely overlap:
    >>> ht.convolve(a, v, mode='valid')
    DNDarray([3., 3., 3.])
    """
    if not isinstance(a, DNDarray) or not isinstance(v, DNDarray):
        raise TypeError("Signal and filter weight must be of type DNDarray")
    if v.is_distributed():
        raise TypeError("Distributed filter weights are not supported")
    if len(a.shape) != 1 or len(v.shape) != 1:
        raise ValueError("Only 1-dimensional input DNDarrays are allowed")
    if a.shape[0] <= v.shape[0]:
        raise ValueError("Filter size must not be greater than or equal to signal size")
    if a.dtype is not v.dtype:
        raise TypeError(
            "Signal and filter weight must be of same dtype, are {} and {}".format(a.dtype, v.dtype)
        )
    if mode == "same" and v.shape[0] % 2 == 0:
        raise ValueError("Mode 'same' cannot be used with even-sized kernel")

    # compute halo size
    halo_size = v.shape[0] // 2

    # pad DNDarray with zeros according to mode
    if mode == "full":
        pad_size = v.shape[0] - 1
        gshape = v.shape[0] + a.shape[0] - 1
    elif mode == "same":
        pad_size = halo_size
        gshape = a.shape[0]
    elif mode == "valid":
        pad_size = 0
        gshape = a.shape[0] - v.shape[0] + 1
    else:
        raise ValueError("Only {'full', 'valid', 'same'} are allowed for mode")

    a = pad(a, pad_size, "constant", 0)

    # fetch halos and store them in a.halo_next/a.halo_prev
    a.get_halo(halo_size)

    # apply halos to local array
    signal = a.array_with_halos

    # check if a local chunk is smaller than the filter size
    if a.is_distributed() and signal.size()[0] < v.shape[0]:
        raise ValueError("Local chunk size is smaller than filter size, this is not supported yet")

    # make signal and filter weight 3D for Pytorch conv1d function
    signal.unsqueeze_(0)
    signal.unsqueeze_(0)

    # flip filter for convolution as Pytorch conv1d computes correlations
    weight = v.larray.flip(dims=(0,))
    weight.unsqueeze_(0)
    weight.unsqueeze_(0)

    # cast to float if on GPU
    if signal.is_cuda:
        float_type = promote_types(signal.dtype, torch.float32).torch_type()
        signal = signal.to(float_type)
        weight = weight.to(float_type)

    # apply torch convolution operator
    signal_filtered = fc.conv1d(signal, weight)

    # unpack 3D result into 1D
    signal_filtered = signal_filtered[0, 0, :]

    # if kernel shape along split axis is even we need to get rid of duplicated values
    if a.comm.rank != 0 and v.shape[0] % 2 == 0:
        signal_filtered = signal_filtered[1:]

    return DNDarray(
        signal_filtered.contiguous(),
        (gshape,),
        signal_filtered.dtype,
        a.split,
        a.device,
        a.comm,
        balanced=False,
    ).astype(a.dtype.torch_type())
