"""Provides a collection of signal-processing operations"""

import torch
from typing import Union, Tuple, Sequence

from .communication import MPI
from .dndarray import DNDarray
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
    if v.split is not None:
        raise TypeError("Distributed filter weights are not supported")
    if len(a.shape) != 1 or len(v.shape) != 1:
        raise ValueError("Only 1-dimensional input DNDarrays are allowed")
    if a.shape[0] <= v.shape[0]:
        raise ValueError("Filter size must not be larger than signal size")
    if a.dtype is not v.dtype:
        raise TypeError("Signal and filter weight must be of same dtype")
    if mode == "same" and v.shape[0] % 2 == 0:
        raise ValueError("Mode 'same' cannot be use with even sized kernel")

    # compute halo size
    halo_size = v.shape[0] // 2 if v.shape[0] % 2 == 0 else (v.shape[0] - 1) // 2

    # fetch halos and store them in a.halo_next/a.halo_prev
    a.get_halo(halo_size)

    # apply halos to local array
    signal = (
        a.array_with_halos
    )  # torch.cat(tuple(_ for _ in (a.halo_prev, a.array, a.halo_next) if _ is not None))

    # check if a local chunk is smaller than the filter size
    if a.is_distributed() and signal.size()[0] < v.shape[0]:
        raise ValueError("Local chunk size is smaller than filter size, this is not supported yet")

    # ----- we need different cases for the first and last processes
    # rank 0:                   only pad on the left
    # rank n-1:                 only pad on the right
    # rank i: 0 < i < n-1:      no padding at all

    has_left = a.halo_prev is not None
    has_right = a.halo_next is not None

    if mode == "full":
        pad_prev = pad_next = None

        if not a.is_distributed():
            pad_prev = pad_next = torch.zeros(
                v.shape[0] - 1, dtype=a.dtype.torch_type(), device=signal.device
            )

        elif (not has_left) and has_right:  # maybe just check for rank?
            # first process, pad only left
            pad_prev = torch.zeros(v.shape[0] - 1, dtype=a.dtype.torch_type(), device=signal.device)
            pad_next = None

        elif has_left and (not has_right):
            # last process, pad only right
            pad_prev = None
            pad_next = torch.zeros(v.shape[0] - 1, dtype=a.dtype.torch_type(), device=signal.device)

        else:
            # all processes in between don't need padding
            pad_prev = pad_next = None

        gshape = v.shape[0] + a.shape[0] - 1

    elif mode == "same":
        # first and last need padding
        pad_prev = pad_next = None
        if a.comm.rank == 0:
            pad_prev = torch.zeros(halo_size, dtype=a.dtype.torch_type(), device=signal.device)
        if a.comm.rank == a.comm.size - 1:
            pad_next = torch.zeros(halo_size, dtype=a.dtype.torch_type(), device=signal.device)

        gshape = a.shape[0]

    elif mode == "valid":
        pad_prev = pad_next = None
        gshape = a.shape[0] - v.shape[0] + 1

    else:
        raise ValueError("Only {'full', 'valid', 'same'} are allowed for mode")

    # add padding to the borders according to mode
    # TODO: why not use ht.pad()?
    signal = a.genpad(signal, pad_prev, pad_next)

    # make signal and filter weight 3D for Pytorch conv1d function
    signal.unsqueeze_(0)
    signal.unsqueeze_(0)

    # flip filter for convolution as Pytorch conv1d computes correlations
    weight = v.larray.clone()
    idx = torch.LongTensor([i for i in range(weight.size(0) - 1, -1, -1)])
    weight = weight.index_select(0, idx)
    weight.unsqueeze_(0)
    weight.unsqueeze_(0)

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
