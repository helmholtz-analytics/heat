"""Provides a collection of signal-processing operations"""

import torch
import numpy as np

from .communication import MPI
from .dndarray import DNDarray
from .types import promote_types
from .manipulations import pad, flip
from .factories import array, zeros
import torch.nn.functional as fc

__all__ = ["convolve"]


def convolve(a: DNDarray, v: DNDarray, mode: str = "full") -> DNDarray:
    """
    Returns the discrete, linear convolution of two one-dimensional `DNDarray`s or scalars.

    Parameters
    ----------
    a : DNDarray or scalar
        One-dimensional signal `DNDarray` of shape (N,) or scalar.
    v : DNDarray or scalar
        One-dimensional filter weight `DNDarray` of shape (M,) or scalar.
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

    Examples
    --------
    Note how the convolution operator flips the second array
    before "sliding" the two across one another:

    >>> a = ht.ones(10)
    >>> v = ht.arange(3).astype(ht.float)
    >>> ht.convolve(a, v, mode='full')
    DNDarray([0., 1., 3., 3., 3., 3., 2.])
    >>> ht.convolve(a, v, mode='same')
    DNDarray([1., 3., 3., 3., 3.])
    >>> ht.convolve(a, v, mode='valid')
    DNDarray([3., 3., 3.])
    >>> a = ht.ones(10, split = 0)
    >>> v = ht.arange(3, split = 0).astype(ht.float)
    >>> ht.convolve(a, v, mode='valid')
    DNDarray([3., 3., 3., 3., 3., 3., 3., 3.])

    [0/3] DNDarray([3., 3., 3.])
    [1/3] DNDarray([3., 3., 3.])
    [2/3] DNDarray([3., 3.])
    >>> a = ht.ones(10, split = 0)
    >>> v = ht.arange(3, split = 0)
    >>> ht.convolve(a, v)
    DNDarray([0., 1., 3., 3., 3., 3., 3., 3., 3., 3., 3., 2.], dtype=ht.float32, device=cpu:0, split=0)

    [0/3] DNDarray([0., 1., 3., 3.])
    [1/3] DNDarray([3., 3., 3., 3.])
    [2/3] DNDarray([3., 3., 3., 2.])
    """
    if np.isscalar(a):
        a = array([a])
    if np.isscalar(v):
        v = array([v])
    if not isinstance(a, DNDarray):
        try:
            a = array(a)
        except TypeError:
            raise TypeError(f"non-supported type for signal: {type(a)}")
    if not isinstance(v, DNDarray):
        try:
            v = array(v)
        except TypeError:
            raise TypeError(f"non-supported type for filter: {type(v)}")
    promoted_type = promote_types(a.dtype, v.dtype)
    a = a.astype(promoted_type)
    v = v.astype(promoted_type)

    if len(a.shape) != 1 or len(v.shape) != 1:
        raise ValueError("Only 1-dimensional input DNDarrays are allowed")
    if mode == "same" and v.shape[0] % 2 == 0:
        raise ValueError("Mode 'same' cannot be used with even-sized kernel")
    if not v.is_balanced():
        raise ValueError("Only balanced kernel weights are allowed")

    if v.shape[0] > a.shape[0]:
        a, v = v, a

    # compute halo size
    halo_size = torch.max(v.lshape_map[:, 0]).item() // 2

    # pad DNDarray with zeros according to mode
    if mode == "full":
        pad_size = v.shape[0] - 1
        gshape = v.shape[0] + a.shape[0] - 1
    elif mode == "same":
        pad_size = v.shape[0] // 2
        gshape = a.shape[0]
    elif mode == "valid":
        pad_size = 0
        gshape = a.shape[0] - v.shape[0] + 1
    else:
        raise ValueError(f"Supported modes are 'full', 'valid', 'same', got {mode}")

    a = pad(a, pad_size, "constant", 0)

    if a.is_distributed():
        if (v.lshape_map[:, 0] > a.lshape_map[:, 0]).any():
            raise ValueError(
                "Local chunk of filter weight is larger than the local chunks of signal"
            )
        # fetch halos and store them in a.halo_next/a.halo_prev
        a.get_halo(halo_size)
        # apply halos to local array
        signal = a.array_with_halos
    else:
        signal = a.larray

    # flip filter for convolution as Pytorch conv1d computes correlations
    v = flip(v, [0])
    if v.larray.shape != v.lshape_map[0]:
        # pads weights if input kernel is uneven
        target = torch.zeros(v.lshape_map[0][0], dtype=v.larray.dtype, device=v.larray.device)
        pad_size = v.lshape_map[0][0] - v.larray.shape[0]
        target[pad_size:] = v.larray
        weight = target
    else:
        weight = v.larray

    t_v = weight  # stores temporary weight

    # make signal and filter weight 3D for Pytorch conv1d function
    signal = signal.reshape(1, 1, signal.shape[0])
    weight = weight.reshape(1, 1, weight.shape[0])

    # cast to float if on GPU
    if signal.is_cuda:
        float_type = promote_types(signal.dtype, torch.float32).torch_type()
        signal = signal.to(float_type)
        weight = weight.to(float_type)
        t_v = t_v.to(float_type)

    if v.is_distributed():
        size = v.comm.size

        for r in range(size):
            rec_v = t_v.clone()
            v.comm.Bcast(rec_v, root=r)
            t_v1 = rec_v.reshape(1, 1, rec_v.shape[0])
            local_signal_filtered = fc.conv1d(signal, t_v1)
            # unpack 3D result into 1D
            local_signal_filtered = local_signal_filtered[0, 0, :]

            if a.comm.rank != 0 and v.lshape_map[0][0] % 2 == 0:
                local_signal_filtered = local_signal_filtered[1:]

            # accumulate filtered signal on the fly
            global_signal_filtered = array(
                local_signal_filtered, is_split=0, device=a.device, comm=a.comm
            )
            if r == 0:
                # initialize signal_filtered, starting point of slice
                signal_filtered = zeros(
                    gshape, dtype=a.dtype, split=a.split, device=a.device, comm=a.comm
                )
                start_idx = 0

            # accumulate relevant slice of filtered signal
            # note, this is a binary operation between unevenly distributed dndarrays and will require communication, check out _operations.__binary_op()
            signal_filtered += global_signal_filtered[start_idx : start_idx + gshape]
            if r != size - 1:
                start_idx += v.lshape_map[r + 1][0].item()
        return signal_filtered

    else:
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
