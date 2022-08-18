"""Provides a collection of signal-processing operations"""

import torch
from typing import Union, Tuple, Sequence

from .communication import MPI
from .dndarray import DNDarray
from .types import promote_types
from .manipulations import pad, flip
from .factories import array, zeros
import torch.nn.functional as fc

__all__ = ["convolve", "convolve2d"]


def convgenpad(a, signal, pad, boundary, fillvalue):
    """
    Adds padding to local PyTorch tensors considering the distributed scheme of the overlying DNDarray.

    Parameters
    ----------
    a : DNDarray
        Overlying N-dimensional `DNDarray` signal
    signal : torch.Tensor
        Local Pytorch tensors to be padded
    pad: list
        list containing paddings per dimensions
    boundary: str{‘fill’, ‘wrap’, ‘symm’}, optional
        A flag indicating how to handle boundaries:
        'fill':
         pad input arrays with fillvalue. (default)
        'wrap':
         circular boundary conditions.
        'symm':
         symmetrical boundary conditions.
    fillvalue: scalar, optional
         Value to fill pad input arrays with. Default is 0.
    """
    dim = len(signal.shape) - 2
    dime = 2 * dim - 1
    dimz = 2 * dim - 2
    # check if more than one rank is involved
    if a.is_distributed() and a.split is not None:

        # set the padding of the first rank
        if a.comm.rank == 0:
            pad[dime - 2 * a.split] = 0
        # set the padding of the last rank
        elif a.comm.rank == a.comm.size - 1:
            pad[dimz - 2 * a.split] = 0
        else:
            pad[dime - 2 * a.split] = 0
            pad[dimz - 2 * a.split] = 0

    if boundary == "fill":
        signal = fc.pad(signal, pad, mode="constant", value=fillvalue)
    elif boundary == "wrap":
        signal = fc.pad(signal, pad, mode="circular")
    elif boundary == "symm":
        signal = fc.pad(signal, pad, mode="reflect")
    else:
        raise ValueError("Only {'fill', 'wrap', 'symm'} are allowed for boundary")

    return signal


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
        Can be 'full', 'valid', or 'same'.
        'full':
          Returns the convolution at
          each point of overlap, with an output shape of (N+M-1,). At
          the end-points of the convolution, the signals do not overlap
          completely, and boundary effects may be seen. Default is 'full'
        'same':
          Mode 'same' returns output  of length 'N'. Boundary
          effects are still visible. This mode is not supported for
          even-sized filter weights
        'valid':
          Mode 'valid' returns output of length 'N-M+1'. The
          convolution product is only given for points where the signals
          overlap completely. Values outside the signal boundary have no
          effect.
    fillvalue: scalar, optional
         Value to fill pad input arrays with. Default is 0.

    Notes
    -----
        Contrary to the original `numpy.convolve`, this function does not
        swap the input arrays if the second one is larger than the first one.
        This is because `a`, the signal, might be memory-distributed,
        whereas the filter `v` is assumed to be non-distributed,
        i.e. a copy of `v` will reside on each process.


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
    """
    if not isinstance(a, DNDarray):
        try:
            a = array(a)
        except TypeError:
            raise TypeError("non-supported type for signal: {}".format(type(a)))
    if not isinstance(v, DNDarray):
        try:
            v = array(v)
        except TypeError:
            raise TypeError("non-supported type for filter: {}".format(type(v)))
    promoted_type = promote_types(a.dtype, v.dtype)
    a = a.astype(promoted_type)
    v = v.astype(promoted_type)

    if v.is_distributed():
        raise TypeError("Distributed filter weights are not supported")
    if len(a.shape) != 1 or len(v.shape) != 1:
        raise ValueError("Only 1-dimensional input DNDarrays are allowed")
    if a.shape[0] <= v.shape[0]:
        raise ValueError("Filter size must not be greater than or equal to signal size")
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
        raise ValueError("Supported modes are 'full', 'valid', 'same', got {}".format(mode))

    print("signal2: ", a.lshape, a.comm.rank)

    a = pad(a, pad_size, "constant", 0)

    if a.is_distributed():
        if (v.shape[0] > a.lshape_map[:, 0]).any():
            raise ValueError("Filter weight is larger than the local chunks of signal")
        # fetch halos and store them in a.halo_next/a.halo_prev
        a.get_halo(halo_size)
        # apply halos to local array
        signal = a.array_with_halos
    else:
        signal = a.larray

    # make signal and filter weight 3D for Pytorch conv1d function
    signal = signal.reshape(1, 1, signal.shape[0])

    # flip filter for convolution as Pytorch conv1d computes correlations
    weight = v.larray.flip(dims=(0,))
    weight = weight.reshape(1, 1, weight.shape[0])

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


def convolve2d(a, v, mode="full", boundary="fill", fillvalue=0):
    """
    Returns the discrete, linear convolution of two two-dimensional HeAT tensors.

    Parameters
    ----------
    a : (N,) ht.tensor
        Two-dimensional signal HeAT tensor
    v : (M,) ht.tensor
        Two-dimensional filter weight HeAT tensor.
    mode : {'full', 'valid', 'same'}, optional
        'full':
          By default, mode is 'full'. This returns the convolution at
          each point of overlap, with an output shape of (N+M-1,). At
          the end-points of the convolution, the signals do not overlap
          completely, and boundary effects may be seen.
        'same':
          Mode 'same' returns output  of length 'N'. Boundary
          effects are still visible. This mode is not supported for
          even sized filter weights
        'valid':
          Mode 'valid' returns output of length 'N-M+1'. The
          convolution product is only given for points where the signals
          overlap completely. Values outside the signal boundary have no
          effect.
    boundary: str{‘fill’, ‘wrap’, ‘symm’}, optional
        A flag indicating how to handle boundaries:
        'fill':
         pad input arrays with fillvalue. (default)
        'wrap':
         circular boundary conditions.
        'symm':
         symmetrical boundary conditions.
    fillvalue: scalar, optional
         Value to fill pad input arrays with. Default is 0.

    Returns
    -------
    out : ht.tensor
        Discrete, linear convolution of 'a' and 'v'.

    Note : There is  differences to the numpy convolve function:
        The inputs are not swapped if v is larger than a. The reason is that v needs to be
        non-splitted. This should not influence performance. If the filter weight is larger
        than fitting into memory, using the FFT for convolution is recommended.

    Example
    --------
    >>> a = ht.ones((5, 5))
    >>> v = ht.ones((3, 3))
    >>> ht.convolve2d(a, v, mode='valid')
    DNDarray([[9., 9., 9.],
              [9., 9., 9.],
              [9., 9., 9.]], dtype=ht.float32, device=cpu:0, split=None)
    """
    if not isinstance(a, DNDarray):
        try:
            a = array(a)
        except TypeError:
            raise TypeError("non-supported type for signal: {}".format(type(a)))
    if not isinstance(v, DNDarray):
        try:
            v = array(v)
        except TypeError:
            raise TypeError("non-supported type for filter: {}".format(type(v)))
    promoted_type = promote_types(a.dtype, v.dtype)
    a = a.astype(promoted_type)
    v = v.astype(promoted_type)

    if v.is_distributed() and (mode == "full" or mode == "same"):
        raise TypeError("Distributed filter weights only supportes valid mode")
    if len(a.shape) != 2 or len(v.shape) != 2:
        raise ValueError("Only 2-dimensional input DNDarrays are allowed")
    if a.shape[0] <= v.shape[0] or a.shape[1] <= v.shape[1]:
        raise ValueError("Filter size must not be greater than or equal to signal size")
    if mode == "same" and v.shape[0] % 2 == 0:
        raise ValueError("Mode 'same' cannot be used with even-sized kernel")

    # compute halo size
    if a.split == 0 or a.split == None:
        halo_size = int(v.lshape_map[0][0]) // 2
    else:
        halo_size = int(v.lshape_map[0][1]) // 2

    # fetch halos and store them in a.halo_next/a.halo_prev
    print("qqa: ", halo_size)
    a.get_halo(halo_size)

    # apply halos to local array
    signal = a.array_with_halos

    # check if a local chunk is smaller than the filter size
    if a.is_distributed() and signal.size()[0] < v.lshape_map[0][0]:
        raise ValueError("Local signal chunk size is smaller than the local filter size.")

    if mode == "full":
        pad_0 = v.shape[0] - 1
        pad_1 = v.shape[1] - 1
        gshape_0 = v.shape[0] + a.shape[0] - 1
        gshape_1 = v.shape[1] + a.shape[1] - 1
        pad = list((pad_0, pad_0, pad_1, pad_1))
        gshape = (gshape_0, gshape_1)

    elif mode == "same":
        pad = list((halo_size,) * 4)
        gshape = (a.shape[0], a.shape[1])

    elif mode == "valid":
        pad = list((0,) * 4)
        gshape_0 = a.shape[0] - v.shape[0] + 1
        gshape_1 = a.shape[1] - v.shape[1] + 1
        gshape = (gshape_0, gshape_1)

    else:
        raise ValueError("Only {'full', 'valid', 'same'} are allowed for mode")

    # make signal and filter weight 4D for Pytorch conv2d function
    signal = signal.reshape(1, 1, signal.shape[0], signal.shape[1])

    # add padding to the borders according to mode
    signal = convgenpad(a, signal, pad, boundary, fillvalue)

    # flip filter for convolution as PyTorch conv2d computes correlation
    v = flip(v, [0, 1])

    # compute weight size
    if a.split == 0 or a.split == None:
        weight_size = int(v.lshape_map[0][0])
        current_size = v.larray.shape[0]
    else:
        weight_size = int(v.lshape_map[0][1])
        current_size = v.larray.shape[1]

    if current_size != weight_size:
        weight_shape = (int(v.lshape_map[0][0]), int(v.lshape_map[0][1]))
        target = torch.zeros(weight_shape, dtype=v.larray.dtype, device=v.larray.device)
        pad_size = weight_size - current_size
        if v.split == 0:
            target[pad_size:] = v.larray
        else:
            target[:, pad_size:] = v.larray
        weight = target
    else:
        weight = v.larray

    t_v = weight  # stores temporary weight
    weight = weight.reshape(1, 1, weight.shape[0], weight.shape[1])

    if v.is_distributed():
        size = v.comm.size
        split_axis = v.split
        for r in range(size):
            rec_v = v.comm.bcast(t_v, root=r)
            t_v1 = rec_v.reshape(1, 1, rec_v.shape[0], rec_v.shape[1])

            # apply torch convolution operator
            local_signal_filtered = fc.conv2d(signal, t_v1)

            # unpack 3D result into 1D
            local_signal_filtered = local_signal_filtered[0, 0, :]

            # if kernel shape along split axis is even we need to get rid of duplicated values
            if a.comm.rank != 0 and weight_size % 2 == 0 and a.split == 0:
                local_signal_filtered = local_signal_filtered[1:, :]
            if a.comm.rank != 0 and weight_size % 2 == 0 and a.split == 1:
                local_signal_filtered = local_signal_filtered[:, 1:]

            # accumulate filtered signal on the fly
            global_signal_filtered = array(
                local_signal_filtered, is_split=split_axis, device=a.device, comm=a.comm
            )
            if r == 0:
                # initialize signal_filtered, starting point of slice
                signal_filtered = zeros(
                    gshape, dtype=a.dtype, split=a.split, device=a.device, comm=a.comm
                )
                start_idx = 0

            # accumulate relevant slice of filtered signal
            # note, this is a binary operation between unevenly distributed dndarrays and will require communication, check out _operations.__binary_op()
            if split_axis == 0:
                signal_filtered += global_signal_filtered[start_idx : start_idx + gshape[0]]
            else:
                signal_filtered += global_signal_filtered[:, start_idx : start_idx + gshape[1]]
            if r != size - 1:
                start_idx += v.lshape_map[r + 1][split_axis]

        signal_filtered.balance()
        return signal_filtered

    else:

        # apply torch convolution operator
        signal_filtered = fc.conv2d(signal, weight)

        # unpack 3D result into 1D
        signal_filtered = signal_filtered[0, 0, :]

        # if kernel shape along split axis is even we need to get rid of duplicated values
        if a.comm.rank != 0 and v.lshape_map[0][0] % 2 == 0 and a.split == 0:
            signal_filtered = signal_filtered[1:, :]
        elif a.comm.rank != 0 and v.lshape_map[0][1] % 2 == 0 and a.split == 1:
            signal_filtered = signal_filtered[:, 1:]

        result = DNDarray(
            signal_filtered.contiguous(),
            gshape,
            signal_filtered.dtype,
            a.split,
            a.device,
            a.comm,
            a.balanced,
        ).astype(a.dtype.torch_type())

        if mode == "full" or mode == "valid":
            result.balance()

        return result
