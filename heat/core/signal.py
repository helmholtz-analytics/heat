"""Provides a collection of signal-processing operations"""

import torch
from typing import Union, Tuple, Sequence

from .communication import MPI
from .dndarray import DNDarray
from .types import promote_types
from .manipulations import pad
from .factories import array
import torch.nn.functional as fc

__all__ = ["convolve", "convolve2d"]


def genpad(a, signal, pad, split, boundary, fillvalue):

    dim = len(signal.shape) - 2

    # check if more than one rank is involved
    if a.is_distributed():

        # set the padding of the first rank
        if a.comm.rank == 0:
            for i in range(dim):
                pad[1 + i * dim] = 0

        # set the padding of the last rank
        elif a.comm.rank == a.comm.size - 1:
            for i in range(dim):
                pad[i * dim] = 0

    if boundary == "fill":
        signal = fc.pad(signal, pad, mode="constant", value=fillvalue)
    elif boundary == "wrap":
        signal = fc.pad(signal, pad, mode="circular")
    elif boundary == "symm":
        signal = fc.pad(signal, pad, mode="reflect")
    else:
        raise ValueError("Only {'fill', 'wrap', 'symm'} are allowed for boundary")

    return signal

def convolve(a: DNDarray, v: DNDarray, mode: str = "full", stride: int = 1) -> DNDarray:
    """
    Returns the discrete, linear convolution of two one-dimensional `DNDarray`s or scalars.
    Unlike `numpy.signal.convolve`, if ``a`` and/or ``v`` have more than one dimension, batch-convolution along the last dimension will be attempted. See `Examples` below.

    Parameters
    ----------
    a : DNDarray or scalar
        One- or N-dimensional signal ``DNDarray`` of shape (..., N), or scalar. If ``a`` has more than one dimension, it will be treated as a batch of 1D signals.
        Distribution along the batch dimension is required for distributed batch processing. See the examples for details.
    v : DNDarray or scalar
        One- or N-dimensional filter weight `DNDarray` of shape (..., M), or scalar. If ``v`` has more than one dimension, it will be treated as a batch of 1D filter weights.
        The batch dimension(s) of ``v`` must match the batch dimension(s) of ``a``.
    mode : str
        Can be 'full', 'valid', or 'same'. Default is 'full'.
        'full':
          Returns the convolution at
          each point of overlap, with a length of '(N+M-2)//stride+1'. At
          the end-points of the convolution, the signals do not overlap
          completely, and boundary effects may be seen.
        'same':
          Mode 'same' returns output  of length 'N'. Boundary
          effects are still visible. This mode is not supported for
          even-sized filter weights
        'valid':
          Mode 'valid' returns output of length '(N-M)//stride+1'. The
          convolution product is only given for points where the signals
          overlap completely. Values outside the signal boundary have no
          effect.
    stride : int
        Stride of the convolution. Must be a positive integer. Default is 1.
        Stride must be 1 for mode 'same'.

    Examples
    --------
    Note how the convolution operator flips the second array
    before "sliding" the two across one another:

    >>> a = ht.ones(5)
    >>> v = ht.arange(3).astype(ht.float)
    >>> ht.convolve(a, v, mode="full")
    DNDarray([0., 1., 3., 3., 3., 3., 2.])
    >>> ht.convolve(a, v, mode="same")
    DNDarray([1., 3., 3., 3., 3.])
    >>> ht.convolve(a, v, mode="valid")
    DNDarray([3., 3., 3.])
    >>> ht.convolve(a, v, stride=2)
    DNDarray([0., 3., 3., 2.])
    >>> ht.convolve(a, v, mode="valid", stride=2)
    DNDarray([3., 3.])

    >>> a = ht.ones(10, split=0)
    >>> v = ht.arange(3, split=0).astype(ht.float)
    >>> ht.convolve(a, v, mode="valid")
    DNDarray([3., 3., 3., 3., 3., 3., 3., 3.])

    [0/3] DNDarray([3., 3., 3.])
    [1/3] DNDarray([3., 3., 3.])
    [2/3] DNDarray([3., 3.])

    >>> a = ht.ones(10, split=0)
    >>> v = ht.arange(3, split=0)
    >>> ht.convolve(a, v)
    DNDarray([0., 1., 3., 3., 3., 3., 3., 3., 3., 3., 3., 2.], dtype=ht.float32, device=cpu:0, split=0)

    [0/3] DNDarray([0., 1., 3., 3.])
    [1/3] DNDarray([3., 3., 3., 3.])
    [2/3] DNDarray([3., 3., 3., 2.])

    >>> a = ht.arange(50, dtype=ht.float64, split=0)
    >>> a = a.reshape(10, 5)  # 10 signals of length 5
    >>> v = ht.arange(3)
    >>> ht.convolve(a, v)  # batch processing: 10 signals convolved with filter v
    DNDarray([[  0.,   0.,   1.,   4.,   7.,  10.,   8.],
          [  0.,   5.,  16.,  19.,  22.,  25.,  18.],
          [  0.,  10.,  31.,  34.,  37.,  40.,  28.],
          [  0.,  15.,  46.,  49.,  52.,  55.,  38.],
          [  0.,  20.,  61.,  64.,  67.,  70.,  48.],
          [  0.,  25.,  76.,  79.,  82.,  85.,  58.],
          [  0.,  30.,  91.,  94.,  97., 100.,  68.],
          [  0.,  35., 106., 109., 112., 115.,  78.],
          [  0.,  40., 121., 124., 127., 130.,  88.],
          [  0.,  45., 136., 139., 142., 145.,  98.]], dtype=ht.float64, device=cpu:0, split=0)

    >>> v = ht.random.randint(0, 3, (10, 3), split=0)  # 10 filters of length 3
    >>> ht.convolve(a, v)  # batch processing: 10 signals convolved with 10 filters
    DNDarray([[  0.,   0.,   2.,   4.,   6.,   8.,   0.],
            [  5.,   6.,   7.,   8.,   9.,   0.,   0.],
            [ 20.,  42.,  56.,  61.,  66.,  41.,  14.],
            [  0.,  15.,  16.,  17.,  18.,  19.,   0.],
            [ 20.,  61.,  64.,  67.,  70.,  48.,   0.],
            [ 50.,  52., 104., 108., 112.,  56.,  58.],
            [  0.,  30.,  61.,  63.,  65.,  67.,  34.],
            [ 35., 106., 109., 112., 115.,  78.,   0.],
            [  0.,  40.,  81.,  83.,  85.,  87.,  44.],
            [  0.,   0.,  45.,  46.,  47.,  48.,  49.]], dtype=ht.float64, device=cpu:0, split=0)
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
    if a.larray.is_mps and promoted_type == float64:
        # cannot cast to float64 on MPS
        promoted_type = float32

    a = a.astype(promoted_type)
    v = v.astype(promoted_type)

    # check if the filter is longer than the signal and swap them if necessary
    if v.shape[-1] > a.shape[-1]:
        a, v = v, a

    # assess whether to perform batch processing, default is False (no batch processing)
    batch_processing = False
    if a.ndim > 1:
        # batch processing requires 1D filter OR matching batch dimensions for signal and filter
        batch_dims = a.shape[:-1]
        # verify that the filter shape is consistent with the signal
        if v.ndim > 1:
            if v.shape[:-1] != batch_dims:
                raise ValueError(
                    f"Batch dimensions of signal and filter must match. Signal: {a.shape}, Filter: {v.shape}"
                )
        if a.is_distributed():
            if a.split == a.ndim - 1:
                raise ValueError(
                    "Please distribute the signal along the batch dimension, not the signal dimension. For in-place redistribution use the `DNDarray.resplit_()` method with `axis=0`"
                )
            if v.is_distributed():
                if v.ndim == 1:
                    # gather filter to all ranks
                    v.resplit_(axis=None)
                else:
                    v.resplit_(axis=a.split)
        batch_processing = True

    if not batch_processing and v.ndim > 1:
        raise ValueError(
            f"1-D convolution only supported for 1-dimensional signal and kernel. Signal: {a.shape}, Filter: {v.shape}"
        )

    # check mode and stride for value errors
    if stride < 1:
        raise ValueError("Stride must be at positive integer")
    if stride > 1 and mode == "same":
        raise ValueError("Stride must be 1 for mode 'same'")

    if mode == "same" and v.shape[-1] % 2 == 0:
        raise ValueError("Mode 'same' cannot be used with even-sized kernel")
    if not v.is_balanced():
        raise ValueError("Only balanced kernel weights are allowed")

    # calculate pad size according to mode
    if mode == "full":
        pad_size = v.shape[-1] - 1
    elif mode == "same":
        pad_size = v.shape[-1] // 2
    elif mode == "valid":
        pad_size = 0
    else:
        raise ValueError(f"Supported modes are 'full', 'valid', 'same', got {mode}")

    gshape = (a.shape[-1] + 2 * pad_size - v.shape[-1]) // stride + 1

    if v.is_distributed() and stride > 1:
        gshape_stride_1 = a.shape[-1] + 2 * pad_size - v.shape[-1] + 1

    if batch_processing:
        # all operations are local torch operations, only the last dimension is convolved
        local_a = a.larray
        local_v = v.larray

        # flip filter for convolution, as Pytorch conv1d computes correlations
        local_v = torch.flip(local_v, [-1])
        local_batch_dims = tuple(local_a.shape[:-1])

        # reshape signal and filter to 3D for Pytorch conv1d function
        # see https://pytorch.org/docs/stable/generated/torch.nn.functional.conv1d.html
        local_a = local_a.reshape(
            torch.prod(torch.tensor(local_batch_dims, device=local_a.device), dim=0).item(),
            local_a.shape[-1],
        )
        channels = local_a.shape[0]
        if v.ndim > 1:
            local_v = local_v.reshape(
                torch.prod(torch.tensor(local_batch_dims, device=local_v.device), dim=0).item(),
                local_v.shape[-1],
            )
            local_v = local_v.unsqueeze(1)
        else:
            local_v = local_v.unsqueeze(0).unsqueeze(0).expand(local_a.shape[0], 1, -1)
        # add batch dimension to signal
        local_a = local_a.unsqueeze(0)

        # cast to single-precision float if on GPU
        if local_a.is_cuda:
            float_type = torch.promote_types(local_a.dtype, torch.float32)
            local_a = local_a.to(float_type)
            local_v = local_v.to(float_type)

        # apply torch convolution operator if local signal isn't empty
        if torch.prod(torch.tensor(local_a.shape, device=local_a.device)) > 0:
            local_convolved = fc.conv1d(
                local_a, local_v, padding=pad_size, groups=channels, stride=stride
            )
        else:
            empty_shape = tuple(local_a.shape[:-1] + (gshape,))
            local_convolved = torch.empty(empty_shape, dtype=local_a.dtype, device=local_a.device)

        # unpack 3D result into original shape
        local_convolved = local_convolved.squeeze(0)
        local_convolved = local_convolved.reshape(local_batch_dims + (gshape,))

        # wrap result in DNDarray
        convolved = array(local_convolved, is_split=a.split, device=a.device, comm=a.comm)
        return convolved

    # pad signal with zeros
    a = pad(a, pad_size, "constant", 0)

    # compute halo size
    halo_size = torch.max(v.lshape_map[:, -1]).item() // 2

    if a.is_distributed():
        if (v.lshape_map[:, 0] > a.lshape_map[:, 0]).any():
            raise ValueError(
                "Local chunk of filter weight is larger than the local chunks of signal"
            )
        # fetch halos and store them in a.halo_next/a.halo_prev
        a.get_halo(halo_size)
        # apply halos to local array
        signal = a.array_with_halos

        # shift signal based on global kernel starts for any rank but first
        if stride > 1 and not v.is_distributed():
            if a.comm.rank == 0:
                local_index = 0
            else:
                local_index = torch.sum(a.lshape_map[: a.comm.rank, 0]).item() - halo_size
                local_index = local_index % stride

                if local_index != 0:
                    local_index = stride - local_index

                # even kernels can produces doubles
                if v.shape[-1] % 2 == 0 and local_index == 0:
                    local_index = stride

            signal = signal[local_index:]
    else:
        signal = a.larray

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

        # any stride is a subset of stride 1
        if stride > 1:
            gshape = gshape_stride_1

        for r in range(size):
            rec_v = t_v.clone()
            v.comm.Bcast(rec_v, root=r)
            t_v1 = rec_v.reshape(1, 1, rec_v.shape[0])
            local_signal_filtered = fc.conv1d(signal, t_v1, stride=1)
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
            try:
                signal_filtered += global_signal_filtered[start_idx : start_idx + gshape]
            except (ValueError, TypeError):
                signal_filtered = (
                    signal_filtered + global_signal_filtered[start_idx : start_idx + gshape]
                )
            if r != size - 1:
                start_idx += v.lshape_map[r + 1][0].item()

        # any stride is a subset of arrays of stride 1
        if stride > 1:
            signal_filtered = signal_filtered[::stride]

        return signal_filtered

    else:
        # apply torch convolution operator
        if signal.shape[-1] >= weight.shape[-1]:
            signal_filtered = fc.conv1d(signal, weight, stride=stride)

            # unpack 3D result into 1D
            signal_filtered = signal_filtered[0, 0, :]
        else:
            signal_filtered = torch.tensor([], device=str(signal.device))

        # if kernel shape along split axis is even we need to get rid of duplicated values
        if a.comm.rank != 0 and v.shape[0] % 2 == 0 and stride == 1:
            signal_filtered = signal_filtered[1:]

        return DNDarray(
            signal_filtered,
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

    if v.is_distributed():
        raise TypeError("Distributed filter weights are not supported")
    if len(a.shape) != 2 or len(v.shape) != 2:
        raise ValueError("Only 2-dimensional input DNDarrays are allowed")
    if a.shape[0] <= v.shape[0] or a.shape[1] <= v.shape[1]:
        raise ValueError("Filter size must not be greater than or equal to signal size")
    if mode == "same" and v.shape[0] % 2 == 0:
        raise ValueError("Mode 'same' cannot be used with even-sized kernel")

    # compute halo size
    if a.split == 0 or a.split == None:
        halo_size = v.shape[0] // 2
    else:
        halo_size = v.shape[1] // 2

    # fetch halos and store them in a.halo_next/a.halo_prev
    a.get_halo(halo_size)

    # apply halos to local array
    signal = a.array_with_halos.clone()

    # check if a local chunk is smaller than the filter size
    if a.is_distributed() and signal.size()[0] < v.shape[0]:
        raise ValueError("Local chunk size is smaller than filter size, this is not supported yet")

    if mode == "full":
        pad_0 = v.shape[0] - 1
        pad_1 = v.shape[1] - 1
        gshape_0 = v.shape[0] + a.shape[0] - 1
        gshape_1 = v.shape[1] + a.shape[1] - 1
        pad = list((pad_0, pad_0, pad_1, pad_1))
        gshape = list((gshape_0, gshape_1))

    elif mode == "same":
        pad = list((halo_size,) * 4)
        gshape = a.shape[0]

    elif mode == "valid":
        pad = list((0,) * 4)
        gshape_0 = a.shape[0] - v.shape[0] + 1
        gshape_1 = a.shape[1] - v.shape[1] + 1
        gshape = list((gshape_0, gshape_1))

    else:
        raise ValueError("Only {'full', 'valid', 'same'} are allowed for mode")

    # make signal and filter weight 4D for Pytorch conv2d function
    signal = signal.reshape(1, 1, signal.shape[0], signal.shape[1])

    # add padding to the borders according to mode
    signal = genpad(a, signal, pad, a.split, boundary, fillvalue)

    # flip filter for convolution as PyTorch conv2d computes correlation
    weight = torch.flip(v._DNDarray__array.clone(), [0, 1])
    weight = weight.reshape(1, 1, weight.shape[0], weight.shape[1])

    # apply torch convolution operator
    signal_filtered = fc.conv2d(signal, weight)

    # unpack 3D result into 1D
    signal_filtered = signal_filtered[0, 0, :]

    # if kernel shape along split axis is even we need to get rid of duplicated values
    if a.comm.rank != 0 and v.shape[0] % 2 == 0:
        signal_filtered = signal_filtered[1:, 1:]

    return DNDarray(
        signal_filtered.contiguous(),
        (gshape,),
        signal_filtered.dtype,
        a.split,
        a.device,
        a.comm,
        a.balanced,
    ).astype(a.dtype.torch_type())
