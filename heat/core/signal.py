"""Provides a collection of signal-processing operations"""

import torch
import numpy as np

from .communication import MPI
from .dndarray import DNDarray
from .types import promote_types, float32, float64
from .manipulations import pad, flip
from .factories import array, zeros
import torch.nn.functional as fc
from perun import monitor

__all__ = ["convolve", "convolve2d"]


def conv_pad(a, convolution_dim, signal, pad, boundary, fillvalue):
    """
    Adds padding to local PyTorch tensors considering the distributed scheme of the overlying DNDarray.

    Parameters
    ----------
    a : DNDarray
        Overlying N-dimensional `DNDarray` signal
    signal : torch.Tensor
        Local Pytorch tensors to be padded
    convolution_dim : int
        Number of dimension along which convolution will be applied
    pad: list
        list containing paddings per dimensions same order as appearing in a (opposite to pytorch),
        2 values per dimension
    boundary: str{‘constant’, ‘circular’, ‘reflect’}, optional
        A flag indicating how to handle boundaries:
        'constant':
         pad input arrays with fillvalue. (default)
        'circular':
         periodic boundary conditions.
        'reflect':
         reflect along border to pad area
         'replicate':
         copy border into pad area
    fillvalue: scalar, optional
         Value to fill pad input arrays with. Default is 0.
    """
    # check if more than one rank is involved

    if a.is_distributed() and a.comm.size > 1:
        # print(a.comm.rank, "check more than one rank")
        dim_split = a.split - a.ndim

        if boundary == "reflect" and dim_split >= -1 * convolution_dim:
            # print("check boundary reflect error")
            # print("Condition 1a:", pad[2*dim_split])
            # print("Condition 1b:", lshape_map[0, split])
            if (pad[2 * dim_split] >= a.lshape_map[0, a.split]) or (
                pad[2 * dim_split + 1] >= a.lshape_map[-1, a.split]
            ):
                # print("I caused a value error in boundary reflect")
                raise ValueError(
                    "Local chunk needs to be larger than padding for boundary mode reflect"
                )
            # print("no value error in boundary reflect")
        # check if split along a convolution dimension

        if dim_split >= -1 * convolution_dim:
            # print(a.comm.rank, "split along convolution dim")
            if boundary == "circular":
                raise ValueError(
                    "Circular boundary for distributed signals in padding dimensions is currently not supported."
                )
            # set the padding of the first rank
            if a.comm.rank == 0:
                pad[2 * dim_split + 1] = 0
            # set the padding of the last rank
            elif a.comm.rank == a.comm.size - 1:
                pad[2 * dim_split] = 0
            else:
                pad[2 * dim_split + 1] = 0
                pad[2 * dim_split] = 0

    # print(a.comm.rank, "pad", pad)
    # rearrange pad for torch
    if convolution_dim == 2:
        pad = [pad[-2], pad[-1], pad[-4], pad[-3]]
    elif convolution_dim == 3:
        pad = [pad[-2], pad[-1], pad[-4], pad[-3], pad[-6], pad[-5]]
    # print(a.comm.rank, "rearranged pad", pad)

    if boundary == "constant":
        signal = fc.pad(signal, pad, mode=boundary, value=fillvalue)
    elif boundary in ("circular", "reflect", "replicate"):
        print(a.comm.rank, "boundary not constant")
        signal = fc.pad(signal, pad, mode=boundary)
    else:
        raise ValueError(
            f"Supported boundaries are 'constant', 'circular', 'reflect' and 'replicate', got {boundary}"
        )

    return signal


def conv_input_check(a, v, stride, mode, convolution_dim=1):
    """
    Check and preprocess input data.

    Parameters
    ----------
    a : scalar, array_like, DNDarray
        Input signal data.
    v : scalar, array_like, DNDarray
        Input filter mask.
    stride : scalar, tuple
        Stride along each axis convolution is applied.
    mode : str
        Convolution mode "full", "same" or "valid"
    convolution_dim : int, optional
        Number of dimension along which convolution will be applied, affects what input_check looks for. Default 1

    Returns
    -------
    tuple
        A tuple containing the processed input signal 'a' and filter mask 'v'.

    Raises
    ------
    TypeError
        If 'a' or 'v' have unsupported data types.

    Description
    -----------
    This function takes two inputs, 'a' (signal data) and 'v' (filter mask), and performs the following checks and
    preprocessing steps:

    1. Check if 'a' and 'v' are scalars. If they are, convert them into DNDarray arrays.

    2. Check if 'a' and 'v' are instances of the 'DNDarray' class. If not, attempt to convert them into DNDarray arrays.
       If conversion is not possible, raise a TypeError.

    3. Determine the promoted data type for 'a' and 'v' based on their existing data types. Convert 'a' and 'v' to this
       promoted data type to ensure consistent data types.

    4. Check if filter is smaller or equal signal, flip if necessary

    5. Check mode and check mode "same" against even sized kernels

    6. Check stride for negative entries and against mode

    7. Return a tuple containing the processed 'a' and 'v'.
    """
    # Check if 'a' is a scalar and convert to a DNDarray if necessary
    if np.isscalar(a):
        a = array([[a]])
        while a.ndim > convolution_dim:
            a = a.squeeze(-1)

    # Check if 'v' is a scalar and convert to a DNDarray if necessary
    if np.isscalar(v):
        v = array([[v]])
        while v.ndim > convolution_dim:
            v = v.squeeze(-1)

    # Check if 'a' is not an instance of DNDarray and try to convert it to a DNDarray array
    if not isinstance(a, DNDarray):
        try:
            a = array(a)
        except TypeError:
            raise TypeError(f"non-supported type for signal: {type(a)}")

    # Check if 'v' is not an instance of DNDarray and try to convert it to a NumPy array
    if not isinstance(v, DNDarray):
        try:
            v = array(v)
        except TypeError:
            raise TypeError(f"non-supported type for filter: {type(v)}")

    # Check if sufficient number of dimensions available
    if a.ndim < convolution_dim or v.ndim < convolution_dim:
        raise ValueError(
            f"{convolution_dim}D-convolution requires at least {convolution_dim}-dimensional input. Signal: {a.shape}, Filter: {v.shape}"
        )

    # Determine the promoted data type for 'a' and 'v' and convert them to this data type
    promoted_type = promote_types(a.dtype, v.dtype)
    if a.larray.is_mps and promoted_type == float64:
        # cannot cast to float64 on MPS
        promoted_type = float32

    a = a.astype(promoted_type)
    v = v.astype(promoted_type)

    # check if the filter is longer than the signal and swap them if necessary
    v_shape = v.shape[-convolution_dim:]
    a_shape = a.shape[-convolution_dim:]

    if all(v_s >= a_s for v_s, a_s in zip(v_shape, a_shape)):
        if not all(v_s == a_s for v_s, a_s in zip(v_shape, a_shape)):
            a, v = v, a
            v_shape = v.shape[-convolution_dim:]
            a_shape = a.shape[-convolution_dim:]

    if any(v_s > a_s for v_s, a_s in zip(v_shape, a_shape)):
        raise ValueError(
            f"Filter size must not be larger in one convolved dimension and smaller in the other. Signal: {a.shape}, Filter: {v.shape}"
        )

    # check mode against even kernel
    if mode not in ("full", "valid", "same"):
        raise ValueError(f"Only 'full', 'valid' or 'same' as mode are allowed, got {mode}.")
    if mode == "same" and any(v_s % 2 == 0 for v_s in v_shape):
        raise ValueError("Mode 'same' cannot be used with even-sized kernel.")

    # check mode and stride for value errors
    if convolution_dim == 1:
        if stride < 1:
            raise ValueError("Stride must be positive")
        if stride > 1 and mode == "same":
            raise ValueError("Stride must be 1 for mode 'same'")
    else:
        if any(s < 1 for s in stride):
            raise ValueError("Stride must be positive for all convolution dimensions")
        if any(s > 1 for s in stride) and mode == "same":
            raise ValueError(f"Stride must be {tuple([1] * convolution_dim)} for mode 'same'")

    # Return the processed 'a' and 'v' as a tuple
    return a, v


def conv_batchprocessing_check(a, v, convolution_dim):
    # assess whether to perform batch processing, default is False (no batch processing)
    batch_processing = False
    if a.ndim > convolution_dim:
        # batch processing requires 1D filter OR matching batch dimensions for signal and filter
        batch_dims = a.shape[:-convolution_dim]
        # verify that the filter shape is consistent with the signal
        if v.ndim > convolution_dim:
            v_batch = v.shape[:-convolution_dim]
            if any(v_s != b_s for v_s, b_s in zip(batch_dims, v_batch)):
                raise ValueError(
                    f"Batch dimensions of signal and filter must match. Signal: {a.shape}, Filter: {v.shape}"
                )
        if a.is_distributed():
            if any(a.split == a.ndim - forbidden for forbidden in range(1, convolution_dim + 1)):
                raise ValueError(
                    "Please distribute the signal along the batch dimension, not the signal dimension. For in-place redistribution use the `DNDarray.resplit_()` method with `axis=0`"
                )
        batch_processing = True

    if (not batch_processing) and (v.ndim > convolution_dim):
        raise ValueError(
            f"{convolution_dim}-D convolution without batch processing only supported for {convolution_dim}-dimensional signal and kernel. Signal: {a.shape}, Filter: {v.shape}"
        )

    return batch_processing


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
    a, v = conv_input_check(a, v, stride, mode, 1)

    # assess whether to perform batch processing, default is False (no batch processing)
    batch_processing = conv_batchprocessing_check(a, v, 1)

    if batch_processing and a.is_distributed() and v.is_distributed():
        if v.ndim == 1:
            # gather filter to all ranks
            v.resplit_(axis=None)
        else:
            v.resplit_(axis=a.split)

    # ensure balanced kernel
    if not (v.is_balanced()):
        raise ValueError("Only balanced kernel weights are allowed")

    # calculate pad size according to mode
    if mode == "full":
        pad_size = v.shape[-1] - 1
    elif mode == "same":
        pad_size = v.shape[-1] // 2
    elif mode == "valid":
        pad_size = 0

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
        if (v.lshape_map[:, a.split] > a.lshape_map[:, a.split]).any():
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


def convolve2d(
    a: DNDarray,
    v: DNDarray,
    mode: str = "full",
    stride: tuple[int, int] = (1, 1),
):
    """
    Returns the discrete, linear convolution of two two-dimensional HeAT tensors.

    Missing: Add batch option, change two last two dimensions are convolved

    Parameters
    ----------
    a : scalar, array_like, DNDarray
        Two-dimensional signal
    v : scalar, array_like, DNDarray
        Two-dimensional filter mask.
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
    stride: Tuple(int,int), optional
        Stride of the convolution in (x,y) direction. Default is (1,1).

    Returns
    -------
    out : ht.tensor
        Discrete, linear convolution of 'a' and 'v',  balanced

    Note : If the filter weight is larger
        than fitting into memory, using the FFT for convolution is recommended.

    Example
    --------
    >>> a = ht.ones((5, 5))
    >>> v = ht.ones((3, 3))
    >>> ht.convolve2d(a, v, mode="valid")
    DNDarray([[9., 9., 9.],
              [9., 9., 9.],
              [9., 9., 9.]], dtype=ht.float32, device=cpu:0, split=None)

    >>> a = ht.ones((5, 5), split=1)
    >>> v = ht.ones((3, 3), split=1)
    >>> ht.convolve2d(a, v)
    DNDarray([[1., 2., 3., 3., 3., 2., 1.],
              [2., 4., 6., 6., 6., 4., 2.],
              [3., 6., 9., 9., 9., 6., 3.],
              [3., 6., 9., 9., 9., 6., 3.],
              [3., 6., 9., 9., 9., 6., 3.],
              [2., 4., 6., 6., 6., 4., 2.],
              [1., 2., 3., 3., 3., 2., 1.]], dtype=ht.float32, device=cpu:0, split=1)
    """
    # check type and size of input
    a, v = conv_input_check(a, v, stride, mode, 2)

    # assess whether to perform batch processing, default is False (no batch processing)
    batch_processing = conv_batchprocessing_check(a, v, 2)

    if a.is_distributed() and v.is_distributed():
        if batch_processing and v.ndim == 2:
            # gather filter to all ranks
            v.resplit_(axis=None)
        else:
            v.resplit_(axis=a.split)

    # ensure balanced kernel
    if not (v.is_balanced()):
        raise ValueError("Only balanced kernel weights are allowed")

    # calculate pad size according to mode
    if mode == "full":
        pad_size = [v.shape[i] - 1 for i in range(-2, 0)]
    elif mode == "same":
        pad_size = [v.shape[i] // 2 for i in range(-2, 0)]
    elif mode == "valid":
        pad_size = [0, 0]

    gshape = tuple(
        [(a.shape[i] + 2 * pad_size[i] - v.shape[i]) // stride[i] + 1 for i in range(-2, 0)]
    )

    if v.is_distributed() and any(s > 1 for s in stride):
        gshape_stride_1 = tuple(
            [(a.shape[i] + 2 * pad_size[i] - v.shape[i]) + 1 for i in range(-2, 0)]
        )

    if batch_processing:
        # all operations are local torch operations, only the last dimension is convolved
        local_a = a.larray
        local_v = v.larray

        # flip filter for convolution, as Pytorch conv1d computes correlations
        local_v = torch.flip(local_v, [-2, -1])
        local_batch_dims = tuple(local_a.shape[:-2])

        # reshape signal and filter to 3D for Pytorch conv1d function
        # see https://pytorch.org/docs/stable/generated/torch.nn.functional.conv1d.html
        local_a = local_a.reshape(
            torch.prod(torch.tensor(local_batch_dims, device=local_a.device), dim=0).item(),
            local_a.shape[-2],
            local_a.shape[-1],
        )
        channels = local_a.shape[0]

        if v.ndim > 2:
            local_v = local_v.reshape(
                torch.prod(torch.tensor(local_batch_dims, device=local_v.device), dim=0).item(),
                local_v.shape[-2],
                local_v.shape[-1],
            )
            local_v = local_v.unsqueeze(1)
        else:
            local_v = (
                local_v.unsqueeze(0)
                .unsqueeze(0)
                .expand(local_a.shape[0], 1, local_v.shape[-2], local_v.shape[-1])
            )

        # add batch dimension to signal
        local_a = local_a.unsqueeze(0)

        # cast to single-precision float if on GPU
        if local_a.is_cuda:
            float_type = torch.promote_types(local_a.dtype, torch.float32)
            local_a = local_a.to(float_type)
            local_v = local_v.to(float_type)

        # apply torch convolution operator if local signal isn't empty, add zero padding
        if torch.prod(torch.tensor(local_a.shape, device=local_a.device)) > 0:
            local_convolved = fc.conv2d(
                local_a, local_v, groups=channels, stride=stride, padding=tuple(pad_size)
            )
        else:
            empty_shape = tuple(local_a.shape[:-1] + (gshape[-2],) + (gshape[-1],))
            local_convolved = torch.empty(empty_shape, dtype=local_a.dtype, device=local_a.device)

        # unpack 3D result into original shape
        local_convolved = local_convolved.squeeze(0)
        local_convolved = local_convolved.reshape(local_batch_dims + (gshape[-2],) + (gshape[-1],))

        # wrap result in DNDarray
        convolved = array(local_convolved, is_split=a.split, device=a.device, comm=a.comm)
        return convolved

    # pad signal with zeros
    pad_array = ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]))
    a = pad(a, pad_array)
    print(a.comm.rank, "Padd area halo prev", a.larray[-20:, :])
    print(a.comm.rank, "Padd area halo next", a.larray[0:20, :])
    # CF: Necessary for convolution2d
    a.comm.Barrier()
    # print("lshape map", a.comm.rank, a.lshape_map, v.lshape_map)
    # no batch processing
    if a.is_distributed():
        if (v.lshape_map[:, a.split] > a.lshape_map[:, a.split]).any():
            raise ValueError(
                "Local chunk of filter weight is larger than the local chunks of signal"
            )

        # compute halo size
        halo_size = int(v.lshape_map[0][a.split]) // 2

        # fetch halos and store them in a.halo_next/a.halo_prev
        a.get_halo(halo_size)

        print(a.comm.rank, "Halo prev", a.halo_prev)
        print(a.comm.rank, "Halo next", a.halo_next)
        # apply halos to local array
        signal = a.array_with_halos
    else:
        # get local array in case of non-distributed a
        signal = a.larray

    # flip filter for convolution as PyTorch conv2d computes correlation
    no_dims = len(v.shape)
    v = flip(v, [no_dims - 2, no_dims - 1])

    # compute weight size
    if v.is_distributed() and v.larray.shape[v.split] != v.lshape_map[0, v.split]:
        # pads weights if input kernel is uneven
        target = torch.zeros(tuple(v.lshape_map[0]), dtype=v.larray.dtype, device=v.larray.device)
        v_pad_size = v.lshape_map[0][v.split] - v.larray.shape[v.split]
        if v.split == 0:
            target[v_pad_size:, :] = v.larray
        else:
            target[:, v_pad_size:] = v.larray
        weight = target

        print(v.comm.rank, "v_pad_size", v_pad_size, weight.shape, v.larray.shape)
    else:
        weight = v.larray

    t_v = weight

    # make signal and filter weight 4D for Pytorch conv2d function
    signal = signal.reshape(1, 1, signal.shape[-2], signal.shape[-1])
    weight = weight.reshape(1, 1, weight.shape[-2], weight.shape[-1])

    if v.is_distributed():
        size = v.comm.size
        split_axis = v.split

        # any stride is a subset of stride 1
        if any(s > 1 for s in stride):
            gshape = gshape_stride_1

        # convoluted signal
        signal_filtered = zeros(gshape, dtype=a.dtype, split=a.split, device=a.device, comm=a.comm)

        for r in range(size):
            rec_v = t_v.clone()
            v.comm.Bcast(rec_v, root=r)
            t_v1 = rec_v.reshape(1, 1, rec_v.shape[0], rec_v.shape[1])

            # apply torch convolution operator
            local_signal_filtered = fc.conv2d(signal, t_v1, stride=1)
            # unpack 3D result into 2D
            local_signal_filtered = local_signal_filtered[0, 0, :, :]

            # if kernel shape along split axis is even we need to get rid of duplicated values
            if a.is_distributed() and v.comm.rank != 0 and v.lshape_map[0][split_axis] % 2 == 0:
                if split_axis == 0:
                    local_signal_filtered = local_signal_filtered[1:, :]
                else:
                    local_signal_filtered = local_signal_filtered[:, 1:]

            # compute offset for local_signal_filtered
            if r > 0:
                v_pad_size = v.lshape_map[0][v.split] - v.lshape_map[r, v.split]
                start_idx = torch.sum(v.lshape_map[:r, split_axis]).item() - v_pad_size
                if v.comm.rank == 0:
                    print(v.comm.rank, r, "start_idx", start_idx, v_pad_size)

            else:
                start_idx = 0

            # if a is distributed, results have to be communicated across ranks
            if a.is_distributed():
                filter_results = array(
                    local_signal_filtered, is_split=a.split, device=a.device, comm=a.comm
                )
            else:
                filter_results = local_signal_filtered

            # apply start_idx
            if split_axis == 0:
                filter_results = filter_results[start_idx : start_idx + gshape[0], :]
            else:
                filter_results = filter_results[:, start_idx : start_idx + gshape[1]]

            # add results
            try:
                # print(v.comm.rank, r, "Not in Exception", v.lshape_map)
                print(
                    "Add results: ",
                    v.comm.rank,
                    r,
                    gshape,
                    signal_filtered.shape,
                    filter_results.shape,
                    local_signal_filtered.shape,
                )
                if a.is_distributed():
                    signal_filtered += filter_results
                else:
                    signal_filtered.larray += filter_results

            except (ValueError, TypeError):
                print(v.comm.rank, "In Exception", signal_filtered.split)
                if a.is_distributed():
                    signal_filtered = signal_filtered + filter_results
                else:
                    signal_filtered.larray = signal_filtered.larray + filter_results

        if any(s > 1 for s in stride):
            signal_filtered = signal_filtered[:: stride[0], :: stride[1]]
        # print(v.comm.rank, "after stride", signal_filtered.larray)

        if a.is_distributed():
            signal_filtered.balance_()

        return signal_filtered

    else:
        # shift signal based on global kernel starts for any rank but first if stride > 1
        if a.is_distributed() and stride[a.split] > 1:
            if a.comm.rank == 0:
                local_index = 0
            else:
                # lshape map does not know about padding, compute pad_offset for last rank
                # pad_offset = pad_size[a.split] if a.comm.rank == a.comm.size - 1 else 0

                local_index = torch.sum(a.lshape_map[: a.comm.rank, a.split]).item() - halo_size
                local_index = local_index % stride[a.split]

                if local_index != 0:
                    local_index = stride[a.split] - local_index

                # even kernels can produces doubles
                if v.shape[a.split] % 2 == 0 and local_index == 0:
                    local_index = stride[a.split]

            if a.split == 0:
                signal = signal[:, :, local_index:, :]
            else:
                signal = signal[:, :, :, local_index:]

        print(a.comm.rank, "Signal min max", signal.min(), signal.max())
        # w_start = weight.shape[-1]
        # if a.comm.rank == 0:
        #    print(0, "Halo", signal[0,0,-halo_size*2:-halo_size,-w_start-1:-w_start+1].shape,
        #          signal[0,0,-halo_size*2:-halo_size,-w_start-1:-w_start+1])
        # if a.comm.rank == 1:
        #    print(1, "Line 53-54", signal[0,0,0:halo_size,-w_start-1:-w_start+1].shape,
        #          signal[0,0,0:halo_size,-w_start-1:-w_start+1])

        if all(a_s >= v_s for v_s, a_s in zip(weight.shape[-2:], signal.shape[-2:])):
            # apply torch convolution operator
            signal_filtered = fc.conv2d(signal, weight, stride=stride)
            # unpack 4D result into 2D
            signal_filtered = signal_filtered[0, 0, :, :]
        else:
            signal_filtered = torch.tensor([[]], device=str(signal.device))

        # if kernel shape along split axis is even we need to get rid of duplicated values
        if (
            a.is_distributed()
            and a.comm.rank != 0
            and stride[a.split] == 1
            and v.shape[a.split] % 2 == 0
        ):
            if a.split == 0:
                signal_filtered = signal_filtered[1:, :]
            elif a.split == 1:
                signal_filtered = signal_filtered[:, 1:]

        print(a.comm.rank, "Signal filtered shape", signal_filtered.shape)
        # if a.comm.rank == 1:
        #    print("Line 53: ", signal_filtered[0,-2:])
        #    print("Line 70: ", signal_filtered[16, -2:])

        result = DNDarray(
            signal_filtered.contiguous(),
            gshape,
            a.dtype,
            a.split,
            a.device,
            a.comm,
            balanced=False,
        ).astype(a.dtype.torch_type())

        print(a.comm.rank, "Result shape, before balancing: ", result.lshape_map)
        if result.is_distributed():
            result.balance_()

        return result
