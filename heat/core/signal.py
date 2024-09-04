"""Provides a collection of signal-processing operations"""

import heat as ht
import torch
import numpy as np
import operator


from .communication import MPI
from .dndarray import DNDarray
from .types import (
    promote_types,
    _complexfloating,
    heat_type_of,
)
from .manipulations import pad, flip
from .factories import array, zeros
import torch.nn.functional as fc
from numbers import Number


__all__ = ["convolve", "fftconvolve"]


def convolve(a: DNDarray, v: DNDarray, mode: str = "full") -> DNDarray:
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

        >>> a = ht.ones(5)
        >>> v = ht.arange(3).astype(ht.float)
        >>> ht.convolve(a, v, mode='full')
        DNDarray([0., 1., 3., 3., 3., 3., 2.], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.convolve(a, v, mode='same')
        DNDarray([1., 3., 3., 3., 3.], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.convolve(a, v, mode='valid')
        DNDarray([3., 3., 3.], dtype=ht.float32, device=cpu:0, split=None)
        >>> a = ht.ones(10, split = 0)
        >>> v = ht.arange(3, split = 0).astype(ht.float)
        >>> ht.convolve(a, v, mode='valid')
        DNDarray([3., 3., 3., 3., 3., 3., 3., 3.], dtype=ht.float32, device=cpu:0, split=0)

        [0/3] tensor([3., 3., 3.])
        [1/3] tensor([3., 3., 3.])
        [2/3] tensor([3., 3.])
        >>> a = ht.ones(10, split = 0)
        >>> v = ht.arange(3, split = 0)
        >>> ht.convolve(a, v)
        DNDarray([0., 1., 3., 3., 3., 3., 3., 3., 3., 3., 3., 2.], dtype=ht.float32, device=cpu:0, split=0)

    <<<<<<< HEAD
        [0/3] tensor([0., 1., 3., 3.])
        [1/3] tensor([3., 3., 3., 3.])
        [2/3] tensor([3., 3., 3., 2.])
    =======
        [0/3] DNDarray([0., 1., 3., 3.])
        [1/3] DNDarray([3., 3., 3., 3.])
        [2/3] DNDarray([3., 3., 3., 2.])

        >>> a = ht.arange(50, dtype = ht.float64, split=0)
        >>> a = a.reshape(10, 5) # 10 signals of length 5
        >>> v = ht.arange(3)
        >>> ht.convolve(a, v) # batch processing: 10 signals convolved with filter v
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

        >>> v = ht.random.randint(0, 3, (10, 3), split=0) # 10 filters of length 3
        >>> ht.convolve(a, v) # batch processing: 10 signals convolved with 10 filters
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
    >>>>>>> main
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

    if mode == "same" and v.shape[-1] % 2 == 0:
        raise ValueError("Mode 'same' cannot be used with even-sized kernel")
    if not v.is_balanced():
        raise ValueError("Only balanced kernel weights are allowed")

    # calculate pad size according to mode
    if mode == "full":
        pad_size = v.shape[-1] - 1
        gshape = v.shape[-1] + a.shape[-1] - 1
    elif mode == "same":
        pad_size = v.shape[-1] // 2
        gshape = a.shape[-1]
    elif mode == "valid":
        pad_size = 0
        gshape = a.shape[-1] - v.shape[-1] + 1
    else:
        raise ValueError(f"Supported modes are 'full', 'valid', 'same', got {mode}")

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
            local_convolved = fc.conv1d(local_a, local_v, padding=pad_size, groups=channels)
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
            try:
                signal_filtered += global_signal_filtered[start_idx : start_idx + gshape]
            except (ValueError, TypeError):
                signal_filtered = (
                    signal_filtered + global_signal_filtered[start_idx : start_idx + gshape]
                )
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


"""
The following implementation is heavily inspired by the one of `fftconvolve` in SciPy
(https://github.com/scipy/scipy/blob/fa9f13e6906e7d00510d593f7f982db30e4e4f14/scipy/signal/_signaltools.py).
"""


def fftconvolve(t1: DNDarray, t2: DNDarray, mode: str = "full", axes=None) -> DNDarray:
    """
    Convolve two N-dimensional arrays using FFT.

    Convolve `t1` and `t2` using the fast Fourier transform method, with the output size determined
    by the `mode` argument.

    This is generally much faster than `convolve` for large arrays (n > ~500), but can be slower
    when only a few output values are needed, and can only output float arrays (int or object array
    inputs will be cast to float).

    Parameters
    ----------
    t1 :    DNDarray or scalar
            First input.
    t2 :    DNDarray or scalar
            Second input. Should have the same number of dimensions as `t1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        ``full``
           The output is the full discrete linear convolution of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not rely on the zero-padding. In
           'valid' mode, either `t1` or `t2` must be at least as large as the other in every
           dimension.
        ``same``
           The output is the same size as `t1`, centered with respect to the 'full' output.
    axes : int or array_like of ints or None, optional
        Axes over which to compute the convolution. The default is over all axes.

    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear convolution of `t1` with
        `t2`.
    """
    if not isinstance(t1, DNDarray):
        t1 = ht.array(t1)
    if not isinstance(t2, DNDarray):
        t2 = ht.array(t2)

    if t1.ndim == t2.ndim == 0:  # scalar inputs
        return t1 * t2
    elif t1.ndim != t2.ndim:
        raise ValueError(
            f"The inputs should have the same dimensionality. But your inputs have dim {t1.ndim} "
            + f"and {t2.ndim}."
        )
    elif t1.size == 0 or t2.size == 0:  # empty arrays
        return ht.array([])

    t1, t2, axes = _init_freq_conv_axes(t1, t2, mode, axes)

    s1 = t1.shape
    s2 = t2.shape

    shape = [max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1 for i in range(t1.ndim)]

    ret = _freq_domain_conv(t1, t2, axes, shape, calc_fast_len=True)

    return _apply_conv_mode(ret, s1, s2, mode, axes)


def _init_freq_conv_axes(t1, t2, mode, axes):
    """
    Handle the axes argument for frequency-domain convolution.

    Returns the inputs and axes in a standard form, eliminating redundant axes, swapping the inputs
    if necessary, and checking for various potential errors.

    Returns
    -------
    t1 : array
        The first input, possibly swapped with the second input.
    t2 : array
        The second input, possibly swapped with the first input.
    axes : Tuple of ints
        Axes over which to compute the FFTs.
    """
    s1 = t1.shape
    s2 = t2.shape
    noaxes = axes is None

    axes = _init_nd_axes(t1, axes=axes)

    if not noaxes and not len(axes):
        raise ValueError("when provided, axes cannot be empty")

    # Axes of length 1 can rely on broadcasting rules for multipy, no fft needed.
    axes = tuple([a for a in axes if s1[a] != 1 and s2[a] != 1])

    if not all(s1[a] == s2[a] or s1[a] == 1 or s2[a] == 1 for a in range(t1.ndim) if a not in axes):
        raise ValueError("incompatible shapes for t1 and t2: {} and {}".format(s1, s2))

    # Check that input sizes are compatible with 'valid' mode.
    if _inputs_swap_needed(mode, s1, s2, axes=axes):
        # Convolution is commutative; order doesn't have any effect on output.
        t1, t2 = t2, t1

    return t1, t2, axes


def _init_nd_axes(x, axes):
    """Handles axes arguments for nd transforms"""
    noaxes = axes is None

    if not noaxes:
        axes = _iterable_of_int(axes, "axes")
        axes = [a + x.ndim if a < 0 else a for a in axes]

        if any(a >= x.ndim or a < 0 for a in axes):
            raise ValueError("axes exceeds dimensionality of input")
        if len(set(axes)) != len(axes):
            raise ValueError("all axes must be unique")

    if noaxes:
        shape = tuple(x.shape)
        axes = range(x.ndim)
    else:
        shape = [x.shape[a] for a in axes]

    if any(s < 1 for s in shape):
        raise ValueError(f"invalid number of data points ({shape}) specified")

    return tuple(axes)


def _iterable_of_int(x, name=None):
    """
    Convert `x` to an iterable sequence of int

    Parameters
    ----------
    x    :  value, or sequence of values,
            convertible to int
    name :  str, optional
            Name of the argument being converted, only used in the error message

    Returns
    -------
    y : ``Tuple[int]``
    """
    if isinstance(x, Number):
        x = (x,)

    try:
        x = [operator.index(a) for a in x]
    except TypeError as e:
        name = name or "value"
        raise ValueError(f"{name} must be a scalar or iterable of integers") from e

    return x


def _inputs_swap_needed(mode, shape1, shape2, axes=None):
    """
    Determine if inputs arrays need to be swapped in `"valid"` mode.

    If in `"valid"` mode, returns whether or not the input arrays need to be
    swapped depending on whether `shape1` is at least as large as `shape2` in
    every calculated dimension.

    This is important for some of the correlation and convolution
    implementations in this module, where the larger array input needs to come
    before the smaller array input when operating in this mode.

    Note that if the mode provided is not 'valid', False is immediately
    returned.
    """
    if mode != "valid":
        return False

    if not shape1:
        return False

    if axes is None:
        axes = range(len(shape1))

    ok1 = all(shape1[a] >= shape2[a] for a in axes)
    ok2 = all(shape2[a] >= shape1[a] for a in axes)

    if not (ok1 or ok2):
        raise ValueError(
            "For 'valid' mode, one must be at least as large as the other in every dimension."
        )

    return not ok1


def _freq_domain_conv(t1, t2, axes, shape, calc_fast_len=False):
    """
    Convolve two arrays in the frequency domain.

    This function implements only base the FFT-related operations. Specifically, it converts the
    signals to the frequency domain, multiplies them, then converts them back to the time domain.
    Calculations of axes, shapes, convolution mode, etc. are implemented in higher level-functions,
    such as `fftconvolve` and `oaconvolve`. Those functions should be used instead of this one.

    Returns
    -------
    out : array
        An N-dimensional array containing the discrete linear convolution of `t1` with `t2`.
    """
    if not len(axes):
        return t1 * t2

    dtype1, dtype2 = (heat_type_of(t1), heat_type_of(t2))
    complex_result = dtype1 in _complexfloating or dtype2 in _complexfloating

    """
    if calc_fast_len:
        # Speed up FFT by padding to optimal size.
        fshape = [
            sp_fft.next_fast_len(shape[a], not complex_result) for a in axes]
    else:
    """
    if not complex_result:
        fft, ifft = ht.fft.rfftn, ht.fft.irfftn
    else:
        fft, ifft = ht.fft.fftn, ht.fft.ifftn

    sp1 = fft(t1, shape, axes=axes)
    sp2 = fft(t2, shape, axes=axes)

    ret = ifft(sp1 * sp2, shape, axes=axes)

    """
    if calc_fast_len:
        fslice = tuple([slice(sz) for sz in shape])
        ret = ret[fslice]
    """

    return ret


def _apply_conv_mode(ret, s1, s2, mode, axes):
    """
    Calculate the convolution result shape based on the `mode` argument.

    Returns the result sliced to the correct size for the given mode.

    Returns
    -------
    ret : array
        A copy of `res`, sliced to the correct size for the given `mode`.
    """
    if mode == "full":
        return ret.copy()  # Why copy and not original??
    elif mode == "same":
        return _centered(ret, s1).copy()
    elif mode == "valid":
        shape_valid = [
            ret.shape[a] if a not in axes else s1[a] - s2[a] + 1 for a in range(ret.ndim)
        ]
        return _centered(ret, shape_valid).copy()
    else:
        raise ValueError("acceptable mode flags are 'valid'," " 'same', or 'full'")


def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]
