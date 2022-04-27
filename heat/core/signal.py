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
                pad[1+i*dim] = 0
                    
        # set the padding of the last rank
        elif a.comm.rank == a.comm.size - 1:
            for i in range(dim):
                pad[i*dim] = 0
                    
    if boundary == 'fill':
        signal = fc.pad(signal, pad, mode='constant', value=fillvalue)
    elif boundary == "wrap": 
        signal = fc.pad(signal, pad, mode='circular')
    elif boundary == 'symm':
        signal = fc.pad(signal, pad, mode='reflect')  
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

def convolve2d(a, v, mode="full", boundary='fill', fillvalue=0):
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

    Returns
    -------
    out : ht.tensor
        Discrete, linear convolution of 'a' and 'v'.

    Note : There is  differences to the numpy convolve function:
        The inputs are not swapped if v is larger than a. The reason is that v needs to be
        non-splitted. This should not influence performance. If the filter weight is larger
        than fitting into memory, using the FFT for convolution is recommended.

    Examples
    --------
    Note how the convolution operator flips the second array
    before "sliding" the two across one another:
    >>> a = ht.ones(10)
    >>> v = ht.arange(3).astype(ht.float)
    >>> ht.convolve1D(a, v, mode='full')
    tensor([0., 1., 3., 3., 3., 3., 2.])

    Only return the middle values of the convolution.
    Contains boundary effects, where zeros are taken
    into account:
    >>> ht.convolve1D(a, v, mode='same')
    tensor([1., 3., 3., 3., 3.])

    Compute only positions where signal and filter weights
    completely overlap:
    >>> ht.convolve1D(a, v, mode='valid')
    tensor([3., 3., 3.])
    """
    if not isinstance(a, dndarray.DNDarray) or not isinstance(v, dndarray.DNDarray):
        raise TypeError("Signal and filter weight must be of type ht.tensor")
    if v.split is not None:
        raise TypeError("Distributed filter weights are not supported")
    if len(a.shape) != 2 or len(v.shape) != 2:
        raise ValueError("Only 2 dimensional input tensors are allowed")
    if a.shape[0] <= v.shape[0] or a.shape[1] <= v.shape[1]:
        raise ValueError("Filter size must not be larger than signal size")
    if a.dtype is not v.dtype:
        raise TypeError("Signal and filter weight must be of same type")
    if mode == "same" and v.shape[0] % 2 == 0  and v.shape[1] % 2 == 0:
        raise ValueError("Mode 'same' cannot be use with even sized kernal")

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
        pad_0 = v.shape[0]-1
        pad_1 = v.shape[1]-1
        gshape_0 = v.shape[0] + a.shape[0] - 1
        gshape_1 = v.shape[1] + a.shape[1] - 1
        pad = list((pad_0, pad_0, pad_1, pad_1))
        gshape = list((gshape_0, gshape_1))

    elif mode == "same":
        pad = list((halo_size,)*4)
        gshape = a.shape[0]

    elif mode == "valid":
        pad = list((0,)*4)
        gshape_0 = a.shape[0] - v.shape[0] + 1
        gshape_1 = a.shape[1] - v.shape[1] + 1
        gshape = list((gshape_0, gshape_1))

    else:
        raise ValueError("Only {'full', 'valid', 'same'} are allowed for mode")

    # make signal and filter weight 4D for Pytorch conv2d function
    signal.unsqueeze_(0)
    signal.unsqueeze_(0)

    # add padding to the borders according to mode
    signal = genpad(a, signal, pad, a.split, boundary, fillvalue)

    # flip filter for convolution as PyTorch conv2d computes correlation
    weight = torch.flip(v._DNDarray__array.clone(), [0, 1])

    weight.unsqueeze_(0)
    weight.unsqueeze_(0)

    # apply torch convolution operator
    signal_filtered = fc.conv2d(signal, weight)

    # unpack 3D result into 1D
    signal_filtered = signal_filtered[0, 0, :]

    # if kernel shape along split axis is even we need to get rid of duplicated values
    if a.comm.rank != 0 and v.shape[0] % 2 == 0:
        signal_filtered = signal_filtered[1:, 1:]

    return dndarray.DNDarray(
        signal_filtered.contiguous(), (gshape,), signal_filtered.dtype, a.split, a.device, a.comm, a.balanced
    ).astype(a.dtype.torch_type())
