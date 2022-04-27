"""Provides a collection of signal-processing functions"""

import torch

from .communication import MPI
from . import dndarray
import torch.nn.functional as fc

__all__ = ["convolve1D", "convolve2D", "convolve3D"]


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



def convolve1D(a, v, mode="full", fillvalue=0):
    """
    Returns the discrete, linear convolution of two one-dimensional HeAT tensors.

    Parameters
    ----------
    a : (N,) ht.tensor
        One-dimensional signal HeAT tensor
    v : (M,) ht.tensor
        One-dimensional filter weight HeAT tensor.
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
    if len(a.shape) != 1 or len(v.shape) != 1:
        raise ValueError("Only 1 dimensional input tensors are allowed")
    if a.shape[0] <= v.shape[0]:
        raise ValueError("Filter size must not be larger than signal size")
    if a.dtype is not v.dtype:
        raise TypeError("Signal and filter weight must be of same type")
    if mode == "same" and v.shape[0] % 2 == 0:
        raise ValueError("Mode 'same' cannot be use with even sized kernal")

    # compute halo size
    halo_size = v.shape[0] // 2 if v.shape[0] % 2 == 0 else (v.shape[0] - 1) // 2

    # fetch halos and store them in a.halo_next/a.halo_prev
    a.get_halo(halo_size)

    # apply halos to local array
    signal = a.array_with_halos

    # check if a local chunk is smaller than the filter size
    if a.is_distributed() and signal.size()[0] < v.shape[0]:
        raise ValueError("Local chunk size is smaller than filter size, this is not supported yet")

   
    if mode == "full":
        pad_prev = pad_next = v.shape[0] - 1
        gshape = v.shape[0] + a.shape[0] - 1
        pad = list((pad_prev, pad_next))

    elif mode == "same":
        pad = list((halo_size, halo_size))
        gshape = a.shape[0]

    elif mode == "valid":
        pad = list((0, 0))
        gshape = a.shape[0] - v.shape[0] + 1

    else:
        raise ValueError("Only {'full', 'valid', 'same'} are allowed for mode")

    # make signal and filter weight 3D for Pytorch conv1d function
    signal.unsqueeze_(0)
    signal.unsqueeze_(0)

    # add padding to the borders according to mode
    signal = genpad(a, signal, pad, 0, mode, fillvalue)


    # flip filter for convolution as Pytorch conv1d computes correlations
    weight = torch.flip(v._DNDarray__array.clone(), [0])

    weight.unsqueeze_(0)
    weight.unsqueeze_(0)

    # apply torch convolution operator
    signal_filtered = fc.conv1d(signal, weight)

    # unpack 3D result into 1D
    signal_filtered = signal_filtered[0, 0, :]

    # if kernel shape along split axis is even we need to get rid of duplicated values
    if a.comm.rank != 0 and v.shape[0] % 2 == 0:
        signal_filtered = signal_filtered[1:]

    return dndarray.DNDarray(
        signal_filtered.contiguous(), (gshape,), signal_filtered.dtype, a.split, a.device, a.comm, a.balanced
    ).astype(a.dtype.torch_type())

# in1, in2, mode='full', boundary='fill', fillvalue=0
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


    signal.unsqueeze_(0)
    signal.unsqueeze_(0)


    # add padding to the borders according to mode
    signal = genpad(a, signal, pad, a.split, boundary, fillvalue)

    
    # make signal and filter weight 4D for Pytorch conv2d function
    #signal.unsqueeze_(0)
    #signal.unsqueeze_(0)

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
        signal_filtered = signal_filtered[1:]

    return dndarray.DNDarray(
        signal_filtered.contiguous(), (gshape,), signal_filtered.dtype, a.split, a.device, a.comm, a.balanced
    ).astype(a.dtype.torch_type())

def convolve3D(a, v, mode="full"):
    """
    Returns the discrete, linear convolution of two three-dimensional HeAT tensors.

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
    if len(a.shape) != 3 or len(v.shape) != 3:
        raise ValueError("Only 2 dimensional input tensors are allowed")
    if a.shape[0] <= v.shape[0] or a.shape[1] <= v.shape[1] or a.shape[2] <= v.shape[2]:
        raise ValueError("Filter size must not be larger than signal size")
    if a.dtype is not v.dtype:
        raise TypeError("Signal and filter weight must be of same type")
    if mode == "same" and v.shape[0] % 2 == 0  and v.shape[1] % 2 == 0:
        raise ValueError("Mode 'same' cannot be use with even sized kernal")

    # compute halo size
    if a.split == 0 or a.split == None:
        halo_size = v.shape[0] // 2 if v.shape[0] % 2 == 0 else (v.shape[0] - 1) // 2
    if a.split == 1 or a.split == None:
        halo_size = v.shape[0] // 2 if v.shape[0] % 2 == 0 else (v.shape[0] - 1) // 2

    # fetch halos and store them in a.halo_next/a.halo_prev
    a.get_halo(halo_size)

    # apply halos to local array
    signal = a.array_with_halos.clone()

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
            pad_prev = pad_next = torch.zeros(v.shape[0] - 1, dtype=a.dtype.torch_type())

        elif (not has_left) and has_right:  # maybe just check for rank?
            # first process, pad only left
            pad_prev = torch.zeros(v.shape[0] - 1, dtype=a.dtype.torch_type())
            pad_next = None

        elif has_left and (not has_right):
            # last process, pad only right
            pad_prev = None
            pad_next = torch.zeros(v.shape[0] - 1, dtype=a.dtype.torch_type())

        else:
            # all processes in between don't need padding
            pad_prev = pad_next = None

        gshape = v.shape[0] + a.shape[0] - 1

    elif mode == "same":
        # first and last need padding
        pad_prev = pad_next = None
        if a.comm.rank == 0:
            pad_prev = torch.zeros(halo_size, dtype=a.dtype.torch_type())
        if a.comm.rank == a.comm.size - 1:
            pad_next = torch.zeros(halo_size, dtype=a.dtype.torch_type())

        gshape = a.shape[0]

    elif mode == "valid":
        pad_prev = pad_next = None
        gshape = a.shape[0] - v.shape[0] + 1

    else:
        raise ValueError("Only {'full', 'valid', 'same'} are allowed for mode")

    # add padding to the borders according to mode
    signal = a.genpad(signal, pad_prev, pad_next, a.split)

    
    # make signal and filter weight 4D for Pytorch conv2d function
    signal.unsqueeze_(0)
    signal.unsqueeze_(0)

    # flip filter for convolution as PyTorch conv2d computes correlation
    weight = torch.flip(v._DNDarray__array.clone(), [0, 1])

    weight.unsqueeze_(0)
    weight.unsqueeze_(0)

    # apply torch convolution operator
    signal_filtered = fc.conv3d(signal, weight)

    # unpack 3D result into 1D
    signal_filtered = signal_filtered[0, 0, :]

    # if kernel shape along split axis is even we need to get rid of duplicated values
    if a.comm.rank != 0 and v.shape[0] % 2 == 0:
        signal_filtered = signal_filtered[1:]

    return dndarray.DNDarray(
        signal_filtered.contiguous(), (gshape,), signal_filtered.dtype, a.split, a.device, a.comm, a.balanced
    ).astype(a.dtype.torch_type())

