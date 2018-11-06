from copy import copy as _copy
import operator
import numpy as np
import warnings
import torch
import torch.nn.functional as fc
from .communicator import mpi, MPICommunicator, NoneCommunicator
from .stride_tricks import *
from . import types
from . import operations
from . import halo
import math
import functools
sign = functools.partial(math.copysign, 1)


class tensor:
    def __init__(self, array, gshape, dtype, split, comm, halo_next=None, halo_prev=None):
        self.__array = array  # One might consider to skip the double underscore and transfer attributes through setters
        self.__gshape = gshape
        self.__dtype = dtype
        self.__split = split
        self.__comm = comm
        self.__halo_next = halo_next
        self.__halo_prev = halo_prev

    @property
    def comm(self):
        return self.__comm
    
    @comm.setter
    def comm(self, comm):
        self.__comm = comm

    @property
    def dtype(self):
        return self.__dtype
    
    @dtype.setter
    def dtype(self, dtype):
        self.__dtype = dtype

    @property
    def gshape(self):
        return self.__gshape

    @gshape.setter
    def gshape(self, gshape):
        self.__gshape = gshape

    @property
    def lshape(self):
        if len(self.array.shape) == len(self.gshape):
            return tuple(self.__array.shape)
        # edge case when the local data tensor receives no elements after chunking
        return self.gshape[:self.split] + (0,) + self.gshape[self.split + 1:]

    @lshape.setter
    def lshape(self, lshape):
        self.__lshape = lshape

    @property
    def split(self):
        return self.__split

    @split.setter
    def split(self, split):
        self.__split = split

    @property
    def array(self):
        return self.__array

    @array.setter
    def array(self, array):
        self.__array = array

    @property
    def halo_next(self):
        return self.__halo_next

    @halo_next.setter
    def halo_next(self, halo_next):
        self.__halo_next = halo_next

    @property
    def halo_prev(self):
        return self.__halo_prev

    @halo_prev.setter
    def halo_prev(self, halo_prev):
        self.__halo_prev = halo_prev

    @property
    def shape(self):
        return self.gshape
    
    def abs(self, out=None):
        """
        Calculate the absolute value element-wise.

        Parameters
        ----------
        out : ht.tensor, optional
            A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
            If not provided or None, a freshly-allocated array is returned.
        
        Returns
        -------
        absolute_values : ht.tensor
            A tensor containing the absolute value of each element in x.
        """
        return operations.abs(self, out)

    def absolute(self, out=None):
        """
        Calculate the absolute value element-wise.

        np.abs is a shorthand for this function.

        Parameters
        ----------
        out : ht.tensor, optional
            A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
            If not provided or None, a freshly-allocated array is returned.
        dtype : ht.type, optional
            Determines the data type of the output array. The values are cast to this type with potential loss of
            precision.

        Returns
        -------
        absolute_values : ht.tensor
            A tensor containing the absolute value of each element in x.
        """

        return self.abs(out)

    def astype(self, dtype, copy=True):
        """
        Returns a casted version of this array.

        Parameters
        ----------
        dtype : ht.dtype
            HeAT type to which the array is cast
        copy : bool, optional
            By default the operation returns a copy of this array. If copy is set to false the cast is performed
            in-place and this tensor is returned

        Returns
        -------
        casted_tensor : ht.tensor
            casted_tensor is a new tensor of the same shape but with given type of this tensor. If copy is True, the
            same tensor is returned instead.
        """
        dtype = types.canonical_heat_type(dtype)
        casted_array = self.__array.type(dtype.torch_type())
        if copy:
            return tensor(casted_array, self.shape, dtype, self.split, _copy(self.__comm))

        self.__array = casted_array
        self.__dtype = dtype

        return self

    def clip(self, a_min, a_max, out=None):
        """
        Parameters
        ----------
        a_min : scalar or None
            Minimum value. If None, clipping is not performed on lower interval edge. Not more than one of a_min and
            a_max may be None.
        a_max : scalar or None
            Maximum value. If None, clipping is not performed on upper interval edge. Not more than one of a_min and
            a_max may be None.
        out : ht.tensor, optional
            The results will be placed in this array. It may be the input array for in-place clipping. out must be of
            the right shape to hold the output. Its type is preserved.

        Returns
        -------
        clipped_values : ht.tensor
            A tensor with the elements of this tensor, but where values < a_min are replaced with a_min, and those >
            a_max with a_max.
        """
        return operations.clip(self, a_min, a_max, out)

    def copy(self):
        """
        Return an array copy of the given object.

        Returns
        -------
        copied : ht.tensor
            A copy of the original
        """
        return operations.copy(self)

    # ToDO: halorize  __reduce_op like for __local_operation
    def __reduce_op(self, partial, op, axis):
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input

        # TODO: make me more numpy API complete
        # TODO: implement type promotion      
        axis = sanitize_axis(self.shape, axis) # need to further checking!
  
        if self.__comm.is_distributed() and (axis is None or axis == self.__split):
            mpi.all_reduce(partial, op, self.__comm.group)
            return tensor(partial, partial.shape, types.canonical_heat_type(partial.dtype), split=None, comm=NoneCommunicator())

        # TODO: verify if this works for negative split axis
        output_shape = self.gshape[:axis] + (1,) + self.gshape[axis + 1:]
        return tensor(partial, output_shape, types.canonical_heat_type(partial.dtype), self.split, comm=_copy(self.__comm))

    def argmin(self, axis):
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        # TODO: make me more numpy API complete
        # TODO: Fix me, I am not reduce_op.MIN!
        _, argmin_axis = self.__array.min(dim=axis, keepdim=True)
        return self.__reduce_op(argmin_axis, mpi.reduce_op.MIN, axis)

    def max(self, axis=None):
        """"
        Return the maximum of an array or maximum along an axis.
        Parameters
        ----------
        a : ht.tensor
        Input data.
        
        axis : None or int  
        Axis or axes along which to operate. By default, flattened input is used.   
        
        #TODO: out : ht.tensor, optional
        Alternative output array in which to place the result. Must be of the same shape and buffer length as the expected output. 

        #TODO: initial : scalar, optional   
        The minimum value of an output element. Must be present to allow computation on empty slice.
        """
        
        return operations.max(self, axis)

    def mean(self, axis):
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        # TODO: make me more numpy API complete
        return self.sum(axis) / self.gshape[axis]

    def min(self, axis=None):
        """"
        Return the minimum of an array or minimum along an axis.

        Parameters
        ----------
        a : ht.tensor
        Input data.
        
        axis : None or int
        Axis or axes along which to operate. By default, flattened input is used.   
        
        #TODO: out : ht.tensor, optional
        Alternative output array in which to place the result. Must be of the same shape and buffer length as the expected output. 

        #TODO: initial : scalar, optional   
        The maximum value of an output element. Must be present to allow computation on empty slice.
        """
        
        return operations.min(self, axis)
       
    def sum(self, axis=None):
        # TODO: Allow also list of axes
        """
        Sum of array elements over a given axis.
        
        Parameters
        ----------   
        axis : None or int, optional
            Axis along which a sum is performed. The default, axis=None, will sum
            all of the elements of the input array. If axis is negative it counts 
            from the last to the first axis.
           
         Returns
         -------
         sum_along_axis : ht.tensor
             An array with the same shape as self.__array except for the specified axis which 
             becomes one, e.g. a.shape = (1,2,3) => ht.ones((1,2,3)).sum(axis=1).shape = (1,1,3)
   
        Examples
        --------
        >>> ht.ones(2).sum()
        tensor([2.])

        >>> ht.ones((3,3)).sum()
        tensor([9.])

        >>> ht.ones((3,3)).astype(ht.int).sum()
        tensor([9])

        >>> ht.ones((3,2,1)).sum(axis=-3)
        tensor([[[3.],
                 [3.]]])
        """
        if axis is not None:
            axis = sanitize_axis(self.shape, axis)
            sum_axis = self.__array.sum(axis, keepdim=True)
  
        else:
            sum_axis = torch.reshape(self.__array.sum(), (1,))
            if not self.__comm.is_distributed():
                return tensor(sum_axis, (1,), types.canonical_heat_type(sum_axis.dtype), self.split, _copy(self.__comm))
            
        return self.__reduce_op(sum_axis, mpi.reduce_op.SUM, axis)

    def expand_dims(self, axis):
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        # TODO: make me more numpy API complete
        # TODO: fix negative axis
        return tensor(
            self.__array.unsqueeze(dim=axis),
            self.shape[:axis] + (1,) + self.shape[axis:],
            self.dtype,
            self.split if self.split is None or self.split < axis else self.split + 1,
            _copy(self.__comm)
        )

    def exp(self, out=None):
        """
        Calculate the exponential of all elements in the input array.
        Parameters
        ----------
        out : ht.tensor or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.
        Returns
        -------
        exponentials : ht.tensor
            A tensor of the same shape as x, containing the positive exponentials of each element in this tensor. If out
            was provided, logarithms is a reference to it.
        Examples
        --------
        >>> ht.arange(5).exp()
        tensor([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981])
        """
        return operations.exp(self, out)

    def floor(self, out=None):
        r"""
        Return the floor of the input, element-wise.
        The floor of the scalar x is the largest integer i, such that i <= x. It is often denoted as :math:`\lfloor x \rfloor`.
        Parameters
        ----------
        out : ht.tensor or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.
        Returns
        -------
        floored : ht.tensor
            A tensor of the same shape as x, containing the floored valued of each element in this tensor. If out was
            provided, logarithms is a reference to it.
        Examples
        --------
        >>> ht.floor(ht.arange(-2.0, 2.0, 0.4))
        tensor([-2., -2., -2., -1., -1.,  0.,  0.,  0.,  1.,  1.])
        """
        return operations.floor(self, out)

    def log(self, out=None):
        """
        Natural logarithm, element-wise.
        The natural logarithm log is the inverse of the exponential function, so that log(exp(x)) = x. The natural
        logarithm is logarithm in base e.
        Parameters
        ----------
        out : ht.tensor or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.
        Returns
        -------
        logarithms : ht.tensor
            A tensor of the same shape as x, containing the positive logarithms of each element in this tensor.
            Negative input elements are returned as nan. If out was provided, logarithms is a reference to it.
        Examples
        --------
        >>> ht.arange(5).log()
        tensor([  -inf, 0.0000, 0.6931, 1.0986, 1.3863])
        """
        return operations.log(self, out)

    def sin(self, out=None):
        """
        Return the trigonometric sine, element-wise.
        Parameters
        ----------
        out : ht.tensor or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.
        Returns
        -------
        sine : ht.tensor
            A tensor of the same shape as x, containing the trigonometric sine of each element in this tensor.
            Negative input elements are returned as nan. If out was provided, square_roots is a reference to it.
        Examples
        --------
        >>> ht.arange(-6, 7, 2).sin()
        tensor([ 0.2794,  0.7568, -0.9093,  0.0000,  0.9093, -0.7568, -0.2794])
        """
        return operations.sin(self, out)

    def sqrt(self, out=None):
        """
        Return the non-negative square-root of the tensor element-wise.
        Parameters
        ----------
        out : ht.tensor or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.
        Returns
        -------
        square_roots : ht.tensor
            A tensor of the same shape as x, containing the positive square-root of each element in this tensor.
            Negative input elements are returned as nan. If out was provided, square_roots is a reference to it.
        Examples
        --------
        >>> ht.arange(5).sqrt()
        tensor([0.0000, 1.0000, 1.4142, 1.7321, 2.0000])
        >>> ht.arange(-5, 0).sqrt()
        tensor([nan, nan, nan, nan, nan])
        """
        return operations.sqrt(self, out)

    def __binop(self, op, other):
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        # TODO: make me more numpy API complete
        # TODO: ... including the actual binops
        if np.isscalar(other):
            return tensor(op(self.__array, other), self.shape, self.dtype, self.split, _copy(self.__comm))

        elif isinstance(other, tensor):
            output_shape = broadcast_shape(self.shape, other.shape)

            # TODO: implement complex NUMPY rules
            if other.dtype != self.dtype:
                other = other.astype(self.dtype)

            if other.split is None or other.split == self.split:
                return tensor(op(self.__array, other.__array), output_shape, self.dtype, self.split, _copy(self.__comm))
            else:
                raise NotImplementedError('Not implemented for other splittings')
        else:
            raise NotImplementedError('Not implemented for non scalar')

    def __add__(self, other):
        return self.__binop(operator.add, other)

    def __sub__(self, other):
        return self.__binop(operator.sub, other)

    def __truediv__(self, other):
        return self.__binop(operator.truediv, other)

    def __mul__(self, other):
        return self.__binop(operator.mul, other)

    def __pow__(self, other):
        return self.__binop(operator.pow, other)

    def __eq__(self, other):
        return self.__binop(operator.eq, other)

    def __ne__(self, other):
        return self.__binop(operator.ne, other)

    def __lt__(self, other):
        return self.__binop(operator.lt, other)

    def __le__(self, other):
        return self.__binop(operator.le, other)

    def __gt__(self, other):
        return self.__binop(operator.gt, other)

    def __ge__(self, other):
        return self.__binop(operator.ge, other)

    def __str__(self, *args):
        # TODO: document me
        # TODO: generate none-PyTorch str
        return self.__array.__str__(*args)

    def __repr__(self, *args):
        # TODO: document me
        # TODO: generate none-PyTorch repr
        return self.__array.__repr__(*args)

    def __getitem__(self, key):
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        # TODO: make me more numpy API complete
        return tensor(self.__array[key], self.shape, self.split, _copy(self.__comm))

    def __setitem__(self, key, value):
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        # TODO: make me more numpy API complete
        if self.__split is not None:
            raise NotImplementedError('Slicing not supported for __split != None')

        if np.isscalar(value):
            self.__array.__setitem__(key, value)
        elif isinstance(value, tensor):
            self.__array.__setitem__(key, value.__array)
        else:
            raise NotImplementedError('Not implemented for {}'.format(value.__class__.__name__))

    def __prephalo(self, start, end):
        """
        Extracts the halo indexed by start, end from self.array in the direction of self.split

        Parameters
        ----------
        start : int 
            start index of the halo extracted from self.array
        end : int
            end index of the halo extracted from self.array
        Returns
        -------
        halo : torch tensor
            The halo extracted from self.array
        """
        if not isinstance(start, int) and start is not None:
            raise TypeError('start needs to be of Python type integer, {} given)'.format(type(start)))
        if not isinstance(end, int) and end is not None: 
            raise TypeError('end needs to be of Python type integer, {} given)'.format(type(end)))

        ix = [slice(None, None, None)] * len(self.shape)
        try: 
            ix[self.split] = slice(start, end)
        except IndexError:
            print('Indices out of bound')

        return self.array[ix].contiguous()

    def is_distributed(self):
        """
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        Returns
        -------
        
        """ 
        return self.comm.size > 1 and self.split is not None

    def halo_next_shape(self): 
        """
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        Returns
        -------
        
        """ 
        if self.halo_next is not None:
            return tuple(self.halo_next.size())
        else:
            return None
 
    def halo_prev_shape(self):  
        """
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        Returns
        -------
        
        """
        if self.halo_prev is not None:
            return tuple(self.halo_prev.size())
        else:
            return None

    def halo_prev_length(self): 
        """
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        Returns
        -------
        
        """
        if self.halo_prev_shape() is not None:
            return self.halo_prev_shape()[self.split]
        else:
            return None 

    def halo_next_length(self): 
        """
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        Returns
        -------
        
        """
        if self.halo_next_shape() is not None:
            return self.halo_next_shape()[self.split]
        else:
            return None 

    def sanitize_halo(self, halo_size):
        """
        In case the of a distributed and splitted tensor, the size will reduced 
        to the smallest chunk if halo_size

        Parameters
        ----------
        halo_size : int 
            Size of the halo. If halo_size exceeds the size of the HeAT tensor in self.split direction
            and the tensor is distributed halo_size will be reduced to the smallest chunk size. 

        Returns
        -------
        halo_size : int
            Sanitized halo size 
        """
        if not isinstance(halo_size, int): 
            raise TypeError('halo_size needs to be of Python type integer, {} given)'.format(type(halo_size)))
        if halo_size < 0: 
            raise ValueError('halo_size needs to be a positive Python integer, {} given)'.format(type(halo_size)))

        if self.is_distributed():

            min_chunksize = self.shape[self.split]//self.comm.size

            if halo_size > min_chunksize:
                warnings.warn('Your halo is larger than the smallest local data array, '
                              'only the local data array will be exchanged')
                halo_size = min_chunksize
 
        return halo_size

    def gethalo(self, halo_size):
        """
        Fetch halos of size 'halo_size' from neighboring ranks and save them in self.halo_next/self.halo_prev
        in case they are not already stored. If 'halo_size' differs from the size of already stored halos,
        the are overwritten. 

        Parameters
        ----------
        halo_size : int 
            Size of the halo. If halo_size exceeds the size of the HeAT tensor in self.split direction
            the whole local tensor.array will be fetched 

        Returns
        -------
        None
        """
        halo_size = self.sanitize_halo(halo_size)

        if self.is_distributed() and halo_size > 0:
            
            a_prev = self.__prephalo(0, halo_size)
            a_next = self.__prephalo(-halo_size, None)
            
            res_prev = None
            res_next = None
           
            if self.comm.rank != self.comm.size-1:
                mpi.send(a_next, dst=self.comm.rank+1)
                res_prev = torch.zeros(a_prev.size(), dtype=a_prev.dtype)
                mpi.recv(res_prev, src=self.comm.rank+1) 

            if self.comm.rank != 0:
                mpi.send(a_prev, dst=self.comm.rank-1)
                res_next = torch.zeros(a_next.size(), dtype=a_next.dtype)
                mpi.recv(res_next, src=self.comm.rank-1) 

            self.halo_prev = res_prev
            self.halo_next = res_next

    def genpad(self, signal, pad_prev, pad_next):
        """
        Generate a zero padding only for local arrays of the first and last rank in case of distributed computing,
        otherwise zero padding is added at begin and end of the global array

        Parameters
        ----------
        signal : torch tensor 
            The data array to which a zero padding is to be added
        pad_prev : int
            The length of the left padding area
        pad_next : int
            The length of the right padding area
            
        Returns
        -------
        out : torch tensor
            The padded data array 
        """
        # ToDo: generalize to ND tensors
        # set the default padding for non distributed arrays

        if len(signal.shape) != 1:
            raise ValueError('Signal must be 1D, but is {}-dimensional'.format(len(signal.shape)))

        # check if more than one rank is involved 
        if self.is_distributed():
            pad_next = None
            pad_prev = None

            # set the padding of the first rank
            if self.comm.rank == 0:
                pad_next = None

            # set the padding of the last rank
            if self.comm.rank == self.comm.size-1:
                pad_prev = None
                    
        return torch.cat(tuple(_ for _ in (pad_prev, signal, pad_next) if _ is not None))


def __factory(shape, dtype, split, local_factory):
    """
    Abstracted factory function for HeAT tensor initialization.

    Parameters
    ----------
    shape : int or sequence of ints
        Desired shape of the output array, e.g. 1 or (1, 2, 3,).
    dtype : ht.dtype
        The desired HeAT data type for the array, defaults to ht.float32.
    split : int
        The axis along which the array is split and distributed.
    local_factory : function
        Function that creates the local PyTorch tensor for the HeAT tensor.

    Returns
    -------
    out : ht.tensor
        Array of ones with given shape, data type and node distribution.
    """
    # clean the user input
    shape = sanitize_shape(shape)
    dtype = types.canonical_heat_type(dtype)
    split = sanitize_axis(shape, split)

    # chunk the shape if necessary
    comm = MPICommunicator() if split is not None else NoneCommunicator()
    _, local_shape, _ = comm.chunk(shape, split)

    return tensor(local_factory(local_shape, dtype=dtype.torch_type()), shape, dtype, split, comm)


def __factory_like(a, dtype, split, factory):
    """
    Abstracted '...-like' factory function for HeAT tensor initialization

    Parameters
    ----------
    a : object
        The shape and data-type of 'a' define these same attributes of the returned array.
    dtype : ht.dtype
        The desired HeAT data type for the array, defaults to ht.float32.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    factory : function
        Function that creates a HeAT tensor.

    Returns
    -------
    out : ht.tensor
        Array of ones with given shape, data type and node distribution that is like a
    """
    # determine the global shape of the object to create
    # attempt in this order: shape property, length of object or default shape (1,)
    try:
        shape = a.shape
    except AttributeError:
        try:
            shape = (len(a),)
        except TypeError:
            shape = (1,)

    # infer the data type, otherwise default to float32
    if dtype is None:
        try:
            dtype = types.heat_type_of(a)
        except TypeError:
            dtype = types.float32

    # infer split axis
    if split is None:
        try:
            split = a.split if not isinstance(a, str) else None
        except AttributeError:
            # do not split at all
            pass

    return factory(shape, dtype, split)


def arange(*args, dtype=None, split=None):
    """
    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range <http://docs.python.org/lib/built-in-funcs.html>`_ function,
    but returns a tensor rather than a list.

    When using a non-integer step, such as 0.1, the results will often not
    be consistent.  It is better to use ``linspace`` for these cases.

    Parameters
    ----------
    start : number, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : number
        End of interval.  The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    step : number, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified as a position argument,
        `start` must also be given.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).

    Returns
    -------
    arange : 1D heat tensor
        1D heat tensor of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.

    See Also
    --------
    linspace : Evenly spaced numbers with careful handling of endpoints.

    Examples
    --------
    >>> ht.arange(3)
    tensor([0, 1, 2])
    >>> ht.arange(3.0)
    tensor([ 0.,  1.,  2.])
    >>> ht.arange(3,7)
    tensor([3, 4, 5, 6])
    >>> ht.arange(3,7,2)
    tensor([3, 5])
    """
    num_of_param = len(args)

    # check if all positional arguments are integers
    all_ints = all([isinstance(_, int) for _ in args])

    # set start, stop, step, num according to *args
    if num_of_param == 1:
        if dtype is None:
            # use int32 as default instead of int64 used in numpy
            dtype = types.int32
        start = 0
        stop = int(np.ceil(args[0]))
        step = 1
        num = stop
    elif num_of_param == 2:
        if dtype is None:
            dtype = types.int32 if all_ints else types.float32
        start = args[0]
        stop = args[1]
        step = 1
        num = int(np.ceil(stop - start))
    elif num_of_param == 3:
        if dtype is None:
            dtype = types.int32 if all_ints else types.float32
        start = args[0]
        stop = args[1]
        step = args[2]
        num = int(np.ceil((stop - start) / step))
    else:
        raise TypeError('function takes minimum one and at most 3 positional arguments ({} given)'.format(num_of_param))

    gshape = (num,)
    split = sanitize_axis(gshape, split)
    comm = MPICommunicator() if split is not None else NoneCommunicator()
    offset, lshape, _ = comm.chunk(gshape, split)

    # compose the local tensor
    start += offset * step
    stop = start + lshape[0] * step
    data = torch.arange(start, stop, step, dtype=types.canonical_heat_type(dtype).torch_type())

    return tensor(data, gshape, types.canonical_heat_type(data.dtype), split, comm)


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, split=None):
    """
    Returns num evenly spaced samples, calculated over the interval [start, stop]. The endpoint of the interval can
    optionally be excluded.

    Parameters
    ----------
    start: scalar, scalar-convertible
        The starting value of the sample interval, maybe a sequence if convertible to scalar
    end: scalar, scalar-convertible
        The end value of the sample interval, unless is set to False. In that case, the sequence consists of all but the
        last of num + 1 evenly spaced samples, so that stop is excluded. Note that the step size changes when endpoint
        is False.
    num: int, optional
        Number of samples to generate, defaults to 50. Must be non-negative.
    endpoint: bool, optional
        If True, stop is the last sample, otherwise, it is not included. Defaults to True.
    retstep: bool, optional
        If True, return (samples, step), where step is the spacing between samples.
    dtype: dtype, optional
        The type of the output array.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).

    Returns
    -------
    samples: ht.tensor
        There are num equally spaced samples in the closed interval [start, stop] or the half-open interval
        [start, stop) (depending on whether endpoint is True or False).
    step: float, optional
        Size of spacing between samples, only returned if retstep is True.

    Examples
    --------
    >>> np.linspace(2.0, 3.0, num=5)
    tensor([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
    >>> np.linspace(2.0, 3.0, num=5, endpoint=False)
    tensor([ 2. ,  2.2,  2.4,  2.6,  2.8])
    >>> np.linspace(2.0, 3.0, num=5, retstep=True)
    (array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)
    """
    # sanitize input parameters
    start = float(start)
    stop = float(stop)
    num = int(num)
    if num <= 0:
        raise ValueError('number of samples \'num\' must be non-negative integer, but was {}'.format(num))
    step = (stop - start) / max(1, num - 1 if endpoint else num)

    # infer local and global shapes
    gshape = (num,)
    split = sanitize_axis(gshape, split)
    comm = MPICommunicator() if split is not None else NoneCommunicator()
    offset, lshape, _ = comm.chunk(gshape, split)

    # compose the local tensor
    start += offset * step
    stop = start + lshape[0] * step - step
    data = torch.linspace(start, stop, lshape[0])
    if dtype is not None:
        data = data.type(types.canonical_heat_type(dtype).torch_type())

    # construct the resulting global tensor
    ht_tensor = tensor(data, gshape, types.canonical_heat_type(data.dtype), split, comm)

    if retstep:
        return ht_tensor, step
    return ht_tensor


def ones(shape, dtype=types.float32, split=None):
    """
    Returns a new array of given shape and data type filled with one values. May be allocated split up across multiple
    nodes along the specified axis.

    Parameters
    ----------
    shape : int or sequence of ints
        Desired shape of the output array, e.g. 1 or (1, 2, 3,).
    dtype : ht.dtype
        The desired HeAT data type for the array, defaults to ht.float32.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).

    Returns
    -------
    out : ht.tensor
        Array of ones with given shape, data type and node distribution.

    Examples
    --------
    >>> ht.ones(3)
    tensor([1., 1., 1.])

    >>> ht.ones(3, dtype=ht.int)
    tensor([1, 1, 1])

    >>> ht.ones((2, 3,))
    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    """
    return __factory(shape, dtype, split, torch.ones)


def ones_like(a, dtype=None, split=None):
    """
    Returns a new array filled with ones with the same type, shape and data distribution of given object. Data type and
    data distribution strategy can be explicitly overriden.

    Parameters
    ----------
    a : object
        The shape and data-type of 'a' define these same attributes of the returned array.
    dtype : ht.dtype, optional
        Overrides the data type of the result.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).

    Returns
    -------
    out : ht.tensor
        Array of ones with the same shape, type and split axis as 'a' unless overriden.

    Examples
    --------
    >>> x = ht.zeros((2, 3,))
    >>> x
    tensor([[0., 0., 0.],
            [0., 0., 0.]])

    >>> ht.ones_like(a)
    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    """
    return __factory_like(a, dtype, split, ones)


def randn(*args, dtype = torch.float32, split = None):
    """
    #based on ht.arange, ht.linspace implementation
    BASIC FUNCTIONALITY:
    Returns a tensor filled with random numbers from a normal distribution 
    with zero mean and variance of one.

    The shape of the tensor is defined by the varargs args.
    
    Parameters	
    ----------

    args (int...) – a set of integers defining the shape of the output tensor.
    #TODO: out (Tensor, optional) – the output tensor


    Examples
    --------
    >>> ht.randn(3)
    tensor([ 0.1921, -0.9635,  0.5047])

    >>> ht.randn(4,4)
    tensor([[-1.1261,  0.5971,  0.2851,  0.9998],
            [-1.8548, -1.2574,  0.2391, -0.3302],
            [ 1.3365, -1.5212,  1.4159, -0.1671],
            [ 0.1260,  1.2126, -0.0804,  0.0907]])
    """
    num_of_param = len(args)

    # check if all positional arguments are integers and greater than zero
    all_ints = all(isinstance(_, int) for _ in args)
    if not all_ints:
        raise TypeError("Only integer-valued dimensions as arguments possible")
    all_positive = all(_ > 0 for _ in args)
    if not all_positive:
        raise ValueError("Not all tensor dimensions are positive")

    # define shape of tensor according to args
    gshape = (args)
    split = sanitize_axis(gshape, split)
    comm = MPICommunicator() if split is not None else NoneCommunicator()
    offset, lshape, _ = comm.chunk(gshape, split)

    #TODO: double-check np.randn/torch.randn overlap

    try:
        torch.randn(args)
    except RuntimeError as exception:
        # re-raise the exception to be consistent with numpy's exception interface
        raise ValueError(str(exception))
    # compose the local tensor
    data = torch.randn(args)

    return tensor(data, gshape, types.canonical_heat_type(data.dtype), split, comm)
    

def zeros(shape, dtype=types.float32, split=None):
    """
    Returns a new array of given shape and data type filled with zero values. May be allocated split up across multiple
    nodes along the specified axis.

    Parameters
    ----------
    shape : int or sequence of ints
        Desired shape of the output array, e.g. 1 or (1, 2, 3,).
    dtype : ht.dtype
        The desired HeAT data type for the array, defaults to ht.float32.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).

    Returns
    -------
    out : ht.tensor
        Array of zeros with given shape, data type and node distribution.

    Examples
    --------
    >>> ht.zeros(3)
    tensor([0., 0., 0.])

    >>> ht.zeros(3, dtype=ht.int)
    tensor([0, 0, 0])

    >>> ht.zeros((2, 3,))
    tensor([[0., 0., 0.],
            [0., 0., 0.]])
    """
    return __factory(shape, dtype, split, torch.zeros)


def zeros_like(a, dtype=None, split=None):
    """
    Returns a new array filled with zeros with the same type, shape and data distribution of given object. Data type and
    data distribution strategy can be explicitly overriden.

    Parameters
    ----------
    a : object
        The shape and data-type of 'a' define these same attributes of the returned array.
    dtype : ht.dtype, optional
        Overrides the data type of the result.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).

    Returns
    -------
    out : ht.tensor
        Array of zeros with the same shape, type and split axis as 'a' unless overriden.

    Examples
    --------
    >>> x = ht.ones((2, 3,))
    >>> x
    tensor([[1., 1., 1.],
            [1., 1., 1.]])

    >>> ht.zeros_like(a)
    tensor([[0., 0., 0.],
            [0., 0., 0.]])
    """
    return __factory_like(a, dtype, split, zeros)


def convolve(a, v, mode='full'):
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
          Mode 'same' returns output of length 'N'. Boundary
          effects are still visible.
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
        Secondly, only float types are allowed in contrast to numpy where also integer types 
        can be utilized.
           
    Examples
    --------
    Note how the convolution operator flips the second array
    before "sliding" the two across one another:
    >>> a = ht.ones(10)
    >>> v = ht.arange(3).astype(ht.float)
    >>> ht.convolve(a, v, mode='full')
    tensor([0., 1., 3., 3., 3., 3., 2.])

    Only return the middle values of the convolution.
    Contains boundary effects, where zeros are taken
    into account:
    >>> ht.convolve(a, v, mode='same')
    tensor([1., 3., 3., 3., 3.])

    Compute only positions where signal and filter weight
    completely overlap:
    >>> ht.convolve(a, v, mode='valid')
    tensor([3., 3., 3.])
    """
    if not isinstance(a, tensor) or not isinstance(v, tensor):
        raise TypeError('Signal and filter weight must be of type ht.tensor')
    if v.split is not None: 
        raise TypeError('Distributed filter weights are not supported')
    if len(a.shape) != 1 or len(v.shape) != 1: 
        raise ValueError("Only 1 dimensional input tensors are allowed")
    if a.shape[0] <= v.shape[0]: 
        raise ValueError("Filter size must not be larger than signal size")
    if not a.dtype.isfloat() or not v.dtype.isfloat():
        raise TypeError('Only float type tensors are supported')
    if a.dtype is not v.dtype:
        raise TypeError('Signal and filter weight must be of same type')
    
    # compute halo size 
    halo_size = (v.shape[0]-1)//2

    # fetch halos and store them in a.halo_next/a.halo_prev
    a.gethalo(halo_size)

    # apply halos to local array
    signal = torch.cat(tuple(_ for _ in (a.halo_prev, a.array, a.halo_next) if _ is not None))

    # check if a local chunk is smaller than the filter size 
    if a.is_distributed() and signal.size()[0] < v.shape[0]:
        raise ValueError("Local chunk size is smaller than filter size, this is not supported yet")
    
    if mode == 'full': 
        pad_prev = pad_next = torch.zeros(v.shape[0]-1, dtype=a.dtype.torch_type())
        gshape = v.shape[0] + a.shape[0] - 1

    elif mode == 'same':  
        if v.shape[0] % 2 == 0:
            pad_prev = torch.zeros(halo_size+1, dtype=a.dtype.torch_type())
            pad_next = torch.zeros(halo_size, dtype=a.dtype.torch_type())
        else: 
            pad_prev = pad_next = torch.zeros(halo_size, dtype=a.dtype.torch_type())

        gshape = a.shape[0]

    elif mode == 'valid': 
        pad_prev = pad_next = None
        gshape = a.shape[0] - v.shape[0] + 1

    else:
        raise ValueError("Only {'full', 'valid', 'same'} are allowed for mode") 

    # add padding to the borders according to mode
    signal = a.genpad(signal, pad_prev, pad_next)

    # make signal and filter weight 3D for Pytorch conv1d function        
    signal.unsqueeze_(0)
    signal.unsqueeze_(0)

    # flip filter for convolution as Pytorch conv1d computes correlations
    weight = v.array.clone()
    idx = torch.LongTensor([i for i in range(weight.size(0)-1, -1, -1)])
    weight = weight.index_select(0, idx)
    weight.unsqueeze_(0)
    weight.unsqueeze_(0)

    # apply torch convolution operator 
    signal_filtered = fc.conv1d(signal, weight) 

    return tensor(signal_filtered[0, 0, :], (gshape,), signal_filtered.dtype, a.split, a.comm)
