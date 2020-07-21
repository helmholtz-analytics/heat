from __future__ import annotations

import numpy as np
import math
import torch
import warnings
from typing import List, Union

from . import arithmetics
from . import devices
from . import exponential
from . import factories
from . import indexing
from . import io
from . import linalg
from . import logical
from . import manipulations
from . import memory
from . import printing
from . import relational
from . import rounding
from . import statistics
from . import stride_tricks
from . import tiling
from . import trigonometrics
from . import types

from .communication import MPI
from .stride_tricks import sanitize_axis

warnings.simplefilter("always", ResourceWarning)

__all__ = ["DNDarray"]


class LocalIndex:
    """
    Indexing class for local operations (primarily for lloc function)
    For docs on __getitem__ and __setitem__ see lloc(self)
    """

    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, key):
        return self.obj[key]

    def __setitem__(self, key, value):
        self.obj[key] = value


class DNDarray:
    """

    Distributed N-Dimensional array. The core element of HeAT. It is composed of
    PyTorch tensors local to each process.

    Parameters
    ----------
    array : torch.Tensor
        local array elements
    gshape : tuple
        the global shape of the DNDarray
    dtype : ht.type
        the datatype of the array
    split : int
        The axis on which the DNDarray is divided between processes
    device : ht.device
        The device on which the local arrays are using (cpu or gpu)
    comm : ht.communication.MPICommunication
        The communications object for sending and recieving data
    """

    def __init__(self, array, gshape, dtype, split, device, comm):
        self.__array = array
        self.__gshape = gshape
        self.__dtype = dtype
        self.__split = split
        self.__device = device
        self.__comm = comm
        self.__ishalo = False
        self.__halo_next = None
        self.__halo_prev = None

        # handle inconsistencies between torch and heat devices
        if (
            isinstance(array, torch.Tensor)
            and isinstance(device, devices.Device)
            and array.device.type != device.device_type
        ):
            self.__array = self.__array.to(devices.sanitize_device(self.__device).torch_device)

    @property
    def halo_next(self):
        return self.__halo_next

    @property
    def halo_prev(self):
        return self.__halo_prev

    @property
    def comm(self):
        return self.__comm

    @property
    def device(self):
        return self.__device

    @property
    def dtype(self):
        return self.__dtype

    @property
    def gshape(self):
        return self.__gshape

    @property
    def numdims(self):
        """
        Returns
        -------
        number_of_dimensions : int
            the number of dimensions of the DNDarray

        .. deprecated:: 0.5.0
          `numdims` will be removed in HeAT 1.0.0, it is replaced by `ndim` because the latter is numpy API compliant.
        """
        warnings.warn("numdims is deprecated, use ndim instead", DeprecationWarning)
        return len(self.__gshape)

    @property
    def ndim(self):
        """
        Returns
        -------
        number_of_dimensions : int
            the number of dimensions of the DNDarray
        """
        return len(self.__gshape)

    @property
    def size(self):
        """

        Returns
        -------
        size : int
            number of total elements of the tensor
        """
        return torch.prod(torch.tensor(self.gshape, device=self.device.torch_device)).item()

    @property
    def gnumel(self):
        """

        Returns
        -------
        global_shape : int
            number of total elements of the tensor
        """
        return self.size

    @property
    def lnumel(self):
        """

        Returns
        -------
        number_of_local_elements : int
            number of elements of the tensor on each process
        """
        return np.prod(self.__array.shape)

    @property
    def lloc(self):
        """

        Local item setter and getter. i.e. this function operates on a local
        level and only on the PyTorch tensors composing the HeAT DNDarray.
        This function uses the LocalIndex class.

        Parameters
        ----------
        key : int, slice, list, tuple
            indices of the desired data
        value : all types compatible with pytorch tensors
            optional (if none given then this is a getter function)

        Returns
        -------
        (getter) -> ht.DNDarray with the indices selected at a *local* level
        (setter) -> nothing

        Examples
        --------
        (2 processes)
        >>> a = ht.zeros((4, 5), split=0)
        (1/2) tensor([[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]])
        (2/2) tensor([[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]])
        >>> a.lloc[1, 0:4]
        (1/2) tensor([0., 0., 0., 0.])
        (2/2) tensor([0., 0., 0., 0.])
        >>> a.lloc[1, 0:4] = torch.arange(1, 5)
        >>> a
        (1/2) tensor([[0., 0., 0., 0., 0.],
                      [1., 2., 3., 4., 0.]])
        (2/2) tensor([[0., 0., 0., 0., 0.],
                      [1., 2., 3., 4., 0.]])
        """
        return LocalIndex(self.__array)

    @property
    def lshape(self):
        """

        Returns
        -------
        local_shape : tuple
            the shape of the data on each node
        """
        return tuple(self.__array.shape)

    @property
    def shape(self):
        """

        Returns
        -------
        global_shape : tuple
            the shape of the tensor as a whole
        """
        return self.__gshape

    @property
    def split(self):
        """

        Returns
        -------
        split_axis : int
            the axis on which the tensor split
        """
        return self.__split

    @property
    def stride(self):
        """

        Returns
        -------
        stride : tuple
            steps in each dimension when traversing a tensor.
            torch-like usage: self.stride()
        """
        return self.__array.stride

    @property
    def strides(self):
        """

        Returns
        -------
        strides : tuple of ints
            bytes to step in each dimension when traversing a tensor.
            numpy-like usage: self.strides
        """
        steps = list(self._DNDarray__array.stride())
        itemsize = self._DNDarray__array.storage().element_size()
        strides = tuple(step * itemsize for step in steps)
        return strides

    @property
    def T(self):
        return linalg.transpose(self, axes=None)

    @property
    def array_with_halos(self):
        return self.__cat_halo()

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
        halo : torch.Tensor
            The halo extracted from self.array
        """
        ix = [slice(None, None, None)] * len(self.shape)
        try:
            ix[self.split] = slice(start, end)
        except IndexError:
            print("Indices out of bound")

        return self.__array[ix].clone().contiguous()

    def get_halo(self, halo_size):
        """
        Fetch halos of size 'halo_size' from neighboring ranks and save them in self.halo_next/self.halo_prev
        in case they are not already stored. If 'halo_size' differs from the size of already stored halos,
        the are overwritten.

        Parameters
        ----------
        halo_size : int
            Size of the halo.
        """
        if not isinstance(halo_size, int):
            raise TypeError(
                "halo_size needs to be of Python type integer, {} given)".format(type(halo_size))
            )
        if halo_size < 0:
            raise ValueError(
                "halo_size needs to be a positive Python integer, {} given)".format(type(halo_size))
            )

        if self.comm.is_distributed() and self.split is not None:
            min_chunksize = self.shape[self.split] // self.comm.size
            if halo_size > min_chunksize:
                raise ValueError(
                    "halo_size {} needs to smaller than chunck-size {} )".format(
                        halo_size, min_chunksize
                    )
                )

            a_prev = self.__prephalo(0, halo_size)
            a_next = self.__prephalo(-halo_size, None)

            res_prev = None
            res_next = None

            req_list = list()

            if self.comm.rank != self.comm.size - 1:
                self.comm.Isend(a_next, self.comm.rank + 1)
                res_prev = torch.zeros(a_prev.size(), dtype=a_prev.dtype)
                req_list.append(self.comm.Irecv(res_prev, source=self.comm.rank + 1))

            if self.comm.rank != 0:
                self.comm.Isend(a_prev, self.comm.rank - 1)
                res_next = torch.zeros(a_next.size(), dtype=a_next.dtype)
                req_list.append(self.comm.Irecv(res_next, source=self.comm.rank - 1))

            for req in req_list:
                req.wait()

            self.__halo_next = res_prev
            self.__halo_prev = res_next
            self.__ishalo = True

    def __cat_halo(self):
        """
        Fetch halos of size 'halo_size' from neighboring ranks and save them in self.halo_next/self.halo_prev
        in case they are not already stored. If 'halo_size' differs from the size of already stored halos,
        the are overwritten.

        Parameters
        ----------
        None

        Returns
        -------
        array + halos: pytorch tensors
        """
        return torch.cat(
            [_ for _ in (self.__halo_prev, self.__array, self.__halo_next) if _ is not None],
            self.split,
        )

    def abs(self, out=None, dtype=None):
        """
        Calculate the absolute value element-wise.

        Parameters
        ----------
        out : ht.DNDarray, optional
            A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
            If not provided or None, a freshly-allocated array is returned.
        dtype : ht.type, optional
            Determines the data type of the output array. The values are cast to this type with potential loss of
            precision.

        Returns
        -------
        absolute_values : ht.DNDarray
            A tensor containing the absolute value of each element in x.
        """
        return rounding.abs(self, out, dtype)

    def absolute(self, out=None, dtype=None):
        """
        Calculate the absolute value element-wise.

        ht.abs is a shorthand for this function.

        Parameters
        ----------
        out : ht.DNDarray, optional
            A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
            If not provided or None, a freshly-allocated array is returned.
        dtype : ht.type, optional
            Determines the data type of the output array. The values are cast to this type with potential loss of
            precision.

        Returns
        -------
        absolute_values : ht.DNDarray
            A tensor containing the absolute value of each element in x.

        """
        return self.abs(out, dtype)

    def __add__(self, other):
        """
        Element-wise addition of another tensor or a scalar to the tensor.
        Takes the second operand (scalar or tensor) whose elements are to be added as argument.

        Parameters
        ----------
        other: tensor or scalar
            The value(s) to be added element-wise to the tensor

        Returns
        -------
        result: ht.DNDarray
            A tensor containing the results of element-wise addition.

        Examples:
        ---------
        >>> import heat as ht
        >>> T1 = ht.float32([[1, 2], [3, 4]])
        >>> T1.__add__(2.0)
        tensor([[3., 4.],
               [5., 6.]])

        >>> T2 = ht.float32([[2, 2], [2, 2]])
        >>> T1.__add__(T2)
        tensor([[3., 4.],
                [5., 6.]])
        """
        return arithmetics.add(self, other)

    def all(self, axis=None, out=None, keepdim=None):
        """
        Test whether all array elements along a given axis evaluate to True.

        Parameters:
        -----------
        axis : None or int or tuple of ints, optional
            Axis or axes along which a logical AND reduction is performed. The default (axis = None) is to perform a
            logical AND over all the dimensions of the input array. axis may be negative, in which case it counts
            from the last to the first axis.
        out : ht.DNDarray, optional
            Alternate output array in which to place the result. It must have the same shape as the expected output
            and its type is preserved.

        Returns:
        --------
        all : ht.DNDarray, bool
            A new boolean or ht.DNDarray is returned unless out is specified, in which case a reference to out is returned.

        Examples:
        ---------
        >>> import heat as ht
        >>> a = ht.random.randn(4,5)
        >>> a
        tensor([[ 0.5370, -0.4117, -3.1062,  0.4897, -0.3231],
                [-0.5005, -1.7746,  0.8515, -0.9494, -0.2238],
                [-0.0444,  0.3388,  0.6805, -1.3856,  0.5422],
                [ 0.3184,  0.0185,  0.5256, -1.1653, -0.1665]])
        >>> x = a < 0.5
        >>> x
        tensor([[0, 1, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 0, 1, 0],
                [1,1, 0, 1, 1]], dtype=torch.uint8)
        >>> x.all()
        tensor([0], dtype=torch.uint8)
        >>> x.all(axis=0)
        tensor([[0, 1, 0, 1, 0]], dtype=torch.uint8)
        >>> x.all(axis=1)
        tensor([[0],
                [0],
                [0],
                [0]], dtype=torch.uint8)

        Write out to predefined buffer:
        >>> out = ht.zeros((1,5))
        >>> x.all(axis=0, out=out)
        >>> out
        tensor([[0, 1, 0, 1, 0]], dtype=torch.uint8)
        """
        return logical.all(self, axis=axis, out=out, keepdim=keepdim)

    def allclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        """
        Test whether self and other are element-wise equal within a tolerance. Returns True if |self - other| <= atol +
        rtol * |other| for all elements, False otherwise.

        Parameters:
        -----------
        other : ht.DNDarray
            Input tensor to compare to
        atol: float, optional
            Absolute tolerance. Defaults to 1e-08
        rtol: float, optional
            Relative tolerance (with respect to y). Defaults to 1e-05
        equal_nan: bool, optional
            Whether to compare NaN’s as equal. If True, NaN’s in a will be considered equal to NaN’s in b in the output array.

        Returns:
        --------
        allclose : bool
            True if the two tensors are equal within the given tolerance; False otherwise.

        Examples:
        ---------
        >>> a = ht.float32([[2, 2], [2, 2]])
        >>> a.allclose(a)
        True

        >>> b = ht.float32([[2.00005,2.00005],[2.00005,2.00005]])
        >>> a.allclose(b)
        False
        >>> a.allclose(b, atol=1e-04)
        True
        """
        return logical.allclose(self, other, rtol, atol, equal_nan)

    def __and__(self, other):
        """
        Compute the bit-wise AND of self and other arrays element-wise.

        Parameters
        ----------
        other: tensor or scalar
            Only integer and boolean types are handled. If self.shape != other.shape, they must be broadcastable to a common shape (which becomes the shape of the output).

        Returns
        -------
        result: ht.DNDarray
            A tensor containing the results of element-wise AND of self and other.

        Examples:
        ---------
        import heat as ht
        >>> ht.array([13]) & 17
        tensor([1])

        >>> ht.array([14]) & ht.array([13])
        tensor([12])

        >>> ht.array([14,3]) & 13
        tensor([12,  1])

        >>> ht.array([11,7]) & ht.array([4,25])
        tensor([0, 1])

        >>> ht.array([2,5,255]) & ht.array([3,14,16])
        tensor([ 2,  4, 16])

        >>> ht.array([True, True]) & ht.array([False, True])
        tensor([False,  True])
        """
        return arithmetics.bitwise_and(self, other)

    def any(self, axis=None, out=None, keepdim=False):
        """
        Test whether any array element along a given axis evaluates to True.
        The returning tensor is one dimensional unless axis is not None.

        Parameters:
        -----------
        axis : int, optional
            Axis along which a logic OR reduction is performed. With axis=None, the logical OR is performed over all
            dimensions of the tensor.
        out : tensor, optional
            Alternative output tensor in which to place the result. It must have the same shape as the expected output.
            The output is a tensor with dtype=bool.

        Returns:
        --------
        boolean_tensor : tensor of type bool
            Returns a tensor of booleans that are 1, if any non-zero values exist on this axis, 0 otherwise.

        Examples:
        ---------
        >>> import heat as ht
        >>> t = ht.float32([[0.3, 0, 0.5]])
        >>> t.any()
        tensor([1], dtype=torch.uint8)
        >>> t.any(axis=0)
        tensor([[1, 0, 1]], dtype=torch.uint8)
        >>> t.any(axis=1)
        tensor([[1]], dtype=torch.uint8)

        >>> t = ht.int32([[0, 0, 1], [0, 0, 0]])
        >>> res = ht.zeros((1, 3), dtype=ht.bool)
        >>> t.any(axis=0, out=res)
        tensor([[0, 0, 1]], dtype=torch.uint8)
        >>> res
        tensor([[0, 0, 1]], dtype=torch.uint8)
        """
        return logical.any(self, axis=axis, out=out, keepdim=keepdim)

    def argmax(self, axis=None, out=None, **kwargs):
        """
        Returns the indices of the maximum values along an axis.

        Parameters:
        ----------
        x : ht.DNDarray
            Input array.
        axis : int, optional
            By default, the index is into the flattened tensor, otherwise along the specified axis.
        out : array, optional
            If provided, the result will be inserted into this tensor. It should be of the appropriate shape and dtype.

        Returns:
        -------
        index_tensor : ht.DNDarray of ints
            Array of indices into the array. It has the same shape as x.shape with the dimension along axis removed.

        Examples:
        --------
        >>> import heat as ht
        >>> import torch
        >>> torch.manual_seed(1)
        >>> a = ht.random.randn(3,3)
        >>> a
        tensor([[-0.5631, -0.8923, -0.0583],
        [-0.1955, -0.9656,  0.4224],
        [ 0.2673, -0.4212, -0.5107]])
        >>> a.argmax()
        tensor([5])
        >>> a.argmax(axis=0)
        tensor([[2, 2, 1]])
        >>> a.argmax(axis=1)
        tensor([[2],
        [2],
        [0]])
        """
        return statistics.argmax(self, axis=axis, out=out, **kwargs)

    def argmin(self, axis=None, out=None, **kwargs):
        """
        Returns the indices of the minimum values along an axis.

        Parameters:
        ----------
        x : ht.DNDarray
            Input array.
        axis : int, optional
            By default, the index is into the flattened tensor, otherwise along the specified axis.
        out : array, optional
            If provided, the result will be inserted into this tensor. It should be of the appropriate shape and dtype.

        Returns:
        -------
        index_tensor : ht.DNDarray of ints
            Array of indices into the array. It has the same shape as x.shape with the dimension along axis removed.

        Examples
        --------
        >>> import heat as ht
        >>> import torch
        >>> torch.manual_seed(1)
        >>> a = ht.random.randn(3,3)
        >>> a
        tensor([[-0.5631, -0.8923, -0.0583],
        [-0.1955, -0.9656,  0.4224],
        [ 0.2673, -0.4212, -0.5107]])
        >>> a.argmin()
        tensor([4])
        >>> a.argmin(axis=0)
        tensor([[0, 1, 2]])
        >>> a.argmin(axis=1)
        tensor([[1],
                [1],
                [2]])
        """
        return statistics.argmin(self, axis=axis, out=out, **kwargs)

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
        casted_tensor : ht.DNDarray
            casted_tensor is a new tensor of the same shape but with given type of this tensor. If copy is True, the
            same tensor is returned instead.
        """
        dtype = types.canonical_heat_type(dtype)
        casted_array = self.__array.type(dtype.torch_type())
        if copy:
            return DNDarray(casted_array, self.shape, dtype, self.split, self.device, self.comm)

        self.__array = casted_array
        self.__dtype = dtype

        return self

    def average(self, axis=None, weights=None, returned=False):
        """
        Compute the weighted average along the specified axis.

        Parameters
        ----------
        x : ht.tensor
            Tensor containing data to be averaged.

        axis : None or int or tuple of ints, optional
            Axis or axes along which to average x.  The default,
            axis=None, will average over all of the elements of the input tensor.
            If axis is negative it counts from the last to the first axis.

            #TODO Issue #351: If axis is a tuple of ints, averaging is performed on all of the axes
            specified in the tuple instead of a single axis or all the axes as
            before.

        weights : ht.tensor, optional
            An tensor of weights associated with the values in x. Each value in
            x contributes to the average according to its associated weight.
            The weights tensor can either be 1D (in which case its length must be
            the size of x along the given axis) or of the same shape as x.
            If weights=None, then all data in x are assumed to have a
            weight equal to one, the result is equivalent to ht.mean(x).

        returned : bool, optional
            Default is False. If True, the tuple (average, sum_of_weights)
            is returned, otherwise only the average is returned.
            If weights=None, sum_of_weights is equivalent to the number of
            elements over which the average is taken.

        Returns
        -------
        average, [sum_of_weights] : ht.tensor or tuple of ht.tensors
            Return the average along the specified axis. When returned=True,
            return a tuple with the average as the first element and the sum
            of the weights as the second element. sum_of_weights is of the
            same type as `average`.

        Raises
        ------
        ZeroDivisionError
            When all weights along axis are zero.

        TypeError
            When the length of 1D weights is not the same as the shape of x
            along axis.


        Examples
        --------
        >>> data = ht.arange(1,5, dtype=float)
        >>> data
        tensor([1., 2., 3., 4.])
        >>> data.average()
        tensor(2.5000)
        >>> ht.arange(1,11, dtype=float).average(weights=ht.arange(10,0,-1))
        tensor([4.])
        >>> data = ht.array([[0, 1],
                             [2, 3],
                            [4, 5]], dtype=float, split=1)
        >>> weights = ht.array([1./4, 3./4])
        >>> data.average(axis=1, weights=weights)
        tensor([0.7500, 2.7500, 4.7500])
        >>> data.average(weights=weights)
        Traceback (most recent call last):
        ...
        TypeError: Axis must be specified when shapes of x and weights differ.
        """
        return statistics.average(self, axis=axis, weights=weights, returned=returned)

    def balance_(self):
        """
        Function for balancing a DNDarray between all nodes. To determine if this is needed use the is_balanced function.
        If the DNDarray is already balanced this function will do nothing. This function modifies the DNDarray itself and will not return anything.

        Examples
        --------
        >>> a = ht.zeros((10, 2), split=0)
        >>> a[:, 0] = ht.arange(10)
        >>> b = a[3:]
        [0/2] tensor([[3., 0.],
        [1/2] tensor([[4., 0.],
                      [5., 0.],
                      [6., 0.]])
        [2/2] tensor([[7., 0.],
                      [8., 0.],
                      [9., 0.]])
        >>> b.balance_()
        >>> print(b.gshape, b.lshape)
        [0/2] (7, 2) (1, 2)
        [1/2] (7, 2) (3, 2)
        [2/2] (7, 2) (3, 2)
        >>> b
        [0/2] tensor([[3., 0.],
                     [4., 0.],
                     [5., 0.]])
        [1/2] tensor([[6., 0.],
                      [7., 0.]])
        [2/2] tensor([[8., 0.],
                      [9., 0.]])
        >>> print(b.gshape, b.lshape)
        [0/2] (7, 2) (3, 2)
        [1/2] (7, 2) (2, 2)
        [2/2] (7, 2) (2, 2)
        """
        if self.is_balanced():
            return
        self.redistribute_()

    def __bool__(self):
        """
        Boolean scalar casting.

        Returns
        -------
        casted : bool
            The corresponding bool scalar value
        """
        return self.__cast(bool)

    def __cast(self, cast_function):
        """
        Implements a generic cast function for HeAT DNDarray objects.

        Parameters
        ----------
        cast_function : function
            The actual cast function, e.g. 'float' or 'int'

        Raises
        ------
        TypeError
            If the DNDarray object cannot be converted into a scalar.

        Returns
        -------
        casted : scalar
            The corresponding casted scalar value
        """
        if np.prod(self.shape) == 1:
            if self.split is None:
                return cast_function(self.__array)

            is_empty = np.prod(self.__array.shape) == 0
            root = self.comm.allreduce(0 if is_empty else self.comm.rank, op=MPI.SUM)

            return self.comm.bcast(None if is_empty else cast_function(self.__array), root=root)

        raise TypeError("only size-1 arrays can be converted to Python scalars")

    def ceil(self, out=None):
        """
        Return the ceil of the input, element-wise.

        The ceil of the scalar x is the smallest integer i, such that i >= x. It is often denoted as :math:`\\lceil x \\rceil`.

        Parameters
        ----------
        out : ht.DNDarray or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        ceiled : ht.DNDarray
            A tensor of the same shape as x, containing the ceiled valued of each element in this tensor. If out was
            provided, ceiled is a reference to it.

        Returns
        -------
        ceiled : ht.DNDarray
            A tensor of the same shape as x, containing the floored valued of each element in this tensor. If out was
            provided, ceiled is a reference to it.

        Examples
        --------
        >>> ht.arange(-2.0, 2.0, 0.4).ceil()
        tensor([-2., -1., -1., -0., -0., -0.,  1.,  1.,  2.,  2.])
        """
        return rounding.ceil(self, out)

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
        out : ht.DNDarray, optional
            The results will be placed in this array. It may be the input array for in-place clipping. out must be of
            the right shape to hold the output. Its type is preserved.

        Returns
        -------
        clipped_values : ht.DNDarray
            A tensor with the elements of this tensor, but where values < a_min are replaced with a_min, and those >
            a_max with a_max.
        """
        return rounding.clip(self, a_min, a_max, out)

    def __complex__(self):
        """
        Complex scalar casting.

        Returns
        -------
        casted : complex
            The corresponding complex scalar value
        """
        return self.__cast(complex)

    def copy(self):
        """
        Return an array copy of the given object.

        Returns
        -------
        copied : ht.DNDarray
            A copy of the original
        """
        return memory.copy(self)

    def cos(self, out=None):
        """
        Return the trigonometric cosine, element-wise.

        Parameters
        ----------
        out : ht.DNDarray or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        cosine : ht.DNDarray
            A tensor of the same shape as x, containing the trigonometric cosine of each element in this tensor.
            Negative input elements are returned as nan. If out was provided, square_roots is a reference to it.

        Examples
        --------
        >>> ht.arange(-6, 7, 2).cos()
        tensor([ 0.9602, -0.6536, -0.4161,  1.0000, -0.4161, -0.6536,  0.9602])
        """
        return trigonometrics.cos(self, out)

    def cosh(self, out=None):
        """
        Return the hyperbolic cosine, element-wise.

        Parameters
        ----------
        x : ht.DNDarray
            The value for which to compute the hyperbolic cosine.
        out : ht.DNDarray or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        hyperbolic cosine : ht.DNDarray
            A tensor of the same shape as x, containing the hyperbolic cosine of each element in this tensor.
            Negative input elements are returned as nan. If out was provided, square_roots is a reference to it.

        Examples
        --------
        >>> ht.cosh(ht.arange(-6, 7, 2))
        tensor([201.7156,  27.3082,   3.7622,   1.0000,   3.7622,  27.3082, 201.7156])
        """
        return trigonometrics.cosh(self, out)

    def cpu(self):
        """
        Returns a copy of this object in main memory. If this object is already in main memory, then no copy is
        performed and the original object is returned.

        Returns
        -------
        tensor_on_device : ht.DNDarray
            A copy of this object on the CPU.
        """
        self.__array = self.__array.cpu()
        self.__device = devices.cpu
        return self

    def create_lshape_map(self):
        """
        Generate a 'map' of the lshapes of the data on all processes.
        Units -> (process rank, lshape)

        Returns
        -------
        lshape_map : torch.Tensor
            Units -> (process rank, lshape)
        """
        lshape_map = torch.zeros(
            (self.comm.size, len(self.gshape)), dtype=torch.int, device=self.device.torch_device
        )
        lshape_map[self.comm.rank, :] = torch.tensor(self.lshape, device=self.device.torch_device)
        self.comm.Allreduce(MPI.IN_PLACE, lshape_map, MPI.SUM)
        return lshape_map

    def __eq__(self, other):
        """
        Element-wise rich comparison of equality with values from second operand (scalar or tensor)
        Takes the second operand (scalar or tensor) to which to compare the first tensor as argument.

        Parameters
        ----------
        other: tensor or scalar
            The value(s) to which to compare equality

        Returns
        -------
        result: ht.DNDarray
            DNDarray holding 1 for all elements in which values of self are equal to values of other, 0 for all other
            elements

        Examples:
        ---------
        >>> import heat as ht
        >>> T1 = ht.float32([[1, 2],[3, 4]])
        >>> T1.__eq__(3.0)
        tensor([[0, 0],
                [1, 0]])

        >>> T2 = ht.float32([[2, 2], [2, 2]])
        >>> T1.__eq__(T2)
        tensor([[0, 1],
                [0, 0]])
        """
        return relational.eq(self, other)

    def isclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        """
        Parameters:
        -----------

        x, y : tensor
            Input tensors to compare.
        rtol : float
            The relative tolerance parameter.
        atol : float
            The absolute tolerance parameter.
        equal_nan : bool
            Whether to compare NaN’s as equal. If True, NaN’s in x will be considered equal to NaN’s in y in the output array.

        Returns:
        --------

        isclose : boolean tensor of where a and b are equal within the given tolerance.
            If both x and y are scalars, returns a single boolean value.
        """
        return logical.isclose(self, other, rtol, atol, equal_nan)

    def exp(self, out=None):
        """
        Calculate the exponential of all elements in the input array.

        Parameters
        ----------
        out : ht.DNDarray or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        exponentials : ht.DNDarray
            A tensor of the same shape as x, containing the positive exponentials of each element in this tensor. If out
            was provided, logarithms is a reference to it.

        Examples
        --------
        >>> ht.arange(5).exp()
        tensor([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981])
        """
        return exponential.exp(self, out)

    def expm1(self, out=None):
        """
        Calculate exp(x) - 1 for all elements in the array.

        Parameters
        ----------
        out : ht.DNDarray or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        exponentials : ht.DNDarray
            A tensor of the same shape as x, containing the positive exponentials minus one of each element in this tensor. If out
            was provided, logarithms is a reference to it.

        Examples
        --------
        >>> ht.arange(5).exp() + 1.
        tensor([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981])
        """
        return exponential.expm1(self, out)

    def exp2(self, out=None):
        """
        Calculate the exponential of all elements in the input array.

        Parameters
        ----------
        out : ht.DNDarray or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        exponentials : ht.DNDarray
            A tensor of the same shape as x, containing the positive exponentials of each element in this tensor. If out
            was provided, logarithms is a reference to it.

        Examples
        --------
        >>> ht.exp2(ht.arange(5))
        tensor([ 1.,  2.,  4.,  8., 16.], dtype=torch.float64)
        """
        return exponential.exp2(self, out)

    def expand_dims(self, axis):
        """
        Expand the shape of an array.

        Insert a new axis that will appear at the axis position in the expanded array shape.

        Parameters
        ----------
        axis : int
            Position in the expanded axes where the new axis is placed.

        Returns
        -------
        res : ht.DNDarray
            Output array. The number of dimensions is one greater than that of the input array.

        Raises
        ------
        ValueError
            If the axis is not in range of the axes.

        Examples
        --------
        >>> x = ht.array([1,2])
        >>> x.shape
        (2,)

        >>> y = ht.expand_dims(x, axis=0)
        >>> y
        array([[1, 2]])
        >>> y.shape
        (1, 2)

        y = ht.expand_dims(x, axis=1)
        >>> y
        array([[1],
               [2]])
        >>> y.shape
        (2, 1)
        """
        return manipulations.expand_dims(self, axis)

    def flatten(self):
        """
        Return a flat tensor.

        Returns
        -------
        flattened : ht.DNDarray
            The flattened tensor

        Examples
        --------
        >>> x = ht.array([[1,2],[3,4]])
        >>> x.flatten()
        tensor([1,2,3,4])
        """
        return manipulations.flatten(self)

    def __float__(self):
        """
        Float scalar casting.

        Returns
        -------
        casted : float
            The corresponding float scalar value
        """
        return self.__cast(float)

    def floor(self, out=None):
        """
        Return the floor of the input, element-wise.

        The floor of the scalar x is the largest integer i, such that i <= x. It is often denoted as :math:`\\lfloor x
        \\rfloor`.

        Parameters
        ----------
        out : ht.DNDarray or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        floored : ht.DNDarray
            A tensor of the same shape as x, containing the floored valued of each element in this tensor. If out was
            provided, floored is a reference to it.

        Examples
        --------
        >>> ht.floor(ht.arange(-2.0, 2.0, 0.4))
        tensor([-2., -2., -2., -1., -1.,  0.,  0.,  0.,  1.,  1.])
        """
        return rounding.floor(self, out)

    def __floordiv__(self, other):
        """
        Element-wise floor division (i.e. result is rounded int (floor))
        of the tensor by another tensor or scalar. Takes the first tensor by which it divides the second
        not-heat-typed-parameter.

        Parameters
        ----------
        other: tensor or scalar
            The second operand by whose values is divided

        Return
        ------
        result: ht.tensor
            A tensor containing the results of element-wise floor division (integer values) of t1 by t2.

        Examples:
        ---------
        >>> import heat as ht
        >>> T1 = ht.float32([[1.7, 2.0], [1.9, 4.2]])
        >>> T1 // 1
        tensor([[1., 2.],
                [1., 4.]])
        >>> T2 = ht.float32([1.5, 2.5])
        >>> T1 // T2
        tensor([[1., 0.],
                [1., 1.]])
        """
        return arithmetics.floordiv(self, other)

    def fabs(self, out=None):
        """
        Calculate the absolute value element-wise and return floating-point tensor.
        This function exists besides abs==absolute since it will be needed in case complex numbers will be introduced in the future.

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
        return rounding.fabs(self, out)

    def fill_diagonal(self, value):
        """
        Fill the main diagonal of a 2D dndarray . This function modifies the input tensor in-place, and returns the input tensor.

        Parameters
        ----------
        value : float
            The value to be placed in the dndarrays main diagonal

        Returns
        -------
        out : ht.DNDarray
            The modified input tensor with value along the diagonal

        """
        # Todo: make this 3D/nD
        if len(self.shape) != 2:
            raise ValueError("Only 2D tensors supported at the moment")

        if self.split is not None and self.comm.is_distributed:
            counts, displ, _ = self.comm.counts_displs_shape(self.shape, self.split)
            k = min(self.shape[0], self.shape[1])
            for p in range(self.comm.size):
                if displ[p] > k:
                    break
                proc = p
            if self.comm.rank <= proc:
                indices = (
                    displ[self.comm.rank],
                    displ[self.comm.rank + 1] if (self.comm.rank + 1) != self.comm.size else k,
                )
                if self.split == 0:
                    self._DNDarray__array[:, indices[0] : indices[1]] = self._DNDarray__array[
                        :, indices[0] : indices[1]
                    ].fill_diagonal_(value)
                elif self.split == 1:
                    self._DNDarray__array[indices[0] : indices[1], :] = self._DNDarray__array[
                        indices[0] : indices[1], :
                    ].fill_diagonal_(value)

        else:
            self._DNDarray__array = self._DNDarray__array.fill_diagonal_(value)

        return self

    def __ge__(self, other):
        """
        Element-wise rich comparison of relation "greater than or equal" with values from second operand (scalar or
        tensor).
        Takes the second operand (scalar or tensor) to which to compare the first tensor as argument.

        Parameters
        ----------
        other: tensor or scalar
            The value(s) to which to compare elements from tensor

        Returns
        -------
        result: ht.DNDarray
            DNDarray holding 1 for all elements in which values in self are greater than or equal to values of other
            (x1 >= x2), 0 for all other elements

        Examples
        -------
        >>> import heat as ht
        >>> T1 = ht.float32([[1, 2],[3, 4]])
        >>> T1.__ge__(3.0)
        tensor([[0, 0],
                [1, 1]], dtype=torch.uint8)
        >>> T2 = ht.float32([[2, 2], [2, 2]])
        >>> T1.__ge__(T2)
        tensor([[0, 1],
                [1, 1]], dtype=torch.uint8)
        """
        return relational.ge(self, other)

    def __getitem__(self, key):
        """
        Global getter function for ht.DNDarrays

        Parameters
        ----------
        key : int, slice, tuple, list, torch.Tensor, DNDarray
            indices to get from the tensor.

        Returns
        -------
        result : ht.DNDarray
            getter returns a new ht.DNDarray composed of the elements of the original tensor selected by the indices
            given. This does *NOT* redistribute or rebalance the resulting tensor. If the selection of values is
            unbalanced then the resultant tensor is also unbalanced!
            To redistributed the tensor use balance() (issue #187)

        Examples
        --------
        (2 processes)
        >>> a = ht.arange(10, split=0)
        (1/2) >>> tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        (2/2) >>> tensor([5, 6, 7, 8, 9], dtype=torch.int32)
        >>> a[1:6]
        (1/2) >>> tensor([1, 2, 3, 4], dtype=torch.int32)
        (2/2) >>> tensor([5], dtype=torch.int32)

        >>> a = ht.zeros((4,5), split=0)
        (1/2) >>> tensor([[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]])
        (2/2) >>> tensor([[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]])
        >>> a[1:4, 1]
        (1/2) >>> tensor([0.])
        (2/2) >>> tensor([0., 0.])
        """
        l_dtype = self.dtype.torch_type()
        kgshape_flag = False
        if isinstance(key, DNDarray) and key.ndim == self.ndim:
            """ if the key is a DNDarray and it has as many dimensions as self, then each of the entries in the 0th
                dim refer to a single element. To handle this, the key is split into the torch tensors for each dimension.
                This signals that advanced indexing is to be used. """
            key.balance_()
            key = manipulations.resplit(key.copy())
            lkey = [slice(None, None, None)] * self.ndim
            kgshape_flag = True
            kgshape = [0] * len(self.gshape)
            if key.ndim > 1:
                for i in range(key.ndim):
                    kgshape[i] = key.gshape[i]
                    lkey[i] = key._DNDarray__array[..., i]
            else:
                kgshape[0] = key.gshape[0]
                lkey[0] = key._DNDarray__array
            key = tuple(lkey)
        elif not isinstance(key, tuple):
            """ this loop handles all other cases. DNDarrays which make it to here refer to advanced indexing slices,
                as do the torch tensors. Both DNDaarrys and torch.Tensors are cast into lists here by PyTorch.
                lists mean advanced indexing will be used"""
            h = [slice(None, None, None)] * self.ndim
            if isinstance(key, DNDarray):
                key.balance_()
                key = manipulations.resplit(key.copy())
                h[0] = key._DNDarray__array.tolist()
            elif isinstance(key, torch.Tensor):
                h[0] = key.tolist()
            else:
                h[0] = key
            key = tuple(h)

        gout_full = [None] * len(self.gshape)
        # below generates the expected shape of the output.
        #   If lists or torch.Tensors remain in the key, some change may be made later
        key = list(key)
        for c, k in enumerate(key):
            if isinstance(k, slice):
                new_slice = stride_tricks.sanitize_slice(k, self.gshape[c])
                gout_full[c] = math.ceil((new_slice.stop - new_slice.start) / new_slice.step)
            elif isinstance(k, list):
                gout_full[c] = len(k)
            elif isinstance(k, (DNDarray, torch.Tensor)):
                gout_full[c] = k.shape[0] if not kgshape_flag else kgshape[c]
            if isinstance(k, DNDarray):
                key[c] = k._DNDarray__array
        if all(g == 1 for g in gout_full):
            gout_full = [1]
        else:
            # delete the dimensions from gout_full if they are not touched (they will be removed)
            for i in range(len(gout_full) - 1, -1, -1):
                if gout_full[i] is None:
                    del gout_full[i]

        key = tuple(key)
        if not self.is_distributed():
            if not self.comm.size == 1:
                return factories.array(
                    self.__array[key],
                    dtype=self.dtype,
                    split=self.split,
                    device=self.device,
                    comm=self.comm,
                )
            else:
                gout = tuple(self.__array[key].shape)
                if self.split is not None and self.split >= len(gout):
                    new_split = len(gout) - 1 if len(gout) - 1 >= 0 else None
                else:
                    new_split = self.split

                return DNDarray(
                    self.__array[key], gout, self.dtype, new_split, self.device, self.comm
                )

        else:
            rank = self.comm.rank
            ends = []
            for pr in range(self.comm.size):
                _, _, e = self.comm.chunk(self.shape, self.split, rank=pr)
                ends.append(e[self.split].stop - e[self.split].start)
            ends = torch.tensor(ends, device=self.device.torch_device)
            chunk_ends = ends.cumsum(dim=0)
            chunk_starts = torch.tensor([0] + chunk_ends.tolist(), device=self.device.torch_device)
            chunk_start = chunk_starts[rank]
            chunk_end = chunk_ends[rank]
            arr = torch.Tensor()
            # all keys should be tuples here
            gout = [0] * len(self.gshape)
            # handle the dimensional reduction for integers
            ints = sum(isinstance(it, int) for it in key)
            gout = gout[: len(gout) - ints]
            if self.split >= len(gout):
                new_split = len(gout) - 1 if len(gout) - 1 > 0 else 0
            else:
                new_split = self.split
            if len(key) == 0:  # handle empty list
                # this will return an array of shape (0, ...)
                arr = self.__array[key]
                gout_full = list(arr.shape)

            """ At the end of the following if/elif/elif block the output array will be set.
                each block handles the case where the element of the key along the split axis is a different type
                and converts the key from global indices to local indices.
            """
            if isinstance(key[self.split], (list, torch.Tensor, DNDarray)):
                # advanced indexing, elements in the split dimension are adjusted to the local indices
                lkey = list(key)
                if isinstance(key[self.split], DNDarray):
                    lkey[self.split] = key[self.split]._DNDarray__array
                inds = (
                    torch.tensor(
                        lkey[self.split], dtype=torch.long, device=self.device.torch_device
                    )
                    if not isinstance(lkey[self.split], torch.Tensor)
                    else lkey[self.split]
                )

                loc_inds = torch.where((inds >= chunk_start) & (inds < chunk_end))
                if len(loc_inds[0]) != 0:
                    # if there are no local indices on a process, then `arr` is empty
                    inds = inds[loc_inds] - chunk_start
                    lkey[self.split] = inds
                    arr = self.__array[tuple(lkey)]
            elif isinstance(key[self.split], slice):
                # standard slicing along the split axis,
                #   adjust the slice start, stop, and step, then run it on the processes which have the requested data
                key = list(key)
                key_start = key[self.split].start if key[self.split].start is not None else 0
                key_stop = (
                    key[self.split].stop
                    if key[self.split].stop is not None
                    else self.gshape[self.split]
                )
                if key_stop < 0:
                    key_stop = self.gshape[self.split] + key[self.split].stop
                key_step = key[self.split].step
                og_key_start = key_start
                st_pr = torch.where(key_start < chunk_ends)[0]
                st_pr = st_pr[0] if len(st_pr) > 0 else self.comm.size
                sp_pr = torch.where(key_stop >= chunk_starts)[0]
                sp_pr = sp_pr[-1] if len(sp_pr) > 0 else 0
                actives = list(range(st_pr, sp_pr + 1))
                if rank in actives:
                    key_start = 0 if rank != actives[0] else key_start - chunk_starts[rank]
                    key_stop = ends[rank] if rank != actives[-1] else key_stop - chunk_starts[rank]
                    if key_step is not None and rank > actives[0]:
                        offset = (chunk_ends[rank - 1] - og_key_start) % key_step
                        if key_step > 2 and offset > 0:
                            key_start += key_step - offset
                        elif key_step == 2 and offset > 0:
                            key_start += (chunk_ends[rank - 1] - og_key_start) % key_step
                    if isinstance(key_start, torch.Tensor):
                        key_start = key_start.item()
                    if isinstance(key_stop, torch.Tensor):
                        key_stop = key_stop.item()
                    key[self.split] = slice(key_start, key_stop, key_step)
                    arr = self.__array[tuple(key)]

            elif isinstance(key[self.split], int):
                # if there is an integer in the key along the split axis, adjust it and then get `arr`
                key = list(key)
                key[self.split] = (
                    key[self.split] + self.gshape[self.split]
                    if key[self.split] < 0
                    else key[self.split]
                )
                if key[self.split] in range(chunk_start, chunk_end):
                    key[self.split] = key[self.split] - chunk_start
                    arr = self.__array[tuple(key)]

            if 0 in arr.shape:
                # arr is empty
                # gout is all 0s as is the shape
                warnings.warn(
                    "This process (rank: {}) is without data after slicing, "
                    "running the .balance_() function is recommended".format(self.comm.rank),
                    ResourceWarning,
                )

            return DNDarray(
                arr.type(l_dtype),
                gout_full if isinstance(gout_full, tuple) else tuple(gout_full),
                self.dtype,
                new_split,
                self.device,
                self.comm,
            )

    if torch.cuda.device_count() > 0:

        def gpu(self):
            """
            Returns a copy of this object in GPU memory. If this object is already in GPU memory, then no copy is
            performed and the original object is returned.

            Returns
            -------
            tensor_on_device : ht.DNDarray
                A copy of this object on the GPU.
            """
            self.__array = self.__array.cuda(devices.gpu.torch_device)
            self.__device = devices.gpu
            return self

    def __gt__(self, other):
        """
        Element-wise rich comparison of relation "greater than" with values from second operand (scalar or tensor)
        Takes the second operand (scalar or tensor) to which to compare the first tensor as argument.

        Parameters
        ----------
        other: tensor or scalar
            The value(s) to which to compare elements from tensor

        Returns
        -------
        result: ht.DNDarray
            DNDarray holding 1 for all elements in which values in self are greater than values of other (x1 > x2),
            0 for all other elements

         Examples
         -------
         >>> import heat as ht
         >>> T1 = ht.float32([[1, 2],[3, 4]])
         >>> T1.__gt__(3.0)
         tensor([[0, 0],
                 [0, 1]], dtype=torch.uint8)

         >>> T2 = ht.float32([[2, 2], [2, 2]])
         >>> T1.__gt__(T2)
         tensor([[0, 0],
                 [1, 1]], dtype=torch.uint8)

        """
        return relational.gt(self, other)

    def __int__(self):
        """
        Integer scalar casting.

        Returns
        -------
        casted : int
            The corresponding float scalar value
        """
        return self.__cast(int)

    def __invert__(self):
        """
        Bit-wise inversion, or bit-wise NOT, element-wise.

        Returns
        -------
        result: ht.DNDarray
            A tensor containing the results of element-wise inversion.
        """
        return arithmetics.invert(self)

    def is_balanced(self):
        """
        Determine if a DNDarray is balanced evenly (or as evenly as possible) across all nodes

        Returns
        -------
        balanced : bool
            True if balanced, False if not
        """
        _, _, chk = self.comm.chunk(self.shape, self.split)
        test_lshape = tuple([x.stop - x.start for x in chk])
        balanced = 1 if test_lshape == self.lshape else 0

        out = self.comm.allreduce(balanced, MPI.SUM)
        return True if out == self.comm.size else False

    def is_distributed(self):
        """
        Determines whether the data of this tensor is distributed across multiple processes.

        Returns
        -------
        is_distributed : bool
            Whether the data of the tensor is distributed across multiple processes
        """
        return self.split is not None and self.comm.is_distributed()

    def item(self):
        """
        Returns the only element of a 1-element tensor. Mirror of the pytorch command by the same name
        If size of tensor is >1 element, then a ValueError is raised (by pytorch)

        Example
        -------
        >>> import heat as ht
        >>> x = ht.zeros((1))
        >>> x.item()
        0.0
        """
        return self.__array.item()

    def kurtosis(self, axis=None, unbiased=True, Fischer=True):
        """
        Compute the kurtosis (Fisher or Pearson) of a dataset.
        TODO: add return type annotation (DNDarray) and x annotation (DNDarray)

        Kurtosis is the fourth central moment divided by the square of the variance.
        If Fisher’s definition is used, then 3.0 is subtracted from the result to give 0.0 for a normal distribution.

        If unbiased is True (defualt) then the kurtosis is calculated using k statistics to
        eliminate bias coming from biased moment estimators

        Parameters
        ----------
        x : ht.DNDarray
            Input array
        axis : NoneType or Int
            Axis along which skewness is calculated, Default is to compute over the whole array `x`
        unbiased : Bool
            if True (default) the calculations are corrected for bias
        Fischer : bool
            Whether use Fischer's definition or not. If true 3. is subtracted from the result.

        Warnings
        --------
        UserWarning: Dependent on the axis given and the split configuration a UserWarning may be thrown during this
            function as data is transferred between processes
        """
        return statistics.kurtosis(self, axis, unbiased, Fischer)

    def __le__(self, other):
        """
        Element-wise rich comparison of relation "less than or equal" with values from second operand (scalar or tensor)
        Takes the second operand (scalar or tensor) to which to compare the first tensor as argument.

        Parameters
        ----------
        other: tensor or scalar
            The value(s) to which to compare elements from tensor

        Returns
        -------
        result: ht.DNDarray
            DNDarray holding 1 for all elements in which values in self are less than or equal to values of other (x1 <= x2),
            0 for all other elements

        Examples
        -------
        >>> import heat as ht
        >>> T1 = ht.float32([[1, 2],[3, 4]])
        >>> T1.__le__(3.0)
        tensor([[1, 1],
                [1, 0]], dtype=torch.uint8)

        >>> T2 = ht.float32([[2, 2], [2, 2]])
        >>> T1.__le__(T2)
        tensor([[1, 1],
                [0, 0]], dtype=torch.uint8)

        """
        return relational.le(self, other)

    def __len__(self):
        """
        The length of the DNDarray, i.e. the number of items in the first dimension.

        Returns
        -------
        length : int
            The number of items in the first dimension
        """
        return self.shape[0]

    def log(self, out=None):
        """
        Natural logarithm, element-wise.

        The natural logarithm log is the inverse of the exponential function, so that log(exp(x)) = x. The natural
        logarithm is logarithm in base e.

        Parameters
        ----------
        out : ht.DNDarray or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        logarithms : ht.DNDarray
            A tensor of the same shape as x, containing the positive logarithms of each element in this tensor.
            Negative input elements are returned as nan. If out was provided, logarithms is a reference to it.

        Examples
        --------
        >>> ht.arange(5).log()
        tensor([  -inf, 0.0000, 0.6931, 1.0986, 1.3863])
        """
        return exponential.log(self, out)

    def log2(self, out=None):
        """
        log base 2, element-wise.

        Parameters
        ----------
        self : ht.DNDarray
            The value for which to compute the logarithm.
        out : ht.DNDarray or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        logarithms : ht.DNDarray
            A tensor of the same shape as x, containing the positive logarithms of each element in this tensor.
            Negative input elements are returned as nan. If out was provided, logarithms is a reference to it.

        Examples
        --------
        >>> ht.log2(ht.arange(5))
        tensor([  -inf, 0.0000, 1.0000, 1.5850, 2.0000])
        """
        return exponential.log2(self, out)

    def log10(self, out=None):
        """
        log base 10, element-wise.

        Parameters
        ----------
        self : ht.DNDarray
            The value for which to compute the logarithm.
        out : ht.DNDarray or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        logarithms : ht.DNDarray
            A tensor of the same shape as x, containing the positive logarithms of each element in this tensor.
            Negative input elements are returned as nan. If out was provided, logarithms is a reference to it.

        Examples
        --------
        >>> ht.log10(ht.arange(5))
        tensor([-inf, 0.0000, 1.0000, 1.5850, 2.0000])
        """
        return exponential.log10(self, out)

    def log1p(self, out=None):
        """
        Return the natural logarithm of one plus the input array, element-wise.

        Parameters
        ----------
        self : ht.DNDarray
            The value for which to compute the logarithm.
        out : ht.DNDarray or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        logarithms : ht.DNDarray
            A tensor of the same shape as x, containing the positive logarithms of each element in this tensor.
            Negative input elements are returned as nan. If out was provided, logarithms is a reference to it.

        Examples
        --------
        >>> ht.log1p(ht.arange(5))
        array([0., 0.69314718, 1.09861229, 1.38629436, 1.60943791])
        """
        return exponential.log1p(self, out)

    def __lshift__(self, other):
        """
        Shift the bits of an integer to the left.

        Parameters
        ----------
        other: scalar or tensor
           number of zero bits to add

        Returns
        -------
        result: ht.NDNarray
           A tensor containing the results of element-wise left shift operation.

        Examples:
        ---------
        >>> ht.array([1, 2, 4]) << 1
        tensor([2, 4, 8])
        """
        return arithmetics.left_shift(self, other)

    def __lt__(self, other):
        """
        Element-wise rich comparison of relation "less than" with values from second operand (scalar or tensor)
        Takes the second operand (scalar or tensor) to which to compare the first tensor as argument.

        Parameters
        ----------
        other: tensor or scalar
            The value(s) to which to compare elements from tensor

        Returns
        -------
        result: ht.DNDarray
            DNDarray holding 1 for all elements in which values in self are less than values of other (x1 < x2),
            0 for all other elements

        Examples
        -------
        >>> import heat as ht
        >>> T1 = ht.float32([[1, 2],[3, 4]])
        >>> T1.__lt__(3.0)
        tensor([[1, 1],
               [0, 0]], dtype=torch.uint8)

        >>> T2 = ht.float32([[2, 2], [2, 2]])
        >>> T1.__lt__(T2)
        tensor([[1, 0],
               [0, 0]], dtype=torch.uint8)

        """
        return relational.lt(self, other)

    def __matmul__(self, other):
        """
        Matrix multiplication of two DNDarrays

        for comment context -> a @ b = c or A @ B = c

        Parameters
        ----------
        a : ht.DNDarray
            2 dimensional: L x P
        b : ht.DNDarray
            2 dimensional: P x Q

        Returns
        -------
        ht.DNDarray
            returns a tensor with the result of a @ b. The split dimension of the returned array is typically the split dimension of a.
            However, if a.split = None then the the c.split will be set as the split dimension of b. If both are None then c.split is also None.
            ** NOTE ** if a is a split vector, then the returned vector will be of shape (1xQ) and will be split in the 1st dimension
            ** NOTE ** if b is a vector and either a or b is split, then the returned vector will be of shape (Lx1) and will be split in the 0th dimension

        References
        ----------
        [1] R. Gu, et al., "Improving Execution Concurrency of Large-scale Matrix Multiplication on Distributed Data-parallel Platforms,"
            IEEE Transactions on Parallel and Distributed Systems, vol 28, no. 9. 2017.
        [2] S. Ryu and D. Kim, "Parallel Huge Matrix Multiplication on a Cluster with GPGPU Accelerators,"
            2018 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), Vancouver, BC, 2018, pp. 877-882.

        Example
        -------
        >>> a = ht.ones((n, m), split=1)
        >>> a[0] = ht.arange(1, m + 1)
        >>> a[:, -1] = ht.arange(1, n + 1)
        (0/1) tensor([[1., 2.],
                      [1., 1.],
                      [1., 1.],
                      [1., 1.],
                      [1., 1.]])
        (1/1) tensor([[3., 1.],
                      [1., 2.],
                      [1., 3.],
                      [1., 4.],
                      [1., 5.]])
        >>> b = ht.ones((j, k), split=0)
        >>> b[0] = ht.arange(1, k + 1)
        >>> b[:, 0] = ht.arange(1, j + 1)
        (0/1) tensor([[1., 2., 3., 4., 5., 6., 7.],
                      [2., 1., 1., 1., 1., 1., 1.]])
        (1/1) tensor([[3., 1., 1., 1., 1., 1., 1.],
                      [4., 1., 1., 1., 1., 1., 1.]])
        >>> linalg.matmul(a, b)
        (0/1) tensor([[18.,  8.,  9., 10.],
                      [14.,  6.,  7.,  8.],
                      [18.,  7.,  8.,  9.],
                      [22.,  8.,  9., 10.],
                      [26.,  9., 10., 11.]])
        (1/1) tensor([[11., 12., 13.],
                      [ 9., 10., 11.],
                      [10., 11., 12.],
                      [11., 12., 13.],
                      [12., 13., 14.]])
        """
        return linalg.matmul(self, other)

    def max(self, axis=None, out=None, keepdim=None):
        """
        Return the maximum of an array or maximum along an axis.

        Parameters
        ----------
        self : ht.DNDarray
            Input data.

        axis : None or int
            Axis or axes along which to operate. By default, flattened input is used.
        #TODO: out : ht.DNDarray, optional
            Alternative output array in which to place the result. Must be of the same shape and buffer length as the
            expected output.
        #TODO: initial : scalar, optional
            The minimum value of an output element. Must be present to allow computation on empty slice.
        """
        return statistics.max(self, axis=axis, out=out, keepdim=keepdim)

    def mean(self, axis=None):
        """
        Calculates and returns the mean of a tensor.
        If a axis is given, the mean will be taken in that direction.

        Parameters
        ----------
        self : ht.DNDarray
            Values for which the mean is calculated for
        axis : None, Int, iterable
            axis which the mean is taken in.
            Default: None -> mean of all data calculated

        Examples
        --------
        >>> a = ht.random.randn(1,3)
        >>> a
        tensor([[-1.2435,  1.1813,  0.3509]])
        >>> ht.mean(a)
        tensor(0.0962)

        >>> a = ht.random.randn(4,4)
        >>> a
        tensor([[ 0.0518,  0.9550,  0.3755,  0.3564],
                [ 0.8182,  1.2425,  1.0549, -0.1926],
                [-0.4997, -1.1940, -0.2812,  0.4060],
                [-1.5043,  1.4069,  0.7493, -0.9384]])
        >>> ht.mean(a, 1)
        tensor([ 0.4347,  0.7307, -0.3922, -0.0716])
        >>> ht.mean(a, 0)
        tensor([-0.2835,  0.6026,  0.4746, -0.0921])

        >>> a = ht.random.randn(4,4)
        >>> a
        tensor([[ 2.5893,  1.5934, -0.2870, -0.6637],
                [-0.0344,  0.6412, -0.3619,  0.6516],
                [ 0.2801,  0.6798,  0.3004,  0.3018],
                [ 2.0528, -0.1121, -0.8847,  0.8214]])
        >>> ht.mean(a, (0,1))
        tensor(0.4730)

        Returns
        -------
        ht.DNDarray containing the mean/s, if split, then split in the same direction as x.
        """
        return statistics.mean(self, axis)

    def min(self, axis=None, out=None, keepdim=None):
        """
        Return the minimum of an array or minimum along an axis.

        Parameters
        ----------
        self : ht.DNDarray
            Input data.
        axis : None or int
            Axis or axes along which to operate. By default, flattened input is used.
        #TODO: out : ht.DNDarray, optional
            Alternative output array in which to place the result. Must be of the same shape and buffer length as the
            expected output.
        #TODO: initial : scalar, optional
            The maximum value of an output element. Must be present to allow computation on empty slice.
        """
        return statistics.min(self, axis=axis, out=out, keepdim=keepdim)

    def __mod__(self, other):
        """
        Element-wise division remainder of values of self by values of operand other (i.e. self % other), not commutative.
        Takes the two operands (scalar or tensor) whose elements are to be divided (operand 1 by operand 2)
        as arguments.

        Parameters
        ----------
        other: tensor or scalar
            The second operand by whose values it self to be divided.

        Returns
        -------
        result: ht.DNDarray
            A tensor containing the remainder of the element-wise division of self by other.

        Examples:
        ---------
        >>> import heat as ht
        >>> ht.mod(2, 2)
        tensor([0])

        >>> T1 = ht.int32([[1, 2], [3, 4]])
        >>> T2 = ht.int32([[2, 2], [2, 2]])
        >>> T1 % T2
        tensor([[1, 0],
                [1, 0]], dtype=torch.int32)

        >>> s = ht.int32([2])
        >>> s % T1
        tensor([[0, 0]
                [2, 2]], dtype=torch.int32)
        """
        return arithmetics.mod(self, other)

    def modf(self, out=None):
        """
        Return the fractional and integral parts of an array, element-wise.
        The fractional and integral parts are negative if the given number is negative.

        Parameters
        ----------
        x : ht.DNDarray
            Input array
        out : ht.DNDarray, optional
            A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
            If not provided or None, a freshly-allocated array is returned.

        Returns
        -------
        tuple(ht.DNDarray: fractionalParts, ht.DNDarray: integralParts)

        fractionalParts : ht.DNDdarray
            Fractional part of x. This is a scalar if x is a scalar.

        integralParts : ht.DNDdarray
            Integral part of x. This is a scalar if x is a scalar.
        """

        return rounding.modf(self, out)

    def __mul__(self, other):
        """
        Element-wise multiplication (not matrix multiplication) with values from second operand (scalar or tensor)
        Takes the second operand (scalar or tensor) whose values to multiply to the first tensor as argument.

        Parameters
        ----------
        other: tensor or scalar
           The value(s) to multiply to the tensor (element-wise)

        Returns
        -------
        result: ht.DNDarray
           A tensor containing the results of element-wise multiplication.

        Examples:
        ---------
        >>> import heat as ht
        >>> T1 = ht.float32([[1, 2], [3, 4]])
        >>> T1.__mul__(3.0)
        tensor([[3., 6.],
            [9., 12.]])

        >>> T2 = ht.float32([[2, 2], [2, 2]])
        >>> T1.__mul__(T2)
        tensor([[2., 4.],
            [6., 8.]])
        """
        return arithmetics.mul(self, other)

    def __ne__(self, other):
        """
        Element-wise rich comparison of non-equality with values from second operand (scalar or tensor)
        Takes the second operand (scalar or tensor) to which to compare the first tensor as argument.

        Parameters
        ----------
        other: tensor or scalar
            The value(s) to which to compare equality

        Returns
        -------
        result: ht.DNDarray
            DNDarray holding 1 for all elements in which values of self are equal to values of other,
            0 for all other elements

        Examples:
        ---------
        >>> import heat as ht
        >>> T1 = ht.float32([[1, 2],[3, 4]])
        >>> T1.__ne__(3.0)
        tensor([[1, 1],
                [0, 1]])

        >>> T2 = ht.float32([[2, 2], [2, 2]])
        >>> T1.__ne__(T2)
        tensor([[1, 0],
                [1, 1]])
        """
        return relational.ne(self, other)

    def nonzero(self):
        """
        Return the indices of the elements that are non-zero. (using torch.nonzero)

        Returns a tuple of arrays, one for each dimension of a, containing the indices of the non-zero elements in that dimension.
        The values in a are always tested and returned in row-major, C-style order. The corresponding non-zero values can be obtained with: a[nonzero(a)].

        Parameters
        ----------
        self: ht.DNDarray

        Returns
        -------
        result: ht.DNDarray
            Indices of elements that are non-zero.
            If 'a' is split then the result is split in the 0th dimension. However, this DNDarray can be UNBALANCED as it contains the indices of the
            non-zero elements on each node.

        Examples
        --------
        >>> x = ht.array([[3, 0, 0], [0, 4, 1], [0, 6, 0]], split=0)
        [0/2] tensor([[3, 0, 0]])
        [1/2] tensor([[0, 4, 1]])
        [2/2] tensor([[0, 6, 0]])
        >>> ht.nonzero(x)
        [0/2] tensor([[0, 0]])
        [1/2] tensor([[1, 1],
        [1/2]         [1, 2]])
        [2/2] tensor([[2, 1]])

        >>> a = ht.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], split=0)
        [0/1] tensor([[1, 2, 3],
        [0/1]         [4, 5, 6]])
        [1/1] tensor([[7, 8, 9]])
        >>> a > 3
        [0/1] tensor([[0, 0, 0],
        [0/1]         [1, 1, 1]], dtype=torch.uint8)
        [1/1] tensor([[1, 1, 1]], dtype=torch.uint8)
        >>> ht.nonzero(a > 3)
        [0/1] tensor([[1, 0],
        [0/1]         [1, 1],
        [0/1]         [1, 2]])
        [1/1] tensor([[2, 0],
        [1/1]         [2, 1],
        [1/1]         [2, 2]])
        >>> a[ht.nonzero(a > 3)]
        [0/1] tensor([[4, 5, 6]])
        [1/1] tensor([[7, 8, 9]])
        """
        return indexing.nonzero(self)

    def numpy(self):
        """
        Convert heat tensor to numpy tensor. If the tensor is distributed it will be merged beforehand. If the tensor
        resides on the GPU, it will be copied to the CPU first.


        Examples
        --------
        >>> import heat as ht

        T1 = ht.random.randn((10,8))
        T1.numpy()
        """
        dist = manipulations.resplit(self, axis=None)
        return dist._DNDarray__array.cpu().numpy()

    def __or__(self, other):
        """
        Compute the bit-wise OR of two arrays element-wise.

        Parameters
        ----------
        other: tensor or scalar
        Only integer and boolean types are handled. If self.shape != other.shape, they must be broadcastable to a common shape (which becomes the shape of the output).

        Returns
        -------
        result: ht.DNDArray
        A tensor containing the results of element-wise OR of self and other.

        Examples:
        ---------
        import heat as ht
        >>> ht.array([13]) | 16
        tensor([29])

        >>> ht.array([32]) | ht.array([2])
        tensor([34])
        >>> ht.array([33, 4]) | 1
        tensor([33,  5])
        >>> ht.array([33, 4]) | ht.array([1, 2])
        tensor([33,  6])

        >>> ht.array([2, 5, 255]) | ht.array([4, 4, 4])
        tensor([  6,   5, 255])
        >>> ht.array([2, 5, 255, 2147483647], dtype=ht.int32) | ht.array([4, 4, 4, 2147483647], dtype=ht.int32)
        tensor([         6,          5,        255, 2147483647])
        >>> ht.array([True, True]) | ht.array([False, True])
        tensor([ True,  True])
        """
        return arithmetics.bitwise_or(self, other)

    def __pow__(self, other):
        """
        Element-wise exponential function with values from second operand (scalar or tensor)
        Takes the second operand (scalar or tensor) whose values are the exponent to be applied to the first
        tensor as argument.

        Parameters
        ----------
        other: tensor or scalar
           The value(s) in the exponent (element-wise)

        Returns
        -------
        result: ht.DNDarray
           A tensor containing the results of element-wise exponential operation.

        Examples:
        ---------
        >>> import heat as ht

        >>> T1 = ht.float32([[1, 2], [3, 4]])
        >>> T1.__pow__(3.0)
        tensor([[1., 8.],
                [27., 64.]])

        >>> T2 = ht.float32([[3, 3], [2, 2]])
        >>> T1.__pow__(T2)
        tensor([[1., 8.],
                [9., 16.]])
        """
        return arithmetics.pow(self, other)

    def prod(self, axis=None, out=None, keepdim=None):
        """
        Return the product of array elements over a given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes along which a product is performed. The default, axis=None, will calculate the product of all
            the elements in the input array. If axis is negative it counts from the last to the first axis.

            If axis is a tuple of ints, a product is performed on all of the axes specified in the tuple instead of a
            single axis or all the axes as before.
        out : ndarray, optional
            Alternative output tensor in which to place the result. It must have the same shape as the expected output,
            but the type of the output values will be cast if necessary.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With
            this option, the result will broadcast correctly against the input array.

        Returns
        -------
        product_along_axis : ht.DNDarray
            An array shaped as a but with the specified axis removed. Returns a reference to out if specified.

        Examples
        --------
        >>> import heat as ht
        >>> ht.array([1.,2.]).prod()
        ht.tensor([2.0])

        >>> ht.tensor([
            [1.,2.],
            [3.,4.]
        ]).prod()
        ht.tensor([24.0])

        >>> ht.array([
            [1.,2.],
            [3.,4.]
        ]).prod(axis=1)
        ht.tensor([  2.,  12.])
        """
        return arithmetics.prod(self, axis, out, keepdim)

    def qr(self, tiles_per_proc=1, calc_q=True, overwrite_a=False):
        """
        Calculates the QR decomposition of a 2D DNDarray. The algorithms are based on the CAQR and TSQR
        algorithms. For more information see the references.

        Parameters
        ----------
        a : DNDarray
            DNDarray which will be decomposed
        tiles_per_proc : int, singlt element torch.Tensor
            optional, default: 1
            number of tiles per process to operate on
        calc_q : bool
            optional, default: True
            whether or not to calculate Q
            if True, function returns (Q, R)
            if False, function returns (None, R)
        overwrite_a : bool
            optional, default: False
            if True, function overwrites the DNDarray a, with R
            if False, a new array will be created for R

        Returns
        -------
        tuple of Q and R
            if calc_q == True, function returns (Q, R)
            if calc_q == False, function returns (None, R)

        References
        ----------
        [0]  W. Zheng, F. Song, L. Lin, and Z. Chen, “Scaling Up Parallel Computation of Tiled QR
                Factorizations by a Distributed Scheduling Runtime System and Analytical Modeling,”
                Parallel Processing Letters, vol. 28, no. 01, p. 1850004, 2018.
        [1] Bilel Hadri, Hatem Ltaief, Emmanuel Agullo, Jack Dongarra. Tile QR Factorization with
                Parallel Panel Processing for Multicore Architectures. 24th IEEE International Parallel
                and DistributedProcessing Symposium (IPDPS 2010), Apr 2010, Atlanta, United States.
                inria-00548899
        [2] Gene H. Golub and Charles F. Van Loan. 1996. Matrix Computations (3rd Ed.).
        """
        return linalg.qr(
            self, tiles_per_proc=tiles_per_proc, calc_q=calc_q, overwrite_a=overwrite_a
        )

    def __repr__(self) -> str:
        """
        Computes a printable representation of the passed DNDarray.
        """
        return printing.__str__(self)

    def redistribute_(self, lshape_map=None, target_map=None):
        """
        Redistributes the data of the DNDarray *along the split axis* to match the given target map.
        This function does not modify the non-split dimensions of the DNDarray.
        This is an abstraction and extension of the balance function.

        Parameters
        ----------
        lshape_map : torch.Tensor, optional
            The current lshape of processes
            Units -> [rank, lshape]
        target_map : torch.Tensor, optional
            The desired distribution across the processes
            Units -> [rank, target lshape]
            Note: the only important parts of the target map are the values along the split axis,
            values which are not along this axis are there to mimic the shape of the lshape_map

        Returns
        -------
        None, the local shapes of the DNDarray are modified
        Examples
        --------
        >>> st = ht.ones((50, 81, 67), split=2)
        >>> target_map = torch.zeros((st.comm.size, 3), dtype=torch.int)
        >>> target_map[0, 2] = 67
        >>> print(target_map)
        [0/2] tensor([[ 0,  0, 67],
        [0/2]         [ 0,  0,  0],
        [0/2]         [ 0,  0,  0]], dtype=torch.int32)
        [1/2] tensor([[ 0,  0, 67],
        [1/2]         [ 0,  0,  0],
        [1/2]         [ 0,  0,  0]], dtype=torch.int32)
        [2/2] tensor([[ 0,  0, 67],
        [2/2]         [ 0,  0,  0],
        [2/2]         [ 0,  0,  0]], dtype=torch.int32)
        >>> print(st.lshape)
        [0/2] (50, 81, 23)
        [1/2] (50, 81, 22)
        [2/2] (50, 81, 22)
        >>> st.redistribute_(target_map=target_map)
        >>> print(st.lshape)
        [0/2] (50, 81, 67)
        [1/2] (50, 81, 0)
        [2/2] (50, 81, 0)
        """
        if not self.is_distributed():
            return
        snd_dtype = self.dtype.torch_type()
        # units -> {pr, 1st index, 2nd index}
        if lshape_map is None:
            # NOTE: giving an lshape map which is incorrect will result in an incorrect distribution
            lshape_map = self.create_lshape_map()
        else:
            if not isinstance(lshape_map, torch.Tensor):
                raise TypeError(
                    "lshape_map must be a torch.Tensor, currently {}".format(type(lshape_map))
                )
            if lshape_map.shape != (self.comm.size, len(self.gshape)):
                raise ValueError(
                    "lshape_map must have the shape ({}, {}), currently {}".format(
                        self.comm.size, len(self.gshape), lshape_map.shape
                    )
                )

        if target_map is None:  # if no target map is given then it will balance the tensor
            target_map = torch.zeros(
                (self.comm.size, len(self.gshape)), dtype=int, device=self.device.torch_device
            )
            _, _, chk = self.comm.chunk(self.shape, self.split)
            target_map = lshape_map.clone()
            target_map[..., self.split] = 0
            for pr in range(self.comm.size):
                target_map[pr, self.split] = self.comm.chunk(self.shape, self.split, rank=pr)[1][
                    self.split
                ]
        else:
            if not isinstance(target_map, torch.Tensor):
                raise TypeError(
                    "target_map must be a torch.Tensor, currently {}".format(type(target_map))
                )
            if target_map[..., self.split].sum() != self.shape[self.split]:
                raise ValueError(
                    "Sum along the split axis of the target map must be equal to the "
                    "shape in that dimension, currently {}".format(target_map[..., self.split])
                )
            if target_map.shape != (self.comm.size, len(self.gshape)):
                raise ValueError(
                    "target_map must have the shape {}, currently {}".format(
                        (self.comm.size, len(self.gshape)), target_map.shape
                    )
                )

        lshape_cumsum = torch.cumsum(lshape_map[..., self.split], dim=0)
        chunk_cumsum = torch.cat(
            (
                torch.tensor([0], device=self.device.torch_device),
                torch.cumsum(target_map[..., self.split], dim=0),
            ),
            dim=0,
        )
        # need the data start as well for process 0
        for rcv_pr in range(self.comm.size - 1):
            st = chunk_cumsum[rcv_pr].item()
            sp = chunk_cumsum[rcv_pr + 1].item()
            # start pr should be the next process with data
            if lshape_map[rcv_pr, self.split] >= target_map[rcv_pr, self.split]:
                # if there is more data on the process than the start process than start == stop
                st_pr = rcv_pr
                sp_pr = rcv_pr
            else:
                # if there is less data on the process than need to get the data from the next data
                # with data
                # need processes > rcv_pr with lshape > 0
                st_pr = (
                    torch.nonzero(input=lshape_map[rcv_pr:, self.split] > 0, as_tuple=False)[
                        0
                    ].item()
                    + rcv_pr
                )
                hld = (
                    torch.nonzero(input=sp <= lshape_cumsum[rcv_pr:], as_tuple=False).flatten()
                    + rcv_pr
                )
                sp_pr = hld[0].item() if hld.numel() > 0 else self.comm.size

            # st_pr and sp_pr are the processes on which the data sits at the beginning
            # need to loop from st_pr to sp_pr + 1 and send the pr
            for snd_pr in range(st_pr, sp_pr + 1):
                if snd_pr == self.comm.size:
                    break
                data_required = abs(sp - st - lshape_map[rcv_pr, self.split].item())
                send_amt = (
                    data_required
                    if data_required <= lshape_map[snd_pr, self.split]
                    else lshape_map[snd_pr, self.split]
                )
                if (sp - st) <= lshape_map[rcv_pr, self.split].item() or snd_pr == rcv_pr:
                    send_amt = 0
                # send amount is the data still needed by recv if that is available on the snd
                if send_amt != 0:
                    self.__redistribute_shuffle(
                        snd_pr=snd_pr, send_amt=send_amt, rcv_pr=rcv_pr, snd_dtype=snd_dtype
                    )
                lshape_cumsum[snd_pr] -= send_amt
                lshape_cumsum[rcv_pr] += send_amt
                lshape_map[rcv_pr, self.split] += send_amt
                lshape_map[snd_pr, self.split] -= send_amt
            if lshape_map[rcv_pr, self.split] > target_map[rcv_pr, self.split]:
                # if there is any data left on the process then send it to the next one
                send_amt = lshape_map[rcv_pr, self.split] - target_map[rcv_pr, self.split]
                self.__redistribute_shuffle(
                    snd_pr=rcv_pr, send_amt=send_amt.item(), rcv_pr=rcv_pr + 1, snd_dtype=snd_dtype
                )
                lshape_cumsum[rcv_pr] -= send_amt
                lshape_cumsum[rcv_pr + 1] += send_amt
                lshape_map[rcv_pr, self.split] -= send_amt
                lshape_map[rcv_pr + 1, self.split] += send_amt

        if any(lshape_map[..., self.split] != target_map[..., self.split]):
            # sometimes need to call the redistribute once more,
            # (in the case that the second to last processes needs to get data from +1 and -1)
            self.redistribute_(lshape_map=lshape_map, target_map=target_map)

    def __redistribute_shuffle(self, snd_pr, send_amt, rcv_pr, snd_dtype):
        """
        Function to abstract the function used during redistribute for shuffling data between
        processes along the split axis

        Parameters
        ----------
        snd_pr : int, single element torch.Tensor
            Sending process
        send_amt : int, single element torch.Tensor
            Amount of data to be sent by the sending process
        rcv_pr : int, single element torch.Tensor
            Recieving process
        snd_dtype : torch.type
            Torch type of the data in question

        Returns
        -------
        None
        """
        rank = self.comm.rank
        send_slice = [slice(None)] * self.ndim
        keep_slice = [slice(None)] * self.ndim
        if rank == snd_pr:
            if snd_pr < rcv_pr:  # data passed to a higher rank (off the bottom)
                send_slice[self.split] = slice(
                    self.lshape[self.split] - send_amt, self.lshape[self.split]
                )
                keep_slice[self.split] = slice(0, self.lshape[self.split] - send_amt)
            if snd_pr > rcv_pr:  # data passed to a lower rank (off the top)
                send_slice[self.split] = slice(0, send_amt)
                keep_slice[self.split] = slice(send_amt, self.lshape[self.split])
            data = self.__array[send_slice].clone()
            self.comm.Send(data, dest=rcv_pr, tag=685)
            self.__array = self.__array[keep_slice]
        if rank == rcv_pr:
            shp = list(self.gshape)
            shp[self.split] = send_amt
            data = torch.zeros(shp, dtype=snd_dtype, device=self.device.torch_device)
            self.comm.Recv(data, source=snd_pr, tag=685)
            if snd_pr < rcv_pr:  # data passed from a lower rank (append to top)
                self.__array = torch.cat((data, self.__array), dim=self.split)
            if snd_pr > rcv_pr:  # data passed from a higher rank (append to bottom)
                self.__array = torch.cat((self.__array, data), dim=self.split)

    def reshape(self, shape, axis=None):
        """
        Returns a tensor with the same data and number of elements as a, but with the specified shape.

        Parameters
        ----------
        a : ht.DNDarray
            The input tensor
        shape : tuple, list
            Shape of the new tensor
        axis : int, optional
            The new split axis. None denotes same axis
            Default : None

        Returns
        -------
        reshaped : ht.DNDarray
            The tensor with the specified shape

        Raises
        ------
        ValueError
            If the number of elements changes in the new shape.

        Examples
        --------
        >>> a = ht.arange(16, split=0)
        >>> a.reshape((4,4))
        (1/2) tensor([[0, 1, 2, 3],
                    [4, 5, 6, 7]], dtype=torch.int32)
        (2/2) tensor([[ 8,  9, 10, 11],
                    [12, 13, 14, 15]], dtype=torch.int32)
        """
        return manipulations.reshape(self, shape, axis)

    def resplit_(self, axis=None):
        """
        In-place option for resplitting a DNDarray.

        Parameters
        ----------
        axis : int, None
            The new split axis, None denotes gathering, an int will set the new split axis

        Returns
        -------
        resplit: ht.DNDarray
            The redistributed tensor. Will overwrite the old DNDarray in memory.

        Examples
        --------
        >>> a = ht.zeros((4, 5,), split=0)
        >>> a.lshape
        (0/2) (2, 5)
        (1/2) (2, 5)
        >>> ht.resplit_(a, None)
        >>> a.split
        None
        >>> a.lshape
        (0/2) (4, 5)
        (1/2) (4, 5)
        >>> a = ht.zeros((4, 5,), split=0)
        >>> a.lshape
        (0/2) (2, 5)
        (1/2) (2, 5)
        >>> ht.resplit_(a, 1)
        >>> a.split
        1
        >>> a.lshape
        (0/2) (4, 3)
        (1/2) (4, 2)
        """
        # sanitize the axis to check whether it is in range
        axis = sanitize_axis(self.shape, axis)

        # early out for unchanged content
        if axis == self.split:
            return self
        if axis is None:
            gathered = torch.empty(
                self.shape, dtype=self.dtype.torch_type(), device=self.device.torch_device
            )
            counts, displs, _ = self.comm.counts_displs_shape(self.shape, self.split)
            self.comm.Allgatherv(self.__array, (gathered, counts, displs), recv_axis=self.split)
            self.__array = gathered
            self.__split = axis
            return self
        # tensor needs be split/sliced locally
        if self.split is None:
            # new_arr = self
            _, _, slices = self.comm.chunk(self.shape, axis)
            temp = self.__array[slices]
            self.__array = torch.empty((1,), device=self.device.torch_device)
            # necessary to clear storage of local __array
            self.__array = temp.clone().detach()
            self.__split = axis
            return self

        tiles = tiling.SplitTiles(self)
        new_tile_locs = tiles.set_tile_locations(
            split=axis, tile_dims=tiles.tile_dimensions, arr=self
        )
        rank = self.comm.rank
        # receive the data with non-blocking, save which process its from
        rcv = {}
        for rpr in range(self.comm.size):
            # need to get where the tiles are on the new one first
            # rpr is the destination
            new_locs = torch.where(new_tile_locs == rpr)
            new_locs = torch.stack([new_locs[i] for i in range(self.ndim)], dim=1)
            for i in range(new_locs.shape[0]):
                key = tuple(new_locs[i].tolist())
                spr = tiles.tile_locations[key].item()
                to_send = tiles[key]
                if spr == rank and spr != rpr:
                    self.comm.Send(to_send.clone(), dest=rpr, tag=rank)
                    del to_send
                elif spr == rpr and rpr == rank:
                    rcv[key] = [None, to_send]
                elif rank == rpr:
                    sz = tiles.get_tile_size(key)
                    buf = torch.zeros(
                        sz, dtype=self.dtype.torch_type(), device=self.device.torch_device
                    )
                    w = self.comm.Irecv(buf=buf, source=spr, tag=spr)
                    rcv[key] = [w, buf]
        dims = list(range(self.ndim))
        del dims[axis]
        sorted_keys = sorted(rcv.keys())
        # todo: reduce the problem to 1D cats for each dimension, then work up
        sz = self.comm.size
        arrays = []
        for prs in range(int(len(sorted_keys) / sz)):
            lp_keys = sorted_keys[prs * sz : (prs + 1) * sz]
            lp_arr = None
            for k in lp_keys:
                if rcv[k][0] is not None:
                    rcv[k][0].wait()
                if lp_arr is None:
                    lp_arr = rcv[k][1]
                else:
                    lp_arr = torch.cat((lp_arr, rcv[k][1]), dim=dims[-1])
                del rcv[k]
            if lp_arr is not None:
                arrays.append(lp_arr)
        del dims[-1]
        # for 4 prs and 4 dims, arrays is now 16 elements long,
        # next need to group the each 4 (sz) and cat in the next dim
        for d in reversed(dims):
            new_arrays = []
            for prs in range(int(len(arrays) / sz)):
                new_arrays.append(torch.cat(arrays[prs * sz : (prs + 1) * sz], dim=d))
            arrays = new_arrays
            del d
        if len(arrays) == 1:
            arrays = arrays[0]

        self.__array = arrays
        self.__split = axis
        return self

    def __rfloordiv__(self, other):
        """
        Element-wise floor division (i.e. result is rounded int (floor))
        of the not-heat-typed parameter by another tensor. Takes the first operand (scalar or tensor) by which to divide
        as argument.

        Parameters
        ----------
        other: scalar or unknown data-type
            this will be divided by the self-tensor

        Return
        ------
        result: ht.tensor
            A tensor containing the results of element-wise floor division (integer values) of t1 by t2.

        Examples:
        ---------
        >>> import heat as ht
        >>> T = ht.float32([[1.7, 2.0], [1.9, 4.2]])
        >>> 5 // T
        tensor([[2., 2.],
                [2., 1.]])
        """
        return arithmetics.floordiv(other, self)

    def __rmod__(self, other):
        """
        Element-wise division remainder of values of other by values of operand self (i.e. other % self),
        not commutative.
        Takes the two operands (scalar or tensor) whose elements are to be divided (operand 2 by operand 1)
        as arguments.

        Parameters
        ----------
        other: scalar or unknown data-type
            The second operand which values will be divided by self.

        Returns
        -------
        result: ht.tensor
            A tensor containing the remainder of the element-wise division of other by self.

        Examples:
        ---------
        >>> import heat as ht
        >>> T = ht.int32([1, 3])
        >>> 2 % T
        tensor([0, 2], dtype=torch.int32)
        """
        return arithmetics.mod(other, self)

    def round(self, decimals=0, out=None, dtype=None):
        """
        Calculate the rounded value element-wise.

        Parameters
        ----------
        x : ht.DNDarray
            The values for which the compute the rounded value.
        out : ht.DNDarray, optional
            A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
            If not provided or None, a freshly-allocated array is returned.
        dtype : ht.type, optional
            Determines the data type of the output array. The values are cast to this type with potential loss of
            precision.

        decimals: int, optional
            Number of decimal places to round to (default: 0).
            If decimals is negative, it specifies the number of positions to the left of the decimal point.

        Returns
        -------
        rounded_values : ht.DNDarray
            A tensor containing the rounded value of each element in x.
        """
        return rounding.round(self, decimals, out, dtype)

    def __rpow__(self, other):
        """
        Element-wise exponential function of second operand (not-heat-typed) with values from first operand (tensor).
        Takes the first operand (tensor) whose values are the exponent to be applied to the second
        scalar or unknown data-type as argument.

        Parameters
        ----------
        other: scalar or unknown data-type
           The value(s) in the base (element-wise)

        Returns
        -------
        result: ht.NDNarray
           A tensor containing the results of element-wise exponential operation.

        Examples:
        ---------
        >>> import heat as ht

        >>> T = ht.float32([[1, 2], [3, 4]])
        >>> 3 ** T
        tensor([[ 3., 9.],
                [27., 81.]])
        """
        return arithmetics.pow(other, self)

    def __rshift__(self, other):
        """
        Shift the bits of an integer to the right.

        Parameters
        ----------
        other: scalar or tensor
           number of bits to remove

        Returns
        -------
        result: ht.NDNarray
           A tensor containing the results of element-wise right shift operation.

        Examples:
        ---------
        >>> ht.array([1, 2, 4]) >> 1
        tensor([0, 1, 2])
        """
        return arithmetics.right_shift(self, other)

    def __rsub__(self, other):
        """
        Element-wise subtraction of another tensor or a scalar from the tensor.
        Takes the first operand (tensor) whose elements are to be subtracted from the second argument
        (scalar or unknown data-type).

        Parameters
        ----------
        other: scalar or unknown data-type
            The value(s) from which the self-tensor will be element wise subtracted.

        Returns
        -------
        result: ht.DNDarray
            A tensor containing the results of element-wise subtraction.

        Examples:
        ---------
        >>> import heat as ht
        >>> T = ht.float32([[1, 2], [3, 4]])
        >>> 5 - T
        tensor([[4., 3.],
                [2., 1.]])
        """
        return arithmetics.sub(other, self)

    def __rtruediv__(self, other):
        """
        Element-wise true division (i.e. result is floating point value rather than rounded int (floor))
        of the not-heat-type parameter by another tensor. Takes the first tensor by which it divides the second
        not-heat-typed-parameter.

        Parameters
        ----------
        other: scalar or unknown data-type
            this will be divided by the self-tensor

        Returns
        -------
        result: ht.DNDarray
           A tensor containing the results of element-wise division.

        Examples:
        ---------
        >>> import heat as ht
        >>> T = ht.float32([2,3])
        >>> 2 / T
        tensor([1.0000, 0.6667])
        """
        return arithmetics.div(other, self)

    def save(self, path, *args, **kwargs):
        """
        Save the tensor's data to disk. Attempts to auto-detect the file format by determining the extension.

        Parameters
        ----------
        self : ht.DNDarray
            The tensor holding the data to be stored
        path : str
            Path to the file to be stored.
        args/kwargs : list/dict
            additional options passed to the particular functions.

        Raises
        -------
        ValueError
            If the file extension is not understood or known.

        Examples
        --------
        >>> a = ht.arange(100, split=0)
        >>> a.save('data.h5', 'DATA', mode='a')
        >>> a.save('data.nc', 'DATA', mode='w')
        """
        return io.save(self, path, *args, **kwargs)

    if io.supports_hdf5():

        def save_hdf5(self, path, dataset, mode="w", **kwargs):
            """
            Saves data to an HDF5 file. Attempts to utilize parallel I/O if possible.

            Parameters
            ----------
            path : str
                Path to the HDF5 file to be written.
            dataset : str
                Name of the dataset the data is saved to.
            mode : str, one of 'w', 'a', 'r+'
                File access mode
            kwargs : dict
                additional arguments passed to the created dataset.

            Raises
            -------
            TypeError
                If any of the input parameters are not of correct type.
            ValueError
                If the access mode is not understood.

            Examples
            --------
            >>> ht.arange(100, split=0).save_hdf5('data.h5', dataset='DATA')
            """
            return io.save_hdf5(self, path, dataset, mode, **kwargs)

    if io.supports_netcdf():

        def save_netcdf(self, path, variable, mode="w", **kwargs):
            """
            Saves data to a netCDF4 file. Attempts to utilize parallel I/O if possible.

            Parameters
            ----------
            path : str
                Path to the netCDF4 file to be written.
            variable : str
                Name of the variable the data is saved to.
            mode : str, one of 'w', 'a', 'r+'
                File access mode
            kwargs : dict
                additional arguments passed to the created dataset.

            Raises
            -------
            TypeError
                If any of the input parameters are not of correct type.
            ValueError
                If the access mode is not understood.

            Examples
            --------
            >>> ht.arange(100, split=0).save_netcdf('data.nc', dataset='DATA')
            """
            return io.save_netcdf(self, path, variable, mode, **kwargs)

    def __setitem__(self, key, value):
        """
        Global item setter

        Parameters
        ----------
        key : int, tuple, list, slice
            index/indices to be set
        value: np.scalar, tensor, torch.Tensor
            value to be set to the specified positions in the ht.DNDarray (self)

        Returns
        -------
        Nothing
            The specified element/s (key) of self is set with the value

        Notes
        -----
        If a DNDarray is given as the value to be set then the split axes are assumed to be equal.
            If they are not, PyTorch will raise an error when the values are attempted to be set
            on the local array

        Examples
        --------
        (2 processes)
        >>> a = ht.zeros((4,5), split=0)
        (1/2) >>> tensor([[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]])
        (2/2) >>> tensor([[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]])
        >>> a[1:4, 1] = 1
        >>> a
        (1/2) >>> tensor([[0., 0., 0., 0., 0.],
                          [0., 1., 0., 0., 0.]])
        (2/2) >>> tensor([[0., 1., 0., 0., 0.],
                          [0., 1., 0., 0., 0.]])
        """
        if isinstance(key, DNDarray) and key.ndim == self.ndim:
            # this splits the key into torch.Tensors in each dimension for advanced indexing
            lkey = [slice(None, None, None)] * self.ndim
            for i in range(key.ndim):
                lkey[i] = key._DNDarray__array[..., i]
            key = tuple(lkey)
        elif not isinstance(key, tuple):
            h = [slice(None, None, None)] * self.ndim
            h[0] = key
            key = tuple(h)

        if not self.is_distributed():
            self.__setter(key, value)
        else:
            # raise RuntimeError("split axis of array and the target value are not equal") removed
            # this will occur if the local shapes do not match
            rank = self.comm.rank
            ends = []
            for pr in range(self.comm.size):
                _, _, e = self.comm.chunk(self.shape, self.split, rank=pr)
                ends.append(e[self.split].stop - e[self.split].start)
            ends = torch.tensor(ends, device=self.device.torch_device)
            chunk_ends = ends.cumsum(dim=0)
            chunk_starts = torch.tensor([0] + chunk_ends.tolist(), device=self.device.torch_device)
            _, _, chunk_slice = self.comm.chunk(self.shape, self.split)
            chunk_start = chunk_slice[self.split].start
            chunk_end = chunk_slice[self.split].stop

            if isinstance(key, tuple):
                if isinstance(key[self.split], slice):
                    key = list(key)
                    key_start = key[self.split].start if key[self.split].start is not None else 0
                    key_stop = (
                        key[self.split].stop
                        if key[self.split].stop is not None
                        else self.gshape[self.split]
                    )
                    if key_stop < 0:
                        key_stop = self.gshape[self.split] + key[self.split].stop
                    key_step = key[self.split].step
                    og_key_start = key_start
                    st_pr = torch.where(key_start < chunk_ends)[0]
                    st_pr = st_pr[0] if len(st_pr) > 0 else self.comm.size
                    sp_pr = torch.where(key_stop >= chunk_starts)[0]
                    sp_pr = sp_pr[-1] if len(sp_pr) > 0 else 0
                    actives = list(range(st_pr, sp_pr + 1))
                    if rank in actives:
                        key_start = 0 if rank != actives[0] else key_start - chunk_starts[rank]
                        key_stop = (
                            ends[rank] if rank != actives[-1] else key_stop - chunk_starts[rank]
                        )
                        if key_step is not None and rank > actives[0]:
                            offset = (chunk_ends[rank - 1] - og_key_start) % key_step
                            if key_step > 2 and offset > 0:
                                key_start += key_step - offset
                            elif key_step == 2 and offset > 0:
                                key_start += (chunk_ends[rank - 1] - og_key_start) % key_step
                        if isinstance(key_start, torch.Tensor):
                            key_start = key_start.item()
                        if isinstance(key_stop, torch.Tensor):
                            key_stop = key_stop.item()
                        key[self.split] = slice(key_start, key_stop, key_step)
                        # todo: need to slice the values to be the right size...
                        if isinstance(value, (torch.Tensor, type(self))):
                            value_slice = [slice(None, None, None)] * value.ndim
                            step2 = key_step if key_step is not None else 1
                            key_start = chunk_starts[rank] - og_key_start
                            key_stop = key_start + key_stop
                            slice_loc = (
                                value.ndim - 1 if self.split > value.ndim - 1 else self.split
                            )
                            value_slice[slice_loc] = slice(
                                key_start.item(), math.ceil(torch.true_divide(key_stop, step2)), 1
                            )
                            self.__setter(tuple(key), value[tuple(value_slice)])
                        else:
                            self.__setter(tuple(key), value)

                elif isinstance(key[self.split], torch.Tensor):
                    key = list(key)
                    key[self.split] -= chunk_start
                    self.__setter(tuple(key), value)

                elif key[self.split] in range(chunk_start, chunk_end):
                    key = list(key)
                    key[self.split] = key[self.split] - chunk_start
                    self.__setter(tuple(key), value)

                elif key[self.split] < 0:
                    key = list(key)
                    if self.gshape[self.split] + key[self.split] in range(chunk_start, chunk_end):
                        key[self.split] = key[self.split] + self.shape[self.split] - chunk_start
                        self.__setter(tuple(key), value)
            else:
                self.__setter(key, value)

    def __setter(self, key, value):
        if np.isscalar(value):
            self.__array.__setitem__(key, value)
        elif isinstance(value, DNDarray):
            self.__array.__setitem__(key, value.__array)
        elif isinstance(value, torch.Tensor):
            self.__array.__setitem__(key, value.data)
        elif isinstance(value, (list, tuple)):
            value = torch.tensor(value, device=self.device.torch_device)
            self.__array.__setitem__(key, value.data)
        elif isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
            self.__array.__setitem__(key, value.data)
        else:
            raise NotImplementedError("Not implemented for {}".format(value.__class__.__name__))

    def sin(self, out=None):
        """
        Return the trigonometric sine, element-wise.

        Parameters
        ----------
        out : ht.DNDarray or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        sine : ht.DNDarray
            A tensor of the same shape as x, containing the trigonometric sine of each element in this tensor.
            Negative input elements are returned as nan. If out was provided, square_roots is a reference to it.

        Examples
        --------
        >>> ht.arange(-6, 7, 2).sin()
        tensor([ 0.2794,  0.7568, -0.9093,  0.0000,  0.9093, -0.7568, -0.2794])
        """
        return trigonometrics.sin(self, out)

    def sinh(self, out=None):
        """
        Return the hyperbolic sine, element-wise.

        Parameters
        ----------
        x : ht.DNDarray
            The value for which to compute the hyperbolic sine.
        out : ht.DNDarray or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        hyperbolic sine : ht.DNDarray
            A tensor of the same shape as x, containing the trigonometric sine of each element in this tensor.
            Negative input elements are returned as nan. If out was provided, square_roots is a reference to it.

        Examples
        --------
        >>> ht.sinh(ht.arange(-6, 7, 2))
        tensor([[-201.7132,  -27.2899,   -3.6269,    0.0000,    3.6269,   27.2899,  201.7132])
        """
        return trigonometrics.sinh(self, out)

    def skew(self, axis=None, unbiased=True):
        """
        Compute the sample skewness of a data set.

        Parameters
        ----------
        x : ht.DNDarray
            Input array
        axis : NoneType or Int
            Axis along which skewness is calculated, Default is to compute over the whole array `x`
        unbiased : Bool
            if True (default) the calculations are corrected for bias

        Warnings
        --------
        UserWarning: Dependent on the axis given and the split configuration a UserWarning may be thrown during this
            function as data is transferred between processes
        """
        return statistics.skew(self, axis, unbiased)

    def sqrt(self, out=None):
        """
        Return the non-negative square-root of the tensor element-wise.

        Parameters
        ----------
        out : ht.DNDarray or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        square_roots : ht.DNDarray
            A tensor of the same shape as x, containing the positive square-root of each element in this tensor.
            Negative input elements are returned as nan. If out was provided, square_roots is a reference to it.

        Examples
        --------
        >>> ht.arange(5).sqrt()
        tensor([0.0000, 1.0000, 1.4142, 1.7321, 2.0000])
        >>> ht.arange(-5, 0).sqrt()
        tensor([nan, nan, nan, nan, nan])
        """
        return exponential.sqrt(self, out)

    def squeeze(self, axis=None):
        """
        Remove single-dimensional entries from the shape of a tensor.

        Parameters:
        -----------
        x : ht.tensor
            Input data.

        axis : None or int or tuple of ints, optional
               Selects a subset of the single-dimensional entries in the shape.
               If axis is None, all single-dimensional entries will be removed from the shape.
               If an axis is selected with shape entry greater than one, a ValueError is raised.



        Returns:
        --------
        squeezed : ht.tensor
                   The input tensor, but with all or a subset of the dimensions of length 1 removed.


        Examples:
        >>> import heat as ht
        >>> import torch
        >>> torch.manual_seed(1)
        <torch._C.Generator object at 0x115704ad0>
        >>> a = ht.random.randn(1,3,1,5)
        >>> a
        tensor([[[[ 0.2673, -0.4212, -0.5107, -1.5727, -0.1232]],

                [[ 3.5870, -1.8313,  1.5987, -1.2770,  0.3255]],

                [[-0.4791,  1.3790,  2.5286,  0.4107, -0.9880]]]])
        >>> a.shape
        (1, 3, 1, 5)
        >>> a.squeeze().shape
        (3, 5)
        >>> a.squeeze
        tensor([[ 0.2673, -0.4212, -0.5107, -1.5727, -0.1232],
                [ 3.5870, -1.8313,  1.5987, -1.2770,  0.3255],
                [-0.4791,  1.3790,  2.5286,  0.4107, -0.9880]])
        >>> a.squeeze(axis=0).shape
        (3, 1, 5)
        >>> a.squeeze(axis=-2).shape
        (1, 3, 5)
        >>> a.squeeze(axis=1).shape
        Traceback (most recent call last):
        ...
        ValueError: Dimension along axis 1 is not 1 for shape (1, 3, 1, 5)
        """
        return manipulations.squeeze(self, axis)

    def std(self, axis=None, ddof=0, **kwargs):
        """
        Calculates and returns the standard deviation of a tensor with the bessel correction
        If a axis is given, the variance will be taken in that direction.

        Parameters
        ----------
        x : ht.DNDarray
            Values for which the std is calculated for
        axis : None, Int
            axis which the mean is taken in.
            Default: None -> std of all data calculated
            NOTE -> if multidemensional var is implemented in pytorch, this can be an iterable. Only thing which muse be changed is the raise
        ddof : int, optional
            Delta Degrees of Freedom: the denominator implicitely used in the calculation is N - ddof, where N
            represents the number of elements. Default: ddof=0. If ddof=1, the Bessel correction will be applied.
            Setting ddof > 1 raises a NotImplementedError.


        Examples
        --------
        >>> a = ht.random.randn(1,3)
        >>> a
        tensor([[ 0.3421,  0.5736, -2.2377]])
        >>> a.std()
        tensor(1.2742)
        >>> a = ht.random.randn(4,4)
        >>> a
        tensor([[-1.0206,  0.3229,  1.1800,  1.5471],
                [ 0.2732, -0.0965, -0.1087, -1.3805],
                [ 0.2647,  0.5998, -0.1635, -0.0848],
                [ 0.0343,  0.1618, -0.8064, -0.1031]])
        >>> ht.std(a, 0, ddof=1)
        tensor([0.6157, 0.2918, 0.8324, 1.1996])
        >>> ht.std(a, 1, ddof=1)
        tensor([1.1405, 0.7236, 0.3506, 0.4324])
        >>> ht.std(a, 1)
        tensor([0.9877, 0.6267, 0.3037, 0.3745])

        Returns
        -------
        ht.DNDarray containing the std/s, if split, then split in the same direction as x.
        """
        return statistics.std(self, axis, ddof=ddof, **kwargs)

    def __str__(self) -> str:
        """
        Computes a string representation of the passed DNDarray.
        """
        return printing.__str__(self)

    def __sub__(self, other):
        """
        Element-wise subtraction of another tensor or a scalar from the tensor.
        Takes the second operand (scalar or tensor) whose elements are to be subtracted  as argument.

        Parameters
        ----------
        other: tensor or scalar
            The value(s) to be subtracted element-wise from the tensor

        Returns
        -------
        result: ht.DNDarray
            A tensor containing the results of element-wise subtraction.

        Examples:
        ---------
        >>> import heat as ht
        >>> T1 = ht.float32([[1, 2], [3, 4]])
        >>> T1.__sub__(2.0)
        tensor([[ 1.,  0.],
                [-1., -2.]])

        >>> T2 = ht.float32([[2, 2], [2, 2]])
        >>> T1.__sub__(T2)
        tensor([[-1., 0.],
                [1., 2.]])
        """
        return arithmetics.sub(self, other)

    def sum(self, axis=None, out=None, keepdim=None):
        """
        Sum of array elements over a given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis along which a sum is performed. The default, axis=None, will sum
            all of the elements of the input array. If axis is negative it counts
            from the last to the first axis.

            If axis is a tuple of ints, a sum is performed on all of the axes specified
            in the tuple instead of a single axis or all the axes as before.

         Returns
         -------
         sum_along_axis : ht.DNDarray
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
        return arithmetics.sum(self, axis=axis, out=out, keepdim=keepdim)

    def tan(self, out=None):
        """
        Compute tangent element-wise.

        Equivalent to ht.sin(x) / ht.cos(x) element-wise.

        Parameters
        ----------
        x : ht.DNDarray
            The value for which to compute the trigonometric tangent.
        out : ht.DNDarray or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        tangent : ht.DNDarray
            A tensor of the same shape as x, containing the trigonometric tangent of each element in this tensor.

        Examples
        --------
        >>> ht.arange(-6, 7, 2).tan()
        tensor([ 0.29100619, -1.15782128,  2.18503986,  0., -2.18503986, 1.15782128, -0.29100619])
        """
        return trigonometrics.tan(self, out)

    def tanh(self, out=None):
        """
        Return the hyperbolic tangent, element-wise.

        Returns
        -------
        hyperbolic tangent : ht.DNDarray
            A tensor of the same shape as x, containing the hyperbolic tangent of each element in this tensor.

        Examples
        --------
        >>> ht.tanh(ht.arange(-6, 7, 2))
        tensor([-1.0000, -0.9993, -0.9640,  0.0000,  0.9640,  0.9993,  1.0000])
        """
        return trigonometrics.tanh(self, out)

    def tolist(self, keepsplit=False) -> List:
        """
        Return a copy of the local array data as a (nested) Python list. For scalars, a standard Python number is returned.

        Parameters
        ----------
        keepsplit: bool
            Whether the list should be returned locally or globally.

        Examples
        --------
        >>> a = ht.array([[0,1],[2,3]])
        >>> a.tolist()
        [[0, 1], [2, 3]]

        >>> a = ht.array([[0,1],[2,3]], split=0)
        >>> a.tolist()
        [[0, 1], [2, 3]]

        >>> a = ht.array([[0,1],[2,3]], split=1)
        >>> a.tolist(keepsplit=True)
        (1/2) [[0], [2]]
        (2/2) [[1], [3]]
        """

        if not keepsplit:
            return manipulations.resplit(self, axis=None).__array.tolist()

        return self.__array.tolist()

    def transpose(self, axes=None):
        """
        Permute the dimensions of an array.

        Parameters
        ----------
        axes : None or list of ints, optional
            By default, reverse the dimensions, otherwise permute the axes according to the values given.

        Returns
        -------
        p : ht.DNDarray
            a with its axes permuted.

        Examples
        --------
        >>> a = ht.array([[1, 2], [3, 4]])
        >>> a
        tensor([[1, 2],
                [3, 4]])
        >>> a.transpose()
        tensor([[1, 3],
                [2, 4]])
        >>> a.transpose((1, 0))
        tensor([[1, 3],
                [2, 4]])
        >>> a.transpose(1, 0)
        tensor([[1, 3],
                [2, 4]])

        >>> x = ht.ones((1, 2, 3))
        >>> ht.transpose(x, (1, 0, 2)).shape
        (2, 1, 3)
        """
        return linalg.transpose(self, axes)

    def tril(self, k=0):
        """
        Returns the lower triangular part of the tensor, the other elements of the result tensor are set to 0.

        The lower triangular part of the tensor is defined as the elements on and below the diagonal.

        The argument k controls which diagonal to consider. If k=0, all elements on and below the main diagonal are
        retained. A positive value includes just as many diagonals above the main diagonal, and similarly a negative
        value excludes just as many diagonals below the main diagonal.

        Parameters
        ----------
        k : int, optional
            Diagonal above which to zero elements. k=0 (default) is the main diagonal, k<0 is below and k>0 is above.

        Returns
        -------
        lower_triangle : ht.DNDarray
            Lower triangle of the input tensor.
        """
        return linalg.tril(self, k)

    def triu(self, k=0):
        """
        Returns the upper triangular part of the tensor, the other elements of the result tensor are set to 0.

        The upper triangular part of the tensor is defined as the elements on and below the diagonal.

        The argument k controls which diagonal to consider. If k=0, all elements on and below the main diagonal are
        retained. A positive value includes just as many diagonals above the main diagonal, and similarly a negative
        value excludes just as many diagonals below the main diagonal.

        Parameters
        ----------
        k : int, optional
            Diagonal above which to zero elements. k=0 (default) is the main diagonal, k<0 is below and k>0 is above.

        Returns
        -------
        upper_triangle : ht.DNDarray
            Upper triangle of the input tensor.
        """
        return linalg.triu(self, k)

    def __truediv__(self, other):
        """
        Element-wise true division (i.e. result is floating point value rather than rounded int (floor))
        of the tensor by another tensor or scalar. Takes the second operand (scalar or tensor) by which to divide
        as argument.

        Parameters
        ----------
        other: tensor or scalar
           The value(s) by which to divide the tensor (element-wise)

        Returns
        -------
        result: ht.DNDarray
           A tensor containing the results of element-wise division.

        Examples:
        ---------
        >>> import heat as ht
        >>> ht.div(2.0, 2.0)
        tensor([1.])
        >>> T1 = ht.float32([[1, 2],[3, 4]])
        >>> T2 = ht.float32([[2, 2], [2, 2]])
        >>> T1 / T2
        tensor([[0.5000, 1.0000],
                [1.5000, 2.0000]])
        >>> s = 2.0
        >>> ht.div(T1, s)
        tensor([[0.5000, 1.0000],
                [1.5, 2.0000]])
        """
        return arithmetics.div(self, other)

    def trunc(self, out=None):
        """
        Return the trunc of the input, element-wise.

        The truncated value of the scalar x is the nearest integer i which is closer to zero than x is. In short, the
        fractional part of the signed number x is discarded.

        Parameters
        ----------
        out : ht.DNDarray or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        trunced : ht.DNDarray
            A tensor of the same shape as x, containing the trunced valued of each element in this tensor. If out was
            provided, trunced is a reference to it.

        Returns
        -------
        trunced : ht.DNDarray
            A tensor of the same shape as x, containing the floored valued of each element in this tensor. If out was
            provided, trunced is a reference to it.

        Examples
        --------
        >>> ht.trunc(ht.arange(-2.0, 2.0, 0.4))
        tensor([-2., -1., -1., -0., -0.,  0.,  0.,  0.,  1.,  1.])
        """
        return rounding.trunc(self, out)

    def unique(self, sorted=False, return_inverse=False, axis=None):
        """
        Finds and returns the unique elements of the tensor.

        Works most effective if axis != self.split.

        Parameters
        ----------
        sorted : bool
            Whether the found elements should be sorted before returning as output.
        return_inverse:
            Whether to also return the indices for where elements in the original input ended up in the returned
            unique list.
        axis : int
            Axis along which unique elements should be found. Default to None, which will return a one dimensional list of
            unique values.

        Returns
        -------
        res : ht.DNDarray
            Output array. The unique elements. Elements are distributed the same way as the input tensor.
        inverse_indices : torch.tensor (optional)
            If return_inverse is True, this tensor will hold the list of inverse indices

        Examples
        --------
        >>> x = ht.array([[3, 2], [1, 3]])
        >>> x.unique(x, sorted=True)
        array([1, 2, 3])

        >>> x.unique(x, sorted=True, axis=0)
        array([[1, 3],
               [2, 3]])

        >>> x.unique(x, sorted=True, axis=1)
        array([[2, 3],
               [3, 1]])
        """
        return manipulations.unique(self, sorted, return_inverse, axis)

    def var(self, axis=None, ddof=0, **kwargs):
        """
        Calculates and returns the variance of a tensor.
        If a axis is given, the variance will be taken in that direction.

        Parameters
        ----------
        x : ht.DNDarray
            Values for which the variance is calculated for
        axis : None, Int
            axis which the variance is taken in.
            Default: None -> var of all data calculated
            NOTE -> if multidemensional var is implemented in pytorch, this can be an iterable. Only thing which muse be changed is the raise
        ddof : int, optional
            Delta Degrees of Freedom: the denominator implicitely used in the calculation is N - ddof, where N
            represents the number of elements. Default: ddof=0. If ddof=1, the Bessel correction will be applied.
            Setting ddof > 1 raises a NotImplementedError.

        Notes on ddof (from numpy)
        --------------------------
        The variance is the average of the squared deviations from the mean, i.e., var = mean(abs(x - x.mean())**2).
        The mean is normally calculated as x.sum() / N, where N = len(x). If, however, ddof is specified, the divisor
        N - ddof is used instead. In standard statistical practice, ddof=1 provides an unbiased estimator of the
        variance of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate of the variance
        for normally distributed variables.


        Examples
        --------
        >>> a = ht.random.randn(1,3)
        >>> a
        tensor([[-1.9755,  0.3522,  0.4751]])
        >>> a.var()
        tensor(1.2710)

        >>> a = ht.random.randn(4,4)
        >>> a
        tensor([[-0.8665, -2.6848, -0.0215, -1.7363],
                [ 0.5886,  0.5712,  0.4582,  0.5323],
                [ 1.9754,  1.2958,  0.5957,  0.0418],
                [ 0.8196, -1.2911, -0.2026,  0.6212]])
        >>> ht.var(a, 1, ddof=1)
        tensor([1.3092, 0.0034, 0.7061, 0.9217])
        >>> ht.var(a, 0, ddof=1)
        tensor([1.3624, 3.2563, 0.1447, 1.2042])
        >>> ht.var(a, 0)
        tensor([1.0218, 2.4422, 0.1085, 0.9032])

        Returns
        -------
        ht.DNDarray containing the var/s, if split, then split in the same direction as x.
        """
        return statistics.var(self, axis, ddof=ddof, **kwargs)

    def __xor__(self, other):
        """
        Compute the bit-wise XOR of two arrays element-wise.

        Parameters
        ----------
        other: tensor or scalar
        Only integer and boolean types are handled. If self.shape != other.shape, they must be broadcastable to a common shape (which becomes the shape of the output).

        Returns
        -------
        result: ht.DNDArray
        A tensor containing the results of element-wise OR of self and other.

        Examples:
        ---------
        import heat as ht
        >>> ht.array([13]) ^ 17
        tensor([28])

        >>> ht.array([31]) ^ ht.array([5])
        tensor([26])
        >>> ht.array[31,3] ^ 5
        tensor([26,  6])

        >>> ht.array([31,3]) ^ ht.array([5,6])
        tensor([26,  5])
        >>> ht.array([True, True]) ^ ht.array([False, True])
        tensor([ True, False])
        """
        return arithmetics.bitwise_xor(self, other)

    """
    This ensures that commutative arithmetic operations work no matter on which side the heat-tensor is placed.

    Examples
    --------
    >>> import heat as ht
    >>> T = ht.float32([[1., 2.], [3., 4.,]])
    >>> T + 1
    tensor([[2., 3.],
            [4., 5.]])
    >>> 1 + T
    tensor([[2., 3.],
        [4., 5.]])
    """
    __radd__ = __add__
    __rmul__ = __mul__
