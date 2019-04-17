import torch
import warnings

from .communication import Communication, MPI, MPI_WORLD
from .stride_tricks import *
from . import types
from . import devices
from . import operations
from . import io
from . import arithmetics
from . import relations
from . import trigonometrics
from . import exponential
from . import rounding
from . import reductions

warnings.simplefilter('always', ResourceWarning)

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


class tensor:
    def __init__(self, array, gshape, dtype, split, device, comm):
        self.__array = array
        self.__gshape = gshape
        self.__dtype = dtype
        self.__split = split
        self.__device = device
        self.__comm = comm

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
    def lshape(self):
        return tuple(self.__array.shape)

    @property
    def shape(self):
        return self.__gshape

    @property
    def split(self):
        return self.__split

    @property
    def T(self, axes=None):
        return operations.transpose(self, axes)

    @property
    def lloc(self):
        """
        Local item setter and getter. i.e. this function operates on a local level and only on the pytorch tesnors composing the ht tensor
        This function uses the LocalIndex class

        Parameters
        ----------
        key : int, slice, list, tuple
            indices of the desired data
        value : all types compatible with pytorch tensors
            optional (if none given then this is a getter function)

        Returns
        -------
        (getter) -> ht.tensor with the indices selected at a *local* level
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

    def abs(self, out=None, dtype=None):
        """
        Calculate the absolute value element-wise.

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
        return rounding.abs(self, out, dtype)

    def absolute(self, out=None, dtype=None):
        """
        Calculate the absolute value element-wise.

        ht.abs is a shorthand for this function.

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
        result: ht.tensor
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

            If this is a tuple of ints, a reduction is performed on multiple axes, instead of a single axis 
            or all the axes as before.


        out : ht.tensor, optional
            Alternate output array in which to place the result. It must have the same shape as the expected output
            and its type is preserved.

        Returns:
        --------
        all : ht.tensor, bool
            A new boolean or ht.tensor is returned unless out is specified, in which case a reference to out is returned.

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
        return operations.all(self, axis=axis, out=out, keepdim=keepdim)

    def allclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        """
        Test whether self and other are element-wise equal within a tolerance. Returns True if |self - other| <= atol +
        rtol * |other| for all elements, False otherwise.

        Parameters:
        -----------
        other : ht.tensor
            Input tensor to compare to

        atol: float, optional
            Absolute tolerance. Default is 1e-08

        rtol: float, optional
            Relative tolerance (with respect to y). Default is 1e-05

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
        return operations.allclose(self, other, rtol, atol, equal_nan)

    def any(self, axis=None, out=None):
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
        return operations.any(self, axis=axis, out=out)

    def argmax(self, axis=None, out=None, **kwargs):
        """
        Returns the indices of the maximum values along an axis.

        Parameters:	
        ----------
        x : ht.tensor
            Input array.
        axis : int, optional
            By default, the index is into the flattened tensor, otherwise along the specified axis.
        out : array, optional
            If provided, the result will be inserted into this tensor. It should be of the appropriate shape and dtype.

        Returns:
        -------	
        index_tensor : ht.tensor of ints
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
        return operations.argmax(self, axis=axis, out=out, **kwargs)

    def argmin(self, axis=None, out=None, **kwargs):
        """
        Returns the indices of the minimum values along an axis.

        Parameters:	
        ----------
        x : ht.tensor
            Input array.
        axis : int, optional
            By default, the index is into the flattened tensor, otherwise along the specified axis.
        out : array, optional
            If provided, the result will be inserted into this tensor. It should be of the appropriate shape and dtype.

        Returns:
        -------	
        index_tensor : ht.tensor of ints
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
        return operations.argmin(self, axis=axis, out=out, **kwargs)

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
            return tensor(casted_array, self.shape, dtype, self.split, self.device, self.comm)

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

    def cos(self, out=None):
        """
        Return the trigonometric cosine, element-wise.

        Parameters
        ----------
        out : ht.tensor or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        cosine : ht.tensor
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
        x : ht.tensor
            The value for which to compute the hyperbolic cosine.
        out : ht.tensor or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        hyperbolic cosine : ht.tensor
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
        tensor_on_device : ht.tensor
            A copy of this object on the CPU.
        """
        self.__array = self.__array.cpu()
        return self

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
        result: ht.tensor
           A tensor containing the results of element-wise division.

        Examples:
        ---------
        >>> import heat as ht
        >>> ht.div(2.0, 2.0)
        tensor([1.])

        >>> T1 = ht.float32([[1, 2],[3, 4]])
        >>> T2 = ht.float32([[2, 2], [2, 2]])
        >>> T1.__div__(T2)
        tensor([[0.5000, 1.0000],
                [1.5000, 2.0000]])

        >>> s = 2.0
        >>> T1.__div__(s)
        tensor([[0.5000, 1.0000],
                [1.5, 2.0000]])
        """
        return arithmetics.div(self, other)

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
            result: ht.tensor
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
        result: ht.tensor
            Tensor holding 1 for all elements in which values of self are equal to values of other, 0 for all other
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
        return relations.eq(self, other)

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
        result: ht.tensor
            Tensor holding 1 for all elements in which values in self are greater than or equal to values of other
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
        return relations.ge(self, other)

    if torch.cuda.device_count() > 0:
        def gpu(self):
            """
            Returns a copy of this object in GPU memory. If this object is already in GPU memory, then no copy is
            performed and the original object is returned.

            Returns
            -------
            tensor_on_device : ht.tensor
                A copy of this object on the GPU.
            """
            self.__array = self.__array.cuda(devices.gpu_index())
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
        result: ht.tensor
            Tensor holding 1 for all elements in which values in self are greater than values of other (x1 > x2),
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
        return relations.gt(self, other)

    def is_distributed(self):
        """
        Determines whether the data of this tensor is distributed across multiple processes.

        Returns
        -------
        is_distributed : bool
            Whether the data of the tensor is distributed across multiple processes
        """
        return self.split is not None and self.comm.is_distributed()

    def max(self, axis=None, out=None, keepdim=None):
        """
        Return the maximum of an array or maximum along an axis.

        Parameters
        ----------
        self : ht.tensor
            Input data.

        axis : None or int or tuple of ints, optional
            Axis or axes along which to operate. By default, flattened input is used.
            If this is a tuple of ints, the maximum is selected over multiple axes, 
            instead of a single axis or all the axes as before.
        out : ht.tensor, optional
            Alternative output array in which to place the result. Must be of the same shape and buffer length as the
            expected output.
        #TODO: initial : scalar, optional   
            The minimum value of an output element. Must be present to allow computation on empty slice.
        """
        return relations.max(self, axis=axis, out=out, keepdim=keepdim)

    def mean(self, axis):
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        # TODO: make me more numpy API complete
        return self.sum(axis) / self.shape[axis]

    def min(self, axis=None, out=None, keepdim=None):
        """
        Return the minimum of an array or minimum along an axis.

        Parameters
        ----------
        self : ht.tensor
            Input data.
        axis : None or int or tuple of ints, optional
            Axis or axes along which to operate. By default, flattened input is used.
            If this is a tuple of ints, the minimum is selected over multiple axes, 
            instead of a single axis or all the axes as before.
        out : ht.tensor, optional
            Alternative output array in which to place the result. Must be of the same shape and buffer length as the
            expected output.
        #TODO: initial : scalar, optional   
            The maximum value of an output element. Must be present to allow computation on empty slice.
        """
        return relations.min(self, axis=axis, out=out, keepdim=keepdim)

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
        return exponential.exp(self, out)

    def exp2(self, out=None):
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
        >>> ht.exp2(ht.arange(5))
        tensor([ 1.,  2.,  4.,  8., 16.], dtype=torch.float64)
        """
        return exponential.exp2(self, out)

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
            self.device,
            self.comm
        )

    def ceil(self, out=None):
        """
        Return the ceil of the input, element-wise.

        The ceil of the scalar x is the largest integer i, such that i <= x. It is often denoted as \lceil x \rceil.

        Parameters
        ----------
        out : ht.tensor or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        ceiled : ht.tensor
            A tensor of the same shape as x, containing the ceiled valued of each element in this tensor. If out was
            provided, ceiled is a reference to it.

        Returns
        -------
        ceiled : ht.tensor
            A tensor of the same shape as x, containing the floored valued of each element in this tensor. If out was
            provided, ceiled is a reference to it.

        Examples
        --------
        >>> ht.arange(-2.0, 2.0, 0.4).ceil()
        tensor([-2., -1., -1., -0., -0., -0.,  1.,  1.,  2.,  2.])
        """
        return rounding.ceil(self, out)

    def floor(self, out=None):
        """
        Return the floor of the input, element-wise.

        The floor of the scalar x is the largest integer i, such that i <= x. It is often denoted as :math:`\lfloor x
        \rfloor`.

        Parameters
        ----------
        out : ht.tensor or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        floored : ht.tensor
            A tensor of the same shape as x, containing the floored valued of each element in this tensor. If out was
            provided, floored is a reference to it.

        Examples
        --------
        >>> ht.floor(ht.arange(-2.0, 2.0, 0.4))
        tensor([-2., -2., -2., -1., -1.,  0.,  0.,  0.,  1.,  1.])
        """
        return rounding.floor(self, out)

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
        result: ht.tensor
            Tensor holding 1 for all elements in which values in self are less than or equal to values of other (x1 <= x2),
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
        return relations.le(self, other)

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
        return exponential.log(self, out)

    def log2(self, out=None):
        """
        log base 2, element-wise.

        Parameters
        ----------
        x : ht.tensor
            The value for which to compute the logarithm.
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
        >>> ht.log2(ht.arange(5))
        tensor([  -inf, 0.0000, 1.0000, 1.5850, 2.0000])
        """
        return exponential.log2(self, out)

    def log10(self, out=None):
        """
        log base 10, element-wise.

        Parameters
        ----------
        x : ht.tensor
            The value for which to compute the logarithm.
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
        >>> ht.log10(ht.arange(5))
        tensor([-inf, 0.0000, 1.0000, 1.5850, 2.0000])
        """
        return exponential.log10(self, out)

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
        result: ht.tensor
            Tensor holding 1 for all elements in which values in self are less than values of other (x1 < x2),
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
        return relations.lt(self, other)

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
        result: ht.tensor
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
        result: ht.tensor
            Tensor holding 1 for all elements in which values of self are equal to values of other,
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
        return relations.ne(self, other)

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
        result: ht.tensor
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

    def resplit(self, axis=None, out=None):
        """
        In-place redistribution of the content of the tensor. Allows to "unsplit" (i.e. gather) all values from all
        nodes as well as the definition of new axis along which the tensor is split without changes to the values.

        WARNING: this operation might involve a significant communication overhead. Use it sparingly and preferably for
        small tensors.

        Parameters
        ----------
        axis : int
            The new split axis, None denotes gathering, an int will set the new split axis
        out : ht.tensor or None, optional
            A location in which to store the resplit tensor. Old data is overwritten.

        Returns
        -------
        resplit: ht.tensor
            The redistributed tensor

        Examples
        --------
        a = ht.zeros((4, 5,), split=0)
        a.lshape
        (0/2) >>> (2, 5)
        (1/2) >>> (2, 5)
        a.resplit(None)
        a.split
        >>> None
        a.lshape
        (0/2) >>> (4, 5)
        (1/2) >>> (4, 5)

        a = ht.zeros((4, 5,), split=0)
        a.lshape
        (0/2) >>> (2, 5)
        (1/2) >>> (2, 5)
        a.resplit(1)
        a.split
        >>> 1
        a.lshape
        (0/2) >>> (4, 3)
        (1/2) >>> (4, 2)
        """
        # sanitize the axis to check whether it is in range
        axis = sanitize_axis(self.shape, axis)

        # sanitize the output tensor
        if out is not None and not isinstance(out, tensor):
            raise TypeError('out must be None or tensor. but was {}'.format(type(out)))

        # early out for unchanged content
        if axis == self.split:
            if out is None:
                return self

            out._tensor__array = self.__array
            out._tensor__gshape = self.__gshape
            out._tensor__dtype = self.__dtype
            out._tensor__split = self.__split
            out._tensor__device = self.__device
            out._tensor__comm = self.__comm

            return out

        # unsplit the tensor
        if axis is None:
            target_data = torch.empty(self.shape)
            target_split = None

            recv_counts, recv_displs, _ = self.comm.counts_displs_shape(self.shape, self.split)
            self.comm.Allgatherv(self.__array, (target_data, recv_counts, recv_displs,), recv_axis=axis)

        # tensor needs be split/sliced locally
        elif self.split is None:
            _, _, slices = self.comm.chunk(self.shape, axis)
            target_data = self.__array[slices]
            target_split = axis

        # entirely new split axis, need to redistribute
        else:
            _, output_shape, _ = self.comm.chunk(self.shape, axis)
            target_data = torch.empty(output_shape)
            target_split = axis

            send_counts, send_displs, _ = self.comm.counts_displs_shape(self.lshape, axis)
            recv_counts, recv_displs, _ = self.comm.counts_displs_shape(self.shape, self.split)

            self.comm.Alltoallv(
                (self.__array, send_counts, send_displs,),
                (target_data, recv_counts, recv_displs,),
                axis=axis, recv_axis=self.split
            )

        if out is not None:
            out._tensor__array = target_data
            out._tensor__gshape = self.__gshape
            out._tensor__dtype = self.__dtype
            out._tensor__split = target_split
            out._tensor__device = self.__device
            out._tensor__comm = self.__comm

            return out

        return tensor(target_data, self.__gshape, self.__dtype, target_split, self.__device, self.__comm)

    def save(self, path, *args, **kwargs):
        """
        Save the tensor's data to disk. Attempts to auto-detect the file format by determining the extension.

        Parameters
        ----------
        self : ht.tensor
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
        def save_hdf5(self, path, dataset, mode='w', **kwargs):
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
        def save_netcdf(self, path, variable, mode='w', **kwargs):
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
        return trigonometrics.sin(self, out)

    def sinh(self, out=None):
        """
        Return the hyperbolic sine, element-wise.

        Parameters
        ----------
        x : ht.tensor
            The value for which to compute the hyperbolic sine.
        out : ht.tensor or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        hyperbolic sine : ht.tensor
            A tensor of the same shape as x, containing the trigonometric sine of each element in this tensor.
            Negative input elements are returned as nan. If out was provided, square_roots is a reference to it.

        Examples
        --------
        >>> ht.sinh(ht.arange(-6, 7, 2))
        tensor([[-201.7132,  -27.2899,   -3.6269,    0.0000,    3.6269,   27.2899,  201.7132])
        """
        return trigonometrics.sinh(self, out)

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
        return exponential.sqrt(self, out)

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
        result: ht.tensor
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
        return reductions.sum(self, axis=axis, out=out, keepdim=keepdim)

    def tan(self, out=None):
        """
        Compute tangent element-wise.

        Equivalent to ht.sin(x) / ht.cos(x) element-wise.

        Parameters
        ----------
        x : ht.tensor
            The value for which to compute the trigonometric tangent.
        out : ht.tensor or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        tangent : ht.tensor
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

        Parameters
        ----------
        x : ht.tensor
            The value for which to compute the hyperbolic tangent.
        out : ht.tensor or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Returns
        -------
        hyperbolic tangent : ht.tensor
            A tensor of the same shape as x, containing the hyperbolic tangent of each element in this tensor.

        Examples
        --------
        >>> ht.tanh(ht.arange(-6, 7, 2))
        tensor([-1.0000, -0.9993, -0.9640,  0.0000,  0.9640,  0.9993,  1.0000])
        """
        return trigonometrics.tanh(self, out)

    def transpose(self, axes=None):
        """
        Permute the dimensions of an array.

        Parameters
        ----------
        axes : None or list of ints, optional
            By default, reverse the dimensions, otherwise permute the axes according to the values given.

        Returns
        -------
        p : ht.tensor
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
        return operations.transpose(self, axes)

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
        lower_triangle : ht.tensor
            Lower triangle of the input tensor.
        """
        return operations.tril(self, k)

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
        upper_triangle : ht.tensor
            Upper triangle of the input tensor.
        """
        return operations.triu(self, k)

    def __str__(self, *args):
        # TODO: document me
        # TODO: generate none-PyTorch str
        return self.__array.__str__(*args)

    def __repr__(self, *args):
        # TODO: document me
        # TODO: generate none-PyTorch repr
        return self.__array.__repr__(*args)

    def __getitem__(self, key):
        """
        Global getter function for ht.tensors

        Parametes
        ---------
        key : int, slice, tuple, list
            indices to get from the tensor.

        Returns
        -------
        result : ht.tensor
            getter returns a new ht.tensor composed of the elements of the original tensor selected by the indices given.
            this does *NOT* redistribute or rebalance the resulting tensor. If the selection of values is unbalanced then
            the resultant tensor is also unbalanced!
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
        if not self.is_distributed():
            if not self.comm.size == 1:
                return tensor(self.__array[key], tuple(self.__array[key].shape), self.dtype, self.split, self.device, self.comm)
            else:
                gout = tuple(self.__array[key].shape)
                if self.split >= len(gout):
                    new_split = len(gout) - 1 if len(gout) - 1 > 0 else 0
                else:
                    new_split = self.split

                return tensor(self.__array[key], gout, self.dtype, new_split, self.device, self.comm)
        else:
            _, _, chunk_slice = self.comm.chunk(self.shape, self.split)
            chunk_start = chunk_slice[self.split].start
            chunk_end = chunk_slice[self.split].stop
            chunk_set = set(range(chunk_start, chunk_end))

            arr = torch.Tensor()

            if isinstance(key, int):  # if a sigular index is given and the tensor is split
                gout = [0] * (len(self.gshape) - 1)
                # handle the reduction of the split to accommodate for the reduced dimension
                if self.split >= len(gout):
                    new_split = len(gout) - 1 if len(gout) - 1 > 0 else 0
                else:
                    new_split = self.split

                if key in range(chunk_start, chunk_end) and self.split == 0:  # only need to adjust the key if split==0
                    gout = list(self.__array[key-chunk_start].shape)
                    arr = self.__array[key - chunk_start]
                elif self.split != 0:
                    _, _, chunk_slice2 = self.comm.chunk(self.shape, self.split)
                    if key in range(chunk_slice2[0].start, chunk_slice2[0].stop):  # need to test if the given axis is on the node and then get the shape
                        arr = self.__array[key]
                        gout = list(arr.shape)
                else:
                    warnings.warn("This process (rank: {}) is without data after slicing".format(self.comm.rank), ResourceWarning)
                    # arr is empty and gout is zeros

            elif isinstance(key, (tuple, list)):  # multi-argument gets are passed as tuples by python
                gout = [0] * len(self.gshape)
                # handle the dimensional reduction for integers
                ints = sum([isinstance(it, int) for it in key])
                gout = gout[:len(gout) - ints]

                # reduce the dims if the slices are only one element in length
                slices = [isinstance(k, slice) for k in key]
                if any(slices):
                    for s, b in enumerate(slices):
                        if b:
                            start = key[s].start if key[s].start is not None else 0
                            stop = key[s].stop if key[s].stop is not None else self.gshape[s]
                            if start == stop - 1:
                                gout = gout[:-1]

                if self.split >= len(gout):
                    new_split = len(gout) - 1 if len(gout) - 1 > 0 else 0
                else:
                    new_split = self.split

                if isinstance(key[self.split], slice):  # if a slice is given in the split direction
                    # below allows for the split given to contain Nones
                    key_set = set(range(key[self.split].start if key[self.split].start is not None else 0,
                                        key[self.split].stop if key[self.split].stop is not None else self.gshape[self.split],
                                        key[self.split].step if key[self.split].step else 1))
                    key = list(key)
                    overlap = list(key_set & chunk_set)
                    if overlap:  # if the slice is requesting data on the nodes
                        hold = [x - chunk_start for x in overlap]
                        key[self.split] = slice(min(hold), max(hold) + 1, key[self.split].step)
                        arr = self.__array[tuple(key)]
                        gout = list(arr.shape)

                # if the given axes are not splits (must be ints for python)
                # this means the whole slice is on one node
                elif key[self.split] in range(chunk_start, chunk_end):
                    key = list(key)
                    key[self.split] = key[self.split] - chunk_start
                    arr = self.__array[tuple(key)]
                    gout = list(arr.shape)
                else:
                    warnings.warn("This process (rank: {}) is without data after slicing".format(self.comm.rank), ResourceWarning)
                    # arr is empty
                    # gout is all 0s and is the proper shape

            elif isinstance(key, slice) and self.split == 0:  # if the given axes are only a slice
                gout = [0] * len(self.gshape)
                # reduce the dims if the slices are only one element in length
                start = key.start if key.start is not None else 0
                stop = key.stop if key.stop is not None else self.gshape[0]
                if start == stop - 1:
                    gout = gout[:-1]

                if self.split >= len(gout):
                    new_split = len(gout) - 1 if len(gout) - 1 > 0 else 0
                else:
                    new_split = self.split

                key_set = set(range(start, stop, key.step if key.step else 1))
                overlap = list(key_set & chunk_set)

                if overlap:
                    hold = [x - chunk_start for x in overlap]
                    key = slice(min(hold), max(hold) + 1, key.step)
                    arr = self.__array[key]
                    gout = list(arr.shape)
                else:
                    warnings.warn("This process (rank: {}) is without data after slicing".format(self.comm.rank), ResourceWarning)
                    # arr is empty
                    # gout is all 0s and is the proper shape

            else:  # handle other cases not accounted for (one is a slice is given and the split != 0)
                gout = [0] * len(self.gshape)
                if isinstance(key, slice):
                    # reduce the dims if the slices are only one element in length
                    start = key.start if key.start is not None else 0
                    stop = key.stop if key.stop is not None else self.gshape[0]
                    if start == stop - 1:
                        gout = gout[:-1]

                if self.split >= len(gout):
                    new_split = len(gout) - 1 if len(gout) - 1 > 0 else 0
                else:
                    new_split = self.split

                gout = list(self.__array[key].shape)
                arr = self.__array[key]

            for e, _ in enumerate(gout):
                if e == new_split:
                    gout[e] = self.comm.allreduce(gout[e], MPI.SUM)
                else:
                    gout[e] = self.comm.allreduce(gout[e], MPI.MAX)

            return tensor(arr, gout if isinstance(gout, tuple) else tuple(gout), self.dtype, new_split, self.device, self.comm)

    def __setitem__(self, key, value):
        """
        Global item setter

        Parameters
        ----------
        key : int, tuple, list, slice
            index/indices to be set
        value: np.scalar, tensor, torch.Tensor
            value to be set to the specified positions in the ht.tensor (self)

        Returns
        -------
        Nothing
            The specified element/s (key) of self is set with the value

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
        if not self.is_distributed():
            self.setter(key, value)
        else:
            _, _, chunk_slice = self.comm.chunk(self.shape, self.split)
            chunk_start = chunk_slice[self.split].start
            chunk_end = chunk_slice[self.split].stop

            if isinstance(key, int) and self.split == 0:
                if key in range(chunk_start, chunk_end):
                    self.setter(key-chunk_start, value)

            elif isinstance(key, (tuple, list, tensor, torch.Tensor)):
                if isinstance(key[self.split], slice):
                    key = list(key)
                    overlap = list(set(range(key[self.split].start if key[self.split].start is not None else 0,
                                             key[self.split].stop if key[self.split].stop is not None else self.gshape[self.split]))
                                   & set(range(chunk_start, chunk_end)))

                    if overlap:
                        hold = [x - chunk_start for x in overlap]
                        key[self.split] = slice(min(hold), max(hold) + 1, key[self.split].step)
                        try:
                            self.setter(tuple(key), value[overlap])
                        except TypeError as te:
                            if str(te) != "'int' object is not subscriptable":
                                raise TypeError
                            self.setter(tuple(key), value)

                elif key[self.split] in range(chunk_start, chunk_end):
                    key = list(key)
                    key[self.split] = key[self.split] - chunk_start
                    self.setter(tuple(key), value)

            elif isinstance(key, slice) and self.split == 0:
                overlap = list(set(range(key.start, key.stop)) & set(range(chunk_start, chunk_end)))
                if overlap:
                    hold = [x - chunk_start for x in overlap]
                    key = slice(min(hold), max(hold) + 1, key.step)
                    self.setter(key, value)
            else:
                self.setter(key, value)

    def setter(self, key, value):
        if np.isscalar(value):
            self.__array.__setitem__(key, value)
        elif isinstance(value, tensor):
            self.__array.__setitem__(key, value.__array)
        elif isinstance(value, torch.Tensor):
            self.__array.__setitem__(key, value.data)
        elif isinstance(value, (list, tuple)):
            value = torch.Tensor(value)
            self.__array.__setitem__(key, value.data)
        elif isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
            self.__array.__setitem__(key, value.data)
        else:
            raise NotImplementedError('Not implemented for {}'.format(value.__class__.__name__))


def __factory(shape, dtype, split, local_factory, device, comm):
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
    device : str or None
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    out : ht.tensor
        Array of ones with given shape, data type and node distribution.
    """
    # clean the user input
    shape = sanitize_shape(shape)
    dtype = types.canonical_heat_type(dtype)
    split = sanitize_axis(shape, split)
    device = devices.sanitize_device(device)

    # chunk the shape if necessary
    _, local_shape, _ = comm.chunk(shape, split)
    # create the torch data using the factory function
    data = local_factory(local_shape, dtype=dtype.torch_type(), device=device.torch_device)

    return tensor(data, shape, dtype, split, device, comm)


def __factory_like(a, dtype, split, factory, device, comm, **kwargs):
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
    device : str or None
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

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

    return factory(shape, dtype=dtype, split=split, device=device, comm=comm, **kwargs)


def arange(*args, dtype=None, split=None, device=None, comm=MPI_WORLD):
    """
    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)`` (in other words, the interval including `start`
    but excluding `stop`). For integer arguments the function is equivalent to the Python built-in `range
    <http://docs.python.org/lib/built-in-funcs.html>`_ function, but returns a tensor rather than a list.

    When using a non-integer step, such as 0.1, the results will often not be consistent. It is better to use
    ``linspace`` for these cases.

    Parameters
    ----------
    start : number, optional
        Start of interval.  The interval includes this value.  The default start value is 0.
    stop : number
        End of interval.  The interval does not include this value, except in some cases where `step` is not an integer
        and floating point round-off affects the length of `out`.
    step : number, optional
        Spacing between values.  For any output `out`, this is the distance between two adjacent values, ``out[i+1] -
        out[i]``. The default step size is 1. If `step` is specified as a position argument, `start` must also be given.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data type from the other input arguments.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    device : str or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    arange : 1D heat tensor
        1D heat tensor of evenly spaced values.

        For floating point arguments, the length of the result is ``ceil((stop - start)/step)``. Because of floating
        point overflow, this rule may result in the last element of `out` being greater than `stop`.

    See Also
    --------
    linspace : Evenly spaced numbers with careful handling of endpoints.

    Examples
    --------
    >>> ht.arange(3)
    tensor([0, 1, 2])
    >>> ht.arange(3.0)
    tensor([ 0.,  1.,  2.])
    >>> ht.arange(3, 7)
    tensor([3, 4, 5, 6])
    >>> ht.arange(3, 7, 2)
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
    offset, lshape, _ = comm.chunk(gshape, split)

    # compose the local tensor
    start += offset * step
    stop = start + lshape[0] * step
    device = devices.sanitize_device(device)
    data = torch.arange(
        start, stop, step,
        dtype=types.canonical_heat_type(dtype).torch_type(),
        device=device.torch_device
    )

    return tensor(data, gshape, types.canonical_heat_type(data.dtype), split, device, comm)


def array(obj, dtype=None, copy=True, ndmin=0, split=None, device=None, comm=MPI_WORLD):
    """
    Create a tensor.
    Parameters
    ----------
    obj : array_like
        A tensor or array, any object exposing the array interface, an object whose __array__ method returns an array,
        or any (nested) sequence.
    dtype : dtype, optional
        The desired data-type for the array. If not given, then the type will be determined as the minimum type required
        to hold the objects in the sequence. This argument can only be used to ‘upcast’ the array. For downcasting, use
        the .astype(t) method.
    copy : bool, optional
        If true (default), then the object is copied. Otherwise, a copy will only be made if obj is a nested sequence or
        if a copy is needed to satisfy any of the other requirements, e.g. dtype.
    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting array should have. Ones will be pre-pended to the
        shape as needed to meet this requirement.
    split : None or int, optional
        The axis along which the array is split and distributed in memory. If not None (default) the shape of the global
        tensor is automatically inferred.
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed tensor chunks.
    Returns
    -------
    out : ht.tensor
        A tensor object satisfying the specified requirements.
    Raises
    ------
    Examples
    --------
    >>> ht.array([1, 2, 3])
    tensor([1, 2, 3])
    Upcasting:
    >>> ht.array([1, 2, 3.0])
    tensor([ 1.,  2.,  3.])
    More than one dimension:
    >>> ht.array([[1, 2], [3, 4]])
    tensor([[1, 2],
           [3, 4]])
    Minimum dimensions given:
    >>> ht.array([1, 2, 3], ndmin=2)
    tensor([[1, 2, 3]])
    Type provided:
    >>> ht.array([1, 2, 3], dtype=float)
    tensor([ 1.0, 2.0, 3.0])
    Pre-split data:
    (0/2) >>> ht.array([1, 2], split=0)
    (1/2) >>> ht.array([3, 4], split=0)
    (0/2) tensor([1, 2, 3, 4])
    (1/2) tensor([1, 2, 3, 4])
    """
    # extract the internal tensor in case of a heat tensor
    if isinstance(obj, tensor):
        obj = obj._tensor__array

    # sanitize the data type
    if dtype is not None:
        dtype = types.canonical_heat_type(dtype)

    # initialize the array
    if bool(copy):
        if isinstance(obj, torch.Tensor):
            obj = obj.clone().detach()
        else:
            try:
                obj = torch.tensor(obj, dtype=dtype.torch_type() if dtype is not None else None)
            except RuntimeError:
                raise TypeError('invalid data of type {}'.format(type(obj)))

    # infer dtype from obj if not explicitly given
    if dtype is None:
        dtype = types.canonical_heat_type(obj.dtype)

    # sanitize minimum number of dimensions
    if not isinstance(ndmin, int):
        raise TypeError('expected ndmin to be int, but was {}'.format(type(ndmin)))

    # reshape the object to encompass additional dimensions
    ndmin -= len(obj.shape)
    if ndmin > 0:
        obj = obj.reshape(obj.shape + ndmin * (1,))

    # sanitize split axis
    split = sanitize_axis(obj.shape, split)

    # sanitize communication object
    if not isinstance(comm, Communication):
        raise TypeError('expected communication object, but got {}'.format(type(comm)))

    # determine the local and the global shape, if not split is given, they are identical
    lshape = np.array(obj.shape)
    gshape = lshape.copy()

    # check with the neighboring rank whether the local shape would fit into a global shape
    if split is not None:
        if comm.rank < comm.size - 1:
            comm.Isend(lshape, dest=comm.rank + 1)
        if comm.rank != 0:
            # look into the message of the neighbor to see whether the shape length fits
            status = MPI.Status()
            comm.Probe(source=comm.rank - 1, status=status)
            length = status.Get_count() // lshape.dtype.itemsize

            # the number of shape elements does not match with the 'left' rank
            if length != len(lshape):
                gshape[split] = np.iinfo(gshape.dtype).min
            else:
                # check whether the individual shape elements match
                comm.Recv(gshape, source=comm.rank - 1)
                for i in range(length):
                    if i == split:
                        continue
                    elif lshape[i] != gshape[i] and lshape[i] - 1 != gshape[i]:
                        gshape[split] = np.iinfo(gshape.dtype).min

        # sum up the elements along the split dimension
        reduction_buffer = np.array(gshape[split])
        comm.Allreduce(MPI.IN_PLACE, reduction_buffer, MPI.SUM)
        if reduction_buffer < 0:
            raise ValueError('unable to construct tensor, shape of local data chunk does not match')
        gshape[split] = reduction_buffer

    return tensor(obj, tuple(int(ele) for ele in gshape), dtype, split, device, comm)


def empty(shape, dtype=types.float32, split=None, device=None, comm=MPI_WORLD):
    """
    Returns a new uninitialized array of given shape and data type. May be allocated split up across multiple
    nodes along the specified axis.

    Parameters
    ----------
    shape : int or sequence of ints
        Desired shape of the output array, e.g. 1 or (1, 2, 3,).
    dtype : ht.dtype
        The desired HeAT data type for the array, defaults to ht.float32.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    out : ht.tensor
        Array of zeros with given shape, data type and node distribution.

    Examples
    --------
    >>> ht.empty(3)
    tensor([ 0.0000e+00, -2.0000e+00,  3.3113e+35])

    >>> ht.empty(3, dtype=ht.int)
    tensor([ 0.0000e+00, -2.0000e+00,  3.3113e+35])

    >>> ht.empty((2, 3,))
    tensor([[ 0.0000e+00, -2.0000e+00,  3.3113e+35],
            [ 3.6902e+19,  1.2096e+04,  7.1846e+22]])
    """
    return __factory(shape, dtype, split, torch.empty, device, comm)


def empty_like(a, dtype=None, split=None, device=None, comm=MPI_WORLD):
    """
    Returns a new uninitialized array with the same type, shape and data distribution of given object. Data type and
    data distribution strategy can be explicitly overriden.

    Parameters
    ----------
    a : object
        The shape and data-type of 'a' define these same attributes of the returned array.
        Uninitialized tensor with the same shape, type and split axis as 'a' unless overriden.
    dtype : ht.dtype, optional
        Overrides the data type of the result.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Examples
    --------
    >>> x = ht.ones((2, 3,))
    >>> x
    tensor([[1., 1., 1.],
            [1., 1., 1.]])

    >>> ht.empty_like(x)
    tensor([[ 0.0000e+00, -2.0000e+00,  3.3113e+35],
            [ 3.6902e+19,  1.2096e+04,  7.1846e+22]])
    """
    return __factory_like(a, dtype, split, empty, device, comm)


def eye(shape, dtype=types.float32, split=None, device=None, comm=MPI_WORLD):
    """
    Returns a new 2-D tensor with ones on the diagonal and zeroes elsewhere.

    Parameters
    ----------
    shape : int or tuple of ints
            The shape of the data-type. If only one number is provided, returning tensor will be square with that size.
            In other cases, the first value represents the number rows, the second the number of columns.
    dtype : ht.dtype, optional
            Overrides the data type of the result.
    split : int, optional
            The axis along which the tensor is split and distributed, defaults to None (no distribution).
    device : str, ht.Device or None, optional
            Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm : Communication, optional
            Handle to the nodes holding distributed parts or copies of this tensor.

    Examples
    --------
    >>> import heat as ht
    >>> ht.eye(2)
    tensor([[1., 0.],
            [0., 1.]])

    >>> ht.eye((2, 3), dtype=ht.int32)
    tensor([[1, 0, 0],
            [0, 1, 0]], dtype=torch.int32)
    """
    # Determine the actual size of the resulting data
    gshape = shape
    if isinstance(gshape, int):
        gshape = (gshape, gshape)
    if len(gshape) is 1:
        gshape = gshape * 2

    split = sanitize_axis(gshape, split)
    device = devices.sanitize_device(device)
    offset, lshape, _ = comm.chunk(gshape, split)
    # Start by creating tensor filled with zeroes
    data = torch.zeros(lshape, dtype=types.canonical_heat_type(dtype).torch_type(), device=device.torch_device)
    # Insert ones at the correct positions
    for i in range(min(lshape)):
        pos_x = i if split is 0 else i + offset
        pos_y = i if split is 1 else i + offset
        data[pos_x][pos_y] = 1

    return tensor(data, gshape, types.canonical_heat_type(data.dtype), split, device, comm)


def full(shape, fill_value, dtype=types.float32, split=None, device=None, comm=MPI_WORLD):
    """
    Return a new array of given shape and type, filled with fill_value.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., (2, 3) or 2.
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    out : ht.tensor
        Array of fill_value with the given shape, dtype and split.

    Examples
    --------
    >>> ht.full((2, 2), np.inf)
    tensor([[ inf,  inf],
            [ inf,  inf]])
    >>> ht.full((2, 2), 10)
    tensor([[10, 10],
            [10, 10]])
    """
    def local_factory(*args, **kwargs):
        return torch.full(*args, fill_value=fill_value, **kwargs)

    return __factory(shape, dtype, split, local_factory, device, comm)


def full_like(a, fill_value, dtype=types.float32, split=None, device=None, comm=MPI_WORLD):
    """
    Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    a : object
        The shape and data-type of 'a' define these same attributes of the returned array.
    fill_value : scalar
        Fill value.
    dtype : ht.dtype, optional
        Overrides the data type of the result.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    out : ht.tensor
        Array of fill_value with the same shape and type as a.


    Examples
    --------
    >>> x = ht.zeros((2, 3,))
    >>> x
    tensor([[0., 0., 0.],
            [0., 0., 0.]])

    >>> ht.full_like(a, 1.0)
    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    """
    return __factory_like(a, dtype, split, full, device, comm, fill_value=fill_value)


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, split=None, device=None, comm=MPI_WORLD):
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
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    samples: ht.tensor
        There are num equally spaced samples in the closed interval [start, stop] or the half-open interval
        [start, stop) (depending on whether endpoint is True or False).
    step: float, optional
        Size of spacing between samples, only returned if retstep is True.

    Examples
    --------
    >>> ht.linspace(2.0, 3.0, num=5)
    tensor([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
    >>> ht.linspace(2.0, 3.0, num=5, endpoint=False)
    tensor([ 2. ,  2.2,  2.4,  2.6,  2.8])
    >>> ht.linspace(2.0, 3.0, num=5, retstep=True)
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
    offset, lshape, _ = comm.chunk(gshape, split)

    # compose the local tensor
    start += offset * step
    stop = start + lshape[0] * step - step
    device = devices.sanitize_device(device)
    data = torch.linspace(start, stop, lshape[0], device=device.torch_device)
    if dtype is not None:
        data = data.type(types.canonical_heat_type(dtype).torch_type())

    # construct the resulting global tensor
    ht_tensor = tensor(data, gshape, types.canonical_heat_type(data.dtype), split, device, comm)

    if retstep:
        return ht_tensor, step
    return ht_tensor


def ones(shape, dtype=types.float32, split=None, device=None, comm=MPI_WORLD):
    """
    Returns a new array of given shape and data type filled with one values. May be allocated split up across multiple
    nodes along the specified axis.

    Parameters
    ----------
    shape : int or sequence of ints
        Desired shape of the output array, e.g. 1 or (1, 2, 3,).
    dtype : ht.dtype
        The desired HeAT data type for the array, defaults to ht.float32.
    split : int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm : Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

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
    return __factory(shape, dtype, split, torch.ones, device, comm)


def ones_like(a, dtype=None, split=None, device=None, comm=MPI_WORLD):
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
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

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
    return __factory_like(a, dtype, split, ones, device, comm)


def zeros(shape, dtype=types.float32, split=None, device=None, comm=MPI_WORLD):
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
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

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
    return __factory(shape, dtype, split, torch.zeros, device, comm)


def zeros_like(a, dtype=None, split=None, device=None, comm=MPI_WORLD):
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
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

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

    >>> ht.zeros_like(x)
    tensor([[0., 0., 0.],
            [0., 0., 0.]])
    """
    return __factory_like(a, dtype, split, zeros, device, comm)
