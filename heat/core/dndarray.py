import numpy as np
import torch
import warnings

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
from . import relational
from . import rounding
from . import statistics
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
    def __init__(self, array, gshape, dtype, split, device, comm):
        self.__array = array
        self.__gshape = gshape
        self.__dtype = dtype
        self.__split = split
        self.__device = device
        self.__comm = comm

        if (
            isinstance(self.__array, torch.Tensor)
            and isinstance(device, devices.Device)
            and self.__array.device.type not in self.__device.torch_device
        ):
            self.__array = self.__array.to(devices.sanitize_device(self.__device).torch_device)

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
        return len(self.__gshape)

    @property
    def size(self):
        """
        Returns
        -------
        int: number of total elements of the tensor
        """
        try:
            return np.prod(self.__gshape)
        except TypeError:
            return 1

    @property
    def gnumel(self):
        """
        Returns
        -------
        int: number of total elements of the tensor
        """
        return self.size

    @property
    def lnumel(self):
        """
        Returns
        -------
        int: number of elements of the tensor on each node
        """
        return np.prod(self.__array.shape)

    @property
    def lloc(self):
        """
        Local item setter and getter. i.e. this function operates on a local level and only on the PyTorch tensors
        composing the HeAT DNDarray. This function uses the LocalIndex class.

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
        tuple : the shape of the data on each node
        """
        return tuple(self.__array.shape)

    @property
    def shape(self):
        """
        Returns
        -------
        tuple : the shape of the tensor as a whole
        """
        return self.__gshape

    @property
    def split(self):
        """
        Returns
        -------
        int : the axis on which the tensor split
        """
        return self.__split

    @property
    def T(self, axes=None):
        return linalg.transpose(self, axes)

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
        sl_dtype = self.dtype.torch_type()
        # units -> {pr, 1st index, 2nd index}
        lshape_map = factories.zeros((self.comm.size, len(self.gshape)), dtype=int)
        lshape_map[self.comm.rank, :] = torch.Tensor(self.lshape)
        lshape_map_comm = self.comm.Iallreduce(MPI.IN_PLACE, lshape_map, MPI.SUM)

        chunk_map = factories.zeros((self.comm.size, len(self.gshape)), dtype=int)
        _, _, chk = self.comm.chunk(self.shape, self.split)
        for i in range(len(self.gshape)):
            chunk_map[self.comm.rank, i] = chk[i].stop - chk[i].start
        chunk_map_comm = self.comm.Iallreduce(MPI.IN_PLACE, chunk_map, MPI.SUM)

        lshape_map_comm.wait()
        chunk_map_comm.wait()

        # create list of which processes need to send data to lower ranked nodes
        send_list = [
            True
            if lshape_map[pr, self.split] != (chunk_map[pr, self.split])
            and lshape_map[pr, self.split] != 0
            else False
            for pr in range(1, self.comm.size)
        ]
        send_list.insert(
            0, True if lshape_map[0, self.split] > (chunk_map[0, self.split]) else False
        )
        first_pr_w_data = send_list.index(True)  # first process with *too much* data
        last_pr_w_data = next(
            (
                i
                for i in reversed(range(len(lshape_map[:, self.split])))
                if lshape_map[i, self.split] > chunk_map[i, self.split]
            )
        )

        # create arbitrary slices for which data to send and which data to keep
        send_slice = [slice(None)] * self.numdims
        keep_slice = [slice(None)] * self.numdims

        # first send the first entries of the data to the 0th node and then the next data to the 1st ...
        # this will redistributed the data forward
        if first_pr_w_data != 0:
            for spr in range(first_pr_w_data, last_pr_w_data + 1):
                if self.comm.rank == spr:
                    for pr in range(spr):
                        send_amt = abs(
                            (chunk_map[pr, self.split] - lshape_map[pr, self.split]).item()
                        )
                        send_amt = (
                            send_amt
                            if send_amt < self.lshape[self.split]
                            else self.lshape[self.split]
                        )
                        if send_amt:
                            send_slice[self.split] = slice(0, send_amt)
                            keep_slice[self.split] = slice(send_amt, self.lshape[self.split])

                            self.comm.Isend(
                                self.__array[send_slice].clone(),
                                dest=pr,
                                tag=pr + self.comm.size + spr,
                            )
                            self.__array = self.__array[keep_slice].clone()

                # else:
                for pr in range(spr):
                    snt = abs((chunk_map[pr, self.split] - lshape_map[pr, self.split]).item())
                    snt = (
                        snt
                        if snt < lshape_map[spr, self.split]
                        else lshape_map[spr, self.split].item()
                    )

                    if self.comm.rank == pr and snt:
                        shp = list(self.gshape)
                        shp[self.split] = snt
                        data = torch.zeros(shp, dtype=sl_dtype, device=self.device.torch_device)
                        self.comm.Recv(data, source=spr, tag=pr + self.comm.size + spr)
                        self.__array = torch.cat((self.__array, data), dim=self.split)
                    lshape_map[pr, self.split] += snt
                    lshape_map[spr, self.split] -= snt

        if self.is_balanced():
            return

        # now the DNDarray is balanced from 0 to x, (by pulling data from the higher ranking nodes)
        # next we balance the data from x to the self.comm.size
        send_list = [
            True if lshape_map[pr, self.split] > (chunk_map[pr, self.split]) else False
            for pr in range(self.comm.size)
        ]
        first_pr_w_data = send_list.index(True)  # first process with *too much* data
        last_pr_w_data = next(
            (
                i
                for i in reversed(range(len(lshape_map[:, self.split])))
                if lshape_map[i, self.split] > chunk_map[i, self.split]
            )
        )

        send_slice = [slice(None)] * self.numdims
        keep_slice = [slice(None)] * self.numdims
        # need to send from the last one with data
        # start from x then push the data to the next one. then do the same at x+1 until the last process
        balanced_process = [False for _ in range(self.comm.size)]
        for pr in range(self.comm.size):
            balanced_process[pr] = (
                True if chunk_map[pr, self.split] == lshape_map[pr, self.split] else False
            )
            if pr > 0:
                if any(i is False for i in balanced_process[:pr]):
                    balanced_process[pr] = False

        for pr, b in enumerate(balanced_process[:-1]):
            if not b:  # if the process is not balanced
                send_amt = abs((chunk_map[pr, self.split] - lshape_map[pr, self.split]).item())
                send_amt = (
                    send_amt
                    if send_amt < lshape_map[pr, self.split]
                    else lshape_map[pr, self.split]
                )
                if send_amt:
                    if self.comm.rank == pr:  # send data to the next process
                        send_slice[self.split] = slice(
                            self.lshape[self.split] - send_amt, self.lshape[self.split]
                        )
                        keep_slice[self.split] = slice(0, self.lshape[self.split] - send_amt)

                        self.comm.Send(
                            self.__array[send_slice].clone(),
                            dest=pr + 1,
                            tag=pr + self.comm.size + pr + 1,
                        )
                        self.__array = self.__array[keep_slice].clone()

                    if self.comm.rank == pr + 1:  # receive data on the next process
                        shp = list(self.gshape)
                        shp[self.split] = send_amt
                        data = torch.zeros(shp, dtype=sl_dtype, device=self.device.torch_device)
                        self.comm.Recv(data, source=pr, tag=pr + self.comm.size + pr + 1)
                        self.__array = torch.cat((data, self.__array), dim=self.split)
                    lshape_map[pr, self.split] -= send_amt
                    lshape_map[pr + 1, self.split] += send_amt

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
        return self

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
        key : int, slice, tuple, list
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
        if isinstance(key, DNDarray) and key.gshape[-1] != len(self.gshape):
            key = tuple(x.item() for x in key)

        if not self.is_distributed():
            if not self.comm.size == 1:
                if isinstance(key, DNDarray) and key.gshape[-1] == len(self.gshape):
                    # this will return a 1D array as the shape cannot be determined automatically
                    arr = self.__array[key._DNDarray__array[..., 0], key._DNDarray__array[..., 1]]
                    return DNDarray(
                        arr, tuple(arr.shape), self.dtype, self.split, self.device, self.comm
                    )
                else:
                    return DNDarray(
                        self.__array[key],
                        tuple(self.__array[key].shape),
                        self.dtype,
                        self.split,
                        self.device,
                        self.comm,
                    )
            else:
                if isinstance(key, DNDarray) and key.gshape[-1] == len(self.gshape):
                    # this will return a 1D array as the shape cannot be determined automatically
                    arr = self.__array[key._DNDarray__array[..., 0], key._DNDarray__array[..., 1]]
                    return DNDarray(arr, tuple(arr.shape), self.dtype, 0, self.device, self.comm)

                else:
                    gout = tuple(self.__array[key].shape)
                    if self.split is not None and self.split >= len(gout):
                        new_split = len(gout) - 1 if len(gout) - 1 > 0 else 0
                    else:
                        new_split = self.split

                    return DNDarray(
                        self.__array[key], gout, self.dtype, new_split, self.device, self.comm
                    )

        else:
            _, _, chunk_slice = self.comm.chunk(self.shape, self.split)
            chunk_start = chunk_slice[self.split].start
            chunk_end = chunk_slice[self.split].stop
            chunk_set = set(range(chunk_start, chunk_end))

            arr = torch.Tensor()

            # if a sigular index is given and the tensor is split
            if isinstance(key, int):
                gout = [0] * (len(self.gshape) - 1)
                if key < 0:
                    key += self.numdims
                # handle the reduction of the split to accommodate for the reduced dimension
                if self.split >= len(gout):
                    new_split = len(gout) - 1 if len(gout) - 1 > 0 else 0
                elif self.split > 0:
                    new_split = self.split - 1
                else:
                    new_split = self.split

                # only need to adjust the key if split==0
                if key in range(chunk_start, chunk_end) and self.split == 0:
                    gout = list(self.__array[key - chunk_start].shape)
                    arr = self.__array[key - chunk_start]
                elif self.split != 0:
                    _, _, chunk_slice2 = self.comm.chunk(self.shape, self.split)
                    # need to test if the given axis is on the node and then get the shape
                    if key in range(chunk_slice2[0].start, chunk_slice2[0].stop):
                        arr = self.__array[key]
                        gout = list(arr.shape)
                else:  # arr is empty and gout is zeros
                    warnings.warn(
                        "This process (rank: {}) is without data after slicing, running the .balance_() function is recommended".format(
                            self.comm.rank
                        ),
                        ResourceWarning,
                    )

            # multi-argument gets are passed as tuples by python
            elif isinstance(key, (tuple, list)):
                gout = [0] * len(self.gshape)
                # handle the dimensional reduction for integers
                ints = sum([isinstance(it, int) for it in key])
                gout = gout[: len(gout) - ints]

                if self.split >= len(gout):
                    new_split = len(gout) - 1 if len(gout) - 1 > 0 else 0
                else:
                    new_split = self.split

                # if a slice is given in the split direction
                # below allows for the split given to contain Nones
                if isinstance(key[self.split], slice):
                    key_stop = key[self.split].stop
                    if key_stop is not None and key_stop < 0:
                        key_stop = self.gshape[self.split] + key[self.split].stop
                    key_set = set(
                        range(
                            key[self.split].start if key[self.split].start is not None else 0,
                            key_stop if key_stop is not None else self.gshape[self.split],
                            key[self.split].step if key[self.split].step else 1,
                        )
                    )
                    key = list(key)
                    overlap = list(key_set & chunk_set)
                    if overlap:  # if the slice is requesting data on the nodes
                        overlap.sort()
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
                elif key[self.split] < 0 and self.gshape[self.split] + key[self.split] in range(
                    chunk_start, chunk_end
                ):
                    key = list(key)
                    key[self.split] = key[self.split] + chunk_end - chunk_start
                    arr = self.__array[tuple(key)]
                    gout = list(arr.shape)
                else:
                    warnings.warn(
                        "This process (rank: {}) is without data after slicing, running the .balance_() function is recommended".format(
                            self.comm.rank
                        ),
                        ResourceWarning,
                    )
                    # arr is empty
                    # gout is all 0s and is the proper shape

            # if the given axes are only a slice
            elif isinstance(key, slice) and self.split == 0:
                gout = [0] * len(self.gshape)
                # reduce the dims if the slices are only one element in length
                start = key.start if key.start is not None else 0
                stop = key.stop if key.stop is not None else self.gshape[0]
                step = key.step if key.step is not None else 1

                if self.split >= len(gout):
                    new_split = len(gout) - 1 if len(gout) - 1 > 0 else 0
                else:
                    new_split = self.split
                key_set = set(range(start, stop, step))
                overlap = list(key_set & chunk_set)
                if overlap:
                    overlap.sort()
                    hold = [x - chunk_start for x in overlap]
                    key = slice(min(hold), max(hold) + 1, step)
                    arr = self.__array[key]
                    gout = list(arr.shape)
                else:
                    warnings.warn(
                        "This process (rank: {}) is without data after slicing, running the .balance_() function is recommended".format(
                            self.comm.rank
                        ),
                        ResourceWarning,
                    )
                    # arr is empty
                    # gout is all 0s and is the proper shape

            elif isinstance(key, DNDarray) and key.gshape[-1] == len(self.gshape):
                # this is for a list of values
                # it will return a 1D DNDarray of the elements on each node which are in the key (will be split in the 0th dimension
                key.lloc[..., self.split] -= chunk_start
                key_new = [key._DNDarray__array[..., i] for i in range(len(self.gshape))]
                arr = self.__array[tuple(key_new)]
                gout = list(arr.shape)
                new_split = 0

            else:  # handle other cases not accounted for (one is a slice is given and the split != 0)
                gout = [0] * len(self.gshape)

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

            return DNDarray(
                arr.type(l_dtype),
                gout if isinstance(gout, tuple) else tuple(gout),
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

    def __repr__(self, *args):
        # TODO: document me
        # TODO: generate none-PyTorch repr
        return self.__array.__repr__(*args)

    def resplit_(self, axis=None):
        """
        In-place redistribution of the content of the tensor. Allows to "unsplit" (i.e. gather) all values from all
        nodes as well as the definition of new axis along which the tensor is split without changes to the values.

        WARNING: this operation might involve a significant communication overhead. Use it sparingly and preferably for
        small tensors.

        Parameters
        ----------
        axis : int
            The new split axis, None denotes gathering, an int will set the new split axis

        Returns
        -------
        resplit: ht.DNDarray
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

        # early out for unchanged content
        if axis == self.split:
            return self

        # unsplit the tensor
        if axis is None:
            gathered = torch.empty(
                self.shape, dtype=self.dtype.torch_type(), device=self.device.torch_device
            )

            recv_counts, recv_displs, _ = self.comm.counts_displs_shape(self.shape, self.split)
            self.comm.Allgatherv(
                self.__array, (gathered, recv_counts, recv_displs), recv_axis=self.split
            )

            self.__array = gathered
            self.__split = None

        # tensor needs be split/sliced locally
        elif self.split is None:
            _, _, slices = self.comm.chunk(self.shape, axis)
            temp = self.__array[slices]
            self.__array = torch.empty((1,), device=self.device.torch_device)
            # necessary to clear storage of local __array
            self.__array = temp.clone().detach()
            self.__split = axis

        # entirely new split axis, need to redistribute
        else:
            _, output_shape, _ = self.comm.chunk(self.shape, axis)
            redistributed = torch.empty(
                output_shape, dtype=self.dtype.torch_type(), device=self.device.torch_device
            )

            send_counts, send_displs, _ = self.comm.counts_displs_shape(self.lshape, axis)
            recv_counts, recv_displs, _ = self.comm.counts_displs_shape(self.shape, self.split)
            self.comm.Alltoallv(
                (self.__array, send_counts, send_displs),
                (redistributed, recv_counts, recv_displs),
                send_axis=axis,
                recv_axis=self.split,
            )

            self.__array = redistributed
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
        if isinstance(key, DNDarray) and key.gshape[-1] != len(self.gshape):
            key = tuple(x.item() for x in key)
        if not self.is_distributed():
            if isinstance(key, DNDarray) and key.gshape[-1] == len(self.gshape):
                # this is for a list of values
                for i in range(key.gshape[0]):
                    self.__setter((key[i, 0].item(), key[i, 1].item()), value)
            else:
                self.__setter(key, value)
        else:
            _, _, chunk_slice = self.comm.chunk(self.shape, self.split)
            chunk_start = chunk_slice[self.split].start
            chunk_end = chunk_slice[self.split].stop

            if isinstance(key, int):
                if key < 0:
                    key += self.numdims
                if self.split == 0:
                    if key in range(chunk_start, chunk_end):
                        self.__setter(key - chunk_start, value)
                if self.split > 0:
                    if (
                        self[key].split is not None
                        and isinstance(value, DNDarray)
                        and value.split is None
                    ):
                        value = factories.array(value, split=self[key].split)
                    self.__setter(key, value)
            elif isinstance(key, (tuple, list, torch.Tensor)):
                if isinstance(key[self.split], slice):
                    key = list(key)
                    overlap = list(
                        set(
                            range(
                                key[self.split].start if key[self.split].start is not None else 0,
                                key[self.split].stop
                                if key[self.split].stop is not None
                                else self.gshape[self.split],
                                key[self.split].step if key[self.split].step is not None else 1,
                            )
                        )
                        & set(range(chunk_start, chunk_end))
                    )
                    if overlap:
                        overlap.sort()
                        hold = [x - chunk_start for x in overlap]
                        key[self.split] = slice(min(hold), max(hold) + 1, key[self.split].step)
                        try:
                            self.__setter(tuple(key), value[overlap])
                        except TypeError as te:
                            if str(te) != "'int' object is not subscriptable":
                                raise TypeError(te)
                            self.__setter(tuple(key), value)
                        except IndexError:
                            self.__setter(tuple(key), value)

                elif key[self.split] in range(chunk_start, chunk_end):
                    key = list(key)
                    key[self.split] = key[self.split] - chunk_start
                    self.__setter(tuple(key), value)

                elif key[self.split] < 0:
                    key = list(key)
                    if self.gshape[self.split] + key[self.split] in range(chunk_start, chunk_end):
                        key[self.split] = key[self.split] + chunk_end - chunk_start
                        self.__setter(tuple(key), value)

            elif isinstance(key, slice) and self.split == 0:
                overlap = list(set(range(key.start, key.stop)) & set(range(chunk_start, chunk_end)))
                if overlap:
                    overlap.sort()
                    hold = [x - chunk_start for x in overlap]
                    key = slice(min(hold), max(hold) + 1, key.step)
                    self.__setter(key, value)

            elif isinstance(key, DNDarray) and key.gshape[-1] == len(self.gshape):
                # this is the case with a list of indices to set
                key = key.copy()
                key.lloc[..., self.split] -= chunk_start
                key_new = [key._DNDarray__array[..., i] for i in range(len(self.gshape))]
                self.__setter(tuple(key_new), value)
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

    def std(self, axis=None, bessel=True):
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
        bessel : Bool
            Default: True
            use the bessel correction when calculating the varaince/std
            toggle between unbiased and biased calculation of the std

        Examples
        --------
        >>> a = ht.random.randn(1,3)
        >>> a
        tensor([[ 0.3421,  0.5736, -2.2377]])
        >>> ht.std(a)
        tensor(1.5606)
        >>> a = ht.random.randn(4,4)
        >>> a
        tensor([[-1.0206,  0.3229,  1.1800,  1.5471],
                [ 0.2732, -0.0965, -0.1087, -1.3805],
                [ 0.2647,  0.5998, -0.1635, -0.0848],
                [ 0.0343,  0.1618, -0.8064, -0.1031]])
        >>> ht.std(a, 0)
        tensor([0.6157, 0.2918, 0.8324, 1.1996])
        >>> ht.std(a, 1)
        tensor([1.1405, 0.7236, 0.3506, 0.4324])
        >>> ht.std(a, 1, bessel=False)
        tensor([0.9877, 0.6267, 0.3037, 0.3745])

        Returns
        -------
        ht.DNDarray containing the std/s, if split, then split in the same direction as x.
        """
        return statistics.std(self, axis, bessel=bessel)

    def __str__(self, *args):
        # TODO: document me
        # TODO: generate none-PyTorch str
        return self.__array.__str__(*args)

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

    def var(self, axis=None, bessel=True):
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
        bessel : Bool
            Default: True
            use the bessel correction when calculating the varaince/std
            toggle between unbiased and biased calculation of the std

        Examples
        --------
        >>> a = ht.random.randn(1,3)
        >>> a
        tensor([[-1.9755,  0.3522,  0.4751]])
        >>> ht.var(a)
        tensor(1.9065)

        >>> a = ht.random.randn(4,4)
        >>> a
        tensor([[-0.8665, -2.6848, -0.0215, -1.7363],
                [ 0.5886,  0.5712,  0.4582,  0.5323],
                [ 1.9754,  1.2958,  0.5957,  0.0418],
                [ 0.8196, -1.2911, -0.2026,  0.6212]])
        >>> ht.var(a, 1)
        tensor([1.3092, 0.0034, 0.7061, 0.9217])
        >>> ht.var(a, 0)
        tensor([1.3624, 3.2563, 0.1447, 1.2042])
        >>> ht.var(a, 0, bessel=True)
        tensor([1.3624, 3.2563, 0.1447, 1.2042])
        >>> ht.var(a, 0, bessel=False)
        tensor([1.0218, 2.4422, 0.1085, 0.9032])

        Returns
        -------
        ht.DNDarray containing the var/s, if split, then split in the same direction as x.
        """
        return statistics.var(self, axis, bessel=bessel)

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
