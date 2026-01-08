Module heat.core.dndarray
=========================
Provides HeAT's core data structure, the DNDarray, a distributed n-dimensional array

Classes
-------

`DNDarray(array: torch.Tensor, gshape: Tuple[int, ...], dtype: datatype, split: Union[int, None], device: Device, comm: Communication, balanced: bool)`
:   Distributed N-Dimensional array. The core element of HeAT. It is composed of
    PyTorch tensors local to each process.

    Parameters
    ----------
    array : torch.Tensor
        Local array elements
    gshape : Tuple[int,...]
        The global shape of the array
    dtype : datatype
        The datatype of the array
    split : int or None
        The axis on which the array is divided between processes
    device : Device
        The device on which the local arrays are using (cpu or gpu)
    comm : Communication
        The communications object for sending and receiving data
    balanced: bool or None
        Describes whether the data are evenly distributed across processes.
        If this information is not available (``self.balanced is None``), it
        can be gathered via the :func:`is_balanced()` method (requires communication).

    ### Instance variables

    `T: heat.core.dndarray.DNDarray`
    :   Permute the dimensions of an array.

        Parameters
        ----------
        a : DNDarray
            Input array.
        axes : None or List[int,...], optional
            By default, reverse the dimensions, otherwise permute the axes according to the values given.

    `array_with_halos: torch.Tensor`
    :   Fetch halos of size ``halo_size`` from neighboring ranks and save them in ``self.halo_next``/``self.halo_prev``
        in case they are not already stored. If ``halo_size`` differs from the size of already stored halos,
        the are overwritten.

    `balanced: bool`
    :   Boolean value indicating if the DNDarray is balanced between the MPI processes

    `comm: Communication`
    :   The :class:`~heat.core.communication.Communication` of the ``DNDarray``

    `device: Device`
    :   The :class:`~heat.core.devices.Device` of the ``DNDarray``

    `dtype: datatype`
    :   The :class:`~heat.core.types.datatype` of the ``DNDarray``

    `gnbytes: int`
    :   Returns the number of bytes consumed by the global ``DNDarray``

        Note
        -----------
            Does not include memory consumed by non-element attributes of the ``DNDarray`` object.

    `gnumel: int`
    :   Returns the number of total elements of the ``DNDarray``

    `gshape: Tuple`
    :   Returns the global shape of the ``DNDarray`` across all processes

    `halo_next: torch.Tensor`
    :   Returns the halo of the next process

    `halo_prev: torch.Tensor`
    :   Returns the halo of the previous process

    `imag: DNDarray`
    :   Return the imaginary part of the ``DNDarray``.

    `larray: torch.Tensor`
    :   Returns the underlying process-local ``torch.Tensor`` of the ``DNDarray``

    `lloc: Union[DNDarray, None]`
    :   Local item setter and getter. i.e. this function operates on a local
        level and only on the PyTorch tensors composing the :class:`DNDarray`.
        This function uses the LocalIndex class. As getter, it returns a ``DNDarray``
        with the indices selected at a *local* level

        Parameters
        ----------
        key : int or slice or Tuple[int,...]
            Indices of the desired data.
        value : scalar, optional
            All types compatible with pytorch tensors, if none given then this is a getter function

        Examples
        --------
        >>> a = ht.zeros((4, 5), split=0)
        DNDarray([[0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.]], dtype=ht.float32, device=cpu:0, split=0)
        >>> a.lloc[1, 0:4]
        (1/2) tensor([0., 0., 0., 0.])
        (2/2) tensor([0., 0., 0., 0.])
        >>> a.lloc[1, 0:4] = torch.arange(1, 5)
        >>> a
        DNDarray([[0., 0., 0., 0., 0.],
                  [1., 2., 3., 4., 0.],
                  [0., 0., 0., 0., 0.],
                  [1., 2., 3., 4., 0.]], dtype=ht.float32, device=cpu:0, split=0)

    `lnbytes: int`
    :   Returns the number of bytes consumed by the local ``torch.Tensor``

        Note
        -------------------
            Does not include memory consumed by non-element attributes of the ``DNDarray`` object.

    `lnumel: int`
    :   Number of elements of the ``DNDarray`` on each process

    `lshape: Tuple[int]`
    :   Returns the shape of the ``DNDarray`` on each node

    `lshape_map: torch.Tensor`
    :   Returns the lshape map. If it hasn't been previously created then it will be created here.

    `nbytes: int`
    :   Returns the number of bytes consumed by the global tensor. Equivalent to property gnbytes.

        Note
        ------------
            Does not include memory consumed by non-element attributes of the ``DNDarray`` object.

    `ndim: int`
    :   Number of dimensions of the ``DNDarray``

    `real: DNDarray`
    :   Return the real part of the ``DNDarray``.

    `shape: Tuple[int]`
    :   Returns the shape of the ``DNDarray`` as a whole

    `size: int`
    :   Number of total elements of the ``DNDarray``

    `split: int`
    :   Returns the axis on which the ``DNDarray`` is split

    `stride: Tuple[int]`
    :   Returns the steps in each dimension when traversing a ``DNDarray``. torch-like usage: ``self.stride()``

    `strides: Tuple[int]`
    :   Returns bytes to step in each dimension when traversing a ``DNDarray``. numpy-like usage: ``self.strides()``

    ### Methods

    `abs(self, out=None, dtype=None)`
    :   Returns :class:`~heat.core.dndarray.DNDarray` containing the elementwise abolute values of the input array ``x``.

        Parameters
        ----------
        x : DNDarray
            The array for which the compute the absolute value.
        out : DNDarray, optional
            A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
            If not provided or ``None``, a freshly-allocated array is returned.
        dtype : datatype, optional
            Determines the data type of the output array. The values are cast to this type with potential loss of
            precision.

        Raises
        ------
        TypeError
            If dtype is not a heat type.

    `absolute(self, out=None, dtype=None)`
    :   Calculate the absolute value element-wise.
        :func:`abs` is a shorthand for this function.

        Parameters
        ----------
        x : DNDarray
            The array for which the compute the absolute value.
        out : DNDarray, optional
            A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
            If not provided or ``None``, a freshly-allocated array is returned.
        dtype : datatype, optional
            Determines the data type of the output array. The values are cast to this type with potential loss of
            precision.

    `acos(self, out=None)`
    :   Compute the trigonometric arccos, element-wise.
        Result is a ``DNDarray`` of the same shape as ``x``.
        Input elements outside [-1., 1.] are returned as ``NaN``. If ``out`` was provided, ``arccos`` is a reference to it.

        Parameters
        ----------
        x : DNDarray
            The array for which to compute the trigonometric cosine.
        out : DNDarray, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to ``None``, a fresh array is allocated.

        Examples
        --------
        >>> ht.arccos(ht.array([-1.0, -0.0, 0.83]))
        DNDarray([3.1416, 1.5708, 0.5917], dtype=ht.float32, device=cpu:0, split=None)

    `add_(t1: DNDarray, t2: Union[DNDarray, float]) ‑> heat.core.dndarray.DNDarray`
    :   Element-wise in-place addition of values of two operands.
        Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise adds the
        element(s) of the second operand (scalar or :class:`~heat.core.dndarray.DNDarray`) in-place,
        i.e. the element(s) of `t1` are overwritten by the results of element-wise addition of `t1` and
        `t2`.
        Can be called as a DNDarray method or with the symbol `+=`.

        Parameters
        ----------
        t1: DNDarray
            The first operand involved in the addition
        t2: DNDarray or scalar
            The second operand involved in the addition

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.float32([[1, 2], [3, 4]])
        >>> T2 = ht.float32([[2, 2], [2, 2]])
        >>> T1 += T2
        >>> T1
        DNDarray([[3., 4.],
                  [5., 6.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[2., 2.],
                  [2., 2.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> s = 2.0
        >>> T2.add_(s)
        DNDarray([[4., 4.],
                  [4., 4.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[4., 4.],
                  [4., 4.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> s
        2.0

    `all(self, axis=None, out=None, keepdims=False)`
    :   Test whether all array elements along a given axis evaluate to ``True``.
        A new boolean or :class:`~heat.core.dndarray.DNDarray` is returned unless out is specified, in which case a
        reference to ``out`` is returned.

        Parameters
        ----------
        x : DNDarray
            Input array or object that can be converted to an array.
        axis : None or int or Tuple[int,...], optional
            Axis or axes along which a logical AND reduction is performed. The default (``axis=None``) is to perform a
            logical AND over all the dimensions of the input array. ``axis`` may be negative, in which case it counts
            from the last to the first axis.
        out : DNDarray, optional
            Alternate output array in which to place the result. It must have the same shape as the expected output
            and its type is preserved.
        keepdims : bool, optional
            If this is set to ``True``, the axes which are reduced are left in the result as dimensions with size one.
            With this option, the result will broadcast correctly against the original array.

        Examples
        --------
        >>> x = ht.random.randn(4, 5)
        >>> x
        DNDarray([[ 0.7199,  1.3718,  1.5008,  0.3435,  1.2884],
                  [ 0.1532, -0.0968,  0.3739,  1.7843,  0.5614],
                  [ 1.1522,  1.9076,  1.7638,  0.4110, -0.2803],
                  [-0.5475, -0.0271,  0.8564, -1.5870,  1.3108]], dtype=ht.float32, device=cpu:0, split=None)
        >>> y = x < 0.5
        >>> y
        DNDarray([[False, False, False,  True, False],
                  [ True,  True,  True, False, False],
                  [False, False, False,  True,  True],
                  [ True,  True, False,  True, False]], dtype=ht.bool, device=cpu:0, split=None)
        >>> ht.all(y)
        DNDarray([False], dtype=ht.bool, device=cpu:0, split=None)
        >>> ht.all(y, axis=0)
        DNDarray([False, False, False, False, False], dtype=ht.bool, device=cpu:0, split=None)
        >>> ht.all(x, axis=1)
        DNDarray([True, True, True, True], dtype=ht.bool, device=cpu:0, split=None)
        >>> out = ht.zeros(5)
        >>> ht.all(y, axis=0, out=out)
        DNDarray([False, False, False, False, False], dtype=ht.float32, device=cpu:0, split=None)
        >>> out
        DNDarray([False, False, False, False, False], dtype=ht.float32, device=cpu:0, split=None)

    `allclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False)`
    :   Test whether all array elements along a given axis evaluate to ``True``.
        A new boolean or :class:`~heat.core.dndarray.DNDarray` is returned unless out is specified, in which case a
        reference to ``out`` is returned.

        Parameters
        ----------
        x : DNDarray
            Input array or object that can be converted to an array.
        axis : None or int or Tuple[int,...], optional
            Axis or axes along which a logical AND reduction is performed. The default (``axis=None``) is to perform a
            logical AND over all the dimensions of the input array. ``axis`` may be negative, in which case it counts
            from the last to the first axis.
        out : DNDarray, optional
            Alternate output array in which to place the result. It must have the same shape as the expected output
            and its type is preserved.
        keepdims : bool, optional
            If this is set to ``True``, the axes which are reduced are left in the result as dimensions with size one.
            With this option, the result will broadcast correctly against the original array.

        Examples
        --------
        >>> x = ht.random.randn(4, 5)
        >>> x
        DNDarray([[ 0.7199,  1.3718,  1.5008,  0.3435,  1.2884],
                  [ 0.1532, -0.0968,  0.3739,  1.7843,  0.5614],
                  [ 1.1522,  1.9076,  1.7638,  0.4110, -0.2803],
                  [-0.5475, -0.0271,  0.8564, -1.5870,  1.3108]], dtype=ht.float32, device=cpu:0, split=None)
        >>> y = x < 0.5
        >>> y
        DNDarray([[False, False, False,  True, False],
                  [ True,  True,  True, False, False],
                  [False, False, False,  True,  True],
                  [ True,  True, False,  True, False]], dtype=ht.bool, device=cpu:0, split=None)
        >>> ht.all(y)
        DNDarray([False], dtype=ht.bool, device=cpu:0, split=None)
        >>> ht.all(y, axis=0)
        DNDarray([False, False, False, False, False], dtype=ht.bool, device=cpu:0, split=None)
        >>> ht.all(x, axis=1)
        DNDarray([True, True, True, True], dtype=ht.bool, device=cpu:0, split=None)
        >>> out = ht.zeros(5)
        >>> ht.all(y, axis=0, out=out)
        DNDarray([False, False, False, False, False], dtype=ht.float32, device=cpu:0, split=None)
        >>> out
        DNDarray([False, False, False, False, False], dtype=ht.float32, device=cpu:0, split=None)

    `any(self, axis=None, out=None, keepdims=False)`
    :   Returns a :class:`~heat.core.dndarray.DNDarray` containing the result of the test whether any array elements along a
        given axis evaluate to ``True``.
        The returning array is one dimensional unless axis is not ``None``.

        Parameters
        ----------
        x : DNDarray
            Input tensor
        axis : int, optional
            Axis along which a logic OR reduction is performed. With ``axis=None``, the logical OR is performed over all
            dimensions of the array.
        out : DNDarray, optional
            Alternative output tensor in which to place the result. It must have the same shape as the expected output.
            The output is a array with ``datatype=bool``.
        keepdims : bool, optional
            If this is set to ``True``, the axes which are reduced are left in the result as dimensions with size one.
            With this option, the result will broadcast correctly against the original array.

        Examples
        --------
        >>> x = ht.float32([[0.3, 0, 0.5]])
        >>> x.any()
        DNDarray([True], dtype=ht.bool, device=cpu:0, split=None)
        >>> x.any(axis=0)
        DNDarray([ True, False,  True], dtype=ht.bool, device=cpu:0, split=None)
        >>> x.any(axis=1)
        DNDarray([True], dtype=ht.bool, device=cpu:0, split=None)
        >>> y = ht.int32([[0, 0, 1], [0, 0, 0]])
        >>> res = ht.zeros(3, dtype=ht.bool)
        >>> y.any(axis=0, out=res)
        DNDarray([False, False,  True], dtype=ht.bool, device=cpu:0, split=None)
        >>> res
        DNDarray([False, False,  True], dtype=ht.bool, device=cpu:0, split=None)

    `argmax(self, axis=None, out=None, **kwargs)`
    :   Returns an array of the indices of the maximum values along an axis. It has the same shape as ``x.shape`` with the
        dimension along axis removed.

        Parameters
        ----------
        x : DNDarray
            Input array.
        axis : int, optional
            By default, the index is into the flattened array, otherwise along the specified axis.
        out : DNDarray, optional.
            If provided, the result will be inserted into this array. It should be of the appropriate shape and dtype.
        **kwargs
            Extra keyword arguments

        Examples
        --------
        >>> a = ht.random.randn(3, 3)
        >>> a
        DNDarray([[ 1.0661,  0.7036, -2.0908],
                  [-0.7534, -0.4986, -0.7751],
                  [-0.4815,  1.9436,  0.6400]], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.argmax(a)
        DNDarray([7], dtype=ht.int64, device=cpu:0, split=None)
        >>> ht.argmax(a, axis=0)
        DNDarray([0, 2, 2], dtype=ht.int64, device=cpu:0, split=None)
        >>> ht.argmax(a, axis=1)
        DNDarray([0, 1, 1], dtype=ht.int64, device=cpu:0, split=None)

    `argmin(self, axis=None, out=None, **kwargs)`
    :   Returns an array of the indices of the minimum values along an axis. It has the same shape as ``x.shape`` with the
        dimension along axis removed.

        Parameters
        ----------
        x : DNDarray
            Input array.
        axis : int, optional
            By default, the index is into the flattened array, otherwise along the specified axis.
        out : DNDarray, optional
            Issue #100 If provided, the result will be inserted into this array. It should be of the appropriate shape and dtype.
        **kwargs
            Extra keyword arguments

        Examples
        --------
        >>> a = ht.random.randn(3, 3)
        >>> a
        DNDarray([[ 1.0661,  0.7036, -2.0908],
                  [-0.7534, -0.4986, -0.7751],
                  [-0.4815,  1.9436,  0.6400]], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.argmin(a)
        DNDarray([2], dtype=ht.int64, device=cpu:0, split=None)
        >>> ht.argmin(a, axis=0)
        DNDarray([1, 1, 0], dtype=ht.int64, device=cpu:0, split=None)
        >>> ht.argmin(a, axis=1)
        DNDarray([2, 2, 0], dtype=ht.int64, device=cpu:0, split=None)

    `asin(self, out=None)`
    :   Compute the trigonometric arcsin, element-wise.
        Result is a ``DNDarray`` of the same shape as ``x``.
        Input elements outside [-1., 1.] are returned as ``NaN``. If ``out`` was provided, ``arcsin`` is a reference to it.

        Parameters
        ----------
        x : DNDarray
            The array for which to compute the trigonometric cosine.
        out : DNDarray, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to ``None``, a fresh array is allocated.

        Examples
        --------
        >>> ht.arcsin(ht.array([-1.0, -0.0, 0.83]))
        DNDarray([-1.5708, -0.0000,  0.9791], dtype=ht.float32, device=cpu:0, split=None)

    `astype(self, dtype, copy=True) ‑> heat.core.dndarray.DNDarray`
    :   Returns a casted version of this array.
        Casted array is a new array of the same shape but with given type of this array. If copy is ``True``, the
        same array is returned instead.

        Parameters
        ----------
        dtype : datatype
            Heat type to which the array is cast
        copy : bool, optional
            By default the operation returns a copy of this array. If copy is set to ``False`` the cast is performed
            in-place and this array is returned

    `atan(self, out=None)`
    :   Compute the trigonometric arctan, element-wise.
        Result is a ``DNDarray`` of the same shape as ``x``.
        Input elements outside [-1., 1.] are returned as ``NaN``. If ``out`` was provided, ``arctan`` is a reference to it.

        Parameters
        ----------
        x : DNDarray
            The array for which to compute the trigonometric cosine.
        out : DNDarray, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to ``None``, a fresh array is allocated.

        Examples
        --------
        >>> ht.arctan(ht.arange(-6, 7, 2))
        DNDarray([-1.4056, -1.3258, -1.1071,  0.0000,  1.1071,  1.3258,  1.4056], dtype=ht.float32, device=cpu:0, split=None)

    `atan2(self, x2)`
    :   Element-wise arc tangent of ``x1/x2`` choosing the quadrant correctly.
        Returns a new ``DNDarray`` with the signed angles in radians between vector (``x2``,``x1``) and vector (1,0)

        Parameters
        ----------
        x1 : DNDarray
             y-coordinates
        x2 : DNDarray
             x-coordinates. If ``x1.shape!=x2.shape``, they must be broadcastable to a common shape (which becomes the shape of the output).

        Examples
        --------
        >>> x = ht.array([-1, +1, +1, -1])
        >>> y = ht.array([-1, -1, +1, +1])
        >>> ht.arctan2(y, x) * 180 / ht.pi
        DNDarray([-135.0000,  -45.0000,   45.0000,  135.0000], dtype=ht.float64, device=cpu:0, split=None)

    `average(self, axis=None, weights=None, returned=False)`
    :   Compute the weighted average along the specified axis.

        If ``returned=True``, return a tuple with the average as the first element and the sum
        of the weights as the second element. ``sum_of_weights`` is of the same type as ``average``.

        Parameters
        ----------
        x : DNDarray
            Array containing data to be averaged.
        axis : None or int or Tuple[int,...], optional
            Axis or axes along which to average ``x``.  The default,
            ``axis=None``, will average over all of the elements of the input array.
            If axis is negative it counts from the last to the first axis.
            #TODO Issue #351: If axis is a tuple of ints, averaging is performed on all of the axes
            specified in the tuple instead of a single axis or all the axes as
            before.
        weights : DNDarray, optional
            An array of weights associated with the values in ``x``. Each value in
            ``x`` contributes to the average according to its associated weight.
            The weights array can either be 1D (in which case its length must be
            the size of ``x`` along the given axis) or of the same shape as ``x``.
            If ``weights=None``, then all data in ``x`` are assumed to have a
            weight equal to one, the result is equivalent to :func:`mean`.
        returned : bool, optional
            If ``True``, the tuple ``(average, sum_of_weights)``
            is returned, otherwise only the average is returned.
            If ``weights=None``, ``sum_of_weights`` is equivalent to the number of
            elements over which the average is taken.

        Raises
        ------
        ZeroDivisionError
            When all weights along axis are zero.
        TypeError
            When the length of 1D weights is not the same as the shape of ``x``
            along axis.

        Examples
        --------
        >>> data = ht.arange(1, 5, dtype=float)
        >>> data
        DNDarray([1., 2., 3., 4.], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.average(data)
        DNDarray(2.5000, dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.average(ht.arange(1, 11, dtype=float), weights=ht.arange(10, 0, -1))
        DNDarray([4.], dtype=ht.float64, device=cpu:0, split=None)
        >>> data = ht.array([[0, 1],
                             [2, 3],
                            [4, 5]], dtype=float, split=1)
        >>> weights = ht.array([1.0 / 4, 3.0 / 4])
        >>> ht.average(data, axis=1, weights=weights)
        DNDarray([0.7500, 2.7500, 4.7500], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.average(data, weights=weights)
        Traceback (most recent call last):
            ...
        TypeError: Axis must be specified when shapes of x and weights differ.

    `balance(self, copy=False)`
    :   Out of place balance function. More information on the meaning of balance can be found in
        :func:`DNDarray.balance_() <heat.core.dndarray.DNDarray.balance_()>`.

        Parameters
        ----------
        array : DNDarray
            the DNDarray to be balanced
        copy : bool, optional
            if the DNDarray should be copied before being balanced. If false (default) this will balance
            the original array and return that array. Otherwise (true), a balanced copy of the array
            will be returned.
            Default: False

    `balance_(self) ‑> heat.core.dndarray.DNDarray`
    :   Function for balancing a :class:`DNDarray` between all nodes. To determine if this is needed use the :func:`is_balanced()` function.
        If the ``DNDarray`` is already balanced this function will do nothing. This function modifies the ``DNDarray``
        itself and will not return anything.

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

    `bitwise_and_(t1: DNDarray, t2: Union[DNDarray, float]) ‑> heat.core.dndarray.DNDarray`
    :   Bitwise AND of two operands computed element-wise and in-place.
        Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise computes the
        bitwise AND with the corresponding element(s) of the second operand (scalar or
        :class:`~heat.core.dndarray.DNDarray`) in-place, i.e. the element(s) of `t1` are overwritten by
        the results of element-wise bitwise AND of `t1` and `t2`.
        Can be called as a DNDarray method or with the symbol `&=`. Only integer and boolean types are
        handled.

        Parameters
        ----------
        t1: DNDarray
            The first operand involved in the operation
        t2: DNDarray or scalar
            The second operand involved in the operation

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.array(13)
        >>> T2 = ht.array(17)
        >>> T1 &= T2
        >>> T1
        DNDarray(1, dtype=ht.int64, device=cpu:0, split=None)
        >>> T2
        DNDarray(17, dtype=ht.int64, device=cpu:0, split=None)
        >>> T3 = ht.array(22)
        >>> T2.bitwise_and_(T3)
        DNDarray(16, dtype=ht.int64, device=cpu:0, split=None)
        >>> T2
        DNDarray(16, dtype=ht.int64, device=cpu:0, split=None)
        >>> T4 = ht.array([14, 3])
        >>> s = 29
        >>> T4 &= s
        >>> T4
        DNDarray([12,  1], dtype=ht.int64, device=cpu:0, split=None)
        >>> s
        29
        >>> T5 = ht.array([2, 5, 255])
        >>> T6 = ht.array([3, 14, 16])
        >>> T5 &= T6
        >>> T5
        DNDarray([ 2,  4, 16], dtype=ht.int64, device=cpu:0, split=None)
        >>> T7 = ht.array([True, True])
        >>> T8 = ht.array([False, True])
        >>> T7 &= T8
        >>> T7
        DNDarray([False,  True], dtype=ht.bool, device=cpu:0, split=None)

    `bitwise_not_(t: DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Computes the bitwise NOT of the given input :class:`~heat.core.dndarray.DNDarray` in-place. The
        elements of the input array must be of integer or Boolean types. For boolean arrays, it computes
        the logical NOT.
        Can only be called as a DNDarray method. `bitwise_not_` is an alias for `invert_`.

        Parameters
        ----------
        t:  DNDarray
            The input array to invert. Must be of integral or Boolean types

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.array(13, dtype=ht.uint8)
        >>> T1.invert_()
        DNDarray(242, dtype=ht.uint8, device=cpu:0, split=None)
        >>> T1
        DNDarray(242, dtype=ht.uint8, device=cpu:0, split=None)
        >>> T2 = ht.array([-1, -2, 3], dtype=ht.int8)
        >>> T2.invert_()
        DNDarray([ 0,  1, -4], dtype=ht.int8, device=cpu:0, split=None)
        >>> T2
        DNDarray([ 0,  1, -4], dtype=ht.int8, device=cpu:0, split=None)
        >>> T3 = ht.array([[True, True], [False, True]])
        >>> T3.invert_()
        DNDarray([[False, False],
                  [ True, False]], dtype=ht.bool, device=cpu:0, split=None)
        >>> T3
        DNDarray([[False, False],
                  [ True, False]], dtype=ht.bool, device=cpu:0, split=None)

    `bitwise_or_(t1: DNDarray, t2: Union[DNDarray, float]) ‑> heat.core.dndarray.DNDarray`
    :   Bitwise OR of two operands computed element-wise and in-place.
        Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise computes the
        bitwise OR with the corresponding element(s) of the second operand (scalar or
        :class:`~heat.core.dndarray.DNDarray`) in-place, i.e. the element(s) of `t1` are overwritten by
        the results of element-wise bitwise OR of `t1` and `t2`.
        Can be called as a DNDarray method or with the symbol `|=`. Only integer and boolean types are
        handled.

        Parameters
        ----------
        t1: DNDarray
            The first operand involved in the operation
        t2: DNDarray or scalar
            The second operand involved in the operation

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.array(13)
        >>> T2 = ht.array(16)
        >>> T1 |= T2
        >>> T1
        DNDarray(29, dtype=ht.int64, device=cpu:0, split=None)
        >>> T2
        DNDarray(16, dtype=ht.int64, device=cpu:0, split=None)
        >>> T3 = ht.array([33, 4])
        >>> s = 1
        >>> T3.bitwise_or_(s)
        DNDarray([33,  5], dtype=ht.int64, device=cpu:0, split=None)
        >>> T3
        DNDarray([33,  5], dtype=ht.int64, device=cpu:0, split=None)
        >>> s
        1
        >>> T4 = ht.array([2, 5, 255])
        >>> T5 = ht.array([4, 4, 4])
        >>> T4 |= T5
        >>> T4
        DNDarray([  6,   5, 255], dtype=ht.int64, device=cpu:0, split=None)
        >>> T6 = ht.array([True, True])
        >>> T7 = ht.array([False, True])
        >>> T6 |= T7
        >>> T6
        DNDarray([True, True], dtype=ht.bool, device=cpu:0, split=None)

    `bitwise_xor_(t1: DNDarray, t2: Union[DNDarray, float]) ‑> heat.core.dndarray.DNDarray`
    :   Bitwise XOR of two operands computed element-wise and in-place.
        Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise computes the
        bitwise XOR with the corresponding element(s) of the second operand (scalar or
        :class:`~heat.core.dndarray.DNDarray`) in-place, i.e. the element(s) of `t1` are overwritten by
        the results of element-wise bitwise XOR of `t1` and `t2`.
        Can be called as a DNDarray method or with the symbol `^=`. Only integer and boolean types are
        handled.

        Parameters
        ----------
        t1: DNDarray
            The first operand involved in the operation
        t2: DNDarray or scalar
            The second operand involved in the operation

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.array(13)
        >>> T2 = ht.array(17)
        >>> T1 ^= T2
        >>> T1
        DNDarray(28, dtype=ht.int64, device=cpu:0, split=None)
        >>> T2
        DNDarray(17, dtype=ht.int64, device=cpu:0, split=None)
        >>> T3 = ht.array([31, 3])
        >>> s = 5
        >>> T3.bitwise_xor_(s)
        DNDarray([26,  6], dtype=ht.int64, device=cpu:0, split=None)
        >>> T3
        DNDarray([26,  6], dtype=ht.int64, device=cpu:0, split=None)
        >>> s
        5
        >>> T4 = ht.array([31, 3, 255])
        >>> T5 = ht.array([5, 6, 4])
        >>> T4 ^= T5
        >>> T4
        DNDarray([ 26,   5, 251], dtype=ht.int64, device=cpu:0, split=None)
        >>> T6 = ht.array([True, True])
        >>> T7 = ht.array([False, True])
        >>> T6 ^= T7
        >>> T6
        DNDarray([ True, False], dtype=ht.bool, device=cpu:0, split=None)

    `ceil(self, out=None)`
    :   Return the ceil of the input, element-wise. Result is a :class:`~heat.core.dndarray.DNDarray` of the same shape as
        ``x``. The ceil of the scalar ``x`` is the smallest integer i, such that ``i>=x``. It is often denoted as
        :math:`\lceil x \rceil`.

        Parameters
        ----------
        x : DNDarray
            The value for which to compute the ceiled values.
        out : DNDarray, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to ``None``, a fresh array is allocated.

        Examples
        --------
        >>> import heat as ht
        >>> ht.ceil(ht.arange(-2.0, 2.0, 0.4))
        DNDarray([-2., -1., -1., -0., -0.,  0.,  1.,  1.,  2.,  2.], dtype=ht.float32, device=cpu:0, split=None)

    `clip(self, a_min, a_max, out=None)`
    :   Returns a :class:`~heat.core.dndarray.DNDarray` with the elements of this array, but where values
        ``<a_min`` are replaced with ``a_min``, and those ``>a_max`` with ``a_max``.

        Parameters
        ----------
        x : DNDarray
            Array containing elements to clip.
        min : scalar or None
            Minimum value. If ``None``, clipping is not performed on lower interval edge. Not more than one of ``a_min`` and
            ``a_max`` may be ``None``.
        max : scalar or None
            Maximum value. If ``None``, clipping is not performed on upper interval edge. Not more than one of ``a_min`` and
            ``a_max`` may be None.
        out : DNDarray, optional
            The results will be placed in this array. It may be the input array for in-place clipping. ``out`` must be of
            the right shape to hold the output. Its type is preserved.

        Raises
        ------
        ValueError
            if either min or max is not set

    `collect(arr, target_rank=0)`
    :   A function collecting a distributed DNDarray to one rank, chosen by the `target_rank` variable.
        It is a specific case of the ``redistribute_`` method.

        Parameters
        ----------
        arr : DNDarray
            The DNDarray to be collected.
        target_rank : int, optional
            The rank to which the DNDarray will be collected. Default: 0.

        Raises
        ------
        TypeError
            If the target rank is not an integer.
        ValueError
            If the target rank is out of bounds.

        Examples
        --------
        >>> st = ht.ones((50, 81, 67), split=2)
        >>> print(st.lshape)
        [0/2] (50, 81, 23)
        [1/2] (50, 81, 22)
        [2/2] (50, 81, 22)
        >>> collected_st = collect(st)
        >>> print(collected_st)
        [0/2] (50, 81, 67)
        [1/2] (50, 81, 0)
        [2/2] (50, 81, 0)
        >>> collected_st = collect(collected_st, 1)
        >>> print(st.lshape)
        [0/2] (50, 81, 0)
        [1/2] (50, 81, 67)
        [2/2] (50, 81, 0)

    `collect_(self, target_rank: Optional[int] = 0) ‑> None`
    :   A method collecting a distributed DNDarray to one MPI rank, chosen by the `target_rank` variable.
        It is a specific case of the ``redistribute_`` method.

        Parameters
        ----------
        target_rank : int, optional
            The rank to which the DNDarray will be collected. Default: 0.

        Raises
        ------
        TypeError
            If the target rank is not an integer.
        ValueError
            If the target rank is out of bounds.

        Examples
        --------
        >>> st = ht.ones((50, 81, 67), split=2)
        >>> print(st.lshape)
        [0/2] (50, 81, 23)
        [1/2] (50, 81, 22)
        [2/2] (50, 81, 22)
        >>> st.collect_()
        >>> print(st.lshape)
        [0/2] (50, 81, 67)
        [1/2] (50, 81, 0)
        [2/2] (50, 81, 0)
        >>> st.collect_(1)
        >>> print(st.lshape)
        [0/2] (50, 81, 0)
        [1/2] (50, 81, 67)
        [2/2] (50, 81, 0)

    `conj(self, out=None)`
    :   Compute the complex conjugate, element-wise.

        Parameters
        ----------
        x : DNDarray
            Input array for which to compute the complex conjugate.
        out : DNDarray, optional
            Output array with the complex conjugates.

        Examples
        --------
        >>> ht.conjugate(ht.array([1.0, 1.0j, 1 + 1j, -2 + 2j, 3 - 3j]))
        DNDarray([ (1-0j),     -1j,  (1-1j), (-2-2j),  (3+3j)], dtype=ht.complex64, device=cpu:0, split=None)

    `copy(self)`
    :   Return a deep copy of the given object.

        Parameters
        ----------
        x : DNDarray
            Input array to be copied.

        Examples
        --------
        >>> a = ht.array([1, 2, 3])
        >>> b = ht.copy(a)
        >>> b
        DNDarray([1, 2, 3], dtype=ht.int64, device=cpu:0, split=None)
        >>> a[0] = 4
        >>> a
        DNDarray([4, 2, 3], dtype=ht.int64, device=cpu:0, split=None)
        >>> b
        DNDarray([1, 2, 3], dtype=ht.int64, device=cpu:0, split=None)

    `copysign_(t1: DNDarray, t2: Union[DNDarray, float]) ‑> heat.core.dndarray.DNDarray`
    :   In-place version of the element-wise operation 'copysign'.
        The magnitudes of the element(s) of 't1' are kept but the sign(s) are adopted from the
        element(s) of 't2'.
        Can only be called as a DNDarray method.

        Parameters
        ----------
        t1:    DNDarray
               The input array
               Entries must be of type float.
        t2:    DNDarray or scalar
               value(s) whose signbit(s) are applied to the magnitudes in 't1'

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            At the moment, the operation only works for DNDarrays whose elements are floats and are not
            complex. This is due to the fact that it relies on the PyTorch function 'copysign_', which
            does not work if the entries of 't1' are integers. The case when 't1' contains floats and
            't2' contains integers works in PyTorch but has not been implemented properly in Heat yet.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.array([3.0, 2.0, -8.0, -2.0, 4.0])
        >>> s = 2.0
        >>> T1.copysign_(s)
        DNDarray([3., 2., 8., 2., 4.], dtype=ht.float32, device=cpu:0, split=None)
        >>> T1
        DNDarray([3., 2., 8., 2., 4.], dtype=ht.float32, device=cpu:0, split=None)
        >>> s
        2.0
        >>> T2 = ht.array([[1.0, -1.0], [1.0, -1.0]])
        >>> T3 = ht.array([-5.0, 2.0])
        >>> T2.copysign_(T3)
        DNDarray([[-1.,  1.],
                  [-1.,  1.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[-1.,  1.],
                  [-1.,  1.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T3
        DNDarray([-5.,  2.], dtype=ht.float32, device=cpu:0, split=None)

    `cos(self, out=None)`
    :   Return the trigonometric cosine, element-wise.

        Parameters
        ----------
        x : ht.DNDarray
            The value for which to compute the trigonometric cosine.
        out : ht.DNDarray or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to None, a fresh tensor is allocated.

        Examples
        --------
        >>> ht.cos(ht.arange(-6, 7, 2))
        DNDarray([ 0.9602, -0.6536, -0.4161,  1.0000, -0.4161, -0.6536,  0.9602], dtype=ht.float32, device=cpu:0, split=None)

    `cosh(self, out=None)`
    :   Compute the hyperbolic cosine, element-wise.
        Result is a ``DNDarray`` of the same shape as ``x``.
        Negative input elements are returned as ``NaN``. If ``out`` was provided, ``cosh`` is a reference to it.

        Parameters
        ----------
        x : DNDarray
            The value for which to compute the hyperbolic cosine.
        out : DNDarray, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to ``None``, a fresh array is allocated.

        Examples
        --------
        >>> ht.cosh(ht.arange(-6, 7, 2))
        DNDarray([201.7156,  27.3082,   3.7622,   1.0000,   3.7622,  27.3082, 201.7156], dtype=ht.float32, device=cpu:0, split=None)

    `counts_displs(self) ‑> Tuple[Tuple[int], Tuple[int]]`
    :   Returns actual counts (number of items per process) and displacements (offsets) of the DNDarray.
        Does not assume load balance.

    `cpu(self) ‑> heat.core.dndarray.DNDarray`
    :   Returns a copy of this object in main memory. If this object is already in main memory, then no copy is
        performed and the original object is returned.

    `create_lshape_map(self, force_check: bool = False) ‑> torch.Tensor`
    :   Generate a 'map' of the lshapes of the data on all processes.
        Units are ``(process rank, lshape)``

        Parameters
        ----------
        force_check : bool, optional
            if False (default) and the lshape map has already been created, use the previous
            result. Otherwise, create the lshape_map

    `create_partition_interface(self)`
    :   Create a partition interface in line with the DPPY proposal. This is subject to change.
        The intention of this to facilitate the usage of a general format for the referencing of
        distributed datasets.

        An example of the output and shape is shown below.

        __partitioned__ = {
            'shape': (27, 3, 2),
            'partition_tiling': (4, 1, 1),
            'partitions': {
                (0, 0, 0): {
                    'start': (0, 0, 0),
                    'shape': (7, 3, 2),
                    'data': tensor([...], dtype=torch.int32),
                    'location': [0],
                    'dtype': torch.int32,
                    'device': 'cpu'
                },
                (1, 0, 0): {
                    'start': (7, 0, 0),
                    'shape': (7, 3, 2),
                    'data': None,
                    'location': [1],
                    'dtype': torch.int32,
                    'device': 'cpu'
                },
                (2, 0, 0): {
                    'start': (14,  0,  0),
                    'shape': (7, 3, 2),
                    'data': None,
                    'location': [2],
                    'dtype': torch.int32,
                    'device': 'cpu'
                },
                (3, 0, 0): {
                    'start': (21,  0,  0),
                    'shape': (6, 3, 2),
                    'data': None,
                    'location': [3],
                    'dtype': torch.int32,
                    'device': 'cpu'
                }
            },
            'locals': [(rank, 0, 0)],
            'get': lambda x: x,
        }

        Returns
        -------
        dictionary containing the partition interface as shown above.

    `cumprod_(t: DNDarray, axis: int) ‑> heat.core.dndarray.DNDarray`
    :   Return the cumulative product of elements along a given axis in-place.
        Can only be called as a DNDarray method.

        Parameters
        ----------
        t:      DNDarray
                Input array.
        axis:   int
                Axis along which the cumulative product is computed.

        Examples
        --------
        >>> import heat as ht
        >>> T = ht.full((3, 3), 2)
        >>> T.cumprod_(0)
        DNDarray([[2., 2., 2.],
                  [4., 4., 4.],
                  [8., 8., 8.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T
        DNDarray([[2., 2., 2.],
                  [4., 4., 4.],
                  [8., 8., 8.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T.cumproduct_(1)
        DNDarray([[  2.,   4.,   8.],
                  [  4.,  16.,  64.],
                  [  8.,  64., 512.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T
        DNDarray([[  2.,   4.,   8.],
                  [  4.,  16.,  64.],
                  [  8.,  64., 512.]], dtype=ht.float32, device=cpu:0, split=None)

    `cumproduct_(t: DNDarray, axis: int) ‑> heat.core.dndarray.DNDarray`
    :   Return the cumulative product of elements along a given axis in-place.
        Can only be called as a DNDarray method.

        Parameters
        ----------
        t:      DNDarray
                Input array.
        axis:   int
                Axis along which the cumulative product is computed.

        Examples
        --------
        >>> import heat as ht
        >>> T = ht.full((3, 3), 2)
        >>> T.cumprod_(0)
        DNDarray([[2., 2., 2.],
                  [4., 4., 4.],
                  [8., 8., 8.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T
        DNDarray([[2., 2., 2.],
                  [4., 4., 4.],
                  [8., 8., 8.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T.cumproduct_(1)
        DNDarray([[  2.,   4.,   8.],
                  [  4.,  16.,  64.],
                  [  8.,  64., 512.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T
        DNDarray([[  2.,   4.,   8.],
                  [  4.,  16.,  64.],
                  [  8.,  64., 512.]], dtype=ht.float32, device=cpu:0, split=None)

    `cumsum_(t: DNDarray, axis: int) ‑> heat.core.dndarray.DNDarray`
    :   Return the cumulative sum of the elements along a given axis in-place.
        Can only be called as a DNDarray method.

        Parameters
        ----------
        t:      DNDarray
                Input array.
        axis:   int
                Axis along which the cumulative sum is computed.

        Examples
        --------
        >>> import heat as ht
        >>> T = ht.ones((3, 3))
        >>> T.cumsum_(0)
        DNDarray([[1., 1., 1.],
                  [2., 2., 2.],
                  [3., 3., 3.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T
        DNDarray([[1., 1., 1.],
                  [2., 2., 2.],
                  [3., 3., 3.]], dtype=ht.float32, device=cpu:0, split=None)

    `div_(t1: DNDarray, t2: Union[DNDarray, float]) ‑> heat.core.dndarray.DNDarray`
    :   Element-wise in-place true division of values of two operands.
        Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise divides its
        element(s) by the element(s) of the second operand (scalar or
        :class:`~heat.core.dndarray.DNDarray`) in-place, i.e. the element(s) of `t1` are overwritten by
        the results of element-wise division of `t1` and `t2`.
        Can be called as a DNDarray method or with the symbol `/=`. `divide_` is an alias for `div_`.

        Parameters
        ----------
        t1: DNDarray
            The first operand whose values are divided.
        t2: DNDarray or scalar
            The second operand by whose values is divided.

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Example
        ---------
        >>> import heat as ht
        >>> T1 = ht.float32([[1, 2], [3, 4]])
        >>> T2 = ht.float32([[2, 2], [2, 2]])
        >>> T1 /= T2
        >>> T1
        DNDarray([[0.5000, 1.0000],
                  [1.5000, 2.0000]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[2., 2.],
                  [2., 2.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> s = 2.0
        >>> T2.div_(s)
        DNDarray([[1., 1.],
                  [1., 1.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[1., 1.],
                  [1., 1.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> s
        2.0
        >>> v = ht.int32([-1, 2])
        >>> T2.divide_(v)
        DNDarray([[-1.0000,  0.5000],
                  [-1.0000,  0.5000]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[-1.0000,  0.5000],
                  [-1.0000,  0.5000]], dtype=ht.float32, device=cpu:0, split=None)
        >>> v
        DNDarray([-1,  2], dtype=ht.int32, device=cpu:0, split=None)

    `divide_(t1: DNDarray, t2: Union[DNDarray, float]) ‑> heat.core.dndarray.DNDarray`
    :   Element-wise in-place true division of values of two operands.
        Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise divides its
        element(s) by the element(s) of the second operand (scalar or
        :class:`~heat.core.dndarray.DNDarray`) in-place, i.e. the element(s) of `t1` are overwritten by
        the results of element-wise division of `t1` and `t2`.
        Can be called as a DNDarray method or with the symbol `/=`. `divide_` is an alias for `div_`.

        Parameters
        ----------
        t1: DNDarray
            The first operand whose values are divided.
        t2: DNDarray or scalar
            The second operand by whose values is divided.

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Example
        ---------
        >>> import heat as ht
        >>> T1 = ht.float32([[1, 2], [3, 4]])
        >>> T2 = ht.float32([[2, 2], [2, 2]])
        >>> T1 /= T2
        >>> T1
        DNDarray([[0.5000, 1.0000],
                  [1.5000, 2.0000]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[2., 2.],
                  [2., 2.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> s = 2.0
        >>> T2.div_(s)
        DNDarray([[1., 1.],
                  [1., 1.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[1., 1.],
                  [1., 1.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> s
        2.0
        >>> v = ht.int32([-1, 2])
        >>> T2.divide_(v)
        DNDarray([[-1.0000,  0.5000],
                  [-1.0000,  0.5000]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[-1.0000,  0.5000],
                  [-1.0000,  0.5000]], dtype=ht.float32, device=cpu:0, split=None)
        >>> v
        DNDarray([-1,  2], dtype=ht.int32, device=cpu:0, split=None)

    `exp(self, out=None)`
    :   Calculate the exponential of all elements in the input array.
        Result is a :py:class:`~heat.core.dndarray.DNDarray` of the same shape as ``x``.

        Parameters
        ----------
        x : DNDarray
            The array for which to compute the exponential.
        out : DNDarray, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to :keyword:`None`, a fresh array is allocated.

        Examples
        --------
        >>> ht.exp(ht.arange(5))
        DNDarray([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981], dtype=ht.float32, device=cpu:0, split=None)

    `exp2(self, out=None)`
    :   Calculate the exponential of two of all elements in the input array (:math:`2^x`).
        Result is a :py:class:`~heat.core.dndarray.DNDarray` of the same shape as ``x``.

        Parameters
        ----------
        x : DNDarray
            The array for which to compute the exponential of two.
        out : DNDarray, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to :keyword:`None`, a fresh array is allocated.

        Examples
        --------
        >>> ht.exp2(ht.arange(5))
        DNDarray([ 1.,  2.,  4.,  8., 16.], dtype=ht.float32, device=cpu:0, split=None)

    `expand_dims(self, axis)`
    :   Expand the shape of an array.
        Insert a new axis that will appear at the axis position in the expanded array shape.

        Parameters
        ----------
        a : DNDarray
            Input array to be expanded.
        axis : int
            Position in the expanded axes where the new axis is placed.

        Raises
        ------
        ValueError
            If `axis` is not consistent with the available dimensions.

        Examples
        --------
        >>> x = ht.array([1, 2])
        >>> x.shape
        (2,)
        >>> y = ht.expand_dims(x, axis=0)
        >>> y
        array([[1, 2]])
        >>> y.shape
        (1, 2)
        >>> y = ht.expand_dims(x, axis=1)
        >>> y
        array([[1],
               [2]])
        >>> y.shape
        (2, 1)

    `expm1(self, out=None)`
    :   Calculate :math:`exp(x) - 1` for all elements in the array.
        Result is a :py:class:`~heat.core.dndarray.DNDarray` of the same shape as ``x``.

        Parameters
        ----------
        x : DNDarray
            The array for which to compute the exponential.
        out : DNDarray, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to :keyword:`None`, a fresh array is allocated.

        Examples
        --------
        >>> ht.expm1(ht.arange(5)) + 1.0
        DNDarray([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981], dtype=ht.float64, device=cpu:0, split=None)

    `fabs(self, out=None)`
    :   Calculate the absolute value element-wise and return floating-point class:`~heat.core.dndarray.DNDarray`.
        This function exists besides ``abs==absolute`` since it will be needed in case complex numbers will be introduced
        in the future.

        Parameters
        ----------
        x : DNDarray
            The array for which the compute the absolute value.
        out : DNDarray, optional
            A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
            If not provided or ``None``, a freshly-allocated array is returned.

    `fill_diagonal(self, value: float) ‑> heat.core.dndarray.DNDarray`
    :   Fill the main diagonal of a 2D :class:`DNDarray`.
        This function modifies the input tensor in-place, and returns the input array.

        Parameters
        ----------
        value : float
            The value to be placed in the ``DNDarrays`` main diagonal

    `flatten(self)`
    :   Flattens an array into one dimension.

        Parameters
        ----------
        a : DNDarray
            Array to collapse

        Warning
        ----------
        If `a.split>0`, the array must be redistributed along the first axis (see :func:`resplit`).


        See Also
        --------
        :func:`ravel`

        Examples
        --------
        >>> a = ht.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        >>> ht.flatten(a)
        DNDarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=ht.int64, device=cpu:0, split=None)

    `floor(self, out=None)`
    :   Return the floor of the input, element-wise.
        The floor of the scalar ``x`` is the largest integer i, such that ``i<=x``.
        It is often denoted as :math:`\lfloor x \rfloor`.

        Parameters
        ----------
        x : DNDarray
            The array for which to compute the floored values.
        out : DNDarray, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to ``None``, a fresh :class:`~heat.core.dndarray.DNDarray` is allocated.

        Examples
        --------
        >>> import heat as ht
        >>> ht.floor(ht.arange(-2.0, 2.0, 0.4))
        DNDarray([-2., -2., -2., -1., -1.,  0.,  0.,  0.,  1.,  1.], dtype=ht.float32, device=cpu:0, split=None)

    `floor_divide_(t1: DNDarray, t2: Union[DNDarray, float]) ‑> heat.core.dndarray.DNDarray`
    :   Element-wise in-place floor division of values of two operands.
        Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise divides its
        element(s) by the element(s) of the second operand (scalar or
        :class:`~heat.core.dndarray.DNDarray`) in-place, then rounds down the result to the next
        integer, i.e. the element(s) of `t1` are overwritten by the results of element-wise floor
        division of `t1` and `t2`.
        Can be called as a DNDarray method or with the symbol `//=`. `floor_divide_` is an alias for
        `floordiv_`.

        Parameters
        ----------
        t1: DNDarray
            The first operand whose values are divided
        t2: DNDarray or scalar
            The second operand by whose values is divided

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.float32([[1.7, 2.0], [1.9, 4.2]])
        >>> s = 1
        >>> T1 //= s
        >>> T1
        DNDarray([[1., 2.],
                  [1., 4.]], dtype=ht.float64, device=cpu:0, split=None)
        >>> s
        1
        >>> T2 = ht.float32([[1.5, 2.5], [1.0, 1.3]])
        >>> T1.floordiv_(T2)
        DNDarray([[0., 0.],
                  [1., 3.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T1
        DNDarray([[0., 0.],
                  [1., 3.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[1.5000, 2.5000],
                  [1.0000, 1.3000]], dtype=ht.float32, device=cpu:0, split=None)
        >>> v = ht.int32([-1, 2])
        >>> T1.floor_divide_(v)
        DNDarray([[-0.,  0.],
                  [-1.,  1.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T1
        DNDarray([[-0.,  0.],
                  [-1.,  1.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> v
        DNDarray([-1,  2], dtype=ht.int32, device=cpu:0, split=None)

    `floordiv_(t1: DNDarray, t2: Union[DNDarray, float]) ‑> heat.core.dndarray.DNDarray`
    :   Element-wise in-place floor division of values of two operands.
        Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise divides its
        element(s) by the element(s) of the second operand (scalar or
        :class:`~heat.core.dndarray.DNDarray`) in-place, then rounds down the result to the next
        integer, i.e. the element(s) of `t1` are overwritten by the results of element-wise floor
        division of `t1` and `t2`.
        Can be called as a DNDarray method or with the symbol `//=`. `floor_divide_` is an alias for
        `floordiv_`.

        Parameters
        ----------
        t1: DNDarray
            The first operand whose values are divided
        t2: DNDarray or scalar
            The second operand by whose values is divided

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.float32([[1.7, 2.0], [1.9, 4.2]])
        >>> s = 1
        >>> T1 //= s
        >>> T1
        DNDarray([[1., 2.],
                  [1., 4.]], dtype=ht.float64, device=cpu:0, split=None)
        >>> s
        1
        >>> T2 = ht.float32([[1.5, 2.5], [1.0, 1.3]])
        >>> T1.floordiv_(T2)
        DNDarray([[0., 0.],
                  [1., 3.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T1
        DNDarray([[0., 0.],
                  [1., 3.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[1.5000, 2.5000],
                  [1.0000, 1.3000]], dtype=ht.float32, device=cpu:0, split=None)
        >>> v = ht.int32([-1, 2])
        >>> T1.floor_divide_(v)
        DNDarray([[-0.,  0.],
                  [-1.,  1.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T1
        DNDarray([[-0.,  0.],
                  [-1.,  1.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> v
        DNDarray([-1,  2], dtype=ht.int32, device=cpu:0, split=None)

    `fmod_(t1: DNDarray, t2: Union[DNDarray, float]) ‑> heat.core.dndarray.DNDarray`
    :   In-place computation of element-wise division remainder of values of operand `t1` by values of
        operand `t2` (i.e. C Library function fmod). The result has the same sign as the dividend `t1`.
        Can only be called as a DNDarray method.

        Parameters
        ----------
        t1: DNDarray
            The first operand whose values are divided
        t2: DNDarray or scalar
            The second operand by whose values is divided (may be floats)

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.array(2)
        >>> T1.fmod_(T1)
        >>> T1
        DNDarray(0, dtype=ht.int64, device=cpu:0, split=None)
        >>> T2 = ht.float32([[1, 2], [3, 4]])
        >>> T3 = ht.int32([[2, 2], [2, 2]])
        >>> T2.fmod_(T3)
        DNDarray([[1., 0.],
                  [1., 0.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[1., 0.],
                  [1., 0.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T3
        DNDarray([[2, 2],
                  [2, 2]], dtype=ht.int32, device=cpu:0, split=None)
        >>> s = -3
        >>> T3.fmod_(s)
        DNDarray([[2, 2],
                  [2, 2]], dtype=ht.int32, device=cpu:0, split=None)
        >>> T3
        DNDarray([[2, 2],
                  [2, 2]], dtype=ht.int32, device=cpu:0, split=None)
        >>> s
        -3

    `gcd_(t1: DNDarray, t2: DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Returns the greatest common divisor of |t1| and |t2| element-wise and in-place.
        Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise computes the
        greatest common divisor with the corresponding element(s) of the second operand (scalar or
        :class:`~heat.core.dndarray.DNDarray`) in-place, i.e. the element(s) of `t1` are overwritten by
        the results of element-wise gcd of `t1` and `t2`.
        Can only be called as a DNDarray method.

        Parameters
        ----------
        t1: DNDarray
             The first input array, must be of integer type
        t2: DNDarray
             The second input array, must be of integer type

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.int(ht.ones(3)) * 9
        >>> T2 = ht.arange(3) + 1
        >>> T1.gcd_(T2)
        DNDarray([1, 1, 3], dtype=ht.int32, device=cpu:0, split=None)
        >>> T1
        DNDarray([1, 1, 3], dtype=ht.int32, device=cpu:0, split=None)
        >>> T2
        DNDarray([1, 2, 3], dtype=ht.int32, device=cpu:0, split=None)
        >>> s = 2
        >>> T2.gcd_(2)
        DNDarray([1, 2, 1], dtype=ht.int32, device=cpu:0, split=None)
        >>> T2
        DNDarray([1, 2, 1], dtype=ht.int32, device=cpu:0, split=None)
        >>> s
        2

    `get_halo(self, halo_size: int, prev: bool = True, next: bool = True) ‑> torch.Tensor`
    :   Fetch halos of size ``halo_size`` from neighboring ranks and save them in ``self.halo_next/self.halo_prev``.

        Parameters
        ----------
        halo_size : int
            Size of the halo.
        prev : bool, optional
            If True, fetch the halo from the previous rank. Default: True.
        next : bool, optional
            If True, fetch the halo from the next rank. Default: True.

    `hypot_(t1: DNDarray, t2: DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Given the 'legs' of a right triangle, return its hypotenuse in-place of the first input.
        Equivalent to :math:`sqrt(a^2 + b^2)`, element-wise.
        Can only be called as a DNDarray method.

        Parameters
        ----------
        t1:  DNDarray
             The first input array
        t2:  DNDarray
             the second input array

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.array([1.0, 3.0, 3.0])
        >>> T2 = ht.array(2.0)
        >>> T1.hypot_(T2)
        DNDarray([2.2361, 3.6056, 3.6056], dtype=ht.float32, device=cpu:0, split=None)
        >>> T1
        DNDarray([2.2361, 3.6056, 3.6056], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray(2., dtype=ht.float32, device=cpu:0, split=None)

    `invert_(t: DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Computes the bitwise NOT of the given input :class:`~heat.core.dndarray.DNDarray` in-place. The
        elements of the input array must be of integer or Boolean types. For boolean arrays, it computes
        the logical NOT.
        Can only be called as a DNDarray method. `bitwise_not_` is an alias for `invert_`.

        Parameters
        ----------
        t:  DNDarray
            The input array to invert. Must be of integral or Boolean types

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.array(13, dtype=ht.uint8)
        >>> T1.invert_()
        DNDarray(242, dtype=ht.uint8, device=cpu:0, split=None)
        >>> T1
        DNDarray(242, dtype=ht.uint8, device=cpu:0, split=None)
        >>> T2 = ht.array([-1, -2, 3], dtype=ht.int8)
        >>> T2.invert_()
        DNDarray([ 0,  1, -4], dtype=ht.int8, device=cpu:0, split=None)
        >>> T2
        DNDarray([ 0,  1, -4], dtype=ht.int8, device=cpu:0, split=None)
        >>> T3 = ht.array([[True, True], [False, True]])
        >>> T3.invert_()
        DNDarray([[False, False],
                  [ True, False]], dtype=ht.bool, device=cpu:0, split=None)
        >>> T3
        DNDarray([[False, False],
                  [ True, False]], dtype=ht.bool, device=cpu:0, split=None)

    `is_balanced(self, force_check: bool = False) ‑> bool`
    :   Determine if ``self`` is balanced evenly (or as evenly as possible) across all nodes
        distributed evenly (or as evenly as possible) across all processes.
        This is equivalent to returning ``self.balanced``. If no information
        is available (``self.balanced = None``), the balanced status will be
        assessed via collective communication.

        Parameters
        ----------
        force_check : bool, optional
            If True, the balanced status of the ``DNDarray`` will be assessed via
            collective communication in any case.

    `is_distributed(self) ‑> bool`
    :   Determines whether the data of this ``DNDarray`` is distributed across multiple processes.

    `isclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False)`
    :   Returns a boolean :class:`~heat.core.dndarray.DNDarray`, with elements ``True`` where ``a`` and ``b`` are equal
        within the given tolerance. If both ``x`` and ``y`` are scalars, returns a single boolean value.

        Parameters
        ----------
        x : DNDarray
            Input array to compare.
        y : DNDarray
            Input array to compare.
        rtol : float
            The relative tolerance parameter.
        atol : float
            The absolute tolerance parameter.
        equal_nan : bool
            Whether to compare NaN’s as equal. If ``True``, NaN’s in x will be considered equal to NaN’s in y in the output
            array.

    `item(self)`
    :   Returns the only element of a 1-element :class:`DNDarray`.
        Mirror of the pytorch command by the same name. If size of ``DNDarray`` is >1 element, then a ``ValueError`` is
        raised (by pytorch)

        Examples
        --------
        >>> import heat as ht
        >>> x = ht.zeros((1))
        >>> x.item()
        0.0

    `kurtosis(x, axis=None, unbiased=True, Fischer=True)`
    :   Compute the weighted average along the specified axis.

        If ``returned=True``, return a tuple with the average as the first element and the sum
        of the weights as the second element. ``sum_of_weights`` is of the same type as ``average``.

        Parameters
        ----------
        x : DNDarray
            Array containing data to be averaged.
        axis : None or int or Tuple[int,...], optional
            Axis or axes along which to average ``x``.  The default,
            ``axis=None``, will average over all of the elements of the input array.
            If axis is negative it counts from the last to the first axis.
            #TODO Issue #351: If axis is a tuple of ints, averaging is performed on all of the axes
            specified in the tuple instead of a single axis or all the axes as
            before.
        weights : DNDarray, optional
            An array of weights associated with the values in ``x``. Each value in
            ``x`` contributes to the average according to its associated weight.
            The weights array can either be 1D (in which case its length must be
            the size of ``x`` along the given axis) or of the same shape as ``x``.
            If ``weights=None``, then all data in ``x`` are assumed to have a
            weight equal to one, the result is equivalent to :func:`mean`.
        returned : bool, optional
            If ``True``, the tuple ``(average, sum_of_weights)``
            is returned, otherwise only the average is returned.
            If ``weights=None``, ``sum_of_weights`` is equivalent to the number of
            elements over which the average is taken.

        Raises
        ------
        ZeroDivisionError
            When all weights along axis are zero.
        TypeError
            When the length of 1D weights is not the same as the shape of ``x``
            along axis.

        Examples
        --------
        >>> data = ht.arange(1, 5, dtype=float)
        >>> data
        DNDarray([1., 2., 3., 4.], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.average(data)
        DNDarray(2.5000, dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.average(ht.arange(1, 11, dtype=float), weights=ht.arange(10, 0, -1))
        DNDarray([4.], dtype=ht.float64, device=cpu:0, split=None)
        >>> data = ht.array([[0, 1],
                             [2, 3],
                            [4, 5]], dtype=float, split=1)
        >>> weights = ht.array([1.0 / 4, 3.0 / 4])
        >>> ht.average(data, axis=1, weights=weights)
        DNDarray([0.7500, 2.7500, 4.7500], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.average(data, weights=weights)
        Traceback (most recent call last):
            ...
        TypeError: Axis must be specified when shapes of x and weights differ.

    `lcm_(t1: DNDarray, t2: Union[DNDarray, int]) ‑> heat.core.dndarray.DNDarray`
    :   Returns the lowest common multiple of |t1| and |t2| element-wise and in-place.
        Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise computes the
        lowest common multiple with the corresponding element(s) of the second operand (scalar or
        :class:`~heat.core.dndarray.DNDarray`) in-place, i.e. the element(s) of `t1` are overwritten by
        the results of element-wise gcd of the absolute values of `t1` and `t2`.
        Can only be called as a DNDarray method.

        Parameters
        ----------
        t1:  DNDarray
             The first input array, must be of integer type
        t2:  DNDarray or scalar
             the second input array, must be of integer type

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.array([6, 12, 15])
        >>> T2 = ht.array([3, 4, 5])
        >>> T1.lcm_(T2)
        DNDarray([ 6, 12, 15], dtype=ht.int64, device=cpu:0, split=None)
        >>> T1
        DNDarray([ 6, 12, 15], dtype=ht.int64, device=cpu:0, split=None)
        >>> T2
        DNDarray([3, 4, 5], dtype=ht.int64, device=cpu:0, split=None)
        >>> s = 2
        >>> T2.lcm_(s)
        DNDarray([ 6,  4, 10], dtype=ht.int64, device=cpu:0, split=None)
        >>> T2
        DNDarray([ 6,  4, 10], dtype=ht.int64, device=cpu:0, split=None)
        >>> s
        2

    `left_shift_(t1: DNDarray, t2: Union[DNDarray, float]) ‑> heat.core.dndarray.DNDarray`
    :   In-place version of `left_shift`.
        Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise shifts the bits
        of each element in-place that many positions to the left as the element(s) of the second operand
        (scalar or :class:`~heat.core.dndarray.DNDarray`) indicate, i.e. the element(s) of `t1` are
        overwritten by the results of element-wise bitwise left shift of `t1` for `t2` positions.
        Can be called as a DNDarray method or with the symbol `<<=`. Only works for inputs with integer
        elements.

        Parameters
        ----------
        t1: DNDarray
            Input array
        t2: DNDarray or float
            Integer number of zero bits to add

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.array([1, 2, 3])
        >>> s = 1
        >>> T1.left_shift_(s)
        DNDarray([2, 4, 6], dtype=ht.int64, device=cpu:0, split=None)
        >>> T1
        DNDarray([2, 4, 6], dtype=ht.int64, device=cpu:0, split=None)
        >>> s
        1
        >>> T2 = ht.array([-1, 1, 0])
        >>> T1 <<= T2
        >>> T1
        DNDarray([0, 8, 6], dtype=ht.int64, device=cpu:0, split=None)
        >>> T2
        DNDarray([-1,  1,  0], dtype=ht.int64, device=cpu:0, split=None)

    `log(self, out=None)`
    :   Natural logarithm, element-wise.
        The natural logarithm is the inverse of the exponential function, so that :math:`log(exp(x)) = x`. The natural
        logarithm is logarithm in base e. Result is a :py:class:`~heat.core.dndarray.DNDarray` of the same shape as ``x``.
        Negative input elements are returned as :abbr:`NaN (Not a Number)`.

        Parameters
        ----------
        x : DNDarray
            The array for which to compute the logarithm.
        out : DNDarray, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to :keyword:`None`, a fresh array is allocated.

        Examples
        --------
        >>> ht.log(ht.arange(5))
        DNDarray([  -inf, 0.0000, 0.6931, 1.0986, 1.3863], dtype=ht.float32, device=cpu:0, split=None)

    `log10(self, out=None)`
    :   Compute the logarithm to the base 10 (:math:`log_{10}(x)`), element-wise.
        Result is a :py:class:`~heat.core.dndarray.DNDarray` of the same shape as ``x``.
        Negative input elements are returned as :abbr:`NaN (Not a Number)`.

        Parameters
        ----------
        x : DNDarray
            The array for which to compute the logarithm.
        out : DNDarray, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to :keyword:`None`, a fresh array is allocated.

        Examples
        --------
        >>> ht.log10(ht.arange(5))
        DNDarray([  -inf, 0.0000, 0.3010, 0.4771, 0.6021], dtype=ht.float32, device=cpu:0, split=None)

    `log1p(self, out=None)`
    :   Return the natural logarithm of one plus the input array, element-wise.
        Result is a :class:`~heat.core.dndarray.DNDarray` of the same shape as ``x``.
        Negative input elements are returned as :abbr:`NaN (Not a Number)`.

        Parameters
        ----------
        x : DNDarray
            The array for which to compute the logarithm.
        out : DNDarray, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to :keyword:`None`, a fresh array is allocated.

        Examples
        --------
        >>> ht.log1p(ht.arange(5))
        DNDarray([0.0000, 0.6931, 1.0986, 1.3863, 1.6094], dtype=ht.float32, device=cpu:0, split=None)

    `log2(self, out=None)`
    :   Compute the logarithm to the base 2 (:math:`log_2(x)`), element-wise.
        Result is a :py:class:`~heat.core.dndarray.DNDarray` of the same shape as ``x``.
        Negative input elements are returned as :abbr:`NaN (Not a Number)`.

        Parameters
        ----------
        x : DNDarray
            The array for which to compute the logarithm.
        out : DNDarray, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to :keyword:`None`, a fresh array is allocated.

        Examples
        --------
        >>> ht.log2(ht.arange(5))
        DNDarray([  -inf, 0.0000, 1.0000, 1.5850, 2.0000], dtype=ht.float32, device=cpu:0, split=None)

    `max(x, axis=None, out=None, keepdims=None)`
    :   Return the maximum along a given axis.

        Parameters
        ----------
        x : DNDarray
            Input array.
        axis : None or int or Tuple[int,...], optional
            Axis or axes along which to operate. By default, flattened input is used.
            If this is a tuple of ints, the maximum is selected over multiple axes,
            instead of a single axis or all the axes as before.
        out : DNDarray, optional
            Tuple of two output arrays ``(max, max_indices)``. Must be of the same shape and buffer length as the expected
            output. The minimum value of an output element. Must be present to allow computation on empty slice.
        keepdims : bool, optional
            If this is set to ``True``, the axes which are reduced are left in the result as dimensions with size one.
            With this option, the result will broadcast correctly against the original array.

        Examples
        --------
        >>> a = ht.float32([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])
        >>> ht.max(a)
        DNDarray([12.], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.max(a, axis=0)
        DNDarray([10., 11., 12.], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.max(a, axis=1)
        DNDarray([ 3.,  6.,  9., 12.], dtype=ht.float32, device=cpu:0, split=None)

    `mean(x, axis=None)`
    :   Calculates and returns the mean of a ``DNDarray``.
        If an axis is given, the mean will be taken in that direction.

        Parameters
        ----------
        x : DNDarray
            Values for which the mean is calculated for.
            The dtype of ``x`` must be a float
        axis : None or int or iterable
            Axis which the mean is taken in. Default ``None`` calculates mean of all data items.

        Notes
        -----
        Split semantics when axis is an integer:

        - if ``axis==x.split``, then ``mean(x).split=None``

        - if ``axis>split``, then ``mean(x).split=x.split``

        - if ``axis<split``, then ``mean(x).split=x.split-1``

        Examples
        --------
        >>> a = ht.random.randn(1, 3)
        >>> a
        DNDarray([[-0.1164,  1.0446, -0.4093]], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.mean(a)
        DNDarray(0.1730, dtype=ht.float32, device=cpu:0, split=None)
        >>> a = ht.random.randn(4, 4)
        >>> a
        DNDarray([[-1.0585,  0.7541, -1.1011,  0.5009],
                  [-1.3575,  0.3344,  0.4506,  0.7379],
                  [-0.4337, -0.6516, -1.3690, -0.8772],
                  [ 0.6929, -1.0989, -0.9961,  0.3547]], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.mean(a, 1)
        DNDarray([-0.2262,  0.0413, -0.8328, -0.2619], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.mean(a, 0)
        DNDarray([-0.5392, -0.1655, -0.7539,  0.1791], dtype=ht.float32, device=cpu:0, split=None)
        >>> a = ht.random.randn(4, 4)
        >>> a
        DNDarray([[-0.1441,  0.5016,  0.8907,  0.6318],
                  [-1.1690, -1.2657,  1.4840, -0.1014],
                  [ 0.4133,  1.4168,  1.3499,  1.0340],
                  [-0.9236, -0.7535, -0.2466, -0.9703]], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.mean(a, (0, 1))
        DNDarray(0.1342, dtype=ht.float32, device=cpu:0, split=None)

    `median(x, axis=None, keepdims=False, sketched=False, sketch_size=1.0)`
    :   Compute the median of the data along the specified axis.
        Returns the median of the ``DNDarray`` elements.
        Per default, the "true" median of the entire data set is computed; however, the argument
        `sketched` allows to switch to a faster but less accurate version that computes
        the median only on behalf of a random subset of the data set ("sketch").

        Parameters
        ----------
        x : DNDarray
            Input tensor
        axis : int, or None, optional
            Axis along which the median is computed. Default is ``None``, i.e.,
            the median is computed along a flattened version of the ``DNDarray``.

        keepdims : bool, optional
            If True, the axes which are reduced are left in the result as dimensions with size one.
            With this option, the result can broadcast correctly against the original array ``a``.

        sketched : bool, optional
            If True, the median is computed on a random subset of the data set ("sketch").
            This is faster but less accurate.  Default is False. The size of the sketch is controlled by the argument `sketch_size`.
        sketch_size : float, optional
            The size of the sketch as a fraction of the data set size. Default is `1./n_proc`  where `n_proc` is the number of MPI processes, e.g. `n_proc =  MPI.COMM_WORLD.size`. Must be in the range (0, 1).
            Ignored for sketched = False.

    `min(self, axis=None, out=None, keepdims=None)`
    :   Return the minimum along a given axis.

        Parameters
        ----------
        x : DNDarray
            Input array.
        axis : None or int or Tuple[int,...]
            Axis or axes along which to operate. By default, flattened input is used.
            If this is a tuple of ints, the minimum is selected over multiple axes,
            instead of a single axis or all the axes as before.
        out : Tuple[DNDarray,DNDarray], optional
            Tuple of two output arrays ``(min, min_indices)``. Must be of the same shape and buffer length as the expected
            output. The maximum value of an output element. Must be present to allow computation on empty slice.
        keepdims : bool, optional
            If this is set to ``True``, the axes which are reduced are left in the result as dimensions with size one.
            With this option, the result will broadcast correctly against the original array.


        Examples
        --------
        >>> a = ht.float32([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12]
            ])
        >>> ht.min(a)
        DNDarray([1.], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.min(a, axis=0)
        DNDarray([1., 2., 3.], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.min(a, axis=1)
        DNDarray([ 1.,  4.,  7., 10.], dtype=ht.float32, device=cpu:0, split=None)

    `mod_(t1: DNDarray, t2: Union[DNDarray, float]) ‑> heat.core.dndarray.DNDarray`
    :   Element-wise in-place division remainder of values of two operands. The result has the same sign
        as the divisor.
        Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise computes the
        modulo regarding the element(s) of the second operand (scalar or
        :class:`~heat.core.dndarray.DNDarray`) in-place, i.e. the element(s) of `t1` are overwritten by
        the results of element-wise `t1` modulo `t2`.
        Can be called as a DNDarray method or with the symbol `%=`. `mod_` is an alias for `remainder_`.

        Parameters
        ----------
        t1: DNDarray
            The first operand whose values are divided
        t2: DNDarray or scalar
            The second operand by whose values is divided

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.array(2)
        >>> T1 %= T1
        >>> T1
        DNDarray(0, dtype=ht.int64, device=cpu:0, split=None)
        >>> T2 = ht.float32([[1, 2], [3, 4]])
        >>> T3 = ht.int32([[2, 2], [2, 2]])
        >>> T2.mod_(T3)
        DNDarray([[1., 0.],
                  [1., 0.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[1., 0.],
                  [1., 0.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T3
        DNDarray([[2, 2],
                  [2, 2]], dtype=ht.int32, device=cpu:0, split=None)
        >>> s = -3
        >>> T3.remainder_(s)
        DNDarray([[-1, -1],
                  [-1, -1]], dtype=ht.int32, device=cpu:0, split=None)
        >>> T3
        DNDarray([[-1, -1],
                  [-1, -1]], dtype=ht.int32, device=cpu:0, split=None)
        >>> s
        -3

    `modf(self, out=None)`
    :   Return the fractional and integral parts of a :class:`~heat.core.dndarray.DNDarray`, element-wise.
        The fractional and integral parts are negative if the given number is negative.

        Parameters
        ----------
        x : DNDarray
            Input array
        out : Tuple[DNDarray, DNDarray], optional
            A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
            If not provided or ``None``, a freshly-allocated array is returned.

        Raises
        ------
        TypeError
            if ``x`` is not a :class:`~heat.core.dndarray.DNDarray`
        TypeError
            if ``out`` is not None or a tuple of :class:`~heat.core.dndarray.DNDarray`
        ValueError
            if ``out`` is a tuple of length unqual 2

        Examples
        --------
        >>> import heat as ht
        >>> ht.modf(ht.arange(-2.0, 2.0, 0.4))
        (DNDarray([ 0.0000, -0.6000, -0.2000, -0.8000, -0.4000,  0.0000,  0.4000,  0.8000,  0.2000,  0.6000], dtype=ht.float32, device=cpu:0, split=None), DNDarray([-2., -1., -1., -0., -0.,  0.,  0.,  0.,  1.,  1.], dtype=ht.float32, device=cpu:0, split=None))

    `mul_(t1: DNDarray, t2: Union[DNDarray, float]) ‑> heat.core.dndarray.DNDarray`
    :   Element-wise in-place multiplication of values of two operands.
        Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise multiplies the
        element(s) of the second operand (scalar or :class:`~heat.core.dndarray.DNDarray`) in-place,
        i.e. the element(s) of `t1` are overwritten by the results of element-wise multiplication of
        `t1` and `t2`.
        Can be called as a DNDarray method or with the symbol `*=`. `multiply_` is an alias for `mul_`.

        Parameters
        ----------
        t1: DNDarray
            The first operand involved in the multiplication.
        t2: DNDarray or scalar
            The second operand involved in the multiplication.

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.float32([[1, 2], [3, 4]])
        >>> T2 = ht.float32([[2, 2], [2, 2]])
        >>> T1 *= T2
        >>> T1
        DNDarray([[2., 4.],
                  [6., 8.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[2., 2.],
                  [2., 2.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> s = 2.0
        >>> T2.mul_(s)
        DNDarray([[4., 4.],
                  [4., 4.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[4., 4.],
                  [4., 4.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> s
        2.0
        >>> v = ht.int32([-1, 2])
        >>> T2.multiply_(v)
        DNDarray([[-4.,  8.],
                  [-4.,  8.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[-4.,  8.],
                  [-4.,  8.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> v
        DNDarray([-1,  2], dtype=ht.int32, device=cpu:0, split=None)

    `multiply_(t1: DNDarray, t2: Union[DNDarray, float]) ‑> heat.core.dndarray.DNDarray`
    :   Element-wise in-place multiplication of values of two operands.
        Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise multiplies the
        element(s) of the second operand (scalar or :class:`~heat.core.dndarray.DNDarray`) in-place,
        i.e. the element(s) of `t1` are overwritten by the results of element-wise multiplication of
        `t1` and `t2`.
        Can be called as a DNDarray method or with the symbol `*=`. `multiply_` is an alias for `mul_`.

        Parameters
        ----------
        t1: DNDarray
            The first operand involved in the multiplication.
        t2: DNDarray or scalar
            The second operand involved in the multiplication.

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.float32([[1, 2], [3, 4]])
        >>> T2 = ht.float32([[2, 2], [2, 2]])
        >>> T1 *= T2
        >>> T1
        DNDarray([[2., 4.],
                  [6., 8.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[2., 2.],
                  [2., 2.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> s = 2.0
        >>> T2.mul_(s)
        DNDarray([[4., 4.],
                  [4., 4.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[4., 4.],
                  [4., 4.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> s
        2.0
        >>> v = ht.int32([-1, 2])
        >>> T2.multiply_(v)
        DNDarray([[-4.,  8.],
                  [-4.,  8.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[-4.,  8.],
                  [-4.,  8.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> v
        DNDarray([-1,  2], dtype=ht.int32, device=cpu:0, split=None)

    `nan_to_num_(t: DNDarray, nan: float = 0.0, posinf: float = None, neginf: float = None) ‑> heat.core.dndarray.DNDarray`
    :   Replaces NaNs, positive infinity values, and negative infinity values in the input 't' in-place
        with the values specified by nan, posinf, and neginf, respectively. By default, NaNs are
        replaced with zero, positive infinity is replaced with the greatest finite value representable
        by input's dtype, and negative infinity is replaced with the least finite value representable by
        input's dtype.
        Can only be called as a DNDarray method.

        Parameters
        ----------
        t:      DNDarray
                Input array.
        nan:    float, optional
                Value to be used to replace NaNs. Default value is 0.0.
        posinf: float, optional
                Value to replace positive infinity values with. If None, positive infinity values are
                replaced with the greatest finite value of the input's dtype. Default value is None.
        neginf: float, optional
                Value to replace negative infinity values with. If None, negative infinity values are
                replaced with the greatest negative finite value of the input's dtype. Default value is
                None.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.array([float("nan"), float("inf"), -float("inf")])
        >>> T1.nan_to_num_()
        DNDarray([ 0.0000e+00,  3.4028e+38, -3.4028e+38], dtype=ht.float32, device=cpu:0, split=None)
        >>> T1
        DNDarray([ 0.0000e+00,  3.4028e+38, -3.4028e+38], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2 = ht.array([1, 2, 3, ht.nan, ht.inf, -ht.inf])
        >>> T2.nan_to_num_(nan=0, posinf=1, neginf=-1)
        DNDarray([ 1.,  2.,  3.,  0.,  1., -1.], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([ 1.,  2.,  3.,  0.,  1., -1.], dtype=ht.float32, device=cpu:0, split=None)

    `neg_(t: DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Element-wise in-place negation of `t`.
        Can only be called as a DNDarray method. `negative_` is an alias for `neg_`.

        Parameter
        ----------
        t:  DNDarray
            The input array

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.array([-1, 1])
        >>> T1.neg_()
        DNDarray([ 1, -1], dtype=ht.int64, device=cpu:0, split=None)
        >>> T1
        DNDarray([ 1, -1], dtype=ht.int64, device=cpu:0, split=None)
        >>> T2 = ht.array([[-1.0, 2.5], [4.0, 0.0]])
        >>> T2.neg_()
        DNDarray([[ 1.0000, -2.5000],
                  [-4.0000, -0.0000]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[ 1.0000, -2.5000],
                  [-4.0000, -0.0000]], dtype=ht.float32, device=cpu:0, split=None)

    `negative_(t: DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Element-wise in-place negation of `t`.
        Can only be called as a DNDarray method. `negative_` is an alias for `neg_`.

        Parameter
        ----------
        t:  DNDarray
            The input array

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.array([-1, 1])
        >>> T1.neg_()
        DNDarray([ 1, -1], dtype=ht.int64, device=cpu:0, split=None)
        >>> T1
        DNDarray([ 1, -1], dtype=ht.int64, device=cpu:0, split=None)
        >>> T2 = ht.array([[-1.0, 2.5], [4.0, 0.0]])
        >>> T2.neg_()
        DNDarray([[ 1.0000, -2.5000],
                  [-4.0000, -0.0000]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[ 1.0000, -2.5000],
                  [-4.0000, -0.0000]], dtype=ht.float32, device=cpu:0, split=None)

    `nonzero(self)`
    :   Return a :class:`~heat.core.dndarray.DNDarray` containing the indices of the elements that are non-zero.. (using ``torch.nonzero``)
        If ``x`` is split then the result is split in the 0th dimension. However, this :class:`~heat.core.dndarray.DNDarray`
        can be UNBALANCED as it contains the indices of the non-zero elements on each node.
        Returns an array with one entry for each dimension of ``x``, containing the indices of the non-zero elements in that dimension.
        The values in ``x`` are always tested and returned in row-major, C-style order.
        The corresponding non-zero values can be obtained with: ``x[nonzero(x)]``.

        Parameters
        ----------
        x: DNDarray
            Input array

        Examples
        --------
        >>> import heat as ht
        >>> x = ht.array([[3, 0, 0], [0, 4, 1], [0, 6, 0]], split=0)
        >>> ht.nonzero(x)
        DNDarray([[0, 0],
                  [1, 1],
                  [1, 2],
                  [2, 1]], dtype=ht.int64, device=cpu:0, split=0)
        >>> y = ht.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], split=0)
        >>> y > 3
        DNDarray([[False, False, False],
                  [ True,  True,  True],
                  [ True,  True,  True]], dtype=ht.bool, device=cpu:0, split=0)
        >>> ht.nonzero(y > 3)
        DNDarray([[1, 0],
                  [1, 1],
                  [1, 2],
                  [2, 0],
                  [2, 1],
                  [2, 2]], dtype=ht.int64, device=cpu:0, split=0)
        >>> y[ht.nonzero(y > 3)]
        DNDarray([4, 5, 6, 7, 8, 9], dtype=ht.int64, device=cpu:0, split=0)

    `norm(self)`
    :   Return the vector or matrix norm of an array.

        Parameters
        ----------
        x : DNDarray
            Input vector
        axis : int, tuple, optional
            Axes along which to compute the norm. If an integer, vector norm is used. If a 2-tuple, matrix norm is used.
            If `None`, it is inferred from the dimension of the array. Default: `None`
        keepdims : bool, optional
            Retains the reduced dimension when `True`. Default: `False`
        ord : int, float, inf, -inf, 'fro', 'nuc'
            The norm order to compute. See Notes

        See Also
        --------
        vector_norm
            Computes the vector norm of an array.
        matrix_norm
            Computes the matrix norm of an array.

        Notes
        -----
        The following norms are supported:

        =====  ============================  ==========================
        ord    norm for matrices             norm for vectors
        =====  ============================  ==========================
        None   Frobenius norm                L2-norm (Euclidean)
        'fro'  Frobenius norm                --
        'nuc'  nuclear norm                  --
        inf    max(sum(abs(x), axis=1))      max(abs(x))
        -inf   min(sum(abs(x), axis=1))      min(abs(x))
        0      --                            sum(x != 0)
        1      max(sum(abs(x), axis=0))      L1-norm (Manhattan)
        -1     min(sum(abs(x), axis=0))      1./sum(1./abs(a))
        2      --                            L2-norm (Euclidean)
        -2     --                            1./sqrt(sum(1./abs(a)**2))
        other  --                            sum(abs(x)**ord)**(1./ord)
        =====  ============================  ==========================

        The following matrix norms are currently **not** supported:

        =====  ============================
        ord    norm for matrices
        =====  ============================
        2      largest singular value
        -2     smallest singular value
        =====  ============================

        Raises
        ------
        ValueError
            If 'axis' has more than 2 elements

        Examples
        --------
        >>> from heat import linalg as LA
        >>> a = ht.arange(9, dtype=ht.float) - 4
        >>> a
        DNDarray([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.], dtype=ht.float32, device=cpu:0, split=None)
        >>> b = a.reshape((3, 3))
        >>> b
        DNDarray([[-4., -3., -2.],
              [-1.,  0.,  1.],
              [ 2.,  3.,  4.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> LA.norm(a)
        DNDarray(7.7460, dtype=ht.float32, device=cpu:0, split=None)
        >>> LA.norm(b)
        DNDarray(7.7460, dtype=ht.float32, device=cpu:0, split=None)
        >>> LA.norm(b, ord="fro")
        DNDarray(7.7460, dtype=ht.float32, device=cpu:0, split=None)
        >>> LA.norm(a, float("inf"))
        DNDarray([4.], dtype=ht.float32, device=cpu:0, split=None)
        >>> LA.norm(b, ht.inf)
        DNDarray([9.], dtype=ht.float32, device=cpu:0, split=None)
        >>> LA.norm(a, -ht.inf))
        DNDarray([0.], dtype=ht.float32, device=cpu:0, split=None)
        >>> LA.norm(b, -ht.inf)
        DNDarray([2.], dtype=ht.float32, device=cpu:0, split=None)
        >>> LA.norm(a, 1)
        DNDarray([20.], dtype=ht.float32, device=cpu:0, split=None)
        >>> LA.norm(b, 1)
        DNDarray([7.], dtype=ht.float32, device=cpu:0, split=None)
        >>> LA.norm(a, -1)
        DNDarray([0.], dtype=ht.float32, device=cpu:0, split=None)
        >>> LA.norm(b, -1)
        DNDarray([6.], dtype=ht.float32, device=cpu:0, split=None)
        >>> LA.norm(a, 2)
        DNDarray(7.7460, dtype=ht.float32, device=cpu:0, split=None)
        >>> LA.norm(a, -2)
        DNDarray([0.], dtype=ht.float32, device=cpu:0, split=None)
        >>> LA.norm(a, 3)
        DNDarray([5.8480], dtype=ht.float32, device=cpu:0, split=None)
        >>> LA.norm(a, -3)
        DNDarray([0.], dtype=ht.float32, device=cpu:0, split=None)
        c = ht.array([[ 1, 2, 3],
                      [-1, 1, 4]])
        >>> LA.norm(c, axis=0)
        DNDarray([1.4142, 2.2361, 5.0000], dtype=ht.float64, device=cpu:0, split=None)
        >>> LA.norm(c, axis=1)
        DNDarray([3.7417, 4.2426], dtype=ht.float64, device=cpu:0, split=None)
        >>> LA.norm(c, axis=1, ord=1)
        DNDarray([6., 6.], dtype=ht.float64, device=cpu:0, split=None)
        >>> m = ht.arange(8).reshape(2, 2, 2)
        >>> LA.norm(m, axis=(1, 2))
        DNDarray([ 3.7417, 11.2250], dtype=ht.float32, device=cpu:0, split=None)
        >>> LA.norm(m[0, :, :]), LA.norm(m[1, :, :])
        (DNDarray(3.7417, dtype=ht.float32, device=cpu:0, split=None), DNDarray(11.2250, dtype=ht.float32, device=cpu:0, split=None))

    `numpy(self) ‑> <built-in function array>`
    :   Returns a copy of the :class:`DNDarray` as numpy ndarray. If the ``DNDarray`` resides on the GPU, the underlying data will be copied to the CPU first.

        If the ``DNDarray`` is distributed, an MPI Allgather operation will be performed before converting to np.ndarray, i.e. each MPI process will end up holding a copy of the entire array in memory.  Make sure process memory is sufficient!

        Examples
        --------
        >>> import heat as ht
        T1 = ht.random.randn((10,8))
        T1.numpy()

    `pow_(t1: DNDarray, t2: Union[DNDarray, float]) ‑> heat.core.dndarray.DNDarray`
    :   Element-wise in-place exponentation.
        Takes the element(s) of the first operand (:class:`~heat.core.dndarray.DNDarray`) element-wise
        to the power of the corresponding element(s) of the second operand (scalar or
        :class:`~heat.core.dndarray.DNDarray`) in-place, i.e. the element(s) of `t1` are overwritten by
        the results of element-wise exponentiation of `t1` and `t2`.
        Can be called as a DNDarray method or with the symbol `**=`. `power_` is an alias for `pow_`.

        Parameters
        ----------
        t1: DNDarray
            The first operand whose values represent the base
        t2: DNDarray or scalar
            The second operand whose values represent the exponent

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.float32([[1, 2], [3, 4]])
        >>> T2 = ht.float32([[3, 3], [2, 2]])
        >>> T1 **= T2
        >>> T1
        DNDarray([[ 1.,  8.],
                  [ 9., 16.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[3., 3.],
                  [2., 2.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> s = -1.0
        >>> T2.pow_(s)
        DNDarray([[0.3333, 0.3333],
                  [0.5000, 0.5000]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[0.3333, 0.3333],
                  [0.5000, 0.5000]], dtype=ht.float32, device=cpu:0, split=None)
        >>> s
        -1.0
        >>> v = ht.int32([-3, 2])
        >>> T2.power_(v)
        DNDarray([[27.0000,  0.1111],
                  [ 8.0000,  0.2500]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[27.0000,  0.1111],
                  [ 8.0000,  0.2500]], dtype=ht.float32, device=cpu:0, split=None)
        >>> v
        DNDarray([-3,  2], dtype=ht.int32, device=cpu:0, split=None)

    `power_(t1: DNDarray, t2: Union[DNDarray, float]) ‑> heat.core.dndarray.DNDarray`
    :   Element-wise in-place exponentation.
        Takes the element(s) of the first operand (:class:`~heat.core.dndarray.DNDarray`) element-wise
        to the power of the corresponding element(s) of the second operand (scalar or
        :class:`~heat.core.dndarray.DNDarray`) in-place, i.e. the element(s) of `t1` are overwritten by
        the results of element-wise exponentiation of `t1` and `t2`.
        Can be called as a DNDarray method or with the symbol `**=`. `power_` is an alias for `pow_`.

        Parameters
        ----------
        t1: DNDarray
            The first operand whose values represent the base
        t2: DNDarray or scalar
            The second operand whose values represent the exponent

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.float32([[1, 2], [3, 4]])
        >>> T2 = ht.float32([[3, 3], [2, 2]])
        >>> T1 **= T2
        >>> T1
        DNDarray([[ 1.,  8.],
                  [ 9., 16.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[3., 3.],
                  [2., 2.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> s = -1.0
        >>> T2.pow_(s)
        DNDarray([[0.3333, 0.3333],
                  [0.5000, 0.5000]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[0.3333, 0.3333],
                  [0.5000, 0.5000]], dtype=ht.float32, device=cpu:0, split=None)
        >>> s
        -1.0
        >>> v = ht.int32([-3, 2])
        >>> T2.power_(v)
        DNDarray([[27.0000,  0.1111],
                  [ 8.0000,  0.2500]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[27.0000,  0.1111],
                  [ 8.0000,  0.2500]], dtype=ht.float32, device=cpu:0, split=None)
        >>> v
        DNDarray([-3,  2], dtype=ht.int32, device=cpu:0, split=None)

    `prod(self, axis=None, out=None, keepdims=None)`
    :   Return the product of array elements over a given axis in form of a DNDarray shaped as a but
        with the specified axis removed.

        Parameters
        ----------
        a : DNDarray
            Input array.
        axis : None or int or Tuple[int,...], optional
            Axis or axes along which a product is performed. The default, ``axis=None``, will calculate
            the product of all the elements in the input array. If axis is negative it counts from the
            last to the first axis. If axis is a tuple of ints, a product is performed on all of the
            axes specified in the tuple instead of a single axis or all the axes as before.
        out : DNDarray, optional
            Alternative output array in which to place the result. It must have the same shape as the
            expected output, but the datatype of the output values will be cast if necessary.
        keepdims : bool, optional
            If this is set to ``True``, the axes which are reduced are left in the result as dimensions
            with size one. With this option, the result will broadcast correctly against the input
            array.

        Examples
        --------
        >>> ht.prod(ht.array([1.0, 2.0]))
        DNDarray(2., dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.prod(ht.array([
            [1.,2.],
            [3.,4.]]))
        DNDarray(24., dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.prod(ht.array([
            [1.,2.],
            [3.,4.]
        ]), axis=1)
        DNDarray([ 2., 12.], dtype=ht.float32, device=cpu:0, split=None)

    `ravel(self)`
    :   Flattens the ``DNDarray``.

        See Also
        --------
        :func:`~heat.core.manipulations.ravel`

        Examples
        --------
        >>> a = ht.ones((2, 3), split=0)
        >>> b = a.ravel()
        >>> a[0, 0] = 4
        >>> b
        DNDarray([4., 1., 1., 1., 1., 1.], dtype=ht.float32, device=cpu:0, split=0)

    `redistribute(arr, lshape_map=None, target_map=None)`
    :   Redistributes the data of the :class:`DNDarray` *along the split axis* to match the given target map.
        This function does not modify the non-split dimensions of the ``DNDarray``.
        This is an abstraction and extension of the balance function.

        Parameters
        ----------
        arr: DNDarray
            DNDarray to redistribute
        lshape_map : torch.Tensor, optional
            The current lshape of processes.
            Units are ``[rank, lshape]``.
        target_map : torch.Tensor, optional
            The desired distribution across the processes.
            Units are ``[rank, target lshape]``.
            Note: the only important parts of the target map are the values along the split axis,
            values which are not along this axis are there to mimic the shape of the ``lshape_map``.

        Examples
        --------
        >>> st = ht.ones((50, 81, 67), split=2)
        >>> target_map = torch.zeros((st.comm.size, 3), dtype=torch.int64)
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
        >>> ht.redistribute_(st, target_map=target_map)
        >>> print(st.lshape)
        [0/2] (50, 81, 67)
        [1/2] (50, 81, 0)
        [2/2] (50, 81, 0)

    `redistribute_(self, lshape_map: Optional[torch.Tensor] = None, target_map: Optional[torch.Tensor] = None)`
    :   Redistributes the data of the :class:`DNDarray` *along the split axis* to match the given target map.
        This function does not modify the non-split dimensions of the ``DNDarray``.
        This is an abstraction and extension of the balance function.

        Parameters
        ----------
        lshape_map : torch.Tensor, optional
            The current lshape of processes.
            Units are ``[rank, lshape]``.
        target_map : torch.Tensor, optional
            The desired distribution across the processes.
            Units are ``[rank, target lshape]``.
            Note: the only important parts of the target map are the values along the split axis,
            values which are not along this axis are there to mimic the shape of the ``lshape_map``.

        Examples
        --------
        >>> st = ht.ones((50, 81, 67), split=2)
        >>> target_map = torch.zeros((st.comm.size, 3), dtype=torch.int64)
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

    `remainder_(t1: DNDarray, t2: Union[DNDarray, float]) ‑> heat.core.dndarray.DNDarray`
    :   Element-wise in-place division remainder of values of two operands. The result has the same sign
        as the divisor.
        Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise computes the
        modulo regarding the element(s) of the second operand (scalar or
        :class:`~heat.core.dndarray.DNDarray`) in-place, i.e. the element(s) of `t1` are overwritten by
        the results of element-wise `t1` modulo `t2`.
        Can be called as a DNDarray method or with the symbol `%=`. `mod_` is an alias for `remainder_`.

        Parameters
        ----------
        t1: DNDarray
            The first operand whose values are divided
        t2: DNDarray or scalar
            The second operand by whose values is divided

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.array(2)
        >>> T1 %= T1
        >>> T1
        DNDarray(0, dtype=ht.int64, device=cpu:0, split=None)
        >>> T2 = ht.float32([[1, 2], [3, 4]])
        >>> T3 = ht.int32([[2, 2], [2, 2]])
        >>> T2.mod_(T3)
        DNDarray([[1., 0.],
                  [1., 0.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[1., 0.],
                  [1., 0.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T3
        DNDarray([[2, 2],
                  [2, 2]], dtype=ht.int32, device=cpu:0, split=None)
        >>> s = -3
        >>> T3.remainder_(s)
        DNDarray([[-1, -1],
                  [-1, -1]], dtype=ht.int32, device=cpu:0, split=None)
        >>> T3
        DNDarray([[-1, -1],
                  [-1, -1]], dtype=ht.int32, device=cpu:0, split=None)
        >>> s
        -3

    `reshape(self, *shape, **kwargs)`
    :   Returns an array with the same data and number of elements as `a`, but with the specified shape.

        Parameters
        ----------
        a : DNDarray
            The input array
        shape : Union[int, Tuple[int,...]]
            Shape of the new array. Must be compatible with the original shape. If an integer, then the result will be a 1-D array of that length.
            One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
        new_split : int, optional
            The distribution axis of the reshaped array. If `new_split` is not provided, the reshaped array will have:
            -  the same split axis as the input array, if the original dimensionality is unchanged;
            -  split axis 0, if the number of dimensions is modified by reshaping.
        **kwargs
            Extra keyword arguments.

        Raises
        ------
        ValueError
            If the number of elements in the new shape is inconsistent with the input data.

        Notes
        -----
        `reshape()` might require significant communication among processes. Communication is minimized if the input array is distributed along axis 0, i.e. `a.split == 0`.

        See Also
        --------
        :func:`ravel`

        Examples
        --------
        >>> a = ht.zeros((3, 4))
        >>> ht.reshape(a, (4, 3))
        DNDarray([[0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> a = ht.linspace(0, 14, 8, split=0)
        >>> ht.reshape(a, (2, 4))
        (1/2) tensor([[0., 2., 4., 6.]])
        (2/2) tensor([[ 8., 10., 12., 14.]])
        # 3-dim array, distributed along axis 1
        >>> a = ht.random.rand(2, 3, 4, split=1)
        >>> a
        DNDarray([[[0.5525, 0.5434, 0.9477, 0.9503],
               [0.4165, 0.3924, 0.3310, 0.3935],
               [0.1008, 0.1750, 0.9030, 0.8579]],

              [[0.0680, 0.4944, 0.4114, 0.6669],
               [0.6423, 0.2625, 0.5413, 0.2225],
               [0.0197, 0.5079, 0.4739, 0.4387]]], dtype=ht.float32, device=cpu:0, split=1)
        >>> a.reshape(-1, 3)  # reshape to 2-dim array: split axis will be set to 0
        DNDarray([[0.5525, 0.5434, 0.9477],
                [0.9503, 0.4165, 0.3924],
                [0.3310, 0.3935, 0.1008],
                [0.1750, 0.9030, 0.8579],
                [0.0680, 0.4944, 0.4114],
                [0.6669, 0.6423, 0.2625],
                [0.5413, 0.2225, 0.0197],
                [0.5079, 0.4739, 0.4387]], dtype=ht.float32, device=cpu:0, split=0)
        >>> a.reshape(2, 3, 2, 2, new_split=1)  # reshape to 4-dim array, specify distribution axis
        DNDarray([[[[0.5525, 0.5434],
                    [0.9477, 0.9503]],

                   [[0.4165, 0.3924],
                    [0.3310, 0.3935]],

                   [[0.1008, 0.1750],
                    [0.9030, 0.8579]]],


                  [[[0.0680, 0.4944],
                    [0.4114, 0.6669]],

                   [[0.6423, 0.2625],
                    [0.5413, 0.2225]],

                   [[0.0197, 0.5079],
                    [0.4739, 0.4387]]]], dtype=ht.float32, device=cpu:0, split=1)

    `resplit(self, axis=None)`
    :   Out-of-place redistribution of the content of the `DNDarray`. Allows to "unsplit" (i.e. gather) all values from all
        nodes,  as well as to define a new axis along which the array is split without changes to the values.

        Parameters
        ----------
        arr : DNDarray
            The array from which to resplit
        axis : int or None
            The new split axis, `None` denotes gathering, an int will set the new split axis

        Warning
        ----------
        This operation might involve a significant communication overhead. Use it sparingly and preferably for
        small arrays.

        Examples
        --------
        >>> a = ht.zeros(
        ...     (
        ...         4,
        ...         5,
        ...     ),
        ...     split=0,
        ... )
        >>> a.lshape
        (0/2) (2, 5)
        (1/2) (2, 5)
        >>> b = resplit(a, None)
        >>> b.split
        None
        >>> b.lshape
        (0/2) (4, 5)
        (1/2) (4, 5)
        >>> a = ht.zeros(
        ...     (
        ...         4,
        ...         5,
        ...     ),
        ...     split=0,
        ... )
        >>> a.lshape
        (0/2) (2, 5)
        (1/2) (2, 5)
        >>> b = resplit(a, 1)
        >>> b.split
        1
        >>> b.lshape
        (0/2) (4, 3)
        (1/2) (4, 2)

    `resplit_(self, axis: int = None)`
    :   In-place option for resplitting a :class:`DNDarray`.

        Parameters
        ----------
        axis : int
            The new split axis, ``None`` denotes gathering, an int will set the new split axis

        Examples
        --------
        >>> a = ht.zeros(
        ...     (
        ...         4,
        ...         5,
        ...     ),
        ...     split=0,
        ... )
        >>> a.lshape
        (0/2) (2, 5)
        (1/2) (2, 5)
        >>> ht.resplit_(a, None)
        >>> a.split
        None
        >>> a.lshape
        (0/2) (4, 5)
        (1/2) (4, 5)
        >>> a = ht.zeros(
        ...     (
        ...         4,
        ...         5,
        ...     ),
        ...     split=0,
        ... )
        >>> a.lshape
        (0/2) (2, 5)
        (1/2) (2, 5)
        >>> ht.resplit_(a, 1)
        >>> a.split
        1
        >>> a.lshape
        (0/2) (4, 3)
        (1/2) (4, 2)

    `right_shift_(t1: DNDarray, t2: Union[DNDarray, float]) ‑> heat.core.dndarray.DNDarray`
    :   In-place version of `right_shift`.
        Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise shifts the bits
        of each element in-place that many positions to the right as the element(s) of the second
        operand (scalar or :class:`~heat.core.dndarray.DNDarray`) indicate, i.e. the element(s) of `t1`
        are overwritten by the results of element-wise bitwise right shift of `t1` for `t2` positions.
        Can be called as a DNDarray method or with the symbol `>>=`. Only works for inputs with integer
        elements.

        Parameters
        ----------
        t1: DNDarray
            Input array
        t2: DNDarray or float
            Integer number of zero bits to remove

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.array([1, 2, 32])
        >>> s = 1
        >>> T1.right_shift_(s)
        DNDarray([ 0,  1, 16], dtype=ht.int64, device=cpu:0, split=None)
        >>> T1
        DNDarray([0, 1, 1], dtype=ht.int64, device=cpu:0, split=None)
        >>> s
        1
        >>> T2 = ht.array([2, -3, 2])
        >>> T1 >>= T2
        >>> T1
        DNDarray([0, 0, 4], dtype=ht.int64, device=cpu:0, split=None)
        >>> T2
        DNDarray([ 2, -3,  2], dtype=ht.int64, device=cpu:0, split=None)

    `rot90(self, k=1, axis=(0, 1))`
    :   Rotate an array by 90 degrees in the plane specified by `axes`.
        Rotation direction is from the first towards the second axis.

        Parameters
        ----------
        m : DNDarray
            Array of two or more dimensions.
        k : integer
            Number of times the array is rotated by 90 degrees.
        axes: (2,) Sequence[int, int]
            The array is rotated in the plane defined by the axes.
            Axes must be different.

        Raises
        ------
        ValueError
            If `len(axis)!=2`.
        ValueError
            If the axes are the same.
        ValueError
            If axes are out of range.

        Notes
        -----
        - ``rot90(m, k=1, axes=(1,0))`` is the reverse of ``rot90(m, k=1, axes=(0,1))``.

        - ``rot90(m, k=1, axes=(1,0))`` is equivalent to ``rot90(m, k=-1, axes=(0,1))``.

        May change the split axis on distributed tensors.

        Examples
        --------
        >>> m = ht.array([[1, 2], [3, 4]], dtype=ht.int)
        >>> m
        DNDarray([[1, 2],
                  [3, 4]], dtype=ht.int32, device=cpu:0, split=None)
        >>> ht.rot90(m)
        DNDarray([[2, 4],
                  [1, 3]], dtype=ht.int32, device=cpu:0, split=None)
        >>> ht.rot90(m, 2)
        DNDarray([[4, 3],
                  [2, 1]], dtype=ht.int32, device=cpu:0, split=None)
        >>> m = ht.arange(8).reshape((2, 2, 2))
        >>> ht.rot90(m, 1, (1, 2))
        DNDarray([[[1, 3],
                   [0, 2]],

                  [[5, 7],
                   [4, 6]]], dtype=ht.int32, device=cpu:0, split=None)

    `round(self, decimals=0, out=None, dtype=None)`
    :   Calculate the rounded value element-wise.

        Parameters
        ----------
        x : DNDarray
            The array for which the compute the rounded value.
        decimals: int, optional
            Number of decimal places to round to.
            If decimals is negative, it specifies the number of positions to the left of the decimal point.
        out : DNDarray, optional
            A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
            If not provided or ``None``, a freshly-allocated array is returned.
        dtype : datatype, optional
            Determines the data type of the output array. The values are cast to this type with potential loss of
            precision.

        Raises
        ------
        TypeError
            if dtype is not a heat data type

        Examples
        --------
        >>> import heat as ht
        >>> ht.round(ht.arange(-2.0, 2.0, 0.4))
        DNDarray([-2., -2., -1., -1., -0.,  0.,  0.,  1.,  1.,  2.], dtype=ht.float32, device=cpu:0, split=None)

    `save(self, path, *args, **kwargs)`
    :   Attempts to save data from a :class:`~heat.core.dndarray.DNDarray` to disk. An auto-detection based on the file
        format extension is performed.

        Parameters
        ----------
        data : DNDarray
            The array holding the data to be stored
        path : str
            Path to the file to be stored.
        args : list, optional
            Additional options passed to the particular functions.
        kwargs : dict, optional
            Additional options passed to the particular functions.

        Raises
        ------
        ValueError
            If the file extension is not understood or known.
        RuntimeError
            If the optional dependency for a file extension is not available.

        Examples
        --------
        >>> x = ht.arange(100, split=0)
        >>> ht.save(x, "data.h5", "DATA", mode="a")

    `save_hdf5(self, path, dataset, mode='w', **kwargs)`
    :   Saves ``data`` to an HDF5 file. Attempts to utilize parallel I/O if possible.

        Parameters
        ----------
        data : DNDarray
            The data to be saved on disk.
        path : str
            Path to the HDF5 file to be written.
        dataset : str
            Name of the dataset the data is saved to.
        mode : str, optional
            File access mode, one of ``'w', 'a', 'r+'``
        kwargs : dict, optional
            Additional arguments passed to the created dataset.

        Raises
        ------
        TypeError
            If any of the input parameters are not of correct type.
        ValueError
            If the access mode is not understood.

        Examples
        --------
        >>> x = ht.arange(100, split=0)
        >>> ht.save_hdf5(x, "data.h5", dataset="DATA")

    `save_netcdf(self, path, variable, mode='w', **kwargs)`
    :   Saves data to a netCDF4 file. Attempts to utilize parallel I/O if possible.

        Parameters
        ----------
        data : DNDarray
            The data to be saved on disk.
        path : str
            Path to the netCDF4 file to be written.
        variable : str
            Name of the variable the data is saved to.
        mode : str, optional
            File access mode, one of ``'w', 'a', 'r+'``.
        dimension_names : list or tuple or string
            Specifies the netCDF Dimensions used by the variable. Ignored if Variable already exists.
        is_unlimited : bool, optional
            If True, every dimension created for this variable (i.e. doesn't already exist) is unlimited. Already
            existing limited dimensions cannot be changed to unlimited and vice versa.
        file_slices : integer iterable, slice, ellipsis or bool
            Keys used to slice the netCDF Variable, as given in the nc.utils._StartCountStride method.
        kwargs : dict, optional
            additional arguments passed to the created dataset.

        Raises
        ------
        TypeError
            If any of the input parameters are not of correct type.
        ValueError
            If the access mode is not understood or if the number of dimension names does not match the number of
            dimensions.

        Examples
        --------
        >>> x = ht.arange(100, split=0)
        >>> ht.save_netcdf(x, "data.nc", dataset="DATA")

    `sin(self, out=None)`
    :   Compute the trigonometric sine, element-wise.
        Result is a ``DNDarray`` of the same shape as ``x``.
        Negative input elements are returned as ``NaN``. If ``out`` was provided, ``sin`` is a reference to it.

        Parameters
        ----------
        x : DNDarray
            The value for which to compute the trigonometric tangent.
        out : DNDarray, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to ``None``, a fresh array is allocated.

        Examples
        --------
        >>> ht.sin(ht.arange(-6, 7, 2))
        DNDarray([ 0.2794,  0.7568, -0.9093,  0.0000,  0.9093, -0.7568, -0.2794], dtype=ht.float32, device=cpu:0, split=None)

    `sinh(self, out=None)`
    :   Compute the hyperbolic sine, element-wise.
        Result is a ``DNDarray`` of the same shape as ``x``.
        Negative input elements are returned as ``NaN``. If ``out`` was provided, ``sinh`` is a reference to it.

        Parameters
        ----------
        x : DNDarray
            The value for which to compute the hyperbolic sine.
        out : DNDarray or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to ``None``, a fresh array is allocated.

        Examples
        --------
        >>> ht.sinh(ht.arange(-6, 7, 2))
        DNDarray([-201.7132,  -27.2899,   -3.6269,    0.0000,    3.6269,   27.2899,  201.7132], dtype=ht.float32, device=cpu:0, split=None)

    `skew(self, axis=None, unbiased=True)`
    :   Compute the sample skewness of a data set.

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
        UserWarning: Dependent on the axis given and the split configuration, a UserWarning may be thrown during this function as data is transferred between processes.

    `sqrt(self, out=None)`
    :   Return the non-negative square-root of a tensor element-wise.
        Result is a :py:class:`~heat.core.dndarray.DNDarray` of the same shape as ``x``.
        Negative input elements are returned as :abbr:`NaN (Not a Number)`.

        Parameters
        ----------
        x : DNDarray
            The array for which to compute the square-roots.
        out : DNDarray, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to :keyword:`None`, a fresh array is allocated.

        Examples
        --------
        >>> ht.sqrt(ht.arange(5))
        DNDarray([0.0000, 1.0000, 1.4142, 1.7321, 2.0000], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.sqrt(ht.arange(-5, 0))
        DNDarray([nan, nan, nan, nan, nan], dtype=ht.float32, device=cpu:0, split=None)

    `square(self, out=None)`
    :   Return a new tensor with the squares of the elements of input.

        Parameters
        ----------
        x : DNDarray
            The array for which to compute the squares.
        out : DNDarray, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to :keyword:`None`, a fresh array is allocated.

        Examples
        --------
        >>> a = ht.random.rand(4)
        >>> a
        DNDarray([0.8654, 0.1432, 0.9164, 0.6179], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.square(a)
        DNDarray([0.7488, 0.0205, 0.8397, 0.3818], dtype=ht.float32, device=cpu:0, split=None)

    `squeeze(self, axis=None)`
    :   Remove single-element entries from the shape of a `DNDarray`.
        Returns the input array, but with all or a subset (indicated by `axis`) of the dimensions of length 1 removed.
        Split semantics: see Notes below.

        Parameters
        ----------
        x : DNDarray
            Input data.
        axis : None or int or Tuple[int,...], optional
               Selects a subset of the single-element entries in the shape.
               If axis is `None`, all single-element entries will be removed from the shape.

        Raises
        ------
        `ValueError`, if an axis is selected with shape entry greater than one.

        Notes
        -----
        Split semantics: a distributed DNDarray will keep its original split dimension after "squeezing",
        which, depending on the squeeze axis, may result in a lower numerical `split` value (see Examples).

        Examples
        --------
        >>> import heat as ht
        >>> a = ht.random.randn(1, 3, 1, 5)
        >>> a
        DNDarray([[[[-0.2604,  1.3512,  0.1175,  0.4197,  1.3590]],
                   [[-0.2777, -1.1029,  0.0697, -1.3074, -1.1931]],
                   [[-0.4512, -1.2348, -1.1479, -0.0242,  0.4050]]]], dtype=ht.float32, device=cpu:0, split=None)
        >>> a.shape
        (1, 3, 1, 5)
        >>> ht.squeeze(a).shape
        (3, 5)
        >>> ht.squeeze(a)
        DNDarray([[-0.2604,  1.3512,  0.1175,  0.4197,  1.3590],
                  [-0.2777, -1.1029,  0.0697, -1.3074, -1.1931],
                  [-0.4512, -1.2348, -1.1479, -0.0242,  0.4050]], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.squeeze(a, axis=0).shape
        (3, 1, 5)
        >>> ht.squeeze(a, axis=-2).shape
        (1, 3, 5)
        >>> ht.squeeze(a, axis=1).shape
        Traceback (most recent call last):
        ...
        ValueError: Dimension along axis 1 is not 1 for shape (1, 3, 1, 5)
        >>> x.shape
        (10, 1, 12, 13)
        >>> x.split
        2
        >>> x.squeeze().shape
        (10, 12, 13)
        >>> x.squeeze().split
        1

    `std(self, axis=None, ddof=0, **kwargs)`
    :   Calculates the standard deviation of a ``DNDarray`` with the bessel correction.
        If an axis is given, the variance will be taken in that direction.

        Parameters
        ----------
        x : DNDarray
            array for which the std is calculated for.
            The datatype of ``x`` must be a float
        axis : None or int or iterable
            Axis which the std is taken in. Default ``None`` calculates std of all data items.
        ddof : int, optional
            Delta Degrees of Freedom: the denominator implicitely used in the calculation is N - ddof, where N
            represents the number of elements. If ``ddof=1``, the Bessel correction will be applied.
            Setting ``ddof>1`` raises a ``NotImplementedError``.
        **kwargs
            Extra keyword arguments

        Examples
        --------
        >>> a = ht.random.randn(1, 3)
        >>> a
        DNDarray([[ 0.5714,  0.0048, -0.2942]], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.std(a)
        DNDarray(0.3590, dtype=ht.float32, device=cpu:0, split=None)
        >>> a = ht.random.randn(4, 4)
        >>> a
        DNDarray([[ 0.8488,  1.2225,  1.2498, -1.4592],
                  [-0.5820, -0.3928,  0.1509, -0.0174],
                  [ 0.6426, -1.8149,  0.1369,  0.0042],
                  [-0.6043, -0.0523, -1.6653,  0.6631]], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.std(a, 1, ddof=1)
        DNDarray([1.2961, 0.3362, 1.0739, 0.9820], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.std(a, 1)
        DNDarray([1.2961, 0.3362, 1.0739, 0.9820], dtype=ht.float32, device=cpu:0, split=None)

    `sub_(t1: DNDarray, t2: Union[DNDarray, float]) ‑> heat.core.dndarray.DNDarray`
    :   Element-wise in-place substitution of values of two operands.
        Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise subtracts the
        element(s) of the second operand (scalar or :class:`~heat.core.dndarray.DNDarray`) in-place,
        i.e. the element(s) of `t1` are overwritten by the results of element-wise subtraction of `t2`
        from `t1`.
        Can be called as a DNDarray method or with the symbol `-=`. `subtract_` is an alias for `sub_`.

        Parameters
        ----------
        t1: DNDarray
            The first operand involved in the subtraction
        t2: DNDarray or scalar
            The second operand involved in the subtraction

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.float32([[1, 2], [3, 4]])
        >>> T2 = ht.float32([[2, 2], [2, 2]])
        >>> T1 -= T2
        >>> T1
        DNDarray([[-1., 0.],
                  [ 1., 2.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[2., 2.],
                  [2., 2.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> s = 2.0
        >>> ht.sub_(T2, s)
        DNDarray([[0., 0.],
                  [0., 0.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[0., 0.],
                  [0., 0.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> s
        2.0
        >>> v = ht.int32([-3, 2])
        >>> T2.subtract_(v)
        DNDarray([[ 3., -2.],
                  [ 3., -2.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[ 3., -2.],
                  [ 3., -2.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> v
        DNDarray([-3,  2], dtype=ht.int32, device=cpu:0, split=None)

    `subtract_(t1: DNDarray, t2: Union[DNDarray, float]) ‑> heat.core.dndarray.DNDarray`
    :   Element-wise in-place substitution of values of two operands.
        Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise subtracts the
        element(s) of the second operand (scalar or :class:`~heat.core.dndarray.DNDarray`) in-place,
        i.e. the element(s) of `t1` are overwritten by the results of element-wise subtraction of `t2`
        from `t1`.
        Can be called as a DNDarray method or with the symbol `-=`. `subtract_` is an alias for `sub_`.

        Parameters
        ----------
        t1: DNDarray
            The first operand involved in the subtraction
        t2: DNDarray or scalar
            The second operand involved in the subtraction

        Raises
        ------
        ValueError
            If both inputs are DNDarrays that do not have the same split axis and the shapes of their
            underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
        TypeError
            If the data type of `t2` cannot be cast to the data type of `t1`. Although the
            corresponding out-of-place operation may work, for the in-place version the requirements
            are stricter, because the data type of `t1` does not change.

        Examples
        --------
        >>> import heat as ht
        >>> T1 = ht.float32([[1, 2], [3, 4]])
        >>> T2 = ht.float32([[2, 2], [2, 2]])
        >>> T1 -= T2
        >>> T1
        DNDarray([[-1., 0.],
                  [ 1., 2.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[2., 2.],
                  [2., 2.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> s = 2.0
        >>> ht.sub_(T2, s)
        DNDarray([[0., 0.],
                  [0., 0.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[0., 0.],
                  [0., 0.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> s
        2.0
        >>> v = ht.int32([-3, 2])
        >>> T2.subtract_(v)
        DNDarray([[ 3., -2.],
                  [ 3., -2.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> T2
        DNDarray([[ 3., -2.],
                  [ 3., -2.]], dtype=ht.float32, device=cpu:0, split=None)
        >>> v
        DNDarray([-3,  2], dtype=ht.int32, device=cpu:0, split=None)

    `sum(self, axis=None, out=None, keepdims=None)`
    :

    `swapaxes(self, axis1, axis2)`
    :   Interchanges two axes of an array.

        Parameters
        ----------
        x : DNDarray
            Input array.
        axis1 : int
            First axis.
        axis2 : int
            Second axis.

        See Also
        --------
        :func:`~heat.core.linalg.basics.transpose`
            Permute the dimensions of an array.

        Examples
        --------
        >>> x = ht.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
        >>> ht.swapaxes(x, 0, 1)
        DNDarray([[[0, 1],
                   [4, 5]],
                  [[2, 3],
                   [6, 7]]], dtype=ht.int64, device=cpu:0, split=None)
        >>> ht.swapaxes(x, 0, 2)
        DNDarray([[[0, 4],
                   [2, 6]],
                  [[1, 5],
                   [3, 7]]], dtype=ht.int64, device=cpu:0, split=None)

    `tan(self, out=None)`
    :   Compute tangent element-wise.
        Result is a ``DNDarray`` of the same shape as ``x``.
        Equivalent to :func:`sin`/:func:`cos` element-wise. If ``out`` was provided, ``tan`` is a reference to it.


        Parameters
        ----------
        x : DNDarray
            The value for which to compute the trigonometric tangent.
        out : DNDarray or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to ``None``, a fresh array is allocated.

        Examples
        --------
        >>> ht.tan(ht.arange(-6, 7, 2))
        DNDarray([ 0.2910, -1.1578,  2.1850,  0.0000, -2.1850,  1.1578, -0.2910], dtype=ht.float32, device=cpu:0, split=None)

    `tanh(self, out=None)`
    :   Compute the hyperbolic tangent, element-wise.
        Result is a ``DNDarray`` of the same shape as ``x``.
        If ``out`` was provided, ``tanh`` is a reference to it.

        Parameters
        ----------
        x : DNDarray
            The value for which to compute the hyperbolic tangent.
        out : DNDarray or None, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to ``None``, a fresh array is allocated.

        Examples
        --------
        >>> ht.tanh(ht.arange(-6, 7, 2))
        DNDarray([-1.0000, -0.9993, -0.9640,  0.0000,  0.9640,  0.9993,  1.0000], dtype=ht.float32, device=cpu:0, split=None)

    `to_sparse_csc(array: DNDarray) ‑> heat.sparse.dcsx_matrix.DCSC_matrix`
    :   Convert the distributed array to a sparse DCSC_matrix representation.

        Parameters
        ----------
        array : DNDarray
            The distributed array to be converted to a sparse DCSC_matrix.

        Returns
        -------
        DCSC_matrix
            A sparse DCSC_matrix representation of the input DNDarray.

        Examples
        --------
        >>> dense_array = ht.array([[1, 0, 0], [0, 0, 2], [0, 3, 0]])
        >>> dense_array.to_sparse_csc()
        (indptr: tensor([0, 1, 2, 3]), indices: tensor([0, 2, 1]), data: tensor([1, 3, 2]), dtype=ht.int64, device=cpu:0, split=None)

    `to_sparse_csr(array: DNDarray) ‑> heat.sparse.dcsx_matrix.DCSR_matrix`
    :   Convert the distributed array to a sparse DCSR_matrix representation.

        Parameters
        ----------
        array : DNDarray
            The distributed array to be converted to a sparse DCSR_matrix.

        Returns
        -------
        DCSR_matrix
            A sparse DCSR_matrix representation of the input DNDarray.

        Examples
        --------
        >>> dense_array = ht.array([[1, 0, 0], [0, 0, 2], [0, 3, 0]])
        >>> dense_array.to_sparse_csr()
        (indptr: tensor([0, 1, 2, 3]), indices: tensor([0, 2, 1]), data: tensor([1, 2, 3]), dtype=ht.int64, device=cpu:0, split=None)

    `tolist(self, keepsplit: bool = False) ‑> List`
    :   Return a copy of the local array data as a (nested) Python list. For scalars, a standard Python number is returned.

        Parameters
        ----------
        keepsplit: bool
            Whether the list should be returned locally or globally.

        Examples
        --------
        >>> a = ht.array([[0, 1], [2, 3]])
        >>> a.tolist()
        [[0, 1], [2, 3]]

        >>> a = ht.array([[0, 1], [2, 3]], split=0)
        >>> a.tolist()
        [[0, 1], [2, 3]]

        >>> a = ht.array([[0, 1], [2, 3]], split=1)
        >>> a.tolist(keepsplit=True)
        (1/2) [[0], [2]]
        (2/2) [[1], [3]]

    `trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None)`
    :   Return the sum along diagonals of the array

        If `a` is 2D, the sum along its diagonal with the given offset is returned, i.e. the sum of
        elements a[i, i+offset] for all i.

        If `a` has more than two dimensions, then the axes specified by `axis1` and `axis2` are used
        to determine the 2D-sub-DNDarrays whose traces are returned.
        The shape of the resulting array is the same as that of `a` with `axis1` and `axis2` removed.

        Parameters
        ----------
        a : array_like
            Input array, from which the diagonals are taken
        offset : int, optional
            Offsets of the diagonal from the main diagonal. Can be both positive and negative. Defaults to 0.
        axis1: int, optional
            Axis to be used as the first axis of the 2D-sub-arrays from which the diagonals
            should be taken. Default is the first axis of `a`
        axis2 : int, optional
            Axis to be used as the second axis of the 2D-sub-arrays from which the diagonals
            should be taken. Default is the second two axis of `a`
        dtype : dtype, optional
            Determines the data-type of the returned array and of the accumulator where the elements are
            summed. If `dtype` has value None than the dtype is the same as that of `a`
        out: ht.DNDarray, optional
            Array into which the output is placed. Its type is preserved and it must be of the right shape
            to hold the output
            Only applicable if `a` has more than 2 dimensions, thus the result is not a scalar.
            If distributed, its split axis might change eventually.

        Returns
        -------
        sum_along_diagonals : number (of defined dtype) or ht.DNDarray
            If `a` is 2D, the sum along the diagonal is returned as a scalar
            If `a` has more than 2 dimensions, then a DNDarray of sums along diagonals is returned

        Examples
        --------
        2D-case
        >>> x = ht.arange(24).reshape((4, 6))
        >>> x
            DNDarray([[ 0,  1,  2,  3,  4,  5],
                      [ 6,  7,  8,  9, 10, 11],
                      [12, 13, 14, 15, 16, 17],
                      [18, 19, 20, 21, 22, 23]], dtype=ht.int32, device=cpu:0, split=None)
        >>> ht.trace(x)
            42
        >>> ht.trace(x, 1)
            46
        >>> ht.trace(x, -2)
            31

        > 2D-case
        >>> x = x.reshape((2, 3, 4))
        >>> x
            DNDarray([[[ 0,  1,  2,  3],
                       [ 4,  5,  6,  7],
                       [ 8,  9, 10, 11]],

                      [[12, 13, 14, 15],
                       [16, 17, 18, 19],
                       [20, 21, 22, 23]]], dtype=ht.int32, device=cpu:0, split=None)
        >>> ht.trace(x)
            DNDarray([16, 18, 20, 22], dtype=ht.int32, device=cpu:0, split=None)
        >>> ht.trace(x, 1)
            DNDarray([24, 26, 28, 30], dtype=ht.int32, device=cpu:0, split=None)
        >>> ht.trace(x, axis1=0, axis2=2)
            DNDarray([13, 21, 29], dtype=ht.int32, device=cpu:0, split=None)

    `transpose(self, axes=None)`
    :   Permute the dimensions of an array.

        Parameters
        ----------
        a : DNDarray
            Input array.
        axes : None or List[int,...], optional
            By default, reverse the dimensions, otherwise permute the axes according to the values given.

    `tril(self, k=0)`
    :   Returns the lower triangular part of the ``DNDarray``.
        The lower triangular part of the array is defined as the elements on and below the diagonal, the other elements of
        the result array are set to 0.
        The argument ``k`` controls which diagonal to consider. If ``k=0``, all elements on and below the main diagonal are
        retained. A positive value includes just as many diagonals above the main diagonal, and similarly a negative
        value excludes just as many diagonals below the main diagonal.

        Parameters
        ----------
        m : DNDarray
            Input array for which to compute the lower triangle.
        k : int, optional
            Diagonal above which to zero elements. ``k=0`` (default) is the main diagonal, ``k<0`` is below and ``k>0`` is above.

    `triu(self, k=0)`
    :   Returns the upper triangular part of the ``DNDarray``.
        The upper triangular part of the array is defined as the elements on and below the diagonal, the other elements of the result array are set to 0.
        The argument ``k`` controls which diagonal to consider. If ``k=0``, all elements on and below the main diagonal are
        retained. A positive value includes just as many diagonals above the main diagonal, and similarly a negative
        value excludes just as many diagonals below the main diagonal.

        Parameters
        ----------
        m : DNDarray
            Input array for which to compute the upper triangle.
        k : int, optional
            Diagonal above which to zero elements. ``k=0`` (default) is the main diagonal, ``k<0`` is below and ``k>0`` is above.

    `trunc(self, out=None)`
    :   Return the trunc of the input, element-wise.
        The truncated value of the scalar ``x`` is the nearest integer ``i`` which is closer to zero than ``x`` is. In short, the
        fractional part of the signed number ``x`` is discarded.

        Parameters
        ----------
        x : DNDarray
            The array for which to compute the trunced values.
        out : DNDarray, optional
            A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
            or set to ``None``, a fresh array is allocated.

        Examples
        --------
        >>> import heat as ht
        >>> ht.trunc(ht.arange(-2.0, 2.0, 0.4))
        DNDarray([-2., -1., -1., -0., -0.,  0.,  0.,  0.,  1.,  1.], dtype=ht.float32, device=cpu:0, split=None)

    `unique(self, sorted=False, return_inverse=False, axis=None)`
    :   Finds and returns the unique elements of a `DNDarray`.
        If return_inverse is `True`, the second tensor will hold the list of inverse indices
        If distributed, it is most efficient if `axis!=a.split`.

        Parameters
        ----------
        a : DNDarray
            Input array.
        sorted : bool, optional
            Whether the found elements should be sorted before returning as output.
            Warning: sorted is not working if `axis!=None and axis!=a.split`
        return_inverse : bool, optional
            Whether to also return the indices for where elements in the original input ended up in the returned
            unique list.
        axis : int, optional
            Axis along which unique elements should be found. Default to `None`, which will return a one dimensional list of
            unique values.

        Examples
        --------
        >>> x = ht.array([[3, 2], [1, 3]])
        >>> ht.unique(x, sorted=True)
        array([1, 2, 3])
        >>> ht.unique(x, sorted=True, axis=0)
        array([[1, 3],
               [2, 3]])
        >>> ht.unique(x, sorted=True, axis=1)
        array([[2, 3],
               [3, 1]])

    `var(self, axis=None, ddof=0, **kwargs)`
    :   Calculates and returns the variance of a ``DNDarray``. If an axis is given, the variance will be
        taken in that direction.

        Parameters
        ----------
        x : DNDarray
            Array for which the variance is calculated for.
            The datatype of ``x`` must be a float
        axis : None or int or iterable
            Axis which the std is taken in. Default ``None`` calculates std of all data items.
        ddof : int, optional
            Delta Degrees of Freedom: the denominator implicitely used in the calculation is N - ddof, where N
            represents the number of elements. If ``ddof=1``, the Bessel correction will be applied.
            Setting ``ddof>1`` raises a ``NotImplementedError``.
        **kwargs
            Extra keyword arguments


        Notes
        -----
        Split semantics when axis is an integer:

        - if ``axis=x.split``, then ``var(x).split=None``

        - if ``axis>split``, then ``var(x).split = x.split``

        - if ``axis<split``, then ``var(x).split=x.split - 1``

        The variance is the average of the squared deviations from the mean, i.e., ``var=mean(abs(x - x.mean())**2)``.
        The mean is normally calculated as ``x.sum()/N``, where ``N = len(x)``. If, however, ``ddof`` is specified, the divisor
        ``N - ddof`` is used instead. In standard statistical practice, ``ddof=1`` provides an unbiased estimator of the
        variance of a hypothetical infinite population. ``ddof=0`` provides a maximum likelihood estimate of the variance
        for normally distributed variables.

        Examples
        --------
        >>> a = ht.random.randn(1, 3)
        >>> a
        DNDarray([[-2.3589, -0.2073,  0.8806]], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.var(a)
        DNDarray(1.8119, dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.var(a, ddof=1)
        DNDarray(2.7179, dtype=ht.float32, device=cpu:0, split=None)
        >>> a = ht.random.randn(4, 4)
        >>> a
        DNDarray([[-0.8523, -1.4982, -0.5848, -0.2554],
                  [ 0.8458, -0.3125, -0.2430,  1.9016],
                  [-0.6778, -0.3584, -1.5112,  0.6545],
                  [-0.9161,  0.0168,  0.0462,  0.5964]], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.var(a, 1)
        DNDarray([0.2777, 1.0957, 0.8015, 0.3936], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.var(a, 0)
        DNDarray([0.7001, 0.4376, 0.4576, 0.7890], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.var(a, 0, ddof=1)
        DNDarray([0.7001, 0.4376, 0.4576, 0.7890], dtype=ht.float32, device=cpu:0, split=None)
        >>> ht.var(a, 0, ddof=0)
        DNDarray([0.7001, 0.4376, 0.4576, 0.7890], dtype=ht.float32, device=cpu:0, split=None)
