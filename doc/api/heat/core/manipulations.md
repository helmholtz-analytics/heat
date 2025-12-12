Module heat.core.manipulations
==============================
Manipulation operations for (potentially distributed) `DNDarray`s.

Functions
---------

`balance(array: DNDarray, copy=False) ‑> heat.core.dndarray.DNDarray`
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

`broadcast_arrays(*arrays: DNDarray) ‑> List[heat.core.dndarray.DNDarray]`
:   Broadcasts one or more arrays against one another. Returns the broadcasted arrays, distributed along the split dimension of the first array in the list. If the first array is not distributed, the output will not be distributed.

    Parameters
    ----------
    arrays : DNDarray
        An arbitrary number of to-be broadcasted ``DNDarray``s.

    Notes
    -----
    Broadcasted arrays are a view of the original arrays if possible, otherwise a copy is made.

    Examples
    --------
    >>> import heat as ht
    >>> a = ht.ones((100, 10), split=0)
    >>> b = ht.ones((10,), split=None)
    >>> c = ht.ones((1, 10), split=1)
    >>> d, e, f = ht.broadcast_arrays(a, b, c)
    >>> d.shape
    (100, 10)
    >>> e.shape
    (100, 10)
    >>> f.shape
    (100, 10)
    >>> d.split
    0
    >>> e.split
    0
    >>> f.split
    0

`broadcast_to(x: DNDarray, shape: Tuple[int, ...]) ‑> heat.core.dndarray.DNDarray`
:   Broadcasts an array to a specified shape. Returns a view of ``x`` if ``x`` is not distributed, otherwise it returns a broadcasted, distributed, load-balanced copy of ``x``.

    Parameters
    ----------
    x : DNDarray
        `DNDarray` to broadcast.
    shape : Tuple[int, ...]
        Array shape. Must be compatible with ``x``.

    Raises
    ------
    ValueError
        If the array is not compatible with the new shape according to PyTorch's broadcasting rules.

    Examples
    --------
    >>> import heat as ht
    >>> a = ht.arange(100, split=0)
    >>> b = ht.broadcast_to(a, (10, 100))
    >>> b.shape
    (10, 100)
    >>> b.split
    1
    >>> c = ht.broadcast_to(a, (100, 10))
    ValueError: Shape mismatch: object cannot be broadcast to the given shape. Original shape: (100,), target shape: (100, 10)

`collect(arr: DNDarray, target_rank: Optional[int] = 0) ‑> heat.core.dndarray.DNDarray`
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

`column_stack(arrays: Sequence[DNDarray, ...]) ‑> DNDarray`
:   Stack 1-D or 2-D `DNDarray`s as columns into a 2-D `DNDarray`.
    If the input arrays are 1-D, they will be stacked as columns. If they are 2-D,
    they will be concatenated along the second axis.

    Parameters
    ----------
    arrays : Sequence[DNDarray, ...]
        Sequence of `DNDarray`s.

    Raises
    ------
    ValueError
        If arrays have more than 2 dimensions

    Notes
    -----
    All `DNDarray`s in the sequence must have the same number of rows.
    All `DNDarray`s must be split along the same axis! Note that distributed
    1-D arrays (`split = 0`) by default will be transposed into distributed
    column arrays with `split == 1`.

    See Also
    --------
    :func:`concatenate`
    :func:`hstack`
    :func:`row_stack`
    :func:`stack`
    :func:`vstack`

    Examples
    --------
    >>> # 1-D tensors
    >>> a = ht.array([1, 2, 3])
    >>> b = ht.array([2, 3, 4])
    >>> ht.column_stack((a, b)).larray
    tensor([[1, 2],
            [2, 3],
            [3, 4]])
    >>> # 1-D and 2-D tensors
    >>> a = ht.array([1, 2, 3])
    >>> b = ht.array([[2, 5], [3, 6], [4, 7]])
    >>> c = ht.array([[7, 10], [8, 11], [9, 12]])
    >>> ht.column_stack((a, b, c)).larray
    tensor([[ 1,  2,  5,  7, 10],
            [ 2,  3,  6,  8, 11],
            [ 3,  4,  7,  9, 12]])
    >>> # distributed DNDarrays, 3 processes
    >>> a = ht.arange(10, split=0).reshape((5, 2))
    >>> b = ht.arange(5, 20, split=0).reshape((5, 3))
    >>> c = ht.arange(20, 40, split=0).reshape((5, 4))
    >>> ht_column_stack((a, b, c)).larray
    [0/2] tensor([[ 0,  1,  5,  6,  7, 20, 21, 22, 23],
    [0/2]         [ 2,  3,  8,  9, 10, 24, 25, 26, 27]], dtype=torch.int32)
    [1/2] tensor([[ 4,  5, 11, 12, 13, 28, 29, 30, 31],
    [1/2]         [ 6,  7, 14, 15, 16, 32, 33, 34, 35]], dtype=torch.int32)
    [2/2] tensor([[ 8,  9, 17, 18, 19, 36, 37, 38, 39]], dtype=torch.int32)
    >>> # distributed 1-D and 2-D DNDarrays, 3 processes
    >>> a = ht.arange(5, split=0)
    >>> b = ht.arange(5, 20, split=1).reshape((5, 3))
    >>> ht_column_stack((a, b)).larray
    [0/2] tensor([[ 0,  5],
    [0/2]         [ 1,  8],
    [0/2]         [ 2, 11],
    [0/2]         [ 3, 14],
    [0/2]         [ 4, 17]], dtype=torch.int32)
    [1/2] tensor([[ 6],
    [1/2]         [ 9],
    [1/2]         [12],
    [1/2]         [15],
    [1/2]         [18]], dtype=torch.int32)
    [2/2] tensor([[ 7],
    [2/2]         [10],
    [2/2]         [13],
    [2/2]         [16],
    [2/2]         [19]], dtype=torch.int32)

`concatenate(arrays: Sequence[DNDarray, ...], axis: int = 0) ‑> DNDarray`
:   Join 2 or more `DNDarrays` along an existing axis.

    Parameters
    ----------
    arrays: Sequence[DNDarray, ...]
        The arrays must have the same shape, except in the dimension corresponding to axis.
    axis: int, optional
        The axis along which the arrays will be joined (default is 0).

    Raises
    ------
    RuntimeError
        If the concatenated :class:`~heat.core.dndarray.DNDarray` meta information, e.g. `split` or `comm`, does not match.
    TypeError
        If the passed parameters are not of correct type.
    ValueError
        If the number of passed arrays is less than two or their shapes do not match.

    Examples
    --------
    >>> x = ht.zeros((3, 5), split=None)
    [0/1] tensor([[0., 0., 0., 0., 0.],
    [0/1]         [0., 0., 0., 0., 0.],
    [0/1]         [0., 0., 0., 0., 0.]])
    [1/1] tensor([[0., 0., 0., 0., 0.],
    [1/1]         [0., 0., 0., 0., 0.],
    [1/1]         [0., 0., 0., 0., 0.]])
    >>> y = ht.ones((3, 6), split=0)
    [0/1] tensor([[1., 1., 1., 1., 1., 1.],
    [0/1]         [1., 1., 1., 1., 1., 1.]])
    [1/1] tensor([[1., 1., 1., 1., 1., 1.]])
    >>> ht.concatenate((x, y), axis=1)
    [0/1] tensor([[0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.],
    [0/1]         [0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.]])
    [1/1] tensor([[0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.]])
    >>> x = ht.zeros((4, 5), split=1)
    [0/1] tensor([[0., 0., 0.],
    [0/1]         [0., 0., 0.],
    [0/1]         [0., 0., 0.],
    [0/1]         [0., 0., 0.]])
    [1/1] tensor([[0., 0.],
    [1/1]         [0., 0.],
    [1/1]         [0., 0.],
    [1/1]         [0., 0.]])
    >>> y = ht.ones((3, 5), split=1)
    [0/1] tensor([[1., 1., 1.],
    [0/1]         [1., 1., 1.],
    [0/1]         [1., 1., 1.]])
    [1/1] tensor([[1., 1.],
    [1/1]         [1., 1.],
    [1/1]         [1., 1.]])
    >>> ht.concatenate((x, y), axis=0)
    [0/1] tensor([[0., 0., 0.],
    [0/1]         [0., 0., 0.],
    [0/1]         [0., 0., 0.],
    [0/1]         [0., 0., 0.],
    [0/1]         [1., 1., 1.],
    [0/1]         [1., 1., 1.],
    [0/1]         [1., 1., 1.]])
    [1/1] tensor([[0., 0.],
    [1/1]         [0., 0.],
    [1/1]         [0., 0.],
    [1/1]         [0., 0.],
    [1/1]         [1., 1.],
    [1/1]         [1., 1.],
    [1/1]         [1., 1.]])

`diag(a: DNDarray, offset: int = 0) ‑> heat.core.dndarray.DNDarray`
:   Extract a diagonal or construct a diagonal array.
    See the documentation for :func:`diagonal` for more information about extracting the diagonal.

    Parameters
    ----------
    a: DNDarray
        The array holding data for creating a diagonal array or extracting a diagonal.
        If `a` is a 1-dimensional array, a diagonal 2d-array will be returned.
        If `a` is a n-dimensional array with n > 1 the diagonal entries will be returned in an n-1 dimensional array.
    offset: int, optional
        The offset from the main diagonal.
        Offset greater than zero means above the main diagonal, smaller than zero is below the main diagonal.

    See Also
    --------
    :func:`diagonal`

    Examples
    --------
    >>> import heat as ht
    >>> a = ht.array([1, 2])
    >>> ht.diag(a)
    DNDarray([[1, 0],
              [0, 2]], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.diag(a, offset=1)
    DNDarray([[0, 1, 0],
              [0, 0, 2],
              [0, 0, 0]], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.equal(ht.diag(ht.diag(a)), a)
    True
    >>> a = ht.array([[1, 2], [3, 4]])
    >>> ht.diag(a)
    DNDarray([1, 4], dtype=ht.int64, device=cpu:0, split=None)

`diagonal(a: DNDarray, offset: int = 0, dim1: int = 0, dim2: int = 1) ‑> heat.core.dndarray.DNDarray`
:   Extract a diagonal of an n-dimensional array with n > 1.
    The returned array will be of dimension n-1.

    Parameters
    ----------
    a: DNDarray
        The array of which the diagonal should be extracted.
    offset: int, optional
        The offset from the main diagonal.
        Offset greater than zero means above the main diagonal, smaller than zero is below the main diagonal.
        Default is 0 which means the main diagonal will be selected.
    dim1: int, optional
        First dimension with respect to which to take the diagonal.
    dim2: int, optional
        Second dimension with respect to which to take the diagonal.

    Examples
    --------
    >>> import heat as ht
    >>> a = ht.array([[1, 2], [3, 4]])
    >>> ht.diagonal(a)
    DNDarray([1, 4], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.diagonal(a, offset=1)
    DNDarray([2], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.diagonal(a, offset=-1)
    DNDarray([3], dtype=ht.int64, device=cpu:0, split=None)
    >>> a = ht.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    >>> ht.diagonal(a)
    DNDarray([[0, 6],
              [1, 7]], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.diagonal(a, dim2=2)
    DNDarray([[0, 5],
              [2, 7]], dtype=ht.int64, device=cpu:0, split=None)

`dsplit(x: Sequence[DNDarray, ...], indices_or_sections: Iterable) ‑> List[DNDarray, ...]`
:   Split array into multiple sub-DNDarrays along the 3rd axis (depth).
    Returns a list of sub-DNDarrays as copies of parts of `x`.

    Parameters
    ----------
    x : DNDarray
        DNDArray to be divided into sub-DNDarrays.
    indices_or_sections : int or 1-dimensional array_like (i.e. undistributed DNDarray, list or tuple)
        If `indices_or_sections` is an integer, N, the DNDarray will be divided into N equal DNDarrays along the 3rd axis.
        If such a split is not possible, an error is raised.
        If `indices_or_sections` is a 1-D DNDarray of sorted integers, the entries indicate where along the 3rd axis
        the array is split.
        If an index exceeds the dimension of the array along the 3rd axis, an empty sub-DNDarray is returned correspondingly.

    Raises
    ------
    ValueError
        If `indices_or_sections` is given as integer, but a split does not result in equal division.

    Notes
    -----
    Please refer to the split documentation. dsplit is equivalent to split with axis=2,
    the array is always split along the third axis provided the array dimension is greater than or equal to 3.

    See Also
    --------
    :func:`split`
    :func:`hsplit`
    :func:`vsplit`

    Examples
    --------
    >>> x = ht.array(24).reshape((2, 3, 4))
    >>> ht.dsplit(x, 2)
        [DNDarray([[[ 0,  1],
                   [ 4,  5],
                   [ 8,  9]],
                   [[12, 13],
                   [16, 17],
                   [20, 21]]]),
        DNDarray([[[ 2,  3],
                   [ 6,  7],
                   [10, 11]],
                   [[14, 15],
                   [18, 19],
                   [22, 23]]])]
    >>> ht.dsplit(x, [1, 4])
        [DNDarray([[[ 0],
                    [ 4],
                    [ 8]],
                   [[12],
                    [16],
                    [20]]]),
        DNDarray([[[ 1,  2,  3],
                    [ 5,  6,  7],
                    [ 9, 10, 11]],
                    [[13, 14, 15],
                     [17, 18, 19],
                     [21, 22, 23]]]),
        DNDarray([])]

`expand_dims(a: DNDarray, axis: int) ‑> heat.core.dndarray.DNDarray`
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

`flatten(a: DNDarray) ‑> heat.core.dndarray.DNDarray`
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

`flip(a: DNDarray, axis: Union[int, Tuple[int, ...]] = None) ‑> heat.core.dndarray.DNDarray`
:   Reverse the order of elements in an array along the given axis.
    The shape of the array is preserved, but the elements are reordered.

    Parameters
    ----------
    a: DNDarray
        Input array to be flipped
    axis: int or Tuple[int,...]
        A list of axes to be flipped

    See Also
    --------
    :func:`fliplr`
    :func:`flipud`

    Examples
    --------
    >>> a = ht.array([[0, 1], [2, 3]])
    >>> ht.flip(a, [0])
    DNDarray([[2, 3],
              [0, 1]], dtype=ht.int64, device=cpu:0, split=None)
    >>> b = ht.array([[0, 1, 2], [3, 4, 5]], split=1)
    >>> ht.flip(a, [0, 1])
    (1/2) tensor([5,4,3])
    (2/2) tensor([2,1,0])

`fliplr(a: DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Flip array in the left/right direction. If `a.ndim>2`, flip along dimension 1.

    Parameters
    ----------
    a: DNDarray
        Input array to be flipped, must be at least 2-D

    See Also
    --------
    :func:`flip`
    :func:`flipud`

    Examples
    --------
    >>> a = ht.array([[0, 1], [2, 3]])
    >>> ht.fliplr(a)
    DNDarray([[1, 0],
              [3, 2]], dtype=ht.int64, device=cpu:0, split=None)
    >>> b = ht.array([[0, 1, 2], [3, 4, 5]], split=0)
    >>> ht.fliplr(b)
    (1/2) tensor([[2, 1, 0]])
    (2/2) tensor([[5, 4, 3]])

`flipud(a: DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Flip array in the up/down direction.

    Parameters
    ----------
    a: DNDarray
        Input array to be flipped

    See Also
    --------
    :func:`flip`
    :func:`fliplr`

    Examples
    --------
    >>> a = ht.array([[0, 1], [2, 3]])
    >>> ht.flipud(a)
    DNDarray([[2, 3],
              [0, 1]], dtype=ht.int64, device=cpu:0, split=None))
    >>> b = ht.array([[0, 1, 2], [3, 4, 5]], split=0)
    >>> ht.flipud(b)
    (1/2) tensor([3,4,5])
    (2/2) tensor([0,1,2])

`hsplit(x: DNDarray, indices_or_sections: Iterable) ‑> List[DNDarray, ...]`
:   Split array into multiple sub-DNDarrays along the 2nd axis (horizontally/column-wise).
    Returns a list of sub-DNDarrays as copies of parts of `x`.

    Parameters
    ----------
    x : DNDarray
        DNDArray to be divided into sub-DNDarrays.
    indices_or_sections : int or 1-dimensional array_like (i.e. undistributed DNDarray, list or tuple)
        If `indices_or_sections` is an integer, N, the DNDarray will be divided into N equal DNDarrays along the 2nd axis.
        If such a split is not possible, an error is raised.
        If `indices_or_sections` is a 1-D DNDarray of sorted integers, the entries indicate where along the 2nd axis
        the array is split.
        If an index exceeds the dimension of the array along the 2nd axis, an empty sub-DNDarray is returned correspondingly.

    Raises
    ------
    ValueError
        If `indices_or_sections` is given as integer, but a split does not result in equal division.

    Notes
    -----
    Please refer to the split documentation. hsplit is nearly equivalent to split with axis=1,
    the array is always split along the second axis though, in contrary to split, regardless of the array dimension.

    See Also
    --------
    :func:`split`
    :func:`dsplit`
    :func:`vsplit`

    Examples
    --------
    >>> x = ht.arange(24).reshape((2, 4, 3))
    >>> ht.hsplit(x, 2)
        [DNDarray([[[ 0,  1,  2],
                   [ 3,  4,  5]],
                  [[12, 13, 14],
                   [15, 16, 17]]]),
        DNDarray([[[ 6,  7,  8],
                   [ 9, 10, 11]],
                  [[18, 19, 20],
                   [21, 22, 23]]])]
    >>> ht.hsplit(x, [1, 3])
        [DNDarray([[[ 0,  1,  2]],
                  [[12, 13, 14]]]),
        DNDarray([[[ 3,  4,  5],
                   [ 6,  7,  8]],
                  [[15, 16, 17],
                   [18, 19, 20]]]),
        DNDarray([[[ 9, 10, 11]],
                  [[21, 22, 23]]])]

`hstack(arrays: Sequence[DNDarray, ...]) ‑> DNDarray`
:   Stack arrays in sequence horizontally (column-wise).
    This is equivalent to concatenation along the second axis, except for 1-D
    arrays where it concatenates along the first axis.

    Parameters
    ----------
    arrays : Sequence[DNDarray, ...]
        The arrays must have the same shape along all but the second axis,
        except 1-D arrays which can be any length.

    See Also
    --------
    :func:`concatenate`
    :func:`stack`
    :func:`vstack`
    :func:`column_stack`
    :func:`row_stack`

    Examples
    --------
    >>> a = ht.array((1, 2, 3))
    >>> b = ht.array((2, 3, 4))
    >>> ht.hstack((a, b)).larray
    [0/1] tensor([1, 2, 3, 2, 3, 4])
    [1/1] tensor([1, 2, 3, 2, 3, 4])
    >>> a = ht.array((1, 2, 3), split=0)
    >>> b = ht.array((2, 3, 4), split=0)
    >>> ht.hstack((a, b)).larray
    [0/1] tensor([1, 2, 3])
    [1/1] tensor([2, 3, 4])
    >>> a = ht.array([[1], [2], [3]], split=0)
    >>> b = ht.array([[2], [3], [4]], split=0)
    >>> ht.hstack((a, b)).larray
    [0/1] tensor([[1, 2],
    [0/1]         [2, 3]])
    [1/1] tensor([[3, 4]])

`moveaxis(x: DNDarray, source: Union[int, Sequence[int]], destination: Union[int, Sequence[int]]) ‑> heat.core.dndarray.DNDarray`
:   Moves axes at the positions in `source` to new positions.

    Parameters
    ----------
    x : DNDarray
        The input array.
    source : int or Sequence[int, ...]
        Original positions of the axes to move. These must be unique.
    destination : int or Sequence[int, ...]
        Destination positions for each of the original axes. These must also be unique.

    See Also
    --------
    ~heat.core.linalg.basics.transpose
        Permute the dimensions of an array.

    Raises
    ------
    TypeError
        If `source` or `destination` are not ints, lists or tuples.
    ValueError
        If `source` and `destination` do not have the same number of elements.


    Examples
    --------
    >>> x = ht.zeros((3, 4, 5))
    >>> ht.moveaxis(x, 0, -1).shape
    (4, 5, 3)
    >>> ht.moveaxis(x, -1, 0).shape
    (5, 3, 4)

`pad(array: DNDarray, pad_width: Union[int, Sequence[Sequence[int, int], ...]], mode: str = 'constant', constant_values: int = 0) ‑> DNDarray`
:   Pads tensor with a specific value (default=0).
    (Not all dimensions supported)

    Parameters
    ----------
    array : DNDarray
        Array to be padded
    pad_width: Union[int, Sequence[Sequence[int, int], ...]]
        Number of values padded to the edges of each axis. ((before_1, after_1),...(before_N, after_N)) unique pad widths for each axis.
        Determines how many elements are padded along which dimension.

        Shortcuts:

            - ((before, after),)  or (before, after): before and after pad width for each axis.
            - (pad_width,) or int: before = after = pad width for all axes.

        Therefore:

        - pad last dimension: (padding_left, padding_right)
        - pad last 2 dimensions: ((padding_top, padding_bottom),(padding_left, padding_right))
        - pad last 3 dimensions: ((padding_front, padding_back),(padding_top, padding_bottom),(paddling_left, padding_right) )
        - ... (same pattern)
    mode : str, optional
        - 'constant' (default): Pads the input tensor boundaries with a constant value. This is available for arbitrary dimensions
    constant_values: Union[int, float, Sequence[Sequence[int,int], ...], Sequence[Sequence[float,float], ...]]
        Number or tuple of 2-element-sequences (containing numbers), optional (default=0)
        The fill values for each axis (1 tuple per axis).
        ((before_1, after_1), ... (before_N, after_N)) unique pad values for each axis.

        Shortcuts:

            - ((before, after),) or (before, after): before and after padding values for each axis.
            - (value,) or int: before = after = padding value for all axes.


    Notes
    -----
    This function follows the principle of datatype integrity.
    Therefore, an array can only be padded with values of the same datatype.
    All values that violate this rule are implicitly cast to the datatype of the `DNDarray`.

    Examples
    --------
    >>> a = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
    >>> b = ht.array(a, split=0)
    Pad last dimension
    >>> c = ht.pad(b, (2, 1), constant_values=1)
    tensor([[[ 1,  1,  0,  1,  2,  3,  1],
         [ 1,  1,  4,  5,  6,  7,  1],
         [ 1,  1,  8,  9, 10, 11,  1]],
        [[ 1,  1, 12, 13, 14, 15,  1],
         [ 1,  1, 16, 17, 18, 19,  1],
         [ 1,  1, 20, 21, 22, 23,  1]]])
    Pad last 2 dimensions
    >>> d = ht.pad(b, [(1, 0), (2, 1)])
    DNDarray([[[ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  1,  2,  3,  0],
               [ 0,  0,  4,  5,  6,  7,  0],
               [ 0,  0,  8,  9, 10, 11,  0]],

              [[ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0, 12, 13, 14, 15,  0],
               [ 0,  0, 16, 17, 18, 19,  0],
               [ 0,  0, 20, 21, 22, 23,  0]]], dtype=ht.int64, device=cpu:0, split=0)
    Pad last 3 dimensions
    >>> e = ht.pad(b, ((2, 1), [1, 0], (2, 1)))
    DNDarray([[[ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0]],

              [[ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0]],

              [[ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  1,  2,  3,  0],
               [ 0,  0,  4,  5,  6,  7,  0],
               [ 0,  0,  8,  9, 10, 11,  0]],

              [[ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0, 12, 13, 14, 15,  0],
               [ 0,  0, 16, 17, 18, 19,  0],
               [ 0,  0, 20, 21, 22, 23,  0]],

              [[ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0]]], dtype=ht.int64, device=cpu:0, split=0)

`ravel(a: DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Return a flattened view of `a` if possible. A copy is returned otherwise.

    Parameters
    ----------
    a : DNDarray
        array to collapse

    Notes
    -----
    Returning a view of distributed data is only possible when `split != 0`. The returned DNDarray may be unbalanced.
    Otherwise, data must be communicated among processes, and `ravel` falls back to `flatten`.

    See Also
    --------
    :func:`flatten`

    Examples
    --------
    >>> a = ht.ones((2, 3), split=0)
    >>> b = ht.ravel(a)
    >>> a[0, 0] = 4
    >>> b
    DNDarray([4., 1., 1., 1., 1., 1.], dtype=ht.float32, device=cpu:0, split=0)

`redistribute(arr: DNDarray, lshape_map: torch.Tensor = None, target_map: torch.Tensor = None) ‑> heat.core.dndarray.DNDarray`
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

`repeat(a: Iterable, repeats: Iterable, axis: Optional[int] = None) ‑> heat.core.dndarray.DNDarray`
:   Creates a new `DNDarray` by repeating elements of array `a`. The output has
    the same shape as `a`, except along the given axis. If axis is None, this
    function returns a flattened `DNDarray`.

    Parameters
    ----------
    a : array_like (i.e. int, float, or tuple/ list/ np.ndarray/ ht.DNDarray of ints/floats)
        Array containing the elements to be repeated.
    repeats : int, or 1-dimensional/ DNDarray/ np.ndarray/ list/ tuple of ints
        The number of repetitions for each element, indicates broadcast if int or array_like of 1 element.
        In this case, the given value is broadcasted to fit the shape of the given axis.
        Otherwise, its length must be the same as a in the specified axis. To put it differently, the
        amount of repetitions has to be determined for each element in the corresponding dimension
        (or in all dimensions if axis is None).
    axis: int, optional
        The axis along which to repeat values. By default, use the flattened input array and return a flat output
        array.

    Examples
    --------
    >>> ht.repeat(3, 4)
    DNDarray([3, 3, 3, 3])

    >>> x = ht.array([[1, 2], [3, 4]])
    >>> ht.repeat(x, 2)
    DNDarray([1, 1, 2, 2, 3, 3, 4, 4])

    >>> x = ht.array([[1, 2], [3, 4]])
    >>> ht.repeat(x, [0, 1, 2, 0])
    DNDarray([2, 3, 3])

    >>> ht.repeat(x, [1, 2], axis=0)
    DNDarray([[1, 2],
            [3, 4],
            [3, 4]])

`reshape(a: DNDarray, *shape: Union[int, Tuple[int, ...]], **kwargs) ‑> heat.core.dndarray.DNDarray`
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

`resplit(arr: DNDarray, axis: Optional[int] = None) ‑> heat.core.dndarray.DNDarray`
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

`roll(x: DNDarray, shift: Union[int, Tuple[int]], axis: Optional[Union[int, Tuple[int]]] = None) ‑> heat.core.dndarray.DNDarray`
:   Rolls array elements along a specified axis. Array elements that roll beyond the last position are re-introduced at the first position.
    Array elements that roll beyond the first position are re-introduced at the last position.

    Parameters
    ----------
    x : DNDarray
        input array
    shift : Union[int, Tuple[int, ...]]
        number of places by which the elements are shifted. If 'shift' is a tuple, then 'axis' must be a tuple of the same size, and each of
        the given axes is shifted by the corrresponding element in 'shift'. If 'shift' is an `int` and 'axis' a `tuple`, then the same shift
        is used for all specified axes.
    axis : Optional[Union[int, Tuple[int, ...]]]
        axis (or axes) along which elements to shift. If 'axis' is `None`, the array is flattened, shifted, and then restored to its original shape.
        Default: `None`.

    Raises
    ------
    TypeError
        If 'shift' or 'axis' is not of type `int`, `list` or `tuple`.
    ValueError
        If 'shift' and 'axis' are tuples with different sizes.

    Examples
    --------
    >>> a = ht.arange(20).reshape((4, 5))
    >>> a
    DNDarray([[ 0,  1,  2,  3,  4],
          [ 5,  6,  7,  8,  9],
          [10, 11, 12, 13, 14],
          [15, 16, 17, 18, 19]], dtype=ht.int32, device=cpu:0, split=None)
    >>> ht.roll(a, 1)
    DNDarray([[19,  0,  1,  2,  3],
          [ 4,  5,  6,  7,  8],
          [ 9, 10, 11, 12, 13],
          [14, 15, 16, 17, 18]], dtype=ht.int32, device=cpu:0, split=None)
    >>> ht.roll(a, -1, 0)
    DNDarray([[ 5,  6,  7,  8,  9],
          [10, 11, 12, 13, 14],
          [15, 16, 17, 18, 19],
          [ 0,  1,  2,  3,  4]], dtype=ht.int32, device=cpu:0, split=None)

`rot90(m: DNDarray, k: int = 1, axes: Sequence[int, int] = (0, 1)) ‑> DNDarray`
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

`row_stack(arrays: Sequence[DNDarray, ...]) ‑> DNDarray`
:   Stack 1-D or 2-D `DNDarray`s as rows into a 2-D `DNDarray`.
    If the input arrays are 1-D, they will be stacked as rows. If they are 2-D,
    they will be concatenated along the first axis.

    Parameters
    ----------
    arrays : Sequence[DNDarrays, ...]
        Sequence of `DNDarray`s.

    Raises
    ------
    ValueError
        If arrays have more than 2 dimensions

    Notes
    -----
    All ``DNDarray``s in the sequence must have the same number of columns.
    All ``DNDarray``s must be split along the same axis!

    See Also
    --------
    :func:`column_stack`
    :func:`concatenate`
    :func:`hstack`
    :func:`stack`
    :func:`vstack`

    Examples
    --------
    >>> # 1-D tensors
    >>> a = ht.array([1, 2, 3])
    >>> b = ht.array([2, 3, 4])
    >>> ht.row_stack((a, b)).larray
    tensor([[1, 2, 3],
            [2, 3, 4]])
    >>> # 1-D and 2-D tensors
    >>> a = ht.array([1, 2, 3])
    >>> b = ht.array([[2, 3, 4], [5, 6, 7]])
    >>> c = ht.array([[7, 8, 9], [10, 11, 12]])
    >>> ht.row_stack((a, b, c)).larray
    tensor([[ 1,  2,  3],
            [ 2,  3,  4],
            [ 5,  6,  7],
            [ 7,  8,  9],
            [10, 11, 12]])
    >>> # distributed DNDarrays, 3 processes
    >>> a = ht.arange(10, split=0).reshape((2, 5))
    >>> b = ht.arange(5, 20, split=0).reshape((3, 5))
    >>> c = ht.arange(20, 40, split=0).reshape((4, 5))
    >>> ht.row_stack((a, b, c)).larray
    [0/2] tensor([[0, 1, 2, 3, 4],
    [0/2]         [5, 6, 7, 8, 9],
    [0/2]         [5, 6, 7, 8, 9]], dtype=torch.int32)
    [1/2] tensor([[10, 11, 12, 13, 14],
    [1/2]         [15, 16, 17, 18, 19],
    [1/2]         [20, 21, 22, 23, 24]], dtype=torch.int32)
    [2/2] tensor([[25, 26, 27, 28, 29],
    [2/2]         [30, 31, 32, 33, 34],
    [2/2]         [35, 36, 37, 38, 39]], dtype=torch.int32)
    >>> # distributed 1-D and 2-D DNDarrays, 3 processes
    >>> a = ht.arange(5, split=0)
    >>> b = ht.arange(5, 20, split=0).reshape((3, 5))
    >>> ht.row_stack((a, b)).larray
    [0/2] tensor([[0, 1, 2, 3, 4],
    [0/2]         [5, 6, 7, 8, 9]])
    [1/2] tensor([[10, 11, 12, 13, 14]])
    [2/2] tensor([[15, 16, 17, 18, 19]])

`shape(a: DNDarray) ‑> Tuple[int, ...]`
:   Returns the global shape of a (potentially distributed) `DNDarray` as a tuple.

    Parameters
    ----------
    a : DNDarray
        The input `DNDarray`.

`sort(a: DNDarray, axis: int = -1, descending: bool = False, out: Optional[DNDarray] = None)`
:   Sorts the elements of `a` along the given dimension (by default in ascending order) by their value.
    The sorting is not stable which means that equal elements in the result may have a different ordering than in the
    original array.
    Sorting where `axis==a.split` needs a lot of communication between the processes of MPI.
    Returns a tuple `(values, indices)` with the sorted local results and the indices of the elements in the original data

    Parameters
    ----------
    a : DNDarray
        Input array to be sorted.
    axis : int, optional
        The dimension to sort along.
        Default is the last axis.
    descending : bool, optional
        If set to `True`, values are sorted in descending order.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to `None`, a fresh array is allocated.

    Raises
    ------
    ValueError
        If `axis` is not consistent with the available dimensions.

    Examples
    --------
    >>> x = ht.array([[4, 1], [2, 3]], split=0)
    >>> x.shape
    (1, 2)
    (1, 2)
    >>> y = ht.sort(x, axis=0)
    >>> y
    (array([[2, 1]], array([[1, 0]]))
    (array([[4, 3]], array([[0, 1]]))
    >>> ht.sort(x, descending=True)
    (array([[4, 1]], array([[0, 1]]))
    (array([[3, 2]], array([[1, 0]]))

`split(x: DNDarray, indices_or_sections: Iterable, axis: int = 0) ‑> List[DNDarray, ...]`
:   Split a DNDarray into multiple sub-DNDarrays.
    Returns a list of sub-DNDarrays as copies of parts of `x`.

    Parameters
    ----------
    x : DNDarray
        DNDArray to be divided into sub-DNDarrays.
    indices_or_sections : int or 1-dimensional array_like (i.e. undistributed DNDarray, list or tuple)
        If `indices_or_sections` is an integer, N, the DNDarray will be divided into N equal DNDarrays along axis.
        If such a split is not possible, an error is raised.
        If `indices_or_sections` is a 1-D DNDarray of sorted integers, the entries indicate where along axis
        the array is split.
        For example, `indices_or_sections = [2, 3]` would, for `axis = 0`, result in

        - `x[:2]`
        - `x[2:3]`
        - `x[3:]`

        If an index exceeds the dimension of the array along axis, an empty sub-array is returned correspondingly.
    axis : int, optional
        The axis along which to split, default is 0.
        `axis` is not allowed to equal `x.split` if `x` is distributed.

    Raises
    ------
    ValueError
        If `indices_or_sections` is given as integer, but a split does not result in equal division.

    Warnings
    --------
    Though it is possible to distribute `x`, this function has nothing to do with the split
    parameter of a DNDarray.

    See Also
    --------
    :func:`dsplit`
    :func:`hsplit`
    :func:`vsplit`

    Examples
    --------
    >>> x = ht.arange(12).reshape((4, 3))
    >>> ht.split(x, 2)
        [ DNDarray([[0, 1, 2],
                    [3, 4, 5]]),
          DNDarray([[ 6,  7,  8],
                    [ 9, 10, 11]])]
    >>> ht.split(x, [2, 3, 5])
        [ DNDarray([[0, 1, 2],
                    [3, 4, 5]]),
          DNDarray([[6, 7, 8]]
          DNDarray([[ 9, 10, 11]]),
          DNDarray([])]
    >>> ht.split(x, [1, 2], 1)
        [DNDarray([[0],
                [3],
                [6],
                [9]]),
        DNDarray([[ 1],
                [ 4],
                [ 7],
                [10]],
        DNDarray([[ 2],
                [ 5],
                [ 8],
                [11]])]

`squeeze(x: DNDarray, axis: Union[int, Tuple[int, ...]] = None) ‑> heat.core.dndarray.DNDarray`
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

`stack(arrays: Sequence[DNDarray, ...], axis: int = 0, out: Optional[DNDarray] = None) ‑> DNDarray`
:   Join a sequence of `DNDarray`s along a new axis.
    The `axis` parameter specifies the index of the new axis in the dimensions of the result.
    For example, if `axis=0`, the arrays will be stacked along the first dimension; if `axis=-1`,
    they will be stacked along the last dimension. See Notes below for split semantics.

    Parameters
    ----------
    arrays : Sequence[DNDarrays, ...]
        Each DNDarray must have the same shape, must be split along the same axis, and must be balanced.
    axis : int, optional
        The axis in the result array along which the input arrays are stacked.
    out : DNDarray, optional
        If provided, the destination to place the result. The shape and split axis must be correct, matching
        that of what stack would have returned if no out argument were specified (see Notes below).

    Raises
    ------
    TypeError
        If arrays in sequence are not `DNDarray`s, or if their `dtype` attribute does not match.
    ValueError
        If `arrays` contains less than 2 `DNDarray`s.
    ValueError
        If the `DNDarray`s are of different shapes, or if they are split along different axes (`split` attribute).
    RuntimeError
        If the `DNDarrays` reside on different devices.

    Notes
    -----
    Split semantics: :func:`stack` requires that all arrays in the sequence be split along the same dimension.
    After stacking, the data are still distributed along the original dimension, however a new dimension has been added at `axis`,
    therefore:

    - if :math:`axis <= split`, output will be distributed along :math:`split+1`

    - if :math:`axis > split`, output will be distributed along `split`

    See Also
    --------
    :func:`column_stack`
    :func:`concatenate`
    :func:`hstack`
    :func:`row_stack`
    :func:`vstack`

    Examples
    --------
    >>> a = ht.arange(20).reshape((4, 5))
    >>> b = ht.arange(20, 40).reshape((4, 5))
    >>> ht.stack((a, b), axis=0).larray
    tensor([[[ 0,  1,  2,  3,  4],
             [ 5,  6,  7,  8,  9],
             [10, 11, 12, 13, 14],
             [15, 16, 17, 18, 19]],
            [[20, 21, 22, 23, 24],
             [25, 26, 27, 28, 29],
             [30, 31, 32, 33, 34],
             [35, 36, 37, 38, 39]]])
    >>> # distributed DNDarrays, 3 processes, stack along last dimension
    >>> a = ht.arange(20, split=0).reshape(4, 5)
    >>> b = ht.arange(20, 40, split=0).reshape(4, 5)
    >>> ht.stack((a, b), axis=-1).larray
    [0/2] tensor([[[ 0, 20],
    [0/2]          [ 1, 21],
    [0/2]          [ 2, 22],
    [0/2]          [ 3, 23],
    [0/2]          [ 4, 24]],
    [0/2]         [[ 5, 25],
    [0/2]          [ 6, 26],
    [0/2]          [ 7, 27],
    [0/2]          [ 8, 28],
    [0/2]          [ 9, 29]]])
    [1/2] tensor([[[10, 30],
    [1/2]          [11, 31],
    [1/2]          [12, 32],
    [1/2]          [13, 33],
    [1/2]          [14, 34]]])
    [2/2] tensor([[[15, 35],
    [2/2]          [16, 36],
    [2/2]          [17, 37],
    [2/2]          [18, 38],
    [2/2]          [19, 39]]])

`swapaxes(x: DNDarray, axis1: int, axis2: int) ‑> heat.core.dndarray.DNDarray`
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

`tile(x: DNDarray, reps: Sequence[int, ...]) ‑> DNDarray`
:   Construct a new DNDarray by repeating 'x' the number of times given by 'reps'.

    If 'reps' has length 'd', the result will have 'max(d, x.ndim)' dimensions:

    - if 'x.ndim < d', 'x' is promoted to be d-dimensional by prepending new axes.
    So a shape (3,) array is promoted to (1, 3) for 2-D replication, or shape (1, 1, 3)
    for 3-D replication (if this is not the desired behavior, promote 'x' to d-dimensions
    manually before calling this function);

    - if 'x.ndim > d', 'reps' will replicate the last 'd' dimensions of 'x', i.e., if
    'x.shape' is (2, 3, 4, 5), a 'reps' of (2, 2) will be expanded to (1, 1, 2, 2).

    Parameters
    ----------
    x : DNDarray
        Input

    reps : Sequence[ints,...]
        Repetitions

    Returns
    -------
    tiled : DNDarray
            Split semantics: if `x` is distributed, the tiled data will be distributed along the
            same dimension. Note that nominally `tiled.split != x.split` in the case where
            `len(reps) > x.ndim`.  See example below.

    Examples
    --------
    >>> x = ht.arange(12).reshape((4, 3)).resplit_(0)
    >>> x
    DNDarray([[ 0,  1,  2],
              [ 3,  4,  5],
              [ 6,  7,  8],
              [ 9, 10, 11]], dtype=ht.int32, device=cpu:0, split=0)
    >>> reps = (1, 2, 2)
    >>> tiled = ht.tile(x, reps)
    >>> tiled
    DNDarray([[[ 0,  1,  2,  0,  1,  2],
               [ 3,  4,  5,  3,  4,  5],
               [ 6,  7,  8,  6,  7,  8],
               [ 9, 10, 11,  9, 10, 11],
               [ 0,  1,  2,  0,  1,  2],
               [ 3,  4,  5,  3,  4,  5],
               [ 6,  7,  8,  6,  7,  8],
               [ 9, 10, 11,  9, 10, 11]]], dtype=ht.int32, device=cpu:0, split=1)

`topk(a: DNDarray, k: int, dim: int = -1, largest: bool = True, sorted: bool = True, out: Optional[Tuple[DNDarray, DNDarray]] = None) ‑> Tuple[heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray]`
:   Returns the :math:`k` highest entries in the array.
    (Not Stable for split arrays)

    Parameters
    ----------
    a: DNDarray
        Input data
    k: int
        Desired number of output items
    dim: int, optional
        Dimension along which to sort, per default the last dimension
    largest: bool, optional
        If `True`, return the :math:`k` largest items, otherwise return the :math:`k` smallest items
    sorted: bool, optional
        Whether to sort the output (descending if `largest` is `True`, else ascending)
    out: Tuple[DNDarray, ...], optional
        output buffer

    Examples
    --------
    >>> a = ht.array([1, 2, 3])
    >>> ht.topk(a, 2)
    (DNDarray([3, 2], dtype=ht.int64, device=cpu:0, split=None), DNDarray([2, 1], dtype=ht.int64, device=cpu:0, split=None))
    >>> a = ht.array([[1, 2, 3], [1, 2, 3]])
    >>> ht.topk(a, 2, dim=1)
    (DNDarray([[3, 2],
               [3, 2]], dtype=ht.int64, device=cpu:0, split=None),
     DNDarray([[2, 1],
               [2, 1]], dtype=ht.int64, device=cpu:0, split=None))
    >>> a = ht.array([[1, 2, 3], [1, 2, 3]], split=1)
    >>> ht.topk(a, 2, dim=1)
    (DNDarray([[3, 2],
               [3, 2]], dtype=ht.int64, device=cpu:0, split=1),
     DNDarray([[2, 1],
               [2, 1]], dtype=ht.int64, device=cpu:0, split=1))

`unfold(a: DNDarray, axis: int, size: int, step: int = 1)`
:   Returns a DNDarray which contains all slices of size `size` in the axis `axis`.
    Behaves like torch.Tensor.unfold for DNDarrays. [torch.Tensor.unfold](https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html)

    Parameters
    ----------
    a : DNDarray
        array to unfold
    axis : int
        axis in which unfolding happens
    size : int
        the size of each slice that is unfolded, must be greater than 1
    step : int
        the step between each slice, must be at least 1
    Example:
    ```
    >>> x = ht.arange(1., 8)
    >>> x
    DNDarray([1., 2., 3., 4., 5., 6., 7.], dtype=ht.float32, device=cpu:0, split=e)
    >>> ht.unfold(x, 0, 2, 1)
    DNDarray([[1., 2.],
              [2., 3.],
              [3., 4.],
              [4., 5.],
              [5., 6.],
              [6., 7.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.unfold(x, 0, 2, 2)
    DNDarray([[1., 2.],
              [3., 4.],
              [5., 6.]], dtype=ht.float32, device=cpu:0, split=None)
    ```
    Note
    ---------
    You have to make sure that every node has at least chunk size size-1 if the split axis of the array is the unfold axis.

`unique(a: DNDarray, sorted: bool = False, return_inverse: bool = False, axis: int = None) ‑> Tuple[heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray]`
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

`vsplit(x: DNDarray, indices_or_sections: Iterable) ‑> List[DNDarray, ...]`
:   Split array into multiple sub-DNDNarrays along the 1st axis (vertically/row-wise).
    Returns a list of sub-DNDarrays as copies of parts of ``x``.

    Parameters
    ----------
    x : DNDarray
        DNDArray to be divided into sub-DNDarrays.
    indices_or_sections : Iterable
        If `indices_or_sections` is an integer, N, the DNDarray will be divided into N equal DNDarrays along the 1st axis.

        If such a split is not possible, an error is raised.

        If `indices_or_sections` is a 1-D DNDarray of sorted integers, the entries indicate where along the 1st axis the array is split.

        If an index exceeds the dimension of the array along the 1st axis, an empty sub-DNDarray is returned correspondingly.


    Raises
    ------
    ValueError
        If `indices_or_sections` is given as integer, but a split does not result in equal division.

    Notes
    -----
    Please refer to the split documentation. :func:`hsplit` is equivalent to split with `axis=0`,
    the array is always split along the first axis regardless of the array dimension.

    See Also
    --------
    :func:`split`
    :func:`dsplit`
    :func:`hsplit`

    Examples
    --------
    >>> x = ht.arange(24).reshape((4, 3, 2))
    >>> ht.vsplit(x, 2)
        [DNDarray([[[ 0,  1],
                   [ 2,  3],
                   [ 4,  5]],
                  [[ 6,  7],
                   [ 8,  9],
                   [10, 11]]]),
        DNDarray([[[12, 13],
                   [14, 15],
                   [16, 17]],
                  [[18, 19],
                   [20, 21],
                   [22, 23]]])]
        >>> ht.vsplit(x, [1, 3])
        [DNDarray([[[0, 1],
                   [2, 3],
                   [4, 5]]]),
        DNDarray([[[ 6,  7],
                   [ 8,  9],
                   [10, 11]],
                  [[12, 13],
                   [14, 15],
                   [16, 17]]]),
        DNDarray([[[18, 19],
                   [20, 21],
                   [22, 23]]])]

`vstack(arrays: Sequence[DNDarray, ...]) ‑> DNDarray`
:   Stack arrays in sequence vertically (row wise).
    This is equivalent to concatenation along the first axis.
    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The :func:`concatenate` function provides more general
    stacking operations.

    Parameters
    ----------
    arrays : Sequence[DNDarray,...]
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length.

    Notes
    -----
    The split axis will be switched to 1 in the case that both elements are 1D and split=0

    See Also
    --------
    :func:`concatenate`
    :func:`stack`
    :func:`hstack`
    :func:`column_stack`
    :func:`row_stack`


    Examples
    --------
    >>> a = ht.array([1, 2, 3])
    >>> b = ht.array([2, 3, 4])
    >>> ht.vstack((a, b)).larray
    [0/1] tensor([[1, 2, 3],
    [0/1]         [2, 3, 4]])
    [1/1] tensor([[1, 2, 3],
    [1/1]         [2, 3, 4]])
    >>> a = ht.array([1, 2, 3], split=0)
    >>> b = ht.array([2, 3, 4], split=0)
    >>> ht.vstack((a, b)).larray
    [0/1] tensor([[1, 2],
    [0/1]         [2, 3]])
    [1/1] tensor([[3],
    [1/1]         [4]])
    >>> a = ht.array([[1], [2], [3]], split=0)
    >>> b = ht.array([[2], [3], [4]], split=0)
    >>> ht.vstack((a, b)).larray
    [0] tensor([[1],
    [0]         [2],
    [0]         [3]])
    [1] tensor([[2],
    [1]         [3],
    [1]         [4]])
