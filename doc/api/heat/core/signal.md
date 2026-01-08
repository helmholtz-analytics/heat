Module heat.core.signal
=======================
Provides a collection of signal-processing operations

Functions
---------

`convolve(a: heat.core.dndarray.DNDarray, v: heat.core.dndarray.DNDarray, mode: str = 'full', stride: int = 1) ‑> heat.core.dndarray.DNDarray`
:   Returns the discrete, linear convolution of two one-dimensional `DNDarray`s or scalars.
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
