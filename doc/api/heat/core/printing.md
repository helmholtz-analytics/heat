Module heat.core.printing
=========================
Allows to output DNDarrays to stdout.

Functions
---------

`get_printoptions() ‑> dict`
:   Returns the currently configured printing options as key-value pairs.

`global_printing() ‑> None`
:   For `DNDarray`s, the builtin `print` function will gather all of the data, format it
    then print it on ONLY rank 0.

    Returns
    -------
    None

    Examples
    --------
    >>> x = ht.arange(15 * 5, dtype=ht.float).reshape((15, 5)).resplit(0)
    >>> print(x)
    [0] DNDarray([[ 0.,  1.,  2.,  3.,  4.],
                 [ 5.,  6.,  7.,  8.,  9.],
                 [10., 11., 12., 13., 14.],
                 [15., 16., 17., 18., 19.],
                 [20., 21., 22., 23., 24.],
                 [25., 26., 27., 28., 29.],
                 [30., 31., 32., 33., 34.],
                 [35., 36., 37., 38., 39.],
                 [40., 41., 42., 43., 44.],
                 [45., 46., 47., 48., 49.],
                 [50., 51., 52., 53., 54.],
                 [55., 56., 57., 58., 59.],
                 [60., 61., 62., 63., 64.],
                 [65., 66., 67., 68., 69.],
                 [70., 71., 72., 73., 74.]], dtype=ht.float32, device=cpu:0, split=0)

`local_printing() ‑> None`
:   The builtin `print` function will now print the local PyTorch Tensor values for
    `DNDarrays` given as arguments.

    Examples
    --------
    >>> x = ht.ht.arange(15 * 5, dtype=ht.float).reshape((15, 5)).resplit(0)
    >>> ht.local_printing()
    [0/2]Printing options set to LOCAL. DNDarrays will print the local PyTorch Tensors
    >>> print(x)
    [0/2] [[ 0.,  1.,  2.,  3.,  4.],
    [0/2]  [ 5.,  6.,  7.,  8.,  9.],
    [0/2]  [10., 11., 12., 13., 14.],
    [0/2]  [15., 16., 17., 18., 19.],
    [0/2]  [20., 21., 22., 23., 24.]]
    [1/2] [[25., 26., 27., 28., 29.],
    [1/2]  [30., 31., 32., 33., 34.],
    [1/2]  [35., 36., 37., 38., 39.],
    [1/2]  [40., 41., 42., 43., 44.],
    [1/2]  [45., 46., 47., 48., 49.]]
    [2/2] [[50., 51., 52., 53., 54.],
    [2/2]  [55., 56., 57., 58., 59.],
    [2/2]  [60., 61., 62., 63., 64.],
    [2/2]  [65., 66., 67., 68., 69.],
    [2/2]  [70., 71., 72., 73., 74.]]

`print0(*args, **kwargs) ‑> None`
:   Wraps the builtin `print` function in such a way that it will only run the command on
    rank 0. If this is called with DNDarrays and local printing, only the data local to
    process 0 is printed. For more information see the examples.

    This function is also available as a builtin when importing heat.

    Examples
    --------
    >>> x = ht.arange(15 * 5, dtype=ht.float).reshape((15, 5)).resplit(0)
    >>> # GLOBAL PRINTING
    >>> ht.print0(x)
    [0] DNDarray([[ 0.,  1.,  2.,  3.,  4.],
                 [ 5.,  6.,  7.,  8.,  9.],
                 [10., 11., 12., 13., 14.],
                 [15., 16., 17., 18., 19.],
                 [20., 21., 22., 23., 24.],
                 [25., 26., 27., 28., 29.],
                 [30., 31., 32., 33., 34.],
                 [35., 36., 37., 38., 39.],
                 [40., 41., 42., 43., 44.],
                 [45., 46., 47., 48., 49.],
                 [50., 51., 52., 53., 54.],
                 [55., 56., 57., 58., 59.],
                 [60., 61., 62., 63., 64.],
                 [65., 66., 67., 68., 69.],
                 [70., 71., 72., 73., 74.]], dtype=ht.float32, device=cpu:0, split=0)
    >>> ht.local_printing()
    [0/2] Printing options set to LOCAL. DNDarrays will print the local PyTorch Tensors
    >>> print0(x)
    [0/2] [[ 0.,  1.,  2.,  3.,  4.],
    [0/2]  [ 5.,  6.,  7.,  8.,  9.],
    [0/2]  [10., 11., 12., 13., 14.],
    [0/2]  [15., 16., 17., 18., 19.],
    [0/2]  [20., 21., 22., 23., 24.]], device: cpu:0, split: 0

`set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)`
:   Configures the printing options. List of items shamelessly taken from NumPy and PyTorch (thanks guys!).

    Parameters
    ----------
    precision : int, optional
        Number of digits of precision for floating point output (default=4).
    threshold : int, optional
        Total number of array elements which trigger summarization rather than full `repr` string (default=1000).
    edgeitems : int, optional
        Number of array items in summary at beginning and end of each dimension (default=3).
    linewidth : int, optional
        The number of characters per line for the purpose of inserting line breaks (default = 80).
    profile : str, optional
        Sane defaults for pretty printing. Can override with any of the above options. Can be any one of `default`,
        `short`, `full`.
    sci_mode : bool, optional
        Enable (True) or disable (False) scientific notation. If None (default) is specified, the value is automatically
        inferred by HeAT.
