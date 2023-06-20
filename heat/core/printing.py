"""Allows to output DNDarrays to stdout."""

import builtins
import copy
import torch
from .communication import MPI_WORLD

from .dndarray import DNDarray

__all__ = ["get_printoptions", "global_printing", "local_printing", "print0", "set_printoptions"]


# set the default printing width to a 120
_DEFAULT_LINEWIDTH = 120
torch.set_printoptions(profile="default", linewidth=_DEFAULT_LINEWIDTH)
LOCAL_PRINT = False

# printing
__PREFIX = "DNDarray"
__INDENT = len(__PREFIX)


def get_printoptions() -> dict:
    """
    Returns the currently configured printing options as key-value pairs.
    """
    return copy.copy(torch._tensor_str.PRINT_OPTS.__dict__)


def local_printing() -> None:
    """
    The builtin `print` function will now print the local PyTorch Tensor values for
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
    """
    global LOCAL_PRINT
    LOCAL_PRINT = True
    print0("Printing options set to LOCAL. DNDarrays will print the local PyTorch Tensors")


def global_printing() -> None:
    """
    For `DNDarray`s, the builtin `print` function will gather all of the data, format it
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
    """
    global LOCAL_PRINT
    if not LOCAL_PRINT:
        return
    LOCAL_PRINT = False
    print0(
        "Printing options set to GLOBAL. DNDarrays will be collected on process 0 before printing"
    )


def print0(*args, **kwargs) -> None:
    """
    Wraps the builtin `print` function in such a way that it will only run the command on
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
    """
    if not LOCAL_PRINT:
        args = list(args)
        for i in range(len(args)):
            if isinstance(args[i], DNDarray):
                args[i] = __str__(args[i])
        args = tuple(args)
    if MPI_WORLD.rank == 0:
        print(*args, **kwargs)


builtins.print0 = print0


def set_printoptions(
    precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None
):
    """
    Configures the printing options. List of items shamelessly taken from NumPy and PyTorch (thanks guys!).

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
    """
    torch.set_printoptions(precision, threshold, edgeitems, linewidth, profile, sci_mode)

    # HeAT profiles will print a bit wider than PyTorch does
    if (
        (profile == "default" and linewidth is None)
        or (profile == "short" and linewidth is None)
        or (profile == "full" and linewidth is None)
    ):
        torch._tensor_str.PRINT_OPTS.linewidth = _DEFAULT_LINEWIDTH


def __str__(dndarray) -> str:
    """
    Computes a printable representation of the passed DNDarray.

    Parameters
    ----------
    dndarray : DNDarray
        The array for which to obtain the corresponding string
    """
    if LOCAL_PRINT:
        return (
            torch._tensor_str._tensor_str(dndarray.larray, 0)
            + f", device: {dndarray.device}, split: {dndarray.split}"
        )
    tensor_string = _tensor_str(dndarray, __INDENT + 1)
    if dndarray.comm.rank != 0:
        return ""

    return (
        f"{__PREFIX}({tensor_string}, dtype=ht.{dndarray.dtype.__name__}, "
        f"device={dndarray.device}, split={dndarray.split})"
    )


def _torch_data(dndarray, summarize) -> DNDarray:
    """
    Extracts the data to be printed from the DNDarray in form of a torch tensor and returns it.

    Parameters
    ----------
    dndarray : DNDarray
        The HeAT DNDarray to be printed.
    summarize : bool
        Flag indicating whether to print the full data or summarized, i.e. ellipsed, version of the data.
    """
    if not dndarray.is_balanced():
        dndarray.balance_()
    # data is not split, we can use it as is
    if dndarray.split is None or dndarray.comm.size == 1:
        data = dndarray.larray
    # split, but no summary required, we collect it
    elif not summarize:
        data = dndarray.copy().resplit_(None).larray
    # split, but summarized, collect the slices from all nodes and pass it on
    else:
        edgeitems = torch._tensor_str.PRINT_OPTS.edgeitems
        double_items = 2 * edgeitems
        ndims = dndarray.ndim
        data = dndarray.larray

        for i in range(ndims):
            # skip over dimensions that are smaller than twice the number of edge items to display
            if dndarray.gshape[i] <= double_items:
                continue

            # non-split dimension, can slice locally
            if i != dndarray.split:
                start_tensor = torch.index_select(data, i, torch.arange(edgeitems + 1))
                end_tensor = torch.index_select(
                    data, i, torch.arange(dndarray.lshape[i] - edgeitems, dndarray.lshape[i])
                )
                data = torch.cat([start_tensor, end_tensor], dim=i)
            # split-dimension , need to respect the global offset
            elif i == dndarray.split and dndarray.gshape[i] > double_items:
                offset, _, _ = dndarray.comm.chunk(dndarray.gshape, i)

                if offset < edgeitems + 1:
                    end = min(dndarray.lshape[i], edgeitems + 1 - offset)
                    data = torch.index_select(data, i, torch.arange(end))
                elif dndarray.gshape[i] - edgeitems < offset - dndarray.lshape[i]:
                    global_start = dndarray.gshape[i] - edgeitems
                    data = torch.index_select(
                        data, i, torch.arange(max(0, global_start - offset), dndarray.lshape[i])
                    )
        # exchange data
        received = dndarray.comm.gather(data)
        if dndarray.comm.rank == 0:
            # concatenate data along the split axis
            data = torch.cat(received, dim=dndarray.split)

    return data


def _tensor_str(dndarray, indent: int) -> str:
    """
    Computes a string representation of the passed DNDarray.

    Parameters
    ----------
    dndarray: DNDarray
        The array for which to obtain the corresponding string
    indent: int
        The number of spaces the array content is indented.
    """
    elements = dndarray.gnumel
    if elements == 0:
        return "[]"

    # we will recycle torch's printing features here
    # to do so, we slice up the torch data and forward it to torch internal printing mechanism
    summarize = elements > get_printoptions()["threshold"]
    torch_data = _torch_data(dndarray, summarize)
    formatter = torch._tensor_str._Formatter(torch_data)

    return torch._tensor_str._tensor_str_with_formatter(torch_data, indent, summarize, formatter)
