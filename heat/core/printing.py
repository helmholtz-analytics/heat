import copy
import torch

from typing import Dict

__all__ = ["get_printoptions", "set_printoptions"]


# set the default printing width to a 120
_DEFAULT_LINEWIDTH = 120
torch.set_printoptions(profile="default", linewidth=_DEFAULT_LINEWIDTH)


def get_printoptions() -> Dict:
    """
    Returns the currently configured printing options.
    """
    return copy.copy(torch._tensor_str.PRINT_OPTS.__dict__)


def set_printoptions(
    precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None
):
    """
    Configures the printing options. List of items shamelessly taken from NumPy and PyTorch (thanks guys!).

    Parameters
    ----------
    precision: int
        Number of digits of precision for floating point output (default=4).
    threshold: int
        Total number of array elements which trigger summarization rather than full `repr` string (default=1000).
    edgeitems: int
        Number of array items in summary at beginning and end of each dimension (default=3).
    linewidth: int
        The number of characters per line for the purpose of inserting line breaks (default = 80).
    profile: str
        Sane defaults for pretty printing. Can override with any of the above options. Can be any one of `default`,
        `short`, `full`.
    sci_mode: bool
        Enable (True) or disable (False) scientific notation. If None (default) is specified, the value is automatically
        inferred by HeAT.
    """
    torch.set_printoptions(precision, threshold, edgeitems, linewidth, profile, sci_mode)

    # HeAT profiles will print a bit wider than PyTorch does
    if profile == "default" and linewidth is None:
        torch._tensor_str.PRINT_OPTS.linewidth = _DEFAULT_LINEWIDTH
    elif profile == "short" and linewidth is None:
        torch._tensor_str.PRINT_OPTS.linewidth = _DEFAULT_LINEWIDTH
    elif profile == "full" and linewidth is None:
        torch._tensor_str.PRINT_OPTS.linewidth = _DEFAULT_LINEWIDTH


__PREFIX = "DNDarray"
__INDENT = len(__PREFIX)


def __repr__(dndarray) -> str:
    """
    Computes a printable representation of the passed DNDarray.

    Parameters
    ----------
    dndarray: DNDarray
        The array for which to obtain the corresponding string
    """
    return "{}({}, dtype=ht.{}, device={}, split={})".format(
        __PREFIX,
        _tensor_str(dndarray, __INDENT + 1),
        dndarray.dtype.__name__,
        dndarray.device,
        dndarray.split,
    )


def _torch_data(dndarray, summarize) -> torch.Tensor:
    """
    Extracts the data to be printed from the DNDarray in form of a torch tensor and returns it.

    Parameters
    ----------
    dndarray: DNDarray
        The HeAT DNDarray to be printed.
    summarize: bool
        Flag indicating whether to print the full data or summarized, i.e. ellipsed, version of the data.
    """
    # data is not split, we can use it as is
    if dndarray.split is None:
        data = dndarray._DNDarray__array
    # split, but no summary required, we collect it
    elif not summarize:
        data = dndarray.copy().resplit_(None)._DNDarray__array
    # split, but summarized, collect the slices from all nodes and pass it on
    else:
        edgeitems = torch._tensor_str.PRINT_OPTS.edgeitems
        double_items = 2 * edgeitems
        ndims = dndarray.ndim
        data = dndarray._DNDarray__array

        for i in range(ndims):
            # skip over dimensions that are smaller than twice the number of edge items to display
            if dndarray.gshape[i] <= double_items:
                continue

            # non-split dimension, can slice locally
            if i != dndarray.split:
                data = torch.stack([data[: edgeitems + 1], data[-edgeitems:]], dim=i)
            # split-dimension , need to respect the global offset
            elif i == dndarray.split and dndarray.gshape[i] > double_items:
                offset, _, _ = dndarray.comm.chunk(dndarray.gshape, i)

                if offset < edgeitems:
                    data = data[: edgeitems + 1 - offset]
                elif dndarray.gshape[i] - edgeitems < offset:
                    edge_start = dndarray.gshape[i] - edgeitems
                    local_end = offset + dndarray.lshape[i]
                    data = data[local_end - edge_start :]

        # TODO: combine the slice parts globally alltoallv
        data = dndarray.comm.alltoallv()

    return data


def _tensor_str(dndarray, indent) -> str:
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

    return torch._tensor_str._tensor_str_with_formatter(torch_data, indent, formatter, summarize)
