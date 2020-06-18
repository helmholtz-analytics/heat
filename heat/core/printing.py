import copy
import math
from typing import Dict

__all__ = ["get_printoptions", "set_printoptions"]


class __PrintingOptions:
    def __init__(self, precision, threshold, edgeitems, linewidth, sci_mode):
        self.precision = precision
        self.threshold = threshold
        self.edgeitems = edgeitems
        self.linewidth = linewidth
        self.sci_mode = sci_mode


# define standard printing profiles
__DEFAULT_OPTIONS = __PrintingOptions(4, 1000, 3, 80, None)
__SHORT_OPTIONS = __PrintingOptions(2, 1000, 2, 80, None)
__FULL_OPTIONS = __PrintingOptions(4, math.inf, 3, 80, None)

# copy over the default profile
__PRINT_OPTIONS = copy.copy(__DEFAULT_OPTIONS)


def get_printoptions() -> Dict:
    """
    Returns the currently configured printing options.
    """
    return {
        "precision": __PRINT_OPTIONS.precision,
        "threshold": __PRINT_OPTIONS.threshold,
        "edgeitems": __PRINT_OPTIONS.edgeitems,
        "linewidth": __PRINT_OPTIONS.linewidth,
        "sci_mode": __PRINT_OPTIONS.sci_mode,
    }


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

    Raises
    ------
    TypeError
        When any parameter except `profile` cannot be interpreted as an integer or bool for `sci_mode` respectively.
    ValueError
        If the profile string is not understood.
    """
    global __PRINT_OPTIONS

    if profile == "default":
        __PRINT_OPTIONS = copy.copy(__DEFAULT_OPTIONS)
    elif profile == "short":
        __PRINT_OPTIONS = copy.copy(__SHORT_OPTIONS)
    elif profile == "full":
        __PRINT_OPTIONS = copy.copy(__FULL_OPTIONS)
    else:
        raise ValueError(
            f"Expected 'profile' to be one of 'default', 'short' or 'full', but was {profile}"
        )

    if precision is not None:
        __PRINT_OPTIONS.precision = max(0, int(precision))
    if threshold is not None:
        __PRINT_OPTIONS.threshold = int(threshold)
    if edgeitems is not None:
        __PRINT_OPTIONS.edgeitems = max(1, int(edgeitems))
    if linewidth is not None:
        __PRINT_OPTIONS.linewidth = max(1, int(linewidth))

    __PRINT_OPTIONS.sci_mode = bool(sci_mode) if sci_mode is not None else None


__PREFIX = "DNDarray"
__INDENT = len(__PREFIX)


def __repr__(dndarray) -> str:
    """
    Computes a printable representation of the passed DNDarray.

    dndarray: DNDarray
        The array for which to obtain the corresponding string
    """
    return "{}({}, device={}, split={})".format(
        __PREFIX, __str__(dndarray, __INDENT), dndarray.device, dndarray.split
    )


def __str__(dndarray, indent=0) -> str:
    """
    Computes a string representation of the passed DNDarray.

    dndarray: DNDarray
        The array for which to obtain the corresponding string
    indent: int
        The number of spaces the array content is indented.
    """
    return ""
