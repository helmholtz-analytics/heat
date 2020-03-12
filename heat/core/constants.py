# Definition of HeAT Constants
import torch

__all__ = ["e", "Euler", "inf", "Inf", "Infty", "Infinity", "nan", "NaN", "pi"]

INF = float("inf")
NAN = float("nan")
NINF = -float("inf")
PI = 3.141592653589793
E = 2.718281828459045

# aliases
inf = Inf = Infty = Infinity = INF
nan = NaN = NAN
pi = PI
e = Euler = E


def sanitize_infinity(dtype):
    """
    Returns largest possible value for the specified dtype.

    Parameters:
    -----------
    dtype: torch dtype

    Returns:
    --------
    large_enough: largest possible value for the given dtype
    """
    if dtype is torch.int8:
        large_enough = (1 << 7) - 1
    elif dtype is torch.int16:
        large_enough = (1 << 15) - 1
    elif dtype is torch.int32:
        large_enough = (1 << 31) - 1
    elif dtype is torch.int64:
        large_enough = (1 << 63) - 1
    else:
        large_enough = float("inf")

    return large_enough
