import torch

from typing import Union

__all__ = ["e", "Euler", "inf", "Inf", "Infty", "Infinity", "nan", "NaN", "pi", "sanitize_infinity"]

INF = float("inf")
"""
:math:`\\infty`, infinity
"""
NAN = float("nan")
"""
Not a number
"""
NINF = -float("inf")
"""
:math:`-\\infty` Negative infinity
"""
PI = 3.141592653589793
"""
:math:`\\pi`, Archimedes' constant
"""
E = 2.718281828459045
"""
:math:`e`, Euler's number
"""

# aliases
inf = Inf = Infty = Infinity = INF
nan = NaN = NAN
pi = PI
e = Euler = E


def sanitize_infinity(dtype: torch.dtype) -> Union[int, float]:
    """
    Returns largest possible value for the specified datatype.

    Parameters:
    -----------
    dtype: torch.dtype
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
