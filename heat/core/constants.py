import torch

from typing import Union

__all__ = ["e", "Euler", "inf", "Inf", "Infty", "Infinity", "nan", "NaN", "pi", "sanitize_infinity"]

# infinity
INF = float("inf")
# Not a number
NAN = float("nan")
# Negative infinity
NINF = -float("inf")
# Archimedes' constant
PI = 3.141592653589793
# Euler's number
E = 2.718281828459045


# aliases
inf = INF
""":math:`\\infty`, infinity"""
Inf = INF
""":math:`\\infty`, infinity"""
Infty = INF
""":math:`\\infty`, infinity"""
Infinity = INF
""":math:`\\infty`, infinity"""
nan = NAN
"""Not a number"""
NaN = NAN
"""Not a number"""
pi = PI
""":math:`\\pi`, Archimedes' constant"""
e = E
""":math:`e`, Euler's number"""
Euler = E
""":math:`e`, Euler's number"""


def sanitize_infinity(dtype: torch.dtype) -> Union[int, float]:
    """
    Return largest possible value for the specified datatype.

    Parameters
    -----------
    dtype: torch.dtype
        The specified datatype
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
