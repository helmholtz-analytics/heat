import torch

from typing import Union

__all__ = ["e", "Euler", "inf", "Inf", "Infty", "Infinity", "nan", "NaN", "pi"]

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
