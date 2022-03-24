"""
This module defines constants used in HeAT.
"""

import torch

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
"""IEEE 754 floating point representation of (positive) infinity (:math:`\\infty`)."""
Inf = INF
"""IEEE 754 floating point representation of (positive) infinity (:math:`\\infty`)."""
Infty = INF
"""IEEE 754 floating point representation of (positive) infinity (:math:`\\infty`)."""
Infinity = INF
"""IEEE 754 floating point representation of (positive) infinity (:math:`\\infty`)."""
nan = NAN
"""IEEE 754 floating point representation of Not a Number (NaN)."""
NaN = NAN
"""IEEE 754 floating point representation of Not a Number (NaN)."""
pi = PI
"""Archimedes' constant (:math:`\\pi`)."""
e = E
"""Euler's number, Euler's constant (:math:`e`)."""
Euler = E
"""Euler's number, Euler's constant (:math:`e`)."""
