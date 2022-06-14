"""
A Heat sub-namespace that conforms to the Python array API standard.
"""
import warnings

warnings.warn("The heat.array_api submodule is not fully implemented.", stacklevel=2)

__all__ = []

from ._dtypes import (
    int8,
    int16,
    int32,
    int64,
    uint8,
    # uint16,
    # uint32,
    # uint64,
    float32,
    float64,
    bool,
)

__all__ += [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    # "uint16",
    # "uint32",
    # "uint64",
    "float32",
    "float64",
    "bool",
]
