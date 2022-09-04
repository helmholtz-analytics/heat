"""
A Heat sub-namespace that conforms to the Python array API standard.
"""
import warnings

warnings.warn("The heat.array_api submodule is not fully implemented.", stacklevel=2)

__all__ = []

from ._constants import e, inf, nan, newaxis, pi

__all__ += ["e", "inf", "nan", "newaxis", "pi"]

from ._creation_functions import (
    arange,
    asarray,
    empty,
    empty_like,
    eye,
    from_dlpack,
    full,
    full_like,
    linspace,
    meshgrid,
    ones,
    ones_like,
    tril,
    triu,
    zeros,
    zeros_like,
)

__all__ += [
    "arange",
    "asarray",
    "empty",
    "empty_like",
    "eye",
    "full",
    "full_like",
    "linspace",
    "meshgrid",
    "ones",
    "ones_like",
    "tril",
    "triu",
    "zeros",
    "zeros_like",
]

from ._data_type_functions import (
    astype,
    broadcast_arrays,
    broadcast_to,
    can_cast,
    finfo,
    iinfo,
    result_type,
)

__all__ += [
    "astype",
    "broadcast_arrays",
    "broadcast_to",
    "can_cast",
    "finfo",
    "iinfo",
    "result_type",
]

from heat.core.devices import cpu

__all__ += ["cpu"]

import heat.core.devices

if hasattr(heat.core.devices, "gpu"):
    from heat.core.devices import gpu

    __all__ += ["gpu"]

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

from ._elementwise_functions import (
    abs,
    acos,
    acosh,
    add,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    bitwise_and,
    bitwise_left_shift,
    bitwise_invert,
    bitwise_or,
    bitwise_right_shift,
    bitwise_xor,
    ceil,
    cos,
    cosh,
    divide,
    equal,
    exp,
    expm1,
    floor,
    floor_divide,
    greater,
    greater_equal,
    isfinite,
    isinf,
    isnan,
    less,
    less_equal,
    log,
    log1p,
    log2,
    log10,
    logaddexp,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    multiply,
    negative,
    not_equal,
    positive,
    pow,
    remainder,
    round,
    sign,
    sin,
    sinh,
    square,
    sqrt,
    subtract,
    tan,
    tanh,
    trunc,
)

__all__ += [
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_invert",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "ceil",
    "cos",
    "cosh",
    "divide",
    "equal",
    "exp",
    "expm1",
    "floor",
    "floor_divide",
    "greater",
    "greater_equal",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "multiply",
    "negative",
    "not_equal",
    "positive",
    "pow",
    "remainder",
    "round",
    "sign",
    "sin",
    "sinh",
    "square",
    "sqrt",
    "subtract",
    "tan",
    "tanh",
    "trunc",
]

from . import linalg

__all__ += ["linalg"]

from .linalg import matmul, matrix_transpose, tensordot, vecdot

__all__ += ["matmul", "matrix_transpose", "tensordot", "vecdot"]

from ._manipulation_functions import (
    concat,
    expand_dims,
    flip,
    permute_dims,
    reshape,
    roll,
    squeeze,
    stack,
)

__all__ += ["concat", "expand_dims", "flip", "permute_dims", "reshape", "roll", "squeeze", "stack"]

from ._searching_functions import argmax, argmin, nonzero, where

__all__ += ["argmax", "argmin", "nonzero", "where"]

from ._set_functions import unique_inverse, unique_values

__all__ += ["unique_inverse", "unique_values"]

from ._sorting_functions import sort

__all__ += ["sort"]

from ._statistical_functions import max, mean, min, prod, std, sum, var

__all__ += ["max", "mean", "min", "prod", "std", "sum", "var"]

from ._utility_functions import all, any

__all__ += ["all", "any"]
