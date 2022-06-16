"""
A Heat sub-namespace that conforms to the Python array API standard.
"""
import warnings

warnings.warn("The heat.array_api submodule is not fully implemented.", stacklevel=2)

__all__ = []

from ._constants import e, inf, nan, newaxis, pi

__all__ += ["e", "inf", "nan", "newaxis", "pi"]

from ._creation_functions import asarray, full, zeros

__all__ += ["asarray", "full", "zeros"]

from ._data_type_functions import astype, finfo, iinfo

__all__ += ["astype", "finfo", "iinfo"]

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

from ._elementwise_functions import equal, isfinite, isinf, isnan

__all__ += ["equal", "isfinite", "isinf", "isnan"]

from ._manipulation_functions import reshape

__all__ += ["reshape"]

from ._utility_functions import all

__all__ += ["all"]
