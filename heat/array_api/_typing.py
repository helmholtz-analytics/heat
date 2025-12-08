from typing import Union
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

from heat.core.devices import Device, cpu

try:
    from heat.core.devices import gpu
except ImportError:
    pass

Dtype = Union[
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
]
