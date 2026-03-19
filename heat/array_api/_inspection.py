import heat.core.devices as ht_devices

from types import SimpleNamespace
from heat.core.types import (
    bool,
    complex64,
    complex128,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
)


def __array_namespace_info__():
    """Returns a namespace with Array API namespace inspection utilities."""
    info = SimpleNamespace()
    info.capabilities = capabilities
    info.default_device = default_device
    info.default_dtypes = default_dtypes
    info.devices = devices
    info.dtypes = dtypes
    return info


def capabilities():
    """Returns a dictionary of array library capabilities."""
    return {"boolean indexing": True, "data-dependent shapes": True, "max dimensions": 64}


def default_device():
    """Returns the default device."""
    return ht_devices.get_device()


def default_dtypes(*, device=None):
    """Returns a dictionary containing default data types."""
    if device is None:
        device = default_device()

    if not isinstance(device, ht_devices.Device):
        raise ValueError(f"Device not understood: {device}")

    if device == ht_devices.cpu:
        return {
            "real floating": float32,
            "complex floating": complex64,
            "integral": int64,
            "indexing": int64,
        }

    if device == ht_devices.gpu:
        return {
            "real floating": float32,
            "complex floating": complex64,
            "integral": int64,
            "indexing": int64,
        }

    raise ValueError(f"Unsupported device: {device}")


def devices():
    """Returns a list of supported devices which are available at runtime."""
    if hasattr(ht_devices, "gpu"):
        return (ht_devices.cpu, ht_devices.gpu)
    else:
        return (ht_devices.cpu,)


def dtypes(*, device=None, kind=None):
    """Returns a dictionary of supported Array API data types"""
    if device is None:
        device = default_device()

    if not isinstance(device, ht_devices.Device):
        raise ValueError(f"Device not understood: {device}")

    if kind is None:
        return {
            "bool": bool,
            "int8": int8,
            "int16": int16,
            "int32": int32,
            "int64": int64,
            "uint8": uint8,
            "float32": float32,
            "float64": float64,
            "complex64": complex64,
            "complex128": complex128,
        }
    if kind == "bool":
        return {
            "bool": bool,
        }
    if kind == "signed integer":
        return {
            "int8": int8,
            "int16": int16,
            "int32": int32,
            "int64": int64,
        }
    if kind == "unsigned integer":
        return {
            "uint8": uint8,
        }
    if kind == "integral":
        return {
            "int8": int8,
            "int16": int16,
            "int32": int32,
            "int64": int64,
            "uint8": uint8,
        }
    if kind == "real floating":
        return {
            "float32": float32,
            "float64": float64,
        }
    if kind == "complex floating":
        return {
            "complex64": complex64,
            "complex128": complex128,
        }
    if kind == "numeric":
        return {
            "int8": int8,
            "int16": int16,
            "int32": int32,
            "int64": int64,
            "uint8": uint8,
            "float32": float32,
            "float64": float64,
            "complex64": complex64,
            "complex128": complex128,
        }
    if isinstance(kind, tuple):
        res = {}
        for k in kind:
            res |= dtypes(device=device, kind=k)
        return res
    raise ValueError(f"Unsupported kind: {kind}")
