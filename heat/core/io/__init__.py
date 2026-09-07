"""Enables parallel I/O with data on disk."""

from . import csv, hdf5, io, netcdf, npy, utils, zarr  # noqa: F401
from ._registry import (  # noqa: F401
    FormatSpec,
    register_format,
    registered_extensions,
    registered_formats,
)
from ..dndarray import DNDarray
from .io import load, save
from .npy import load_npy_from_path
from .utils import size_from_slice  # noqa: F401

# Format-specific functions are only defined when the optional dependency they need is
# installed, so each format module is star-imported and its own `__all__` decides what it
# contributes -- `ht.load_hdf5` exists exactly when h5py does, as before.
from .csv import *  # noqa: F403
from .hdf5 import *  # noqa: F403
from .netcdf import *  # noqa: F403
from .zarr import *  # noqa: F403

__all__ = ["load", "save", "load_npy_from_path"]
for _module in (csv, hdf5, netcdf, zarr):
    __all__.extend(name for name in _module.__all__ if name not in __all__)
del _module


def _bind_save_method(format_name: str, saver) -> None:
    """
    Attaches ``DNDarray.save_<format_name>`` as a thin forwarder to ``saver``.

    Generated rather than written by hand so that every registered format gets a method,
    and so that no argument can be dropped on the way through.
    """

    def method(self, path, *args, **kwargs):
        return saver(self, path, *args, **kwargs)

    method.__doc__ = saver.__doc__
    method.__name__ = f"save_{format_name}"
    method.__qualname__ = f"DNDarray.save_{format_name}"
    setattr(DNDarray, f"save_{format_name}", method)


for _format_name, _spec in registered_formats().items():
    if _spec.saver is not None:
        _bind_save_method(_format_name, _spec.saver)
del _format_name, _spec
