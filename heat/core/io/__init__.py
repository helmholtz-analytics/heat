"""Enables parallel I/O with data on disk."""

from . import csv, hdf5, io, netcdf, npy, utils, zarr  # noqa: F401
from ._registry import (  # noqa: F401
    FormatSpec,
    register_format,
    registered_extensions,
    registered_formats,
)
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
