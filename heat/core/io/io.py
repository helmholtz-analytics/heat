"""Enables parallel I/O with data on disk."""

from __future__ import annotations

from functools import reduce
import glob

import operator
import os.path
from math import log10
from pathlib import Path
import numpy as np
import torch
import warnings
import fnmatch

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

from .. import devices
from .. import factories
from .. import types

from ..communication import Communication, MPI, MPI_WORLD, sanitize_comm
from ..dndarray import DNDarray
from ..manipulations import hsplit, vsplit
from ..statistics import max as smax, min as smin
from ..stride_tricks import sanitize_axis
from ..types import datatype
from ._registry import loader_for_path, register_format, saver_for_path
from .utils import sanitize_path

__VALID_WRITE_MODES = frozenset(["w", "a", "r+"])

__all__ = [
    "load",
    "save",
]


def size_from_slice(size: int, s: slice) -> Tuple[int, int]:
    """
    Determines the size of a slice object.

    Parameters
    ----------
    size: int
        The size of the array the slice object is applied to.
    s : slice
        The slice object to determine the size of.

    Returns
    -------
    int
        The size of the sliced object.
    int
        The start index of the slice object.
    """
    new_range = range(size)[s]
    return len(new_range), new_range.start if len(new_range) > 0 else 0


def load(
    path: str, *args: Optional[List[object]], **kwargs: Optional[Dict[str, object]]
) -> DNDarray:
    """
    Attempts to load data from a file stored on disk. Attempts to auto-detect the file format by determining the
    extension. Supports at least CSV files, HDF5 and netCDF4 are additionally possible if the corresponding libraries
    are installed.

    Parameters
    ----------
    path : str
        Path to the file to be read.
    args : list, optional
        Additional options passed to the particular functions.
    kwargs : dict, optional
        Additional options passed to the particular functions.

    Raises
    ------
    ValueError
        If the file extension is not understood or known.
    RuntimeError
        If the optional dependency for a file extension is not available.

    Examples
    --------
    >>> ht.load("data.h5", dataset="DATA")
    DNDarray([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.load("data.nc", variable="DATA")
    DNDarray([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.load("my_data.zarr", variable="RECEIVER_1/DATA")
    DNDarray([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981], dtype=ht.float32, device=cpu:0, split=0)
    >>> ht.load("my_data.zarr", variable="RECEIVER_*/DATA")
    DNDarray([[ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981],
                [ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981],
                [ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981]], dtype=ht.float32, device=cpu:0, split=0)

    See Also
    --------
    :func:`load_csv` : Loads data from a CSV file.
    :func:`load_csv_from_folder` : Loads multiple .csv files into one DNDarray which will be returned.
    :func:`load_hdf5` : Loads data from an HDF5 file.
    :func:`load_netcdf` : Loads data from a NetCDF4 file.
    :func:`load_npy_from_path` : Loads multiple .npy files into one DNDarray which will be returned.
    :func:`load_zarr` : Loads zarr-Format into DNDarray which will be returned.

    """
    path = sanitize_path(path)
    return loader_for_path(path)(path, *args, **kwargs)


def save(
    data: DNDarray, path: str, *args: Optional[List[object]], **kwargs: Optional[Dict[str, object]]
):
    """
    Attempts to save data from a :class:`~heat.core.dndarray.DNDarray` to disk. An auto-detection based on the file
    format extension is performed.

    Parameters
    ----------
    data : DNDarray
        The array holding the data to be stored
    path : str
        Path to the file to be stored.
    args : list, optional
        Additional options passed to the particular functions.
    kwargs : dict, optional
        Additional options passed to the particular functions.

    Raises
    ------
    ValueError
        If the file extension is not understood or known.
    RuntimeError
        If the optional dependency for a file extension is not available.

    Examples
    --------
    >>> x = ht.arange(100, split=0)
    >>> ht.save(x, "data.h5", "DATA", mode="a")
    """
    path = sanitize_path(path)
    return saver_for_path(path)(data, path, *args, **kwargs)


DNDarray.save = lambda self, path, *args, **kwargs: save(self, path, *args, **kwargs)
DNDarray.save.__doc__ = save.__doc__


# --------------------------------------------------------------------------------------
# Format registration
# --------------------------------------------------------------------------------------
# Formats register even when their optional dependency is missing, with `loader`/`saver`
# left as None, so that `load`/`save` report the missing dependency rather than claiming
# the extension is unknown. Directory-based loaders (`load_npy_from_path`,
# `load_csv_from_folder`, `load_multiple_hdf5`) are deliberately not registered: they take
# a directory, so there is no extension to dispatch on.
