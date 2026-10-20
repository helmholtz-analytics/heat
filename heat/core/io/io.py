"""Generic, extension-driven entry points for reading and writing Heat arrays.

`load` and `save` dispatch on the file extension through the format registry; the
formats themselves live in the sibling modules of this package.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..dndarray import DNDarray
from ._registry import loader_for_path, saver_for_path
from .utils import sanitize_path

__all__ = [
    "load",
    "save",
]


def load(
    path: str, *args: Optional[List[object]], **kwargs: Optional[Dict[str, object]]
) -> DNDarray:
    """
    Loads data from a file, choosing the file format from its extension.

    The extension is looked up in the format registry, see
    :func:`~heat.core.io.available_formats`. CSV is always available; HDF5, netCDF4 and
    Zarr require their optional dependency to be installed.

    Parameters
    ----------
    path : str
        Path to the file to be read.
    *args : list, optional
        Additional positional arguments passed to the format-specific loader.
    **kwargs : dict, optional
        Additional keyword arguments passed to the format-specific loader.

    Returns
    -------
    DNDarray
        The data read from ``path``.

    Raises
    ------
    TypeError
        If ``path`` is not a string.
    ValueError
        If the file extension is not understood or known.
    RuntimeError
        If the optional dependency for a file extension is not available.

    See Also
    --------
    :func:`load_csv` : Loads data from a CSV file.
    :func:`load_csv_from_folder` : Loads multiple .csv files into one DNDarray.
    :func:`load_hdf5` : Loads data from an HDF5 file.
    :func:`load_multiple_hdf5` : Loads multiple HDF5 files into one DNDarray.
    :func:`load_netcdf` : Loads data from a NetCDF4 file.
    :func:`load_npy_from_path` : Loads multiple .npy files into one DNDarray.
    :func:`load_zarr` : Loads data from a Zarr store.

    Examples
    --------
    >>> ht.load("data.h5", dataset="DATA")
    DNDarray([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.load("data.nc", variable="DATA")
    DNDarray([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.load("my_data.zarr", variable="RECEIVER_1/DATA")
    DNDarray([[ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.load("my_data.zarr", variable="RECEIVER_*/DATA", split=0)
    DNDarray([[ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981],
              [ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981],
              [ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981]], dtype=ht.float32, device=cpu:0, split=0)
    """
    path = sanitize_path(path)
    return loader_for_path(path)(path, *args, **kwargs)


def save(
    data: DNDarray, path: str, *args: Optional[List[object]], **kwargs: Optional[Dict[str, object]]
):
    """
    Saves a ``DNDarray`` to a file, choosing the file format from its extension.

    The extension is looked up in the format registry, see
    :func:`~heat.core.io.available_formats`.

    Parameters
    ----------
    data : DNDarray
        The array holding the data to be stored.
    path : str
        Path to the file to be stored.
    *args : list, optional
        Additional positional arguments passed to the format-specific saver.
    **kwargs : dict, optional
        Additional keyword arguments passed to the format-specific saver.

    Returns
    -------
    object
        Whatever the format-specific saver returns, ``None`` for the built-in formats.

    Raises
    ------
    TypeError
        If ``path`` is not a string.
    ValueError
        If the file extension is not understood or known.
    RuntimeError
        If the optional dependency for a file extension is not available.

    See Also
    --------
    :func:`save_csv` : Saves data to a CSV file.
    :func:`save_hdf5` : Saves data to an HDF5 file.
    :func:`save_netcdf` : Saves data to a NetCDF4 file.
    :func:`save_zarr` : Saves data to a Zarr store.

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
