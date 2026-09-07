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
__HDF5_EXTENSIONS = frozenset([".h5", ".hdf5"])

__all__ = [
    "load",
    "save",
    "supports_hdf5",
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


try:
    import h5py
except ImportError:
    # HDF5 support is optional
    def supports_hdf5() -> bool:
        """
        Returns ``True`` if Heat supports reading from and writing to HDF5 files, ``False`` otherwise.
        """
        return False

else:
    # add functions to exports
    __all__.extend(["load_hdf5", "save_hdf5", "load_multiple_hdf5"])

    # warn the user about serial hdf5
    if not h5py.get_config().mpi and MPI_WORLD.rank == 0:
        warnings.warn(
            "h5py does not support parallel I/O, falling back to slower serial I/O", ImportWarning
        )

    def supports_hdf5() -> bool:
        """
        Returns ``True`` if Heat supports reading from and writing to HDF5 files, ``False`` otherwise.
        """
        return True

    def load_hdf5(
        path: str,
        dataset: str,
        dtype: Optional[datatype] = None,
        slices: Optional[Tuple[Optional[slice], ...]] = None,
        split: Optional[int] = None,
        device: Optional[str] = None,
        comm: Optional[Communication] = None,
    ) -> DNDarray:
        """
        Loads data from an HDF5 file. The data may be distributed among multiple processing nodes via the split flag.

        Parameters
        ----------
        path : str
            Path to the HDF5 file to be read.
        dataset : str
            Name of the dataset to be read.
        dtype : datatype, optional
            Data type of the resulting array, defaults to the loaded datasets type.
        slices : tuple of slice objects, optional
            Load only the specified slices of the dataset.
        split : int or None, optional
            The axis along which the data is distributed among the processing cores.
        device : str, optional
            The device id on which to place the data, defaults to globally set default device.
        comm : Communication, optional
            The communication to use for the data distribution.

        Raises
        ------
        TypeError
            If any of the input parameters are not of correct type

        Examples
        --------
        >>> a = ht.load_hdf5("data.h5", dataset="DATA")
        >>> a.shape
        [0/2] (5,)
        [1/2] (5,)
        >>> a.lshape
        [0/2] (5,)
        [1/2] (5,)
        >>> b = ht.load_hdf5("data.h5", dataset="DATA", split=0)
        >>> b.shape
        [0/2] (5,)
        [1/2] (5,)
        >>> b.lshape
        [0/2] (3,)
        [1/2] (2,)

        Using the slicing argument:
        >>> not_sliced = ht.load_hdf5("other_data.h5", dataset="DATA", split=0)
        >>> not_sliced.shape
        [0/2] (10,2)
        [1/2] (10,2)
        >>> not_sliced.lshape
        [0/2] (5,2)
        [1/2] (5,2)
        >>> not_sliced.larray
        [0/2] [[ 0,  1],
               [ 2,  3],
               [ 4,  5],
               [ 6,  7],
               [ 8,  9]]
        [1/2] [[10, 11],
               [12, 13],
               [14, 15],
               [16, 17],
               [18, 19]]

        >>> sliced = ht.load_hdf5("other_data.h5", dataset="DATA", split=0, slices=[slice(8)])
        >>> sliced.shape
        [0/2] (8,2)
        [1/2] (8,2)
        >>> sliced.lshape
        [0/2] (4,2)
        [1/2] (4,2)
        >>> sliced.larray
        [0/2] [[ 0,  1],
               [ 2,  3],
               [ 4,  5],
               [ 6,  7]]
        [1/2] [[ 8,  9],
               [10, 11],
               [12, 13],
               [14, 15],
               [16, 17]]

        >>> sliced = ht.load_hdf5(
        ...     "other_data.h5", dataset="DATA", split=0, slices=[slice(2, 8), slice(0, 1)]
        ... )
        >>> sliced.shape
        [0/2] (6,1)
        [1/2] (6,1)
        >>> sliced.lshape
        [0/2] (3,1)
        [1/2] (3,1)
        >>> sliced.larray
        [0/2] [[ 4, ],
               [ 6, ],
               [ 8, ]]
        [1/2] [[10, ],
               [12, ],
               [14, ]]
        """
        if not isinstance(path, str):
            raise TypeError(f"path must be str, not {type(path)}")
        elif not isinstance(dataset, str):
            raise TypeError(f"dataset must be str, not {type(dataset)}")
        elif split is not None and not isinstance(split, int):
            raise TypeError(f"split must be None or int, not {type(split)}")

        # determine the comm and device the data will be placed on
        device = devices.sanitize_device(device)
        comm = sanitize_comm(comm)

        # actually load the data from the HDF5 file
        with h5py.File(path, "r") as handle:
            data = handle[dataset]
            gshape = data.shape
            new_gshape = tuple()
            offsets = [0] * len(gshape)
            if dtype is None:
                dtype = data.dtype
            dtype = types.canonical_heat_type(dtype)
            if slices is not None:
                for i in range(len(gshape)):
                    if i < len(slices) and slices[i]:
                        s = slices[i]
                        if s.step is not None and s.step != 1:
                            raise ValueError("Slices with step != 1 are not supported")
                        new_axis_size, offset = size_from_slice(gshape[i], s)
                        new_gshape += (new_axis_size,)
                        offsets[i] = offset
                    else:
                        new_gshape += (gshape[i],)
                        offsets[i] = 0

                gshape = new_gshape

            dims = len(gshape)
            split = sanitize_axis(gshape, split)
            _, _, indices = comm.chunk(gshape, split)

            if slices is not None:
                new_indices = tuple()
                for offset, index in zip(offsets, indices):
                    new_indices += (slice(index.start + offset, index.stop + offset),)
                indices = new_indices

            balanced = True
            if split is None:
                data = torch.tensor(
                    data[indices], dtype=dtype.torch_type(), device=device.torch_device
                )
            elif indices[split].stop > indices[split].start:
                data = torch.tensor(
                    data[indices], dtype=dtype.torch_type(), device=device.torch_device
                )
            else:
                warnings.warn("More MPI ranks are used then the length of splitting dimension!")
                slice1 = tuple(
                    slice(0, gshape[i]) if i != split else slice(0, 1) for i in range(dims)
                )
                slice2 = tuple(
                    slice(0, gshape[i]) if i != split else slice(0, 0) for i in range(dims)
                )
                data = torch.tensor(
                    data[slice1], dtype=dtype.torch_type(), device=device.torch_device
                )
                data = data[slice2]

            return DNDarray(data, gshape, dtype, split, device, comm, balanced)

    def save_hdf5(
        data: DNDarray,
        path: str,
        dataset: str,
        mode: str = "w",
        dtype: Optional[datatype] = None,
        **kwargs: Dict[str, object],
    ):
        """
        Saves ``data`` to an HDF5 file. Attempts to utilize parallel I/O if possible.

        Parameters
        ----------
        data : DNDarray
            The data to be saved on disk.
        path : str
            Path to the HDF5 file to be written.
        dataset : str
            Name of the dataset the data is saved to.
        mode : str, optional
            File access mode, one of ``'w', 'a', 'r+'``
        dtype : datatype, optional
            Data type of the saved data
        kwargs : dict, optional
            Additional arguments passed to the created dataset.

        Raises
        ------
        TypeError
            If any of the input parameters are not of correct type.
        ValueError
            If the access mode is not understood.

        Examples
        --------
        >>> x = ht.arange(100, split=0)
        >>> ht.save_hdf5(x, "data.h5", dataset="DATA")
        """
        if not isinstance(data, DNDarray):
            raise TypeError(f"data must be heat tensor, not {type(data)}")
        if not isinstance(path, str):
            raise TypeError(f"path must be str, not {type(path)}")
        if not isinstance(dataset, str):
            raise TypeError(f"dataset must be str, not {type(path)}")

        # we only support a subset of possible modes
        if mode not in __VALID_WRITE_MODES:
            raise ValueError(f"mode was {mode}, not in possible modes {__VALID_WRITE_MODES}")

        # chunk the data, if no split is set maximize parallel I/O and chunk first axis
        is_split = data.split is not None
        _, _, slices = data.comm.chunk(data.gshape, data.split if is_split else 0)

        if dtype is None:
            dtype = data.dtype
        elif type(dtype) == torch.dtype:
            dtype = str(dtype).split(".")[-1]
        if type(dtype) is not str:
            dtype = dtype.__name__

        # attempt to perform parallel I/O if possible
        if h5py.get_config().mpi:
            with h5py.File(path, mode, driver="mpio", comm=data.comm.handle) as handle:
                dset = handle.create_dataset(dataset, data.shape, dtype=dtype, **kwargs)
                dset[slices] = data.larray.cpu() if is_split else data.larray[slices].cpu()

        # otherwise a single rank only write is performed in case of local data (i.e. no split)
        elif data.comm.rank == 0:
            with h5py.File(path, mode) as handle:
                dset = handle.create_dataset(dataset, data.shape, dtype=dtype, **kwargs)
                if is_split:
                    dset[slices] = data.larray.cpu()
                else:
                    dset[...] = data.larray.cpu()

            # ping next rank if it exists
            if is_split and data.comm.size > 1:
                data.comm.Isend([None, 0, MPI.INT], dest=1)
                data.comm.Recv([None, 0, MPI.INT], source=data.comm.size - 1)

        # no MPI, but split data is more tricky, we have to serialize the writes
        elif is_split:
            # wait for the previous rank to finish writing its chunk, then write own part
            data.comm.Recv([None, 0, MPI.INT], source=data.comm.rank - 1)
            with h5py.File(path, "r+") as handle:
                handle[dataset][slices] = data.larray.cpu()

            # ping the next node in the communicator, wrap around to 0 to complete barrier behavior
            next_rank = (data.comm.rank + 1) % data.comm.size
            data.comm.Isend([None, 0, MPI.INT], dest=next_rank)

    DNDarray.save_hdf5 = lambda self, path, dataset, mode="w", dtype=None, **kwargs: save_hdf5(
        self, path, dataset, mode, **kwargs
    )
    DNDarray.save_hdf5.__doc__ = save_hdf5.__doc__

    def load_multiple_hdf5(
        folder: str | Path,
        dataset: str,
        dtype=None,
        sorting_func: Callable[[list[Path]], list[Path]] | None = None,
    ) -> DNDarray:
        """Loads all .hdf5 or .h5 files inside the given folder into a single DNDarray that is split along axis 0; the arrays
        from the different files are concatenated along axis 0.
        The files are sorted by the sorting_func, if given, or by the default sorted function.

        Parameters
        ----------
        folder : str | Path
            folder containing all .h5 or .hdf5 files
        dataset : str
            dataset to load
        dtype : _type_, optional
            dtype to create array with, by default None
        sorting_func : Callable[[list[Path]], list[Path]] | None, optional
            how to sort the files, by default None, ie the sorted function
        """
        if not isinstance(folder, (str, Path)):
            raise TypeError(f"path must be an String or Path, not {type(folder)}")
        if not isinstance(dataset, str):
            raise TypeError(f"dataset must be a string not, {type(dataset)}")
        if not isinstance(sorting_func, Callable) and sorting_func is not None:
            raise TypeError(f"sorting_func must be None or Callable, not {type(sorting_func)}")
        if not Path(folder).is_dir():
            raise ValueError("Path must be a Folder")

        files: list[Path] = list(
            filter(lambda x: x.suffix == ".h5" or x.suffix == ".hdf5", Path(folder).iterdir())
        )
        files: list[Path] = sorted(files) if sorting_func is None else sorting_func(files)

        if len(files) == 0:
            raise ValueError("No files inside the directory.")

        h5_files: list[h5py.File] = [h5py.File(x) for x in files]
        shapes: list[tuple[int, ...]] = [x[dataset].shape for x in h5_files]
        dtypes: list[np.dtype] = [x[dataset].dtype for x in h5_files]
        bytes: list[int] = [x[dataset].dtype.itemsize for x in h5_files]

        if len(set(len(x) for x in shapes)) != 1:
            raise ValueError("Amount of dimensions of all hdf5 files must be the same")

        if dtype is None:
            warnings.warn("No explicit dtype given, will use biggest dtype across all files")
            dtype = dtypes[np.argmax(bytes)]

        shapes_arr = np.array(shapes, dtype=int)
        gshape = np.zeros(shapes_arr.shape[1], dtype=int)

        for i in range(1, len(gshape)):  # rows can be diffrent
            if len(set(shapes_arr[:, i])) != 1:
                raise ValueError(f"Dimension missmatch on ndim {i + 1}")
            gshape[i] = shapes_arr[0][i]
        gshape[0] = shapes_arr[:, 0].sum()
        gshape = tuple(gshape.astype(int))

        ranges: list[tuple[int, int]] = list()

        n = 0
        for i, shape in enumerate(shapes):
            rows = shape[0]
            ranges.append((n, n + rows))
            n += rows

        def find_file_ids_for_range(rslice: slice) -> list[int]:
            files = list()
            for i, (start, end) in enumerate(ranges):
                if start <= rslice.start and rslice.start < end:
                    files.append(i)
                elif start < rslice.stop and rslice.stop <= end:
                    files.append(i)
                elif rslice.start <= start and end <= rslice.stop:
                    files.append(i)
            return files

        _, _, slices = MPI_WORLD.chunk(gshape, split=0)
        local_slice: slice = slices[0]  # global row slice of local rank

        local_h5_file_ids: list[int] = find_file_ids_for_range(local_slice)

        data: DNDarray = factories.empty(gshape, dtype=dtype, split=0)

        n = 0
        start_h5_index = local_slice.start - shapes_arr[: local_h5_file_ids[0], 0].sum()
        max_n: int = data.larray.shape[0]
        for i, idx in enumerate(local_h5_file_ids):
            if i == 0:
                start_index = start_h5_index
            else:
                start_index = 0
            h5_file: h5py.File = h5_files[idx]
            rows: int = shapes[idx][0] - start_index

            diff = n + rows - min(n + rows, max_n)  # amount of overshoot

            data.larray[n : (n + rows - diff)] = torch.from_numpy(
                h5_file[dataset][start_index : (start_index + rows - diff)]
            )
            n += rows - diff
        return data


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

register_format(
    "hdf5",
    __HDF5_EXTENSIONS,
    dependency="h5py",
    loader=load_hdf5 if supports_hdf5() else None,
    saver=save_hdf5 if supports_hdf5() else None,
)
