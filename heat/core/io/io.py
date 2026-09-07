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
__NETCDF_EXTENSIONS = frozenset([".nc", ".nc4", "netcdf"])
__NETCDF_DIM_TEMPLATE = "{}_dim_{}"
__ZARR_EXTENSIONS = frozenset([".zarr"])

__all__ = [
    "load",
    "save",
    "supports_hdf5",
    "supports_netcdf",
    "supports_zarr",
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
    import netCDF4 as nc
except ImportError:
    # netCDF4 support is optional
    def supports_netcdf() -> bool:
        """
        Returns ``True`` if Heat supports reading from and writing to netCDF4 files, ``False`` otherwise.
        """
        return False

else:
    # add functions to visible exports
    __all__.extend(["load_netcdf", "save_netcdf"])

    # determine netCDF's parallel I/O support
    __nc_has_par = (
        nc.__dict__.get("__has_parallel4_support__", False)
        or nc.__dict__.get("__has_pnetcdf_support__", False)
        or nc.__dict__.get("__has_nc_par__", False)
    )

    # warn the user about serial netcdf
    if not __nc_has_par and MPI_WORLD.rank == 0:
        warnings.warn(
            "netCDF4 does not support parallel I/O, falling back to slower serial I/O",
            ImportWarning,
        )

    def supports_netcdf() -> bool:
        """
        Returns ``True`` if Heat supports reading from and writing to netCDF4 files, ``False`` otherwise.
        """
        return True

    def load_netcdf(
        path: str,
        variable: str,
        dtype: datatype = types.float32,
        split: Optional[int] = None,
        device: Optional[str] = None,
        comm: Optional[Communication] = None,
    ) -> DNDarray:
        """
        Loads data from a NetCDF4 file. The data may be distributed among multiple processing nodes via the split flag.

        Parameters
        ----------
        path : str
            Path to the NetCDF4 file to be read.
        variable : str
            Name of the variable to be read.
        dtype : datatype, optional
            Data type of the resulting array
        split : int or None, optional
            The axis along which the data is distributed among the processing cores.
        comm : Communication, optional
            The communication to use for the data distribution. Defaults to MPI_COMM_WORLD.
        device : str, optional
            The device id on which to place the data, defaults to globally set default device.

        Raises
        ------
        TypeError
            If any of the input parameters are not of correct type.

        Examples
        --------
        >>> a = ht.load_netcdf("data.nc", variable="DATA")
        >>> a.shape
        [0/2] (5,)
        [1/2] (5,)
        >>> a.lshape
        [0/2] (5,)
        [1/2] (5,)
        >>> b = ht.load_netcdf("data.nc", variable="DATA", split=0)
        >>> b.shape
        [0/2] (5,)
        [1/2] (5,)
        >>> b.lshape
        [0/2] (3,)
        [1/2] (2,)
        """
        if not isinstance(path, str):
            raise TypeError(f"path must be str, not {type(path)}")
        if not isinstance(variable, str):
            raise TypeError(f"dataset must be str, not {type(variable)}")
        if split is not None and not isinstance(split, int):
            raise TypeError(f"split must be None or int, not {type(split)}")

        # infer the canonical heat datatype
        dtype = types.canonical_heat_type(dtype)
        # determine the device and comm the data will be placed on
        device = devices.sanitize_device(device)
        comm = sanitize_comm(comm)

        # actually load the data
        with nc.Dataset(path, "r", parallel=__nc_has_par, comm=comm.handle) as handle:
            data = handle[variable]

            # prepare meta information
            gshape = tuple(data.shape)
            split = sanitize_axis(gshape, split)

            # chunk up the data portion
            _, local_shape, indices = comm.chunk(gshape, split)
            balanced = True
            if split is None or local_shape[split] > 0:
                data = torch.tensor(
                    data[indices], dtype=dtype.torch_type(), device=device.torch_device
                )
            else:
                data = torch.empty(
                    local_shape, dtype=dtype.torch_type(), device=device.torch_device
                )

            return DNDarray(data, gshape, dtype, split, device, comm, balanced)

    def save_netcdf(
        data: DNDarray,
        path: str,
        variable: str,
        mode: str = "w",
        dimension_names: Union[list, tuple, str] = None,
        is_unlimited: bool = False,
        file_slices: Union[Iterable[int], slice, bool] = slice(None),
        **kwargs: Dict[str, object],
    ):
        """
        Saves data to a netCDF4 file. Attempts to utilize parallel I/O if possible.

        Parameters
        ----------
        data : DNDarray
            The data to be saved on disk.
        path : str
            Path to the netCDF4 file to be written.
        variable : str
            Name of the variable the data is saved to.
        mode : str, optional
            File access mode, one of ``'w', 'a', 'r+'``.
        dimension_names : list or tuple or string
            Specifies the netCDF Dimensions used by the variable. Ignored if Variable already exists.
        is_unlimited : bool, optional
            If True, every dimension created for this variable (i.e. doesn't already exist) is unlimited. Already
            existing limited dimensions cannot be changed to unlimited and vice versa.
        file_slices : integer iterable, slice, ellipsis or bool
            Keys used to slice the netCDF Variable, as given in the nc.utils._StartCountStride method.
        kwargs : dict, optional
            additional arguments passed to the created dataset.

        Raises
        ------
        TypeError
            If any of the input parameters are not of correct type.
        ValueError
            If the access mode is not understood or if the number of dimension names does not match the number of
            dimensions.

        Examples
        --------
        >>> x = ht.arange(100, split=0)
        >>> ht.save_netcdf(x, "data.nc", dataset="DATA")
        """
        if not isinstance(data, DNDarray):
            raise TypeError(f"data must be heat tensor, not {type(data)}")
        if not isinstance(path, str):
            raise TypeError(f"path must be str, not {type(path)}")
        if not isinstance(variable, str):
            raise TypeError(f"variable must be str, not {type(path)}")
        if dimension_names is None:
            dimension_names = [
                __NETCDF_DIM_TEMPLATE.format(variable, dim) for dim, _ in enumerate(data.shape)
            ]
        elif isinstance(dimension_names, str):
            dimension_names = [dimension_names]
        elif isinstance(dimension_names, tuple):
            dimension_names = list(dimension_names)
        elif not isinstance(dimension_names, list):
            raise TypeError(
                "dimension_names must be list or tuple or string, not{}".format(
                    type(dimension_names)
                )
            )
        elif not len(dimension_names) == len(data.shape):
            raise ValueError(f"{len(dimension_names)} names given for {len(data.shape)} dimensions")

        # we only support a subset of possible modes
        if mode not in __VALID_WRITE_MODES:
            raise ValueError(f"mode was {mode}, not in possible modes {__VALID_WRITE_MODES}")

        failed = 0
        excep = None
        # chunk the data, if no split is set maximize parallel I/O and chunk first axis
        is_split = data.split is not None
        _, _, slices = data.comm.chunk(data.gshape, data.split if is_split else 0)

        def __get_expanded_split(
            shape: Tuple[int], expanded_shape: Tuple[int], split: Optional[int]
        ) -> int:
            """
            Returns the hypothetical split-axis of a dndarray of shape=shape and
            split=split if it was expanded to expandedShape by adding empty dimensions.

            Parameters
            ----------
            shape : tuple[int]
                Shape of a DNDarray.
            expanded_shape : tuple[int]
                Shape of hypothetical expanded DNDarray.
            split : int or None
                split-axis of dndarray.

            Raises
            ------
            ValueError
                If resulting shapes do not match.
            """
            if np.prod(shape) != np.prod(expanded_shape):
                raise ValueError(f"Shapes {shape} and {expanded_shape} do not have the same size")
            if np.prod(shape) == 1:  # size 1 array
                return split
            if len(shape) == len(expanded_shape):  # actually not expanded at all
                return split
            if split is None:  # not split at all
                return None
            # Get indices of non-empty dimensions and squeezed shapes
            enumerated = [[i, v] for i, v in enumerate(shape) if v != 1]
            ind_nonempty, sq_shape = list(zip(*enumerated))  # transpose
            enumerated = [[i, v] for i, v in enumerate(expanded_shape) if v != 1]
            ex_ind_nonempty, sq_ex = list(zip(*enumerated))  # transpose
            if not sq_shape == sq_ex:
                raise ValueError(
                    f"Shapes {shape} and {expanded_shape} differ in non-empty dimensions"
                )
            if split in ind_nonempty:  # split along non-empty dimension
                split_sq = ind_nonempty.index(split)  # split-axis in squeezed shape
                return ex_ind_nonempty[split_sq]
            # split along empty dimension: split doesnt matter, only one process contains data
            # return the last empty dimension (in expanded shape) before (the first nonempty dimension after split)
            # number of nonempty elems before split
            ne_before_split = split - shape[:split].count(1)
            ind_ne_after_split = ind_nonempty[
                ne_before_split
            ]  # index of (first nonempty element after split) in squeezed shape
            return max(
                i
                for i, v in enumerate(expanded_shape[: max(ex_ind_nonempty[:ind_ne_after_split])])
                if v == 1
            )

        def __merge_slices(
            var: nc.Variable,
            var_slices: Tuple[int, slice],
            data: DNDarray,
            data_slices: Optional[Tuple[int, slice]] = None,
        ) -> Tuple[Union[int, slice]]:
            """
            Allows replacing:
                ``var[var_slices][data_slices] = data``
            (a `netcdf4.Variable.__getitem__` and a `numpy.ndarray.__setitem__` call)

            with:
                ``var[ __merge_slices(var, var_slices, data, data_slices) ] = data``
            (a single `netcdf4.Variable.__setitem__` call).

            This is necessary because performing the former would, in the ``__getitem__``, load the global dataset onto
            every process in local ``np.ndarray``s. Then, the ``__setitem__`` would write the local `chunk` into the
            ``np.ndarray``.

            The latter allows the netcdf4 library to parallelize the write-operation by directly using the
            `netcdf4.Variable.__setitem__` method.

            Parameters
            ----------
            var : nc.Variable
                Variable to which data is to be saved.
            var_slices : tuple[int, slice]
                Keys to pass to the set-operator.
            data : DNDarray
                Data to be saved.
            data_slices: tuple[int, slice]
                As returned by the data.comm.chunk method.
            """
            slices = data_slices
            if slices is None:
                _, _, slices = data.comm.chunk(data.gshape, data.split if is_split else 0)
            start, count, stride, _ = nc.utils._StartCountStride(
                elem=var_slices,
                shape=var.shape,
                dimensions=var.dimensions,
                grp=var.group(),
                datashape=data.shape,
                put=True,
            )
            out_shape = nc._netCDF4._out_array_shape(count)
            out_split = __get_expanded_split(data.shape, out_shape, data.split)

            start, count, stride = start.T, count.T, stride.T  # transpose for iteration
            stop = start + stride * count
            new_slices = []
            for begin, end, step in zip(start, stop, stride):
                if begin.size == 1:
                    begin, end, step = begin.item(), end.item(), step.item()
                    new_slices.append(slice(begin, end, step))
                else:
                    begin, end, step = begin.flatten(), end.flatten(), step.flatten()
                    new_slices.append(
                        np.r_[
                            tuple(
                                slice(b.item(), e.item(), s.item())
                                for b, e, s in zip(begin, end, step)
                            )
                        ]
                    )
            if out_split is not None:  # add split-slice
                if isinstance(new_slices[out_split], slice):
                    start, stop, step = (
                        new_slices[out_split].start,
                        new_slices[out_split].stop,
                        new_slices[out_split].step,
                    )
                    sliced = range(start, stop, step)[slices[data.split]]
                    a, b, c = sliced.start, sliced.stop, sliced.step
                    a = None if a < 0 else a
                    b = None if b < 0 else b
                    new_slices[out_split] = slice(a, b, c)
                    # new_slices[out_split] = sliced
                elif isinstance(new_slices[out_split], np.ndarray):
                    new_slices[out_split] = new_slices[out_split][slices[data.split]]
                else:
                    new_slices[out_split] = np.r_[new_slices[out_split]][slices[data.split]]
            return tuple(new_slices)

        # attempt to perform parallel I/O if possible
        if __nc_has_par:
            try:
                with nc.Dataset(path, mode, parallel=True, comm=data.comm.handle) as handle:
                    if variable in handle.variables:
                        var = handle.variables[variable]
                    else:
                        for name, elements in zip(dimension_names, data.shape):
                            if name not in handle.dimensions:
                                handle.createDimension(name, elements if not is_unlimited else None)
                        var = handle.createVariable(
                            variable, data.dtype.char(), dimension_names, **kwargs
                        )
                    merged_slices = __merge_slices(var, file_slices, data)
                    try:
                        var[merged_slices] = (
                            data.larray.cpu() if is_split else data.larray[slices].cpu()
                        )
                    except RuntimeError:
                        var.set_collective(True)
                        var[merged_slices] = (
                            data.larray.cpu() if is_split else data.larray[slices].cpu()
                        )
            except Exception as e:
                failed = data.comm.rank + 1
                excep = e
        # otherwise a single rank only write is performed in case of local data (i.e. no split)
        elif data.comm.rank == 0:
            try:
                with nc.Dataset(path, mode) as handle:
                    if variable in handle.variables:
                        var = handle.variables[variable]
                    else:
                        for name, elements in zip(dimension_names, data.shape):
                            if name not in handle.dimensions:
                                handle.createDimension(name, elements if not is_unlimited else None)
                        var = handle.createVariable(
                            variable, data.dtype.char(), dimension_names, **kwargs
                        )
                    var.set_collective(False)  # not possible with non-parallel netcdf
                    if is_split:
                        merged_slices = __merge_slices(var, file_slices, data)
                        var[merged_slices] = data.larray.cpu()
                    else:
                        var[file_slices] = data.larray.cpu()
            except Exception as e:
                failed = 1
                excep = e
            finally:
                if data.comm.size > 1:
                    data.comm.isend(failed, dest=1)
                    data.comm.recv()

        # non-root
        else:
            # wait for the previous rank to finish writing its chunk, then write own part
            failed = data.comm.recv()
            try:
                # no MPI, but data is split, we have to serialize the writes
                if not failed and is_split:
                    with nc.Dataset(path, "r+") as handle:
                        var = handle.variables[variable]
                        var.set_collective(False)  # not possible with non-parallel netcdf
                        merged_slices = __merge_slices(var, file_slices, data)
                        var[merged_slices] = data.larray.cpu()
            except Exception as e:
                failed = data.comm.rank + 1
                excep = e
            finally:
                # ping the next node in the communicator, wrap around to 0 to complete barrier behavior
                next_rank = (data.comm.rank + 1) % data.comm.size
                data.comm.isend(failed, dest=next_rank)

        failed = data.comm.allreduce(failed, op=MPI.MAX)
        if failed - 1 == data.comm.rank:
            data.comm.bcast(excep, root=failed - 1)
            raise excep
        elif failed:
            excep = data.comm.bcast(excep, root=failed - 1)
            excep.args = f"raised by process rank {failed - 1}", *excep.args
            raise excep from None  # raise the same error but without traceback
            # because that is on a different process

    DNDarray.save_netcdf = lambda self, path, variable, mode="w", **kwargs: save_netcdf(
        self, path, variable, mode, **kwargs
    )
    DNDarray.save_netcdf.__doc__ = save_netcdf.__doc__

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


try:
    import zarr
except ModuleNotFoundError:

    def supports_zarr() -> bool:
        """
        Returns ``True`` if zarr is installed, ``False`` otherwise.
        """
        return False

else:
    __all__.extend(["load_zarr", "save_zarr"])

    def supports_zarr() -> bool:
        """
        Returns ``True`` if zarr is installed, ``False`` otherwise.
        """
        return True

    def load_zarr(
        path: str,
        variable: str = None,
        split: int = 0,
        device: Optional[str] = None,
        comm: Optional[Communication] = None,
        slices: Union[None, slice, Iterable[Union[slice, None]]] = None,
        **kwargs,
    ) -> DNDarray:
        """
        Loads data from a zarr store into DNDarray. `path` can either point to a single zarr array or a zarr group. In the latter case, `variable` must be provided to specify which array in the group to load. If `variable` contains a wildcard pattern (e.g. `RECEIVER_*/DATA`), all matching arrays will be loaded and concatenated along the specified `split` axis.

        Parameters
        ----------
        path : str
            Path to the directory in which a .zarr-file is located.
        variable : str, optional
            If the zarr store is a group, the variable (or path to variable) to load from the group.
            Can contain a wildcard pattern to load and concatenate arrays stored in slices in different directories.
        split : int
            Along which axis the loaded arrays should be concatenated.
        device : str, optional
            The device id on which to place the data, defaults to globally set default device.
        comm : Communication, optional
            The communication to use for the data distribution, default is 'heat.MPI_WORLD'
        slices: Union[None, slice, Iterable[Union[slice, None]]]
            Load only a slice of the array instead of everything
        **kwargs : Any
            extra Arguments to pass to zarr.open
        """
        # sanitize inputs
        device = devices.sanitize_device(device)
        torch_device = device.torch_device
        comm = sanitize_comm(comm)

        if not isinstance(path, str):
            raise TypeError(f"path must be str, not {type(path)}")
        if not isinstance(slices, (slice, Iterable)) and slices is not None:
            raise TypeError(f"Slices Argument must be slice, tuple or None and not {type(slices)}")
        if isinstance(slices, Iterable):
            for elem in slices:
                if isinstance(elem, slice) or elem is None:
                    continue
                raise TypeError(f"Tuple values of slices must be slice or None, not {type(elem)}")

        for extension in __ZARR_EXTENSIONS:
            if fnmatch.fnmatch(path, f"*{extension}"):
                break
        else:
            raise ValueError("File has no zarr extension.")

        store_path = os.path.join(path, variable) if variable else path

        output_dtype = kwargs.pop("dtype", None)
        torch_output_dtype = output_dtype.torch_type() if output_dtype else None

        if variable and "*" in variable:
            # `variable` contains a wildcard pattern
            # e.g. data were chunked at write-out and stored in multiple directories
            if slices is not None:
                raise NotImplementedError("Slicing is not supported when loading with a wildcard.")

            base_paths = sorted(glob.glob(store_path))

            if not base_paths:
                raise FileNotFoundError(
                    f"Zarr wildcard pattern '{variable}' did not match any arrays in store '{path}'"
                )

            variable_paths = [os.path.relpath(p, start=path) for p in base_paths]

            # each rank reads data from its assigned directories and concatenates locally

            # determine which directories to open on rank
            dummy_array = factories.empty((len(base_paths),), dtype=types.float32)
            _, _, local_dir_slice = dummy_array.comm.chunk(
                dummy_array.shape, rank=dummy_array.comm.rank, split=0
            )

            # load data to torch tensors
            local_tensors = []
            for i, var_path in enumerate(variable_paths[local_dir_slice[0]]):
                local_tensor = torch.from_numpy(zarr.open(path)[var_path][:])
                if torch_output_dtype:
                    local_tensor = local_tensor.to(torch_output_dtype)
                local_tensors.append(local_tensor)

            # Have rank 0 determine the single-store shape and broadcast it to all ranks for sanitation
            target_ndims = torch.zeros(1, dtype=torch.int32)
            if dummy_array.comm.rank == 0:
                if len(local_tensors) == 0:
                    raise ValueError(
                        f"Zarr wildcard pattern '{variable}' did not match any arrays in store '{path}'"
                    )
                # broadcast shape of first local tensor to allow sanitation on empty ranks
                target_ndims = torch.tensor(local_tensors[0].ndim, dtype=torch.int32)
            dummy_array.comm.Bcast(target_ndims, root=0)
            # sanitize split axis
            proxy_shape = (1,) * target_ndims.item()
            split = sanitize_axis(proxy_shape, axis=split)

            # prepare sorted list of all heat datatypes to convert them to integer for communication with MPI
            all_heat_type_names = [dtype.__name__ for dtype in types.__get_all_heat_types()]
            all_heat_type_names.sort()

            # concatenate locally
            if len(local_tensors) >= 1:
                if len(local_tensors) == 1:
                    local_tensor = local_tensors[0]
                else:
                    local_tensor = torch.cat(local_tensors, dim=split if split is not None else 0)
                empty_ranks = torch.tensor([0], dtype=torch.int32)
                ht_type_code = all_heat_type_names.index(
                    types.canonical_heat_type(local_tensor.dtype).__name__
                )
            else:
                # no local tensors i.e. no data assigned to rank
                local_tensor = torch.empty((0,))
                empty_ranks = torch.tensor([1], dtype=torch.int32)
                # dummy dtype code
                ht_type_code = -1
            # check for empty ranks
            dummy_array.comm.Allreduce(MPI.IN_PLACE, empty_ranks, op=MPI.SUM)
            if empty_ranks.item() > 0:
                # fix local shape and dtype of empty tensors, otherwise DNDarray construction will fail
                # Rank 0 broadcasts the info to all other ranks
                target_shape = torch.zeros(
                    (
                        1,
                        target_ndims.item() + 1,
                    ),
                    dtype=torch.int64,
                )
                if local_tensor.numel() > 0:
                    target_shape[0, :-1] = torch.tensor(local_tensor.shape, dtype=torch.int64)
                # encode dtype as last entry
                target_shape[0, -1] = ht_type_code
                # share info about target shape and dtype
                target_shapes = torch.zeros(
                    (dummy_array.comm.size, target_ndims.item() + 1), dtype=torch.int64
                )
                dummy_array.comm.Allgather(target_shape, target_shapes)
                if local_tensor.numel() == 0:
                    ht_type_code = target_shapes[0, -1].item()
                    target_shape = target_shapes[0, :-1].clone()
                    target_shape[split] = 0
                    ht_type = getattr(types, all_heat_type_names[ht_type_code])
                    local_tensor = torch.empty(
                        tuple(target_shape.tolist()), dtype=ht_type.torch_type()
                    )
                # discard dtype code column
                target_shapes = target_shapes[:, :-1]
                # calculate global array shape
                out_gshape = target_shapes[0, :].clone()
                out_gshape[split] = target_shapes[:, split].sum().item()
                # wrap local tensors in DNDarray
                dndarray = DNDarray(
                    local_tensor.to(device=torch_device),
                    gshape=tuple(out_gshape.tolist()),
                    dtype=output_dtype
                    if output_dtype
                    else types.canonical_heat_type(local_tensor.dtype),
                    split=split,
                    device=device,
                    comm=comm,
                    balanced=False,
                )
            else:
                # all ranks are populated, create DNDarray directly
                dndarray = factories.array(local_tensor, is_split=split, device=device, comm=comm)
            dndarray.balance_()
            return dndarray

        # standard single zarr array
        arr: zarr.Array = zarr.open_array(store=store_path, **kwargs)
        shape = arr.shape

        if isinstance(slices, slice) or slices is None:
            slices = [slices]

        if len(shape) < len(slices):
            raise ValueError(
                f"slices Argument has more arguments than the length of the shape of the array. {len(shape)} < {len(slices)}"
            )

        slices = [elem if elem is not None else slice(None) for elem in slices]
        slices.extend([slice(None) for _ in range(abs(len(slices) - len(shape)))])

        dtype = types.canonical_heat_type(arr.dtype)
        split = sanitize_axis(shape, axis=split)

        # slices = tuple(slice(*tslice.indices(length)) for length, tslice in zip(shape, slices))
        slices = tuple(slices)
        shape = [len(range(*tslice.indices(length))) for length, tslice in zip(shape, slices)]
        offset, local_shape, local_slices = comm.chunk(shape, split)

        return factories.array(
            arr[slices][local_slices], dtype=dtype, is_split=split, device=device, comm=comm
        )

    def save_zarr(dndarray: DNDarray, path: str, overwrite: bool = False, **kwargs) -> None:
        """
        Writes the DNDArray into the zarr-format.

        Parameters
        ----------
        dndarray : DNDarray
            DNDArray to save.
        path : str
            path to save to.
        overwrite : bool
            Wether to overwrite an existing array.
        **kwargs : Any
            extra Arguments to pass to zarr.open and zarr.create

        Raises
        ------
        TypeError
            - If given parameters do not match or have conflicting information.
            - If it already exists and no overwrite is specified.

        Notes
        -----
        Zarr functions by chunking the data, were a chunk is a file inside the store.
        The problem ist that only one process writes to it at a time. Therefore when two
        processes try to write to the same chunk one will fail, unless the other finishes before
        the other starts.

        To alleviate it we can define the chunk sizes ourselves. To do this we just get the lowest size of
        the distributed axis, ex: split=0 with a (4,4) shape with a worldsize of 4 you would chunk it with (1,4).

        A problem arises when a process gets a bigger chunk and interferes with another process. Example:
        N_PROCS = 4
        SHAPE = (9,10)
        SPLIT = 0
        CHUNKS => (2,10)

        In this problem one process will have a write region of 3 rows and therefore be able to either not write
        or overwrite what another process does therefore destroying the parallel write as it would at the end load
        2 chunks to write 3 rows.
        To counter act this we just set the chunk size in the split axis to 1. This allows for no overwrites but can
        cripple write speeds and or even speed it up.

        Another Problem with this approach is that we tell zarr have full chunks, i.e if array has shape (10_000, 10_000)
        and we split it at axis=0 with 4 processes we have chunks of (2_500, 10_000). Zarr will load the whole chunk into
        memory making it memory intensive and probably inefficient. Better approach would be to have a smaller chunk size
        for example half of it but that cannot be determined at all times so the current approach is a compromise.

        Another Problem is the split=None scenario. In this case every processs has the same data, so only one needs to write
        so we ignore chunking and let zarr decide the chunk size and let only one process, aka rank=0 write.

        To avoid errors when using NumPy arrays as chunk shape, the chunks argument is only passed to zarr.create if it is
        not None. This prevents issues with ambiguous truth values or attribute errors on None.

        """
        if not isinstance(path, str):
            raise TypeError(f"path must be str, not {type(path)}")

        for extension in __ZARR_EXTENSIONS:
            if fnmatch.fnmatch(path, f"*{extension}"):
                break
        else:
            raise ValueError("path does not end on an Zarr extension.")

        if os.path.exists(path) and not overwrite:
            raise RuntimeError("Given Path already exists.")

        if MPI_WORLD.rank == 0:
            if dndarray.split is None or MPI_WORLD.size == 1:
                chunks = None
            else:
                chunks = np.array(dndarray.gshape)
                axis = dndarray.split

                if chunks[axis] % MPI_WORLD.size != 0:
                    chunks[axis] = 1
                else:
                    chunks[axis] //= MPI_WORLD.size

                    CODEC_LIMIT_BYTES = 2**31 - 1  # PR#1766

                    for _ in range(
                        10
                    ):  # Use for loop instead of while true for better handling of edge cases
                        byte_size = reduce(operator.mul, chunks, 1) * dndarray.larray.element_size()
                        if byte_size > CODEC_LIMIT_BYTES:
                            if chunks[axis] % 2 == 0:
                                chunks[axis] /= 2
                                continue
                            else:
                                chunks[axis] = 1
                                break
                        else:
                            break
                    else:
                        chunks[axis] = 1
                        warnings.warn(
                            "Calculation of chunk size for zarr format unexpectadly defaulted to 1 on the split axis"
                        )

            dtype = dndarray.dtype.char()

            zarr_create_kwargs = {
                "store": path,
                "shape": dndarray.gshape,
                "dtype": dtype,
                "overwrite": overwrite,
                **kwargs,
            }

            if chunks is not None:
                zarr_create_kwargs["chunks"] = chunks.tolist()

            zarr_array = zarr.create(**zarr_create_kwargs)

        # Wait for the file creation to finish
        MPI_WORLD.Barrier()
        zarr_array = zarr.open(store=path, mode="r+", **kwargs)

        if dndarray.split is not None:
            _, _, slices = MPI_WORLD.chunk(dndarray.gshape, dndarray.split)

            zarr_array[slices] = (
                dndarray.larray.cpu().numpy()  # Numpy array needed as zarr can only understand numpy dtypes and infers it.
            )
        else:
            if MPI_WORLD.rank == 0:
                zarr_array[:] = dndarray.larray.cpu().numpy()

        MPI_WORLD.Barrier()


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
register_format(
    "netcdf",
    __NETCDF_EXTENSIONS,
    dependency="netCDF4",
    loader=load_netcdf if supports_netcdf() else None,
    saver=save_netcdf if supports_netcdf() else None,
)
register_format(
    "zarr",
    __ZARR_EXTENSIONS,
    dependency="zarr",
    loader=load_zarr if supports_zarr() else None,
    saver=save_zarr if supports_zarr() else None,
)
