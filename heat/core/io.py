"""Enables parallel I/O with data on disk."""

from __future__ import annotations

from functools import reduce
import operator
import os.path
from math import log10
import numpy as np
import torch
import warnings
import fnmatch

from typing import Dict, Iterable, List, Optional, Tuple, Union

from . import devices
from . import factories
from . import types

from .communication import Communication, MPI, MPI_WORLD, sanitize_comm
from .dndarray import DNDarray
from .manipulations import hsplit, vsplit
from .statistics import max as smax, min as smin
from .stride_tricks import sanitize_axis
from .types import datatype

__VALID_WRITE_MODES = frozenset(["w", "a", "r+"])
__CSV_EXTENSION = frozenset([".csv"])
__HDF5_EXTENSIONS = frozenset([".h5", ".hdf5"])
__NETCDF_EXTENSIONS = frozenset([".nc", ".nc4", "netcdf"])
__NETCDF_DIM_TEMPLATE = "{}_dim_{}"
__ZARR_EXTENSIONS = frozenset([".zarr"])

__all__ = [
    "load",
    "load_csv",
    "save_csv",
    "save",
    "supports_hdf5",
    "supports_netcdf",
    "load_npy_from_path",
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
    __all__.extend(["load_hdf5", "save_hdf5"])

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
        dtype: datatype = types.float32,
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
            Data type of the resulting array.
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

        >>> sliced = ht.load_hdf5("other_data.h5", dataset="DATA", split=0, slices=slice(8))
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

        >>> sliced = ht.load_hdf5('other_data.h5', dataset='DATA', split=0, slices=(slice(2,8), slice(0,1))
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

        # infer the type and communicator for the loaded array
        dtype = types.canonical_heat_type(dtype)
        # determine the comm and device the data will be placed on
        device = devices.sanitize_device(device)
        comm = sanitize_comm(comm)

        # actually load the data from the HDF5 file
        with h5py.File(path, "r") as handle:
            data = handle[dataset]
            gshape = data.shape
            new_gshape = tuple()
            offsets = [0] * len(gshape)
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
        data: DNDarray, path: str, dataset: str, mode: str = "w", **kwargs: Dict[str, object]
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

        # attempt to perform parallel I/O if possible
        if h5py.get_config().mpi:
            with h5py.File(path, mode, driver="mpio", comm=data.comm.handle) as handle:
                dset = handle.create_dataset(dataset, data.shape, **kwargs)
                dset[slices] = data.larray.cpu() if is_split else data.larray[slices].cpu()

        # otherwise a single rank only write is performed in case of local data (i.e. no split)
        elif data.comm.rank == 0:
            with h5py.File(path, mode) as handle:
                dset = handle.create_dataset(dataset, data.shape, **kwargs)
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

    DNDarray.save_hdf5 = lambda self, path, dataset, mode="w", **kwargs: save_hdf5(
        self, path, dataset, mode, **kwargs
    )
    DNDarray.save_hdf5.__doc__ = save_hdf5.__doc__


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

    See Also
    --------
    :func:`load_csv` : Loads data from a CSV file.
    :func:`load_csv_from_folder` : Loads multiple .csv files into one DNDarray which will be returned.
    :func:`load_hdf5` : Loads data from an HDF5 file.
    :func:`load_netcdf` : Loads data from a NetCDF4 file.
    :func:`load_npy_from_path` : Loads multiple .npy files into one DNDarray which will be returned.
    :func:`load_zarr` : Loads zarr-Format into DNDarray which will be returned.

    """
    if not isinstance(path, str):
        raise TypeError(f"Expected path to be str, but was {type(path)}")
    extension = os.path.splitext(path)[-1].strip().lower()

    if extension in __CSV_EXTENSION:
        return load_csv(path, *args, **kwargs)
    elif extension in __HDF5_EXTENSIONS:
        if supports_hdf5():
            return load_hdf5(path, *args, **kwargs)
        else:
            raise RuntimeError(f"hdf5 is required for file extension {extension}")
    elif extension in __NETCDF_EXTENSIONS:
        if supports_netcdf():
            return load_netcdf(path, *args, **kwargs)
        else:
            raise RuntimeError(f"netcdf is required for file extension {extension}")
    elif extension in __ZARR_EXTENSIONS:
        if supports_zarr():
            return load_zarr(path, *args, **kwargs)
        else:
            raise RuntimeError(f"Package zarr is required for file extension {extension}")

    else:
        raise ValueError(f"Unsupported file extension {extension}")


def load_csv(
    path: str,
    header_lines: int = 0,
    sep: str = ",",
    dtype: datatype = types.float32,
    encoding: str = "utf-8",
    split: Optional[int] = None,
    device: Optional[str] = None,
    comm: Optional[Communication] = None,
) -> DNDarray:
    """
    Loads data from a CSV file. The data will be distributed along the axis 0.

    Parameters
    ----------
    path : str
        Path to the CSV file to be read.
    header_lines : int, optional
        The number of columns at the beginning of the file that should not be considered as data.
    sep : str, optional
        The single ``char`` or ``str`` that separates the values in each row.
    dtype : datatype, optional
        Data type of the resulting array.
    encoding : str, optional
        The type of encoding which will be used to interpret the lines of the csv file as strings.
    split : int or None : optional
        Along which axis the resulting array should be split.
        Default is ``None`` which means each node will have the full array.
    device : str, optional
        The device id on which to place the data, defaults to globally set default device.
    comm : Communication, optional
        The communication to use for the data distribution, defaults to global default

    Raises
    ------
    TypeError
        If any of the input parameters are not of correct type.

    Examples
    --------
    >>> import heat as ht
    >>> a = ht.load_csv("data.csv")
    >>> a.shape
    [0/3] (150, 4)
    [1/3] (150, 4)
    [2/3] (150, 4)
    [3/3] (150, 4)
    >>> a.lshape
    [0/3] (38, 4)
    [1/3] (38, 4)
    [2/3] (37, 4)
    [3/3] (37, 4)
    >>> b = ht.load_csv("data.csv", header_lines=10)
    >>> b.shape
    [0/3] (140, 4)
    [1/3] (140, 4)
    [2/3] (140, 4)
    [3/3] (140, 4)
    >>> b.lshape
    [0/3] (35, 4)
    [1/3] (35, 4)
    [2/3] (35, 4)
    [3/3] (35, 4)
    """
    if not isinstance(path, str):
        raise TypeError(f"path must be str, not {type(path)}")
    if not isinstance(sep, str):
        raise TypeError(f"separator must be str, not {type(sep)}")
    if not isinstance(header_lines, int):
        raise TypeError(f"header_lines must int, not {type(header_lines)}")
    if split not in [None, 0, 1]:
        raise ValueError(f"split must be in [None, 0, 1], but is {split}")

    # infer the type and communicator for the loaded array
    dtype = types.canonical_heat_type(dtype)
    # determine the comm and device the data will be placed on
    device = devices.sanitize_device(device)
    comm = sanitize_comm(comm)

    file_size = os.stat(path).st_size
    rank = comm.rank
    size = comm.size

    if split is None:
        with open(path) as f:
            data = f.readlines()
            data = data[header_lines:]
            result = []
            for line in data:
                values = line.replace("\n", "").replace("\r", "").split(sep)
                values = [float(val) for val in values]
                result.append(values)
            resulting_tensor = factories.array(
                result, dtype=dtype, split=split, device=device, comm=comm
            )

    elif split == 0:
        counts, displs, _ = comm.counts_displs_shape((file_size, 1), 0)
        # in case lines are terminated with '\r\n' we need to skip 2 bytes later
        lineter_len = 1
        # Read a chunk of bytes and count the linebreaks
        with open(path, "rb") as f:
            f.seek(displs[rank], 0)
            line_starts = []
            r = f.read(counts[rank])
            for pos, line in enumerate(r):
                if chr(line) == "\n":
                    # Check if it is part of '\r\n'
                    if chr(r[pos - 1]) != "\r":
                        line_starts.append(pos + 1)
                elif chr(line) == "\r":
                    # check if file line is terminated by '\r\n'
                    if pos + 1 < len(r) and chr(r[pos + 1]) == "\n":
                        line_starts.append(pos + 2)
                        lineter_len = 2
                    else:
                        line_starts.append(pos + 1)

            if rank == 0:
                line_starts = [0] + line_starts

            # Find the correct starting point
            total_lines = torch.empty(size, dtype=torch.int32)
            comm.Allgather(torch.tensor([len(line_starts)], dtype=torch.int32), total_lines)

            cumsum = total_lines.cumsum(dim=0).tolist()
            start = next(i for i in range(size) if cumsum[i] > header_lines)
            if rank < start:
                line_starts = []
            if rank == start:
                rem = header_lines - (0 if start == 0 else cumsum[start - 1])
                line_starts = line_starts[rem:]

            # Determine the number of columns that each line consists of
            if len(line_starts) > 1:
                columns = 1
                for li in r[line_starts[0] : line_starts[1]]:
                    if chr(li) == sep:
                        columns += 1
            else:
                columns = 0

            columns = torch.tensor([columns], dtype=torch.int32)
            comm.Allreduce(MPI.IN_PLACE, columns, MPI.MAX)

            # Share how far the processes need to reed in their last line
            last_line = file_size
            if size - start > 1:
                if rank == start:
                    last_line = torch.empty(1, dtype=torch.int32)
                    comm.Recv(last_line, source=rank + 1)
                    last_line = last_line.item()
                elif rank == size - 1:
                    first_line = torch.tensor(displs[rank] + line_starts[0] - 1, dtype=torch.int32)
                    comm.Send(first_line, dest=rank - 1)
                elif start < rank < size - 1:
                    last_line = torch.empty(1, dtype=torch.int32)
                    first_line = torch.tensor(displs[rank] + line_starts[0] - 1, dtype=torch.int32)
                    comm.Send(first_line, dest=rank - 1)
                    comm.Recv(last_line, source=rank + 1)
                    last_line = last_line.item()

            # Create empty tensor and iteratively fill it with the values
            local_shape = (len(line_starts), columns)
            actual_length = 0
            local_tensor = torch.empty(
                local_shape, dtype=dtype.torch_type(), device=device.torch_device
            )

            for ind, start in enumerate(line_starts):
                if ind == len(line_starts) - 1:
                    f.seek(displs[rank] + start, 0)
                    line = f.read(last_line - displs[rank] - start)
                else:
                    line = r[start : line_starts[ind + 1] - lineter_len]
                # Decode byte array
                line = line.decode(encoding)
                if len(line) > 0:
                    sep_values = [float(val) for val in line.split(sep)]
                    local_tensor[actual_length] = torch.tensor(sep_values, dtype=dtype.torch_type())
                    actual_length += 1

        # In case there are some empty lines in the csv file
        local_tensor = local_tensor[:actual_length]

        total_actual_lines = torch.tensor(
            actual_length, dtype=torch.int64, device=local_tensor.device
        )
        comm.Allreduce(MPI.IN_PLACE, total_actual_lines, MPI.SUM)

        gshape = (total_actual_lines.item(), columns[0].item())

        resulting_tensor = DNDarray(
            local_tensor,
            gshape=gshape,
            dtype=dtype,
            split=0,
            device=device,
            comm=comm,
            balanced=None,
        )
        resulting_tensor.balance_()

    elif split == 1:
        data = []

        with open(path) as f:
            for i in range(header_lines):
                f.readline()
            line = f.readline()
            values = line.replace("\n", "").replace("\r", "").split(sep)
            values = [float(val) for val in values]
            rows = len(values)

            chunk, displs, _ = comm.counts_displs_shape((1, rows), 1)
            data.append(values[displs[rank] : displs[rank] + chunk[rank]])
            # Read file line by line till EOF reached
            for line in iter(f.readline, ""):
                values = line.replace("\n", "").replace("\r", "").split(sep)
                values = [float(val) for val in values]
                data.append(values[displs[rank] : displs[rank] + chunk[rank]])
        resulting_tensor = factories.array(data, dtype=dtype, is_split=1, device=device, comm=comm)

    return resulting_tensor


def save_csv(
    data: DNDarray,
    path: str,
    header_lines: Iterable[str] = None,
    sep: str = ",",
    decimals: int = -1,
    encoding: str = "utf-8",
    comm: Optional[Communication] = None,
    truncate: bool = True,
):
    """
    Saves data to CSV files. Only 2D data, all split axes.

    Parameters
    ----------
    data : DNDarray
        The DNDarray to be saved to CSV.
    path : str
        The path as a string.
    header_lines : Iterable[str]
        Optional iterable of str to prepend at the beginning of the file. No
        pound sign or any other comment marker will be inserted.
    sep : str
        The separator character used in this CSV.
    decimals: int
        Number of digits after decimal point.
    encoding : str
        The encoding to be used in this CSV.
    comm : Optional[Communication]
        An optional object of type Communication to be used.
    truncate : bool
        Whether to truncate an existing file before writing, i.e. fully overwrite it.
        The sane default is True. Setting it to False will not shorten files if
        needed and thus may leave garbage at the end of existing files.
    """
    if not isinstance(path, str):
        raise TypeError(f"path must be str, not {type(path)}")
    if not isinstance(sep, str):
        raise TypeError(f"separator must be str, not {type(sep)}")
    # check this to allow None
    if not isinstance(header_lines, Iterable) and header_lines is not None:
        raise TypeError(f"header_lines must Iterable[str], not {type(header_lines)}")
    if data.split not in [None, 0, 1]:
        raise ValueError(f"split must be in [None, 0, 1], but is {data.split}")

    if os.path.exists(path) and truncate:
        if data.comm.rank == 0:
            os.truncate(path, 0)
        # avoid truncating and writing at the same time
        data.comm.handle.Barrier()

    amode = MPI.MODE_WRONLY | MPI.MODE_CREATE
    csv_out = MPI.File.Open(data.comm.handle, path, amode)

    # will be needed as an additional offset later
    hl_displacement = 0
    if header_lines is not None:
        hl_displacement = sum(len(hl) for hl in header_lines)
        # count additions everywhere, but write only on rank 0, avoiding reduce op to share final hl_displacement
        for hl in header_lines:
            if not hl.endswith("\n"):
                hl = hl + "\n"
                hl_displacement = hl_displacement + 1
            if data.comm.rank == 0 and header_lines:
                csv_out.Write(hl.encode(encoding))

    # formatting and element width
    data_min = smin(data).item()  # at least min is used twice, so cache it here
    data_max = smax(data).item()
    sign = 1 if data_min < 0 else 0
    if abs(data_max) > 0 or abs(data_min) > 0:
        pre_point_digits = int(log10(max(abs(data_max), abs(data_min)))) + 1
    else:
        pre_point_digits = 1

    dec_sep = 1
    fmt = ""
    if types.issubdtype(data.dtype, types.integer):
        decimals = 0
        dec_sep = 0
        if sign == 1:
            fmt = "%%%-dd" % (pre_point_digits + 1)
        else:
            fmt = "%%%dd" % (pre_point_digits)
    elif types.issubdtype(data.dtype, types.floating):
        if decimals == -1:
            decimals = 7 if data.dtype is types.float32 else 15
        if sign == 1:
            fmt = "%%%-d.%df" % (pre_point_digits + decimals + 2, decimals)
        else:
            fmt = "%%%d.%df" % (pre_point_digits + decimals + 1, decimals)

    # sign + decimal separator + pre separator digits + decimals (post separator)
    item_size = decimals + dec_sep + sign + pre_point_digits
    # each item is one position larger than its representation, either b/c of separator or line break
    row_width = item_size + 1
    if len(data.shape) > 1:
        row_width = data.shape[1] * (item_size + 1)

    offset = hl_displacement  # all splits
    if data.split == 0:
        _, displs = data.counts_displs()
        offset = offset + displs[data.comm.rank] * row_width
    elif data.split == 1:
        _, displs = data.counts_displs()
        offset = offset + displs[data.comm.rank] * (item_size + 1)

    for i in range(data.lshape[0]):
        # if lshape is of the form (x,), then there will only be a single element per row
        if len(data.lshape) == 1:
            row = fmt % (data.larray[i])
        else:
            if data.lshape[1] == 0:
                break
            row = sep.join(fmt % (item) for item in data.larray[i])

        if (
            data.split is None
            or data.split == 0
            or displs[data.comm.rank] + data.lshape[1] == data.shape[1]
        ):
            row = row + "\n"
        else:
            row = row + sep

        if data.split is not None or data.comm.rank == 0:
            csv_out.Write_at(offset, row.encode("utf-8"))

        offset = offset + row_width

    csv_out.Close()
    data.comm.handle.Barrier()


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
    if not isinstance(path, str):
        raise TypeError(f"Expected path to be str, but was {type(path)}")
    extension = os.path.splitext(path)[-1].strip().lower()

    if extension in __HDF5_EXTENSIONS:
        if supports_hdf5():
            save_hdf5(data, path, *args, **kwargs)
        else:
            raise RuntimeError(f"hdf5 is required for file extension {extension}")
    elif extension in __NETCDF_EXTENSIONS:
        if supports_netcdf():
            save_netcdf(data, path, *args, **kwargs)
        else:
            raise RuntimeError(f"netcdf is required for file extension {extension}")
    elif extension in __CSV_EXTENSION:
        save_csv(data, path, *args, **kwargs)
    elif extension in __ZARR_EXTENSIONS:
        if supports_zarr():
            return save_zarr(data, path, *args, **kwargs)
        else:
            raise RuntimeError(f"Package zarr is required for file extension {extension}")
    else:
        raise ValueError(f"Unsupported file extension {extension}")


DNDarray.save = lambda self, path, *args, **kwargs: save(self, path, *args, **kwargs)
DNDarray.save.__doc__ = save.__doc__


def load_npy_from_path(
    path: str,
    dtype: datatype = types.int32,
    split: int = 0,
    device: Optional[str] = None,
    comm: Optional[Communication] = None,
) -> DNDarray:
    """
    Loads multiple .npy files into one DNDarray which will be returned. The data will be concatenated along the split axis provided as input.

    Parameters
    ----------
    path : str
        Path to the directory in which .npy-files are located.
    dtype : datatype, optional
        Data type of the resulting array.
    split : int
        Along which axis the loaded arrays should be concatenated.
    device : str, optional
        The device id on which to place the data, defaults to globally set default device.
    comm : Communication, optional
        The communication to use for the data distribution, default is 'heat.MPI_WORLD'
    """
    if not isinstance(path, str):
        raise TypeError(f"path must be str, not {type(path)}")
    elif split is not None and not isinstance(split, int):
        raise TypeError(f"split must be None or int, not {type(split)}")

    process_number = MPI_WORLD.size
    file_list = []
    for file in os.listdir(path):
        if fnmatch.fnmatch(file, "*.npy"):
            file_list.append(file)
    n_files = len(file_list)

    if n_files == 0:
        raise ValueError("No .npy Files were found")
    if (n_files < process_number) and (process_number > 1):
        raise RuntimeError("Number of processes can't exceed number of files")

    rank = MPI_WORLD.rank
    if rank < (n_files % process_number):
        n_for_procs = n_files // process_number + 1
        idx = rank * n_for_procs
    else:
        n_for_procs = n_files // process_number
        idx = rank * n_for_procs + (n_files % process_number)
    array_list = [np.load(path + "/" + element) for element in file_list[idx : idx + n_for_procs]]

    larray = np.concatenate(array_list, split)
    larray = torch.from_numpy(larray)

    x = factories.array(larray, dtype=dtype, device=device, is_split=split, comm=comm)
    return x


try:
    import pandas as pd
except ModuleNotFoundError:
    # pandas support is optional
    def supports_pandas() -> bool:
        """
        Returns ``True`` if pandas is installed , ``False`` otherwise.
        """
        return False

else:
    # add functions to visible exports
    __all__.extend(["load_csv_from_folder"])

    def supports_pandas() -> bool:
        """
        Returns ``True`` if pandas is installed, ``False`` otherwise.
        """
        return True

    def load_csv_from_folder(
        path: str,
        dtype: datatype = types.int32,
        split: int = 0,
        device: Optional[str] = None,
        comm: Optional[Communication] = None,
        func: Optional[callable] = None,
    ) -> DNDarray:
        """
        Loads multiple .csv files into one DNDarray which will be returned. The data will be concatenated along the split axis provided as input.

        Parameters
        ----------
        path : str
            Path to the directory in which .csv-files are located.
        dtype : datatype, optional
            Data type of the resulting array.
        split : int
            Along which axis the loaded arrays should be concatenated.
        device : str, optional
            The device id on which to place the data, defaults to globally set default device.
        comm : Communication, optional
            The communication to use for the data distribution, default is 'heat.MPI_WORLD'
        func : pandas.DataFrame, optional
            The function the files have to go through before being added to the array.
        """
        if not isinstance(path, str):
            raise TypeError(f"path must be str, not {type(path)}")
        elif split is not None and not isinstance(split, int):
            raise TypeError(f"split must be None or int, not {type(split)}")
        elif (func is not None) and not callable(func):
            raise TypeError("func needs to be a callable function or None")

        process_number = MPI_WORLD.size
        file_list = []
        for file in os.listdir(path):
            if fnmatch.fnmatch(file, "*.csv"):
                file_list.append(file)
        n_files = len(file_list)

        if n_files == 0:
            raise ValueError("No .csv Files were found")
        if (n_files < process_number) and (process_number > 1):
            raise RuntimeError("Number of processes can't exceed number of files")

        rank = MPI_WORLD.rank
        if rank < (n_files % process_number):
            n_for_procs = n_files // process_number + 1
            idx = rank * n_for_procs
        else:
            n_for_procs = n_files // process_number
            idx = rank * n_for_procs + (n_files % process_number)
        array_list = [
            (
                (func(pd.read_csv(path + "/" + element))).to_numpy()
                if ((func is not None) and (callable(func)))
                else (pd.read_csv(path + "/" + element)).to_numpy()
            )
            for element in file_list[idx : idx + n_for_procs]
        ]

        larray = np.concatenate(array_list, split)
        larray = torch.from_numpy(larray)
        x = factories.array(larray, dtype=dtype, device=device, is_split=split, comm=comm)
        return x


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
        split: int = 0,
        device: Optional[str] = None,
        comm: Optional[Communication] = None,
        slices: Union[None, slice, Iterable[Union[slice, None]]] = None,
        **kwargs,
    ) -> DNDarray:
        """
        Loads zarr-Format into DNDarray which will be returned.

        Parameters
        ----------
        path : str
            Path to the directory in which a .zarr-file is located.
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
        if not isinstance(path, str):
            raise TypeError(f"path must be str, not {type(path)}")
        if split is not None and not isinstance(split, int):
            raise TypeError(f"split must be None or int, not {type(split)}")
        if device is not None and not isinstance(device, str):
            raise TypeError(f"device must be None or str, not {type(split)}")
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

        arr: zarr.Array = zarr.open_array(store=path, **kwargs)
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
        device = devices.sanitize_device(device)
        comm = sanitize_comm(comm)

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
