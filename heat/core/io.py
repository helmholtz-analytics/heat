"""Enables parallel I/O with data on disk."""
from __future__ import annotations

import os.path
from math import log10
import numpy as np
import torch
import warnings

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

__all__ = ["load", "load_csv", "save_csv", "save", "supports_hdf5", "supports_netcdf"]

try:
    import h5py
except ImportError:
    # HDF5 support is optional
    def supports_hdf5() -> bool:
        """
        Returns ``True`` if HeAT supports reading from and writing to HDF5 files, ``False`` otherwise.
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
        Returns ``True`` if HeAT supports reading from and writing to HDF5 files, ``False`` otherwise.
        """
        return True

    def load_hdf5(
        path: str,
        dataset: str,
        dtype: datatype = types.float32,
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
        split : int or None, optional
            The axis along which the data is distributed among the processing cores.
        device : str, optional
            The device id on which to place the data, defaults to globally set default device.
        comm : Communication, optional
            The communication to use for the data distribution.

        Raises
        -------
        TypeError
            If any of the input parameters are not of correct type

        Examples
        --------
        >>> a = ht.load_hdf5('data.h5', dataset='DATA')
        >>> a.shape
        [0/2] (5,)
        [1/2] (5,)
        >>> a.lshape
        [0/2] (5,)
        [1/2] (5,)
        >>> b = ht.load_hdf5('data.h5', dataset='DATA', split=0)
        >>> b.shape
        [0/2] (5,)
        [1/2] (5,)
        >>> b.lshape
        [0/2] (3,)
        [1/2] (2,)
        """
        if not isinstance(path, str):
            raise TypeError(f"path must be str, not {type(path)}")
        elif not isinstance(dataset, str):
            raise TypeError("dataset must be str, not {}".format(type(dataset)))
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
            gshape = tuple(data.shape)
            dims = len(gshape)
            split = sanitize_axis(gshape, split)
            _, _, indices = comm.chunk(gshape, split)
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
        -------
        TypeError
            If any of the input parameters are not of correct type.
        ValueError
            If the access mode is not understood.

        Examples
        --------
        >>> x = ht.arange(100, split=0)
        >>> ht.save_hdf5(x, 'data.h5', dataset='DATA')
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


try:
    import netCDF4 as nc
except ImportError:
    # netCDF4 support is optional
    def supports_netcdf() -> bool:
        """
        Returns ``True`` if HeAT supports reading from and writing to netCDF4 files, ``False`` otherwise.
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
        Returns ``True`` if HeAT supports reading from and writing to netCDF4 files, ``False`` otherwise.
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
        -------
        TypeError
            If any of the input parameters are not of correct type.

        Examples
        --------
        >>> a = ht.load_netcdf('data.nc', variable='DATA')
        >>> a.shape
        [0/2] (5,)
        [1/2] (5,)
        >>> a.lshape
        [0/2] (5,)
        [1/2] (5,)
        >>> b = ht.load_netcdf('data.nc', variable='DATA', split=0)
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
        -------
        TypeError
            If any of the input parameters are not of correct type.
        ValueError
            If the access mode is not understood or if the number of dimension names does not match the number of
            dimensions.

        Examples
        --------
        >>> x = ht.arange(100, split=0)
        >>> ht.save_netcdf(x, 'data.nc', dataset='DATA')
        """
        if not isinstance(data, DNDarray):
            raise TypeError("data must be heat tensor, not {}".format(type(data)))
        if not isinstance(path, str):
            raise TypeError("path must be str, not {}".format(type(path)))
        if not isinstance(variable, str):
            raise TypeError("variable must be str, not {}".format(type(path)))
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
            raise ValueError(
                "{0} names given for {1} dimensions".format(len(dimension_names), len(data.shape))
            )

        # we only support a subset of possible modes
        if mode not in __VALID_WRITE_MODES:
            raise ValueError(
                "mode was {}, not in possible modes {}".format(mode, __VALID_WRITE_MODES)
            )

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
            -------
            ValueError
                If resulting shapes do not match.
            """
            if np.prod(shape) != np.prod(expanded_shape):
                raise ValueError(
                    "Shapes %s and %s do not have the same size" % (shape, expanded_shape)
                )
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
                    "Shapes %s and %s differ in non-empty dimensions" % (shape, expanded_shape)
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
            This method allows replacing:
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
            excep.args = "raised by process rank {}".format(failed - 1), *excep.args
            raise excep from None  # raise the same error but without traceback
            # because that is on a different process

    DNDarray.save_netcdf = lambda self, path, variable, mode="w", **kwargs: save_netcdf(
        self, path, variable, mode, **kwargs
    )
    DNDarray.save_netcdf.__doc__ = save_netcdf.__doc__


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
    -------
    ValueError
        If the file extension is not understood or known.
    RuntimeError
        If the optional dependency for a file extension is not available.

    Examples
    --------
    >>> ht.load('data.h5', dataset='DATA')
    DNDarray([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.load('data.nc', variable='DATA')
    DNDarray([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981], dtype=ht.float32, device=cpu:0, split=None)
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
    -------
    TypeError
        If any of the input parameters are not of correct type.

    Examples
    --------
    >>> import heat as ht
    >>> a = ht.load_csv('data.csv')
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
    >>> b = ht.load_csv('data.csv', header_lines=10)
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
            for pos, l in enumerate(r):
                if chr(l) == "\n":
                    # Check if it is part of '\r\n'
                    if chr(r[pos - 1]) != "\r":
                        line_starts.append(pos + 1)
                elif chr(l) == "\r":
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

        resulting_tensor = factories.array(
            local_tensor, dtype=dtype, is_split=0, device=device, comm=comm
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
    -------
    ValueError
        If the file extension is not understood or known.
    RuntimeError
        If the optional dependency for a file extension is not available.

    Examples
    --------
    >>> x = ht.arange(100, split=0)
    >>> ht.save(x, 'data.h5', 'DATA', mode='a')
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
    else:
        raise ValueError(f"Unsupported file extension {extension}")


DNDarray.save = lambda self, path, *args, **kwargs: save(self, path, *args, **kwargs)
DNDarray.save.__doc__ = save.__doc__
