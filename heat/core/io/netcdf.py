"""Reading and writing Heat arrays in the NetCDF4 format."""

from __future__ import annotations

import functools
import warnings
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from .. import devices
from .. import types
from .._config import is_installed, netcdf_has_parallel, require_dependency
from ..communication import Communication, MPI, MPI_WORLD, sanitize_comm
from ..dndarray import DNDarray
from ..stride_tricks import sanitize_axis
from ..types import datatype
from ._registry import register_format
from .utils import sanitize_write_mode

__all__ = ["supports_netcdf"]

#: File extensions handled by this module.
EXTENSIONS = frozenset([".nc", ".nc4"])

#: Template for the names given to dimensions Heat has to create itself.
DIM_TEMPLATE = "{}_dim_{}"


def supports_netcdf() -> bool:
    """
    Returns ``True`` if Heat supports reading from and writing to netCDF4 files, ``False`` otherwise.
    """
    return is_installed("netCDF4")


@functools.lru_cache(maxsize=None)
def _warn_if_serial() -> None:
    """
    Warns once, on rank 0, if the installed netCDF4 cannot do parallel I/O.
    """
    if not netcdf_has_parallel() and MPI_WORLD.rank == 0:
        warnings.warn(
            "netCDF4 does not support parallel I/O, falling back to slower serial I/O",
            ImportWarning,
        )


if supports_netcdf():
    __all__.extend(["load_netcdf", "save_netcdf"])

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
        nc = require_dependency("netCDF4")
        _warn_if_serial()
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
        with nc.Dataset(path, "r", parallel=netcdf_has_parallel(), comm=comm.handle) as handle:
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
        nc = require_dependency("netCDF4")
        _warn_if_serial()
        if not isinstance(data, DNDarray):
            raise TypeError(f"data must be heat tensor, not {type(data)}")
        if not isinstance(path, str):
            raise TypeError(f"path must be str, not {type(path)}")
        if not isinstance(variable, str):
            raise TypeError(f"variable must be str, not {type(path)}")
        if dimension_names is None:
            dimension_names = [
                DIM_TEMPLATE.format(variable, dim) for dim, _ in enumerate(data.shape)
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
        sanitize_write_mode(mode)

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
        if netcdf_has_parallel():
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

register_format(
    "netcdf",
    EXTENSIONS,
    dependency="netCDF4",
    loader=load_netcdf if supports_netcdf() else None,
    saver=save_netcdf if supports_netcdf() else None,
)
