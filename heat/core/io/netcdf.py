"""Reading and writing Heat arrays in the NetCDF4 format."""

from __future__ import annotations

import functools
import warnings
from types import EllipsisType
from typing import Any, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import torch

from .. import devices
from .. import types
from .._config import is_installed, netcdf_has_parallel, require_dependency
from ..communication import Communication, MPI, MPI_WORLD, sanitize_comm
from ..devices import Device
from ..dndarray import DNDarray
from ..stride_tricks import sanitize_axis
from ..types import datatype
from ._registry import register_format
from .utils import propagate_failure, sanitize_extension, sanitize_write_mode, serialized_write

__all__ = ["supports_netcdf"]

#: File extensions handled by this module.
EXTENSIONS = frozenset([".nc", ".nc4"])

#: Template for the names given to dimensions Heat has to create itself.
DIM_TEMPLATE = "{}_dim_{}"

#: Keys accepted for indexing a ``netCDF4.Variable``.
NetCDFKey = Union[int, slice, EllipsisType, bool, Iterable[Union[int, slice]]]


_AVAILABLE = is_installed("netCDF4")


def supports_netcdf() -> bool:
    """
    Returns ``True`` if Heat supports reading from and writing to netCDF4 files, ``False`` otherwise.

    .. deprecated:: 1.10.0
        Use :func:`heat.io.supports` instead, e.g. ``ht.io.supports("netcdf")``.

    Returns
    -------
    bool
        Whether netCDF4 is installed.
    """
    warnings.warn(
        "supports_netcdf() is deprecated and will be removed in a future release; "
        "use ht.io.supports('netcdf') instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return _AVAILABLE


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


if _AVAILABLE:
    __all__.extend(["load_netcdf", "save_netcdf"])

    def load_netcdf(
        path: str,
        variable: str,
        dtype: Type[datatype] = types.float32,
        split: Optional[int] = None,
        device: Optional[Union[str, Device]] = None,
        comm: Optional[Communication] = None,
    ) -> DNDarray:
        """
        Loads data from a NetCDF4 file.

        The data may be distributed among multiple processing nodes via the ``split`` flag.

        Parameters
        ----------
        path : str
            Path to the NetCDF4 file to be read.
        variable : str
            Name of the variable to be read.
        dtype : datatype, optional
            Data type of the resulting array.
        split : int or None, optional
            The axis along which the data is distributed among the processing cores.
        device : str or Device, optional
            The device id on which to place the data, defaults to globally set default device.
        comm : Communication, optional
            The communication to use for the data distribution, defaults to global default.

        Returns
        -------
        DNDarray
            The data read from the variable.

        Raises
        ------
        TypeError
            If any of the input parameters are not of correct type.
        ValueError
            If ``path`` does not have a netCDF extension.

        See Also
        --------
        :func:`save_netcdf` : Saves data to a NetCDF4 file.

        Examples
        --------
        >>> a = ht.load_netcdf("data.nc", variable="DATA")
        >>> a.shape
        [0/1] (5,)
        [1/1] (5,)
        >>> a.lshape
        [0/1] (5,)
        [1/1] (5,)
        >>> b = ht.load_netcdf("data.nc", variable="DATA", split=0)
        >>> b.shape
        [0/1] (5,)
        [1/1] (5,)
        >>> b.lshape
        [0/1] (3,)
        [1/1] (2,)
        """
        nc = require_dependency("netCDF4")
        _warn_if_serial()
        if not isinstance(path, str):
            raise TypeError(f"path must be str, not {type(path)}")
        if not isinstance(variable, str):
            raise TypeError(f"dataset must be str, not {type(variable)}")
        if split is not None and not isinstance(split, int):
            raise TypeError(f"split must be None or int, not {type(split)}")

        path = sanitize_extension(path, EXTENSIONS, "netcdf")
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
        dimension_names: Optional[Union[List[str], Tuple[str, ...], str]] = None,
        is_unlimited: bool = False,
        file_slices: NetCDFKey = slice(None),
        **kwargs: Any,
    ) -> None:
        """
        Saves data to a netCDF4 file.

        Uses parallel I/O if netCDF4 was built with parallel support, otherwise the
        processes write one after another.

        Parameters
        ----------
        data : DNDarray
            The data to be saved on disk.
        path : str
            Path to the netCDF4 file to be written.
        variable : str
            Name of the variable the data is saved to.
        mode : str, optional
            File access mode, one of ``'w'``, ``'a'``, ``'r+'``.
        dimension_names : list or tuple or str, optional
            Specifies the netCDF Dimensions used by the variable. Ignored if Variable already exists.
            Defaults to ``"<variable>_dim_<axis>"`` for every axis.
        is_unlimited : bool, optional
            If True, every dimension created for this variable (i.e. doesn't already exist) is unlimited. Already
            existing limited dimensions cannot be changed to unlimited and vice versa.
        file_slices : int, slice, Ellipsis, bool or Iterable[int or slice], optional
            Keys used to slice the netCDF Variable, i.e. the region of the variable ``data`` is
            written to. Any key accepted by :class:`netCDF4.Variable` indexing can be used.
            Defaults to the whole variable.
        **kwargs : Any
            Additional arguments passed to :meth:`netCDF4.Dataset.createVariable`.

        Raises
        ------
        TypeError
            If any of the input parameters are not of correct type.
        ValueError
            If ``path`` does not have a netCDF extension, the access mode is not understood, or
            the number of dimension names does not match the number of dimensions.

        See Also
        --------
        :func:`load_netcdf` : Loads data from a NetCDF4 file.

        Examples
        --------
        >>> x = ht.arange(100, split=0)
        >>> ht.save_netcdf(x, "data.nc", variable="DATA")
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
        path = sanitize_extension(path, EXTENSIONS, "netcdf")
        sanitize_write_mode(mode)

        failed = 0
        excep: Optional[Exception] = None
        # chunk the data, if no split is set maximize parallel I/O and chunk first axis
        is_split = data.split is not None
        _, _, slices = data.comm.chunk(data.gshape, data.split if is_split else 0)

        def __get_expanded_split(
            shape: Tuple[int, ...], expanded_shape: Tuple[int, ...], split: Optional[int]
        ) -> Optional[int]:
            """
            Returns the hypothetical split-axis of a dndarray of shape=shape and
            split=split if it was expanded to expandedShape by adding empty dimensions.

            Parameters
            ----------
            shape : tuple[int, ...]
                Shape of a DNDarray.
            expanded_shape : tuple[int, ...]
                Shape of hypothetical expanded DNDarray.
            split : int or None
                Split axis of the DNDarray.

            Returns
            -------
            int or None
                The split axis in ``expanded_shape``.

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
                return int(ex_ind_nonempty[split_sq])
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
            var: Any,
            var_slices: NetCDFKey,
            data: DNDarray,
            data_slices: Optional[Tuple[slice, ...]] = None,
        ) -> Tuple[Union[slice, np.ndarray], ...]:
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
            var : netCDF4.Variable
                Variable to which data is to be saved.
            var_slices : int, slice, Ellipsis, bool or Iterable[int or slice]
                Keys to pass to the set-operator.
            data : DNDarray
                Data to be saved.
            data_slices : tuple[slice, ...], optional
                As returned by the ``data.comm.chunk`` method. Computed if not given.

            Returns
            -------
            tuple[slice or numpy.ndarray, ...]
                The keys that write the local chunk of ``data`` with a single
                ``var.__setitem__`` call.
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
            new_slices: List[Union[slice, np.ndarray]] = []
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
            if out_split is not None and data.split is not None:  # add split-slice
                split_slice = new_slices[out_split]
                if isinstance(split_slice, slice):
                    start, stop, step = split_slice.start, split_slice.stop, split_slice.step
                    sliced = range(start, stop, step)[slices[data.split]]
                    new_slices[out_split] = slice(
                        None if sliced.start < 0 else sliced.start,
                        None if sliced.stop < 0 else sliced.stop,
                        sliced.step,
                    )
                    # new_slices[out_split] = sliced
                elif isinstance(split_slice, np.ndarray):
                    new_slices[out_split] = split_slice[slices[data.split]]
                else:
                    new_slices[out_split] = np.r_[split_slice][slices[data.split]]
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
            propagate_failure(data.comm, failed, excep)

        # otherwise the writes have to be serialized: rank 0 creates the file and writes
        # its own chunk, then hands a token around the ring
        else:

            def write_root() -> None:
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
                        var[__merge_slices(var, file_slices, data)] = data.larray.cpu()
                    else:
                        var[file_slices] = data.larray.cpu()

            def write_other() -> None:
                with nc.Dataset(path, "r+") as handle:
                    var = handle.variables[variable]
                    var.set_collective(False)  # not possible with non-parallel netcdf
                    var[__merge_slices(var, file_slices, data)] = data.larray.cpu()

            serialized_write(data.comm, is_split, write_root, write_other)


register_format(
    "netcdf",
    EXTENSIONS,
    dependency="netCDF4",
    loader=load_netcdf if _AVAILABLE else None,
    saver=save_netcdf if _AVAILABLE else None,
)
