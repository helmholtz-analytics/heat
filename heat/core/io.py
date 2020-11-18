import os.path

import torch
import warnings

from heat.core import factories
from .communication import MPI, MPI_WORLD, sanitize_comm
from . import devices
from .stride_tricks import sanitize_axis
from . import types

__VALID_WRITE_MODES = frozenset(["w", "a", "r+"])
__CSV_EXTENSION = frozenset([".csv"])
__HDF5_EXTENSIONS = frozenset([".h5", ".hdf5"])
__NETCDF_EXTENSIONS = frozenset([".nc", ".nc4", "netcdf"])
__NETCDF_DIM_TEMPLATE = "{}_dim_{}"

__all__ = ["load", "load_csv", "save"]


try:
    import h5py
except ImportError:
    # HDF5 support is optional
    def supports_hdf5():
        return False


else:
    # warn the user about serial hdf5
    if not h5py.get_config().mpi and MPI_WORLD.rank == 0:
        warnings.warn(
            "h5py does not support parallel I/O, falling back to slower serial I/O", ImportWarning
        )

    # add functions to exports
    __all__.extend(["load_hdf5", "save_hdf5"])

    def supports_hdf5():
        return True

    def load_hdf5(path, dataset, dtype=types.float32, split=None, device=None, comm=None):
        """
        Loads data from an HDF5 file. The data may be distributed among multiple processing nodes via the split flag.

        Parameters
        ----------
        path : str
            Path to the HDF5 file to be read.
        dataset : str
            Name of the dataset to be read.
        dtype : ht.dtype
            Data type of the resulting array; default: ht.float32.
        split : int, optional
            The axis along which the data is distributed among the processing cores.
        device : None or str, optional
            The device id on which to place the data, defaults to globally set default device.
        comm : Communication, optional
            The communication to use for the data distribution. defaults to MPI_COMM_WORLD.

        Returns
        -------
        out : ht.DNDarray
            Data read from the HDF5 file.

        Raises
        -------
        TypeError
            If any of the input parameters are not of correct type

        Examples
        --------
        >>> a = ht.load_hdf5('data.h5', dataset='DATA')
        >>> a.shape
        (5,)
        >>> a.lshape
        (5,)
        >>> b = ht.load_hdf5('data.h5', dataset='DATA', split=0)
        >>> b.shape
        (5,)
        >>> b.lshape
        (3,)
        """
        if not isinstance(path, str):
            raise TypeError("path must be str, not {}".format(type(path)))
        if not isinstance(dataset, str):
            raise TypeError("dataset must be str, not {}".format(type(dataset)))
        if split is not None and not isinstance(split, int):
            raise TypeError("split must be None or int, not {}".format(type(split)))

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

            return dndarray.DNDarray(data, gshape, dtype, split, device, comm, balanced)

    def save_hdf5(data, path, dataset, mode="w", **kwargs):
        """
        Saves data to an HDF5 file. Attempts to utilize parallel I/O if possible.

        Parameters
        ----------
        data : ht.DNDarray
            The data to be saved on disk.
        path : str
            Path to the HDF5 file to be written.
        dataset : str
            Name of the dataset the data is saved to.
        mode : str, one of 'w', 'a', 'r+'
            File access mode
        kwargs : dict
            additional arguments passed to the created dataset.

        Raises
        -------
        TypeError
            If any of the input parameters are not of correct type.
        ValueError
            If the access mode is not understood.

        Examples
        --------
        >>> a_range = ht.arange(100, split=0)
        >>> ht.save_hdf5(a_range, 'data.h5', dataset='DATA')
        """
        if not isinstance(data, dndarray.DNDarray):
            raise TypeError("data must be heat tensor, not {}".format(type(data)))
        if not isinstance(path, str):
            raise TypeError("path must be str, not {}".format(type(path)))
        if not isinstance(dataset, str):
            raise TypeError("dataset must be str, not {}".format(type(path)))

        # we only support a subset of possible modes
        if mode not in __VALID_WRITE_MODES:
            raise ValueError(
                "mode was {}, not in possible modes {}".format(mode, __VALID_WRITE_MODES)
            )

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


try:
    import netCDF4 as nc
except ImportError:

    def supports_netcdf():
        return False


else:
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

    # add functions to visible exports
    __all__.extend(["load_netcdf", "save_netcdf"])

    def supports_netcdf():
        return True

    def load_netcdf(path, variable, dtype=types.float32, split=None, device=None, comm=None):
        """
        Loads data from a NetCDF4 file. The data may be distributed among multiple processing nodes via the split flag.

        Parameters
        ----------
        path : str
            Path to the NetCDF4 file to be read.
        variable : str
            Name of the variable to be read.
        dtype : ht.dtype
            Data type of the resulting array; default: ht.float32.
        split : int, optional
            The axis along which the data is distributed among the processing cores.
        comm : ht.Communication, optional
            The communication to use for the data distribution. defaults to MPI_COMM_WORLD.
        device : None or str, optional
            The device id on which to place the data, defaults to globally set default device.

        Returns
        -------
        out : ht.DNDarray
            Data read from the NetCDF4 file.

        Raises
        -------
        TypeError
            If any of the input parameters are not of correct type

        Examples
        --------
        >>> a = ht.load_netcdf('data.nc', variable='DATA')
        >>> a.shape
        (5,)
        >>> a.lshape
        (5,)
        >>> b = ht.load_netcdf('data.nc', variable='DATA', split=0)
        >>> b.shape
        (5,)
        >>> b.lshape
        (3,)
        """
        if not isinstance(path, str):
            raise TypeError("path must be str, not {}".format(type(path)))
        if not isinstance(variable, str):
            raise TypeError("dataset must be str, not {}".format(type(variable)))
        if split is not None and not isinstance(split, int):
            raise TypeError("split must be None or int, not {}".format(type(split)))

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

            return dndarray.DNDarray(data, gshape, dtype, split, device, comm, balanced)

    def save_netcdf(data, path, variable, mode="w", **kwargs):
        """
        Saves data to a netCDF4 file. Attempts to utilize parallel I/O if possible.

        Parameters
        ----------
        data : ht.DNDarray
            The data to be saved on disk.
        path : str
            Path to the netCDF4 file to be written.
        variable : str
            Name of the variable the data is saved to.
        mode : str, one of 'w', 'a', 'r+'
            File access mode
        kwargs : dict
            additional arguments passed to the created dataset.

        Raises
        -------
        TypeError
            If any of the input parameters are not of correct type.
        ValueError
            If the access mode is not understood.

        Examples
        --------
        >>> a_range = ht.arange(100, split=0)
        >>> ht.save_netcdf(a_range, 'data.nc', dataset='DATA')
        """
        if not isinstance(data, dndarray.DNDarray):
            raise TypeError("data must be heat tensor, not {}".format(type(data)))
        if not isinstance(path, str):
            raise TypeError("path must be str, not {}".format(type(path)))
        if not isinstance(variable, str):
            raise TypeError("variable must be str, not {}".format(type(path)))

        # we only support a subset of possible modes
        if mode not in __VALID_WRITE_MODES:
            raise ValueError(
                "mode was {}, not in possible modes {}".format(mode, __VALID_WRITE_MODES)
            )

        # chunk the data, if no split is set maximize parallel I/O and chunk first axis
        is_split = data.split is not None
        _, _, slices = data.comm.chunk(data.gshape, data.split if is_split else 0)

        # attempt to perform parallel I/O if possible
        if __nc_has_par:
            with nc.Dataset(path, mode, parallel=True, comm=data.comm.handle) as handle:
                dimension_names = []
                for dimension, elements in enumerate(data.shape):
                    name = __NETCDF_DIM_TEMPLATE.format(variable, dimension)
                    handle.createDimension(name, elements)
                    dimension_names.append(name)

                var = handle.createVariable(variable, data.dtype.char(), dimension_names, **kwargs)
                var[slices] = data.larray.cpu() if is_split else data.larray[slices].cpu()

        # otherwise a single rank only write is performed in case of local data (i.e. no split)
        elif data.comm.rank == 0:
            with nc.Dataset(path, mode) as handle:
                dimension_names = []
                for dimension, elements in enumerate(data.shape):
                    name = __NETCDF_DIM_TEMPLATE.format(variable, dimension)
                    handle.createDimension(name, elements)
                    dimension_names.append(name)

                var = handle.createVariable(
                    variable, data.dtype.char(), tuple(dimension_names), **kwargs
                )
                if is_split:
                    var[slices] = data.larray.cpu()
                else:
                    var[:] = data.larray.cpu()

            # ping next rank if it exists
            if is_split and data.comm.size > 1:
                data.comm.Isend([None, 0, MPI.INT], dest=1)
                data.comm.Recv([None, 0, MPI.INT], source=data.comm.size - 1)

        # no MPI, but data is split, we have to serialize the writes
        elif is_split:
            # wait for the previous rank to finish writing its chunk, then write own part
            data.comm.Recv([None, 0, MPI.INT], source=data.comm.rank - 1)
            with nc.Dataset(path, "r+") as handle:
                handle[variable][slices] = data.larray.cpu()

            # ping the next node in the communicator, wrap around to 0 to complete barrier behavior
            next_rank = (data.comm.rank + 1) % data.comm.size
            data.comm.Isend([None, 0, MPI.INT], dest=next_rank)


def load(path, *args, **kwargs):
    """
    Attempts to load data from a file stored on disk. Attempts to auto-detect the file format by determining the
    extension.

    Parameters
    ----------
    path : str
        Path to the file to be read.
    args/kwargs : list/dict
        additional options passed to the particular functions.

    Returns
    -------
    out : ht.DNDarray
        Data read from the file.

    Raises
    -------
    ValueError
        If the file extension is not understood or known.

    Examples
    --------
    >>> ht.load('data.h5', dataset='DATA')
    tensor([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981])
    >>> ht.load('data.nc', variable='DATA')
    tensor([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981])
    """
    if not isinstance(path, str):
        raise TypeError("Expected path to be str, but was {}".format(type(path)))
    extension = os.path.splitext(path)[-1].strip().lower()

    if extension in __CSV_EXTENSION:
        return load_csv(path, *args, **kwargs)
    elif supports_hdf5() and extension in __HDF5_EXTENSIONS:
        return load_hdf5(path, *args, **kwargs)
    elif supports_netcdf() and extension in __NETCDF_EXTENSIONS:
        return load_netcdf(path, *args, **kwargs)
    else:
        raise ValueError("Unsupported file extension {}".format(extension))


def load_csv(
    path,
    header_lines=0,
    sep=",",
    dtype=types.float32,
    encoding="UTF-8",
    split=None,
    device=None,
    comm=MPI_WORLD,
):
    """
    Loads data from a CSV file. The data will be distributed along the 0 axis.

    Parameters
    ----------
    path : str
        Path to the CSV file to be read.
    header_lines : int, optional
        The number of columns at the beginning of the file that should not be considered as data.
        default: 0.
    sep : str, optional
        The single char or string that separates the values in each row.
        default: ';'
    dtype : ht.dtype, optional
        Data type of the resulting array;
        default: ht.float32.
    encoding : str, optional
        The type of encoding which will be used to interpret the lines of the csv file as strings.
        default: 'UTF-8'
    split : None, 0, 1 : optional
        Along which axis the resulting tensor should be split.
        Default is None which means each node will have the full tensor.
    device : None or str, optional
        The device id on which to place the data, defaults to globally set default device.
    comm : Communication, optional
        The communication to use for the data distribution. defaults to MPI_COMM_WORLD.

    Returns
    -------
    out : ht.DNDarray
        Data read from the CSV file.

    Raises
    -------
    TypeError
        If any of the input parameters are not of correct type

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
        raise TypeError("path must be str, not {}".format(type(path)))
    if not isinstance(sep, str):
        raise TypeError("separator must be str, not {}".format(type(sep)))
    if not isinstance(header_lines, int):
        raise TypeError("header_lines must int, not {}".format(type(header_lines)))
    if split not in [None, 0, 1]:
        raise ValueError("split must be in [None, 0, 1], but is {}".format(split))

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
            for i, line in enumerate(data):
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
                    if not chr(r[pos - 1]) == "\r":
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


def save(data, path, *args, **kwargs):
    """
    Attempts to save data from a tensor to disk. Attempts to auto-detect the file format by determining the extension.

    Parameters
    ----------
    data : ht.DNDarray
        The tensor holding the data to be stored
    path : str
        Path to the file to be stored.
    args/kwargs : list/dict
        additional options passed to the particular functions.

    Raises
    -------
    ValueError
        If the file extension is not understood or known.

    Examples
    --------
    >>> a_range = ht.arange(100, split=0)
    >>> ht.save(a_range, 'data.h5', 'DATA', mode='a')
    >>> ht.save(a_range, 'data.nc', 'DATA', mode='w')
    """
    if not isinstance(path, str):
        raise TypeError("Expected path to be str, but was {}".format(type(path)))
    extension = os.path.splitext(path)[-1].strip().lower()

    if supports_hdf5() and extension in __HDF5_EXTENSIONS:
        save_hdf5(data, path, *args, **kwargs)
    elif supports_netcdf() and extension in __NETCDF_EXTENSIONS:
        save_netcdf(data, path, *args, **kwargs)
    else:
        raise ValueError("Unsupported file extension {}".format(extension))


# tensor is imported at the very end to break circular dependency
from . import dndarray
