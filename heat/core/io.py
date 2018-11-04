import os.path
import torch
import warnings

from .communication import MPI, MPI_WORLD
from . import types

__VALID_WRITE_MODES = frozenset(['w', 'a', 'r+'])
__HDF5_EXTENSIONS = frozenset(['.h5', '.hdf5'])
__NETCDF_EXTENSIONS = frozenset(['.nc', '.nc4', 'netcdf'])
__NETCDF_DIM_TEMPLATE = '{}_dim_{}'

__all__ = [
    'load',
    'save'
]

try:
    import h5py
except ImportError:
    # HDF5 support is optional
    def supports_hdf5():
        return False
else:
    # warn the user about serial hdf5
    if not h5py.get_config().mpi and MPI_WORLD.rank == 0:
        warnings.warn('h5py does not support parallel I/O, falling back to slower serial I/O', ImportWarning)

    # add functions to exports
    __all__.extend([
        'load_hdf5',
        'save_hdf5'
    ])

    def supports_hdf5():
        return True


    def load_hdf5(path, dataset, dtype=types.float32, split=None, comm=MPI_WORLD):
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
        group : mpi.group
            The communicator group to use for the data distribution

        Returns
        -------
        out : ht.tensor
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
            raise TypeError('path must be str, not {}'.format(type(path)))
        if not isinstance(dataset, str):
            raise TypeError('dataset must be str, not {}'.format(type(dataset)))
        if split is not None and not isinstance(split, int):
            raise TypeError('split must be None or int, not {}'.format(type(split)))

        # infer the type and communicator for the loaded array
        dtype = types.canonical_heat_type(dtype)

        # actually load the data from the HDF5 file
        with h5py.File(path, 'r') as handle:
            data = handle[dataset]
            gshape = tuple(data.shape)
            _, _, indices = comm.chunk(gshape, split)

            return tensor.tensor(torch.tensor(data[indices], dtype=dtype.torch_type()), gshape, dtype, split, comm)


    def save_hdf5(data, path, dataset, mode='w', **kwargs):
        """
        Saves data to an HDF5 file. Attempts to utilize parallel I/O if possible.

        Parameters
        ----------
        data : ht.tensor
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
        if not isinstance(data, tensor.tensor):
            raise TypeError('data must be heat tensor, not {}'.format(type(data)))
        if not isinstance(path, str):
            raise TypeError('path must be str, not {}'.format(type(path)))
        if not isinstance(dataset, str):
            raise TypeError('dataset must be str, not {}'.format(type(path)))

        # we only support a subset of possible modes
        if mode not in __VALID_WRITE_MODES:
            raise ValueError('mode was {}, not in possible modes {}'.format(mode, __VALID_WRITE_MODES))

        # chunk the data, if no split is set maximize parallel I/O and chunk first axis
        is_split = data.split is not None
        _, _, slices = data.comm.chunk(data.gshape, data.split if is_split else 0)

        # attempt to perform parallel I/O if possible
        if h5py.get_config().mpi:
            with h5py.File(path, mode, driver='mpio', comm=data.comm.handle) as handle:
                dset = handle.create_dataset(dataset, data.shape, **kwargs)
                dset[slices] = data._tensor__array if is_split else data._tensor__array[slices]

        # otherwise a single rank only write is performed in case of local data (i.e. no split)
        elif data.comm.rank == 0:
            with h5py.File(path, mode) as handle:
                dset = handle.create_dataset(dataset, data.shape, **kwargs)
                if is_split:
                    dset[slices] = data._tensor__array
                else:
                    dset[...] = data._tensor__array

            # ping next rank if it exists
            if is_split and data.comm.size > 1:
                data.comm.Isend([None, 0, MPI.INT], dest=1)
                data.comm.Recv([None, 0, MPI.INT], source=data.comm.size - 1)

        # no MPI, but split data is more tricky, we have to serialize the writes
        elif is_split:
            # wait for the previous rank to finish writing its chunk, then write own part
            data.comm.Recv([None, 0, MPI.INT], source=data.comm.rank - 1)
            with h5py.File(path, 'r+') as handle:
                handle[dataset][slices] = data._tensor__array

            # ping the next node in the communicator, wrap around to 0 to complete barrier behavior
            next_rank = (data.comm.rank + 1) % data.comm.size
            data.comm.Isend([None, 0, MPI.INT], dest=next_rank)

try:
    import netCDF4 as nc
except ImportError:
    def supports_netcdf():
        return False
else:
    # warn the user about serial netcdf
    if not nc.__has_nc_par__ and MPI_WORLD.rank == 0:
        warnings.warn('netCDF4 does not support parallel I/O, falling back to slower serial I/O', ImportWarning)

    # add functions to visible exports
    __all__.extend([
        'load_netcdf',
        'save_netcdf'
    ])

    def supports_netcdf():
        return True

    def load_netcdf(path, variable, dtype=types.float32, split=None, comm=MPI_WORLD):
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

        Returns
        -------
        out : ht.tensor
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
            raise TypeError('path must be str, not {}'.format(type(path)))
        if not isinstance(variable, str):
            raise TypeError('dataset must be str, not {}'.format(type(variable)))
        if split is not None and not isinstance(split, int):
            raise TypeError('split must be None or int, not {}'.format(type(split)))

        # infer the canonical heat datatype
        dtype = types.canonical_heat_type(dtype)

        # actually load the data
        with nc.Dataset(path, 'r', parallel=nc.__has_nc_par__, comm=comm.handle) as handle:
            data = handle[variable][:]
            gshape = tuple(data.shape)
            _, _, indices = comm.chunk(gshape, split)

            return tensor.tensor(torch.tensor(data[indices], dtype=dtype.torch_type()), gshape, dtype, split, comm)


    def save_netcdf(data, path, variable, mode='w', **kwargs):
        """
        Saves data to a netCDF4 file. Attempts to utilize parallel I/O if possible.

        Parameters
        ----------
        data : ht.tensor
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
        if not isinstance(data, tensor.tensor):
            raise TypeError('data must be heat tensor, not {}'.format(type(data)))
        if not isinstance(path, str):
            raise TypeError('path must be str, not {}'.format(type(path)))
        if not isinstance(variable, str):
            raise TypeError('variable must be str, not {}'.format(type(path)))

        # we only support a subset of possible modes
        if mode not in __VALID_WRITE_MODES:
            raise ValueError('mode was {}, not in possible modes {}'.format(mode, __VALID_WRITE_MODES))

        # chunk the data, if no split is set maximize parallel I/O and chunk first axis
        is_split = data.split is not None
        _, _, slices = data.comm.chunk(data.gshape, data.split if is_split else 0)

        # attempt to perform parallel I/O if possible
        if nc.__has_nc_par__:
            with nc.Dataset(path, mode, parallel=True, comm=data.comm.handle) as handle:
                dimension_names = []
                for dimension, elements in enumerate(data.shape):
                    name = __NETCDF_DIM_TEMPLATE.format(variable, dimension)
                    handle.createDimension(name, elements)
                    dimension_names.append(name)

                var = handle.createVariable(variable, data.dtype.char(), dimension_names, **kwargs)
                var[slices] = data._tensor__array if is_split else data._tensor__array[slices]

        # otherwise a single rank only write is performed in case of local data (i.e. no split)
        elif data.comm.rank == 0:
            with nc.Dataset(path, mode) as handle:
                dimension_names = []
                for dimension, elements in enumerate(data.shape):
                    name = __NETCDF_DIM_TEMPLATE.format(variable, dimension)
                    handle.createDimension(name, elements)
                    dimension_names.append(name)

                var = handle.createVariable(variable, data.dtype.char(), tuple(dimension_names), **kwargs)
                if is_split:
                    var[slices] = data._tensor__array
                else:
                    var[:] = data._tensor__array

            # ping next rank if it exists
            if is_split and data.comm.size > 1:
                data.comm.Isend([None, 0, MPI.INT], dest=1)
                data.comm.Recv([None, 0, MPI.INT], source=data.comm.size - 1)

        # no MPI, but data is split, we have to serialize the writes
        elif is_split:
            # wait for the previous rank to finish writing its chunk, then write own part
            data.comm.Recv([None, 0, MPI.INT], source=data.comm.rank - 1)
            with nc.Dataset(path, 'r+') as handle:
                handle[variable][slices] = data._tensor__array

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
    out : ht.tensor
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
        raise TypeError('Expected path to be str, but was {}'.format(type(path)))
    extension = os.path.splitext(path)[-1].strip().lower()

    if supports_hdf5() and extension in __HDF5_EXTENSIONS:
        return load_hdf5(path, *args, **kwargs)
    elif supports_netcdf() and extension in __NETCDF_EXTENSIONS:
        return load_netcdf(path, *args, **kwargs)
    else:
        raise ValueError('Unsupported file extension {}'.format(extension))


def save(data, path, *args, **kwargs):
    """
    Attempts to save data from a tensor to disk. Attempts to auto-detect the file format by determining the extension.

    Parameters
    ----------
    data : ht.tensor
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
        raise TypeError('Expected path to be str, but was {}'.format(type(path)))
    extension = os.path.splitext(path)[-1].strip().lower()

    if supports_hdf5() and extension in __HDF5_EXTENSIONS:
        save_hdf5(data, path, *args, **kwargs)
    elif supports_netcdf() and extension in __NETCDF_EXTENSIONS:
        save_netcdf(data, path, *args, **kwargs)
    else:
        raise ValueError('Unsupported file extension {}'.format(extension))


# tensor is imported at the very end to break circular dependency
from . import tensor
