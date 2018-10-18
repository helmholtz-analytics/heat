import os.path
import torch

from .communicator import mpi, MPICommunicator, NoneCommunicator
from . import tensor
from . import types

try:
    import h5py
except ImportError:
    # HDF5 support is optional
    pass
else:
    def load_hdf5(path, dataset, dtype=types.float32, split=None, group=mpi.group.WORLD):
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
        """
        if not isinstance(path, str):
            raise TypeError('path must be str, not {}'.format(type(path)))
        if not isinstance(dataset, str):
            raise TypeError('dataset must be str, not {}'.format(type(dataset)))
        if split is not None and not isinstance(split, int):
            raise TypeError('split must be None or int, not {}'.format(type(split)))

        # infer the type and communicator for the loaded array
        dtype = types.canonical_heat_type(dtype)
        comm = MPICommunicator(group) if split is not None else NoneCommunicator()

        # actually load the data from the HDF5 file
        with h5py.File(path, 'r') as handle:
            data = handle[dataset]
            gshape = tuple(data.shape)
            _, _, indices = comm.chunk(gshape, split)

            return tensor(torch.tensor(data[indices], dtype=dtype.torch_type()), gshape, dtype, split, comm)

try:
    import netCDF4
except ImportError:
    # netCDF support is optional
    pass
else:
    def load_netcdf(path, variable, dtype=types.float32, split=None, group=mpi.group.WORLD):
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
        group : mpi.group
            The communicator group to use for the data distribution

        Returns
        -------
        out : ht.tensor
            Data read from the NetCDF4 file.

        Raises
        -------
        TypeError
            If any of the input parameters are not of correct type
        """
        if not isinstance(path, str):
            raise TypeError('path must be str, not {}'.format(type(path)))
        if not isinstance(variable, str):
            raise TypeError('dataset must be str, not {}'.format(type(variable)))
        if split is not None and not isinstance(split, int):
            raise TypeError('split must be None or int, not {}'.format(type(split)))

        # infer the canonical heat datatype
        dtype = types.canonical_heat_type(dtype)
        comm = MPICommunicator(group) if split is not None else NoneCommunicator()

        # actually load the data
        try:
            handle = netCDF4.Dataset(path, 'r', parallel=True)
        except ValueError:
            handle = netCDF4.Dataset(path, 'r')

        data = handle[variable][:]
        gshape = tuple(data.shape)
        _, _, indices = comm.chunk(gshape, split)

        try:
            return tensor(torch.tensor(data[indices], dtype=dtype.torch_type()), gshape, dtype, split, comm)
        finally:
            handle.close()


def load(path, *args, **kwargs):
    """
    Attempts to load data from a file stored on disk. Attempts to auto-detect the file format by determining the
    extension.

    Parameters
    ----------
    path : str
        Path to the file to be read.

    Returns
    -------
    out : ht.tensor
        Data read from the file.

    Raises
    -------
    ValueError
        If the file extension is not understood or known.
    """
    extension = os.path.splitext(path)[-1].strip().lower()

    if (extension == '.h5' or extension == '.hdf5') and 'load_hdf5' in globals():
        return load_hdf5(path, *args, **kwargs)
    elif (extension == '.nc' or extension == '.nc4' or extension == '.netcdf') and 'load_netcdf' in globals():
        return load_netcdf(path, *args, **kwargs)
    else:
        raise ValueError('Unsupported file extension {}'.format(extension))
