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
    def loadh5(path, dataset, dtype=types.float32, split=None, group=mpi.group.WORLD):
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
