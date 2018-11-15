from mpi4py import MPI
import abc
import subprocess
import torch

from .stride_tricks import sanitize_axis

# check whether OpenMPI support CUDA-aware MPI
try:
    buffer = subprocess.check_output(['ompi_info', '--parsable', '--all'])
    CUDA_AWARE_MPI = b'mpi_built_with_cuda_support:value:true' in buffer
except FileNotFoundError:
    CUDA_AWARE_MPI = False


class Communication(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def is_distributed():
        pass

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def chunk(self, shape, split):
        """
        Calculates the chunk of data that will be assigned to this compute node given a global data shape and a split
        axis.
        Parameters
        ----------
        shape : tuple of ints
            the global shape of the data to be split
        split : int
            the axis along which to chunk the data
        Returns
        -------
        offset : int
            the offset in the split dimension
        local_shape : tuple of ints
            the resulting local shape if the global input shape is chunked on the split axis
        slices : tuple of slices
            the chunk slices with respect to the given shape
        """
        pass


class MPICommunication(Communication):
    def __init__(self, handle=MPI.COMM_WORLD):
        self.handle = handle
        self.rank = handle.Get_rank()
        self.size = handle.Get_size()

    def is_distributed(self):
        return self.size > 1

    def chunk(self, shape, split):
        """
        Calculates the chunk of data that will be assigned to this compute node given a global data shape and a split
        axis.

        Parameters
        ----------
        shape : tuple of ints
            the global shape of the data to be split
        split : int
            the axis along which to chunk the data

        Returns
        -------
        offset : int
            the offset in the split dimension
        local_shape : tuple of ints
            the resulting local shape if the global input shape is chunked on the split axis
        slices : tuple of slices
            the chunk slices with respect to the given shape
        """
        # ensure the split axis is valid, we actually do not need it
        split = sanitize_axis(shape, split)
        if split is None:
            return 0, shape, tuple(slice(0, end) for end in shape)

        dims = len(shape)
        size = shape[split]
        chunk = size // self.size
        remainder = size % self.size

        if remainder > self.rank:
            chunk += 1
            start = self.rank * chunk
        else:
            start = self.rank * chunk + remainder
        end = start + chunk

        return start, \
            tuple(shape[i] if i != split else end - start for i in range(dims)), \
            tuple(slice(0, shape[i]) if i != split else slice(start, end) for i in range(dims))

    @staticmethod
    def as_buffer(obj):
        if isinstance(obj, tensor.tensor):
            obj = obj._tensor__array
        if not isinstance(obj, torch.Tensor):
            return obj

        pointer = obj.data_ptr() if CUDA_AWARE_MPI else obj.cpu().data_ptr()

        return MPI.memory.fromaddress(pointer, obj.element_size() * torch.numel(obj))

    def convert_tensors(self, a_callable):
        def wrapped(*args, **kwargs):
            args = list(args)
            for index, arg in enumerate(args):
                args[index] = self.as_buffer(arg)
            for key, arg in kwargs.items():
                kwargs[key] = self.as_buffer(arg)

            return a_callable(*args, **kwargs)

        return wrapped

    def __getattr__(self, name):
        return self.convert_tensors(getattr(self.handle, name))


MPI_WORLD = MPICommunication()
MPI_SELF = MPICommunication(MPI.COMM_SELF)

# tensor is imported at the very end to break circular dependency
from . import tensor
