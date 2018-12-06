from mpi4py import MPI
import abc
import os
import subprocess
import torch

from .stride_tricks import sanitize_axis

# check whether OpenMPI support CUDA-aware MPI
try:
    # OpenMPI
    if 'openmpi' in os.environ.get('MPI_SUFFIX', '').lower():
        buffer = subprocess.check_output(['ompi_info', '--parsable', '--all'])
        CUDA_AWARE_MPI = b'mpi_built_with_cuda_support:value:true' in buffer
    else:
        CUDA_AWARE_MPI = False
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
    # static mapping of torch types to the respective MPI type handle
    __mpi_type_mappings = {
        torch.uint8: MPI.UNSIGNED_CHAR,
        torch.int8: MPI.SIGNED_CHAR,
        torch.int16: MPI.SHORT_INT,
        torch.int32: MPI.INT,
        torch.int64: MPI.LONG,
        torch.float32: MPI.FLOAT,
        torch.float64: MPI.DOUBLE
    }

    def __init__(self, handle=MPI.COMM_WORLD):
        self.handle = handle
        self.rank = handle.Get_rank()
        self.size = handle.Get_size()

    def is_distributed(self):
        """
        Determines whether the communicator is distributed, i.e. handles more than one node.

        Returns
        -------
            distribution_flag : bool
                flag indicating whether the communicator contains distributed resources
        """
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

    @classmethod
    def get_type_and_size(cls, obj):
        """
        Determines the MPI data type and number of respective elements for the given tensor. In case the tensor is
        contiguous in memory, a native MPI data type can be used. Otherwise, a derived data type is automatically
        constructed using the storage information of the passed object.

        Parameters
        ----------
        obj : ht.tensor or torch.Tensor
            The object for which to construct the MPI data type and number of elements

        Returns
        -------
        type : MPI.Datatype
            The data type object
        elements : int
            The number of elements of the respective data type
        """
        mpi_type, elements = cls.__mpi_type_mappings[obj.dtype], torch.numel(obj)

        # simple case, continuous memory can be transmitted as is
        if obj.is_contiguous():
            return mpi_type, elements

        # non-continuous memory, e.g. after a transpose, has to be packed in derived MPI types
        elements = obj.shape[0]
        shape = obj.shape[1:]
        strides = [1] * len(shape)
        strides[0] = obj.stride()[-1]
        offsets = [obj.element_size() * stride for stride in obj.strides()[:-1]]

        # chain the types based on the
        for i in range(len(shape), -1, -1):
            mpi_type = mpi_type.Create_vector(shape[i], 1, strides[i]).Create_resized(0, offsets[i])
            mpi_type.Commit()

        return mpi_type, elements

    @classmethod
    def as_buffer(cls, obj):
        """
        Converts a passed HeAT or torch tensor into a memory buffer object with associated number of elements and MPI
        data type.

        Parameters
        ----------
        obj : ht.tensor or torch.Tensor
            The object to be converted into a buffer representation.

        Returns
        -------
        buffer : list[MPI.memory, int, MPI.Datatype]
            The buffer information of the passed tensor, ready to be passed as MPI send or receive buffer.
        """
        # unpack heat tensors, only the torch tensor is needed
        if isinstance(obj, tensor.tensor):
            obj = obj._tensor__array
        # non-torch tensors are assumed to support the buffer interface or will not be send
        if not isinstance(obj, torch.Tensor):
            return obj

        # ensure that the underlying memory is contiguous
        # may be improved by constructing an appropriate MPI derived data type?
        mpi_type, elements = cls.get_type_and_size(obj)

        # prepare the memory pointer
        # in case of GPUs, the memory has to be copied to host memory if cuda aware MPI is not supported
        pointer = obj.data_ptr() if CUDA_AWARE_MPI else obj.cpu().data_ptr()
        pointer += obj.storage_offset()

        return [MPI.memory.fromaddress(pointer, 0), elements, mpi_type]

    def convert_tensors(self, a_callable):
        """
        Constructs a decorator function for a given callable. The decorator ensures, that all the positional and keyword
        arguments to the passed callable are converted from HeAT or torch tensor to a buffer before invocation.

        Parameters
        ----------
        a_callable : callable
            The callable to be decorated.

        Returns
        -------
        wrapped : function
            The decorated callable.

        See Also
        --------
        as_buffer
        """
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
