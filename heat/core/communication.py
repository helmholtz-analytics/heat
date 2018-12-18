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
    def mpi_type_and_elements_of(cls, obj):
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
        offsets = [obj.element_size() * stride for stride in obj.stride()[:-1]]

        # chain the types based on the
        for i in range(len(shape) - 1, -1, -1):
            mpi_type = mpi_type.Create_vector(shape[i], 1, strides[i]).Create_resized(0, offsets[i])
            mpi_type.Commit()

        return mpi_type, elements

    @classmethod
    def as_mpi_memory(cls, obj):
        """
        Converts the passed Torch tensor into an MPI compatible memory view.

        Parameters
        ----------
        obj : torch.Tensor
            The tensor to be converted into a MPI memory view.

        Returns
        -------
        mpi_memory : MPI.memory
            The MPI memory objects of the passed tensor.
        """
        # in case of GPUs, the memory has to be copied to host memory if CUDA-aware MPI is not supported
        pointer = obj.data_ptr() if CUDA_AWARE_MPI else obj.cpu().data_ptr()
        pointer += obj.storage_offset()

        return MPI.memory.fromaddress(pointer, 0)

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
        mpi_type, elements = cls.mpi_type_and_elements_of(obj)

        return [cls.as_mpi_memory(obj), elements, mpi_type]

    def __recv(self, func, buf, source, tag, status):
        if isinstance(buf, tensor.tensor):
            buf = buf._tensor__array
        if isinstance(buf, torch.Tensor):
            buf = self.as_buffer(buf)

        return func(buf, source, tag, status)

    def Irecv(self, buf, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=None):
        return self.__recv(self.handle.Irecv, buf, source, tag, status)
    Irecv.__doc__ = MPI.Comm.Irecv.__doc__

    def Recv(self, buf, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=None):
        return self.__recv(self.handle.Recv, buf, source, tag, status)
    Recv.__doc__ = MPI.Comm.Recv.__doc__

    def __send(self, func, buf, dest, tag):
        if isinstance(buf, tensor.tensor):
            buf = buf._tensor__array
        if isinstance(buf, torch.Tensor):
            buf = self.as_buffer(buf)

        return func(buf, dest, tag)

    def Bsend(self, buf, dest, tag=0):
        return self.__send(self.handle.Bsend, buf, dest, tag)
    Bsend.__doc__ = MPI.Comm.Bsend.__doc__

    def Ibsend(self, buf, dest, tag=0):
        return self.__send(self.handle.Ibsend, buf, dest, tag)
    Ibsend.__doc__ = MPI.Comm.Ibsend.__doc__

    def Irsend(self, buf, dest, tag=0):
        return self.__send(self.handle.Irsend, buf, dest, tag)
    Irsend.__doc__ = MPI.Comm.Irsend.__doc__

    def Isend(self, buf, dest, tag=0):
        return self.__send(self.handle.Isend, buf, dest, tag)
    Isend.__doc__ = MPI.Comm.Isend.__doc__

    def Issend(self, buf, dest, tag=0):
        return self.__send(self.handle.Issend, buf, dest, tag)
    Issend.__doc__ = MPI.Comm.Issend.__doc__

    def Rsend(self, buf, dest, tag=0):
        return self.__send(self.handle.Rsend, buf, dest, tag)
    Rsend.__doc__ = MPI.Comm.Rsend.__doc__

    def Ssend(self, buf, dest, tag=0):
        return self.__send(self.handle.Ssend, buf, dest, tag)
    Ssend.__doc__ = MPI.Comm.Ssend.__doc__

    def Send(self, buf, dest, tag=0):
        return self.__send(self.handle.Send, buf, dest, tag)
    Send.__doc__ = MPI.Comm.Send.__doc__

    def __broadcast(self, func, buf, root):
        # unpack the buffer if it is a HeAT tensor
        if isinstance(buf, tensor.tensor):
            buf = buf._tensor__array
        # convert torch tensors to MPI memory buffers
        if isinstance(buf, torch.Tensor):
            buf = self.as_buffer(buf)

        return func(buf, root)

    def Bcast(self, buf, root=0):
        return self.__broadcast(self.handle.Bcast, buf, root)
    Bcast.__doc__ = MPI.Comm.Bcast.__doc__

    def Ibcast(self, buf, root=0):
        return self.__broadcast(self.handle.Ibcast, buf, root)
    Ibcast.__doc__ = MPI.Comm.Ibcast.__doc__

    def __collective_single_type(self, func, sendbuf, recvbuf, *args, **kwargs):
        # unpack the receive buffer if it is a HeAT tensor
        if isinstance(recvbuf, tensor.tensor):
            recvbuf = recvbuf._tensor__array
        # unpack the send buffer if it is a HeAT tensor
        if isinstance(sendbuf, tensor.tensor):
            sendbuf = sendbuf._tensor__array

        # determine whether the buffers are torch tensors
        sendbuf_is_torch = isinstance(sendbuf, torch.Tensor)
        recvbuf_is_torch = isinstance(recvbuf, torch.Tensor)

        # harmonize the input and output buffers
        # MPI requires send and receive buffers to be of same type and length. If the torch tensors are either not both
        # contiguous or differently strided, they have to be made matching (if possible) first.
        if sendbuf_is_torch and recvbuf_is_torch:
            # convert the send buffer to a pointer, number of elements and type are identical to the receive buffer
            dummy = sendbuf.contiguous()  # make a contiguous copy and reassign the storage, old will be collected
            sendbuf.set_(dummy.storage(), dummy.storage_offset(), size=dummy.shape, stride=dummy.stride())
            sendbuf = self.as_buffer(sendbuf)

            # nothing matches, the buffers have to be made contiguous
            dummy = recvbuf.contiguous()
            recvbuf.set_(dummy.storage(), dummy.storage_offset(), size=dummy.shape, stride=dummy.stride())
            recvbuf = [self.as_mpi_memory(recvbuf), sendbuf[1], sendbuf[2]]

        elif sendbuf_is_torch:
            sendbuf = self.as_buffer(sendbuf)
        elif recvbuf_is_torch:
            recvbuf = self.as_buffer(recvbuf)

        # perform the actual reduction operation
        return func(sendbuf, recvbuf, *args, **kwargs)

    def Exscan(self, sendbuf, recvbuf, op=MPI.SUM):
        return self.__collective_single_type(self.handle.Exscan, sendbuf, recvbuf, op)
    Exscan.__doc__ = MPI.COMM_WORLD.Exscan.__doc__

    def Iexscan(self, sendbuf, recvbuf, op=MPI.SUM):
        return self.__collective_single_type(self.handle.Iexscan, sendbuf, recvbuf, op)
    Iexscan.__doc__ = MPI.COMM_WORLD.Iexscan.__doc__

    def Iscan(self, sendbuf, recvbuf, op=MPI.SUM):
        return self.__collective_single_type(self.handle.Iscan, sendbuf, recvbuf, op)
    Iscan.__doc__ = MPI.COMM_WORLD.Iscan.__doc__

    def Ireduce(self, sendbuf, recvbuf, op=MPI.SUM, root=0):
        return self.__collective_single_type(self.handle.Ireduce, sendbuf, recvbuf, op, root)
    Ireduce.__doc__ = MPI.Comm.Ireduce.__doc__

    def Reduce(self, sendbuf, recvbuf, op=MPI.SUM, root=0):
        return self.__collective_single_type(self.handle.Reduce, sendbuf, recvbuf, op, root)
    Reduce.__doc__ = MPI.Comm.Reduce.__doc__

    def Scan(self, sendbuf, recvbuf, op=MPI.SUM):
        return self.__collective_single_type(self.handle.Scan, sendbuf, recvbuf, op)
    Scan.__doc__ = MPI.COMM_WORLD.Scan.__doc__


MPI_WORLD = MPICommunication()
MPI_SELF = MPICommunication(MPI.COMM_SELF)

# tensor is imported at the very end to break circular dependency
from . import tensor
