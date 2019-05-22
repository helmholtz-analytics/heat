from mpi4py import MPI

import abc
import numpy as np
import os
import subprocess
import torch

from .stride_tricks import sanitize_axis

# check whether OpenMPI support CUDA-aware MPI
if 'openmpi' in os.environ.get('MPI_SUFFIX', '').lower():
    buffer = subprocess.check_output(['ompi_info', '--parsable', '--all'])
    CUDA_AWARE_MPI = b'mpi_built_with_cuda_support:value:true' in buffer
else:
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

    def counts_displs_shape(self, shape, axis):
        """
        Calculates the item counts, displacements and output shape for a variable sized all-to-all MPI-call (e.g.
        MPI_Alltoallv). The passed shape is regularly chunk along the given axis and for all nodes.

        Parameters
        ----------
        shape : tuple(int)
            The object for which to calculate the chunking.
        axis : int
            The axis along which the chunking is performed.

        Returns
        -------
        counts_and_displs : two-tuple of tuple of ints
            The counts and displacements for all nodes
        """
        # the elements send/received by all nodes
        counts = np.full((self.size,), shape[axis] // self.size)
        counts[:shape[axis] % self.size] += 1

        # the displacements into the buffer
        displs = np.zeros((self.size,), dtype=counts.dtype)
        np.cumsum(counts[:-1], out=displs[1:])

        # helper that calculates the output shape for a receiving buffer under the assumption all nodes have an equally
        # sized input compared to this node
        output_shape = list(shape)
        output_shape[axis] = self.size * counts[self.rank]

        return tuple(counts), tuple(displs), tuple(output_shape)

    @classmethod
    def mpi_type_and_elements_of(cls, obj, counts, displs):
        """
        Determines the MPI data type and number of respective elements for the given tensor. In case the tensor is
        contiguous in memory, a native MPI data type can be used. Otherwise, a derived data type is automatically
        constructed using the storage information of the passed object.

        Parameters
        ----------
        obj : ht.DNDarray or torch.Tensor
            The object for which to construct the MPI data type and number of elements
        counts : tuple of ints, optional
            Optional counts arguments for variable MPI-calls (e.g. Alltoallv)
        displs : tuple of ints, optional
            Optional displacements arguments for variable MPI-calls (e.g. Alltoallv)

        Returns
        -------
        type : MPI.Datatype
            The data type object
        elements : int or tuple of ints
            The number of elements of the respective data type
        """
        mpi_type, elements = cls.__mpi_type_mappings[obj.dtype], torch.numel(obj)

        # simple case, continuous memory can be transmitted as is
        if obj.is_contiguous():
            if counts is None:
                return mpi_type, elements
            else:
                factor = np.prod(obj.shape[1:])
                return mpi_type, (tuple(factor * ele for ele in counts), (tuple(factor * ele for ele in displs)),)

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

        if counts is not None:
            return mpi_type, (counts, displs,)
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
    def as_buffer(cls, obj, counts=None, displs=None):
        """
        Converts a passed torch tensor into a memory buffer object with associated number of elements and MPI data type.

        Parameters
        ----------
        obj : torch.Tensor
            The object to be converted into a buffer representation.
        counts : tuple of ints, optional
            Optional counts arguments for variable MPI-calls (e.g. Alltoallv)
        displs : tuple of ints, optional
            Optional displacements arguments for variable MPI-calls (e.g. Alltoallv)

        Returns
        -------
        buffer : list[MPI.memory, int, MPI.Datatype] or list[MPI.memory, tuple of int, MPI.Datatype]
            The buffer information of the passed tensor, ready to be passed as MPI send or receive buffer.
        """
        mpi_type, elements = cls.mpi_type_and_elements_of(obj, counts, displs)

        return [cls.as_mpi_memory(obj), elements, mpi_type]

    def __recv_like(self, func, buf, source, tag, status):
        if isinstance(buf, dndarray.DNDarray):
            buf = buf._DNDarray__array
        if not isinstance(buf, torch.Tensor):
            return func(buf, source, tag, status)

        return func(self.as_buffer(buf), source, tag, status)

    def Irecv(self, buf, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=None):
        return self.__recv_like(self.handle.Irecv, buf, source, tag, status)
    Irecv.__doc__ = MPI.Comm.Irecv.__doc__

    def Recv(self, buf, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=None):
        return self.__recv_like(self.handle.Recv, buf, source, tag, status)
    Recv.__doc__ = MPI.Comm.Recv.__doc__

    def __send_like(self, func, buf, dest, tag):
        if isinstance(buf, dndarray.DNDarray):
            buf = buf._DNDarray__array
        if not isinstance(buf, torch.Tensor):
            return func(buf, dest, tag)

        return func(self.as_buffer(buf), dest, tag)

    def Bsend(self, buf, dest, tag=0):
        return self.__send_like(self.handle.Bsend, buf, dest, tag)
    Bsend.__doc__ = MPI.Comm.Bsend.__doc__

    def Ibsend(self, buf, dest, tag=0):
        return self.__send_like(self.handle.Ibsend, buf, dest, tag)
    Ibsend.__doc__ = MPI.Comm.Ibsend.__doc__

    def Irsend(self, buf, dest, tag=0):
        return self.__send_like(self.handle.Irsend, buf, dest, tag)
    Irsend.__doc__ = MPI.Comm.Irsend.__doc__

    def Isend(self, buf, dest, tag=0):
        return self.__send_like(self.handle.Isend, buf, dest, tag)
    Isend.__doc__ = MPI.Comm.Isend.__doc__

    def Issend(self, buf, dest, tag=0):
        return self.__send_like(self.handle.Issend, buf, dest, tag)
    Issend.__doc__ = MPI.Comm.Issend.__doc__

    def Rsend(self, buf, dest, tag=0):
        return self.__send_like(self.handle.Rsend, buf, dest, tag)
    Rsend.__doc__ = MPI.Comm.Rsend.__doc__

    def Ssend(self, buf, dest, tag=0):
        return self.__send_like(self.handle.Ssend, buf, dest, tag)
    Ssend.__doc__ = MPI.Comm.Ssend.__doc__

    def Send(self, buf, dest, tag=0):
        return self.__send_like(self.handle.Send, buf, dest, tag)
    Send.__doc__ = MPI.Comm.Send.__doc__

    def __broadcast_like(self, func, buf, root):
        # unpack the buffer if it is a HeAT tensor
        if isinstance(buf, dndarray.DNDarray):
            buf = buf._DNDarray__array
        # convert torch tensors to MPI memory buffers
        if not isinstance(buf, torch.Tensor):
            return func(buf, root)

        return func(self.as_buffer(buf), root)

    def Bcast(self, buf, root=0):
        return self.__broadcast_like(self.handle.Bcast, buf, root)
    Bcast.__doc__ = MPI.Comm.Bcast.__doc__

    def Ibcast(self, buf, root=0):
        return self.__broadcast_like(self.handle.Ibcast, buf, root)
    Ibcast.__doc__ = MPI.Comm.Ibcast.__doc__

    def __reduce_like(self, func, sendbuf, recvbuf, *args, **kwargs):
        # unpack the send buffer if it is a HeAT tensor
        if isinstance(sendbuf, dndarray.DNDarray):
            sendbuf = sendbuf._DNDarray__array
        # unpack the receive buffer if it is a HeAT tensor
        if isinstance(recvbuf, dndarray.DNDarray):
            recvbuf = recvbuf._DNDarray__array

        # harmonize the input and output buffers
        # MPI requires send and receive buffers to be of same type and length. If the torch tensors are either not both
        # contiguous or differently strided, they have to be made matching (if possible) first.
        if isinstance(sendbuf, torch.Tensor):
            # convert the send buffer to a pointer, number of elements and type are identical to the receive buffer
            dummy = sendbuf.contiguous()  # make a contiguous copy and reassign the storage, old will be collected
            sendbuf.set_(dummy.storage(), dummy.storage_offset(), size=dummy.shape, stride=dummy.stride())
            sendbuf = self.as_buffer(sendbuf)
        if isinstance(recvbuf, torch.Tensor):
            # nothing matches, the buffers have to be made contiguous
            dummy = recvbuf.contiguous()
            recvbuf.set_(dummy.storage(), dummy.storage_offset(), size=dummy.shape, stride=dummy.stride())
            if sendbuf is MPI.IN_PLACE:
                recvbuf = self.as_buffer(recvbuf)
            else:
                recvbuf = (self.as_mpi_memory(recvbuf), sendbuf[1], sendbuf[2],)

        # perform the actual reduction operation
        return func(sendbuf, recvbuf, *args, **kwargs)

    def Allreduce(self, sendbuf, recvbuf, op=MPI.SUM):
        return self.__reduce_like(self.handle.Allreduce, sendbuf, recvbuf, op)
    Allreduce.__doc__ = MPI.Comm.Allreduce.__doc__

    def Exscan(self, sendbuf, recvbuf, op=MPI.SUM):
        return self.__reduce_like(self.handle.Exscan, sendbuf, recvbuf, op)
    Exscan.__doc__ = MPI.COMM_WORLD.Exscan.__doc__

    def Iallreduce(self, sendbuf, recvbuf, op=MPI.SUM):
        return self.__reduce_like(self.handle.Iallreduce, sendbuf, recvbuf, op)
    Iallreduce.__doc__ = MPI.Comm.Iallreduce.__doc__

    def Iexscan(self, sendbuf, recvbuf, op=MPI.SUM):
        return self.__reduce_like(self.handle.Iexscan, sendbuf, recvbuf, op)
    Iexscan.__doc__ = MPI.COMM_WORLD.Iexscan.__doc__

    def Iscan(self, sendbuf, recvbuf, op=MPI.SUM):
        return self.__reduce_like(self.handle.Iscan, sendbuf, recvbuf, op)
    Iscan.__doc__ = MPI.COMM_WORLD.Iscan.__doc__

    def Ireduce(self, sendbuf, recvbuf, op=MPI.SUM, root=0):
        return self.__reduce_like(self.handle.Ireduce, sendbuf, recvbuf, op, root)
    Ireduce.__doc__ = MPI.Comm.Ireduce.__doc__

    def Reduce(self, sendbuf, recvbuf, op=MPI.SUM, root=0):
        return self.__reduce_like(self.handle.Reduce, sendbuf, recvbuf, op, root)
    Reduce.__doc__ = MPI.Comm.Reduce.__doc__

    def Scan(self, sendbuf, recvbuf, op=MPI.SUM):
        return self.__reduce_like(self.handle.Scan, sendbuf, recvbuf, op)
    Scan.__doc__ = MPI.COMM_WORLD.Scan.__doc__

    def __scatter_like(self, func, sendbuf, recvbuf, send_axis, recv_axis, send_factor=1, recv_factor=1, **kwargs):
        # align the output buffer in the same way as the input buffer by default
        if recv_axis is None:
            recv_axis = send_axis

        # dummy allocation for *v calls
        send_counts, send_displs, recv_counts, recv_displs = None, None, None, None,

        # unpack the send buffer
        if isinstance(sendbuf, tuple):
            sendbuf, send_counts, send_displs = sendbuf
        if isinstance(sendbuf, dndarray.DNDarray):
            sendbuf = sendbuf._DNDarray__array
        if not isinstance(sendbuf, torch.Tensor) and send_axis != 0:
            raise TypeError('sendbuf of type {} does not support send_axis != 0'.format(type(sendbuf)))

        # unpack the receive buffer
        if isinstance(recvbuf, tuple):
            recvbuf, recv_counts, recv_displs = recvbuf
        if isinstance(recvbuf, dndarray.DNDarray):
            recvbuf = recvbuf._DNDarray__array
        if not isinstance(recvbuf, torch.Tensor) and send_axis != 0:
            raise TypeError('recvbuf of type {} does not support send_axis != 0'.format(type(recvbuf)))

        # keep a reference to the original buffer object
        original_recvbuf = recvbuf

        # permute the send_axis order so that the split send_axis is the first to be transmitted
        send_axis_permutation = list(range(recvbuf.ndimension()))
        send_axis_permutation[0], send_axis_permutation[send_axis] = send_axis, 0
        if self.rank == kwargs.get('root', -1) or send_counts is not None:
            sendbuf = sendbuf.permute(*send_axis_permutation)

        recv_axis_permutation = list(range(recvbuf.ndimension()))
        recv_axis_permutation[0], recv_axis_permutation[recv_axis] = recv_axis, 0
        recvbuf = recvbuf.permute(*recv_axis_permutation)

        # prepare buffer objects
        if sendbuf is not MPI.IN_PLACE:
            mpi_sendbuf = self.as_buffer(sendbuf, send_counts, send_displs)
            if send_counts is None:
                mpi_sendbuf[1] //= send_factor

        if recvbuf is not MPI.IN_PLACE:
            mpi_recvbuf = self.as_buffer(recvbuf, recv_counts, recv_displs)
            if recv_counts is None:
                mpi_recvbuf[1] //= recv_factor

        # perform the scatter operation
        exit_code = func(mpi_sendbuf, mpi_recvbuf, **kwargs)

        # undo the recvbuf permutation and assign the temporary buffer to the original recvbuf
        if recv_axis != 0:
            recvbuf = recvbuf.permute(*recv_axis_permutation)
            original_recvbuf.set_(recvbuf.storage(), recvbuf.storage_offset(), recvbuf.shape, recvbuf.stride())

        return exit_code

    def Allgather(self, sendbuf, recvbuf, axis=0, recv_axis=None):
        return self.__scatter_like(self.handle.Allgather, sendbuf, recvbuf, axis, recv_axis, recv_factor=self.size)
    Allgather.__doc__ = MPI.Comm.Allgather.__doc__

    def Allgatherv(self, sendbuf, recvbuf, axis=0, recv_axis=None):
        return self.__scatter_like(self.handle.Allgatherv, sendbuf, recvbuf, axis, recv_axis, recv_factor=self.size)
    Allgatherv.__doc__ = MPI.Comm.Allgatherv.__doc__

    def Alltoall(self, sendbuf, recvbuf, axis=0, recv_axis=None):
        return self.__scatter_like(
            self.handle.Alltoall, sendbuf, recvbuf, axis, recv_axis, send_factor=self.size, recv_factor=self.size
        )
    Alltoall.__doc__ = MPI.Comm.Alltoall.__doc__

    def Alltoallv(self, sendbuf, recvbuf, axis=0, recv_axis=None):
        return self.__scatter_like(self.handle.Alltoallv, sendbuf, recvbuf, axis, recv_axis)
    Alltoallv.__doc__ = MPI.Comm.Alltoallv.__doc__

    def Gather(self, sendbuf, recvbuf, root=0, axis=0, recv_axis=None):
        return self.__scatter_like(
            self.handle.Gather, sendbuf, recvbuf, axis, recv_axis, root=root, recv_factor=self.size
        )
    Gather.__doc__ = MPI.Comm.Gather.__doc__

    def Gatherv(self, sendbuf, recvbuf, root=0, axis=0, recv_axis=None):
        return self.__scatter_like(self.handle.Gatherv, sendbuf, recvbuf, axis, recv_axis, root=root)
    Gatherv.__doc__ = MPI.Comm.Gatherv.__doc__

    def Iallgather(self, sendbuf, recvbuf, axis=0, recv_axis=None):
        return self.__scatter_like(self.handle.Iallgather, sendbuf, recvbuf, axis, recv_axis, recv_factor=self.size)
    Iallgather.__doc__ = MPI.Comm.Iallgather.__doc__

    def Iallgatherv(self, sendbuf, recvbuf, axis=0, recv_axis=None):
        return self.__scatter_like(self.handle.Iallgatherv, sendbuf, recvbuf, axis, recv_axis, recv_factor=self.size)
    Iallgatherv.__doc__ = MPI.Comm.Iallgatherv.__doc__

    def Ialltoall(self, sendbuf, recvbuf, axis=0, recv_axis=None):
        return self.__scatter_like(
            self.handle.Ialltoall, sendbuf, recvbuf, axis, recv_axis, send_factor=self.size, recv_factor=self.size
        )
    Ialltoall.__doc__ = MPI.Comm.Ialltoall.__doc__

    def Ialltoallv(self, sendbuf, recvbuf, axis=0, recv_axis=None):
        return self.__scatter_like(self.handle.Ialltoallv, sendbuf, recvbuf, axis, recv_axis)
    Ialltoallv.__doc__ = MPI.Comm.Ialltoallv.__doc__

    def Igather(self, sendbuf, recvbuf, root=0, axis=0, recv_axis=None):
        return self.__scatter_like(
            self.handle.Igather, sendbuf, recvbuf, axis, recv_axis, root=root, recv_factor=self.size
        )
    Igather.__doc__ = MPI.Comm.Igather.__doc__

    def Igatherv(self, sendbuf, recvbuf, root=0, axis=0, recv_axis=None):
        return self.__scatter_like(
            self.handle.Igatherv, sendbuf, recvbuf, axis, recv_axis, root=root, recv_factor=self.size
        )
    Igatherv.__doc__ = MPI.Comm.Igatherv.__doc__

    def Iscatter(self, sendbuf, recvbuf, root=0, axis=0, recv_axis=None):
        return self.__scatter_like(
            self.handle.Iscatter, sendbuf, recvbuf, axis, recv_axis, root=root, send_factor=self.size
        )
    Iscatter.__doc__ = MPI.Comm.Iscatter.__doc__

    def Iscatterv(self, sendbuf, recvbuf, root=0, axis=0, recv_axis=None):
        return self.__scatter_like(
            self.handle.Iscatterv, sendbuf, recvbuf, axis, recv_axis, root=root, send_factor=self.size
        )
    Iscatterv.__doc__ = MPI.Comm.Iscatterv.__doc__

    def Scatter(self, sendbuf, recvbuf, root=0, axis=0, recv_axis=None):
        return self.__scatter_like(
            self.handle.Scatter, sendbuf, recvbuf, axis, recv_axis, root=root, send_factor=self.size
        )
    Scatter.__doc__ = MPI.Comm.Scatter.__doc__

    def Scatterv(self, sendbuf, recvbuf, root=0, axis=0, recv_axis=None):
        return self.__scatter_like(
            self.handle.Scatterv, sendbuf, recvbuf, axis, recv_axis, root=root, send_factor=self.size
        )
    Scatterv.__doc__ = MPI.Comm.Scatterv.__doc__

    def __getattr__(self, name):
        """
        Default pass-through for the communicator methods.

        Parameters
        ----------
        name : str
            The name of the method to be called.

        Returns
        -------
        method : function
            The handle's method
        """
        return getattr(self.handle, name)


MPI_WORLD = MPICommunication()
MPI_SELF = MPICommunication(MPI.COMM_SELF)

# tensor is imported at the very end to break circular dependency
from . import dndarray
