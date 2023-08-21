"""
Module implementing the communication layer of HeAT
"""
from __future__ import annotations

import numpy as np
import os
import subprocess
import torch

from mpi4py import MPI
from typing import Any, Callable, Optional, List, Tuple, Union

from .stride_tricks import sanitize_axis

CUDA_AWARE_MPI = False
# check whether OpenMPI support CUDA-aware MPI
if "openmpi" in os.environ.get("MPI_SUFFIX", "").lower():
    buffer = subprocess.check_output(["ompi_info", "--parsable", "--all"])
    CUDA_AWARE_MPI = b"mpi_built_with_cuda_support:value:true" in buffer
# MVAPICH
CUDA_AWARE_MPI = CUDA_AWARE_MPI or os.environ.get("MV2_USE_CUDA") == "1"
# MPICH
CUDA_AWARE_MPI = CUDA_AWARE_MPI or os.environ.get("MPIR_CVAR_ENABLE_HCOLL") == "1"
# ParaStationMPI
CUDA_AWARE_MPI = CUDA_AWARE_MPI or os.environ.get("PSP_CUDA") == "1"


class MPIRequest:
    """
    Represents a handle on a non-blocking operation

    Parameters
    ----------
    handle: MPI.Communicator
        Handle for the mpi4py Communicator
    sendbuf: DNDarray or torch.Tensor or Any
        The buffer for the data to be send
    recvbuf: DNDarray or torch.Tensor or Any
        The buffer to the receive data
    tensor: torch.Tensor
        Internal Data
    permutation: Tuple[int,...]
        Permutation of the tensor axes
    """

    def __init__(
        self,
        handle,
        sendbuf: Union[DNDarray, torch.Tensor, Any] = None,
        recvbuf: Union[DNDarray, torch.Tensor, Any] = None,
        tensor: torch.Tensor = None,
        permutation: Tuple[int, ...] = None,
    ):
        self.handle = handle
        self.tensor = tensor
        self.recvbuf = recvbuf
        self.sendbuf = sendbuf
        self.permutation = permutation

    def Wait(self, status: MPI.Status = None):
        """
        Waits for an MPI request to complete
        """
        self.handle.Wait(status)
        if (
            self.tensor is not None
            and isinstance(self.tensor, torch.Tensor)
            and self.tensor.is_cuda
            and not CUDA_AWARE_MPI
        ):
            if self.permutation is not None:
                self.recvbuf = self.recvbuf.permute(self.permutation)
            self.tensor.copy_(self.recvbuf)

    def __getattr__(self, name: str) -> Callable:
        """
        Default pass-through for the communicator methods.

        Parameters
        ----------
        name : str
            The name of the method to be called.
        """
        return getattr(self.handle, name)


class Communication:
    """
    Base class for Communications (inteded for other backends)
    """

    @staticmethod
    def is_distributed() -> NotImplementedError:
        """
        Whether or not the Communication is distributed
        """
        raise NotImplementedError()

    def __init__(self) -> NotImplementedError:
        raise NotImplementedError()

    def chunk(self, shape, split) -> NotImplementedError:
        """
        Calculates the chunk of data that will be assigned to this compute node given a global data shape and a split
        axis. Returns ``(offset, local_shape, slices)``: the offset in the split dimension, the resulting local shape if the
        global input shape is chunked on the split axis and the chunk slices with respect to the given shape

        Parameters
        ----------
        shape : Tuple[int,...]
            The global shape of the data to be split
        split : int
            The axis along which to chunk the data

        """
        raise NotImplementedError()


class MPICommunication(Communication):
    """
    Class encapsulating all MPI Communication

    Parameters
    ----------
    handle: MPI.Communicator
        Handle for the mpi4py Communicator
    """

    __mpi_type_mappings = {
        torch.bool: MPI.BOOL,
        torch.uint8: MPI.UNSIGNED_CHAR,
        torch.int8: MPI.SIGNED_CHAR,
        torch.int16: MPI.SHORT,
        torch.int32: MPI.INT,
        torch.int64: MPI.LONG,
        torch.bfloat16: MPI.INT16_T,
        torch.float16: MPI.INT16_T,
        torch.float32: MPI.FLOAT,
        torch.float64: MPI.DOUBLE,
        torch.complex64: MPI.COMPLEX,
        torch.complex128: MPI.DOUBLE_COMPLEX,
    }

    def __init__(self, handle=MPI.COMM_WORLD):
        self.handle = handle
        try:
            self.rank = handle.Get_rank()
            self.size = handle.Get_size()
        except MPI.Exception:
            # ranks not within the group will fail with an MPI.Exception, this is expected
            self.rank = None
            self.size = None

    def is_distributed(self) -> bool:
        """
        Determines whether the communicator is distributed, i.e. handles more than one node.
        """
        return self.size > 1

    def chunk(
        self,
        shape: Tuple[int],
        split: int,
        rank: int = None,
        w_size: int = None,
        sparse: bool = False,
    ) -> Tuple[int, Tuple[int], Tuple[slice]]:
        """
        Calculates the chunk of data that will be assigned to this compute node given a global data shape and a split
        axis.
        Returns ``(offset, local_shape, slices)``: the offset in the split dimension, the resulting local shape if the
        global input shape is chunked on the split axis and the chunk slices with respect to the given shape

        Parameters
        ----------
        shape : Tuple[int,...]
            The global shape of the data to be split
        split : int
            The axis along which to chunk the data
        rank : int, optional
            Process for which the chunking is calculated for, defaults to ``self.rank``.
            Intended for creating chunk maps without communication
        w_size : int, optional
            The MPI world size, defaults to ``self.size``.
            Intended for creating chunk maps without communication
        sparse : bool, optional
            Specifies whether the array is a sparse matrix
        """
        # ensure the split axis is valid, we actually do not need it
        split = sanitize_axis(shape, split)
        if split is None:
            return 0, shape, tuple(slice(0, end) for end in shape)
        rank = self.rank if rank is None else rank
        w_size = self.size if w_size is None else w_size
        if not isinstance(rank, int) or not isinstance(w_size, int):
            raise TypeError("rank and size must be integers")

        dims = len(shape)
        size = shape[split]
        chunk = size // w_size
        remainder = size % w_size

        if remainder > rank:
            chunk += 1
            start = rank * chunk
        else:
            start = rank * chunk + remainder
        end = start + chunk

        if sparse:
            return start, end

        return (
            start,
            tuple(shape[i] if i != split else end - start for i in range(dims)),
            tuple(slice(0, shape[i]) if i != split else slice(start, end) for i in range(dims)),
        )

    def counts_displs_shape(
        self, shape: Tuple[int], axis: int
    ) -> Tuple[Tuple[int], Tuple[int], Tuple[int]]:
        """
        Calculates the item counts, displacements and output shape for a variable sized all-to-all MPI-call (e.g.
        ``MPI_Alltoallv``). The passed shape is regularly chunk along the given axis and for all nodes.

        Parameters
        ----------
        shape : Tuple[int,...]
            The object for which to calculate the chunking.
        axis : int
            The axis along which the chunking is performed.

        """
        # the elements send/received by all nodes
        counts = torch.full((self.size,), shape[axis] // self.size)
        counts[: shape[axis] % self.size] += 1

        # the displacements into the buffer
        displs = torch.zeros((self.size,), dtype=counts.dtype)
        torch.cumsum(counts[:-1], out=displs[1:], dim=0)

        # helper that calculates the output shape for a receiving buffer under the assumption all nodes have an equally
        # sized input compared to this node
        output_shape = list(shape)
        output_shape[axis] = self.size * counts[self.rank].item()

        return tuple(counts.tolist()), tuple(displs.tolist()), tuple(output_shape)

    @classmethod
    def mpi_type_and_elements_of(
        cls,
        obj: Union[DNDarray, torch.Tensor],
        counts: Tuple[int],
        displs: Tuple[int],
        is_contiguous: Optional[bool],
    ) -> Tuple[MPI.Datatype, Tuple[int, ...]]:
        """
        Determines the MPI data type and number of respective elements for the given tensor (:class:`~heat.core.dndarray.DNDarray`
        or ``torch.Tensor). In case the tensor is contiguous in memory, a native MPI data type can be used.
        Otherwise, a derived data type is automatically constructed using the storage information of the passed object.

        Parameters
        ----------
        obj : DNDarray or torch.Tensor
            The object for which to construct the MPI data type and number of elements
        counts : Tuple[ints,...], optional
            Optional counts arguments for variable MPI-calls (e.g. Alltoallv)
        displs : Tuple[ints,...], optional
            Optional displacements arguments for variable MPI-calls (e.g. Alltoallv)
        is_contiguous: bool
            Information on global contiguity of the memory-distributed object. If `None`, it will be set to local contiguity via ``torch.Tensor.is_contiguous()``.
        # ToDo: The option to explicitely specify the counts and displacements to be send still needs propper implementation
        """
        mpi_type, elements = cls.__mpi_type_mappings[obj.dtype], torch.numel(obj)

        # simple case, contiguous memory can be transmitted as is
        if is_contiguous is None:
            # determine local contiguity
            is_contiguous = obj.is_contiguous()

        if is_contiguous:
            if counts is None:
                return mpi_type, elements
            factor = np.prod(obj.shape[1:])
            return (
                mpi_type,
                (
                    tuple(factor * ele for ele in counts),
                    (tuple(factor * ele for ele in displs)),
                ),
            )

        # non-contiguous memory, e.g. after a transpose, has to be packed in derived MPI types
        elements = obj.shape[0]
        shape = obj.shape[1:]
        strides = [1] * len(shape)
        strides[0] = obj.stride()[-1]
        strides = strides[::-1]
        offsets = [obj.element_size() * stride for stride in obj.stride()[:-1]]

        # chain the types based on the
        for i in range(len(shape) - 1, -1, -1):
            mpi_type = mpi_type.Create_vector(shape[i], 1, strides[i]).Create_resized(0, offsets[i])
            mpi_type.Commit()

        if counts is not None:
            return mpi_type, (counts, displs)

        return mpi_type, elements

    @classmethod
    def as_mpi_memory(cls, obj) -> MPI.memory:
        """
        Converts the passed ``torch.Tensor`` into an MPI compatible memory view.

        Parameters
        ----------
        obj : torch.Tensor
            The tensor to be converted into a MPI memory view.
        """
        return MPI.memory.fromaddress(obj.data_ptr(), 0)

    @classmethod
    def as_buffer(
        cls,
        obj: torch.Tensor,
        counts: Tuple[int] = None,
        displs: Tuple[int] = None,
        is_contiguous: Optional[bool] = None,
    ) -> List[Union[MPI.memory, Tuple[int, int], MPI.Datatype]]:
        """
        Converts a passed ``torch.Tensor`` into a memory buffer object with associated number of elements and MPI data type.

        Parameters
        ----------
        obj : torch.Tensor
            The object to be converted into a buffer representation.
        counts : Tuple[int,...], optional
            Optional counts arguments for variable MPI-calls (e.g. Alltoallv)
        displs : Tuple[int,...], optional
            Optional displacements arguments for variable MPI-calls (e.g. Alltoallv)
        is_contiguous: bool, optional
            Optional information on global contiguity of the memory-distributed object.
        """
        squ = False
        if not obj.is_contiguous() and obj.ndim == 1:
            # this makes the math work below this function.
            obj.unsqueeze_(-1)
            squ = True

        mpi_type, elements = cls.mpi_type_and_elements_of(obj, counts, displs, is_contiguous)
        mpi_mem = cls.as_mpi_memory(obj)
        if squ:
            # the squeeze happens in the mpi_type_and_elements_of function in the case of a
            # non-contiguous 1D tensor. Squeezing it puts the memory back to where it should be
            obj.squeeze_(-1)
        return [mpi_mem, elements, mpi_type]

    def alltoall_sendbuffer(
        self, obj: torch.Tensor
    ) -> List[Union[MPI.memory, Tuple[int, int], MPI.Datatype]]:
        """
        Converts a passed ``torch.Tensor`` into a memory buffer object with associated number of elements and MPI data type.
        XXX: might not work for all MPI stacks. Might require multiple type commits or so

        Parameters
        ----------
        obj: torch.Tensor
             The object to be transformed into a custom MPI datatype
        """
        mpi_type = self.__mpi_type_mappings[obj.dtype]

        nproc = self.size
        shape = obj.shape
        strides = [1] * len(shape)
        strides[-1] = obj.stride()[-1]
        offsets = [0] * len(shape)
        offsets[1:] = [obj.element_size() * stride for stride in obj.stride()[:-1]]

        # Step 1: Wrap along axes > 1 (all axes except send_axis and recv_axis
        for i in range(len(shape) - 1, 1, -1):
            mpi_type = mpi_type.Create_vector(shape[i], 1, strides[i]).Create_resized(0, offsets[i])
            mpi_type.Commit()

        # Step 2: Create Custom sized vector datatypes, according to rank-specific size along send_axis
        # send_elements has nproc entries, defining how many vectors of mpi_type are stacked together for each process to receive along the send_axis
        send_elements = np.full((nproc,), obj.shape[1] // nproc)
        send_elements[: obj.shape[1] % nproc] += 1

        # Create short_Type from the last entry of send_elements
        mpi_short_type = mpi_type.Create_vector(send_elements[-1], 1, strides[1]).Create_resized(
            0, offsets[1]
        )
        mpi_short_type.Commit()
        # Create long_Type from the first entry of send_elements (wraps one more mpi_type vector than short_Type
        mpi_long_type = mpi_type.Create_vector(send_elements[0], 1, strides[1]).Create_resized(
            0, offsets[1]
        )
        mpi_long_type.Commit()

        # Step 3: Pack short_type and long_type along the recv_axis
        mpi_short_type = mpi_short_type.Create_vector(shape[0], 1, strides[0]).Create_resized(
            0, send_elements[-1] * obj.stride()[1] * obj.element_size()
        )
        mpi_short_type.Commit()
        mpi_long_type = mpi_long_type.Create_vector(shape[0], 1, strides[0]).Create_resized(
            0, send_elements[0] * obj.stride()[1] * obj.element_size()
        )
        mpi_long_type.Commit()

        # Step 4: Prepare sencounts, senddispls and sendtypes for alltoallw
        # to each process 1 element (=sendcount) of the custom prepared long or short type will be send
        sendcount = [1] * nproc
        tmp_displs = [0] * nproc
        tmp_displs[1:] = np.cumsum(send_elements[:-1])
        element_size = obj.element_size()
        senddispls = [element_size * obj.stride()[1] * d for d in tmp_displs]
        sendtypes = [mpi_short_type] * nproc
        for i in range(obj.shape[1] % nproc):
            sendtypes[i] = mpi_long_type

        return self.as_mpi_memory(obj), (sendcount, senddispls), sendtypes

    def alltoall_recvbuffer(
        self, obj: torch.Tensor
    ) -> List[Union[MPI.memory, Tuple[int, int], MPI.Datatype]]:
        """
        Converts a passed ``torch.Tensor`` into a memory buffer object with associated number of elements and MPI data type.
        XXX: might not work for all MPI stacks. Might require multiple type commits or so

        Parameters
        ----------
        obj: torch.Tensor
             The object to be transformed into a custom MPI datatype
        """
        mpi_type, _ = self.__mpi_type_mappings[obj.dtype], torch.numel(obj)

        nproc = self.size
        shape = obj.shape[1:]
        strides = [1] * len(shape)
        strides[0] = obj.stride()[-1]
        strides = strides[::-1]
        offsets = [obj.element_size() * stride for stride in obj.stride()[:-1]]

        # Step 1: Wrap along axes > 0 (all axes except recv_axis)
        for i in range(len(shape) - 1, -1, -1):
            mpi_type = mpi_type.Create_vector(shape[i], 1, strides[i]).Create_resized(0, offsets[i])
            mpi_type.Commit()

        # Step 2: Receive blocks along the recv axis
        # Prepare recvcount, senddispls and sendtypes for alltoallw
        recvcount = np.full((nproc,), obj.shape[0] // nproc)
        recvcount[: obj.shape[0] % nproc] += 1
        # size/extent of mpitype = offsets[0]
        tmp_displs = [0] * nproc
        tmp_displs[1:] = np.cumsum(recvcount[:-1])
        recvdispls = [offsets[0] * d for d in tmp_displs]
        recvtypes = [mpi_type] * nproc

        return self.as_mpi_memory(obj), (recvcount, recvdispls), recvtypes

    def Free(self) -> None:
        """
        Free a communicator.
        """
        self.handle.Free()

    def Split(self, color: int = 0, key: int = 0) -> MPICommunication:
        """
        Split communicator by color and key.

        Parameters
        ----------
        color : int, optional
            Determines the new communicator for a process.
        key: int, optional
            Ordering within the new communicator.
        """
        return MPICommunication(self.handle.Split(color, key))

    def Irecv(
        self,
        buf: Union[DNDarray, torch.Tensor, Any],
        source: int = MPI.ANY_SOURCE,
        tag: int = MPI.ANY_TAG,
    ) -> MPIRequest:
        """
        Nonblocking receive

        Parameters
        ------------
        buf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to place the received message
        source: int, optional
            Rank of source process, that send the message
        tag: int, optional
            A Tag to identify the message
        """
        if isinstance(buf, DNDarray):
            buf = buf.larray
        if not isinstance(buf, torch.Tensor):
            return MPIRequest(self.handle.Irecv(buf, source, tag))

        rbuf = buf if CUDA_AWARE_MPI else buf.cpu()
        return MPIRequest(self.handle.Irecv(self.as_buffer(rbuf), source, tag), None, rbuf, buf)

    Irecv.__doc__ = MPI.Comm.Irecv.__doc__

    def Recv(
        self,
        buf: Union[DNDarray, torch.Tensor, Any],
        source: int = MPI.ANY_SOURCE,
        tag: int = MPI.ANY_TAG,
        status: MPI.Status = None,
    ):
        """
        Blocking receive

        Parameters
        ------------
        buf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to place the received message
        source: int, optional
            Rank of the source process, that send the message
        tag: int, optional
            A Tag to identify the message
        status: MPI.Status, optional
            Details on the communication
        """
        if isinstance(buf, DNDarray):
            buf = buf.larray
        if not isinstance(buf, torch.Tensor):
            return self.handle.Recv(buf, source, tag, status)

        rbuf = buf if CUDA_AWARE_MPI else buf.cpu()
        ret = self.handle.Recv(self.as_buffer(rbuf), source, tag, status)

        if isinstance(buf, torch.Tensor) and buf.is_cuda and not CUDA_AWARE_MPI:
            buf.copy_(rbuf)
        return ret

    Recv.__doc__ = MPI.Comm.Recv.__doc__

    def __send_like(
        self, func: Callable, buf: Union[DNDarray, torch.Tensor, Any], dest: int, tag: int
    ) -> Tuple[Optional[Union[DNDarray, torch.Tensor]]]:
        """
        Generic function for sending a message to process with rank "dest"

        Parameters
        ------------
        func: Callable
            The respective MPI sending function
        buf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the message to be send
        dest: int, optional
            Rank of the destination process, that receives the message
        tag: int, optional
            A Tag to identify the message
        """
        if isinstance(buf, DNDarray):
            buf = buf.larray
        if not isinstance(buf, torch.Tensor):
            return func(buf, dest, tag), None

        # in case of GPUs, the memory has to be copied to host memory if CUDA-aware MPI is not supported
        sbuf = buf if CUDA_AWARE_MPI else buf.cpu()
        return func(self.as_buffer(sbuf), dest, tag), sbuf

    def Bsend(self, buf: Union[DNDarray, torch.Tensor, Any], dest: int, tag: int = 0):
        """
        Blocking buffered send

        Parameters
        ------------
        buf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the message to be send
        dest: int, optional
            Index of the destination process, that receives the message
        tag: int, optional
            A Tag to identify the message
        """
        return self.__send_like(self.handle.Bsend, buf, dest, tag)[0]

    Bsend.__doc__ = MPI.Comm.Bsend.__doc__

    def Ibsend(
        self, buf: Union[DNDarray, torch.Tensor, Any], dest: int, tag: int = 0
    ) -> MPIRequest:
        """
        Nonblocking buffered send

        Parameters
        ------------
        buf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the message to be send
        dest: int, optional
            Rank of the destination process, that receives the message
        tag: int, optional
            A Tag to identify the message
        """
        return MPIRequest(*self.__send_like(self.handle.Ibsend, buf, dest, tag))

    Ibsend.__doc__ = MPI.Comm.Ibsend.__doc__

    def Irsend(
        self, buf: Union[DNDarray, torch.Tensor, Any], dest: int, tag: int = 0
    ) -> MPIRequest:
        """
        Nonblocking ready send

        Parameters
        ------------
        buf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the message to be send
        dest: int, optional
            Rank of the destination process, that receives the message
        tag: int, optional
            A Tag to identify the message
        """
        return MPIRequest(*self.__send_like(self.handle.Irsend, buf, dest, tag))

    Irsend.__doc__ = MPI.Comm.Irsend.__doc__

    def Isend(self, buf: Union[DNDarray, torch.Tensor, Any], dest: int, tag: int = 0) -> MPIRequest:
        """
        Nonblocking send

        Parameters
        ------------
        buf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the message to be send
        dest: int, optional
            Rank of the destination process, that receives the message
        tag: int, optional
            A Tag to identify the message
        """
        return MPIRequest(*self.__send_like(self.handle.Isend, buf, dest, tag))

    Isend.__doc__ = MPI.Comm.Isend.__doc__

    def Issend(
        self, buf: Union[DNDarray, torch.Tensor, Any], dest: int, tag: int = 0
    ) -> MPIRequest:
        """
        Nonblocking synchronous send

        Parameters
        ------------
        buf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the message to be send
        dest: int, optional
            Rank of the destination process, that receives the message
        tag: int, optional
            A Tag to identify the message
        """
        return MPIRequest(*self.__send_like(self.handle.Issend, buf, dest, tag))

    Issend.__doc__ = MPI.Comm.Issend.__doc__

    def Rsend(self, buf: Union[DNDarray, torch.Tensor, Any], dest: int, tag: int = 0):
        """
        Blocking ready send

        Parameters
        ------------
        buf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the message to be send
        dest: int, optional
            Rank of the destination process, that receives the message
        tag: int, optional
            A Tag to identify the message
        """
        return self.__send_like(self.handle.Rsend, buf, dest, tag)[0]

    Rsend.__doc__ = MPI.Comm.Rsend.__doc__

    def Ssend(self, buf: Union[DNDarray, torch.Tensor, Any], dest: int, tag: int = 0):
        """
        Blocking synchronous send

        Parameters
        ------------
        buf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the message to be send
        dest: int, optional
            Rank of the destination process, that receives the message
        tag: int, optional
            A Tag to identify the message
        """
        return self.__send_like(self.handle.Ssend, buf, dest, tag)[0]

    Ssend.__doc__ = MPI.Comm.Ssend.__doc__

    def Send(self, buf: Union[DNDarray, torch.Tensor, Any], dest: int, tag: int = 0):
        """
        Blocking send

        Parameters
        ------------
        buf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the message to be send
        dest: int, optional
            Rank of the destination process, that receives the message
        tag: int, optional
            A Tag to identify the message
        """
        return self.__send_like(self.handle.Send, buf, dest, tag)[0]

    Send.__doc__ = MPI.Comm.Send.__doc__

    def __broadcast_like(
        self, func: Callable, buf: Union[DNDarray, torch.Tensor, Any], root: int
    ) -> Tuple[Optional[DNDarray, torch.Tensor]]:
        """
        Generic function for broadcasting a message from the process with rank "root" to all other processes of the
        communicator

        Parameters
        ------------
        func: Callable
            The respective MPI broadcast function
        buf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the message to be broadcasted
        root: int
            Rank of the root process, that broadcasts the message
        """
        # unpack the buffer if it is a HeAT tensor
        if isinstance(buf, DNDarray):
            buf = buf.larray
        # convert torch tensors to MPI memory buffers
        if not isinstance(buf, torch.Tensor):
            return func(buf, root), None, None, None

        srbuf = buf if CUDA_AWARE_MPI else buf.cpu()

        return func(self.as_buffer(srbuf), root), srbuf, srbuf, buf

    def Bcast(self, buf: Union[DNDarray, torch.Tensor, Any], root: int = 0) -> None:
        """
        Blocking Broadcast

        Parameters
        ------------
        buf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the message to be broadcasted
        root: int
            Rank of the root process, that broadcasts the message
        """
        ret, sbuf, rbuf, buf = self.__broadcast_like(self.handle.Bcast, buf, root)
        if buf is not None and isinstance(buf, torch.Tensor) and buf.is_cuda and not CUDA_AWARE_MPI:
            buf.copy_(rbuf)
        return ret

    Bcast.__doc__ = MPI.Comm.Bcast.__doc__

    def Ibcast(self, buf: Union[DNDarray, torch.Tensor, Any], root: int = 0) -> MPIRequest:
        """
        Nonblocking Broadcast

        Parameters
        ------------
        buf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the message to be broadcasted
        root: int
            Rank of the root process, that broadcasts the message
        """
        return MPIRequest(*self.__broadcast_like(self.handle.Ibcast, buf, root))

    Ibcast.__doc__ = MPI.Comm.Ibcast.__doc__

    def __reduce_like(
        self,
        func: Callable,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        *args,
        **kwargs,
    ) -> Tuple[Optional[DNDarray, torch.Tensor]]:
        """
        Generic function for reduction operations.

        Parameters
        ------------
        func: Callable
            The respective MPI reduction operation
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result of the reduction
        """
        sbuf = None
        rbuf = None
        buf = None
        # unpack the send buffer if it is a HeAT tensor
        if isinstance(sendbuf, DNDarray):
            sendbuf = sendbuf.larray
        # unpack the receive buffer if it is a HeAT tensor
        if isinstance(recvbuf, DNDarray):
            recvbuf = recvbuf.larray

        # harmonize the input and output buffers
        # MPI requires send and receive buffers to be of same type and length. If the torch tensors are either not both
        # contiguous or differently strided, they have to be made matching (if possible) first.
        if isinstance(sendbuf, torch.Tensor):
            # convert the send buffer to a pointer, number of elements and type are identical to the receive buffer
            dummy = (
                sendbuf.contiguous()
            )  # make a contiguous copy and reassign the storage, old will be collected
            sendbuf.set_(
                dummy.storage(), dummy.storage_offset(), size=dummy.shape, stride=dummy.stride()
            )
            sbuf = sendbuf if CUDA_AWARE_MPI else sendbuf.cpu()
            sendbuf = self.as_buffer(sbuf)
        if isinstance(recvbuf, torch.Tensor):
            buf = recvbuf
            # nothing matches, the buffers have to be made contiguous
            dummy = recvbuf.contiguous()
            recvbuf.set_(
                dummy.storage(), dummy.storage_offset(), size=dummy.shape, stride=dummy.stride()
            )
            rbuf = recvbuf if CUDA_AWARE_MPI else recvbuf.cpu()
            if sendbuf is MPI.IN_PLACE:
                recvbuf = self.as_buffer(rbuf)
            else:
                recvbuf = (self.as_mpi_memory(rbuf), sendbuf[1], sendbuf[2])

        # perform the actual reduction operation
        return func(sendbuf, recvbuf, *args, **kwargs), sbuf, rbuf, buf

    def Allreduce(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        op: MPI.Op = MPI.SUM,
    ):
        """
        Combines values from all processes and distributes the result back to all processes

        Parameters
        ---------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result of the reduction
        op: MPI.Op
            The operation to perform upon reduction
        """
        ret, sbuf, rbuf, buf = self.__reduce_like(self.handle.Allreduce, sendbuf, recvbuf, op)
        if buf is not None and isinstance(buf, torch.Tensor) and buf.is_cuda and not CUDA_AWARE_MPI:
            buf.copy_(rbuf)
        return ret

    Allreduce.__doc__ = MPI.Comm.Allreduce.__doc__

    def Exscan(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        op: MPI.Op = MPI.SUM,
    ):
        """
        Computes the exclusive scan (partial reductions) of data on a collection of processes

        Parameters
        ------------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result of the reduction
        op: MPI.Op
            The operation to perform upon reduction
        """
        ret, sbuf, rbuf, buf = self.__reduce_like(self.handle.Exscan, sendbuf, recvbuf, op)
        if buf is not None and isinstance(buf, torch.Tensor) and buf.is_cuda and not CUDA_AWARE_MPI:
            buf.copy_(rbuf)
        return ret

    Exscan.__doc__ = MPI.COMM_WORLD.Exscan.__doc__

    def Iallreduce(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        op: MPI.Op = MPI.SUM,
    ) -> MPIRequest:
        """
        Nonblocking allreduce reducing values on all processes to a single value

        Parameters
        ---------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result of the reduction
        op: MPI.Op
            The operation to perform upon reduction
        """
        return MPIRequest(*self.__reduce_like(self.handle.Iallreduce, sendbuf, recvbuf, op))

    Iallreduce.__doc__ = MPI.Comm.Iallreduce.__doc__

    def Iexscan(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        op: MPI.Op = MPI.SUM,
    ) -> MPIRequest:
        """
        Nonblocking Exscan

        Parameters
        ------------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result of the reduction
        op: MPI.Op
            The operation to perform upon reduction
        """
        return MPIRequest(*self.__reduce_like(self.handle.Iexscan, sendbuf, recvbuf, op))

    Iexscan.__doc__ = MPI.COMM_WORLD.Iexscan.__doc__

    def Iscan(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        op: MPI.Op = MPI.SUM,
    ) -> MPIRequest:
        """
        Nonblocking Scan

        Parameters
        ------------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result of the reduction
        op: MPI.Op
            The operation to perform upon reduction
        """
        return MPIRequest(*self.__reduce_like(self.handle.Iscan, sendbuf, recvbuf, op))

    Iscan.__doc__ = MPI.COMM_WORLD.Iscan.__doc__

    def Ireduce(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        op: MPI.Op = MPI.SUM,
        root: int = 0,
    ) -> MPIRequest:
        """
        Nonblocking reduction operation

        Parameters
        ---------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result of the reduction
        op: MPI.Op
            The operation to perform upon reduction
        root: int
            Rank of the root process
        """
        return MPIRequest(*self.__reduce_like(self.handle.Ireduce, sendbuf, recvbuf, op, root))

    Ireduce.__doc__ = MPI.Comm.Ireduce.__doc__

    def Reduce(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        op: MPI.Op = MPI.SUM,
        root: int = 0,
    ):
        """
        Reduce values from all processes to a single value on process "root"

        Parameters
        ---------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result of the reduction
        op: MPI.Op
            The operation to perform upon reduction
        root: int
            Rank of the root process
        """
        ret, sbuf, rbuf, buf = self.__reduce_like(self.handle.Reduce, sendbuf, recvbuf, op, root)
        if buf is not None and isinstance(buf, torch.Tensor) and buf.is_cuda and not CUDA_AWARE_MPI:
            buf.copy_(rbuf)
        return ret

    Reduce.__doc__ = MPI.Comm.Reduce.__doc__

    def Scan(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        op: MPI.Op = MPI.SUM,
    ):
        """
        Computes the scan (partial reductions) of data on a collection of processes in a nonblocking way

        Parameters
        ------------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result of the reduction
        op: MPI.Op
            The operation to perform upon reduction
        """
        ret, sbuf, rbuf, buf = self.__reduce_like(self.handle.Scan, sendbuf, recvbuf, op)
        if buf is not None and isinstance(buf, torch.Tensor) and buf.is_cuda and not CUDA_AWARE_MPI:
            buf.copy_(rbuf)
        return ret

    Scan.__doc__ = MPI.COMM_WORLD.Scan.__doc__

    def __allgather_like(
        self,
        func: Callable,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        axis: int,
        **kwargs,
    ):
        """
        Generic function for allgather operations.

        Parameters
        ----------
        func: Callable
            Type of MPI Allgather function (i.e. allgather, allgatherv, iallgather)
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result
        axis: int
            Concatenation axis: The axis along which ``sendbuf`` is packed and along which ``recvbuf`` puts together individual chunks
        """
        # dummy allocation for *v calls
        # ToDO: Propper implementation of usage
        send_counts, send_displs, recv_counts, recv_displs = None, None, None, None

        # unpack the send buffer
        if isinstance(sendbuf, tuple):
            sendbuf, send_counts, send_displs = sendbuf
        if isinstance(sendbuf, DNDarray):
            sendbuf = sendbuf.larray
        if not isinstance(sendbuf, torch.Tensor) and axis != 0:
            raise TypeError(
                f"sendbuf of type {type(sendbuf)} does not support concatenation axis != 0"
            )
        # unpack the receive buffer
        if isinstance(recvbuf, tuple):
            recvbuf, recv_counts, recv_displs = recvbuf
        if isinstance(recvbuf, DNDarray):
            recvbuf = recvbuf.larray
        if not isinstance(recvbuf, torch.Tensor) and axis != 0:
            raise TypeError(
                f"recvbuf of type {type(recvbuf)} does not support concatenation axis != 0"
            )

        # keep a reference to the original buffer object
        original_recvbuf = recvbuf
        sbuf_is_contiguous, rbuf_is_contiguous = None, None
        # permute the send_axis order so that the split send_axis is the first to be transmitted
        if axis != 0:
            send_axis_permutation = list(range(sendbuf.ndimension()))
            send_axis_permutation[0], send_axis_permutation[axis] = axis, 0
            sendbuf = sendbuf.permute(*send_axis_permutation)
            sbuf_is_contiguous = False

            recv_axis_permutation = list(range(recvbuf.ndimension()))
            recv_axis_permutation[0], recv_axis_permutation[axis] = axis, 0
            recvbuf = recvbuf.permute(*recv_axis_permutation)
            rbuf_is_contiguous = False
        else:
            recv_axis_permutation = None

        sbuf = sendbuf if CUDA_AWARE_MPI or not isinstance(sendbuf, torch.Tensor) else sendbuf.cpu()
        rbuf = recvbuf if CUDA_AWARE_MPI or not isinstance(recvbuf, torch.Tensor) else recvbuf.cpu()

        # prepare buffer objects
        if sendbuf is MPI.IN_PLACE or not isinstance(sendbuf, torch.Tensor):
            mpi_sendbuf = sbuf
        else:
            mpi_sendbuf = self.as_buffer(sbuf, send_counts, send_displs, sbuf_is_contiguous)
            if send_counts is not None:
                mpi_sendbuf[1] = mpi_sendbuf[1][0][self.rank]

        if recvbuf is MPI.IN_PLACE or not isinstance(recvbuf, torch.Tensor):
            mpi_recvbuf = rbuf
        else:
            mpi_recvbuf = self.as_buffer(rbuf, recv_counts, recv_displs, rbuf_is_contiguous)
            if recv_counts is None:
                mpi_recvbuf[1] //= self.size
        # perform the scatter operation
        exit_code = func(mpi_sendbuf, mpi_recvbuf, **kwargs)
        return exit_code, sbuf, rbuf, original_recvbuf, recv_axis_permutation

    def Allgather(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        recv_axis: int = 0,
    ):
        """
        Gathers data from all tasks and distribute the combined data to all tasks

        Parameters
        ----------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result
        recv_axis: int
            Concatenation axis: The axis along which ``sendbuf`` is packed and along which ``recvbuf`` puts together individual chunks
        """
        ret, sbuf, rbuf, buf, permutation = self.__allgather_like(
            self.handle.Allgather, sendbuf, recvbuf, recv_axis
        )
        if buf is not None and isinstance(buf, torch.Tensor) and buf.is_cuda and not CUDA_AWARE_MPI:
            if permutation is not None:
                rbuf = rbuf.permute(permutation)
            buf.copy_(rbuf)
        return ret

    Allgather.__doc__ = MPI.Comm.Allgather.__doc__

    def Allgatherv(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        recv_axis: int = 0,
    ):
        """
        v-call of Allgather: Each process may contribute a different amount of data.

        Parameters
        ----------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result
        recv_axis: int
            Concatenation axis: The axis along which ``sendbuf`` is packed and along which ``recvbuf`` puts together individual chunks
        """
        ret, sbuf, rbuf, buf, permutation = self.__allgather_like(
            self.handle.Allgatherv, sendbuf, recvbuf, recv_axis
        )
        if buf is not None and isinstance(buf, torch.Tensor) and buf.is_cuda and not CUDA_AWARE_MPI:
            if permutation is not None:
                rbuf = rbuf.permute(permutation)
            buf.copy_(rbuf)
        return ret

    Allgatherv.__doc__ = MPI.Comm.Allgatherv.__doc__

    def Iallgather(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        recv_axis: int = 0,
    ) -> MPIRequest:
        """
        Nonblocking Allgather.

        Parameters
        ----------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result
        recv_axis: int
            Concatenation axis: The axis along which ``sendbuf`` is packed and along which ``recvbuf`` puts together individual chunks
        """
        return MPIRequest(
            *self.__allgather_like(self.handle.Iallgather, sendbuf, recvbuf, recv_axis)
        )

    Iallgather.__doc__ = MPI.Comm.Iallgather.__doc__

    def Iallgatherv(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        recv_axis: int = 0,
    ):
        """
        Nonblocking v-call of Allgather: Each process may contribute a different amount of data.

        Parameters
        ----------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result
        recv_axis: int
            Concatenation axis: The axis along which ``sendbuf`` is packed and along which ``recvbuf`` puts together individual chunks
        """
        return MPIRequest(
            *self.__allgather_like(self.handle.Iallgatherv, sendbuf, recvbuf, recv_axis)
        )

    Iallgatherv.__doc__ = MPI.Comm.Iallgatherv.__doc__

    def __alltoall_like(
        self,
        func: Callable,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        send_axis: int,
        recv_axis: int,
        **kwargs,
    ):
        """
        Generic function for alltoall operations.

        Parameters
        ----------
        func: Callable
            Specific alltoall function
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result
        send_axis: int
            Future split axis, along which data blocks will be created that will be send to individual ranks

                - if ``send_axis==recv_axis``, an error will be thrown
                - if ``send_axis`` or ``recv_axis`` are ``None``, an error will be thrown
        recv_axis: int
            Prior split axis, along which blocks are received from the individual ranks
        """
        if send_axis is None:
            raise NotImplementedError(
                f"AllToAll needs send_axis and recv_axis to be specified but was send_axis = {send_axis}, recv_axis = {recv_axis}. Please set send_axis and recv_axis"
            )
        # align the output buffer in the same way as the input buffer by default
        if recv_axis is None:
            recv_axis = send_axis

        # dummy allocation for *v calls
        send_counts, send_displs, recv_counts, recv_displs = None, None, None, None

        # unpack the send buffer
        if isinstance(sendbuf, tuple):
            sendbuf, send_counts, send_displs = sendbuf
        if isinstance(sendbuf, DNDarray):
            sendbuf = sendbuf.larray
        if not isinstance(sendbuf, torch.Tensor) and send_axis != 0:
            raise TypeError(f"sendbuf of type {type(sendbuf)} does not support send_axis != 0")

        # unpack the receive buffer
        if isinstance(recvbuf, tuple):
            recvbuf, recv_counts, recv_displs = recvbuf
        if isinstance(recvbuf, DNDarray):
            recvbuf = recvbuf.larray
        if not isinstance(recvbuf, torch.Tensor) and send_axis != 0:
            raise TypeError(f"recvbuf of type {type(recvbuf)} does not support send_axis != 0")

        # keep a reference to the original buffer object
        original_recvbuf = recvbuf

        # Simple case, contiguous buffers can be transmitted as is
        if send_axis < 2 and recv_axis < 2:
            send_axis_permutation = list(range(recvbuf.ndimension()))
            recv_axis_permutation = list(range(recvbuf.ndimension()))

            # Minimal Fix; Could possibly be improved when reworking counts, displs algorithmics
            if self.size > 1:
                send_axis_permutation[0], send_axis_permutation[send_axis] = (send_axis, 0)
                recv_axis_permutation[0], recv_axis_permutation[recv_axis] = (recv_axis, 0)

            else:
                recv_counts = send_counts

            sendbuf = sendbuf.permute(*send_axis_permutation)
            recvbuf = recvbuf.permute(*recv_axis_permutation)

            # prepare buffer objects
            sbuf = (
                sendbuf
                if CUDA_AWARE_MPI or not isinstance(sendbuf, torch.Tensor)
                else sendbuf.cpu()
            )
            mpi_sendbuf = self.as_buffer(sbuf, send_counts, send_displs)
            if send_counts is None:
                mpi_sendbuf[1] //= self.size

            rbuf = (
                recvbuf
                if CUDA_AWARE_MPI or not isinstance(recvbuf, torch.Tensor)
                else recvbuf.cpu()
            )
            mpi_recvbuf = self.as_buffer(rbuf, recv_counts, recv_displs)
            if recv_counts is None:
                mpi_recvbuf[1] //= self.size

            # perform the scatter operation
            exit_code = func(mpi_sendbuf, mpi_recvbuf, **kwargs)
        # slightly more difficult situation, send and receive buffer need custom datatype preparation;
        # operation is performed via alltoallw
        else:
            if recv_axis == send_axis:
                raise NotImplementedError(
                    "AllToAll for same axes not supported. Please choose send_axis and recv_axis to be different."
                )

            # Send_axis-Permutation: [recv_axis, send_axis, rest ...]
            axis_permutation = list(range(recvbuf.ndimension()))
            if send_axis == 0:
                axis_permutation[1], axis_permutation[send_axis] = send_axis, 1
                axis_permutation[recv_axis] = axis_permutation[0]
                axis_permutation[0] = recv_axis

            else:
                axis_permutation[0], axis_permutation[recv_axis] = recv_axis, 0
                axis_permutation[send_axis] = axis_permutation[1]
                axis_permutation[1] = send_axis

            sendbuf = sendbuf.permute(*axis_permutation)
            recvbuf = recvbuf.permute(*axis_permutation)

            # prepare buffer objects
            sbuf = (
                sendbuf
                if CUDA_AWARE_MPI or not isinstance(sendbuf, torch.Tensor)
                else sendbuf.cpu()
            )
            rbuf = (
                recvbuf
                if CUDA_AWARE_MPI or not isinstance(recvbuf, torch.Tensor)
                else recvbuf.cpu()
            )
            mpi_sendbuf = self.alltoall_sendbuffer(sbuf)
            mpi_recvbuf = self.alltoall_recvbuffer(rbuf)

            exit_code = self.handle.Alltoallw(mpi_sendbuf, mpi_recvbuf, **kwargs)
            # original_recvbuf.set_(recvbuf.storage(), recvbuf.storage_offset(), original_recvbuf.shape, original_recvbuf.stride())
            recv_axis_permutation = list(np.argsort(np.array(axis_permutation)))

        return exit_code, sbuf, rbuf, original_recvbuf, recv_axis_permutation

    def Alltoall(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        send_axis: int = 0,
        recv_axis: int = None,
    ):
        """
        All processes send data to all processes: The jth block sent from process i is received by process j and is
        placed in the ith block of recvbuf.

        Parameters
        ----------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result
        send_axis: int
            Future split axis, along which data blocks will be created that will be send to individual ranks

                - if ``send_axis==recv_axis``, an error will be thrown
                - if ``send_axis`` or ``recv_axis`` are ``None``, an error will be thrown
        recv_axis: int
            Prior split axis, along which blocks are received from the individual ranks
        """
        ret, sbuf, rbuf, buf, permutation = self.__alltoall_like(
            self.handle.Alltoall, sendbuf, recvbuf, send_axis, recv_axis
        )
        if buf is not None and isinstance(buf, torch.Tensor) and buf.is_cuda and not CUDA_AWARE_MPI:
            if permutation is not None:
                rbuf = rbuf.permute(permutation)
            buf.copy_(rbuf)
        return ret

    Alltoall.__doc__ = MPI.Comm.Alltoall.__doc__

    def Alltoallv(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        send_axis: int = 0,
        recv_axis: int = None,
    ):
        """
        v-call of Alltoall: All processes send different amount of data to, and receive different amount of data
        from, all processes

        Parameters
        ----------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result
        send_axis: int
            Future split axis, along which data blocks will be created that will be send to individual ranks

                - if ``send_axis==recv_axis``, an error will be thrown
                - if ``send_axis`` or ``recv_axis`` are ``None``, an error will be thrown
        recv_axis: int
            Prior split axis, along which blocks are received from the individual ranks
        """
        ret, sbuf, rbuf, buf, permutation = self.__alltoall_like(
            self.handle.Alltoallv, sendbuf, recvbuf, send_axis, recv_axis
        )
        if buf is not None and isinstance(buf, torch.Tensor) and buf.is_cuda and not CUDA_AWARE_MPI:
            if permutation is not None:
                rbuf = rbuf.permute(permutation)
            buf.copy_(rbuf)
        return ret

    Alltoallv.__doc__ = MPI.Comm.Alltoallv.__doc__

    def Ialltoall(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        send_axis: int = 0,
        recv_axis: int = None,
    ) -> MPIRequest:
        """
        Nonblocking Alltoall

        Parameters
        ----------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result
        send_axis: int
            Future split axis, along which data blocks will be created that will be send to individual ranks

                - if ``send_axis==recv_axis``, an error will be thrown
                - if ``send_axis`` or ``recv_axis`` are ``None``, an error will be thrown
        recv_axis: int
            Prior split axis, along which blocks are received from the individual ranks
        """
        return MPIRequest(
            *self.__alltoall_like(self.handle.Ialltoall, sendbuf, recvbuf, send_axis, recv_axis)
        )

    Ialltoall.__doc__ = MPI.Comm.Ialltoall.__doc__

    def Ialltoallv(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        send_axis: int = 0,
        recv_axis: int = None,
    ) -> MPIRequest:
        """
        Nonblocking v-call of Alltoall: All processes send different amount of data to, and receive different amount of
        data from, all processes

        Parameters
        ----------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result
        send_axis: int
            Future split axis, along which data blocks will be created that will be send to individual ranks

                - if ``send_axis==recv_axis``, an error will be thrown
                - if ``send_axis`` or ``recv_axis`` are ``None``, an error will be thrown
        recv_axis: int
            Prior split axis, along which blocks are received from the individual ranks
        """
        return MPIRequest(
            *self.__alltoall_like(self.handle.Ialltoallv, sendbuf, recvbuf, send_axis, recv_axis)
        )

    Ialltoallv.__doc__ = MPI.Comm.Ialltoallv.__doc__

    def __scatter_like(
        self,
        func: Callable,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        send_axis: int,
        recv_axis: int,
        send_factor: int = 1,
        recv_factor: int = 1,
        **kwargs,
    ):
        """
        Generic function for scatter and gather operations.

        Parameters
        ----------
        func: Callable
            Type of MPI Scatter/Gather function
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result
        send_axis: int
            The axis along which ``sendbuf`` is packed
        recv_axis: int
            The axis along which ``recvbuf`` is packed
        send_factor: int
            Number of elements to be scattered (vor non-v-calls)
        recv_factor: int
            Number of elements to be gathered (vor non-v-calls)
        """
        sbuf, rbuf, recv_axis_permutation = None, None, None

        # align the output buffer in the same way as the input buffer by default
        if recv_axis is None:
            recv_axis = send_axis

        # dummy allocation for *v calls
        send_counts, send_displs, recv_counts, recv_displs = None, None, None, None

        # unpack the send buffer
        if isinstance(sendbuf, tuple):
            sendbuf, send_counts, send_displs = sendbuf
        if isinstance(sendbuf, DNDarray):
            sendbuf = sendbuf.larray
        if not isinstance(sendbuf, torch.Tensor) and send_axis != 0:
            raise TypeError(f"sendbuf of type {type(sendbuf)} does not support send_axis != 0")

        # unpack the receive buffer
        if isinstance(recvbuf, tuple):
            recvbuf, recv_counts, recv_displs = recvbuf
        if isinstance(recvbuf, DNDarray):
            recvbuf = recvbuf.larray
        if not isinstance(recvbuf, torch.Tensor) and send_axis != 0:
            raise TypeError(f"recvbuf of type {type(recvbuf)} does not support send_axis != 0")

        # keep a reference to the original buffer object
        original_recvbuf = recvbuf

        # permute the send_axis order so that the split send_axis is the first to be transmitted
        send_axis_permutation = list(range(recvbuf.ndimension()))
        send_axis_permutation[0], send_axis_permutation[send_axis] = send_axis, 0
        if self.rank == kwargs.get("root", -1) or send_counts is not None:
            sendbuf = sendbuf.permute(*send_axis_permutation)

        recv_axis_permutation = list(range(recvbuf.ndimension()))
        recv_axis_permutation[0], recv_axis_permutation[recv_axis] = recv_axis, 0
        recvbuf = recvbuf.permute(*recv_axis_permutation)

        # prepare buffer objects
        sbuf = sendbuf if CUDA_AWARE_MPI or not isinstance(sendbuf, torch.Tensor) else sendbuf.cpu()
        rbuf = recvbuf if CUDA_AWARE_MPI or not isinstance(recvbuf, torch.Tensor) else recvbuf.cpu()

        if sendbuf is not MPI.IN_PLACE:
            mpi_sendbuf = self.as_buffer(sbuf, send_counts, send_displs)
            if send_counts is None:
                mpi_sendbuf[1] //= send_factor
        else:
            mpi_sendbuf = sbuf
        if recvbuf is not MPI.IN_PLACE:
            mpi_recvbuf = self.as_buffer(rbuf, recv_counts, recv_displs)
            if recv_counts is None:
                mpi_recvbuf[1] //= recv_factor
        else:
            mpi_recvbuf = rbuf

        # perform the scatter operation
        exit_code = func(mpi_sendbuf, mpi_recvbuf, **kwargs)

        # undo the recvbuf permutation and assign the temporary buffer to the original recvbuf
        # if recv_axis != 0:
        #    recvbuf = recvbuf.permute(*recv_axis_permutation)
        #    original_recvbuf.set_(recvbuf.storage(), recvbuf.storage_offset(), recvbuf.shape, recvbuf.stride())

        return exit_code, sbuf, rbuf, original_recvbuf, recv_axis_permutation

    def Gather(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        root: int = 0,
        axis: int = 0,
        recv_axis: int = None,
    ):
        """
        Gathers together values from a group of processes

        Parameters
        ----------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result
        root: int
            Rank of receiving process
        axis: int
            The axis along which ``sendbuf`` is packed
        recv_axis: int
            The axis along which ``recvbuf`` is packed
        """
        ret, sbuf, rbuf, buf, permutation = self.__scatter_like(
            self.handle.Gather, sendbuf, recvbuf, axis, recv_axis, root=root, recv_factor=self.size
        )
        if buf is not None and isinstance(buf, torch.Tensor) and buf.is_cuda and not CUDA_AWARE_MPI:
            if permutation is not None:
                rbuf = rbuf.permute(permutation)
            buf.copy_(rbuf)
        return ret

    Gather.__doc__ = MPI.Comm.Gather.__doc__

    def Gatherv(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        root: int = 0,
        axis: int = 0,
        recv_axis: int = None,
    ):
        """
        v-call for Gather: All processes send different amount of data

        Parameters
        ----------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result
        root: int
            Rank of receiving process
        axis: int
            The axis along which ``sendbuf`` is packed
        recv_axis: int
            The axis along which ``recvbuf`` is packed
        """
        ret, sbuf, rbuf, buf, permutation = self.__scatter_like(
            self.handle.Gatherv, sendbuf, recvbuf, axis, recv_axis, root=root
        )
        if buf is not None and isinstance(buf, torch.Tensor) and buf.is_cuda and not CUDA_AWARE_MPI:
            if permutation is not None:
                rbuf = rbuf.permute(permutation)
            buf.copy_(rbuf)
        return ret

    Gatherv.__doc__ = MPI.Comm.Gatherv.__doc__

    def Igather(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        root: int = 0,
        axis: int = 0,
        recv_axis: int = None,
    ) -> MPIRequest:
        """
        Non-blocking Gather

        Parameters
        ----------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result
        root: int
            Rank of receiving process
        axis: int
            The axis along which ``sendbuf`` is packed
        recv_axis: int
            The axis along which ``recvbuf`` is packed
        """
        return MPIRequest(
            *self.__scatter_like(
                self.handle.Igather,
                sendbuf,
                recvbuf,
                axis,
                recv_axis,
                root=root,
                recv_factor=self.size,
            )
        )

    Igather.__doc__ = MPI.Comm.Igather.__doc__

    def Igatherv(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        root: int = 0,
        axis: int = 0,
        recv_axis: int = None,
    ) -> MPIRequest:
        """
        Non-blocking v-call for Gather: All processes send different amount of data

        Parameters
        ----------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result
        root: int
            Rank of receiving process
        axis: int
            The axis along which ``sendbuf`` is packed
        recv_axis: int
            The axis along which ``recvbuf`` is packed
        """
        return MPIRequest(
            *self.__scatter_like(
                self.handle.Igatherv,
                sendbuf,
                recvbuf,
                axis,
                recv_axis,
                root=root,
                recv_factor=self.size,
            )
        )

    Igatherv.__doc__ = MPI.Comm.Igatherv.__doc__

    def Iscatter(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        root: int = 0,
        axis: int = 0,
        recv_axis: int = None,
    ) -> MPIRequest:
        """
        Non-blocking Scatter

        Parameters
        ----------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result
        root: int
            Rank of sending process
        axis: int
            The axis along which ``sendbuf`` is packed
        recv_axis: int
            The axis along which ``recvbuf`` is packed
        """
        return MPIRequest(
            *self.__scatter_like(
                self.handle.Iscatter,
                sendbuf,
                recvbuf,
                axis,
                recv_axis,
                root=root,
                send_factor=self.size,
            )
        )

    Iscatter.__doc__ = MPI.Comm.Iscatter.__doc__

    def Iscatterv(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        root: int = 0,
        axis: int = 0,
        recv_axis: int = None,
    ) -> MPIRequest:
        """
        Non-blocking v-call for Scatter: Sends different amounts of data to different processes

        Parameters
        ----------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result
        root: int
            Rank of sending process
        axis: int
            The axis along which ``sendbuf`` is packed
        recv_axis: int
            The axis along which ``recvbuf`` is packed
        """
        return MPIRequest(
            *self.__scatter_like(
                self.handle.Iscatterv,
                sendbuf,
                recvbuf,
                axis,
                recv_axis,
                root=root,
                send_factor=self.size,
            )
        )

    Iscatterv.__doc__ = MPI.Comm.Iscatterv.__doc__

    def Scatter(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: Union[DNDarray, torch.Tensor, Any],
        root: int = 0,
        axis: int = 0,
        recv_axis: int = None,
    ):
        """
        Sends data parts from one process to all other processes in a communicator

        Parameters
        ----------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result
        root: int
            Rank of sending process
        axis: int
            The axis along which ``sendbuf`` is packed
        recv_axis: int
            The axis along which ``recvbuf`` is packed
        """
        ret, sbuf, rbuf, buf, permutation = self.__scatter_like(
            self.handle.Scatter, sendbuf, recvbuf, axis, recv_axis, root=root, send_factor=self.size
        )
        if buf is not None and isinstance(buf, torch.Tensor) and buf.is_cuda and not CUDA_AWARE_MPI:
            if permutation is not None:
                rbuf = rbuf.permute(permutation)
            buf.copy_(rbuf)
        return ret

    Scatter.__doc__ = MPI.Comm.Scatter.__doc__

    def Scatterv(
        self,
        sendbuf: Union[DNDarray, torch.Tensor, Any],
        recvbuf: int,
        root: int = 0,
        axis: int = 0,
        recv_axis: int = None,
    ):
        """
        v-call for Scatter: Sends different amounts of data to different processes

        Parameters
        ----------
        sendbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address of the send message
        recvbuf: Union[DNDarray, torch.Tensor, Any]
            Buffer address where to store the result
        root: int
            Rank of sending process
        axis: int
            The axis along which ``sendbuf`` is packed
        recv_axis: int
            The axis along which ``recvbuf`` is packed
        """
        ret, sbuf, rbuf, buf, permutation = self.__scatter_like(
            self.handle.Scatterv,
            sendbuf,
            recvbuf,
            axis,
            recv_axis,
            root=root,
            send_factor=self.size,
        )
        if buf is not None and isinstance(buf, torch.Tensor) and buf.is_cuda and not CUDA_AWARE_MPI:
            if permutation is not None:
                rbuf = rbuf.permute(permutation)
            buf.copy_(rbuf)
        return ret

    Scatterv.__doc__ = MPI.Comm.Scatterv.__doc__

    def __getattr__(self, name: str):
        """
        Default pass-through for the communicator methods.

        Parameters
        ----------
        name : str
            The name of the method to be called.
        """
        return getattr(self.handle, name)


# creating a duplicate COMM
comm = MPI.COMM_WORLD
dup_comm = comm.Dup()

MPI_WORLD = MPICommunication(dup_comm)
MPI_SELF = MPICommunication(MPI.COMM_SELF.Dup())

# set the default communicator to be MPI_WORLD
__default_comm = MPI_WORLD


def get_comm() -> Communication:
    """
    Retrieves the currently globally set default communication.
    """
    return __default_comm


def sanitize_comm(comm: Optional[Communication]) -> Communication:
    """
    Sanitizes a device or device identifier, i.e. checks whether it is already an instance of :class:`heat.core.devices.Device`
    or a string with known device identifier and maps it to a proper ``Device``.

    Parameters
    ----------
    comm : Communication
        The comm to be sanitized

    Raises
    ------
    TypeError
        If the given communication is not the proper type
    """
    if comm is None:
        return get_comm()
    elif isinstance(comm, Communication):
        return comm

    raise TypeError(f"Unknown communication, must be instance of {Communication}")


def use_comm(comm: Communication = None):
    """
    Sets the globally used default communicator.

    Parameters
    ----------
    comm : Communication or None
        The communication to be set
    """
    global __default_comm
    __default_comm = sanitize_comm(comm)


# import at the end of file to break circular dependencies
from .dndarray import DNDarray
