Module heat.core.communication
==============================
Module implementing the communication layer of HeAT

Functions
---------

`get_comm() ‑> heat.core.communication.Communication`
:   Retrieves the currently globally set default communication.

`sanitize_comm(comm: Optional[Communication]) ‑> heat.core.communication.Communication`
:   Sanitizes a device or device identifier, i.e. checks whether it is already an instance of :class:`heat.core.devices.Device`
    or a string with known device identifier and maps it to a proper ``Device``.

    Parameters
    ----------
    comm : Communication
        The comm to be sanitized

    Raises
    ------
    TypeError
        If the given communication is not the proper type

`use_comm(comm: Communication = None)`
:   Sets the globally used default communicator.

    Parameters
    ----------
    comm : Communication or None
        The communication to be set

Classes
-------

`Communication()`
:   Base class for Communications (inteded for other backends)

    ### Descendants

    * heat.core.communication.MPICommunication

    ### Static methods

    `is_distributed() ‑> NotImplementedError`
    :   Whether or not the Communication is distributed

    ### Methods

    `chunk(self, shape, split) ‑> NotImplementedError`
    :   Calculates the chunk of data that will be assigned to this compute node given a global data shape and a split
        axis. Returns ``(offset, local_shape, slices)``: the offset in the split dimension, the resulting local shape if the
        global input shape is chunked on the split axis and the chunk slices with respect to the given shape

        Parameters
        ----------
        shape : Tuple[int,...]
            The global shape of the data to be split
        split : int
            The axis along which to chunk the data

`MPICommunication(handle=<mpi4py.MPI.Intracomm object>)`
:   Class encapsulating all MPI Communication

    Parameters
    ----------
    handle: MPI.Communicator
        Handle for the mpi4py Communicator

    ### Ancestors (in MRO)

    * heat.core.communication.Communication

    ### Class variables

    `COUNT_LIMIT`
    :

    ### Static methods

    `as_buffer(obj: torch.Tensor, counts: Optional[Tuple[int]] = None, displs: Optional[Tuple[int]] = None, is_contiguous: Optional[bool] = None) ‑> List[mpi4py.MPI.buffer | Tuple[int, int] | mpi4py.MPI.Datatype]`
    :   Converts a passed ``torch.Tensor`` into a memory buffer object with associated number of elements and MPI data type.

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

    `as_mpi_memory(obj: torch.Tensor) ‑> mpi4py.MPI.buffer`
    :   Converts the passed ``torch.Tensor`` into an MPI compatible memory view.

        Parameters
        ----------
        obj : torch.Tensor
            The tensor to be converted into a MPI memory view.

    `mpi_type_and_elements_of(obj: Union[DNDarray, torch.Tensor], counts: Optional[Tuple[int]], displs: Tuple[int], is_contiguous: Optional[bool]) ‑> Tuple[mpi4py.MPI.Datatype, Tuple[int, ...]]`
    :   Determines the MPI data type and number of respective elements for the given tensor (:class:`~heat.core.dndarray.DNDarray`
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

    `mpi_type_of(dtype: torch.dtype) ‑> mpi4py.MPI.Datatype`
    :   Determines the MPI Datatype from the torch dtype.

        Parameters
        ----------
        dtype : torch.dtype
            PyTorch data type

    ### Methods

    `Allgather(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], recv_axis: int = 0)`
    :   Allgather(self, sendbuf: BufSpec | InPlace, recvbuf: BufSpecB) -> None

        Gather to All.

        Gather data from all processes and broadcast the combined data to all
        other processes.

    `Allgatherv(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], recv_axis: int = 0)`
    :   Allgatherv(self, sendbuf: BufSpec | InPlace, recvbuf: BufSpecV) -> None

        Gather to All Vector.

        Gather data from all processes and send it to all other processes
        providing different amounts of data and displacements.

    `Allreduce(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], op: MPI.Op = <mpi4py.MPI.Op object>)`
    :   Allreduce(self, sendbuf: BufSpec | InPlace, recvbuf: BufSpec, op: Op = SUM) -> None

        Reduce to All.

    `Alltoall(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], send_axis: int = 0, recv_axis: int = None)`
    :   Alltoall(self, sendbuf: BufSpecB | InPlace, recvbuf: BufSpecB) -> None

        All to All Scatter/Gather.

        Send data to all processes and recv data from all processes.

    `Alltoallv(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], send_axis: int = 0, recv_axis: int = None)`
    :   Alltoallv(self, sendbuf: BufSpecV | InPlace, recvbuf: BufSpecV) -> None

        All to All Scatter/Gather Vector.

        Send data to all processes and recv data from all processes
        providing different amounts of data and displacements.

    `Alltoallw(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any])`
    :   Alltoallw(self, sendbuf: BufSpecW | InPlace, recvbuf: BufSpecW) -> None

        All to All Scatter/Gather General.

        Send/recv data to/from all processes allowing the specification of
        different counts, displacements, and datatypes for each dest/source.

    `Bcast(self, buf: Union[DNDarray, torch.Tensor, Any], root: int = 0) ‑> None`
    :   Bcast(self, buf: BufSpec, root: int = 0) -> None

        Broadcast data from one process to all other processes.

    `Bsend(self, buf: Union[DNDarray, torch.Tensor, Any], dest: int, tag: int = 0)`
    :   Bsend(self, buf: BufSpec, dest: int, tag: int = 0) -> None

        Blocking send in buffered mode.

    `Exscan(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], op: MPI.Op = <mpi4py.MPI.Op object>)`
    :   Exscan(self, sendbuf: BufSpec | InPlace, recvbuf: BufSpec, op: Op = SUM) -> None

        Exclusive Scan.

    `Free(self) ‑> None`
    :   Free a communicator.

    `Gather(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], root: int = 0, axis: int = 0, recv_axis: int = None)`
    :   Gather(self, sendbuf: BufSpec | InPlace, recvbuf: BufSpecB | None, root: int = 0) -> None

        Gather data to one process from all other processes.

    `Gatherv(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], root: int = 0, axis: int = 0, recv_axis: int = None)`
    :   Gatherv(self, sendbuf: BufSpec | InPlace, recvbuf: BufSpecV | None, root: int = 0) -> None

        Gather Vector.

        Gather data to one process from all other processes
        providing different amounts of data and displacements.

    `Iallgather(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], recv_axis: int = 0) ‑> heat.core.communication.MPIRequest`
    :   Iallgather(self, sendbuf: BufSpec | InPlace, recvbuf: BufSpecB) -> Request

        Nonblocking Gather to All.

    `Iallgatherv(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], recv_axis: int = 0)`
    :   Iallgatherv(self, sendbuf: BufSpec | InPlace, recvbuf: BufSpecV) -> Request

        Nonblocking Gather to All Vector.

    `Iallreduce(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], op: MPI.Op = <mpi4py.MPI.Op object>) ‑> heat.core.communication.MPIRequest`
    :   Iallreduce(self, sendbuf: BufSpec | InPlace, recvbuf: BufSpec, op: Op = SUM) -> Request

        Nonblocking Reduce to All.

    `Ialltoall(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], send_axis: int = 0, recv_axis: int = None) ‑> heat.core.communication.MPIRequest`
    :   Ialltoall(self, sendbuf: BufSpecB | InPlace, recvbuf: BufSpecB) -> Request

        Nonblocking All to All Scatter/Gather.

    `Ialltoallv(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], send_axis: int = 0, recv_axis: int = None) ‑> heat.core.communication.MPIRequest`
    :   Ialltoallv(self, sendbuf: BufSpecV | InPlace, recvbuf: BufSpecV) -> Request

        Nonblocking All to All Scatter/Gather Vector.

    `Ibcast(self, buf: Union[DNDarray, torch.Tensor, Any], root: int = 0) ‑> heat.core.communication.MPIRequest`
    :   Ibcast(self, buf: BufSpec, root: int = 0) -> Request

        Nonblocking Broadcast.

    `Ibsend(self, buf: Union[DNDarray, torch.Tensor, Any], dest: int, tag: int = 0) ‑> heat.core.communication.MPIRequest`
    :   Ibsend(self, buf: BufSpec, dest: int, tag: int = 0) -> Request

        Nonblocking send in buffered mode.

    `Iexscan(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], op: MPI.Op = <mpi4py.MPI.Op object>) ‑> heat.core.communication.MPIRequest`
    :   Iexscan(self, sendbuf: BufSpec | InPlace, recvbuf: BufSpec, op: Op = SUM) -> Request

        Inclusive Scan.

    `Igather(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], root: int = 0, axis: int = 0, recv_axis: int = None) ‑> heat.core.communication.MPIRequest`
    :   Igather(self, sendbuf: BufSpec | InPlace, recvbuf: BufSpecB | None, root: int = 0) -> Request

        Nonblocking Gather.

    `Igatherv(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], root: int = 0, axis: int = 0, recv_axis: int = None) ‑> heat.core.communication.MPIRequest`
    :   Igatherv(self, sendbuf: BufSpec | InPlace, recvbuf: BufSpecV | None, root: int = 0) -> Request

        Nonblocking Gather Vector.

    `Irecv(self, buf: Union[DNDarray, torch.Tensor, Any], source: int = -1, tag: int = -1) ‑> heat.core.communication.MPIRequest`
    :   Irecv(self, buf: BufSpec, source: int = ANY_SOURCE, tag: int = ANY_TAG) -> Request

        Nonblocking receive.

    `Ireduce(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], op: MPI.Op = <mpi4py.MPI.Op object>, root: int = 0) ‑> heat.core.communication.MPIRequest`
    :   Ireduce(self, sendbuf: BufSpec | InPlace, recvbuf: BufSpec | None, op: Op = SUM, root: int = 0) -> Request

        Nonblocking Reduce to Root.

    `Irsend(self, buf: Union[DNDarray, torch.Tensor, Any], dest: int, tag: int = 0) ‑> heat.core.communication.MPIRequest`
    :   Irsend(self, buf: BufSpec, dest: int, tag: int = 0) -> Request

        Nonblocking send in ready mode.

    `Iscan(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], op: MPI.Op = <mpi4py.MPI.Op object>) ‑> heat.core.communication.MPIRequest`
    :   Iscan(self, sendbuf: BufSpec | InPlace, recvbuf: BufSpec, op: Op = SUM) -> Request

        Inclusive Scan.

    `Iscatter(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], root: int = 0, axis: int = 0, recv_axis: int = None) ‑> heat.core.communication.MPIRequest`
    :   Iscatter(self, sendbuf: BufSpecB | None, recvbuf: BufSpec | InPlace, root: int = 0) -> Request

        Nonblocking Scatter.

    `Iscatterv(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], root: int = 0, axis: int = 0, recv_axis: int = None) ‑> heat.core.communication.MPIRequest`
    :   Iscatterv(self, sendbuf: BufSpecV | None, recvbuf: BufSpec | InPlace, root: int = 0) -> Request

        Nonblocking Scatter Vector.

    `Isend(self, buf: Union[DNDarray, torch.Tensor, Any], dest: int, tag: int = 0) ‑> heat.core.communication.MPIRequest`
    :   Isend(self, buf: BufSpec, dest: int, tag: int = 0) -> Request

        Nonblocking send.

    `Issend(self, buf: Union[DNDarray, torch.Tensor, Any], dest: int, tag: int = 0) ‑> heat.core.communication.MPIRequest`
    :   Issend(self, buf: BufSpec, dest: int, tag: int = 0) -> Request

        Nonblocking send in synchronous mode.

    `Recv(self, buf: Union[DNDarray, torch.Tensor, Any], source: int = -1, tag: int = -1, status: MPI.Status = None)`
    :   Recv(self, buf: BufSpec, source: int = ANY_SOURCE, tag: int = ANY_TAG, status: Status | None = None) -> None

        Blocking receive.

        .. note:: This function blocks until the message is received.

    `Reduce(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], op: MPI.Op = <mpi4py.MPI.Op object>, root: int = 0)`
    :   Reduce(self, sendbuf: BufSpec | InPlace, recvbuf: BufSpec | None, op: Op = SUM, root: int = 0) -> None

        Reduce to Root.

    `Rsend(self, buf: Union[DNDarray, torch.Tensor, Any], dest: int, tag: int = 0)`
    :   Rsend(self, buf: BufSpec, dest: int, tag: int = 0) -> None

        Blocking send in ready mode.

    `Scan(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], op: MPI.Op = <mpi4py.MPI.Op object>)`
    :   Scan(self, sendbuf: BufSpec | InPlace, recvbuf: BufSpec, op: Op = SUM) -> None

        Inclusive Scan.

    `Scatter(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: Union[DNDarray, torch.Tensor, Any], root: int = 0, axis: int = 0, recv_axis: int = None)`
    :   Scatter(self, sendbuf: BufSpecB | None, recvbuf: BufSpec | InPlace, root: int = 0) -> None

        Scatter data from one process to all other processes.

    `Scatterv(self, sendbuf: Union[DNDarray, torch.Tensor, Any], recvbuf: int, root: int = 0, axis: int = 0, recv_axis: int = None)`
    :   Scatterv(self, sendbuf: BufSpecV | None, recvbuf: BufSpec | InPlace, root: int = 0) -> None

        Scatter Vector.

        Scatter data from one process to all other processes
        providing different amounts of data and displacements.

    `Send(self, buf: Union[DNDarray, torch.Tensor, Any], dest: int, tag: int = 0)`
    :   Send(self, buf: BufSpec, dest: int, tag: int = 0) -> None

        Blocking send.

        .. note:: This function may block until the message is received.
           Whether `Send` blocks or not depends on several factors and is
           implementation dependent.

    `Split(self, color: int = 0, key: int = 0) ‑> heat.core.communication.MPICommunication`
    :   Split communicator by color and key.

        Parameters
        ----------
        color : int, optional
            Determines the new communicator for a process.
        key: int, optional
            Ordering within the new communicator.

    `Ssend(self, buf: Union[DNDarray, torch.Tensor, Any], dest: int, tag: int = 0)`
    :   Ssend(self, buf: BufSpec, dest: int, tag: int = 0) -> None

        Blocking send in synchronous mode.

    `alltoall_recvbuffer(self, obj: torch.Tensor) ‑> List[mpi4py.MPI.buffer | Tuple[int, int] | mpi4py.MPI.Datatype]`
    :   Converts a passed ``torch.Tensor`` into a memory buffer object with associated number of elements and MPI data type.
        XXX: might not work for all MPI stacks. Might require multiple type commits or so

        Parameters
        ----------
        obj: torch.Tensor
             The object to be transformed into a custom MPI datatype

    `alltoall_sendbuffer(self, obj: torch.Tensor) ‑> List[mpi4py.MPI.buffer | Tuple[int, int] | mpi4py.MPI.Datatype]`
    :   Converts a passed ``torch.Tensor`` into a memory buffer object with associated number of elements and MPI data type.
        XXX: might not work for all MPI stacks. Might require multiple type commits or so

        Parameters
        ----------
        obj: torch.Tensor
             The object to be transformed into a custom MPI datatype

    `chunk(self, shape: Tuple[int], split: int, rank: int = None, w_size: int = None, sparse: bool = False) ‑> Tuple[int, Tuple[int], Tuple[slice]]`
    :   Calculates the chunk of data that will be assigned to this compute node given a global data shape and a split
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

    `counts_displs_shape(self, shape: Tuple[int], axis: int) ‑> Tuple[Tuple[int], Tuple[int], Tuple[int]]`
    :   Calculates the item counts, displacements and output shape for a variable sized all-to-all MPI-call (e.g.
        ``MPI_Alltoallv``). The passed shape is regularly chunk along the given axis and for all nodes.

        Parameters
        ----------
        shape : Tuple[int,...]
            The object for which to calculate the chunking.
        axis : int
            The axis along which the chunking is performed.

    `is_distributed(self) ‑> bool`
    :   Determines whether the communicator is distributed, i.e. handles more than one node.

`MPIRequest(handle, sendbuf: Union[DNDarray, torch.Tensor, Any] = None, recvbuf: Union[DNDarray, torch.Tensor, Any] = None, tensor: torch.Tensor = None, permutation: Tuple[int, ...] = None)`
:   Represents a handle on a non-blocking operation

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

    ### Methods

    `Wait(self, status: MPI.Status = None)`
    :   Waits for an MPI request to complete
