"""Test script for alltoallw exchange using MPI4py."""

from mpi4py import MPI
import heat as ht
import torch


def decompose(axis_size, world_size, rank):
    """
    Decompose the axis size based on the world size and rank.

    Parameters
    ----------
    axis_size : int
        The size of the axis
    world_size : int
        The number of processes in the world
    rank : int
        The rank of the current process

    Returns
    -------
    chunk_size : int
        The size of the chunk for the current process
    chunk_start : int
        The starting index of the chunk for the current process
    """
    block_size = axis_size // world_size  # Integer division
    rest = axis_size % world_size  # Modulus (remainder)

    chunk_size = block_size + (rest > rank)
    chunk_start = block_size * rank + min(rank, rest)
    return chunk_size, chunk_start  # Return the calculated values as a tuple


def subarrays(com, g_shape, l_shape, split_axis, datatype=MPI.DOUBLE):
    """
    Create subarray types for each process based on the global shape, local shape, and split axis.

    Parameters
    ----------
    com : MPI.Comm
        The MPI communicator
    g_shape : list of int
        The global shape of the array
    l_shape : list of int
        The local shape of the array
    split_axis : int
        The axis along which to split the array
    datatype : MPI.Datatype, optional
        The datatype of the array elements (default is MPI.DOUBLE)

    Returns
    -------
    subarray_types : list of MPI.Datatype
        The subarray types for each process
    """
    world_size = com.Get_size()

    subsizes = list(l_shape)
    substarts = [0] * len(l_shape)

    subarray_types = []
    for i in range(world_size):
        chunk_size, chunk_start = decompose(g_shape[split_axis], world_size, i)
        subsizes[split_axis] = chunk_size
        substarts[split_axis] = chunk_start
        print("DEBUGGING:rank, sizes, subsizes, substarts = ", l_shape, subsizes, substarts)
        subarray_type = datatype.Create_subarray(
            list(l_shape), subsizes, substarts, order=MPI.ORDER_C
        ).Commit()
        subarray_types.append(subarray_type)

    return subarray_types


def exchange(comm, datatype, sizesA, arrayA, axisA, sizesB, arrayB, axisB):
    """
    Perform data exchange between processes using MPI_Alltoallw.

    Parameters
    ----------
    comm : MPI.Comm
        The MPI communicator
    datatype : MPI.Datatype
        The datatype of the array elements
    sizesA : list of int
        The size of arrayA along each dimension
    arrayA : numpy.ndarray
        The array to send
    axisA : int
        The axis along which to split arrayA
    sizesB : list of int
        The size of arrayB along each dimension
    arrayB : numpy.ndarray
        The array to receive
    axisB : int
        The axis along which to split arrayB
    """
    nparts = comm.Get_size()

    # Create subarray datatypes for arrayA and arrayB
    send_subarrays = subarrays(comm, sizesA, list(arrayA.shape), axisB, datatype=datatype)
    recv_subarrays = subarrays(comm, sizesB, list(arrayB.shape), axisA, datatype=datatype)

    counts = [1] * nparts
    displs = [0] * nparts

    # Perform the data exchange using MPI_Alltoallw
    print(type(arrayA), type(arrayB))
    comm.Alltoallw(
        [arrayA, (counts, displs), send_subarrays],
        [arrayB, (counts, displs), recv_subarrays],
    )

    print("DEBUGGING: arrayA = ", arrayA)
    print("DEBUGGING: arrayB = ", arrayB)
    # Free the subarray datatypes
    for p in range(nparts):
        send_subarrays[p].Free()
        recv_subarrays[p].Free()


a = ht.arange(26 * 4, split=0, dtype=ht.int32).reshape(26, 4)
new_split = 1
send_buf = a.larray  # .clone()
print("Send buffer: ", send_buf)

recv_buf_heat = ht.zeros(a.gshape, dtype=a.dtype, split=new_split, device=a.device, comm=a.comm)
recv_buf = recv_buf_heat.larray
print("Recv buffer: ", recv_buf)
# send_buf = ht.MPICommunication.as_mpi_memory(send_buf)
# recv_buf = ht.MPICommunication.as_mpi_memory(recv_buf)

exchange(a.comm, MPI.INT, list(a.gshape), send_buf, a.split, list(a.gshape), recv_buf, new_split)
