"""
This script demonstrates the usage of the `exchange` function to perform data exchange using MPI_Alltoallw.
It also includes two functions, `AllToAllWResplit` and `HeatResplit`, which utilize the `exchange` function.

The `AllToAllWResplit` function performs data exchange between arrays `arrayA` and `arrayB` using MPI_Alltoallw.
The `HeatResplit` function resplits an array `a` along the specified axis.

The script also includes a main section where it creates an array `a` and performs data exchange and resplit operations.

Note: This script requires the `mpi4py`, `heat`, and `torch` libraries to be installed.
"""

import time
from mpi4py import MPI
import heat as ht
import torch
import perun


def decompose(axis_size, world_size, rank):
    """
    Decompose the axis size based on the world size and rank.

    Parameters
    ----------
    axis_size : int
        The size of the axis.
    world_size : int
        The number of processes.
    rank : int
        The rank of the current process.

    Returns
    -------
    tuple
        A tuple containing the chunk size and chunk start.
    """
    block_size = axis_size // world_size  # Integer division
    rest = axis_size % world_size  # Modulus (remainder)

    chunk_size = block_size + (rest > rank)
    chunk_start = block_size * rank + min(rank, rest)
    return chunk_size, chunk_start  # Return the calculated values as a tuple


def subarrays(com, g_shape, l_shape, split_axis, datatype=MPI.DOUBLE):
    """
    Create subarray datatypes for data exchange.

    Parameters
    ----------
    com : MPI.Comm
        The MPI communicator.
    g_shape : tuple
        The global shape of the array.
    l_shape : tuple
        The local shape of the array.
    split_axis : int
        The axis along which the array will be split.
    datatype : MPI.Datatype, optional
        The datatype of the array elements. Default is MPI.DOUBLE.

    Returns
    -------
    list
        A list of subarray datatypes.
    """
    world_size = com.Get_size()

    subsizes = list(l_shape)
    substarts = [0] * len(l_shape)

    subarray_types = []
    for i in range(world_size):
        chunk_size, chunk_start = decompose(g_shape[split_axis], world_size, i)
        subsizes[split_axis] = chunk_size
        substarts[split_axis] = chunk_start
        subarray_type = datatype.Create_subarray(
            list(l_shape), subsizes, substarts, order=MPI.ORDER_C
        ).Commit()
        subarray_types.append(subarray_type)

    return subarray_types


def exchange(comm, datatype, sizesA, arrayA, axisA, sizesB, arrayB, axisB):
    """
    Perform data exchange using MPI_Alltoallw.

    Parameters
    ----------
    comm : MPI.Comm
        The MPI communicator.
    datatype : MPI.Datatype
        The datatype of the array elements.
    sizesA : list
        The sizes of arrayA.
    arrayA : ndarray
        The array to be sent.
    axisA : int
        The axis along which arrayA will be split.
    sizesB : list
        The sizes of arrayB.
    arrayB : ndarray
        The array to receive the data.
    axisB : int
        The axis along which arrayB will be split.
    """
    nparts = comm.Get_size()

    # Create subarray datatypes for arrayA and arrayB
    send_subarrays = subarrays(comm, sizesA, list(arrayA.shape), axisB, datatype=datatype)
    recv_subarrays = subarrays(comm, sizesB, list(arrayB.shape), axisA, datatype=datatype)

    counts = [1] * nparts
    displs = [0] * nparts

    # Perform the data exchange using MPI_Alltoallw
    comm.Alltoallw(
        [arrayA, (counts, displs), send_subarrays],
        [arrayB, (counts, displs), recv_subarrays],
    )

    # Free the subarray datatypes
    for p in range(nparts):
        send_subarrays[p].Free()
        recv_subarrays[p].Free()


I = 200
J = 200
K = 20
L = 4
M = 19
shape = (I, J, K, L, M)
n_elements = I * J * K * L * M


@perun.monitor()
def AllToAllWResplit(a):
    """
    Perform data exchange between arrays `arrayA` and `arrayB` using MPI_Alltoallw.

    Parameters
    ----------
    a : ht.DNDarray
        The input array.

    Notes
    -----
    - The function resplits the array `a` along the specified axis.
    - The function requires the `mpi4py`, `heat`, and `torch` libraries to be installed.
    """
    new_split = 2
    send_buf = a.larray  # .clone()

    b = ht.zeros(a.gshape, dtype=a.dtype, split=new_split, device=a.device, comm=a.comm)
    recv_buf = b.larray

    exchange(
        a.comm, MPI.INT, list(a.gshape), send_buf, a.split, list(a.gshape), recv_buf, new_split
    )
    del b


@perun.monitor()
def HeatResplit(a):
    """
    Resplit an array `a` along the specified axis.

    Parameters
    ----------
    a : ht.DNDarray
        The input array.

    Notes
    -----
    - The function requires the `heat` library to be installed.
    """
    b = a.resplit(2)
    del b


a = ht.arange(n_elements, dtype=ht.int32).reshape(shape).resplit(3)
print(f"Expected memory: {1 * n_elements * 4 / 1000**3 } GB")
time.sleep(5)
AllToAllWResplit(a)
time.sleep(5)
HeatResplit(a)
time.sleep(5)
