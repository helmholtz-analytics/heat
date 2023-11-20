"""
Module implementing the communication layer of HeAT
"""
from __future__ import annotations
import torch
from typing import Optional, Tuple
from ..core.stride_tricks import sanitize_axis

from mpi4py import MPI
from .mpi4py4torch import MPICommunication


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
