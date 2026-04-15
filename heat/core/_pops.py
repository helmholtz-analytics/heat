import numpy as np
import torch

from enum import IntEnum

from .communication import MPI, HAVE_MPI

from typing import Any


class POps(IntEnum):
    # predefined
    OP_NULL: 0
    MAX = 1
    MIN = 2
    SUM = 3
    PROD = 4
    LAND = 5
    BAND = 6
    LOR = 7
    BOR = 8
    LXOR = 9
    BXOR = 10
    MAX_LOC = 11
    MIN_LOC = 12
    REPLACE = 13
    NO_OP = 14
    # redefined
    ARGMAX = 15
    ARGMIN = 16
    TOPK = 17
    SUM_F16 = 18
    SUM_BLOAT = 19
    MINMAX = 20


if HAVE_MPI:

    def mpi_argmax(a: str, b: str, _: Any) -> torch.Tensor:
        """
        Create the MPI function for doing argmax, for more info see :func:`argmax <argmax>`

        Parameters
        ----------
        a : str
            left hand side buffer
        b : str
            right hand side buffer
        _ : Any
            placeholder
        """
        lhs = torch.from_numpy(np.frombuffer(a, dtype=np.float64))
        rhs = torch.from_numpy(np.frombuffer(b, dtype=np.float64))

        # extract the values and minimal indices from the buffers (first half are values, second are indices)
        idx_l, idx_r = lhs.chunk(2)[1], rhs.chunk(2)[1]

        if idx_l[0] < idx_r[0]:
            values = torch.stack((lhs.chunk(2)[0], rhs.chunk(2)[0]), dim=1)
            indices = torch.stack((idx_l, idx_r), dim=1)
        else:
            values = torch.stack((rhs.chunk(2)[0], lhs.chunk(2)[0]), dim=1)
            indices = torch.stack((idx_r, idx_l), dim=1)

        # determine the minimum value and select the indices accordingly
        max, max_indices = torch.max(values, dim=1)
        result = torch.cat((max, indices[torch.arange(values.shape[0]), max_indices]))

        rhs.copy_(result)

    MPI_ARGMAX = MPI.Op.Create(mpi_argmax, commute=True)

    def mpi_argmin(a: str, b: str, _: Any) -> torch.Tensor:
        """
        Create the MPI function for doing argmin, for more info see :func:`argmin <argmin>`

        Parameters
        ----------
        a : str
            left hand side
        b : str
            right hand side
        _ : Any
            placeholder
        """
        lhs = torch.from_numpy(np.frombuffer(a, dtype=np.float64))
        rhs = torch.from_numpy(np.frombuffer(b, dtype=np.float64))
        # extract the values and minimal indices from the buffers (first half are values, second are indices)
        idx_l, idx_r = lhs.chunk(2)[1], rhs.chunk(2)[1]

        if idx_l[0] < idx_r[0]:
            values = torch.stack((lhs.chunk(2)[0], rhs.chunk(2)[0]), dim=1)
            indices = torch.stack((idx_l, idx_r), dim=1)
        else:
            values = torch.stack((rhs.chunk(2)[0], lhs.chunk(2)[0]), dim=1)
            indices = torch.stack((idx_r, idx_l), dim=1)

        # determine the minimum value and select the indices accordingly
        min, min_indices = torch.min(values, dim=1)
        result = torch.cat((min, indices[torch.arange(values.shape[0]), min_indices]))

        rhs.copy_(result)

    MPI_ARGMIN = MPI.Op.Create(mpi_argmin, commute=True)

    def mpi_topk(a, b, mpi_type):
        """
        MPI function for distributed :func:`topk`
        """
        # Parse Buffer
        a_parsed = torch.from_numpy(np.frombuffer(a, dtype=np.float64))
        b_parsed = torch.from_numpy(np.frombuffer(b, dtype=np.float64))

        # Collect metadata from Buffer
        k = int(a_parsed[0].item())
        dim = int(a_parsed[1].item())
        largest = bool(a_parsed[2].item())
        sorted = bool(a_parsed[3].item())

        # Offset is the length of the shape on the buffer
        len_shape_a = int(a_parsed[4])
        shape_a = a_parsed[5 : 5 + len_shape_a].int().tolist()
        len_shape_b = int(b_parsed[4])
        shape_b = b_parsed[5 : 5 + len_shape_b].int().tolist()

        # separate the data into values, indices
        a_values, a_indices = a_parsed[len_shape_a + 5 :].chunk(2)
        b_values, b_indices = b_parsed[len_shape_b + 5 :].chunk(2)

        # reconstruct the flattened data by shape
        a_values = a_values.reshape(shape_a)
        a_indices = a_indices.reshape(shape_a)
        b_values = b_values.reshape(shape_b)
        b_indices = b_indices.reshape(shape_b)

        # concatenate the data to actually run topk on
        values = torch.cat((a_values, b_values), dim=dim)
        indices = torch.cat((a_indices, b_indices), dim=dim)

        result, k_indices = torch.topk(values, k, dim=dim, largest=largest, sorted=sorted)
        indices = torch.gather(indices, dim, k_indices)

        metadata = a_parsed[0 : len_shape_a + 5]
        final_result = torch.cat((metadata, result.double().flatten(), indices.double().flatten()))

        b_parsed.copy_(final_result)

    MPI_TOPK = MPI.Op.Create(mpi_topk, commute=True)
