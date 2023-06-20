"""
Module for (pairwise) distance functions
"""
import torch
import numpy as np
from mpi4py import MPI
from typing import Callable

from ..core import factories
from ..core import types
from ..core.dndarray import DNDarray

__all__ = ["cdist", "manhattan", "rbf"]


def _euclidian(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    """
    Helper function to calculate Euclidian distance between ``torch.tensors`` ``x`` and ``y``: :math:`\\sqrt(|x-y|^2)`
    Based on ``torch.cdist``. Returns 2D torch.Tensor of size :math:`m \\times n`

    Parameters
    ----------
    x : torch.Tensor
        2D tensor of size :math:`m x f`
    y : torch.Tensor
        2D tensor of size :math:`n x f`
    """
    return torch.cdist(x, y)


def _euclidian_fast(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    """
    Helper function to calculate Euclidian distance between ``torch.tensors`` ``x`` and ``y``: :math:`\\sqrt(|x-y|^2)`
    Uses quadratic expansion to calculate :math:`(x-y)^2`. Returns 2D torch.Tensor of size :math:`m \\times n`

    Parameters
    ----------
    x : torch.Tensor
        2D tensor of size :math:`m x f`
    y : torch.Tensor
        2D tensor of size :math:`n x f`
    """
    return torch.sqrt(_quadratic_expand(x, y))


def _quadratic_expand(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    """
    Helper function to calculate quadratic expansion :math:`|x-y|^2=|x|^2 + |y|^2 - 2xy`
    Returns 2D torch.Tensor of size :math:`m x n`

    Parameters
    ----------
    x : torch.Tensor
        2D tensor of size :math:`m x f`
    y : torch.Tensor
        2D tensor of size :math:`n x f`
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def _gaussian(x: torch.tensor, y: torch.tensor, sigma: float = 1.0) -> torch.tensor:
    """
    Helper function to calculate Gaussian distance between ``torch.Tensors`` ``x`` and ``y``: :math:`exp(-(|x-y|^2/2\\sigma^2)`
    Based on ``torch.cdist``. Returns a 2D tensor of size :math:`m x n`

    Parameters
    ----------
    x : torch.Tensor
        2D tensor of size :math:`m x f`
    y : torch.Tensor
        2D tensor of size :math:`n x f`
    sigma: float
        Scaling factor for Gaussian kernel

    """
    d2 = _euclidian(x, y) ** 2
    result = torch.exp(-d2 / (2 * sigma * sigma))
    return result


def _gaussian_fast(x: torch.tensor, y: torch.tensor, sigma: float = 1.0) -> torch.tensor:
    """
    Helper function to calculate Gaussian distance between ``torch.Tensors`` ``x`` and ``y``: :math:`exp(-(|x-y|^2/2\\sigma^2)`
    Uses quadratic expansion to calculate :math:`(x-y)^2`. Returns a 2D tensor of size :math:`m x n`

    Parameters
    ----------
    x : torch.Tensor
        2D tensor of size :math:`m x f`
    y : torch.Tensor
        2D tensor of size :math:`n x f`
    sigma: float
        Scaling factor for Gaussian kernel
    """
    d2 = _quadratic_expand(x, y)
    result = torch.exp(-d2 / (2 * sigma * sigma))
    return result


def _manhattan(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    """
    Helper function to calculate Manhattan distance between ``torch.Tensors`` ``x`` and ``y``: :math:`sum(|x-y|)`
    Based on ``torch.cdist``. Returns a 2D tensor of size :math:`m x n`.

    Parameters
    ----------
    x : torch.Tensor
        2D tensor of size :math:`m x f`
    y : torch.Tensor
        2D tensor of size :math:`n x f`
    """
    return torch.cdist(x, y, p=1)


def _manhattan_fast(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Helper function to calculate manhattan distance between ``torch.Tensors`` ``x`` and ``y``: :math:`sum(|x-y|)`
    Uses dimension expansion. Returns a 2D tensor of size :math:`m x n`.

    Parameters
    ----------
    x : torch.Tensor
        2D tensor of size :math:`m x f`
    y : torch.Tensor
        2D tensor of size :math:`n x f`
    """
    return torch.sum(torch.abs(x.unsqueeze(1) - y.unsqueeze(0)), dim=2)


def cdist(X: DNDarray, Y: DNDarray = None, quadratic_expansion: bool = False) -> DNDarray:
    """
    Calculate Euclidian distance between two DNDarrays:

    .. math:: d(x,y) = \\sqrt{(|x-y|^2)}

    Returns 2D DNDarray of size :math: `m \\times n`

    Parameters
    ----------
    X : DNDarray
        2D array of size :math: `m \\times f`
    Y : DNDarray
        2D array of size :math: `n \\times f`
    quadratic_expansion : bool
        Whether to use quadratic expansion for :math:`\\sqrt{(|x-y|^2)}` (Might yield speed-up)
    """
    if quadratic_expansion:
        return _dist(X, Y, _euclidian_fast)
    else:
        return _dist(X, Y, _euclidian)


def rbf(
    X: DNDarray, Y: DNDarray = None, sigma: float = 1.0, quadratic_expansion: bool = False
) -> DNDarray:
    """
    Calculate Gaussian distance between two DNDarrays:

    .. math:: d(x,y) = exp(-(|x-y|^2/2\\sigma^2)

    Returns 2D DNDarray of size :math: `m \\times n`

    Parameters
    ----------
    X : DNDarray
        2D array of size :math: `m \\times f`
    Y : DNDarray
        2D array of size `n \\times f`
    sigma: float
        Scaling factor for gaussian kernel
    quadratic_expansion : bool
        Whether to use quadratic expansion for :math:`\\sqrt{(|x-y|^2)}` (Might yield speed-up)
    """
    if quadratic_expansion:
        return _dist(X, Y, lambda x, y: _gaussian_fast(x, y, sigma))
    else:
        return _dist(X, Y, lambda x, y: _gaussian(x, y, sigma))


def manhattan(X: DNDarray, Y: DNDarray = None, expand: bool = False):
    """
    Calculate Manhattan distance between two DNDarrays:

    .. math:: d(x,y) = \\sum{|x_i-y_i|}

    Returns 2D DNDarray of size :math: `m \\times n`

    Parameters
    ----------
    X : DNDarray
        2D array of size :math: `m \\times f`
    Y : DNDarray
        2D array of size :math: `n \\times f`
    expand : bool
        Whether to use dimension expansion (Might yield speed-up)
    """
    if expand:
        return _dist(X, Y, lambda x, y: _manhattan_fast(x, y))
    else:
        return _dist(X, Y, lambda x, y: _manhattan(x, y))


def _dist(X: DNDarray, Y: DNDarray = None, metric: Callable = _euclidian) -> DNDarray:
    """
    Pairwise distance calculation between all elements along axis 0 of ``X`` and ``Y`` Returns 2D DNDarray of size :math: `m \\times n`
    ``X.split`` and ``Y.split`` can be distributed among axis 0.
        - if neither ``X`` nor ``Y`` is split, result will also be ``split = None \n
        - if ``X.split == 0``, result will be ``split = 0`` regardless of ``Y.split`` \n
    The distance matrix is calculated tile-wise with ring communication between the processes
    holding each a piece of ``X`` and/or ``Y``.

    Parameters
    ----------
    X : DNDarray
        2D Array of size :math: `m \\times f`
    Y : DNDarray, optional
        2D array of size `n \\times f``.
        If `Y in None, the distances will be calculated between all elements of ``X``
    metric: Callable
        The distance to be calculated between ``X`` and ``Y``
        If metric requires additional arguments, it must be handed over as a lambda function: ``lambda x, y: metric(x, y, **args)``

    Notes
    -------
    If ``X.split=None`` and ``Y.split=0``, result will be ``split=1``

    """
    if len(X.shape) > 2:
        raise NotImplementedError("Only 2D data matrices are currently supported")

    if Y is None:
        if X.dtype == types.float32:
            torch_type = torch.float32
            mpi_type = MPI.FLOAT
        elif X.dtype == types.float64:
            torch_type = torch.float64
            mpi_type = MPI.DOUBLE
        else:
            promoted_type = types.promote_types(X.dtype, types.float32)
            X = X.astype(promoted_type)
            if promoted_type == types.float32:
                torch_type = torch.float32
                mpi_type = MPI.FLOAT
            elif promoted_type == types.float64:
                torch_type = torch.float64
                mpi_type = MPI.DOUBLE
            else:
                raise NotImplementedError(f"Datatype {X.dtype} currently not supported as input")

        d = factories.zeros(
            (X.shape[0], X.shape[0]), dtype=X.dtype, split=X.split, device=X.device, comm=X.comm
        )

        if X.split is None:
            d.larray = metric(X.larray, X.larray)

        elif X.split == 0:
            comm = X.comm
            rank = comm.Get_rank()
            size = comm.Get_size()
            K, f = X.shape

            counts, displ, _ = comm.counts_displs_shape(X.shape, X.split)
            num_iter = (size + 1) // 2

            stationary = X.larray
            rows = (displ[rank], displ[rank + 1] if (rank + 1) != size else K)

            # 0th iteration, calculate diagonal
            d_ij = metric(stationary, stationary)
            d.larray[:, rows[0] : rows[1]] = d_ij
            for iter in range(1, num_iter):
                # Send rank's part of the matrix to the next process in a circular fashion
                receiver = (rank + iter) % size
                sender = (rank - iter) % size

                col1 = displ[sender]
                col2 = displ[sender + 1] if sender != size - 1 else K
                columns = (col1, col2)
                # All but the first iter processes are receiving, then sending
                if (rank // iter) != 0:
                    stat = MPI.Status()
                    comm.handle.Probe(source=sender, tag=iter, status=stat)
                    count = int(stat.Get_count(mpi_type) / f)
                    moving = torch.zeros((count, f), dtype=torch_type, device=X.device.torch_device)
                    comm.Recv(moving, source=sender, tag=iter)
                # Sending to next Process
                comm.Send(stationary, dest=receiver, tag=iter)
                # The first iter processes can now receive after sending
                if (rank // iter) == 0:
                    stat = MPI.Status()
                    comm.handle.Probe(source=sender, tag=iter, status=stat)
                    count = int(stat.Get_count(mpi_type) / f)
                    moving = torch.zeros((count, f), dtype=torch_type, device=X.device.torch_device)
                    comm.Recv(moving, source=sender, tag=iter)

                d_ij = metric(stationary, moving)
                d.larray[:, columns[0] : columns[1]] = d_ij

                # Receive calculated tile
                scol1 = displ[receiver]
                scol2 = displ[receiver + 1] if receiver != size - 1 else K
                scolumns = (scol1, scol2)
                symmetric = torch.zeros(
                    scolumns[1] - scolumns[0],
                    (rows[1] - rows[0]),
                    dtype=torch_type,
                    device=X.device.torch_device,
                )
                if (rank // iter) != 0:
                    comm.Recv(symmetric, source=receiver, tag=iter)

                # sending result back to sender of moving matrix (for symmetry)
                comm.Send(d_ij, dest=sender, tag=iter)
                if (rank // iter) == 0:
                    comm.Recv(symmetric, source=receiver, tag=iter)
                d.larray[:, scolumns[0] : scolumns[1]] = symmetric.transpose(0, 1)

            if (size + 1) % 2 != 0:  # we need one mor iteration for the first n/2 processes
                receiver = (rank + num_iter) % size
                sender = (rank - num_iter) % size
                # Case 1: only receiving
                if rank < (size // 2):
                    stat = MPI.Status()
                    comm.handle.Probe(source=sender, tag=num_iter, status=stat)
                    count = int(stat.Get_count(mpi_type) / f)
                    moving = torch.zeros((count, f), dtype=torch_type, device=X.device.torch_device)
                    comm.Recv(moving, source=sender, tag=num_iter)

                    col1 = displ[sender]
                    col2 = displ[sender + 1] if sender != size - 1 else K
                    columns = (col1, col2)

                    d_ij = metric(stationary, moving)
                    d.larray[:, columns[0] : columns[1]] = d_ij

                    # sending result back to sender of moving matrix (for symmetry)
                    comm.Send(d_ij, dest=sender, tag=num_iter)

                # Case 2 : only sending processes
                else:
                    comm.Send(stationary, dest=receiver, tag=num_iter)

                    # Receiving back result
                    scol1 = displ[receiver]
                    scol2 = displ[receiver + 1] if receiver != size - 1 else K
                    scolumns = (scol1, scol2)
                    symmetric = torch.zeros(
                        (scolumns[1] - scolumns[0], rows[1] - rows[0]),
                        dtype=torch_type,
                        device=X.device.torch_device,
                    )
                    comm.Recv(symmetric, source=receiver, tag=num_iter)
                    d.larray[:, scolumns[0] : scolumns[1]] = symmetric.transpose(0, 1)

        else:
            raise NotImplementedError(
                f"Input split was X.split = {X.split}. Splittings other than 0 or None currently not supported."
            )
    else:
        if len(Y.shape) > 2:
            raise NotImplementedError(
                f"Only 2D data matrices are supported, but input shapes were X: {X.shape}, Y: {Y.shape}"
            )

        if X.comm != Y.comm:
            raise NotImplementedError("Differing communicators not supported")

        if X.split is None:
            if Y.split is None:
                split = None
            elif Y.split == 0:
                split = 1
            else:
                raise NotImplementedError(
                    f"Input splits were X.split = {X.split}, Y.split = {Y.split}. Splittings other than 0 or None currently not supported."
                )
        elif X.split == 0:
            split = X.split
        else:
            # ToDo: Find out if even possible
            raise NotImplementedError(
                f"Input splits were X.split = {X.split}, Y.split = {Y.split}. Splittings other than 0 or None currently not supported."
            )

        promoted_type = types.promote_types(X.dtype, Y.dtype)
        promoted_type = types.promote_types(promoted_type, types.float32)
        X = X.astype(promoted_type)
        Y = Y.astype(promoted_type)
        if promoted_type == types.float32:
            torch_type = torch.float32
            mpi_type = MPI.FLOAT
        elif promoted_type == types.float64:
            torch_type = torch.float64
            mpi_type = MPI.DOUBLE
        else:
            raise NotImplementedError(f"Datatype {X.dtype} currently not supported as input")

        d = factories.zeros(
            (X.shape[0], Y.shape[0]), dtype=promoted_type, split=split, device=X.device, comm=X.comm
        )

        if X.split is None:
            d.larray = metric(X.larray, Y.larray)

        elif X.split == 0:
            if Y.split is None:
                d.larray = metric(X.larray, Y.larray)

            elif Y.split == 0:
                if X.shape[1] != Y.shape[1]:
                    raise ValueError("Inputs must have same shape[1]")

                comm = X.comm
                rank = comm.Get_rank()
                size = comm.Get_size()

                m, f = X.shape
                n = Y.shape[0]

                xcounts, xdispl, _ = X.comm.counts_displs_shape(X.shape, X.split)
                ycounts, ydispl, _ = Y.comm.counts_displs_shape(Y.shape, Y.split)
                num_iter = size

                x_ = X.larray
                stationary = Y.larray
                # rows = (xdispl[rank], xdispl[rank + 1] if (rank + 1) != size else m)
                cols = (ydispl[rank], ydispl[rank + 1] if (rank + 1) != size else n)

                # 0th iteration, calculate diagonal
                d_ij = metric(x_, stationary)
                d.larray[:, cols[0] : cols[1]] = d_ij

                for iter in range(1, num_iter):
                    # Send rank's part of the matrix to the next process in a circular fashion
                    receiver = (rank + iter) % size
                    sender = (rank - iter) % size

                    col1 = ydispl[sender]
                    col2 = ydispl[sender + 1] if sender != size - 1 else n
                    columns = (col1, col2)

                    # All but the first iter processes are receiving, then sending
                    if (rank // iter) != 0:
                        stat = MPI.Status()
                        Y.comm.handle.Probe(source=sender, tag=iter, status=stat)
                        count = int(stat.Get_count(mpi_type) / f)
                        moving = torch.zeros(
                            (count, f), dtype=torch_type, device=X.device.torch_device
                        )
                        Y.comm.Recv(moving, source=sender, tag=iter)

                    # Sending to next Process
                    Y.comm.Send(stationary, dest=receiver, tag=iter)

                    # The first iter processes can now receive after sending
                    if (rank // iter) == 0:
                        stat = MPI.Status()
                        Y.comm.handle.Probe(source=sender, tag=iter, status=stat)
                        count = int(stat.Get_count(mpi_type) / f)
                        moving = torch.zeros(
                            (count, f), dtype=torch_type, device=X.device.torch_device
                        )
                        Y.comm.Recv(moving, source=sender, tag=iter)

                    d_ij = metric(x_, moving)
                    d.larray[:, columns[0] : columns[1]] = d_ij

        elif X.split == 0:
            raise NotImplementedError(
                f"Input splits were X.split = {X.split}, Y.split = {Y.split}. Splittings other than 0 or None currently not supported."
            )
    return d
