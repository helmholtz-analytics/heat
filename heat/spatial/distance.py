import torch
import numpy as np
from mpi4py import MPI

from ..core import factories
from ..core import types

__all__ = ["cdist", "rbf"]


def _euclidian(x, y):
    """
    Helper function to calculate euclidian distance between torch.tensors x and y: sqrt(|x-y|**2)
    Based on torch.cdist

    Parameters
    ----------
    x : torch.tensor
        2D tensor of size m x f
    y : torch.tensor
        2D tensor of size n x f

    Returns
    -------
    torch.tensor
        2D tensor of size m x n
    """
    return torch.cdist(x, y)


def _euclidian_fast(x, y):
    """
    Helper function to calculate euclidian distance between torch.tensors x and y: sqrt(|x-y|**2)
    Uses quadratic expansion to calculate (x-y)**2

    Parameters
    ----------
    x : torch.tensor
        2D tensor of size m x f
    y : torch.tensor
        2D tensor of size n x f

    Returns
    -------
    torch.tensor
        2D tensor of size m x n
    """
    return torch.sqrt(_quadratic_expand(x, y))


def _quadratic_expand(x, y):
    """
    Helper function to calculate quadratic expansion |x-y|**2=|x|**2 + |y|**2 - 2xy

    Parameters
    ----------
    x : torch.tensor
        2D tensor of size m x f
    y : torch.tensor
        2D tensor of size n x f

    Returns
    -------
    torch.tensor
        2D tensor of size m x n
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def _gaussian(x, y, sigma=1.0):
    """
    Helper function to calculate gaussian distance between torch.tensors x and y: exp(-(|x-y|**2/2sigma**2)
    Based on torch.cdist

    Parameters
    ----------
    x : torch.tensor
        2D tensor of size m x f
    y : torch.tensor
        2D tensor of size n x f
    sigma: float, default=1.0
        scaling factor for gaussian kernel

    Returns
    -------
    torch.tensor
        2D tensor of size m x n
    """
    d2 = _euclidian(x, y) ** 2
    result = torch.exp(-d2 / (2 * sigma * sigma))
    return result


def _gaussian_fast(x, y, sigma=1.0):
    """
    Helper function to calculate gaussian distance between torch.tensors x and y: exp(-(|x-y|**2/2sigma**2)
    Uses quadratic expansion to calculate (x-y)**2

    Parameters
    ----------
    x : torch.tensor
        2D tensor of size m x f
    y : torch.tensor
        2D tensor of size n x f
    sigma: float, default=1.0
        scaling factor for gaussian kernel

    Returns
    -------
    torch.tensor
        2D tensor of size m x n
    """

    d2 = _quadratic_expand(x, y)
    result = torch.exp(-d2 / (2 * sigma * sigma))
    return result


def cdist(X, Y=None, quadratic_expansion=False):
    if quadratic_expansion:
        return _dist(X, Y, _euclidian_fast)
    else:
        return _dist(X, Y, _euclidian)


def rbf(X, Y=None, sigma=1.0, quadratic_expansion=False):
    if quadratic_expansion:
        return _dist(X, Y, lambda x, y: _gaussian_fast(x, y, sigma))
    else:
        return _dist(X, Y, lambda x, y: _gaussian(x, y, sigma))


def _dist(X, Y=None, metric=_euclidian):
    """
    Pairwise distance caclualation between all elements along axis 0 of X and Y
    X.split and Y.split can be distributed among axis 0
    The distance matrix is calculated tile-wise with ring communication between the processes
    holding each a piece of X and/or Y.

    Parameters
    ----------
    X : ht.DNDarray
        2D Array of size m x f
    Y : ht.DNDarray, optional
        2D array of size n x f
        if Y in None, the distances will be calculated between all elements of X
    metric: function
        the distance to be calculated between X and Y
        if metric requires additional arguments, it must be handed over as a lambda function: lambda x, y: metric(x, y, **args)

    Returns
    -------
    ht.DNDarray
        2D array of size m x n
        if neither X nor Y is split, result will also be split=None
        if X.split == 0, result will be split=0 regardless of Y.split
        Caution: if X.split=None and Y.split=0, result will be split=1

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
                raise NotImplementedError(
                    "Datatype {} currently not supported as input".format(X.dtype)
                )

        d = factories.zeros(
            (X.shape[0], X.shape[0]), dtype=X.dtype, split=X.split, device=X.device, comm=X.comm
        )

        if X.split is None:
            d._DNDarray__array = metric(X._DNDarray__array, X._DNDarray__array)

        elif X.split == 0:
            comm = X.comm
            rank = comm.Get_rank()
            size = comm.Get_size()
            K, f = X.shape

            counts, displ, _ = comm.counts_displs_shape(X.shape, X.split)
            num_iter = (size + 1) // 2

            stationary = X._DNDarray__array
            rows = (displ[rank], displ[rank + 1] if (rank + 1) != size else K)

            # 0th iteration, calculate diagonal
            d_ij = metric(stationary, stationary)
            d._DNDarray__array[:, rows[0] : rows[1]] = d_ij
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
                d._DNDarray__array[:, columns[0] : columns[1]] = d_ij

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
                d._DNDarray__array[:, scolumns[0] : scolumns[1]] = symmetric.transpose(0, 1)

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
                    d._DNDarray__array[:, columns[0] : columns[1]] = d_ij

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
                    d._DNDarray__array[:, scolumns[0] : scolumns[1]] = symmetric.transpose(0, 1)

        else:
            raise NotImplementedError(
                "Input split was X.split = {}. Splittings other than 0 or None currently not supported.".format(
                    X.split
                )
            )
    # Y is not None
    else:
        if len(Y.shape) > 2:
            raise NotImplementedError(
                "Only 2D data matrices are supported, but input shapes were X: {}, Y: {}".format(
                    X.shape, Y.shape
                )
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
                    "Input splits were X.split = {}, Y.split = {}. Splittings other than 0 or None currently not supported.".format(
                        X.split, Y.split
                    )
                )
        elif X.split == 0:
            split = X.split
        else:
            # ToDo: Find out if even possible
            raise NotImplementedError(
                "Input splits were X.split = {}, Y.split = {}. Splittings other than 0 or None currently not supported.".format(
                    X.split, Y.split
                )
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
            raise NotImplementedError(
                "Datatype {} currently not supported as input".format(X.dtype)
            )

        d = factories.zeros(
            (X.shape[0], Y.shape[0]), dtype=promoted_type, split=split, device=X.device, comm=X.comm
        )

        if X.split is None:
            d._DNDarray__array = metric(X._DNDarray__array, Y._DNDarray__array)

        elif X.split == 0:
            if Y.split is None:
                d._DNDarray__array = metric(X._DNDarray__array, Y._DNDarray__array)

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

                x_ = X._DNDarray__array
                stationary = Y._DNDarray__array
                # rows = (xdispl[rank], xdispl[rank + 1] if (rank + 1) != size else m)
                cols = (ydispl[rank], ydispl[rank + 1] if (rank + 1) != size else n)

                # 0th iteration, calculate diagonal
                d_ij = metric(x_, stationary)
                d._DNDarray__array[:, cols[0] : cols[1]] = d_ij

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
                    d._DNDarray__array[:, columns[0] : columns[1]] = d_ij

            else:
                raise NotImplementedError(
                    "Input splits were X.split = {}, Y.split = {}. Splittings other than 0 or None currently not supported.".format(
                        X.split, Y.split
                    )
                )
    return d
