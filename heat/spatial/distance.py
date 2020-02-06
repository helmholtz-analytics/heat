import torch
import numpy as np
from mpi4py import MPI

from .. import core

__all__ = ["cdist", "rbf"]


def _euclidian(x, y):
    return torch.cdist(x, y)


def _euclidian_fast(x, y):
    return torch.sqrt(_quadratic_expand(x, y))


def _quadratic_expand(x, y):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def _gaussian(x, y, sigma=1.0):
    d2 = _euclidian(x, y) ** 2
    result = torch.exp(-d2 / (2 * sigma * sigma))
    return result


def _gaussian_fast(x, y, sigma=1.0):
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
    if len(X.shape) > 2:
        raise NotImplementedError("Only 2D data matrices are currently supported")

    if X.dtype != core.float32:
        raise NotImplementedError("Currently only float32 is supported as input datatype")

    if Y is None:
        d = core.factories.zeros(
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
            d[rows[0] : rows[1], rows[0] : rows[1]] = d_ij
            for iter in range(1, num_iter):
                # Send rank's part of the matrix to the next process in a circular fashion
                receiver = (rank + iter) % size
                sender = (rank - iter) % size

                col1 = displ[sender]
                if sender != size - 1:
                    col2 = displ[sender + 1]
                else:
                    col2 = K
                columns = (col1, col2)

                # All but the first iter processes are receiving, then sending
                if (rank // iter) != 0:
                    stat = MPI.Status()
                    comm.handle.Probe(source=sender, tag=iter, status=stat)
                    count = int(stat.Get_count(MPI.FLOAT) / f)
                    moving = torch.zeros((count, f), dtype=torch.float32)
                    comm.Recv(moving, source=sender, tag=iter)

                # Sending to next Process
                comm.Send(stationary, dest=receiver, tag=iter)

                # The first iter processes can now receive after sending
                if (rank // iter) == 0:
                    stat = MPI.Status()
                    comm.handle.Probe(source=sender, tag=iter, status=stat)
                    count = int(stat.Get_count(MPI.FLOAT) / f)
                    moving = torch.zeros((count, f), dtype=torch.float32)
                    comm.Recv(moving, source=sender, tag=iter)

                d_ij = metric(stationary, moving)
                d[rows[0] : rows[1], columns[0] : columns[1]] = d_ij

                # Receive calculated tile
                scol1 = displ[receiver]
                if receiver != size - 1:
                    scol2 = displ[receiver + 1]
                else:
                    scol2 = K
                scolumns = (scol1, scol2)
                symmetric = torch.zeros(
                    scolumns[1] - scolumns[0], (rows[1] - rows[0]), dtype=torch.float64
                )
                # Receive calculated tile
                if (rank // iter) != 0:
                    comm.Recv(symmetric, source=receiver, tag=iter)

                # sending result back to sender of moving matrix (for symmetry)
                comm.Send(d_ij, dest=sender, tag=iter)
                if (rank // iter) == 0:
                    comm.Recv(symmetric, source=receiver, tag=iter)
                d[rows[0] : rows[1], scolumns[0] : scolumns[1]] = symmetric.transpose(0, 1)

            if (size + 1) % 2 != 0:  # we need one mor iteration for the first n/2 processes
                receiver = (rank + num_iter) % size
                sender = (rank - num_iter) % size

                # Case 1: only receiving
                if rank < (size // 2):
                    stat = MPI.Status()
                    comm.handle.Probe(source=sender, tag=num_iter, status=stat)
                    count = int(stat.Get_count(MPI.FLOAT) / f)
                    moving = torch.zeros((count, f), dtype=torch.float32)
                    comm.Recv(moving, source=sender, tag=num_iter)

                    col1 = displ[sender]
                    if sender != size - 1:
                        col2 = displ[sender + 1]
                    else:
                        col2 = K
                    columns = (col1, col2)

                    d_ij = metric(stationary, moving)
                    d[rows[0] : rows[1], columns[0] : columns[1]] = d_ij

                    # sending result back to sender of moving matrix (for symmetry)
                    comm.Send(d_ij, dest=sender, tag=num_iter)

                # Case 2 : only sending processes
                else:
                    comm.Send(stationary, dest=receiver, tag=num_iter)

                    # Receiving back result
                    scol1 = displ[receiver]
                    if receiver != size - 1:
                        scol2 = displ[receiver + 1]
                    else:
                        scol2 = K
                    scolumns = (scol1, scol2)
                    symmetric = torch.zeros(
                        (scolumns[1] - scolumns[0], rows[1] - rows[0]), dtype=torch.float32
                    )
                    comm.Recv(symmetric, source=receiver, tag=num_iter)
                    d[rows[0] : rows[1], scolumns[0] : scolumns[1]] = symmetric.transpose(0, 1)

        else:
            raise NotImplementedError("Splittings other than 0 or None currently not supported.")

    #########################################################################
    else:
        if len(Y.shape) > 2:
            raise NotImplementedError("Only 2D data matrices are supported")

        if X.comm != Y.comm:
            raise NotImplementedError("Differing communicators not supported")

        if Y.dtype != core.float32:
            raise NotImplementedError("Currently only float32 is supported as input datatype")

        if X.split is None:
            if Y.split is None:
                split = None
            elif Y.split == 0:
                split = 1
            else:
                raise NotImplementedError(
                    "Splittings other than 0 or None currently not supported."
                )
        else:
            # ToDo Für split implementation >1: split = min(X,split, Y.split)
            split = X.split

        d = core.factories.zeros(
            (X.shape[0], Y.shape[0]), dtype=X.dtype, split=split, device=X.device, comm=X.comm
        )

        if X.split is None:
            if Y.split == 0 or Y.split is None:
                d._DNDarray__array = metric(X._DNDarray__array, Y._DNDarray__array)

            else:
                raise NotImplementedError(
                    "Splittings other than 0 or None currently not supported."
                )

        elif X.split == 0:
            if Y.split is None:
                d._DNDarray__array = metric(X._DNDarray__array, Y._DNDarray__array)

            elif Y.split == 0:
                if X.shape[1] != Y.shape[1]:
                    raise ValueError("Inputs must have same shape[1]")

                # print("Rank {}\n X._DNDarray__array  = {}, \nY._DNDarray__array = {}".format(X.comm.rank,X._DNDarray__array, Y._DNDarray__array))
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
                rows = (xdispl[rank], xdispl[rank + 1] if (rank + 1) != size else m)
                cols = (ydispl[rank], ydispl[rank + 1] if (rank + 1) != size else n)

                # 0th iteration, calculate diagonal
                d_ij = metric(x_, stationary)
                d[rows[0] : rows[1], cols[0] : cols[1]] = d_ij

                for iter in range(1, num_iter):
                    # Send rank's part of the matrix to the next process in a circular fashion
                    receiver = (rank + iter) % size
                    sender = (rank - iter) % size

                    col1 = ydispl[sender]
                    if sender != size - 1:
                        col2 = ydispl[sender + 1]
                    else:
                        col2 = n
                    columns = (col1, col2)

                    # All but the first iter processes are receiving, then sending
                    if (rank // iter) != 0:
                        stat = MPI.Status()
                        Y.comm.handle.Probe(source=sender, tag=iter, status=stat)
                        count = int(stat.Get_count(MPI.FLOAT) / f)
                        moving = torch.zeros((count, f), dtype=torch.float32)
                        Y.comm.Recv(moving, source=sender, tag=iter)

                    # Sending to next Process
                    Y.comm.Send(stationary, dest=receiver, tag=iter)

                    # The first iter processes can now receive after sending
                    if (rank // iter) == 0:
                        stat = MPI.Status()
                        Y.comm.handle.Probe(source=sender, tag=iter, status=stat)
                        count = int(stat.Get_count(MPI.FLOAT) / f)
                        moving = torch.zeros((count, f), dtype=torch.float32)
                        Y.comm.Recv(moving, source=sender, tag=iter)

                    d_ij = metric(x_, moving)
                    d[rows[0] : rows[1], columns[0] : columns[1]] = d_ij

            else:
                raise NotImplementedError(
                    "Splittings other than 0 or None currently not supported."
                )

        else:
            # ToDo: Find out if even possible
            raise NotImplementedError("Splittings other than 0 or None currently not supported.")

    return d
