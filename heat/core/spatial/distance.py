import torch

from .. import communication
from .. import factories
from .. import types


def _euclidian(x, y):
    # X and Y are torch tensors
    # k1, f1 = x.shape
    # k2, f2 = y.shape
    # if f1 != f2:
    #    raise RuntimeError(
    #        "X and Y have differing feature dimensions (dim = 1), should be equal, but are {} and {}".format(
    #            f1, f2
    #        )
    #    )
    # xd = x.unsqueeze(dim=1)
    # yd = y.unsqueeze(dim=0)
    # result = torch.zeros((k1, k2), dtype=torch.float64)

    # for i in range(xd.shape[0]):
    #    result[i, :] = ((yd - xd[i, :, :]) ** 2).sum(dim=-1).sqrt()
    result = torch.cdist(x, y)
    return result


def _gaussian(x, y, sigma=1.0):
    # X and Y are torch tensors
    # k1, f1 = x.shape
    # k2, f2 = y.shape
    # if f1 != f2:
    #    raise RuntimeError(
    #        "X and Y have differing feature dimensions (dim = 1), should be equal, but are {} and {}".format(
    #            f1, f2
    #        )
    #    )
    # xd = x.unsqueeze(dim=1)
    # yd = y.unsqueeze(dim=0)
    # result = torch.zeros((k1, k2), dtype=torch.float64)
    # for i in range(xd.shape[0]):
    #   result[i, :] = torch.exp(
    #      -((yd - yd[i, :, :]) ** 2).sum(dim=-1) / (2 * sigma * sigma)
    # )
    d = torch.cdist(x, y)
    result = torch.exp(-d ** 2 / (2 * sigma * sigma))
    return result


def similarity(X, metric=_euclidian):
    if (X.split is not None) and (X.split != 0):
        raise NotImplementedError("Feature Splitting is not supported")
    if len(X.shape) > 2:
        raise NotImplementedError("Only 2D data matrices are supported")

    comm = X.comm
    rank = comm.Get_rank()
    size = comm.Get_size()

    K, f = X.shape

    S = factories.zeros((K, K), dtype=types.float64, split=0)

    counts, displ, _ = comm.counts_displs_shape(X.shape, X.split)
    num_iter = (size + 1) // 2

    stationary = X._DNDarray__array
    rows = (displ[rank], displ[rank + 1] if (rank + 1) != size else K)

    # 0th iteration, calculate diagonal
    d_ij = metric(stationary, stationary)
    S[rows[0] : rows[1], rows[0] : rows[1]] = d_ij

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
            stat = communication.Status()
            comm.handle.Probe(source=sender, tag=iter, status=stat)
            count = int(stat.Get_count(communication.FLOAT) / f)
            moving = torch.zeros((count, f), dtype=torch.float32)
            comm.Recv(moving, source=sender, tag=iter)

        # Sending to next Process
        comm.Send(stationary, dest=receiver, tag=iter)

        # The first iter processes can now receive after sending
        if (rank // iter) == 0:
            stat = communication.Status()
            comm.handle.Probe(source=sender, tag=iter, status=stat)
            count = int(stat.Get_count(communication.FLOAT) / f)
            moving = torch.zeros((count, f), dtype=torch.float32)
            comm.Recv(moving, source=sender, tag=iter)

        d_ij = metric(stationary, moving)
        S[rows[0] : rows[1], columns[0] : columns[1]] = d_ij

        # Receive calculated tile
        scol1 = displ[receiver]
        if receiver != size - 1:
            scol2 = displ[receiver + 1]
        else:
            scol2 = K
        scolumns = (scol1, scol2)
        symmetric = torch.zeros(scolumns[1] - scolumns[0], (rows[1] - rows[0]), dtype=torch.float64)
        # Receive calculated tile
        if (rank // iter) != 0:
            comm.Recv(symmetric, source=receiver, tag=iter)

        # sending result back to sender of moving matrix (for symmetry)
        comm.Send(d_ij, dest=sender, tag=iter)
        if (rank // iter) == 0:
            comm.Recv(symmetric, source=receiver, tag=iter)
        S[rows[0] : rows[1], scolumns[0] : scolumns[1]] = symmetric.transpose(0, 1)

    if (size + 1) % 2 != 0:  # we need one mor iteration for the first n/2 processes
        receiver = (rank + num_iter) % size
        sender = (rank - num_iter) % size

        # Case 1: only receiving
        if rank < (size // 2):
            stat = communication.Status()
            comm.handle.Probe(source=sender, tag=num_iter, status=stat)
            count = int(stat.Get_count(communication.FLOAT) / f)
            moving = torch.zeros((count, f), dtype=torch.float32)
            comm.Recv(moving, source=sender, tag=num_iter)

            col1 = displ[sender]
            if sender != size - 1:
                col2 = displ[sender + 1]
            else:
                col2 = K
            columns = (col1, col2)

            d_ij = metric(stationary, moving)
            S[rows[0] : rows[1], columns[0] : columns[1]] = d_ij
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
                (scolumns[1] - scolumns[0], rows[1] - rows[0]), dtype=torch.float64
            )
            comm.Recv(symmetric, source=receiver, tag=num_iter)
            S[rows[0] : rows[1], scolumns[0] : scolumns[1]] = symmetric.transpose(0, 1)

    return S


def pairwise(X, Y, metric=_euclidian):
    if ((X.split is not None) and (X.split != 0)) or ((X.split is not None) and (X.split != 0)):
        raise NotImplementedError("Feature Splitting is not supported")
    if len(X.shape) > 2 or len(Y.shape) > 2:
        raise NotImplementedError("Only 2D data matrices are supported")
    if X.comm != Y.comm:
        raise NotImplementedError("Differing communicators not supported")

    if X.shape[1] != Y.shape[1]:
        raise ValueError("inputs must have same number of features")

    comm = X.comm
    rank = comm.Get_rank()
    size = comm.Get_size()

    m, f = X.shape
    n = Y.shape[0]

    S = factories.zeros((m, n), dtype=types.float64, split=0)

    xcounts, xdispl, _ = X.comm.counts_displs_shape(X.shape, X.split)
    ycounts, ydispl, _ = Y.comm.counts_displs_shape(Y.shape, Y.split)
    num_iter = size

    x_ = X._DNDarray__array
    stationary = Y._DNDarray__array
    rows = (xdispl[rank], xdispl[rank + 1] if (rank + 1) != size else m)
    cols = (ydispl[rank], ydispl[rank + 1] if (rank + 1) != size else n)

    # 0th iteration, calculate diagonal
    d_ij = metric(x_, stationary)
    S[rows[0] : rows[1], cols[0] : cols[1]] = d_ij

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
            stat = communication.Status()
            Y.comm.handle.Probe(source=sender, tag=iter, status=stat)
            count = int(stat.Get_count(communication.FLOAT) / f)
            moving = torch.zeros((count, f), dtype=torch.float32)
            Y.comm.Recv(moving, source=sender, tag=iter)

        # Sending to next Process
        Y.comm.Send(stationary, dest=receiver, tag=iter)

        # The first iter processes can now receive after sending
        if (rank // iter) == 0:
            stat = communication.Status()
            Y.comm.handle.Probe(source=sender, tag=iter, status=stat)
            count = int(stat.Get_count(communication.FLOAT) / f)
            moving = torch.zeros((count, f), dtype=torch.float32)
            Y.comm.Recv(moving, source=sender, tag=iter)

        d_ij = metric(stationary, moving)
        S[rows[0] : rows[1], columns[0] : columns[1]] = d_ij

    return S
