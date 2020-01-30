import torch
import numpy as np

from .. import core

__all__ = [
    "cdist"
]
def cdist(X,Y, quadratic_expansion=False):
    # ToDo Case X==Y
    result = core.factories.zeros((X.shape[0], Y.shape[0]), dtype=core.types.float32, split=X.split, device=X.device, comm=X.comm)

    if X.split is not None:
        if X.split!=0:
            # ToDo: Find out if even possible
            raise NotImplementedError('Splittings other than 0 or None currently not supported.')
        if Y.split is not None:
            # ToDo: This requires communication of calculated blocks, will be implemented with Similarity Matrix Calculation
            raise NotImplementedError('Currently not supported')

    if quadratic_expansion:
        x_norm = (X._DNDarray__array ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(Y._DNDarray__array, 0, 1)
        y_norm = (Y._DNDarray__array ** 2).sum(1).view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(X._DNDarray__array, y_t)
        result._DNDarray__array = torch.sqrt(torch.clamp(dist, 0.0, np.inf))

    else:
        result._DNDarray__array = torch.cdist(X._DNDarray__array, Y._DNDarray__array)

    return result
