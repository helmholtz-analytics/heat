import torch

from .communication import MPI
from . import dndarray
from . import factories
from . import operations
from . import stride_tricks
from . import types

__all__ = [
"genpad"
]


def genpad(self, signal, pad_prev, pad_next):
    """
    Generate a zero padding only for local arrays of the first and last rank in case of distributed computing,
    otherwise zero padding is added at begin and end of the global array

    Parameters
    ----------
    signal : torch tensor 
        The data array to which a zero padding is to be added
    pad_prev : int
        The length of the left padding area
    pad_next : int
       The length of the right padding area
            
    Returns
    -------
    out : torch tensor
        The padded data array 
    """
    # ToDo: generalize to ND tensors
    # set the default padding for non distributed arrays

    if len(signal.shape) != 1:
        raise ValueError('Signal must be 1D, but is {}-dimensional'.format(len(signal.shape)))

    # check if more than one rank is involved 
    if self.is_distributed():
   
        # set the padding of the first rank
        if self.comm.rank == 0:
            pad_next = None

        # set the padding of the last rank
        if self.comm.rank == self.comm.size-1:
            pad_prev = None
                    
    return torch.cat([_ for _ in (pad_prev, signal, pad_next) if _ is not None])
