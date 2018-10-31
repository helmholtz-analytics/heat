import functools
import torch
from . import types
from .communicator import mpi


# We use decorators to modulize methods and updating halos, later decorators can be integrated in the main code  

def halorize_local_operation(func):
    """
    A decorator for function "__local_operation" in heat.core.operations.py
    that updates halos after local operations
    Parameters
    ----------
    func : HeAT function

    Returns
    -------
    res : wrapper
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        res = func(*args, **kwargs)

        promoted_type = types.promote_types(args[1].dtype, types.float32)
        torch_type = promoted_type.torch_type()

        if args[1].halo_next is not None:
            res.halo_next = args[0](args[1].halo_next.type(torch_type))

        if args[1].halo_prev is not None:
            res.halo_prev = args[0](args[1].halo_prev.type(torch_type))

        return res

    return wrapper

