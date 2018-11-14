import torch

from .communication import MPI_WORLD
from . import tensor
from . import types


def set_gseed(seed):
    # TODO: think about proper random number generation
    # TODO: comment me
    # TODO: test me
    torch.manual_seed(seed)


def uniform(low=0.0, high=1.0, size=None, comm=MPI_WORLD):
    # TODO: comment me
    # TODO: test me
    if size is None:
        size = (1,)

    return tensor(torch.Tensor(*size).uniform_(low, high), size, types.float32, None, comm)


def randn(*args):
    """
    Returns a tensor filled with random numbers from a standard normal distribution with zero mean and variance of one.

    The shape of the tensor is defined by the varargs args.

    Parameters
    ----------
    d0, d1, …, dn : int, optional
        The dimensions of the returned array, should be all positive.

    Returns
    -------
    broadcast_shape : tuple of ints
        the broadcast shape

    Raises
    -------
    TypeError
        If one of d0 to dn is not an int.
    ValueError
        If one of d0 to dn is less or equal to 0.

    Examples
    --------
    >>> ht.randn(3)
    tensor([ 0.1921, -0.9635,  0.5047])

    >>> ht.randn(4, 4)
    tensor([[-1.1261,  0.5971,  0.2851,  0.9998],
            [-1.8548, -1.2574,  0.2391, -0.3302],
            [ 1.3365, -1.5212,  1.4159, -0.1671],
            [ 0.1260,  1.2126, -0.0804,  0.0907]])
    """
    # TODO: make me splitable
    # check if all positional arguments are integers
    if not all(isinstance(_, int) for _ in args):
        raise TypeError('dimensions have to be integers')
    if not all(_ > 0 for _ in args):
        raise ValueError('negative dimension are not allowed')

    gshape = tuple(args) if args else (1,)
    try:
        torch.randn(gshape)
    except RuntimeError as exception:
        # re-raise the exception to be consistent with numpy's exception interface
        raise ValueError(str(exception))

    # compose the local tensor
    data = torch.randn(args)

    return tensor(data, gshape, types.canonical_heat_type(data.dtype), None, MPI_WORLD)
