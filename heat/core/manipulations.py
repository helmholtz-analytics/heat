import torch

from . import dndarray
from . import factories
from . import operations
from . import stride_tricks
from . import types

__all__ = [
    'squeeze']


def squeeze(x, axis=None):
    """
    Remove single-dimensional entries from the shape of a tensor.

    Parameters:
    -----------
    x : ht.DNDarray
        Input data.

    axis : None or int or tuple of ints, optional


    Returns:
    --------	
    squeezed : ht.DNDarray
               The input tensor, but with all or a subset of the dimensions of length 1 removed. 


    Examples:
    >>> import heat as ht
    >>> import torch
    >>> torch.manual_seed(1)
    <torch._C.Generator object at 0x115704ad0>
    >>> a = ht.random.randn(1,3,1,5)
    >>> a
    tensor([[[[ 0.2673, -0.4212, -0.5107, -1.5727, -0.1232]],

            [[ 3.5870, -1.8313,  1.5987, -1.2770,  0.3255]],

            [[-0.4791,  1.3790,  2.5286,  0.4107, -0.9880]]]])
    >>> a.shape
    (1, 3, 1, 5)
    >>> ht.squeeze(a).shape
    (3, 5)
    >>> ht.squeeze(a)
    tensor([[ 0.2673, -0.4212, -0.5107, -1.5727, -0.1232],
            [ 3.5870, -1.8313,  1.5987, -1.2770,  0.3255],
            [-0.4791,  1.3790,  2.5286,  0.4107, -0.9880]])
    >>> ht.squeeze(a,axis=0).shape
    (3, 1, 5)
    >>> ht.squeeze(a,axis=-2).shape
    (1, 3, 5)
    >>> ht.squeeze(a,axis=1).shape
    Traceback (most recent call last):
    ...
    ValueError: Dimension along axis 1 is not 1 for shape (1, 3, 1, 5)
    """

    # Sanitize input
    if not isinstance(x, dndarray.DNDarray):
        raise TypeError('expected x to be a ht.DNDarray, but was {}'.format(type(x)))
    # Sanitize axis
    axis = stride_tricks.sanitize_axis(x.shape, axis)
    if axis is not None:
        if isinstance(axis, int):
            dim_is_one = (x.shape[axis] == 1)
        if isinstance(axis, tuple):
            dim_is_one = bool(factories.array(list(x.shape[dim] == 1 for dim in axis)).all()._DNDarray__array)
        if not dim_is_one:
            raise ValueError('Dimension along axis {} is not 1 for shape {}'.format(axis, x.shape))

    # Local squeeze
    if axis is None:
        axis = tuple(i for i, dim in enumerate(x.shape) if dim == 1)
    if isinstance(axis, int):
        axis = (axis,)
    out_shape = tuple(x.lshape[dim] for dim in range(len(x.lshape)) if not dim in axis)
    x_lsqueezed = x._DNDarray__array.reshape(out_shape)

    # Distributed squeeze
    if x.split is not None:
        split = None
        if x.comm.is_distributed():
            if x.split in axis:
                raise ValueError('Cannot split AND squeeze along same axis. Split is {}, axis is {} for shape {}'.format(
                    x.split, axis, x.shape))
            out_shape = tuple(x.gshape[dim] for dim in range(len(x.gshape)) if not dim in axis)
            x_gsqueezed = dndarray.empty(out_shape, dtype=x.dtype)
            x.comm.Allgather(x_lsqueezed, x_gsqueezed)
            return dndarray.DNDarray(
                x_gsqueezed,
                out_shape,
                x.dtype,
                split=split,
                device=x.device,
                comm=x.comm)

    return dndarray.DNDarray(
        x_lsqueezed,
        out_shape,
        x.dtype,
        split=None,
        device=x.device,
        comm=x.comm)
