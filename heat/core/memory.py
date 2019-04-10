from . import tensor

__all__ = [
    'copy'
]


def copy(a):
    """
    Return an array copy of the given object.

    Parameters
    ----------
    a : ht.Tensor
        Input data to be copied.

    Returns
    -------
    copied : ht.Tensor
        A copy of the original
    """
    if not isinstance(a, tensor.Tensor):
        raise TypeError('input needs to be a tensor')
    return tensor.Tensor(a._Tensor__array.clone(), a.shape, a.dtype, a.split, a.device, a.comm)
