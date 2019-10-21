import numpy as np
import torch

from .communication import MPI
from . import factories
from . import manipulations
from . import operations
from . import dndarray

__all__ = ["all", "allclose", "any"]


def all(x, axis=None, out=None, keepdim=None):
    """
    Test whether all array elements along a given axis evaluate to True.

    Parameters:
    -----------
    x : ht.DNDarray
        Input array or object that can be converted to an array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a logical AND reduction is performed. The default (axis = None) is to perform a
        logical AND over all the dimensions of the input array. axis may be negative, in which case it counts
        from the last to the first axis.
    out : ht.DNDarray, optional
        Alternate output array in which to place the result. It must have the same shape as the expected output
        and its type is preserved.

    Returns:
    --------
    all : ht.DNDarray, bool
        A new boolean or ht.DNDarray is returned unless out is specified, in which case a reference to out is returned.

    Examples:
    ---------
    >>> import heat as ht
    >>> a = ht.random.randn(4, 5)
    >>> a
    tensor([[ 0.5370, -0.4117, -3.1062,  0.4897, -0.3231],
            [-0.5005, -1.7746,  0.8515, -0.9494, -0.2238],
            [-0.0444,  0.3388,  0.6805, -1.3856,  0.5422],
            [ 0.3184,  0.0185,  0.5256, -1.1653, -0.1665]])
    >>> x = a < 0.5
    >>> x
    tensor([[0, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 1, 1]], dtype=ht.uint8)
    >>> ht.all(x)
    tensor([0], dtype=ht.uint8)
    >>> ht.all(x, axis=0)
    tensor([[0, 1, 0, 1, 0]], dtype=ht.uint8)
    >>> ht.all(x, axis=1)
    tensor([[0],
            [0],
            [0],
            [0]], dtype=ht.uint8)

    Write out to predefined buffer:
    >>> out = ht.zeros((1, 5))
    >>> ht.all(x, axis=0, out=out)
    >>> out
    tensor([[0, 1, 0, 1, 0]], dtype=ht.uint8)
    """
    # TODO: make me more numpy API complete. Issue #101
    def local_all(t, *args, **kwargs):
        return torch.all(t != 0, *args, **kwargs)

    return operations.__reduce_op(x, local_all, MPI.LAND, axis=axis, out=out, keepdim=keepdim)


def allclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Test whether two tensors are element-wise equal within a tolerance. Returns True if |x - y| <= atol + rtol * |y|
    for all elements of x and y, False otherwise

    Parameters:
    -----------
    x : ht.DNDarray
        First tensor to compare
    y : ht.DNDarray
        Second tensor to compare
    atol: float, optional
        Absolute tolerance. Default is 1e-08
    rtol: float, optional
        Relative tolerance (with respect to y). Default is 1e-05
    equal_nan: bool, optional
        Whether to compare NaN’s as equal. If True, NaN’s in a will be considered equal to NaN’s in b in the output
        array.

    Returns:
    --------
    allclose : bool
        True if the two tensors are equal within the given tolerance; False otherwise.

    Examples:
    ---------
    >>> a = ht.float32([[2, 2], [2, 2]])
    >>> ht.allclose(a, a)
    True
    >>> b = ht.float32([[2.00005, 2.00005], [2.00005, 2.00005]])
    >>> ht.allclose(a, b)
    False
    >>> ht.allclose(a, b, atol=1e-04)
    True
    """
    if np.isscalar(x):
        try:
            x = factories.array([float(x)])
        except (ValueError, TypeError):
            raise TypeError("Data type not supported, input was {}".format(type(x)))

    elif not isinstance(x, dndarray.DNDarray):
        raise TypeError(
            "Only tensors and numeric scalars are supported, but input was {}".format(type(x))
        )

    if np.isscalar(y):
        try:
            y = factories.array([float(y)])
        except (ValueError, TypeError):
            raise TypeError("Data type not supported, input was {}".format(type(y)))

    elif not isinstance(y, dndarray.DNDarray):
        raise TypeError(
            "Only tensors and numeric scalars are supported, but input was {}".format(type(y))
        )

    # Do redistribution out-of-place
    # If only one of the tensors is distributed, unsplit/gather it
    if (x.split is not None) and (y.split is None):
        t1 = manipulations.resplit(x, axis=None)
        t2 = y.copy()

    elif (x.split is None) and (y.split is not None):
        t1 = x.copy()
        t2 = manipulations.resplit(y, axis=None)

    # If both x and y are split, but along different axes, y is redistributed to be split along the same axis as x
    elif (x.split is not None) and (y.split is not None) and (x.split != y.split):
        t1 = x.copy()
        t2 = manipulations.resplit(y, axis=x.split)

    else:
        t1 = x.copy()
        t2 = y.copy()

    # no sanitation for shapes of x and y needed, torch.allclose raises relevant errors
    _local_allclose = torch.tensor(
        torch.allclose(t1._DNDarray__array, t2._DNDarray__array, rtol, atol, equal_nan)
    )

    # If x is distributed, then y is also distributed along the same axis
    if t1.comm.is_distributed():
        t1.comm.Allreduce(MPI.IN_PLACE, _local_allclose, MPI.LAND)

    return bool(_local_allclose.item())


def any(x, axis=None, out=None, keepdim=False):
    """
    Test whether any array element along a given axis evaluates to True.
    The returning tensor is one dimensional unless axis is not None.

    Parameters:
    -----------
    x : tensor
        Input tensor
    axis : int, optional
        Axis along which a logic OR reduction is performed. With axis=None, the logical OR is performed over all
        dimensions of the tensor.
    out : tensor, optional
        Alternative output tensor in which to place the result. It must have the same shape as the expected output.
        The output is a tensor with dtype=bool.

    Returns:
    --------
    boolean_tensor : tensor of type bool
        Returns a tensor of booleans that are 1, if any non-zero values exist on this axis, 0 otherwise.

    Examples:
    ---------
    >>> import heat as ht
    >>> t = ht.float32([[0.3, 0, 0.5]])
    >>> t.any()
    tensor([1], dtype=torch.uint8)
    >>> t.any(axis=0)
    tensor([[1, 0, 1]], dtype=torch.uint8)
    >>> t.any(axis=1)
    tensor([[1]], dtype=torch.uint8)

    >>> t = ht.int32([[0, 0, 1], [0, 0, 0]])
    >>> res = ht.zeros((1, 3), dtype=ht.bool)
    >>> t.any(axis=0, out=res)
    tensor([[0, 0, 1]], dtype=torch.uint8)
    >>> res
    tensor([[0, 0, 1]], dtype=torch.uint8)
    """

    def local_any(t, *args, **kwargs):
        return torch.any(t != 0, *args, **kwargs)

    return operations.__reduce_op(x, local_any, MPI.LOR, axis=axis, out=out, keepdim=keepdim)
