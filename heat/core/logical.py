import numpy as np
import torch

from .communication import MPI
from . import factories
from . import manipulations
from . import operations
from . import dndarray
from . import stride_tricks
from . import types

__all__ = [
    "all",
    "allclose",
    "any",
    "isclose",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
]


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

    return operations.__reduce_op(
        x, local_all, MPI.LAND, axis=axis, out=out, neutral=1, keepdim=keepdim
    )


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

    t1, t2 = __sanitize_close_input(x, y)

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

    return operations.__reduce_op(
        x, local_any, MPI.LOR, axis=axis, out=out, neutral=0, keepdim=keepdim
    )


def isclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Parameters:
    -----------
    x, y : tensor
        Input tensors to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN’s as equal. If True, NaN’s in x will be considered equal to NaN’s in y in the output array.

    Returns:
    --------
    isclose : boolean tensor of where a and b are equal within the given tolerance.
        If both x and y are scalars, returns a single boolean value.
    """
    t1, t2 = __sanitize_close_input(x, y)

    # no sanitation for shapes of x and y needed, torch.isclose raises relevant errors
    _local_isclose = torch.isclose(t1._DNDarray__array, t2._DNDarray__array, rtol, atol, equal_nan)

    # If x is distributed, then y is also distributed along the same axis
    if t1.comm.is_distributed() and t1.split is not None:
        output_gshape = stride_tricks.broadcast_shape(t1.gshape, t2.gshape)
        res = torch.empty(output_gshape).bool()
        t1.comm.Allgather(_local_isclose, res)
        result = factories.array(res, dtype=types.bool, device=t1.device, split=t1.split)
    else:
        if _local_isclose.dim() == 0:
            # both x and y are scalars, return a single boolean value
            result = bool(factories.array(_local_isclose).item())
        else:
            result = factories.array(_local_isclose, dtype=types.bool, device=t1.device)

    return result


def logical_and(t1, t2):
    """
    Compute the truth value of t1 AND t2 element-wise.

    Parameters:
    -----------
    t1, t2: tensor
        input tensors of same shape

    Returns:
    --------
    boolean_tensor : tensor of type bool
        Element-wise result of t1 AND t2.

    Examples:
    ---------
    >>> ht.logical_and(ht.array([True, False]), ht.array([False, False]))
    tensor([ False, False])
    """
    return operations.__binary_op(
        torch.Tensor.__and__, types.bool(t1, device=t1.device), types.bool(t2, device=t2.device)
    )


def logical_not(t, out=None):
    """
    Computes the element-wise logical NOT of the given input tensor.

    Parameters:
    -----------
    t1: tensor
        input tensor
    out : tensor, optional
        Alternative output tensor in which to place the result. It must have the same shape as the expected output.
        The output is a tensor with dtype=bool.

    Returns:
    --------
    boolean_tensor : tensor of type bool
        Element-wise result of NOT t.

    Examples:
    ---------
    >>> ht.logical_not(ht.array([True, False]))
    tensor([ False,  True])
    """
    return operations.__local_op(torch.logical_not, t, out)


def logical_or(t1, t2):
    """
    Compute the truth value of t1 OR t2 element-wise.

    Parameters:
    -----------
    t1, t2: tensor
        input tensors of same shape

    Returns:
    --------
    boolean_tensor : tensor of type bool
        Element-wise result of t1 OR t2.

    Examples:
    ---------
    >>> ht.logical_or(ht.array([True, False]), ht.array([False, False]))
    tensor([True, False])
    """
    return operations.__binary_op(
        torch.Tensor.__or__, types.bool(t1, device=t1.device), types.bool(t2, device=t2.device)
    )


def logical_xor(t1, t2):
    """
    Computes the element-wise logical XOR of the given input tensors.

    Parameters:
    -----------
    t1, t2: tensor
        input tensors of same shape

    Returns:
    --------
    boolean_tensor : tensor of type bool
        Element-wise result of t1 XOR t2.

    Examples:
    ---------
    >>> ht.logical_xor(ht.array([True, False, True]), ht.array([True, False, False]))
    tensor([ False, False,  True])
    """
    return operations.__binary_op(torch.logical_xor, t1, t2)


def __sanitize_close_input(x, y):
    """
    Makes sure that both x and y are ht.DNDarrays.
    Provides copies of x and y distributed along the same split axis (if original split axes do not match).
    """

    def sanitize_input_type(x, y):
        """
        Verifies that x is either a scalar, or a ht.DNDarray. If a scalar, x gets wrapped in a ht.DNDarray.
        Raises TypeError if x is neither.
        """
        if not isinstance(x, dndarray.DNDarray):
            if np.ndim(x) == 0:
                dtype = getattr(x, "dtype", float)
                device = getattr(y, "device", None)
                x = factories.array(x, dtype=dtype, device=device)
            else:
                raise TypeError("Expected DNDarray or numeric scalar, input was {}".format(type(x)))

        return x

    x = sanitize_input_type(x, y)
    y = sanitize_input_type(y, x)

    # Do redistribution out-of-place
    # If only one of the tensors is distributed, unsplit/gather it
    if x.split is not None and y.split is None:
        t1 = manipulations.resplit(x, axis=None)
        return t1, y

    elif x.split != y.split:
        t2 = manipulations.resplit(y, axis=x.split)
        return x, t2

    else:
        return x, y
