import numpy as np
import torch

from typing import Callable, Optional, Tuple, Union

from . import factories
from . import manipulations
from . import operations
from . import stride_tricks
from . import types

from .communication import MPI
from .dndarray import DNDarray

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


def all(
    x: DNDarray,
    axis: Union[int, Tuple[int], None] = None,
    out: Optional[DNDarray] = None,
    keepdim: bool = False,
) -> Union[DNDarray, bool]:
    """
    Test whether all array elements along a given axis evaluate to ``True``.
    A new boolean or :class:`~heat.core.dndarray.DNDarray` is returned unless out is specified, in which case a
    reference to ``out`` is returned.

    Parameters
    -----------
    x : DNDarray
        Input array or object that can be converted to an array.
    axis : None or int or Tuple[int,...], optional
        Axis or axes along which a logical AND reduction is performed. The default (``axis=None``) is to perform a
        logical AND over all the dimensions of the input array. ``axis`` may be negative, in which case it counts
        from the last to the first axis.
    out : DNDarray, optional
        Alternate output array in which to place the result. It must have the same shape as the expected output
        and its type is preserved.
    keepdim : bool, optional
        If this is set to ``True``, the axes which are reduced are left in the result as dimensions with size one.
        With this option, the result will broadcast correctly against the original array.

    Examples
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


DNDarray.all: Callable[
    [Union[int, Tuple[int], None], Optional[DNDarray], bool], Union[DNDarray, bool]
] = lambda self, axis=None, out=None, keepdim=False: all(self, axis, out, keepdim)
DNDarray.all.__doc__ = all.__doc__


def allclose(
    x: DNDarray, y: DNDarray, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> bool:
    """
    Test whether two tensors are element-wise equal within a tolerance. Returns ``True`` if ``|x-y|<=atol+rtol*|y|``
    for all elements of ``x`` and ``y``, ``False`` otherwise

    Parameters
    -----------
    x : DNDarray
        First array to compare
    y : DNDarray
        Second array to compare
    atol: float, optional
        Absolute tolerance.
    rtol: float, optional
        Relative tolerance (with respect to ``y``).
    equal_nan: bool, optional
        Whether to compare NaN’s as equal. If ``True``, NaN’s in a will be considered equal to NaN’s in b in the output
        array.

    Examples
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


DNDarray.allclose: Callable[
    [DNDarray, DNDarray, float, float, bool], bool
] = lambda self, other, rtol=1e-05, atol=1e-08, equal_nan=False: allclose(
    self, other, rtol, atol, equal_nan
)
DNDarray.allclose.__doc__ = all.__doc__


def any(
    x, axis: Optional[int] = None, out: Optional[DNDarray] = None, keepdim: bool = False
) -> DNDarray:
    """
    Test whether any array element along a given axis evaluates to ``True``.
    The returning array is one dimensional unless axis is not ``None``.

    Parameters
    -----------
    x : DNDarray
        Input tensor
    axis : int, optional
        Axis along which a logic OR reduction is performed. With ``axis=None``, the logical OR is performed over all
        dimensions of the array.
    out : DNDarray, optional
        Alternative output tensor in which to place the result. It must have the same shape as the expected output.
        The output is a array with ``datatype=bool``.
    keepdim : bool, optional
        If this is set to ``True``, the axes which are reduced are left in the result as dimensions with size one.
        With this option, the result will broadcast correctly against the original array.

    Examples
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


DNDarray.any: Callable[
    [DNDarray, Optional[int], Optional[DNDarray], bool], DNDarray
] = lambda self, axis=None, out=None, keepdim=False: allclose(self, axis, out, keepdim)
DNDarray.any.__doc__ = any.__doc__


def isclose(
    x: DNDarray, y: DNDarray, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> DNDarray:
    """
    Returns a boolean :class:`~heat.core.dndarray.DNDarray`, with elements ``True`` where ``a`` and ``b`` are equal
    within the given tolerance. If both ``x`` and ``y`` are scalars, returns a single boolean value.

    Parameters
    -----------
    x : DNDarray
        Input array to compare.
    y : DNDarray
        Input array to compare.
    rtol : float
        The relative tolerance parameter.
    atol : float
        The absolute tolerance parameter.
    equal_nan : bool
        Whether to compare NaN’s as equal. If ``True``, NaN’s in x will be considered equal to NaN’s in y in the output array.
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


DNDarray.isclose: Callable[
    [DNDarray, DNDarray, float, float, bool], DNDarray
] = lambda self, other, rtol=1e-05, atol=1e-08, equal_nan=False: isclose(
    self, other, rtol, atol, equal_nan
)
DNDarray.isclose.__doc__ = isclose.__doc__


def logical_and(t1: DNDarray, t2: DNDarray) -> DNDarray:
    """
    Compute the truth value of ``t1`` AND ``t2`` element-wise.

    Parameters
    -----------
    t1 : DNDarray
        Input array of same shape
    t2 : DNDarray
        Input array of same shape

    Examples
    ---------
    >>> ht.logical_and(ht.array([True, False]), ht.array([False, False]))
    tensor([ False, False])
    """
    return operations.__binary_op(
        torch.Tensor.__and__, types.bool(t1, device=t1.device), types.bool(t2, device=t2.device)
    )


def logical_not(t: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Computes the element-wise logical NOT of the given input :class:`~heat.core.dndarray.DNDarray` .

    Parameters
    -----------
    t: DNDarray
        Input array
    out : DNDarray, optional
        Alternative output array in which to place the result. It must have the same shape as the expected output.
        The output is a ``DNDarray`` with ``datatype=bool``.

    Examples
    ---------
    >>> ht.logical_not(ht.array([True, False]))
    tensor([ False,  True])
    """
    return operations.__local_op(torch.logical_not, t, out)


def logical_or(t1: DNDarray, t2: DNDarray) -> DNDarray:
    """
    Compute the truth value of ``t1`` OR ``t2`` element-wise.

    Parameters
    -----------
    t1 : DNDarray
        Input array of same shape
    t2 : DNDarray
        Input array of same shape

    Examples
    ---------
    >>> ht.logical_or(ht.array([True, False]), ht.array([False, False]))
    tensor([True, False])
    """
    return operations.__binary_op(
        torch.Tensor.__or__, types.bool(t1, device=t1.device), types.bool(t2, device=t2.device)
    )


def logical_xor(t1: DNDarray, t2: DNDarray) -> DNDarray:
    """
    Computes the element-wise logical XOR of the given input :class:`~heat.core.dndarray.DNDarray` .

    Parameters
    -----------
    t1 : DNDarray
        Input array of same shape
    t2 : DNDarray
        Input array of same shape

    Examples
    ---------
    >>> ht.logical_xor(ht.array([True, False, True]), ht.array([True, False, False]))
    tensor([ False, False,  True])
    """
    return operations.__binary_op(torch.logical_xor, t1, t2)


def __sanitize_close_input(x: DNDarray, y: DNDarray) -> Tuple[DNDarray, DNDarray]:
    """
    Makes sure that both ``x`` and ``y`` are :class:`~heat.core.dndarray.DNDarray`.
    Provides copies of ``x`` and ``y`` distributed along the same split axis (if original split axes do not match).

    Parameters
    -----------
    x : DNDarray
        The left-hand side operand.
    y : DNDarray
        The right-hand side operand.

    Raises
    ------
    TypeError
        If ``x`` is neither.
    """

    def sanitize_input_type(
        x: Union[int, float, DNDarray], y: Union[int, float, DNDarray]
    ) -> DNDarray:
        """
        Verifies that ``x`` and ``y`` are either scalar, or a :class:`~heat.core.dndarray.DNDarray`.
        In the former case, the scalar is wrapped in a :class:`~heat.core.dndarray.DNDarray`.

        Raises
        ------
        TypeError
            If ``x`` or ``y`` are not
        """
        if not isinstance(x, DNDarray):
            if np.ndim(x) != 0:
                raise TypeError("Expected DNDarray or numeric scalar, input was {}".format(type(x)))

            dtype = getattr(x, "dtype", float)
            device = getattr(y, "device", None)
            x = factories.array(x, dtype=dtype, device=device)

        return x

    x = sanitize_input_type(x, y)
    y = sanitize_input_type(y, x)

    # if one of the tensors is distributed, unsplit/gather it
    if x.split is not None and y.split is None:
        t1 = manipulations.resplit(x, axis=None)
        return t1, y

    elif x.split != y.split:
        t2 = manipulations.resplit(y, axis=x.split)
        return x, t2

    else:
        return x, y
