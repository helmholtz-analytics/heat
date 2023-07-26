"""
Logical functions for the DNDarrays
"""

import numpy as np
import torch

from typing import Callable, Optional, Tuple, Union

from . import factories
from . import manipulations

from . import _operations
from . import stride_tricks
from . import types

from .communication import MPI
from .dndarray import DNDarray

__all__ = [
    "all",
    "allclose",
    "any",
    "isclose",
    "isfinite",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "signbit",
]


def all(
    x: DNDarray,
    axis: Union[int, Tuple[int], None] = None,
    out: Optional[DNDarray] = None,
    keepdims: bool = False,
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
    keepdims : bool, optional
        If this is set to ``True``, the axes which are reduced are left in the result as dimensions with size one.
        With this option, the result will broadcast correctly against the original array.

    Examples
    ---------
    >>> x = ht.random.randn(4, 5)
    >>> x
    DNDarray([[ 0.7199,  1.3718,  1.5008,  0.3435,  1.2884],
              [ 0.1532, -0.0968,  0.3739,  1.7843,  0.5614],
              [ 1.1522,  1.9076,  1.7638,  0.4110, -0.2803],
              [-0.5475, -0.0271,  0.8564, -1.5870,  1.3108]], dtype=ht.float32, device=cpu:0, split=None)
    >>> y = x < 0.5
    >>> y
    DNDarray([[False, False, False,  True, False],
              [ True,  True,  True, False, False],
              [False, False, False,  True,  True],
              [ True,  True, False,  True, False]], dtype=ht.bool, device=cpu:0, split=None)
    >>> ht.all(y)
    DNDarray([False], dtype=ht.bool, device=cpu:0, split=None)
    >>> ht.all(y, axis=0)
    DNDarray([False, False, False, False, False], dtype=ht.bool, device=cpu:0, split=None)
    >>> ht.all(x, axis=1)
    DNDarray([True, True, True, True], dtype=ht.bool, device=cpu:0, split=None)
    >>> out = ht.zeros(5)
    >>> ht.all(y, axis=0, out=out)
    DNDarray([False, False, False, False, False], dtype=ht.float32, device=cpu:0, split=None)
    >>> out
    DNDarray([False, False, False, False, False], dtype=ht.float32, device=cpu:0, split=None)
    """

    def local_all(t, *args, **kwargs):
        return torch.all(t != 0, *args, **kwargs)

    if keepdims and axis is None:
        axis = tuple(range(x.ndim))

    return _operations.__reduce_op(
        x, local_all, MPI.LAND, axis=axis, out=out, neutral=1, keepdims=keepdims
    )


DNDarray.all: Callable[
    [Union[int, Tuple[int], None], Optional[DNDarray], bool], Union[DNDarray, bool]
] = lambda self, axis=None, out=None, keepdims=False: all(self, axis, out, keepdims)
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
        Whether to compare NaN’s as equal. If ``True``, NaN’s in ``x`` will be considered equal to NaN’s in ``y`` in
        the output array.

    Examples
    ---------
    >>> x = ht.float32([[2, 2], [2, 2]])
    >>> ht.allclose(x, x)
    True
    >>> y = ht.float32([[2.00005, 2.00005], [2.00005, 2.00005]])
    >>> ht.allclose(x, y)
    False
    >>> ht.allclose(x, y, atol=1e-04)
    True
    """
    t1, t2 = __sanitize_close_input(x, y)

    # no sanitation for shapes of x and y needed, torch.allclose raises relevant errors
    try:
        _local_allclose = torch.tensor(torch.allclose(t1.larray, t2.larray, rtol, atol, equal_nan))
    except RuntimeError:
        promoted_dtype = torch.promote_types(t1.larray.dtype, t2.larray.dtype)
        _local_allclose = torch.tensor(
            torch.allclose(
                t1.larray.type(promoted_dtype),
                t2.larray.type(promoted_dtype),
                rtol,
                atol,
                equal_nan,
            )
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
    x, axis: Optional[int] = None, out: Optional[DNDarray] = None, keepdims: bool = False
) -> DNDarray:
    """
    Returns a :class:`~heat.core.dndarray.DNDarray` containing the result of the test whether any array elements along a
    given axis evaluate to ``True``.
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
    keepdims : bool, optional
        If this is set to ``True``, the axes which are reduced are left in the result as dimensions with size one.
        With this option, the result will broadcast correctly against the original array.

    Examples
    ---------
    >>> x = ht.float32([[0.3, 0, 0.5]])
    >>> x.any()
    DNDarray([True], dtype=ht.bool, device=cpu:0, split=None)
    >>> x.any(axis=0)
    DNDarray([ True, False,  True], dtype=ht.bool, device=cpu:0, split=None)
    >>> x.any(axis=1)
    DNDarray([True], dtype=ht.bool, device=cpu:0, split=None)
    >>> y = ht.int32([[0, 0, 1], [0, 0, 0]])
    >>> res = ht.zeros(3, dtype=ht.bool)
    >>> y.any(axis=0, out=res)
    DNDarray([False, False,  True], dtype=ht.bool, device=cpu:0, split=None)
    >>> res
    DNDarray([False, False,  True], dtype=ht.bool, device=cpu:0, split=None)
    """

    def local_any(t, *args, **kwargs):
        return torch.any(t != 0, *args, **kwargs)

    if keepdims and axis is None:
        axis = tuple(range(x.ndim))

    return _operations.__reduce_op(
        x, local_any, MPI.LOR, axis=axis, out=out, neutral=0, keepdims=keepdims
    )


DNDarray.any: Callable[
    [DNDarray, Optional[int], Optional[DNDarray], bool], DNDarray
] = lambda self, axis=None, out=None, keepdims=False: any(self, axis, out, keepdims)
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
        Whether to compare NaN’s as equal. If ``True``, NaN’s in x will be considered equal to NaN’s in y in the output
        array.
    """
    t1, t2 = __sanitize_close_input(x, y)

    # no sanitation for shapes of x and y needed, torch.isclose raises relevant errors
    _local_isclose = torch.isclose(t1.larray, t2.larray, rtol, atol, equal_nan)

    # If x is distributed, then y is also distributed along the same axis
    if t1.comm.is_distributed() and t1.split is not None:
        output_gshape = stride_tricks.broadcast_shape(t1.gshape, t2.gshape)
        res = torch.empty(output_gshape, device=t1.device.torch_device).bool()
        t1.comm.Allgather(_local_isclose, res)
        result = DNDarray(
            res,
            gshape=output_gshape,
            dtype=types.bool,
            split=t1.split,
            device=t1.device,
            comm=t1.comm,
            balanced=t1.is_balanced,
        )
    else:
        if _local_isclose.dim() == 0:
            # both x and y are scalars, return a single boolean value
            result = bool(_local_isclose.item())
        else:
            result = DNDarray(
                _local_isclose,
                gshape=tuple(_local_isclose.shape),
                dtype=types.bool,
                split=None,
                device=t1.device,
                comm=t1.comm,
                balanced=t1.is_balanced,
            )

    return result


def isfinite(x: DNDarray) -> DNDarray:
    """
    Test element-wise for finiteness (not infinity or not Not a Number) and return result as a boolean
    :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x : DNDarray
        Input tensor

    Examples
    --------
    >>> ht.isfinite(ht.array([1, ht.inf, -ht.inf, ht.nan]))
    DNDarray([ True, False, False, False], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.isfinite, x, None, no_cast=True)


def isinf(x: DNDarray) -> DNDarray:
    """
    Test element-wise for positive or negative infinity and return result as a boolean
    :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x : DNDarray
        Input tensor

    Examples
    --------
    >>> ht.isinf(ht.array([1, ht.inf, -ht.inf, ht.nan]))
    DNDarray([False,  True,  True, False], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.isinf, x, None, no_cast=True)


def isnan(x: DNDarray) -> DNDarray:
    """
    Test element-wise for NaN and return result as a boolean :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x   : DNDarray
          Input tensor

    Examples
    --------
    >>> ht.isnan(ht.array([1, ht.inf, -ht.inf, ht.nan]))
    DNDarray([False, False, False,  True], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.isnan, x, None, no_cast=True)


def isneginf(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Test if each element of `x` is negative infinite, return result as a boolean :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x   : DNDarray
          Input tensor
    out : DNDarray, optional
          Alternate output array in which to place the result. It must have the same shape as the expected output
          and its type is preserved.

    Examples
    --------
    >>> ht.isnan(ht.array([1, ht.inf, -ht.inf, ht.nan]))
    DNDarray([False, False, True, False], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.isneginf, x, out, no_cast=True)


def isposinf(x: DNDarray, out: Optional[DNDarray] = None):
    """
    Test if each element of `x` is positive infinite, return result as a boolean :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x   : DNDarray
          Input tensor
    out : DNDarray, optional
          Alternate output array in which to place the result. It must have the same shape as the expected output
          and its type is preserved.

    Examples
    --------
    >>> ht.isnan(ht.array([1, ht.inf, -ht.inf, ht.nan]))
    DNDarray([False, True, False, False], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.isposinf, x, out, no_cast=True)


DNDarray.isclose: Callable[
    [DNDarray, DNDarray, float, float, bool], DNDarray
] = lambda self, other, rtol=1e-05, atol=1e-08, equal_nan=False: isclose(
    self, other, rtol, atol, equal_nan
)
DNDarray.isclose.__doc__ = isclose.__doc__


def logical_and(x: DNDarray, y: DNDarray) -> DNDarray:
    """
    Compute the truth value of ``x`` AND ``y`` element-wise. Returns a boolean :class:`~heat.core.dndarray.DNDarray` containing the truth value of ``x`` AND ``y`` element-wise.

    Parameters
    -----------
    x : DNDarray
        Input array of same shape
    y : DNDarray
        Input array of same shape

    Examples
    ---------
    >>> ht.logical_and(ht.array([True, False]), ht.array([False, False]))
    DNDarray([False, False], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__binary_op(
        torch.logical_and, types.bool(x, device=x.device), types.bool(y, device=y.device)
    )


def logical_not(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Computes the element-wise logical NOT of the given input :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    -----------
    x : DNDarray
        Input array
    out : DNDarray, optional
        Alternative output array in which to place the result. It must have the same shape as the expected output.
        The output is a :class:`~heat.core.dndarray.DNDarray` with ``datatype=bool``.

    Examples
    ---------
    >>> ht.logical_not(ht.array([True, False]))
    DNDarray([False,  True], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.logical_not, x, out)


def logical_or(x: DNDarray, y: DNDarray) -> DNDarray:
    """
    Returns boolean :class:`~heat.core.dndarray.DNDarray` containing the element-wise logical NOT of the given
    input :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    -----------
    x : DNDarray
        Input array of same shape
    y : DNDarray
        Input array of same shape

    Examples
    ---------
    >>> ht.logical_or(ht.array([True, False]), ht.array([False, False]))
    DNDarray([ True, False], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__binary_op(
        torch.logical_or, types.bool(x, device=x.device), types.bool(y, device=y.device)
    )


def logical_xor(x: DNDarray, y: DNDarray) -> DNDarray:
    """
    Computes the element-wise logical XOR of the given input :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    -----------
    x : DNDarray
        Input array of same shape
    y : DNDarray
        Input array of same shape

    Examples
    ---------
    >>> ht.logical_xor(ht.array([True, False, True]), ht.array([True, False, False]))
    DNDarray([False, False,  True], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__binary_op(torch.logical_xor, x, y)


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
        If ``x`` is neither :class:`~heat.core.dndarray.DNDarray` or numeric scalar
    """

    def sanitize_input_type(
        x: Union[int, float, DNDarray], y: Union[int, float, DNDarray]
    ) -> DNDarray:
        """
        Verifies that ``x`` and ``y`` are either scalar, or a :class:`~heat.core.dndarray.DNDarray`.
        In the former case, the scalar is wrapped in a :class:`~heat.core.dndarray.DNDarray`.

        Parameters
        -----------
        x : Union[int, float, DNDarray]
            The left-hand side operand.
        y : Union[int, float, DNDarray]
            The right-hand side operand.

        Raises
        ------
        TypeError
            If ``x`` or ``y`` are not
        """
        if not isinstance(x, DNDarray):
            if np.ndim(x) != 0:
                raise TypeError(f"Expected DNDarray or numeric scalar, input was {type(x)}")

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


def signbit(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Checks if signbit is set element-wise (less than zero).

    Parameters
    ----------
    x : DNDarray
        The input array.
    out : DNDarray, optional
        The output array.

    Examples
    --------
    >>> a = ht.array([2, -1.3, 0])
    >>> ht.signbit(a)
    DNDarray([False,  True, False], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.signbit, x, out, no_cast=True)
