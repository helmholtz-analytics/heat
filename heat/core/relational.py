"""
Functions for relational oprations, i.e. equal/no equal...
"""
from __future__ import annotations

import torch
import numpy as np

from typing import Union

from .communication import MPI
from .dndarray import DNDarray
from . import _operations
from . import dndarray
from . import types
from . import sanitation
from . import factories

__all__ = [
    "eq",
    "equal",
    "ge",
    "greater",
    "greater_equal",
    "gt",
    "le",
    "less",
    "less_equal",
    "lt",
    "ne",
    "not_equal",
]


def eq(x: Union[DNDarray, float, int], y: Union[DNDarray, float, int]) -> DNDarray:
    """
    Returns a :class:`~heat.core.dndarray.DNDarray` containing the results of element-wise comparision.
    Takes the first and second operand (scalar or :class:`~heat.core.dndarray.DNDarray`) whose elements are to be
    compared as argument.

    Parameters
    ----------
    x: DNDarray or scalar
        The first operand involved in the comparison
    y: DNDarray or scalar
        The second operand involved in the comparison

    Examples
    ---------
    >>> import heat as ht
    >>> x = ht.float32([[1, 2],[3, 4]])
    >>> ht.eq(x, 3.0)
    DNDarray([[False, False],
              [ True, False]], dtype=ht.bool, device=cpu:0, split=None)
    >>> y = ht.float32([[2, 2], [2, 2]])
    >>> ht.eq(x, y)
    DNDarray([[False,  True],
              [False, False]], dtype=ht.bool, device=cpu:0, split=None)
    """
    res = _operations.__binary_op(torch.eq, x, y)

    if res.dtype != types.bool:
        res = dndarray.DNDarray(
            res.larray.type(torch.bool),
            res.gshape,
            types.bool,
            res.split,
            res.device,
            res.comm,
            res.balanced,
        )

    return res


DNDarray.__eq__ = lambda self, other: eq(self, other)
DNDarray.__eq__.__doc__ = eq.__doc__


def equal(x: Union[DNDarray, float, int], y: Union[DNDarray, float, int]) -> bool:
    """
    Overall comparison of equality between two :class:`~heat.core.dndarray.DNDarray`. Returns ``True`` if two arrays
    have the same size and elements, and ``False`` otherwise.

    Parameters
    ----------
    x: DNDarray or scalar
        The first operand involved in the comparison
    y: DNDarray or scalar
        The second operand involved in the comparison

    Examples
    ---------
    >>> import heat as ht
    >>> x = ht.float32([[1, 2],[3, 4]])
    >>> ht.equal(x, ht.float32([[1, 2],[3, 4]]))
    True
    >>> y = ht.float32([[2, 2], [2, 2]])
    >>> ht.equal(x, y)
    False
    >>> ht.equal(x, 3.0)
    False
    """
    if np.isscalar(x) and np.isscalar(y):
        x = factories.array(x)
        y = factories.array(y)
    elif isinstance(x, DNDarray) and np.isscalar(y):
        if x.gnumel == 1:
            return equal(x.item(), y)
        return False
        # y = factories.full_like(x, fill_value=y)
    elif np.isscalar(x) and isinstance(y, DNDarray):
        if y.gnumel == 1:
            return equal(x, y.item())
        return False
        # x = factories.full_like(y, fill_value=x)
    else:  # elif isinstance(x, DNDarray) and isinstance(y, DNDarray):
        if x.gnumel == 1:
            return equal(x.item(), y)
        elif y.gnumel == 1:
            return equal(x, y.item())
        elif x.comm != y.comm:
            raise NotImplementedError("Not implemented for other comms")
        elif x.gshape != y.gshape:
            return False

        if x.split is None and y.split is None:
            pass
        elif x.split is None and y.split is not None:
            if y.is_balanced(force_check=False):
                x = factories.array(x, split=y.split, copy=False, comm=x.comm, device=x.device)
            else:
                target_map = y.lshape_map
                idx = [slice(None)] * x.ndim
                idx[y.split] = slice(
                    target_map[: x.comm.rank, y.split].sum(),
                    target_map[: x.comm.rank + 1, y.split].sum(),
                )
                x = factories.array(
                    x.larray[tuple(idx)], is_split=y.split, copy=False, comm=x.comm, device=x.device
                )
        elif x.split is not None and y.split is None:
            if x.is_balanced(force_check=False):
                y = factories.array(y, split=x.split, copy=False, comm=y.comm, device=y.device)
            else:
                target_map = x.lshape_map
                idx = [slice(None)] * y.ndim
                idx[x.split] = slice(
                    target_map[: y.comm.rank, x.split].sum(),
                    target_map[: y.comm.rank + 1, x.split].sum(),
                )
                y = factories.array(
                    y.larray[tuple(idx)], is_split=x.split, copy=False, comm=y.comm, device=y.device
                )
        elif x.split != y.split:
            raise ValueError(
                "DNDarrays must have the same split axes, found {x.split} and {y.split}"
            )
        elif not (x.is_balanced(force_check=False) and y.is_balanced(force_check=False)):
            x_lmap = x.lshape_map
            y_lmap = y.lshape_map
            if not torch.equal(x_lmap, y_lmap):
                x = x.balance()
                y = y.balance()

    result_type = types.result_type(x, y)
    x = x.astype(result_type)
    y = y.astype(result_type)

    if x.larray.numel() > 0:
        result_value = torch.equal(x.larray, y.larray)
    else:
        result_value = True

    return x.comm.allreduce(result_value, MPI.LAND)


def ge(x: Union[DNDarray, float, int], y: Union[DNDarray, float, int]) -> DNDarray:
    """
    Returns a D:class:`~heat.core.dndarray.DNDarray` containing the results of element-wise rich greater than or equal comparison between values from operand ``x`` with respect to values of
    operand ``y`` (i.e. ``x>=y``), not commutative. Takes the first and second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) whose elements are to be compared as argument.

    Parameters
    ----------
    x: DNDarray or scalar
        The first operand to be compared greater than or equal to second operand
    y: DNDarray or scalar
       The second operand to be compared less than or equal to first operand

    Examples
    -------
    >>> import heat as ht
    >>> x = ht.float32([[1, 2],[3, 4]])
    >>> ht.ge(x, 3.0)
    DNDarray([[False, False],
              [ True,  True]], dtype=ht.bool, device=cpu:0, split=None)
    >>> y = ht.float32([[2, 2], [2, 2]])
    >>> ht.ge(x, y)
    DNDarray([[False,  True],
              [ True,  True]], dtype=ht.bool, device=cpu:0, split=None)
    """
    res = _operations.__binary_op(torch.ge, x, y)

    if res.dtype != types.bool:
        res = dndarray.DNDarray(
            res.larray.type(torch.bool),
            res.gshape,
            types.bool,
            res.split,
            res.device,
            res.comm,
            res.balanced,
        )

    return res


DNDarray.__ge__ = lambda self, other: ge(self, other)
DNDarray.__ge__.__doc__ = ge.__doc__

# alias
greater_equal = ge
greater_equal.__doc__ = ge.__doc__


def gt(x: Union[DNDarray, float, int], y: Union[DNDarray, float, int]) -> DNDarray:
    """
    Returns a :class:`~heat.core.dndarray.DNDarray` containing the results of element-wise rich greater than comparison between values from operand ``x`` with respect to values of
    operand ``y`` (i.e. ``x>y``), not commutative. Takes the first and second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) whose elements are to be compared as argument.

    Parameters
    ----------
    x: DNDarray or scalar
       The first operand to be compared greater than second operand
    y: DNDarray or scalar
       The second operand to be compared less than first operand

    Examples
    -------
    >>> import heat as ht
    >>> x = ht.float32([[1, 2],[3, 4]])
    >>> ht.gt(x, 3.0)
    DNDarray([[False, False],
              [False,  True]], dtype=ht.bool, device=cpu:0, split=None)
    >>> y = ht.float32([[2, 2], [2, 2]])
    >>> ht.gt(x, y)
    DNDarray([[False, False],
              [ True,  True]], dtype=ht.bool, device=cpu:0, split=None)
    """
    res = _operations.__binary_op(torch.gt, x, y)

    if res.dtype != types.bool:
        res = dndarray.DNDarray(
            res.larray.type(torch.bool),
            res.gshape,
            types.bool,
            res.split,
            res.device,
            res.comm,
            res.balanced,
        )

    return res


DNDarray.__gt__ = lambda self, other: gt(self, other)
DNDarray.__gt__.__doc__ = gt.__doc__

# alias
greater = gt
greater.__doc__ = gt.__doc__


def le(x: Union[DNDarray, float, int], y: Union[DNDarray, float, int]) -> DNDarray:
    """
    Return a :class:`~heat.core.dndarray.DNDarray` containing the results of element-wise rich less than or equal comparison between values from operand ``x`` with respect to values of
    operand ``y`` (i.e. ``x<=y``), not commutative. Takes the first and second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) whose elements are to be compared as argument.

    Parameters
    ----------
    x: DNDarray or scalar
       The first operand to be compared less than or equal to second operand
    y: DNDarray or scalar
       The second operand to be compared greater than or equal to first operand

    Examples
    -------
    >>> import heat as ht
    >>> x = ht.float32([[1, 2],[3, 4]])
    >>> ht.le(x, 3.0)
    DNDarray([[ True,  True],
              [ True, False]], dtype=ht.bool, device=cpu:0, split=None)
    >>> y = ht.float32([[2, 2], [2, 2]])
    >>> ht.le(x, y)
    DNDarray([[ True,  True],
              [False, False]], dtype=ht.bool, device=cpu:0, split=None)
    """
    res = _operations.__binary_op(torch.le, x, y)

    if res.dtype != types.bool:
        res = dndarray.DNDarray(
            res.larray.type(torch.bool),
            res.gshape,
            types.bool,
            res.split,
            res.device,
            res.comm,
            res.balanced,
        )

    return res


DNDarray.__le__ = lambda self, other: le(self, other)
DNDarray.__le__.__doc__ = le.__doc__

# alias
less_equal = le
less_equal.__doc__ = le.__doc__


def lt(x: Union[DNDarray, float, int], y: Union[DNDarray, float, int]) -> DNDarray:
    """
    Returns a :class:`~heat.core.dndarray.DNDarray` containing the results of element-wise rich less than comparison between values from operand ``x`` with respect to values of
    operand ``y`` (i.e. ``x<y``), not commutative. Takes the first and second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) whose elements are to be compared as argument.

    Parameters
    ----------
    x: DNDarray or scalar
        The first operand to be compared less than second operand
    y: DNDarray or scalar
        The second operand to be compared greater than first operand

    Examples
    -------
    >>> import heat as ht
    >>> x = ht.float32([[1, 2],[3, 4]])
    >>> ht.lt(x, 3.0)
    DNDarray([[ True,  True],
              [False, False]], dtype=ht.bool, device=cpu:0, split=None)
    >>> y = ht.float32([[2, 2], [2, 2]])
    >>> ht.lt(x, y)
    DNDarray([[ True, False],
              [False, False]], dtype=ht.bool, device=cpu:0, split=None)
    """
    res = _operations.__binary_op(torch.lt, x, y)

    if res.dtype != types.bool:
        res = dndarray.DNDarray(
            res.larray.type(torch.bool),
            res.gshape,
            types.bool,
            res.split,
            res.device,
            res.comm,
            res.balanced,
        )

    return res


DNDarray.__lt__ = lambda self, other: lt(self, other)
DNDarray.__lt__.__doc__ = lt.__doc__

# alias
less = lt
less.__doc__ = lt.__doc__


def ne(x: Union[DNDarray, float, int], y: Union[DNDarray, float, int]) -> DNDarray:
    """
    Returns a :class:`~heat.core.dndarray.DNDarray` containing the results of element-wise rich comparison of non-equality between values from two operands, commutative.
    Takes the first and second operand (scalar or :class:`~heat.core.dndarray.DNDarray`) whose elements are to be
    compared as argument.

    Parameters
    ----------
    x: DNDarray or scalar
        The first operand involved in the comparison
    y: DNDarray or scalar
        The second operand involved in the comparison

    Examples
    ---------
    >>> import heat as ht
    >>> x = ht.float32([[1, 2],[3, 4]])
    >>> ht.ne(x, 3.0)
    DNDarray([[ True,  True],
              [False,  True]], dtype=ht.bool, device=cpu:0, split=None)
    >>> y = ht.float32([[2, 2], [2, 2]])
    >>> ht.ne(x, y)
    DNDarray([[ True, False],
              [ True,  True]], dtype=ht.bool, device=cpu:0, split=None)
    """
    res = _operations.__binary_op(torch.ne, x, y)

    if res.dtype != types.bool:
        res = dndarray.DNDarray(
            res.larray.type(torch.bool),
            res.gshape,
            types.bool,
            res.split,
            res.device,
            res.comm,
            res.balanced,
        )

    return res


DNDarray.__ne__ = lambda self, other: ne(self, other)
DNDarray.__ne__.__doc__ = ne.__doc__

# alias
not_equal = ne
not_equal.__doc__ = ne.__doc__
