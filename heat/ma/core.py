"""MaskedDNDarray support"""

from typing import Iterable, Optional, Type, Union, Tuple, Callable
from functools import update_wrapper

import torch
from torch.masked import masked_tensor

import heat as ht
from heat import DNDarray

__all__ = [
    "MaskedDNDarray",
]


def _binary_op(operation, t1, t2, out, where):
    if isinstance(t1, MaskedDNDarray):
        data1 = t1._data
        mask1 = t1._mask
    else:
        data1 = t1
        mask1 = False

    if isinstance(t2, MaskedDNDarray):
        data2 = t2._data
        mask2 = t2._mask
    else:
        data2 = t2
        mask2 = False

    data = operation(data1, data2, out=out, where=where)
    mask = ht.bitwise_or(mask1, mask2)

    return MaskedDNDarray(data, mask)


def _masked_tensor_str(data, mask, formatter):
    if data.layout in {torch.sparse_coo, torch.sparse_csr}:
        data = data.to_dense()
        mask = mask.to_dense()
    if data.dim() == 1:
        formatted_elements = [
            formatter.format(d.item()) if isinstance(d.item(), float) else str(d.item())
            for d in data
        ]
        max_len = max(8 if x[1] else len(x[0]) for x in zip(formatted_elements, ~mask))
        return (
            "["
            + ", ".join(
                ["--".rjust(max_len) if m else e for (e, m) in zip(formatted_elements, ~mask)]
            )
            + "]"
        )
    sub_strings = [_masked_tensor_str(d, m, formatter) for (d, m) in zip(data, mask)]
    sub_strings = ["\n".join(["  " + si for si in s.split("\n")]) for s in sub_strings]
    return "[\n" + ",\n".join(sub_strings) + "\n]"


HANDLED_FUNCTIONS = {}


class MaskedDNDarray:
    """Handles missing or invalid values"""

    def __init__(self, data, mask, copy=False, fill_value=None, keep_mask=True, hard_mask=None, shrink=True):
        # Checks
        ht.sanitize_in(data)
        ht.sanitize_in(mask)

        if data.gshape != mask.gshape or data.split != mask.split or data.lshape != mask.lshape:
            raise ValueError("data and mask do not have the same layout")

        self._data, self._mask = data, mask
        self._fill_value = fill_value
        self._hardmask = hard_mask

    @property
    def data(self):
        return self._data
    
    @property
    def mask(self):
        return self._mask

    @property
    def fill_value(self):
        return self._fill_value
    
    @property.setter
    def fill_value(self, value):
        self._fill_value = value

    @property
    def hardmask(self):
        return self._hardmask

    @classmethod
    def __heat_function__(cls, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, cls) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __repr__(self) -> str:
        formatter = "{0:8.4f}"
        if self._data.ndim == 0:
            scalar_data = self._data.item()
            data_formatted = (
                formatter.format(scalar_data)
                if isinstance(scalar_data, float)
                else str(scalar_data)
            )
            if not self._mask.item():
                data_formatted = "--"
            return (
                ("MaskedDNDarrray(" + data_formatted + ", " + str(self._mask.item()) + ")")
                if self._data.comm.rank == 0
                else ""
            )

        if not self._data.is_balanced():
            self._data.balance_()
            self._mask.balance_()
        # data is not split, we can use it as is
        if self._data.split is None or self._data.comm.size == 1:
            data = self._data.larray
            mask = self._mask.larray
        # split, we collect it
        else:
            data = self._data.copy().resplit_(None).larray
            mask = self._mask.copy().resplit_(None).larray

        s = _masked_tensor_str(data, mask, formatter)
        s = "\n".join("  " + si for si in s.split("\n"))
        return "MaskedDNDarray(\n" + s + "\n)" if self._data.comm.rank == 0 else ""

    def __add__(self, other):
        try:
            return add(self, other)
        except TypeError:
            return NotImplemented

    def __radd__(self, other):
        try:
            return add(other, self)
        except TypeError:
            return NotImplemented

    def __mul__(self, other):
        try:
            return mul(self, other)
        except TypeError:
            return NotImplemented

    def __rmul__(self, other):
        try:
            return mul(other, self)
        except TypeError:
            return NotImplemented

    def __sub__(self, other):
        try:
            return sub(self, other)
        except TypeError:
            return NotImplemented

    def __rsub__(self, other):
        try:
            return sub(other, self)
        except TypeError:
            return NotImplemented


#####################
# Function Override
#####################


def implements(heat_function):
    """Register override implementation"""

    def decorator(func):
        update_wrapper(func, heat_function)
        HANDLED_FUNCTIONS[heat_function] = func
        return func

    return decorator


# arithmetics.py
@implements(ht.add)
def add(
    t1: Union[DNDarray, float],
    t2: Union[DNDarray, float],
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    return _binary_op(ht.add, t1, t2, out, where)


@implements(ht.bitwise_and)
def bitwise_and(
    t1: Union[DNDarray, float],
    t2: Union[DNDarray, float],
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    return _binary_op(ht.bitwise_and, t1, t2, out, where)


@implements(ht.div)
def div(
    t1: Union[DNDarray, float],
    t2: Union[DNDarray, float],
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    return _binary_op(ht.div, t1, t2, out, where)


@implements(ht.mul)
def mul(
    t1: Union[DNDarray, float],
    t2: Union[DNDarray, float],
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    return _binary_op(ht.mul, t1, t2, out, where)


@implements(ht.sub)
def sub(
    t1: Union[DNDarray, float],
    t2: Union[DNDarray, float],
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    return _binary_op(ht.sub, t1, t2, out, where)


# manipulations.py


@implements(ht.balance)
def balance(array, copy=False):
    data = ht.balance(array._data, copy)
    mask = ht.balance(array._mask, copy)
    return MaskedDNDarray(data, mask)


@implements(ht.broadcast_arrays)
def broadcast_arrays(*arrays):
    data = (array._data for array in arrays)
    mask = (array._mask for array in arrays)
    data = ht.broadcast_arrays(*data)
    mask = ht.broadcast_arrays(*mask)
    return [MaskedDNDarray(d, m) for d, m in zip(data, mask)]


@implements(ht.broadcast_to)
def broadcast_to(x, shape):
    data = ht.broadcast_to(x._data, shape)
    mask = ht.broadcast_to(x._mask, shape)
    return MaskedDNDarray(data, mask)


@implements(ht.collect)
def collect(arr, target_rank=0):
    data = ht.collect(arr._data, target_rank)
    mask = ht.collect(arr._mask, target_rank)
    return MaskedDNDarray(data, mask)


@implements(ht.repeat)
def repeat(a, repeats, axis):
    data = ht.repeat(a._data, repeats, axis)
    mask = ht.repeat(a._mask, repeats, axis)
    return MaskedDNDarray(data, mask)
