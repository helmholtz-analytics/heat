from __future__ import annotations

from ._dtypes import (
    _all_dtypes,
    _boolean_dtypes,
    _integer_dtypes,
    _integer_or_boolean_dtypes,
    _floating_dtypes,
    _numeric_dtypes,
    _result_type,
    _dtype_categories,
)
from ._typing import Dtype

import heat as ht
from heat import array_api
from heat.core.devices import Device
from heat.core.dndarray import DNDarray
from typing import Any, Tuple, Union, Optional


class Array:
    """
    DNDarray object for the array API namespace.
    This is a wrapper around `heat.DNDarray` that restricts the usage to only
    those things that are required by the array API namespace. Note,
    attributes on this object that start with a single underscore are not part
    of the API specification and should only be used internally. This object
    should not be constructed directly. Rather, use one of the creation
    functions, such as asarray().
    """

    _array: DNDarray

    @classmethod
    def _new(cls, x, /):
        """
        This is a private method for initializing the array API Array
        object.
        Functions outside of the array_api submodule should not use this
        method. Use one of the creation functions instead, such as
        `asarray`.
        """
        obj = super().__new__(cls)
        obj._array = x
        return obj

    def __new__(cls, *args, **kwargs):
        """
        Prevent `Array()` from working.
        """
        raise TypeError(
            "The array_api Array object should not be instantiated directly. Use an array creation function, such as asarray(), instead."
        )

    def _check_allowed_dtypes(
        self, other: Union[bool, int, float, Array], dtype_category: str, op: str
    ) -> Array:
        """
        Helper function for operators to only allow specific input dtypes
        Use like
            other = self._check_allowed_dtypes(other, 'numeric', '__add__')
            if other is NotImplemented:
                return other
        """
        if self.dtype not in _dtype_categories[dtype_category]:
            raise TypeError(f"Only {dtype_category} dtypes are allowed in {op}")
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        elif isinstance(other, Array):
            if other.dtype not in _dtype_categories[dtype_category]:
                raise TypeError(f"Only {dtype_category} dtypes are allowed in {op}")
        else:
            return NotImplemented
        return other

    def _promote_scalar(self, scalar):
        """
        Returns a promoted version of a Python scalar appropriate for use with
        operations on self.
        This may raise a `TypeError` when the scalar type is incompatible with
        the dtype of self.
        """
        # Note: Only Python scalar types that match the array dtype are
        # allowed.
        if isinstance(scalar, bool):
            if self.dtype not in _boolean_dtypes:
                raise TypeError("Python bool scalars can only be promoted with bool arrays")
        elif isinstance(scalar, int):
            if self.dtype in _boolean_dtypes:
                raise TypeError("Python int scalars cannot be promoted with bool arrays")
        elif isinstance(scalar, float):
            if self.dtype not in _floating_dtypes:
                raise TypeError(
                    "Python float scalars can only be promoted with floating-point arrays."
                )
        else:
            raise TypeError("'scalar' must be a Python scalar")

        # Note: scalars are unconditionally cast to the same dtype as the
        # array.

        return Array._new(ht.array(scalar, self.dtype))

    def __abs__(self: Array, /) -> Array:
        """
        Calculates the absolute value for each element of an array instance
        (i.e., the element-wise result has the same magnitude as the respective
        element but has positive sign).
        """
        if self.dtype not in _numeric_dtypes:
            raise TypeError("Only numeric dtypes are allowed in __abs__")
        res = ht.abs(self._array)
        return self.__class__._new(res)

    def __add__(self: Array, other: Union[int, float, Array], /) -> Array:
        """
        Calculates the sum for each element of an array instance with the
        respective element of the array `other`.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__add__")
        if other is NotImplemented:
            return other
        res = ht.add(self._array, other._array)
        return self.__class__._new(res)

    def __and__(self: Array, other: Union[int, bool, Array], /) -> Array:
        """
        Evaluates `self_i & other_i` for each element of an array instance
        with the respective element of the array `other`.
        """
        other = self._check_allowed_dtypes(other, "integer or boolean", "__and__")
        if other is NotImplemented:
            return other
        res = ht.bitwise_and(self._array, other._array)
        return self.__class__._new(res)

    def __array_namespace__(self: Array, /, *, api_version: Optional[str] = None) -> Any:
        if api_version is not None and api_version != "2021.12":
            raise ValueError(f"Unrecognized array API version: {api_version}")
        return array_api

    @property
    def dtype(self) -> Dtype:
        """
        Array API compatible wrapper for `heat.DNDarray.dtype`.
        """
        return self._array.dtype

    @property
    def device(self) -> Device:
        """
        Array API compatible wrapper for `heat.DNDarray.device`.
        """
        return self._array.device

    @property
    def ndim(self) -> int:
        """
        Array API compatible wrapper for `heat.DNDarray.ndim`.
        See its docstring for more information.
        """
        return self._array.ndim

    @property
    def shape(self) -> Tuple[Optional[int], ...]:
        """
        Array API compatible wrapper for `heat.DNDarray.shape`.
        """
        return self._array.shape

    @property
    def size(self) -> Optional[int]:
        """
        Array API compatible wrapper for `heat.DNDarray.size`.
        """
        return self._array.size
