from __future__ import annotations

import heat as ht

from typing import TYPE_CHECKING, Any, Tuple, Union, Optional

from ._creation_functions import asarray
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

if TYPE_CHECKING:
    from ._typing import Dtype, Device
    from builtins import ellipsis

from heat import array_api
from heat.core.dndarray import DNDarray


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

    def __str__(self: Array, /) -> str:
        """
        Computes a printable representation of the Array.
        """
        return self._array.__str__().replace("DNDarray", "Array")

    def __repr__(self: Array, /) -> str:
        """
        Computes a printable representation of the Array.
        """
        return self._array.__str__().replace("DNDarray", "Array")

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
        _result_type(self.dtype, other.dtype)
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
        """
        Returns an object that has all the array API functions on it.
        """
        if api_version is not None and api_version != "2021.12":
            raise ValueError(f"Unrecognized array API version: {api_version}")
        return array_api

    def __bool__(self: Array, /) -> bool:
        """
        Converts a zero-dimensional boolean array to a Python `bool` object.
        """
        if self._array.ndim != 0:
            raise TypeError("bool is only allowed on arrays with 0 dimensions")
        if self.dtype not in _boolean_dtypes:
            raise ValueError("bool is only allowed on boolean arrays")
        res = self._array.__bool__()
        return res

    def __float__(self: Array, /) -> float:
        """
        Converts a zero-dimensional floating-point array to a Python `float` object.
        """
        if self._array.ndim != 0:
            raise TypeError("float is only allowed on arrays with 0 dimensions")
        if self.dtype not in _floating_dtypes:
            raise ValueError("float is only allowed on floating-point arrays")
        res = self._array.__float__()
        return res

    def __getitem__(
        self: Array,
        key: Union[int, slice, ellipsis, Tuple[Union[int, slice, ellipsis], ...], Array],
        /,
    ) -> Array:
        """
        Returns `self[key]`.
        """
        # Note: Only indices required by the spec are allowed. See the
        # docstring of _validate_index
        # self._validate_index(key)
        if isinstance(key, Array):
            # Indexing self._array with array_api arrays can be erroneous
            key = key._array
        res = self._array.__getitem__(key)
        return self._new(res)

    # def __index__(self: Array, /) -> int:
    #     """
    #     Performs the operation __index__.
    #     """
    #     res = self.__int__()
    #     return res

    def __int__(self: Array, /) -> int:
        """
        Converts a zero-dimensional integer array to a Python `int` object.
        """
        if self._array.ndim != 0:
            raise TypeError("int is only allowed on arrays with 0 dimensions")
        if self.dtype not in _integer_dtypes:
            raise ValueError("int is only allowed on integer arrays")
        res = self._array.__int__()
        return res

    def __setitem__(
        self,
        key: Union[int, slice, ellipsis, Tuple[Union[int, slice, ellipsis], ...], Array],
        value: Union[int, float, bool, Array],
        /,
    ) -> None:
        """
        Sets `self[key]` to `value`.
        """
        # Note: Only indices required by the spec are allowed. See the
        # docstring of _validate_index
        # self._validate_index(key)
        if isinstance(key, Array):
            # Indexing self._array with array_api arrays can be erroneous
            key = key._array
        self._array.__setitem__(key, asarray(value)._array)

    @property
    def dtype(self) -> Dtype:
        """
        Data type of the array elements.
        """
        return self._array.dtype

    @property
    def device(self) -> Device:
        """
        Hardware device the array data resides on.
        """
        return self._array.device

    @property
    def ndim(self) -> int:
        """
        Number of array dimensions (axes).
        """
        return self._array.ndim

    @property
    def shape(self) -> Tuple[Optional[int], ...]:
        """
        Array dimensions.
        """
        return self._array.shape

    @property
    def size(self) -> Optional[int]:
        """
        Number of elements in an array.
        """
        return self._array.size
