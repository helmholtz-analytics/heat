from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

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
    from ._typing import Device, Dtype
    from builtins import ellipsis

import heat as ht
from heat import array_api


class Array:
    """
    DNDarray object for the array API namespace.
    This is a wrapper around ``heat.DNDarray`` that restricts the usage to only
    those things that are required by the array API namespace. Note,
    attributes on this object that start with a single underscore are not part
    of the API specification and should only be used internally. This object
    should not be constructed directly. Rather, use one of the creation
    functions, such as ``asarray``.
    """

    _array: ht.DNDarray

    @classmethod
    def _new(cls, x, /):
        """
        This is a private method for initializing the array API Array
        object.
        Functions outside of the array_api submodule should not use this
        method. Use one of the creation functions instead, such as
        ``asarray``.

        Parameters
        ----------
        x : DNDarray
            Underlying ``DNDarray``
        """
        obj = super().__new__(cls)
        obj._array = x
        return obj

    def __new__(cls, *args, **kwargs):
        """
        Prevent ``Array()`` from working.
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
        This may raise a ``TypeError`` when the scalar type is incompatible with
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
            raise TypeError(f"{scalar} must be a Python scalar")

        # Note: scalars are unconditionally cast to the same dtype as the
        # array.

        return Array._new(ht.array(scalar, self.dtype))

    @staticmethod
    def _normalize_two_args(x1, x2) -> Tuple[Array, Array]:
        """
        Normalize inputs to two arg functions to fix type promotion rules
        Heat deviates from the spec type promotion rules in cases where one
        argument is 0-dimensional and the other is not. For example:
        >>> import heat as ht
        >>> a = ht.array([1.0], dtype=ht.float32)
        >>> b = ht.array(1.0, dtype=ht.float64)
        >>> ht.add(a, b) # The spec says this should be float64
        DNDarray([2.], dtype=ht.float32, device=cpu:0, split=None)
        To fix this, we add a dimension to the 0-dimension array before passing it
        through. This works because a dimension would be added anyway from
        broadcasting, so the resulting shape is the same, but this prevents Heat
        from not promoting the dtype.
        """
        if x1.ndim == 0 and x2.ndim != 0:
            # The _array[None] workaround was chosen because it is relatively
            # performant. broadcast_to(x1._array, x2.shape) is much slower. We
            # could also manually type promote x2, but that is more complicated
            # and about the same performance as this.
            x1 = Array._new(x1._array[None])
        elif x2.ndim == 0 and x1.ndim != 0:
            x2 = Array._new(x2._array[None])
        return (x1, x2)

    def __abs__(self: Array, /) -> Array:
        """
        Calculates the absolute value for each element of an array instance
        (i.e., the element-wise result has the same magnitude as the respective
        element but has positive sign).
        """
        if self.dtype not in _numeric_dtypes:
            raise TypeError("Only numeric dtypes are allowed in __abs__")
        res = self._array.abs(dtype=self.dtype)
        return self.__class__._new(res)

    def __add__(self: Array, other: Union[int, float, Array], /) -> Array:
        """
        Calculates the sum for each element of an array instance with the
        respective element of the array ``other``.

        Parameters
        ----------
        other : Union[int, float, Array]
            Addend array. Must have a numeric data type.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__add__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__add__(other._array)
        return self.__class__._new(res)

    def __and__(self: Array, other: Union[int, bool, Array], /) -> Array:
        """
        Evaluates ``self_i & other_i`` for each element of an array instance
        with the respective element of the array ``other``.

        Parameters
        ----------
        other : Union[int, bool, Array]
            Other array. Must have an integer or boolean data type.
        """
        other = self._check_allowed_dtypes(other, "integer or boolean", "__and__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__and__(other._array)
        return self.__class__._new(res)

    def __array_namespace__(self: Array, /, *, api_version: Optional[str] = None) -> Any:
        """
        Returns an object that has all the array API functions on it.

        Parameters
        ----------
        api_version : Optional[str]
            string representing the version of the array API specification to
            be returned, in ``'YYYY.MM'`` form. If it is ``None`` (default), it
            returns the namespace corresponding to latest version of the
            array API specification.
        """
        if api_version is not None and api_version != "2021.12":
            raise ValueError(f"Unrecognized array API version: {api_version}")
        return array_api

    def __bool__(self: Array, /) -> bool:
        """
        Converts a zero-dimensional boolean array to a Python ``bool`` object.
        """
        if self._array.ndim != 0:
            raise TypeError("bool is only allowed on arrays with 0 dimensions")
        if self.dtype not in _boolean_dtypes:
            raise ValueError("bool is only allowed on boolean arrays")
        res = self._array.__bool__()
        return res

    def __eq__(self: Array, other: Union[int, float, bool, Array], /) -> Array:
        """
        Computes the truth value of ``self_i == other_i`` for each element of an
        array instance with the respective element of the array ``other``.

        Parameters
        ----------
        other : Union[int, float, bool, Array]
            Other array.
        """
        # Even though "all" dtypes are allowed, we still require them to be
        # promotable with each other.
        other = self._check_allowed_dtypes(other, "all", "__eq__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__eq__(other._array)
        return self.__class__._new(res)

    def __float__(self: Array, /) -> float:
        """
        Converts a zero-dimensional floating-point array to a Python ``float`` object.
        """
        if self._array.ndim != 0:
            raise TypeError("float is only allowed on arrays with 0 dimensions")
        if self.dtype not in _floating_dtypes:
            raise ValueError("float is only allowed on floating-point arrays")
        res = self._array.__float__()
        return res

    def __floordiv__(self: Array, other: Union[int, float, Array], /) -> Array:
        """
        Evaluates ``self_i // other_i`` for each element of an array instance
        with the respective element of the array ``other``.

        Parameters
        ----------
        other : Union[int, float, Array]
            Other array. Must have a numeric data type.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__floordiv__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__floordiv__(other._array)
        return self.__class__._new(res)

    def __ge__(self: Array, other: Union[int, float, Array], /) -> Array:
        """
        Computes the truth value of ``self_i >= other_i`` for each element of
        an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        other : Union[int, float, Array]
            Other array. Must have a numeric data type.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__ge__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__ge__(other._array)
        return self.__class__._new(res)

    def __getitem__(
        self: Array,
        key: Union[int, slice, ellipsis, Tuple[Union[int, slice, ellipsis], ...], Array],
        /,
    ) -> Array:
        """
        Returns ``self[key]``.

        Parameters
        ----------
        key : Union[int, slice, ellipsis, Tuple[Union[int, slice, ellipsis], ...], array]
            Index key
        """
        # Note: Only indices required by the spec are allowed. See the
        # docstring of _validate_index
        # self._validate_index(key)
        if isinstance(key, Array):
            # Indexing self._array with array_api arrays can be erroneous
            key = key._array
        # if 0 in self.shape:
        #     new_array = self._array
        #     if isinstance(key, tuple):
        #         for i, index in enumerate((x for x in key if x != ...)):
        #             if index is None:
        #                 new_array = new_array.expand_dims(i)
        #     return self._new(new_array)
        res = self._array.__getitem__(key)
        return self._new(res)

    def __gt__(self: Array, other: Union[int, float, Array], /) -> Array:
        """
        Computes the truth value of ``self_i > other_i`` for each element
        of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        other : Union[int, float, Array]
            Other array. Must have a numeric data type.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__gt__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__gt__(other._array)
        return self.__class__._new(res)

    def __index__(self: Array, /) -> int:
        """
        Converts a zero-dimensional integer array to a Python ``int`` object.
        """
        return self.__int__()

    def __int__(self: Array, /) -> int:
        """
        Converts a zero-dimensional integer array to a Python ``int`` object.
        """
        if self._array.ndim != 0:
            raise TypeError("int is only allowed on arrays with 0 dimensions")
        if self.dtype not in _integer_dtypes:
            raise ValueError("int is only allowed on integer arrays")
        res = self._array.__int__()
        return res

    def __invert__(self: Array, /) -> Array:
        """
        Evaluates ``~self_i`` for each element of an array instance.
        """
        if self.dtype not in _integer_or_boolean_dtypes:
            raise TypeError("Only integer or boolean dtypes are allowed in __invert__")
        res = self._array.__invert__()
        return self.__class__._new(res)

    def __le__(self: Array, other: Union[int, float, Array], /) -> Array:
        """
        Computes the truth value of ``self_i <= other_i`` for each element of an
        array instance with the respective element of the array ``other``.

        Parameters
        ----------
        other : Union[int, float, Array]
            Other array. Must have a numeric data type.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__le__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__le__(other._array)
        return self.__class__._new(res)

    def __lshift__(self: Array, other: Union[int, Array], /) -> Array:
        """
        Evaluates ``self_i << other_i`` for each element of an array instance
        with the respective element of the array ``other``.

        Parameters
        ----------
        other : Union[int, Array]
            Other array. Must have an integer data type. Each element
            must be greater than or equal to ``0``.
        """
        other = self._check_allowed_dtypes(other, "integer", "__lshift__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__lshift__(other._array)
        return self.__class__._new(res)

    def __lt__(self: Array, other: Union[int, float, Array], /) -> Array:
        """
        Computes the truth value of ``self_i < other_i`` for each element
        of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        other : Union[int, Array]
            Other array. Must have a numeric data type.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__lt__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__lt__(other._array)
        return self.__class__._new(res)

    def __matmul__(self: Array, other: Array, /) -> Array:
        """
        Computes the matrix product.

        Parameters
        ----------
        other : Array
            Other array. Must have a numeric data type and at least one dimension.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__matmul__")
        if other is NotImplemented:
            return other
        res = self._array.__matmul__(other._array)
        return self.__class__._new(res)

    def __mod__(self: Array, other: Union[int, float, Array], /) -> Array:
        """
        Evaluates ``self_i % other_i`` for each element of an array instance
        with the respective element of the array ``other``.

        Parameters
        ----------
        other : Union[int, float, array]
            Other array. Must have a numeric data type.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__mod__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__mod__(other._array)
        return self.__class__._new(res)

    def __mul__(self: Array, other: Union[int, float, Array], /) -> Array:
        """
        Calculates the product for each element of an array instance with
        the respective element of the array ``other``.

        Parameters
        ----------
        other : Union[int, float, array]
            Other array. Must have a numeric data type.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__mul__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__mul__(other._array)
        return self.__class__._new(res)

    def __ne__(self: Array, other: Union[int, float, bool, Array], /) -> Array:
        """
        Computes the truth value of ``self_i != other_i`` for each element of
        an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        other : Union[int, float, bool, array]
            Other array.
        """
        other = self._check_allowed_dtypes(other, "all", "__ne__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__ne__(other._array)
        return self.__class__._new(res)

    def __neg__(self: Array, /) -> Array:
        """
        Evaluates ``-self_i`` for each element of an array instance.
        """
        if self.dtype not in _numeric_dtypes:
            raise TypeError("Only numeric dtypes are allowed in __neg__")
        res = self._array.__neg__()
        return self.__class__._new(res)

    def __or__(self: Array, other: Union[int, bool, Array], /) -> Array:
        """
        Evaluates ``self_i | other_i`` for each element of an array instance
        with the respective element of the array ``other``.

        Parameters
        ----------
        other : Union[int, bool, array]
            Other array. Must have an integer or boolean data type.
        """
        other = self._check_allowed_dtypes(other, "integer or boolean", "__or__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__or__(other._array)
        return self.__class__._new(res)

    def __pos__(self: Array, /) -> Array:
        """
        Evaluates ``+self_i`` for each element of an array instance.
        """
        if self.dtype not in _numeric_dtypes:
            raise TypeError("Only numeric dtypes are allowed in __pos__")
        res = self._array.__pos__()
        return self.__class__._new(res)

    def __pow__(self: Array, other: Union[int, float, Array], /) -> Array:
        """
        Calculates an approximation of exponentiation by raising each element
        (the base) of an array instance to the power of ``other_i`` (the exponent),
        where ``other_i`` is the corresponding element of the array ``other``.

        Parameters
        ----------
        other : Union[int, float, array]
            Other array. Must have a numeric data type.
        """
        # from ._elementwise_functions import pow

        other = self._check_allowed_dtypes(other, "numeric", "__pow__")
        if other is NotImplemented:
            return other
        # Note: NumPy's __pow__ does not follow type promotion rules for 0-d
        # arrays, so we use pow() here instead.
        # return pow(self, other)
        res = self._array.__pow__()
        return self.__class__._new(res)

    def __rshift__(self: Array, other: Union[int, Array], /) -> Array:
        """
        Evaluates ``self_i >> other_i`` for each element of an array instance
        with the respective element of the array ``other``.

        Parameters
        ----------
        other : Union[int, array]
            Other array. Must have an integer data type. Each element must be
            greater than or equal to ``0``.
        """
        other = self._check_allowed_dtypes(other, "integer", "__rshift__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rshift__(other._array)
        return self.__class__._new(res)

    def __setitem__(
        self,
        key: Union[int, slice, ellipsis, Tuple[Union[int, slice, ellipsis], ...], Array],
        value: Union[int, float, bool, Array],
        /,
    ) -> None:
        """
        Sets ``self[key]`` to ``value``.
        """
        # Note: Only indices required by the spec are allowed. See the
        # docstring of _validate_index
        # self._validate_index(key)
        if isinstance(key, Array):
            # Indexing self._array with array_api arrays can be erroneous
            key = key._array
        if isinstance(value, Array):
            value = value._array
        if self.ndim == 0 and isinstance(key, type(...)):
            self._array = ht.asarray(value)
        else:
            self._array.__setitem__(key, value)

    def __sub__(self: Array, other: Union[int, float, Array], /) -> Array:
        """
        Calculates the difference for each element of an array instance
        with the respective element of the array ``other``.

        Parameters
        ----------
        other : Union[int, float, array]
            Subtrahend array. Must have a numeric data type.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__sub__")
        if other is NotImplemented:
            return other
        # self, other = self._normalize_two_args(self, other)
        res = self._array.__sub__(other._array)
        return self.__class__._new(res)

    def __truediv__(self: Array, other: Union[int, float, Array], /) -> Array:
        """
        Evaluates ``self_i / other_i`` for each element of an array instance
        with the respective element of the array ``other``.

        Parameters
        ----------
        other : Union[int, float, array]
            Subtrahend array. Must have a numeric data type.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__truediv__")
        if other is NotImplemented:
            return other
        # self, other = self._normalize_two_args(self, other)
        res = self._array.__truediv__(other._array)
        return self.__class__._new(res)

    def __xor__(self: Array, other: Union[int, bool, Array], /) -> Array:
        """
        Evaluates ``self_i ^ other_i`` for each element of an array instance
        with the respective element of the array ``other``.

        Parameters
        ----------
        other : Union[int, bool, array]
            Subtrahend array. Must have an integer or boolean data type.
        """
        other = self._check_allowed_dtypes(other, "integer or boolean", "__xor__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__xor__(other._array)
        return self.__class__._new(res)

    # def to_device(self: Array, device: Device, /, stream: Optional[Union[int, Any]] = None) -> Array:
    #     """
    #     Copy the array from the device on which it currently resides to the specified ``device``.
    #     """
    #     if stream is not None:
    #         raise ValueError("The stream argument to to_device() is not supported")
    #     if device == 'cpu':
    #         return self
    #     raise ValueError(f"Unsupported device {device!r}")

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
    def mT(self) -> Array:
        """
        Transpose of a matrix (or a stack of matrices).
        """
        if self.ndim < 2:
            raise ValueError("x.mT requires x to have at least 2 dimensions.")
        return self._array.transpose(axes=list(range(self.ndim))[:-2] + [-1, -2])

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

    @property
    def T(self) -> Array:
        """
        Transpose of the array.
        """
        # Note: T only works on 2-dimensional arrays, as outlined in the specification:
        if self.ndim != 2:
            raise ValueError(
                "x.T requires x to have 2 dimensions. Use x.mT to transpose stacks of matrices and permute_dims() to permute dimensions."
            )
        return self.__class__._new(self._array.T)
