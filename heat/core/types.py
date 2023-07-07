"""
implementations of the different dtypes supported in heat and the
"""

from __future__ import annotations

import builtins
import collections
import numpy as np
import torch

from . import communication
from . import devices
from . import factories
from . import _operations
from . import sanitation

from typing import Type, Union, Tuple, Any, Iterable, Optional

__all__ = [
    "datatype",
    "number",
    "integer",
    "signedinteger",
    "unsignedinteger",
    "bool",
    "bool_",
    "floating",
    "int8",
    "byte",
    "int16",
    "short",
    "int32",
    "int",
    "int64",
    "long",
    "uint8",
    "ubyte",
    "float32",
    "float",
    "float_",
    "float64",
    "double",
    "flexible",
    "can_cast",
    "canonical_heat_type",
    "heat_type_is_exact",
    "heat_type_is_inexact",
    "iscomplex",
    "isreal",
    "issubdtype",
    "heat_type_of",
    "promote_types",
    "result_type",
    "complex64",
    "cfloat",
    "csingle",
    "complex128",
    "cdouble",
]
__api__ = []


class datatype:
    """
    Defines the basic heat data types in the hierarchy as shown below. Design inspired by the Python package numpy.
    As part of the type-hierarchy: xx -- is bit-width \n
        - generic \n
            - bool, bool_ (kind=?) \n
            - number \n
                - integer \n
                    - signedinteger (intxx)(kind=b, i) \n
                        - int8, byte \n
                        - int16, short \n
                        - int32, int \n
                        - int64, long
                    - unsignedinteger (uintxx)(kind=B, u) \n
                        - uint8, ubyte
                - floating (floatxx) (kind=f) \n
                    - float32, float, float_ \n
                    - float64, double (double)
            - flexible (currently unused, placeholder for characters) \n
    """

    def __new__(
        cls,
        *value,
        device: Optional[Union[str, devices.Device]] = None,
        comm: Optional[communication.Communication] = None,
    ) -> dndarray.DNDarray:
        """
        Create a new DNDarray. See :func:`ht.array <heat.core.factories.array>` for more info on general
        DNDarray creation.

        Parameters
        ----------
        value: array_like
            The values for the DNDarray which will be created
        device: devices.Device
            The device on which to place the created DNDarray
        comm: communication.Communication
            The MPI communication object to use in distribution and further operations
        """
        torch_type = cls.torch_type()
        if torch_type is NotImplemented:
            raise TypeError(f"cannot create '{cls}' instances")

        value_count = len(value)

        # sanitize the distributed processing flags
        comm = communication.sanitize_comm(comm)
        device = devices.sanitize_device(device)

        # check whether there are too many arguments
        if value_count >= 2:
            raise TypeError(f"function takes at most 1 argument ({value_count} given)")
        elif value_count == 0:
            value = ((0,),)

        # otherwise, attempt to create a torch tensor of given type
        try:
            array = value[0]._DNDarray__array.type(torch_type)
            return dndarray.DNDarray(
                array,
                gshape=value[0].shape,
                dtype=cls,
                split=value[0].split,
                comm=comm,
                device=device,
                balanced=value[0].balanced,
            )
        except AttributeError:
            # this is the case of that the first/only element of value is not a DNDarray
            array = torch.tensor(*value, dtype=torch_type, device=device.torch_device)
        except TypeError as exception:
            # re-raise the exception to be consistent with numpy's exception interface
            raise ValueError(str(exception))

        return dndarray.DNDarray(
            array, tuple(array.shape), cls, split=None, device=device, comm=comm, balanced=True
        )

    @classmethod
    def torch_type(cls) -> NotImplemented:
        """
        Torch Datatype
        """
        return NotImplemented

    @classmethod
    def char(cls) -> NotImplemented:
        """
        Datatype short-hand name
        """
        return NotImplemented


class bool(datatype):
    """
    The boolean datatype in Heat
    """

    @classmethod
    def torch_type(cls) -> torch.dtype:
        """
        Torch Datatype
        """
        return torch.bool

    @classmethod
    def char(cls) -> str:
        """
        Datatype short-hand name
        """
        return "u1"


class number(datatype):
    """
    The general number datatype. Integer and Float classes will inherit from this.
    """

    pass


class integer(number):
    """
    The general integer datatype. Specific integer classes inherit from this.
    """

    pass


class signedinteger(integer):
    """
    The general signed integer datatype.
    """

    pass


class int8(signedinteger):
    """
    8 bit signed integer datatype
    """

    @classmethod
    def torch_type(cls) -> torch.dtype:
        """
        Torch Datatype
        """
        return torch.int8

    @classmethod
    def char(cls) -> str:
        """
        Datatype short-hand name
        """
        return "i1"


class int16(signedinteger):
    """
    16 bit signed integer datatype
    """

    @classmethod
    def torch_type(cls) -> torch.dtype:
        """
        Torch Datatype
        """
        return torch.int16

    @classmethod
    def char(cls) -> str:
        """
        Datatype short-hand name
        """
        return "i2"


class int32(signedinteger):
    """
    32 bit signed integer datatype
    """

    @classmethod
    def torch_type(cls) -> torch.dtype:
        """
        Torch Datatype
        """
        return torch.int32

    @classmethod
    def char(cls) -> str:
        """
        Datatype short-hand name
        """
        return "i4"


class int64(signedinteger):
    """
    64 bit signed integer datatype
    """

    @classmethod
    def torch_type(cls) -> torch.dtype:
        """
        Torch Datatype
        """
        return torch.int64

    @classmethod
    def char(cls) -> str:
        """
        Datatype short-hand name
        """
        return "i8"


class unsignedinteger(integer):
    """
    The general unsigned integer datatype
    """

    pass


class uint8(unsignedinteger):
    """
    8 bit unsigned integer datatype
    """

    @classmethod
    def torch_type(cls) -> torch.dtype:
        """
        Torch Datatype
        """
        return torch.uint8

    @classmethod
    def char(cls) -> str:
        """
        Datatype short-hand name
        """
        return "u1"


class floating(number):
    """
    The general floating point datatype class.
    """

    pass


class float32(floating):
    """
    The 32 bit floating point datatype
    """

    @classmethod
    def torch_type(cls) -> torch.dtype:
        """
        Torch Datatype
        """
        return torch.float32

    @classmethod
    def char(cls) -> str:
        """
        Datatype short-hand name
        """
        return "f4"


class float64(floating):
    """
    The 64 bit floating point datatype
    """

    @classmethod
    def torch_type(cls) -> torch.dtype:
        """
        Torch Datatye
        """
        return torch.float64

    @classmethod
    def char(cls) -> str:
        """
        Datatype short-hand name
        """
        return "f8"


class flexible(datatype):
    """
    The general flexible datatype. Currently unused, placeholder for characters
    """

    pass


class complex(number):
    """
    The general complex datatype class.
    """

    pass


class complex64(complex):
    """
    The complex 64 bit datatype. Both real and imaginary are 32 bit floating point
    """

    @classmethod
    def torch_type(cls):
        """
        Torch Datatype
        """
        return torch.complex64

    @classmethod
    def char(cls):
        """
        Datatype short-hand name
        """
        return "c8"


class complex128(complex):
    """
    The complex 128 bit datatype. Both real and imaginary are 64 bit floating point
    """

    @classmethod
    def torch_type(cls):
        """
        Torch Datatype
        """
        return torch.complex128

    @classmethod
    def char(cls):
        """
        Datatype short-hand name
        """
        return "c16"


# definition of aliases
bool_ = bool
ubyte = uint8
byte = int8
short = int16
int = int32
int_ = int32
long = int64
float = float32
float_ = float32
double = float64
cfloat = complex64
csingle = complex64
cdouble = complex128

_complexfloating = (complex64, complex128)

_inexact = (
    # float16,
    float32,
    float64,
    *_complexfloating,
)

_exact = (uint8, int8, int16, int32, int64)

# type mappings for type strings and builtins types
__type_mappings = {
    # type strings
    "?": bool,
    "B": uint8,
    "b": int8,
    "h": int16,
    "i": int32,
    "l": int64,
    "f": float32,
    "d": float64,
    "F": complex64,
    "D": complex128,
    "b1": bool,
    "u": uint8,
    "u1": uint8,
    "i1": int8,
    "i2": int16,
    "i4": int32,
    "i8": int64,
    "f4": float32,
    "f8": float64,
    "c8": complex64,
    "c16": complex128,
    # numpy types
    np.bool_: bool,
    np.uint8: uint8,
    np.int8: int8,
    np.int16: int16,
    np.int32: int32,
    np.int64: int64,
    np.float32: float32,
    np.float64: float64,
    np.complex64: complex64,
    np.complex128: complex128,
    # torch types
    torch.bool: bool,
    torch.uint8: uint8,
    torch.int8: int8,
    torch.int16: int16,
    torch.int32: int32,
    torch.int64: int64,
    torch.float32: float32,
    torch.float64: float64,
    torch.complex64: complex64,
    torch.complex128: complex128,
    # builtins
    builtins.bool: bool,
    builtins.int: int32,
    builtins.float: float32,
    builtins.complex: complex64,
}


def canonical_heat_type(a_type: Union[str, Type[datatype], Any]) -> Type[datatype]:
    """
    Canonicalize the builtin Python type, type string or HeAT type into a canonical HeAT type.

    Parameters
    ----------
    a_type : type, str, datatype
        A description for the type. It may be a a Python builtin type, string or an HeAT type already.
        In the three former cases the according mapped type is looked up, in the latter the type is simply returned.

    Raises
    -------
    TypeError
        If the type cannot be converted.
    """
    # already a heat type
    try:
        if issubclass(a_type, datatype):
            return a_type
    except TypeError:
        pass

    # extract type of numpy.dtype
    a_type = getattr(a_type, "type", a_type)

    # try to look the corresponding type up
    try:
        return __type_mappings[a_type]
    except KeyError:
        raise TypeError(f"data type {a_type} is not understood")


def heat_type_is_exact(ht_dtype: Type[datatype]) -> bool:
    """
    Check if HeAT type is an exact type, i.e an integer type. True if ht_dtype is an integer, False otherwise

    Parameters
    ----------
    ht_dtype: Type[datatype]
        HeAT type to check
    """
    return ht_dtype in _exact


def heat_type_is_inexact(ht_dtype: Type[datatype]) -> bool:
    """
    Check if HeAT type is an inexact type, i.e floating point type. True if ht_dtype is a float, False otherwise

    Parameters
    ----------
    ht_dtype: Type[datatype]
        HeAT type to check
    """
    return ht_dtype in _inexact


def heat_type_is_complexfloating(ht_dtype: Type[datatype]) -> bool:
    """
    Check if HeAT type is a complex floading point number, i.e complex64

    Parameters
    ----------
    ht_dtype: ht.dtype
        HeAT type to check

    Returns
    -------
    out: bool
        True if ht_dtype is a complex float, False otherwise
    """
    return ht_dtype in _complexfloating


def heat_type_of(
    obj: Union[str, Type[datatype], Any, Iterable[str, Type[datatype], Any]]
) -> Type[datatype]:
    """
    Returns the corresponding HeAT data type of given object, i.e. scalar, array or iterable. Attempts to determine the
    canonical data type based on the following priority list:
        1. dtype property
        2. type(obj)
        3. type(obj[0])

    Parameters
    ----------
    obj : scalar or DNDarray or iterable
        The object for which to infer the type.

    Raises
    -------
    TypeError
        If the object's type cannot be inferred.
    """
    # attempt to access the dtype property
    try:
        return canonical_heat_type(obj.dtype)
    except (AttributeError, TypeError):
        pass

    # attempt type of object itself
    try:
        return canonical_heat_type(type(obj))
    except TypeError:
        pass

    # last resort, type of the object at first position
    try:
        return canonical_heat_type(type(obj[0]))
    except (KeyError, IndexError, TypeError):
        raise TypeError(f"data type of {obj} is not understood")


# type code assignment
__type_codes = collections.OrderedDict(
    [
        (bool, 0),
        (uint8, 1),
        (int8, 2),
        (int16, 3),
        (int32, 4),
        (int64, 5),
        (float32, 6),
        (float64, 7),
        (complex64, 8),
        (complex128, 9),
    ]
)

# safe cast table
__safe_cast = [
    # bool  uint8  int8   int16  int32  int64  float32 float64 complex64 complex128
    [True, True, True, True, True, True, True, True, True, True],  # bool
    [False, True, False, True, True, True, True, True, True, True],  # uint8
    [False, False, True, True, True, True, True, True, True, True],  # int8
    [False, False, False, True, True, True, True, True, True, True],  # int16
    [False, False, False, False, True, True, False, True, False, True],  # int32
    [False, False, False, False, False, True, False, True, False, True],  # int64
    [False, False, False, False, False, False, True, True, True, True],  # float32
    [False, False, False, False, False, False, False, True, False, True],  # float64
    [False, False, False, False, False, False, False, False, True, True],  # complex64
    [False, False, False, False, False, False, False, False, False, True],  # complex128
]

# intuitive cast table
__intuitive_cast = [
    # bool  uint8  int8   int16  int32  int64  float32 float64 complex64 complex128
    [True, True, True, True, True, True, True, True, True, True],  # bool
    [False, True, False, True, True, True, True, True, True, True],  # uint8
    [False, False, True, True, True, True, True, True, True, True],  # int8
    [False, False, False, True, True, True, True, True, True, True],  # int16
    [False, False, False, False, True, True, True, True, True, True],  # int32
    [False, False, False, False, False, True, False, True, False, True],  # int64
    [False, False, False, False, False, False, True, True, True, True],  # float32
    [False, False, False, False, False, False, False, True, False, True],  # float64
    [False, False, False, False, False, False, False, False, True, True],  # complex64
    [False, False, False, False, False, False, False, False, False, True],  # complex128
]


# same kind table
__same_kind = [
    # bool  uint8  int8   int16  int32  int64  float32 float64 complex64 complex128
    [True, False, False, False, False, False, False, False, False, False],  # bool
    [False, True, True, True, True, True, False, False, False, False],  # uint8
    [False, True, True, True, True, True, False, False, False, False],  # int8
    [False, True, True, True, True, True, False, False, False, False],  # int16
    [False, True, True, True, True, True, False, False, False, False],  # int32
    [False, True, True, True, True, True, False, False, False, False],  # int64
    [False, False, False, False, False, False, True, True, False, False],  # float32
    [False, False, False, False, False, False, True, True, False, False],  # float64
    [False, False, False, False, False, False, False, False, True, True],  # complex64
    [False, False, False, False, False, False, False, False, True, True],  # complex128
]


# static list of possible casting methods
__cast_kinds = ["no", "safe", "same_kind", "unsafe", "intuitive"]


def can_cast(
    from_: Union[str, Type[datatype], Any],
    to: Union[str, Type[datatype], Any],
    casting: str = "intuitive",
) -> bool:
    """
    Returns True if cast between data types can occur according to the casting rule. If from is a scalar or array
    scalar, also returns True if the scalar value can be cast without overflow or truncation to an integer.

    Parameters
    ----------
    from_ : Union[str, Type[datatype], Any]
        Scalar, data type or type specifier to cast from.
    to : Union[str, Type[datatype], Any]
        Target type to cast to.
    casting: str, optional
        options: {"no", "safe", "same_kind", "unsafe", "intuitive"}, optional
        Controls the way the cast is evaluated
            * "no" the types may not be cast, i.e. they need to be identical
            * "safe" allows only casts that can preserve values with complete precision
            * "same_kind" safe casts are possible and down_casts within the same type family, e.g. int32 -> int8
            * "unsafe" means any conversion can be performed, i.e. this casting is always possible
            * "intuitive" allows all of the casts of safe plus casting from int32 to float32


    Raises
    -------
    TypeError
        If the types are not understood or casting is not a string
    ValueError
        If the casting rule is not understood

    Examples
    --------
    >>> ht.can_cast(ht.int32, ht.int64)
    True
    >>> ht.can_cast(ht.int64, ht.float64)
    True
    >>> ht.can_cast(ht.int16, ht.int8)
    False
    >>> ht.can_cast(1, ht.float64)
    True
    >>> ht.can_cast(2.0e200, "u1")
    False
    >>> ht.can_cast('i8', 'i4', 'no')
    False
    >>> ht.can_cast("i8", "i4", "safe")
    False
    >>> ht.can_cast("i8", "i4", "same_kind")
    True
    >>> ht.can_cast("i8", "i4", "unsafe")
    True
    """
    if not isinstance(casting, str):
        raise TypeError(f"expected string, found {type(casting)}")
    if casting not in __cast_kinds:
        raise ValueError(f"casting must be one of {str(__cast_kinds)[1:-1]}")

    # obtain the types codes of the canonical HeAT types
    try:
        typecode_from = __type_codes[canonical_heat_type(from_)]
    except TypeError:
        typecode_from = __type_codes[heat_type_of(from_)]
    typecode_to = __type_codes[canonical_heat_type(to)]

    # unsafe casting allows everything
    if casting == "unsafe":
        return True

    # types have to match exactly
    elif casting == "no":
        return typecode_from == typecode_to

    # safe casting or same_kind
    can_safe_cast = __safe_cast[typecode_from][typecode_to]
    if casting == "safe":
        return can_safe_cast
    can_intuitive_cast = __intuitive_cast[typecode_from][typecode_to]
    if casting == "intuitive":
        return can_intuitive_cast
    return can_safe_cast or __same_kind[typecode_from][typecode_to]


# compute possible type promotions dynamically
__type_promotions = [[None] * len(row) for row in __same_kind]
for i, operand_a in enumerate(__type_codes.keys()):
    for j, operand_b in enumerate(__type_codes.keys()):
        for target in __type_codes.keys():
            if can_cast(operand_a, target) and can_cast(operand_b, target):
                __type_promotions[i][j] = target
                break


def iscomplex(x: dndarray.DNDarray) -> dndarray.DNDarray:
    """
    Test element-wise if input is complex.

    Parameters
    ----------
    x : DNDarray
        The input DNDarray

    Examples
    --------
    >>> ht.iscomplex(ht.array([1+1j, 1]))
    DNDarray([ True, False], dtype=ht.bool, device=cpu:0, split=None)
    """
    sanitation.sanitize_in(x)

    if issubclass(x.dtype, _complexfloating):
        return x.imag != 0
    else:
        return factories.zeros(x.shape, bool, split=x.split, device=x.device, comm=x.comm)


def isreal(x: dndarray.DNDarray) -> dndarray.DNDarray:
    """
    Test element-wise if input is real-valued.

    Parameters
    ----------
    x : DNDarray
        The input DNDarray

    Examples
    --------
    >>> ht.iscomplex(ht.array([1+1j, 1]))
    DNDarray([ True, False], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.isreal, x, None, no_cast=True)


def issubdtype(
    arg1: Union[str, Type[datatype], Any], arg2: Union[str, Type[datatype], Any]
) -> builtins.bool:
    """
    Returns True if first argument is a typecode lower/equal in type hierarchy.

    Parameters
    ----------
    arg1 : type, str, ht.dtype
        A description representing the type. It may be a a Python builtin type, string or an HeAT type already.
    arg2 : type, str, ht.dtype
        A description representing the type. It may be a a Python builtin type, string or an HeAT type already.


    Examples
    --------
    >>> ints = ht.array([1, 2, 3], dtype=ht.int32)
    >>> ht.issubdtype(ints.dtype, ht.integer)
    True
    >>> ht.issubdype(ints.dtype, ht.floating)
    False
    >>> ht.issubdtype(ht.float64, ht.float32)
    False
    >>> ht.issubdtype('i', ht.integer)
    True
    """
    # Assure that each argument is a ht.dtype
    arg1 = canonical_heat_type(arg1)
    arg2 = canonical_heat_type(arg2)

    return issubclass(arg1, arg2)


def promote_types(
    type1: Union[str, Type[datatype], Any], type2: Union[str, Type[datatype], Any]
) -> Type[datatype]:
    """
    Returns the data type with the smallest size and smallest scalar kind to which both ``type1`` and ``type2`` may be
    intuitively cast to, where intuitive casting refers to maintaining the same bit length if possible. This function
    is symmetric.

    Parameters
    ----------
    type1 : type or str or datatype
        type of first operand
    type2 : type or str or datatype
        type of second operand

    Examples
    --------
    >>> ht.promote_types(ht.uint8, ht.uint8)
    <class 'heat.core.types.uint8'>
    >>> ht.promote_types(ht.int32, ht.float32)
    <class 'heat.core.types.float32'>
    >>> ht.promote_types(ht.int8, ht.uint8)
    <class 'heat.core.types.int16'>
    >>> ht.promote_types("i8", "f4")
    <class 'heat.core.types.float64'>
    """
    typecode_type1 = __type_codes[canonical_heat_type(type1)]
    typecode_type2 = __type_codes[canonical_heat_type(type2)]

    return __type_promotions[typecode_type1][typecode_type2]


def result_type(
    *arrays_and_types: Tuple[Union[dndarray.DNDarray, Type[datatype], Any]]
) -> Type[datatype]:
    """
    Returns the data type that results from type promotions rules performed in an arithmetic operation.

    Parameters
    ----------
    arrays_and_types: List of arrays and types
        Input arrays, types or numbers of the operation.

    Examples
    --------
    >>> ht.result_type(ht.array([1], dtype=ht.int32), 1)
    ht.int32
    >>> ht.result_type(ht.float32, ht.array(1, dtype=ht.int8))
    ht.float32
    >>> ht.result_type("i8", "f4")
    ht.float64
    """

    def result_type_rec(*arrays_and_types):
        # derive type and set precedence (lower number, higher precedence)
        arg = arrays_and_types[0]

        try:
            # array / tensor
            if isinstance(arg, np.ndarray):
                type1 = canonical_heat_type(arg.dtype.char)
            else:
                type1 = canonical_heat_type(arg.dtype)

            if len(arg.shape) > 0:
                prec1 = 0  # array
            else:
                prec1 = 2  # scalar
        except (AttributeError, TypeError):
            try:
                # type
                if isinstance(arg, np.dtype):
                    arg = arg.char
                type1 = canonical_heat_type(arg)
                prec1 = 1
            except TypeError:
                # type instance
                type1 = canonical_heat_type(type(arg))
                prec1 = 3

        # multiple arguments
        if len(arrays_and_types) > 1:
            type2, prec2 = result_type_rec(*arrays_and_types[1:])

            # fast check same type
            if type1 == type2:
                return type1, min(prec1, prec2)
            # fast check same precedence
            if prec1 == prec2:
                return promote_types(type1, type2), prec1

            # check if parent type is identical and decide by precedence
            for sclass in (bool, integer, floating, complex):
                if issubdtype(type1, sclass) and issubdtype(type2, sclass):
                    if prec1 < prec2:
                        return type1, min(prec1, prec2)
                    else:
                        return type2, min(prec1, prec2)

            # different parent type: bool < int < float < complex
            tc1 = __type_codes[type1]
            tc2 = __type_codes[type2]

            if tc1 < tc2:
                return type2, min(prec1, prec2)
            else:
                return type1, min(prec1, prec2)

        # single argument
        return type1, prec1

    return result_type_rec(*arrays_and_types)[0]


class finfo:
    """
    Class describing machine limits (bit representation) of floating point types.

    Attributes
    ----------
    bits : int
        The number of bits occupied by the type.
    eps : float
        The smallest representable positive number such that
        ``1.0 + eps != 1.0``.  Type of ``eps`` is an appropriate floating
        point type.
    max : float
        The largest representable number.
    min : float
        The smallest representable number, typically ``-max``.
    tiny : float
        The smallest positive usable number.  Type of ``tiny`` is an
        appropriate floating point type.

    Parameters
    ----------
    dtype : datatype
        Kind of floating point data-type about which to get information.

    Examples
    ---------
    >>> import heat as ht
    >>> info = ht.types.finfo(ht.float32)
    >>> info.bits
    32
    >>> info.eps
    1.1920928955078125e-07
    """

    def __new__(cls, dtype: Type[datatype]):
        try:
            dtype = heat_type_of(dtype)
        except (KeyError, IndexError, TypeError):
            # If given type is not heat type
            pass

        if dtype not in _inexact:
            raise TypeError(f"Data type {dtype} not inexact, not supported")

        return super(finfo, cls).__new__(cls)._init(dtype)

    def _init(self, dtype: Type[datatype]):
        _torch_finfo = torch.finfo(dtype.torch_type())
        for word in ["bits", "eps", "max", "min", "tiny"]:
            setattr(self, word, getattr(_torch_finfo, word))

        return self


class iinfo:
    """
    Class describing machine limits (bit representation) of integer types.

    Attributes
    ----------
    bits : int
        The number of bits occupied by the type.
    max : float
        The largest representable number.
    min : float
        The smallest representable number, typically ``-max``.

    Parameters
    ----------
    dtype : datatype
        Kind of floating point data-type about which to get information.

    Examples
    ---------
    >>> import heat as ht
    >>> info = ht.types.iinfo(ht.int32)
    >>> info.bits
    32
    """

    def __new__(cls, dtype: Type[datatype]):
        try:
            dtype = heat_type_of(dtype)
        except (KeyError, IndexError, TypeError):
            # If given type is not heat type
            pass

        if dtype not in _exact:
            raise TypeError(f"Data type {dtype} not exact, not supported")

        return super(iinfo, cls).__new__(cls)._init(dtype)

    def _init(self, dtype: Type[datatype]):
        _torch_iinfo = torch.iinfo(dtype.torch_type())
        for word in ["bits", "min", "max"]:
            setattr(self, word, getattr(_torch_iinfo, word))

        return self


# dndarray is imported at the very end to break circular dependency
from . import dndarray
