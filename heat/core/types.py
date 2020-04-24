"""
types: Defines the basic heat data types in the hierarchy as shown below. Design inspired by the Python package numpy.

As part of the type-hierarchy: xx -- is bit-width
generic
 +-> bool, bool_                                (kind=?)
 +-> number
 |   +-> integer
 |   |   +-> signedinteger         (intxx)      (kind=b, i)
 |   |   |     int8, byte
 |   |   |     int16, short
 |   |   |     int32, int
 |   |   |     int64, long
 |   |   \\-> unsignedinteger      (uintxx)     (kind=B, u)
 |   |         uint8, ubyte
 |   \\-> floating                 (floatxx)    (kind=f)
 |         float32, float, float_
 |         float64, double         (double)
 \\-> flexible (currently unused, placeholder for characters)
"""

import builtins
import collections
import numpy as np
import torch

from . import communication
from . import devices
from . import factories


__all__ = [
    "generic",
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
    "promote_types",
]


class generic:
    def __new__(cls, *value, device=None, comm=None):
        torch_type = cls.torch_type()
        if torch_type is NotImplemented:
            raise TypeError("cannot create '{}' instances".format(cls))

        value_count = len(value)

        # sanitize the distributed processing flags
        comm = communication.sanitize_comm(comm)
        device = devices.sanitize_device(device)

        # check whether there are too many arguments
        if value_count >= 2:
            raise TypeError("function takes at most 1 argument ({} given)".format(value_count))
        # if no value is given, we will initialize the value to be zero
        elif value_count == 0:
            value = ((0,),)

        # otherwise, attempt to create a torch tensor of given type
        try:
            array = value[0]._DNDarray__array.type(torch_type)
            if value[0].split is None:
                return factories.array(array, dtype=cls, split=None, comm=comm, device=device)
            else:
                return factories.array(
                    array, dtype=cls, is_split=value[0].split, comm=comm, device=device
                )
        except AttributeError:
            # this is the case of that the first/only element of value is not a DNDarray
            array = torch.tensor(*value, dtype=torch_type, device=device.torch_device)
        except TypeError as exception:
            # re-raise the exception to be consistent with numpy's exception interface
            raise ValueError(str(exception))

        return dndarray.DNDarray(
            array, tuple(array.shape), cls, split=None, device=device, comm=comm
        )

    @classmethod
    def torch_type(cls):
        return NotImplemented

    @classmethod
    def char(cls):
        return NotImplemented


class bool(generic):
    @classmethod
    def torch_type(cls):
        return torch.bool

    @classmethod
    def char(cls):
        return "u1"


class number(generic):
    pass


class integer(number):
    pass


class signedinteger(integer):
    pass


class int8(signedinteger):
    @classmethod
    def torch_type(cls):
        return torch.int8

    @classmethod
    def char(cls):
        return "i1"


class int16(signedinteger):
    @classmethod
    def torch_type(cls):
        return torch.int16

    @classmethod
    def char(cls):
        return "i2"


class int32(signedinteger):
    @classmethod
    def torch_type(cls):
        return torch.int32

    @classmethod
    def char(cls):
        return "i4"


class int64(signedinteger):
    @classmethod
    def torch_type(cls):
        return torch.int64

    @classmethod
    def char(cls):
        return "i8"


class unsignedinteger(integer):
    pass


class uint8(unsignedinteger):
    @classmethod
    def torch_type(cls):
        return torch.uint8

    @classmethod
    def char(cls):
        return "u1"


class floating(number):
    pass


class float32(floating):
    @classmethod
    def torch_type(cls):
        return torch.float32

    @classmethod
    def char(cls):
        return "f4"


class float64(floating):
    @classmethod
    def torch_type(cls):
        return torch.float64

    @classmethod
    def char(cls):
        return "f8"


class flexible(generic):
    pass


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

_inexact = (
    # float16,
    float32,
    float64,
)

_exact = (uint8, int8, int16, int32, int64)

# type mappings for type strings and builtins types
__type_mappings = {
    # type strings
    "?": bool,
    "b": int8,
    "i": int32,
    "i1": int8,
    "i2": int16,
    "i4": int32,
    "i8": int64,
    "B": uint8,
    "u": uint8,
    "u1": uint8,
    "f": float32,
    "f4": float32,
    "f8": float64,
    # numpy types
    np.bool: bool,
    np.uint8: uint8,
    np.int8: int8,
    np.int16: int16,
    np.int32: int32,
    np.int64: int64,
    np.float32: float32,
    np.float64: float64,
    # torch types
    torch.bool: bool,
    torch.uint8: uint8,
    torch.int8: int8,
    torch.int16: int16,
    torch.int32: int32,
    torch.int64: int64,
    torch.float32: float32,
    torch.float64: float64,
    # builtins
    builtins.bool: bool,
    builtins.int: int32,
    builtins.float: float32,
}


def canonical_heat_type(a_type):
    """
    Canonicalize the builtin Python type, type string or HeAT type into a canonical HeAT type.

    Parameters
    ----------
    a_type : type, str, ht.dtype
        A description for the type. It may be a a Python builtin type, string or an HeAT type already.
        In the three former cases the according mapped type is looked up, in the latter the type is simply returned.

    Returns
    -------
    out : ht.dtype
        The canonical HeAT type.

    Raises
    -------
    TypeError
        If the type cannot be converted.
    """
    # already a heat type
    try:
        if issubclass(a_type, generic):
            return a_type
    except TypeError:
        pass

    # try to look the corresponding type up
    try:
        return __type_mappings[a_type]
    except KeyError:
        raise TypeError("data type {} is not understood".format(a_type))


def heat_type_is_exact(ht_dtype):
    """
    Check if HeAT type is an exact type, i.e an integer type

    Parameters
    ----------
    ht_dtype: ht.dtype
        HeAT type to check

    Returns
    -------
    out: bool
        True if ht_dtype is an integer, False otherwise
    """
    return ht_dtype in _exact


def heat_type_is_inexact(ht_dtype):
    """
    Check if HeAT type is an inexact type, i.e floating point type

    Parameters
    ----------
    ht_dtype: ht.dtype
        HeAT type to check

    Returns
    -------
    out: bool
        True if ht_dtype is a float, False otherwise
    """
    return ht_dtype in _inexact


def heat_type_of(obj):
    """
    Returns the corresponding HeAT data type of given object, i.e. scalar, array or iterable. Attempts to determine the
    canonical data type based on the following priority list:
        1. dtype property
        2. type(obj)
        3. type(obj[0])

    Parameters
    ----------
    obj : scalar, array, iterable
        The object for which to infer the type.

    Returns
    -------
    out : ht.dtype
        The object's corresponding HeAT type.

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
        raise TypeError("data type of {} is not understood".format(obj))


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
    ]
)

# safe cast table
__safe_cast = [
    # bool  uint8  int8   int16  int32  int64  float32 float64
    [True, True, True, True, True, True, True, True],  # bool
    [False, True, False, True, True, True, True, True],  # uint8
    [False, False, True, True, True, True, True, True],  # int8
    [False, False, False, True, True, True, True, True],  # int16
    [False, False, False, False, True, True, False, True],  # int32
    [False, False, False, False, False, True, False, True],  # int64
    [False, False, False, False, False, False, True, True],  # float32
    [False, False, False, False, False, False, False, True],  # float64
]


# same kind table
__same_kind = [
    # bool  uint8  int8   int16  int32  int64  float32 float64
    [True, False, False, False, False, False, False, False],  # bool
    [False, True, True, True, True, True, False, False],  # uint8
    [False, True, True, True, True, True, False, False],  # int8
    [False, True, True, True, True, True, False, False],  # int16
    [False, True, True, True, True, True, False, False],  # int32
    [False, True, True, True, True, True, False, False],  # int64
    [False, False, False, False, False, False, True, True],  # float32
    [False, False, False, False, False, False, True, True],  # float64
]


# static list of possible casting methods
__cast_kinds = ["no", "safe", "same_kind", "unsafe"]


def can_cast(from_, to, casting="safe"):
    """
    Returns True if cast between data types can occur according to the casting rule. If from is a scalar or array
    scalar, also returns True if the scalar value can be cast without overflow or truncation to an integer.

    Parameters
    ----------
    from_ : scalar, tensor, type, str, ht.dtype
        Scalar, data type or type specifier to cast from.
    to : type, str, ht.dtype
        Target type to cast to.
    casting: str {'no', 'safe', 'same_kind', 'unsafe'}, optional
        Controls the way the cast is evaluated
            * 'no' the types may not be cast, i.e. they need to be identical
            * 'safe' allows only casts that can preserve values with complete precision
            * 'same_kind' safe casts are possible and down_casts within the same type family, e.g. int32 -> int8
            * 'unsafe' means any conversion can be performed, i.e. this casting is always possible

    Returns
    -------
    out : bool
        True if cast can occur according to the casting rules, False otherwise.

    Raises
    -------
    TypeError
        If the types are not understood or casting is not a string
    ValueError
        If the casting rule is not understood

    Examples
    --------
    Basic examples with types

    >>> ht.can_cast(ht.int32, ht.int64)
    True
    >>> ht.can_cast(ht.int64, ht.float64)
    True
    >>> ht.can_cast(ht.int16, ht.int8)
    False

    The usage of scalars is also possible
    >>> ht.can_cast(1, ht.float64)
    True
    >>> ht.can_cast(2.0e200, 'u1')
    False

    can_cast supports different casting rules
    >>> ht.can_cast('i8', 'i4', 'no')
    False
    >>> ht.can_cast('i8', 'i4', 'safe')
    False
    >>> ht.can_cast('i8', 'i4', 'same_kind')
    True
    >>> ht.can_cast('i8', 'i4', 'unsafe')
    True
    """
    if not isinstance(casting, str):
        raise TypeError("expected string, found {}".format(type(casting)))
    if casting not in __cast_kinds:
        raise ValueError("casting must be one of {}".format(str(__cast_kinds)[1:-1]))

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
    return can_safe_cast or __same_kind[typecode_from][typecode_to]


# compute possible type promotions dynamically
__type_promotions = [[None] * len(row) for row in __same_kind]
for i, operand_a in enumerate(__type_codes.keys()):
    for j, operand_b in enumerate(__type_codes.keys()):
        for target in __type_codes.keys():
            if can_cast(operand_a, target) and can_cast(operand_b, target):
                __type_promotions[i][j] = target
                break


def promote_types(type1, type2):
    """
    Returns the data type with the smallest size and smallest scalar kind to which both type1 and type2 may be safely
    cast. This function is symmetric.

    Parameters
    ----------
    type1 : type, str, ht.dtype
        type of first operand
    type2 : type, str, ht.dtype
        type of second operand

    Returns
    -------
    out : ht.dtype
        The promoted data type.

    Examples
    --------
    >>> ht.promote_types(ht.uint8, ht.uint8)
    ht.uint8
    >>> ht.promote_types(ht.int8, ht.uint8)
    ht.int16
    >>> ht.promote_types('i8', 'f4')
    ht.float64
    """
    typecode_type1 = __type_codes[canonical_heat_type(type1)]
    typecode_type2 = __type_codes[canonical_heat_type(type2)]

    return __type_promotions[typecode_type1][typecode_type2]


class finfo:
    """
    finfo(dtype)

    Class describing machine limits (bit representation) of floating point types.

    Attributes
    ----------
    bits : int
        The number of bits occupied by the type.
    eps : float
        The smallest representable positive number such that
        ``1.0 + eps != 1.0``.  Type of `eps` is an appropriate floating
        point type.
    max : floating point number of the appropriate type
        The largest representable number.
    min : floating point number of the appropriate type
        The smallest representable number, typically ``-max``.
    tiny : float
        The smallest positive usable number.  Type of `tiny` is an
        appropriate floating point type.

    Parameters
    ----------
    dtype : ht.dtype
        Kind of floating point data-type about which to get information.

    Examples:
    ---------
    >>> import heat as ht
    >>> info = ht.types.finfo(ht.float32)
    >>> info.bits
    32

    >>> info.eps
    1.1920928955078125e-07
    """

    def __new__(cls, dtype):
        try:
            dtype = heat_type_of(dtype)
        except (KeyError, IndexError, TypeError):
            # If given type is not heat type
            pass

        if dtype not in _inexact:
            raise TypeError("Data type {} not inexact, not supported".format(dtype))

        return super(finfo, cls).__new__(cls)._init(dtype)

    def _init(self, dtype):
        _torch_finfo = torch.finfo(dtype.torch_type())
        for word in ["bits", "eps", "max", "tiny"]:
            setattr(self, word, getattr(_torch_finfo, word))

        self.min = -self.max

        return self


class iinfo:
    """
    iinfo(dtype)

    Class describing machine limits (bit representation) of integer types.

    Attributes
    ----------
    bits : int
        The number of bits occupied by the type.
    max : floating point number of the appropriate type
        The largest representable number.
    min : floating point number of the appropriate type
        The smallest representable number, typically ``-max``.

    Parameters
    ----------
    dtype : ht.dtype
        Kind of floating point data-type about which to get information.

    Examples:
    ---------
    >>> import heat as ht
    >>> info = ht.types.finfo(ht.int32)
    >>> info.bits
    32
    """

    def __new__(cls, dtype):
        try:
            dtype = heat_type_of(dtype)
        except (KeyError, IndexError, TypeError):
            # If given type is not heat type
            pass

        if dtype not in _exact:
            raise TypeError("Data type {} not inexact, not supported".format(dtype))

        return super(iinfo, cls).__new__(cls)._init(dtype)

    def _init(self, dtype):
        _torch_iinfo = torch.iinfo(dtype.torch_type())
        for word in ["bits", "max"]:
            setattr(self, word, getattr(_torch_iinfo, word))

        self.min = -(self.max + 1)

        return self


# tensor is imported at the very end to break circular dependency
from . import dndarray
