"""
types: Defines the basic heat data types in the hierarchy as shown below. Design inspired by the Python package numpy.

As part of the type-hierarchy: xx -- is bit-width
generic
 +-> bool, bool_                            (kind=?)
 +-> number
 |   +-> integer
 |   |   +-> signedinteger     (intxx)      (kind=b, i)
 |   |   |     int8, byte
 |   |   |     int16, short
 |   |   |     int32, int
 |   |   |     int64, long
 |   |   \\-> unsignedinteger  (uintxx)     (kind=B, u)
 |   |         uint8, ubyte
 |   \\-> floating             (floatxx)    (kind=f)
 |         float32, float, float_
 |         float64, double     (double)
 \\-> flexible (currently unused, placeholder for characters)
"""

import abc
import builtins
import torch

from .communicator import NoneCommunicator
from . import tensor


__all__ = [
    'generic',
    'number',
    'integer',
    'signedinteger',
    'unsignedinteger',
    'bool',
    'bool_',
    'floating',
    'int8',
    'byte',
    'int16',
    'short',
    'int32',
    'int',
    'int64',
    'long',
    'uint8',
    'ubyte',
    'float32',
    'float',
    'float_',
    'float64',
    'double',
    'flexible',
]


class generic(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __new__(cls, *value):
        try:
            torch_type = cls.torch_type()
        except TypeError:
            raise TypeError('cannot create \'{}\' instances'.format(cls))

        value_count = len(value)

        # check whether there are too many arguments
        if value_count >= 2:
            raise TypeError('function takes at most 1 argument ({} given)'.format(value_count))
        # if no value is given, we will initialize the value to be zero
        elif value_count == 0:
            value = ((0,),)

        # otherwise, attempt to create a torch tensor of given type
        try:
            array = torch.tensor(*value, dtype=torch_type)
        except TypeError as exception:
            # re-raise the exception to be consistent with numpy's exception interface
            raise ValueError(str(exception))

        return tensor.tensor(array, tuple(array.shape), split=None, comm=NoneCommunicator())

    @classmethod
    @abc.abstractclassmethod
    def torch_type(cls):
        pass


class bool(generic):
    @classmethod
    def torch_type(cls):
        return torch.uint8


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


class int16(signedinteger):
    @classmethod
    def torch_type(cls):
        return torch.int16


class int32(signedinteger):
    @classmethod
    def torch_type(cls):
        return torch.int32


class int64(signedinteger):
    @classmethod
    def torch_type(cls):
        return torch.int64


class unsignedinteger(integer):
    pass


class uint8(unsignedinteger):
    @classmethod
    def torch_type(cls):
        return torch.uint8


class floating(number):
    pass


class float32(floating):
    @classmethod
    def torch_type(cls):
        return torch.float32


class float64(floating):
    @classmethod
    def torch_type(cls):
        return torch.float64


class flexible(generic):
    pass


# definition of aliases
bool_ = bool
byte = int8
short = int16
int = int32
int_ = int32
long = int64
ubyte = uint8
float = float32
float_ = float32
double = float64

# type mappings for type strings and builtins types
__type_mappings = {
    '?':            bool,
    'b':            int8,
    'i':            int32,
    'i1':           int8,
    'i2':           int16,
    'i4':           int32,
    'i8':           int64,
    'B':            uint8,
    'u':            uint8,
    'u1':           uint8,
    'f':            float32,
    'f4':           float32,
    'f8':           float64,
    builtins.bool:  bool,
    builtins.int:   int32,
    builtins.float: float32,
}


def as_torch_type(a_type):
    """
    Infers the PyTorch type from a given HeAT type, type string or Python builtin type.

    Parameters
    ----------
    a_type : type, str, ht.dtype
        A description for the type. It may be a Python builtin type, string or an HeAT type already.
        In the two former cases the according mapped type is looked up, in the latter the type is simply returned.

    Returns
    -------
    out : ht.dtype
        The matching HeAT type.

    Raises
    -------
    TypeError
        If the type cannot be converted.
    """
    if issubclass(a_type, generic):
        return a_type.torch_type()
    try:
        return __type_mappings[a_type].torch_type()
    except KeyError:
        raise TypeError('data type {} is not understood'.format(a_type))


# inverse type mappings for reconstructing heat from torch types
__inverse_type_mappings = {
    torch.int8:    int8,
    torch.int16:   int16,
    torch.int32:   int32,
    torch.int64:   int64,
    torch.uint8:   uint8,
    torch.float32: float32,
    torch.float64: float64,
}


def as_heat_type(torch_type):
    """
    Converts a PyTorch type to its equivalent HeAT type.

    Parameters
    ----------
    torch_type : torch.dtype
        The PyTorch type to convert.

    Returns
    -------
    out : ht.dtype
        The equivalent HeAT type.

    Raises
    -------
    TypeError
        If the type cannot be converted.
    """
    try:
        return __inverse_type_mappings[torch_type]
    except KeyError:
        raise TypeError('data type {} is not understood'.format(torch_type))
