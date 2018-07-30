"""
types: Defines the basic heat data types in the hierarchy as shown below. Design inspired by the Python package numpy.

As part of the type-hierarchy: xx -- is bit-width
generic
 +-> bool                                   (kind=b)
 +-> number
 |   +-> integer
 |   |   +-> signedinteger     (intxx)      (kind=i)
 |   |   |     int8, byte
 |   |   |     int16, short
 |   |   |     int32, int
 |   |   |     int64, long
 |   |   \\-> unsignedinteger  (uintxx)     (kind=u)
 |   |         uint8, ubyte
 |   \\-> floating             (floatxx)    (kind=f)
 |         float16, half
 |         float32, float
 |         float64, double     (double)
 \\-> flexible (currently unused, placeholder for characters)
"""

import abc
import builtins
import torch

from .communicator import NoneCommunicator
from .tensor import tensor


class generic(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __new__(cls, *args, **kwargs):
        try:
            torch_type = cls.torch_type()

            typed_tensor = tensor(Comm=NoneCommunicator)
            typed_tensor.array = torch.zeros(1, dtype=torch_type)
            typed_tensor._tensor__gshape = (1,)

            return typed_tensor

        except TypeError:
            raise TypeError('cannot create \'{}\' instances'.format(cls))

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


class float16(floating):
    @classmethod
    def torch_type(cls):
        return torch.float16


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

half = float16
float = float32
float_ = float32
double = float64

# type mappings for type strings and builtins types
__type_mappings = {
    'b':  bool,
    'i':  int32,
    'i1': int8,
    'i2': int16,
    'i4': int32,
    'i8': int64,
    'u':  uint8,
    'u1': uint8,
    'f':  float32,
    'f2': float16,
    'f4': float32,
    'f8': float64,
    builtins.bool:  bool,
    builtins.int:   int32,
    builtins.float: float32,
}

# inverse type mappings for reconstructing heat from torch types
__inverse_type_mappings = {
    torch.int8:    int8,
    torch.int16:   int16,
    torch.int32:   int32,
    torch.int64:   int64,
    torch.uint8:   uint8,
    torch.float16: float16,
    torch.float32: float32,
    torch.float64: float64,
}
