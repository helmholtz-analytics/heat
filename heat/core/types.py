"""
types: Defines the basic heat data types in the hierarchy as shown below. Design inspired by the Python package numpy.

As part of the type-hierarchy: xx -- is bit-width
generic
 +-> bool                                   (kind=b)
 +-> number
 |   +-> integer
 |   |   +-> signedinteger     (intxx)      (kind=i)
 |   |   |     byte
 |   |   |     short
 |   |   |     int
 |   |   |     long
 |   |   |     longlong
 |   |   \\-> unsignedinteger  (uintxx)     (kind=u)
 |   |         ubyte
 |   |         ushort
 |   |         uint
 |   |         ulong
 |   |         ulonglong
 |   +-> floating              (floatxx)    (kind=f)
 |   |     half
 |   |     float
 |   |     double              (double)
 \\-> flexible (currently unused, placeholder for characters)
"""

import abc

# maintain references to Python's builtin types
_bool = bool
_float = float
_int = int


class generic(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __new__(cls, *args, **kwargs):
        pass


class bool(generic):
    pass


bool_ = bool


class number(generic):
    pass


class integer(number):
    pass


class signedinteger(integer):
    pass


class int8(signedinteger):
    pass


byte = int8


class int16(signedinteger):
    pass


short = int16


class int32(signedinteger):
    pass


int = int32
int_ = int32


class int64(signedinteger):
    pass


long = int64
longlong = int64


class unsignedinteger(integer):
    pass


class uint8(unsignedinteger):
    pass


ubyte = uint8


class uint16(unsignedinteger):
    pass


ushort = uint16


class uint32(unsignedinteger):
    pass


uint = uint32


class uint64(unsignedinteger):
    pass


ulong = uint64
ulonglong = uint64


class floating(number):
    pass


class float16(floating):
    pass


half = float16


class float32(floating):
    pass


float = float32
float_ = float32


class float64(floating):
    pass


double = float64


class flexible(generic):
    pass
