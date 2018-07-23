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


class byte(signedinteger):
    pass


class short(signedinteger):
    pass


class int(signedinteger):
    pass
int_ = int


class long(signedinteger):
    pass


class longlong(signedinteger):
    pass


class unsignedinteger(integer):
    pass


class ubyte(unsignedinteger):
    pass


class ushort(unsignedinteger):
    pass


class uint(unsignedinteger):
    pass


class ulong(unsignedinteger):
    pass


class ulonglong(unsignedinteger):
    pass


class floating(number):
    pass


class half(floating):
    pass


class float(floating):
    pass
float_ = float


class double(floating):
    pass


class flexible(generic):
    pass
