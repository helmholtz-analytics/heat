"""
types: Defines the basic heat data types in the hierarchy as shown below. Design inspired by the Python package numpy.

As part of the type-hierarchy: xx -- is bit-width
generic
 +-> bool_                                  (kind=b)
 +-> number
 |   +-> integer
 |   |   +-> signedinteger     (intxx)      (kind=i)
 |   |   |     byte
 |   |   |     short
 |   |   |     int_
 |   |   |     longlong
 |   |   \\-> unsignedinteger  (uintxx)     (kind=u)
 |   |         ubyte
 |   |         ushort
 |   |         uint_
 |   |         ulonglong
 |   +-> floating              (floatxx)    (kind=f)
 |   |     half
 |   |     float_
 |   |     double              (double)
 \\-> flexible (currently unused, placeholder for characters)
"""
