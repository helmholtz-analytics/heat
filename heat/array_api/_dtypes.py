import heat as ht

bool = ht.bool
"""Boolean (``True`` or ``False``)."""
int8 = ht.int8
"""An 8-bit signed integer whose values exist on the interval ``[-128, +127]``."""
int16 = ht.int16
"""A 16-bit signed integer whose values exist on the interval ``[-32,767, +32,767]``."""
int32 = ht.int32
"""A 32-bit signed integer whose values exist on the interval ``[-2,147,483,647, +2,147,483,647]``."""
int64 = ht.int64
"""A 64-bit signed integer whose values exist on the interval ``[-9,223,372,036,854,775,807, +9,223,372,036,854,775,807]``."""
uint8 = ht.uint8
"""An 8-bit unsigned integer whose values exist on the interval ``[0, +255]``."""
# For the status of Pytorch support for these 3 data types,
# see https://github.com/pytorch/pytorch/issues/58734
# uint16 =
# """A 16-bit unsigned integer whose values exist on the interval ``[0, +65,535]``."""
# uint32 =
# """A 32-bit unsigned integer whose values exist on the interval ``[0, +4,294,967,295]``."""
# uint64 =
# """A 64-bit unsigned integer whose values exist on the interval ``[0, +18,446,744,073,709,551,615]``."""
float32 = ht.float32
"""IEEE 754 single-precision (32-bit) binary floating-point number (see IEEE 754-2019)."""
float64 = ht.float64
"""IEEE 754 double-precision (64-bit) binary floating-point number (see IEEE 754-2019)."""
complex64 = ht.complex64
"""Single-precision (64-bit) complex floating-point number whose real and imaginary components must be IEEE 754 single-precision (32-bit) binary floating-point numbers (see IEEE 754-2019)."""
complex128 = ht.complex128
"""Double-precision (128-bit) complex floating-point number whose real and imaginary components must be IEEE 754 double-precision (64-bit) binary floating-point numbers (see IEEE 754-2019)."""

default_int = int64
"""Default integer data type is ``int64``"""
default_float = float64
"""Default floating-point data type is ``float64``"""

_all_dtypes = (
    bool,
    int8,
    int16,
    int32,
    int64,
    uint8,
    # uint16,
    # uint32,
    # uint64,
    float32,
    float64,
    complex64,
    complex128,
)
_boolean_dtypes = (bool,)
_real_floating_dtypes = (float32, float64)
_floating_dtypes = (float32, float64, complex64, complex128)
_complex_floating_dtypes = (complex64, complex128)
_integer_dtypes = (
    int8,
    int16,
    int32,
    int64,
    uint8,
    # uint16,
    # uint32,
    # uint64
)
_signed_integer_dtypes = (int8, int16, int32, int64)
_unsigned_integer_dtypes = (
    uint8,
    # uint16,
    # utnt32,
    # uint64
)
_integer_or_boolean_dtypes = (
    bool,
    int8,
    int16,
    int32,
    int64,
    uint8,
    # uint16,
    # uint32,
    # uint64,
)
_real_numeric_dtypes = (
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    # uint16,
    # uint32,
    # uint64,
)
_numeric_dtypes = (
    float32,
    float64,
    complex64,
    complex128,
    int8,
    int16,
    int32,
    int64,
    uint8,
    # uint16,
    # uint32,
    # uint64,
)

_dtype_categories = {
    "all": _all_dtypes,
    "real numeric": _real_numeric_dtypes,
    "numeric": _numeric_dtypes,
    "integer": _integer_dtypes,
    "integer or boolean": _integer_or_boolean_dtypes,
    "boolean": _boolean_dtypes,
    "real floating-point": _floating_dtypes,
    "complex floating-point": _complex_floating_dtypes,
    "floating-point": _floating_dtypes,
}

_promotion_table = {
    (int8, int8): int8,
    (int8, int16): int16,
    (int8, int32): int32,
    (int8, int64): int64,
    (int16, int8): int16,
    (int16, int16): int16,
    (int16, int32): int32,
    (int16, int64): int64,
    (int32, int8): int32,
    (int32, int16): int32,
    (int32, int32): int32,
    (int32, int64): int64,
    (int64, int8): int64,
    (int64, int16): int64,
    (int64, int32): int64,
    (int64, int64): int64,
    (uint8, uint8): uint8,
    # (uint8, uint16): uint16,
    # (uint8, uint32): uint32,
    # (uint8, uint64): uint64,
    # (uint16, uint8): uint16,
    # (uint16, uint16): uint16,
    # (uint16, uint32): uint32,
    # (uint16, uint64): uint64,
    # (uint32, uint8): uint32,
    # (uint32, uint16): uint32,
    # (uint32, uint32): uint32,
    # (uint32, uint64): uint64,
    # (uint64, uint8): uint64,
    # (uint64, uint16): uint64,
    # (uint64, uint32): uint64,
    # (uint64, uint64): uint64,
    (int8, uint8): int16,
    # (int8, uint16): int32,
    # (int8, uint32): int64,
    (int16, uint8): int16,
    # (int16, uint16): int32,
    # (int16, uint32): int64,
    (int32, uint8): int32,
    # (int32, uint16): int32,
    # (int32, uint32): int64,
    (int64, uint8): int64,
    # (int64, uint16): int64,
    # (int64, uint32): int64,
    (uint8, int8): int16,
    # (uint16, int8): int32,
    # (uint32, int8): int64,
    (uint8, int16): int16,
    # (uint16, int16): int32,
    # (uint32, int16): int64,
    (uint8, int32): int32,
    # (uint16, int32): int32,
    # (uint32, int32): int64,
    (uint8, int64): int64,
    # (uint16, int64): int64,
    # (uint32, int64): int64,
    (float32, float32): float32,
    (float32, float64): float64,
    (float64, float32): float64,
    (float64, float64): float64,
    (complex64, complex64): complex64,
    (complex128, complex128): complex128,
    (complex64, complex128): complex128,
    (complex128, complex64): complex128,
    (float32, complex64): complex64,
    (float32, complex128): complex128,
    (float64, complex64): complex128,
    (float64, complex128): complex128,
    (complex64, float32): complex64,
    (complex64, float64): complex128,
    (complex128, float32): complex128,
    (complex128, float64): complex128,
    (bool, bool): bool,
}


def _result_type(type1, type2):
    if (type1, type2) in _promotion_table:
        return _promotion_table[type1, type2]
    raise TypeError(f"{type1} and {type2} cannot be type promoted together")
