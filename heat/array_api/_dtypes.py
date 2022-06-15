import heat as ht

int8 = ht.int8
int16 = ht.int16
int32 = ht.int32
int64 = ht.int64
uint8 = ht.uint8
# uint16 =
# uint32 =
# uint64 =
float32 = ht.float32
float64 = ht.float64
bool = ht.bool
default_float = float32

_all_dtypes = (
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
    bool,
)
_boolean_dtypes = (bool,)
_floating_dtypes = (float32, float64)
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
_numeric_dtypes = (
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

_dtype_categories = {
    "all": _all_dtypes,
    "numeric": _numeric_dtypes,
    "integer": _integer_dtypes,
    "integer or boolean": _integer_or_boolean_dtypes,
    "boolean": _boolean_dtypes,
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
    (bool, bool): bool,
}


def _result_type(type1, type2):
    if (type1, type2) in _promotion_table:
        return _promotion_table[type1, type2]
    raise TypeError(f"{type1} and {type2} cannot be type promoted together")
