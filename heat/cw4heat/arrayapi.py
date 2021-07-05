"""
Functions, data-types etc. defined by Array API.
See https://data-apis.org/array-api/latest
"""

__all__ = [
    "aa_creators",
    "aa_attributes",
    "aa_methods",
    "aa_elementwises",
    "aa_statisticals",
    "aa_inplace_operators",
    "aa_reflected_operators",
    "aa_datatypes",
    "aa_datatype_functions",
    "aa_searching",
    "aa_sorting",
    "aa_set",
    "aa_utility",
    "aa_constants",
    "aa_arraydir",
    "aa_tldir",
    "aa_tlfuncs",
    "aa_arrayfuncs",
    "aa_methods_s",
    "aa_methods_a",
]

aa_creators = [
    "arange",  # (start, /, stop=None, step=1, *, dtype=None, device=None)
    "asarray",  # (obj, /, *, dtype=None, device=None, copy=None)
    "empty",  # (shape, *, dtype=None, device=None)
    "empty_like",  # (x, /, *, dtype=None, device=None)
    "eye",  # (n_rows, n_cols=None, /, *, k=0, dtype=None, device=None)
    "from_dlpack",  # (x, /)
    "full",  # (shape, fill_value, *, dtype=None, device=None)
    "full_like",  # (x, /, fill_value, *, dtype=None, device=None)
    "linspace",  # (start, stop, /, num, *, dtype=None, device=None, endpoint=True)
    "meshgrid",  # (*arrays, indexing=’xy’)
    "ones",  # (shape, *, dtype=None, device=None)
    "ones_like",  # (x, /, *, dtype=None, device=None)
    "zeros",  # (shape, *, dtype=None, device=None)
    "zeros_like",  # (x, /, *, dtype=None, device=None)
]

aa_attributes = ["dtype", "device", "ndim", "shape", "size", "T"]

aa_inplace_operators = [
    "__iadd__",
    "__isub__",
    "__imul__",
    "__itruediv__",
    "__iflowdiv__",
    "__ipow__",
    "__imatmul__",
    "__imod__",
    "__iand__",
    "__ior__",
    "__ixor__",
    "__ilshift__",
    "__irshift__",
]

aa_reflected_operators = [
    "__radd__",
    "__rsub__",
    "__rmul__",
    "__rtruediv__",
    "__rflowdiv__",
    "__rpow__",
    "__rmatmul__",
    "__rmod__",
    "__rand__",
    "__ror__",
    "__rxor__",
    "__rlshift__",
    "__rrshift__",
]

aa_datatypes = [
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
]

aa_datatype_functions = [
    "broadcast_arrays",  # (*arrays)
    "broadcast_to",  # (x, /, shape)
    "can_cast",  # (from_, to, /)
    "finfo",  # (type, /)
    "iinfo",  # (type, /)
    "result_type",  # (*arrays_and_dtypes)
]

aa_methods = [
    "__abs__",  # (self, /)
    "__add__",  # (self, other, /)
    "__and__",  # (self, other, /)
    "__array_namespace__",  # (self, /, *, api_version=None)
    "__bool__",  # (self, /)
    "__dlpack__",  # (self, /, *, stream=None)
    "__dlpack_device__",  # (self, /)
    "__eq__",  # (self, other, /)
    "__float__",  # (self, /)
    "__floordiv__",  # (self, other, /)
    "__ge__",  # (self, other, /)
    "__getitem__",  # (self, key, /)
    "__gt__",  # (self, other, /)
    "__int__",  # (self, /)
    "__invert__",  # (self, /)
    "__le__",  # (self, other, /)
    "__len__",  # (self, /)
    "__lshift__",  # (self, other, /)
    "__lt__",  # (self, other, /)
    "__matmul__",  # (self, other, /)
    "__mod__",  # (self, other, /)
    "__mul__",  # (self, other, /)
    "__ne__",  # (self, other, /)
    "__neg__",  # (self, /)
    "__or__",  # (self, other, /)
    "__pos__",  # (self, /)
    "__pow__",  # (self, other, /)
    "__rshift__",  # (self, other, /)
    "__setitem__",  # (self, key, value, /)
    "__sub__",  # (self, other, /)
    "__truediv__",  # (self, other, /)
    "__xor__",  # (self, other, /)
]

aa_creators = [
    "arange",  # (start, /, stop=None, step=1, *, dtype=None, device=None)
    "asarray",  # (obj, /, *, dtype=None, device=None, copy=None)
    "empty",  # (shape, *, dtype=None, device=None)
    "empty_like",  # (x, /, *, dtype=None, device=None)
    "eye",  # (n_rows, n_cols=None, /, *, k=0, dtype=None, device=None)
    "from_dlpack",  # (x, /)
    "full",  # (shape, fill_value, *, dtype=None, device=None)
    "full_like",  # (x, /, fill_value, *, dtype=None, device=None)
    "linspace",  # (start, stop, /, num, *, dtype=None, device=None, endpoint=True)
    "meshgrid",  # (*arrays, indexing=’xy’)
    "ones",  # (shape, *, dtype=None, device=None)
    "ones_like",  # (x, /, *, dtype=None, device=None)
    "zeros",  # (shape, *, dtype=None, device=None)
    "zeros_like",  # (x, /, *, dtype=None, device=None)
]

aa_attributes = ["dtype", "device", "ndim", "shape", "size", "T"]

aa_methods_a = [
    "__abs__",  # (self, /)
    "__add__",  # (self, other, /)
    "__floordiv__",  # (self, other, /)
    "__invert__",  # (self, /)
    "__lshift__",  # (self, other, /)
    "__matmul__",  # (self, other, /)
    "__mod__",  # (self, other, /)
    "__mul__",  # (self, other, /)
    "__neg__",  # (self, /)
    "__pos__",  # (self, /)
    "__pow__",  # (self, other, /)
    "__rshift__",  # (self, other, /)
    "__sub__",  # (self, other, /)
    "__truediv__",  # (self, other, /)
    "__getitem__",  # (self, key, /)
    "__setitem__",  # (self, key, value, /)
    "__eq__",  # (self, other, /)
    "__ge__",  # (self, other, /)
    "__gt__",  # (self, other, /)
    "__le__",  # (self, other, /)
    "__lt__",  # (self, other, /)
    "__ne__",  # (self, other, /)
    "__and__",  # (self, other, /)
    "__or__",  # (self, other, /)
    "__xor__",  # (self, other, /)
]

aa_methods_s = [
    "__array_namespace__",  # (self, /, *, api_version=None)
    "__bool__",  # (self, /)
    "__dlpack__",  # (self, /, *, stream=None)
    "__dlpack_device__",  # (self, /)
    "__float__",  # (self, /)
    "__int__",  # (self, /)
    "__len__",  # (self, /)
]

aa_methods = aa_methods_s + aa_methods_a

aa_elementwises = [
    "abs",  # (x, /)
    "acos",  # (x, /)
    "acosh",  # (x, /)
    "add",  # (x1, x2, /)
    "asin",  # (x, /)
    "asinh",  # (x, /)
    "atan",  # (x, /)
    "atan2",  # (x1, x2, /)
    "atanh",  # (x, /)
    "bitwise_and",  # (x1, x2, /)
    "bitwise_left_shift",  # (x1, x2, /)
    "bitwise_invert",  # (x, /)
    "bitwise_or",  # (x1, x2, /)
    "bitwise_right_shift",  # (x1, x2, /)
    "bitwise_xor",  # (x1, x2, /)
    "ceil",  # (x, /)
    "cos",  # (x, /)
    "cosh",  # (x, /)
    "divide",  # (x1, x2, /)
    "equal",  # (x1, x2, /)
    "exp",  # (x, /)
    "expm1",  # (x, /)
    "floor",  # (x, /)
    "floor_divide",  # (x1, x2, /)
    "greater",  # (x1, x2, /)
    "greater_equal",  # (x1, x2, /)
    "isfinite",  # (x, /)
    "isinf",  # (x, /)
    "isnan",  # (x, /)
    "less",  # (x1, x2, /)
    "less_equal",  # (x1, x2, /)
    "log",  # (x, /)
    "log1p",  # (x, /)
    "log2",  # (x, /)
    "log10",  # (x, /)
    "logaddexp",  # (x1, x2)
    "logical_and",  # (x1, x2, /)
    "logical_not",  # (x, /)
    "logical_or",  # (x1, x2, /)
    "logical_xor",  # (x1, x2, /)
    "multiply",  # (x1, x2, /)
    "negative",  # (x, /)
    "not_equal",  # (x1, x2, /)
    "positive",  # (x, /)
    "pow",  # (x1, x2, /)
    "remainder",  # (x1, x2, /)
    "round",  # (x, /)
    "sign",  # (x, /)
    "sin",  # (x, /)
    "sinh",  # (x, /)
    "square",  # (x, /)
    "sqrt",  # (x, /)
    "subtract",  # (x1, x2, /)
    "tan",  # (x, /)
    "tanh",  # (x, /)
    "trunc",  # (x, /)
]

aa_statisticals = [
    "max",  # (x, /, *, axis=None, keepdims=False)
    "mean",  # (x, /, *, axis=None, keepdims=False)
    "min",  # (x, /, *, axis=None, keepdims=False)
    "prod",  # (x, /, *, axis=None, keepdims=False)
    "std",  # (x, /, *, axis=None, correction=0.0, keepdims=False)
    "sum",  # (x, /, *, axis=None, keepdims=False)
    "var",  # (x, /, *, axis=None, correction=0.0, keepdims=False)
]

aa_searching = ["argmax", "argmin", "nonzero", "where"]

aa_sorting = ["argsort", "sort"]

aa_set = ["unique"]

aa_utility = ["all", "any"]

aa_constants = ["e", "inf", "nan", "pi"]

aa_tlfuncs = (
    aa_creators
    + aa_elementwises
    + aa_statisticals
    + aa_datatype_functions
    + aa_searching
    + aa_sorting
    + aa_set
    + aa_utility
)
aa_tldir = aa_tlfuncs + aa_datatypes + aa_constants
aa_arrayfuncs = aa_methods + aa_inplace_operators + aa_reflected_operators
aa_arraydir = aa_attributes + aa_arrayfuncs
