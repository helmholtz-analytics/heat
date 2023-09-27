import heat

numpy_functions = []

# List of numpy functions
headers = {"0": "NumPy  Mathematical Functions"}
numpy_mathematical_functions = [
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "hypot",
    "arctan2",
    "degrees",
    "radians",
    "unwrap",
    "deg2rad",
    "rad2deg",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "round",
    "around",
    "rint",
    "fix",
    "floor",
    "ceil",
    "trunc",
    "prod",
    "sum",
    "nanprod",
    "nansum",
    "cumprod",
    "cumsum",
    "nancumprod",
    "nancumsum",
    "diff",
    "ediff1d",
    "gradient",
    "cross",
    "trapz",
    "exp",
    "expm1",
    "exp2",
    "log",
    "log10",
    "log2",
    "log1p",
    "logaddexp",
    "logaddexp2",
    "i0",
    "sinc",
    "signbit",
    "copysign",
    "frexp",
    "ldexp",
    "nextafter",
    "spacing",
    "lcm",
    "gcd",
    "add",
    "reciprocal",
    "positive",
    "negative",
    "multiply",
    "divide",
    "power",
    "subtract",
    "true_divide",
    "floor_divide",
    "float_power",
    "fmod",
    "mod",
    "modf",
    "remainder",
    "divmod",
    "angle",
    "real",
    "imag",
    "conj",
    "conjugate",
    "maximum",
    "max",
    "amax",
    "fmax",
    "nanmax",
    "minimum",
    "min",
    "amin",
    "fmin",
    "nanmin",
    "convolve",
    "clip",
    "sqrt",
    "cbrt",
    "square",
    "absolute",
    "fabs",
    "sign",
    "heaviside",
    "nan_to_num",
    "real_if_close",
    "interp",
]
numpy_functions.append(numpy_mathematical_functions)

numpy_array_creation = [
    "empty",
    "empty_like",
    "eye",
    "identity",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
    "full",
    "full_like",
    "array",
    "asarray",
    "asanyarray",
    "ascontiguousarray",
    "asmatrix",
    "copy",
    "frombuffer",
    "from_dlpack",
    "fromfile",
    "fromfunction",
    "fromiter",
    "fromstring",
    "loadtxt",
    "arange",
    "linspace",
    "logspace",
    "geomspace",
    "meshgrid",
    "mgrid",
    "ogrid",
    "diag",
    "diagflat",
    "tri",
    "tril",
    "triu",
    "vander",
    "mat",
    "bmat",
]
numpy_functions.append(numpy_array_creation)
headers[str(len(headers))] = "NumPy Array Creation"

numpy_array_manipulation = [
    "copyto",
    "shape",
    "reshape",
    "ravel",
    "flat",
    "flatten",
    "moveaxis",
    "rollaxis",
    "swapaxes",
    "T",
    "transpose",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "broadcast",
    "broadcast_to",
    "broadcast_arrays",
    "expand_dims",
    "squeeze",
    "asarray",
    "asanyarray",
    "asmatrix",
    "asfarray",
    "asfortranarray",
    "ascontiguousarray",
    "asarray_chkfinite",
    "require",
    "concatenate",
    "stack",
    "block",
    "vstack",
    "hstack",
    "dstack",
    "column_stack",
    "row_stack",
    "split",
    "array_split",
    "dsplit",
    "hsplit",
    "vsplit",
    "tile",
    "repeat",
    "delete",
    "insert",
    "append",
    "resize",
    "trim_zeros",
    "unique",
    "flip",
    "fliplr",
    "flipud",
    "reshape",
    "roll",
    "rot90",
]
numpy_functions.append(numpy_array_manipulation)
headers[str(len(headers))] = "NumPy Array Manipulation"

numpy_binary_operations = [
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "invert",
    "left_shift",
    "right_shift",
    "packbits",
    "unpackbits",
    "binary_repr",
]
numpy_functions.append(numpy_binary_operations)
headers[str(len(headers))] = "NumPy Binary Operations"

numpy_io_operations = [
    # numpy.load
    # numpy.save
    # numpy.savez_compressed
    # numpy.loadtxt
    # numpy.savez
    # numpy.savetxt
    # numpy.genfromtxt
    # numpy.fromregex
    # numpy.fromstring
    # numpy.ndarray.tofile
    # numpy.ndarray.tolist
    # numpy.array2string
    # numpy.array_repr
    # numpy.array_str
    # numpy.format_float_positional
    # numpy.format_float_scientific
    # numpy.memmap
    # numpy.lib.format.open_memmap
    # numpy.set_printoptions
    # numpy.get_printoptions
    # numpy.set_string_function
    # numpy.printoptions
    # numpy.binary_repr
    # numpy.base_repr
    # numpy.DataSource
    # numpy.lib.format
    "load",
    "save",
    "savez",
    "savez_compressed",
    "loadtxt",
    "savetxt",
    "genfromtxt",
    "fromregex",
    "fromstring",
    "tofile",
    "tolist",
    "array2string",
    "array_repr",
    "array_str",
    "format_float_positional",
    "format_float_scientific",
    "memmap",
    "open_memmap",
    "set_printoptions",
    "get_printoptions",
    "set_string_function",
    "printoptions",
    "binary_repr",
    "base_repr",
    "DataSource",
    "format",
]
numpy_functions.append(numpy_io_operations)
headers[str(len(headers))] = "NumPy IO Operations"

numpy_linalg_operations = [
    # numpy.dot
    # numpy.linalg.multi_dot
    # numpy.vdot
    # numpy.inner
    # numpy.outer
    # numpy.matmul
    # numpy.tensordot
    # numpy.einsum
    # numpy.einsum_path
    # numpy.linalg.matrix_power
    # numpy.kron
    # numpy.linalg.cholesky
    # numpy.linalg.qr
    # numpy.linalg.svd
    # numpy.linalg.eig
    # numpy.linalg.eigh
    # numpy.linalg.eigvals
    # numpy.linalg.eigvalsh
    # numpy.linalg.norm
    # numpy.linalg.cond
    # numpy.linalg.det
    # numpy.linalg.matrix_rank
    # numpy.linalg.slogdet
    # numpy.trace
    # numpy.linalg.solve
    # numpy.linalg.tensorsolve
    # numpy.linalg.lstsq
    # numpy.linalg.inv
    # numpy.linalg.pinv
    # numpy.linalg.tensorinv
    "dot",
    "linalg.multi_dot",
    "vdot",
    "inner",
    "outer",
    "matmul",
    "tensordot",
    "einsum",
    "einsum_path",
    "linalg.matrix_power",
    "kron",
    "linalg.cholesky",
    "linalg.qr",
    "linalg.svd",
    "linalg.eig",
    "linalg.eigh",
    "linalg.eigvals",
    "linalg.eigvalsh",
    "linalg.norm",
    "linalg.cond",
    "linalg.det",
    "linalg.matrix_rank",
    "linalg.slogdet",
    "trace",
    "linalg.solve",
    "linalg.tensorsolve",
    "linalg.lstsq",
    "linalg.inv",
    "linalg.pinv",
    "linalg.tensorinv",
]
numpy_functions.append(numpy_linalg_operations)
headers[str(len(headers))] = "NumPy LinAlg Operations"

numpy_logic_operations = [
    # numpy.all
    # numpy.any
    # numpy.isinf
    # numpy.isfinite
    # numpy.isnan
    # numpy.isnat
    # numpy.isneginf
    # numpy.isposinf
    # numpy.iscomplex
    # numpy.iscomplexobj
    # numpy.isfortran
    # numpy.isreal
    # numpy.isrealobj
    # numpy.isscalar
    # numpy.logical_and
    # numpy.logical_or
    # numpy.logical_not
    # numpy.logical_xor
    # numpy.allclose
    # numpy.isclose
    # numpy.array_equal
    # numpy.array_equiv
    # numpy.greater
    # numpy.greater_equal
    # numpy.less
    # numpy.less_equal
    # numpy.equal
    # numpy.not_equal
    "all",
    "any",
    "isfinite",
    "isinf",
    "isnan",
    "isnat",
    "isneginf",
    "isposinf",
    "iscomplex",
    "iscomplexobj",
    "isfortran",
    "isreal",
    "isrealobj",
    "isscalar",
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_xor",
    "allclose",
    "isclose",
    "array_equal",
    "array_equiv",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "equal",
    "not_equal",
]
numpy_functions.append(numpy_logic_operations)
headers[str(len(headers))] = "NumPy Logic Functions"

numpy_sorting_operations = [
    # numpy.sort
    # numpy.lexsort
    # numpy.argsort
    # numpy.ndarray.sort
    # numpy.sort_complex
    # numpy.partition
    # numpy.argpartition
    # numpy.argmax
    # numpy.nanargmax
    # numpy.argmin
    # numpy.nanargmin
    # numpy.argwhere
    # numpy.nonzero
    # numpy.flatnonzero
    # numpy.where
    # numpy.searchsorted
    # numpy.extract
    # numpy.count_nonzero
    "sort",
    "lexsort",
    "argsort",
    "sort",
    "sort_complex",
    "partition",
    "argpartition",
    "argmax",
    "nanargmax",
    "argmin",
    "nanargmin",
    "argwhere",
    "nonzero",
    "flatnonzero",
    "where",
    "searchsorted",
    "extract",
    "count_nonzero",
]
numpy_functions.append(numpy_sorting_operations)
headers[str(len(headers))] = "NumPy Sorting Operations"

numpy_statistics_operations = [
    # numpy.ptp
    # numpy.percentile
    # numpy.nanpercentile
    # numpy.quantile
    # numpy.nanquantile
    # numpy.median
    # numpy.average
    # numpy.mean
    # numpy.std
    # numpy.var
    # numpy.nanmedian
    # numpy.nanmean
    # numpy.nanstd
    # numpy.nanvar
    # numpy.corrcoef
    # numpy.correlate
    # numpy.cov
    # numpy.histogram
    # numpy.histogram2d
    # numpy.histogramdd
    # numpy.bincount
    # numpy.histogram_bin_edges
    # numpy.digitize
    "ptp",
    "percentile",
    "nanpercentile",
    "quantile",
    "nanquantile",
    "median",
    "average",
    "mean",
    "std",
    "var",
    "nanmedian",
    "nanmean",
    "nanstd",
    "nanvar",
    "corrcoef",
    "correlate",
    "cov",
    "histogram",
    "histogram2d",
    "histogramdd",
    "bincount",
    "histogram_bin_edges",
    "digitize",
]
numpy_functions.append(numpy_statistics_operations)
headers[str(len(headers))] = "NumPy Statistical Operations"

# initialize markdown file
# open the file in write mode
f = open("numpy_coverage_tables.md", "w")
# write in file
f.write("# NumPy Coverage Tables\n")
f.write("This file is automatically generated by `./scripts/numpy_coverage_tables.py`.\n")
f.write(
    "Please do not edit this file directly, but instead edit `./scripts/numpy_coverage_tables.py` and run it to generate this file.\n"
)
f.write("The following tables show the NumPy functions supported by Heat.\n")

# create Table of Contents
f.write("## Table of Contents\n")
for i, header in enumerate(headers):
    f.write(f"{i+1}. [{headers[header]}](#{headers[header].lower().replace(' ', '-')})\n")
f.write("\n")

for i, function_list in enumerate(numpy_functions):
    f.write(f"## {headers[str(i)]}\n")
    # Initialize a list to store the rows of the Markdown table
    table_rows = []

    # Check if functions exist in the heat library and create table rows
    for func_name in function_list:
        if hasattr(heat, func_name):
            support_status = "✅"  # Green checkmark for supported functions
        else:
            support_status = "❌"  # Red cross for unsupported functions

        table_row = f"| {func_name} | {support_status} |"
        table_rows.append(table_row)

    # Create the Markdown table header
    table_header = f"| {headers[str(i)]} | Heat |\n|---|---|\n"

    # Combine the header and table rows
    markdown_table = table_header + "\n".join(table_rows)

    # write link to table of contents
    f.write("[Back to Table of Contents](#table-of-contents)\n\n")
    # Print the Markdown table
    f.write(markdown_table)
    f.write("\n")
