"""Reading and writing Heat arrays as comma-separated values.

The optional ``pandas`` dependency, needed only by :func:`load_csv_from_folder`, is
imported lazily through :func:`heat.core._config.require_dependency` rather than at module
level, so that ``import heat`` does not pay for it.
"""

from __future__ import annotations

import warnings

import os
from functools import reduce
from math import log10
from typing import Any, Callable, Iterable, List, Optional, Type, Union

import numpy as np
import torch

from .. import devices
from .. import factories
from .. import types
from .._config import is_installed, require_dependency
from ..communication import Communication, MPI, sanitize_comm
from ..devices import Device
from ..dndarray import DNDarray
from ..statistics import max as smax, min as smin
from ..types import datatype
from ._registry import register_format
from .utils import load_from_folder

__all__ = ["load_csv", "save_csv", "supports_pandas"]

#: File extensions handled by this module.
EXTENSIONS = frozenset([".csv"])


def load_csv(
    path: str,
    header_lines: int = 0,
    sep: str = ",",
    dtype: Type[datatype] = types.float32,
    encoding: str = "utf-8",
    split: Optional[int] = None,
    device: Optional[Union[str, Device]] = None,
    comm: Optional[Communication] = None,
) -> DNDarray:
    """
    Loads data from a CSV file.

    The data may be distributed among multiple processing nodes via the ``split`` flag.
    With ``split=0`` every process reads only its own portion of the file.

    Parameters
    ----------
    path : str
        Path to the CSV file to be read.
    header_lines : int, optional
        The number of lines at the beginning of the file that should not be considered as data.
    sep : str, optional
        The single ``char`` or ``str`` that separates the values in each row.
    dtype : datatype, optional
        Data type of the resulting array.
    encoding : str, optional
        The type of encoding which will be used to interpret the lines of the csv file as strings.
    split : int or None, optional
        Along which axis the resulting array should be split, one of ``None``, 0 or 1.
        Default is ``None`` which means each node will have the full array.
    device : str or Device, optional
        The device id on which to place the data, defaults to globally set default device.
    comm : Communication, optional
        The communication to use for the data distribution, defaults to global default.

    Returns
    -------
    DNDarray
        The data read from the file.

    Raises
    ------
    TypeError
        If any of the input parameters are not of correct type.
    ValueError
        If ``split`` is not one of ``None``, 0 or 1.

    See Also
    --------
    :func:`save_csv` : Saves data to a CSV file.
    :func:`load_csv_from_folder` : Loads multiple .csv files into one DNDarray.

    Examples
    --------
    >>> a = ht.load_csv("heat/datasets/iris.csv", sep=";", split=0)
    >>> a.shape
    [0/3] (150, 4)
    [1/3] (150, 4)
    [2/3] (150, 4)
    [3/3] (150, 4)
    >>> a.lshape
    [0/3] (38, 4)
    [1/3] (38, 4)
    [2/3] (37, 4)
    [3/3] (37, 4)
    >>> b = ht.load_csv("heat/datasets/iris.csv", sep=";", header_lines=10, split=0)
    >>> b.shape
    [0/3] (140, 4)
    [1/3] (140, 4)
    [2/3] (140, 4)
    [3/3] (140, 4)
    >>> b.lshape
    [0/3] (35, 4)
    [1/3] (35, 4)
    [2/3] (35, 4)
    [3/3] (35, 4)
    """
    if not isinstance(path, str):
        raise TypeError(f"path must be str, not {type(path)}")
    if not isinstance(sep, str):
        raise TypeError(f"separator must be str, not {type(sep)}")
    if not isinstance(header_lines, int):
        raise TypeError(f"header_lines must int, not {type(header_lines)}")
    if split not in [None, 0, 1]:
        raise ValueError(f"split must be in [None, 0, 1], but is {split}")

    # infer the type and communicator for the loaded array
    dtype = types.canonical_heat_type(dtype)
    # determine the comm and device the data will be placed on
    device = devices.sanitize_device(device)
    comm = sanitize_comm(comm)

    file_size = os.stat(path).st_size
    rank = comm.rank
    size = comm.size

    if split is None:
        with open(path) as f:
            data = f.readlines()
            data = data[header_lines:]
            result: List[List[float]] = []
            for line in data:
                values = line.replace("\n", "").replace("\r", "").split(sep)
                result.append([float(val) for val in values])
            resulting_tensor = factories.array(
                result, dtype=dtype, split=split, device=device, comm=comm
            )

    elif split == 0:
        counts, displs, _ = comm.counts_displs_shape((file_size, 1), 0)
        # in case lines are terminated with '\r\n' we need to skip 2 bytes later
        lineter_len = 1
        # Read a chunk of bytes and count the linebreaks
        with open(path, "rb") as f:
            f.seek(displs[rank], 0)
            line_starts: List[int] = []
            r = f.read(counts[rank])
            for pos, byte in enumerate(r):
                if chr(byte) == "\n":
                    # Check if it is part of '\r\n'
                    if chr(r[pos - 1]) != "\r":
                        line_starts.append(pos + 1)
                elif chr(byte) == "\r":
                    # check if file line is terminated by '\r\n'
                    if pos + 1 < len(r) and chr(r[pos + 1]) == "\n":
                        line_starts.append(pos + 2)
                        lineter_len = 2
                    else:
                        line_starts.append(pos + 1)

            if rank == 0:
                line_starts = [0] + line_starts

            # Find the correct starting point
            total_lines = torch.empty(size, dtype=torch.int32)
            comm.Allgather(torch.tensor([len(line_starts)], dtype=torch.int32), total_lines)

            cumsum = total_lines.cumsum(dim=0).tolist()
            start = next(i for i in range(size) if cumsum[i] > header_lines)
            if rank < start:
                line_starts = []
            if rank == start:
                rem = header_lines - (0 if start == 0 else cumsum[start - 1])
                line_starts = line_starts[rem:]

            # Determine the number of columns that each line consists of
            if len(line_starts) > 1:
                columns = 1
                for li in r[line_starts[0] : line_starts[1]]:
                    if chr(li) == sep:
                        columns += 1
            else:
                columns = 0

            max_columns = torch.tensor([columns], dtype=torch.int32)
            comm.Allreduce(MPI.IN_PLACE, max_columns, MPI.MAX)
            columns = int(max_columns.item())

            # Share how far the processes need to reed in their last line
            last_line = file_size
            if size - start > 1:
                if rank == start:
                    next_first_line = torch.empty(1, dtype=torch.int32)
                    comm.Recv(next_first_line, source=rank + 1)
                    last_line = int(next_first_line.item())
                elif rank == size - 1:
                    first_line = torch.tensor(displs[rank] + line_starts[0] - 1, dtype=torch.int32)
                    comm.Send(first_line, dest=rank - 1)
                elif start < rank < size - 1:
                    next_first_line = torch.empty(1, dtype=torch.int32)
                    first_line = torch.tensor(displs[rank] + line_starts[0] - 1, dtype=torch.int32)
                    comm.Send(first_line, dest=rank - 1)
                    comm.Recv(next_first_line, source=rank + 1)
                    last_line = int(next_first_line.item())

            # Create empty tensor and iteratively fill it with the values
            local_shape = (len(line_starts), columns)
            actual_length = 0
            local_tensor = torch.empty(
                local_shape, dtype=dtype.torch_type(), device=device.torch_device
            )

            for ind, start in enumerate(line_starts):
                if ind == len(line_starts) - 1:
                    f.seek(displs[rank] + start, 0)
                    raw_line = f.read(last_line - displs[rank] - start)
                else:
                    raw_line = r[start : line_starts[ind + 1] - lineter_len]
                # Decode byte array
                line = raw_line.decode(encoding)
                if len(line) > 0:
                    sep_values = [float(val) for val in line.split(sep)]
                    local_tensor[actual_length] = torch.tensor(sep_values, dtype=dtype.torch_type())
                    actual_length += 1

        # In case there are some empty lines in the csv file
        local_tensor = local_tensor[:actual_length]

        total_actual_lines = torch.tensor(
            actual_length, dtype=torch.int64, device=local_tensor.device
        )
        comm.Allreduce(MPI.IN_PLACE, total_actual_lines, MPI.SUM)

        gshape = (int(total_actual_lines.item()), columns)

        resulting_tensor = DNDarray(
            local_tensor,
            gshape=gshape,
            dtype=dtype,
            split=0,
            device=device,
            comm=comm,
            balanced=None,
        )
        resulting_tensor.balance_()

    elif split == 1:
        local_rows: List[List[float]] = []

        with open(path) as f:
            for i in range(header_lines):
                f.readline()
            line = f.readline()
            row = [float(val) for val in line.replace("\n", "").replace("\r", "").split(sep)]
            rows = len(row)

            chunk, displs, _ = comm.counts_displs_shape((1, rows), 1)
            local_rows.append(row[displs[rank] : displs[rank] + chunk[rank]])
            # Read file line by line till EOF reached
            for line in iter(f.readline, ""):
                row = [float(val) for val in line.replace("\n", "").replace("\r", "").split(sep)]
                local_rows.append(row[displs[rank] : displs[rank] + chunk[rank]])
        resulting_tensor = factories.array(
            local_rows, dtype=dtype, is_split=1, device=device, comm=comm
        )

    return resulting_tensor


def save_csv(
    data: DNDarray,
    path: str,
    header_lines: Optional[Iterable[str]] = None,
    sep: str = ",",
    decimals: int = -1,
    encoding: str = "utf-8",
    truncate: bool = True,
) -> None:
    """
    Saves data to a CSV file.

    Only 1D and 2D data is supported, with any split axis. All processes write their part
    of the data to the file in parallel through MPI-IO.

    Parameters
    ----------
    data : DNDarray
        The DNDarray to be saved to CSV.
    path : str
        Path to the CSV file to be written.
    header_lines : Iterable[str], optional
        Lines to prepend at the beginning of the file. No pound sign or any other comment
        marker will be inserted.
    sep : str, optional
        The separator character used in this CSV.
    decimals : int, optional
        Number of digits after the decimal point. Ignored for integer data. The default,
        -1, writes 7 digits for ``float32`` and 15 digits for ``float64``.
    encoding : str, optional
        The encoding used to write the header lines.
    truncate : bool, optional
        Whether to truncate an existing file before writing, i.e. fully overwrite it.
        The sane default is True. Setting it to False will not shorten files if
        needed and thus may leave garbage at the end of existing files.

    Raises
    ------
    TypeError
        If any of the input parameters are not of correct type.
    ValueError
        If the split axis of ``data`` is not one of ``None``, 0 or 1.

    See Also
    --------
    :func:`load_csv` : Loads data from a CSV file.

    Examples
    --------
    >>> x = ht.arange(6, split=0).reshape((3, 2), new_split=0)
    >>> ht.save_csv(x, "data.csv", header_lines=["a,b"])
    >>> print(open("data.csv").read(), end="")
    a,b
    0,1
    2,3
    4,5
    """
    if not isinstance(path, str):
        raise TypeError(f"path must be str, not {type(path)}")
    if not isinstance(sep, str):
        raise TypeError(f"separator must be str, not {type(sep)}")
    # check this to allow None
    if not isinstance(header_lines, Iterable) and header_lines is not None:
        raise TypeError(f"header_lines must Iterable[str], not {type(header_lines)}")
    if data.split not in [None, 0, 1]:
        raise ValueError(f"split must be in [None, 0, 1], but is {data.split}")

    if os.path.exists(path) and truncate:
        if data.comm.rank == 0:
            os.truncate(path, 0)
        # avoid truncating and writing at the same time
        data.comm.handle.Barrier()

    amode = MPI.MODE_WRONLY | MPI.MODE_CREATE
    csv_out = MPI.File.Open(data.comm.handle, path, amode)

    # will be needed as an additional offset later
    hl_displacement = 0
    if header_lines is not None:
        hl_displacement = sum(len(hl) for hl in header_lines)
        # count additions everywhere, but write only on rank 0, avoiding reduce op to share final hl_displacement
        for hl in header_lines:
            if not hl.endswith("\n"):
                hl = hl + "\n"
                hl_displacement = hl_displacement + 1
            if data.comm.rank == 0 and header_lines:
                csv_out.Write(hl.encode(encoding))

    # formatting and element width
    data_min = smin(data).item()  # at least min is used twice, so cache it here
    data_max = smax(data).item()
    sign = 1 if data_min < 0 else 0
    if abs(data_max) > 0 or abs(data_min) > 0:
        pre_point_digits = int(log10(max(abs(data_max), abs(data_min)))) + 1
    else:
        pre_point_digits = 1

    dec_sep = 1
    fmt = ""
    if types.issubdtype(data.dtype, types.integer):
        decimals = 0
        dec_sep = 0
        if sign == 1:
            fmt = "%%%-dd" % (pre_point_digits + 1)
        else:
            fmt = "%%%dd" % (pre_point_digits)
    elif types.issubdtype(data.dtype, types.floating):
        if decimals == -1:
            decimals = 7 if data.dtype is types.float32 else 15
        if sign == 1:
            fmt = "%%%-d.%df" % (pre_point_digits + decimals + 2, decimals)
        else:
            fmt = "%%%d.%df" % (pre_point_digits + decimals + 1, decimals)

    # sign + decimal separator + pre separator digits + decimals (post separator)
    item_size = decimals + dec_sep + sign + pre_point_digits
    # each item is one position larger than its representation, either b/c of separator or line break
    row_width = item_size + 1
    if len(data.shape) > 1:
        row_width = data.shape[1] * (item_size + 1)

    offset = hl_displacement  # all splits
    if data.split == 0:
        _, displs = data.counts_displs()
        offset = offset + displs[data.comm.rank] * row_width
    elif data.split == 1:
        _, displs = data.counts_displs()
        offset = offset + displs[data.comm.rank] * (item_size + 1)

    for i in range(data.lshape[0]):
        # if lshape is of the form (x,), then there will only be a single element per row
        if len(data.lshape) == 1:
            row = fmt % (data.larray[i])
        else:
            if data.lshape[1] == 0:
                break
            row = sep.join(fmt % (item) for item in data.larray[i])

        if (
            data.split is None
            or data.split == 0
            or displs[data.comm.rank] + data.lshape[1] == data.shape[1]
        ):
            row = row + "\n"
        else:
            row = row + sep

        if data.split is not None or data.comm.rank == 0:
            csv_out.Write_at(offset, row.encode("utf-8"))

        offset = offset + row_width

    csv_out.Close()
    data.comm.handle.Barrier()


_AVAILABLE = is_installed("pandas")


def supports_pandas() -> bool:
    """
    Returns ``True`` if pandas is installed, ``False`` otherwise.

    .. deprecated:: 1.10.0
        Use :func:`heat.io.supports` instead, e.g. ``ht.io.supports("pandas")``.

    Returns
    -------
    bool
        Whether pandas is installed.
    """
    warnings.warn(
        "supports_pandas() is deprecated and will be removed in a future release; "
        "use ht.io.supports('pandas') instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return _AVAILABLE


if _AVAILABLE:
    __all__.append("load_csv_from_folder")

    def load_csv_from_folder(
        path: str,
        dtype: Type[datatype] = types.int32,
        split: int = 0,
        device: Optional[Union[str, Device]] = None,
        comm: Optional[Communication] = None,
        func: Optional[Callable[[Any], Any]] = None,
    ) -> DNDarray:
        """
        Loads all ``.csv`` files in a directory into one DNDarray.

        The files are sorted by name and distributed over the processes, so every process
        reads only its own files with :func:`pandas.read_csv`. The arrays are then
        concatenated along ``split``, which is also the split axis of the result. Requires
        pandas.

        Parameters
        ----------
        path : str
            Path to the directory in which the ``.csv`` files are located.
        dtype : datatype, optional
            Data type of the resulting array. Default is ``ht.int32``.
        split : int, optional
            Axis along which the loaded arrays are concatenated and distributed. Default is 0.
        device : str or Device, optional
            The device id on which to place the data, defaults to globally set default device.
        comm : Communication, optional
            The communication to use for the data distribution, defaults to global default.
        func : Callable, optional
            Applied to the :class:`pandas.DataFrame` read from each file before it is added
            to the array; must return a ``DataFrame``.

        Returns
        -------
        DNDarray
            The concatenated data, split along ``split``.

        Raises
        ------
        TypeError
            If ``func`` is not callable, ``path`` is not a string or ``split`` is not an
            integer.
        ValueError
            If ``path`` contains no ``.csv`` files.
        RuntimeError
            If there are more processes than files.

        See Also
        --------
        :func:`load_csv` : Loads data from a single CSV file.
        :func:`load_npy_from_path` : Loads multiple .npy files into one DNDarray.

        Examples
        --------
        Assuming ``csv_dir`` holds four files, each with a header line ``a,b`` and three
        rows of data:

        >>> a = ht.load_csv_from_folder("csv_dir", split=0)
        >>> a.shape
        [0/1] (12, 2)
        [1/1] (12, 2)
        >>> b = ht.load_csv_from_folder("csv_dir", split=0, func=lambda df: df[["b"]])
        >>> b.shape
        [0/1] (12, 1)
        [1/1] (12, 1)
        """
        if (func is not None) and not callable(func):
            raise TypeError("func needs to be a callable function or None")

        pandas = require_dependency("pandas")

        def reader(file_path: str) -> np.ndarray:
            frame = pandas.read_csv(file_path)
            return np.asarray(func(frame) if func is not None else frame)

        return load_from_folder(
            path,
            pattern="*.csv",
            reader=reader,
            format_name=".csv",
            dtype=dtype,
            split=split,
            device=device,
            comm=comm,
        )


register_format("csv", EXTENSIONS, loader=load_csv, saver=save_csv)
