"""Reading Heat arrays from directories of NumPy ``.npy`` files."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .. import types
from ..communication import Communication
from ..dndarray import DNDarray
from ..types import datatype
from .utils import load_from_folder

__all__ = ["load_npy_from_path"]


def load_npy_from_path(
    path: str,
    dtype: datatype = types.int32,
    split: int = 0,
    device: Optional[str] = None,
    comm: Optional[Communication] = None,
) -> DNDarray:
    """
    Loads all ``.npy`` files in a directory into one DNDarray.

    The files are sorted by name and distributed over the processes, so every process
    reads only its own files. The arrays are then concatenated along ``split``, which is
    also the split axis of the result.

    Parameters
    ----------
    path : str
        Path to the directory in which the ``.npy`` files are located.
    dtype : datatype, optional
        Data type of the resulting array. Default is ``ht.int32``.
    split : int, optional
        Axis along which the loaded arrays are concatenated and distributed. Default is 0.
    device : str, optional
        The device id on which to place the data, defaults to globally set default device.
    comm : Communication, optional
        The communication to use for the data distribution, defaults to global default.

    Returns
    -------
    DNDarray
        The concatenated data, split along ``split``.

    Raises
    ------
    TypeError
        If ``path`` is not a string or ``split`` is not an integer.
    ValueError
        If ``path`` contains no ``.npy`` files.
    RuntimeError
        If there are more processes than files.

    See Also
    --------
    :func:`load_csv_from_folder` : Loads multiple .csv files into one DNDarray.
    :func:`load_multiple_hdf5` : Loads multiple HDF5 files into one DNDarray.

    Examples
    --------
    >>> import numpy as np
    >>> for i in range(4):
    ...     np.save(f"npy_dir/part_{i}.npy", np.arange(6 * i, 6 * (i + 1)).reshape(3, 2))
    >>> a = ht.load_npy_from_path("npy_dir", dtype=ht.float32, split=0)
    >>> a.shape
    [0/1] (12, 2)
    [1/1] (12, 2)
    >>> a.lshape
    [0/1] (6, 2)
    [1/1] (6, 2)
    """
    return load_from_folder(
        path,
        pattern="*.npy",
        reader=np.load,
        format_name=".npy",
        dtype=dtype,
        split=split,
        device=device,
        comm=comm,
    )
