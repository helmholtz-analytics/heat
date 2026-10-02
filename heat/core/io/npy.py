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
    Loads multiple .npy files into one DNDarray which will be returned. The data will be concatenated along the split axis provided as input.

    Parameters
    ----------
    path : str
        Path to the directory in which .npy-files are located.
    dtype : datatype, optional
        Data type of the resulting array.
    split : int
        Along which axis the loaded arrays should be concatenated.
    device : str, optional
        The device id on which to place the data, defaults to globally set default device.
    comm : Communication, optional
        The communication to use for the data distribution, default is 'heat.MPI_WORLD'
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
