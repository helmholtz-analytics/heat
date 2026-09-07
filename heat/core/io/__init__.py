"""Enables parallel I/O with data on disk."""

from typing import Any

from .._config import require_dependency

from . import io as _impl
from .io import *  # noqa: F403
from .io import __all__ as _io_all
from .csv import *  # noqa: F403
from .csv import __all__ as _csv_all
from .hdf5 import *  # noqa: F403
from .hdf5 import __all__ as _hdf5_all
from .netcdf import *  # noqa: F403
from .netcdf import __all__ as _netcdf_all
from .npy import *  # noqa: F403
from .npy import __all__ as _npy_all
from .zarr import *  # noqa: F403
from .zarr import __all__ as _zarr_all

__all__ = [*_io_all, *_csv_all, *_hdf5_all, *_netcdf_all, *_npy_all, *_zarr_all]


#: Third-party handles the monolithic io.py happened to expose as module globals, and
#: which the test suite reaches for. Kept resolvable while the formats migrate out.
_LEGACY_LIBRARY_ALIASES = {"h5py": "h5py", "nc": "netCDF4", "pd": "pandas"}


def __getattr__(name: str) -> Any:
    """
    Forward attribute lookups to the implementation module.

    Transitional shim for the split of the former monolithic ``heat/core/io.py`` into
    per-format modules. It keeps names that were never part of ``__all__`` -- such as
    ``ht.io.size_from_slice`` and the third-party handles ``ht.io.h5py`` / ``ht.io.nc``
    used by the test suite -- resolvable while functions migrate out of ``io.io``.
    """
    if name in _LEGACY_LIBRARY_ALIASES:
        return require_dependency(_LEGACY_LIBRARY_ALIASES[name])
    try:
        return getattr(_impl, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
