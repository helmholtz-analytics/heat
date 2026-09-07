"""Enables parallel I/O with data on disk."""

from typing import Any

from . import io as _impl
from .io import *  # noqa: F403
from .io import __all__ as _io_all
from .npy import *  # noqa: F403
from .npy import __all__ as _npy_all

__all__ = [*_io_all, *_npy_all]


def __getattr__(name: str) -> Any:
    """
    Forward attribute lookups to the implementation module.

    Transitional shim for the split of the former monolithic ``heat/core/io.py`` into
    per-format modules. It keeps names that were never part of ``__all__`` -- such as
    ``ht.io.size_from_slice`` and the third-party handles ``ht.io.h5py`` / ``ht.io.nc``
    used by the test suite -- resolvable while functions migrate out of ``io.io``.
    """
    try:
        return getattr(_impl, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
