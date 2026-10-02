"""Maps file extensions onto the functions that read and write them.

Each format module registers itself here, so that :func:`heat.core.io.io.load` and
:func:`heat.core.io.io.save` can dispatch on the file extension without knowing about
any format in particular. Formats register even when the third-party library they need
is missing -- with ``loader``/``saver`` left as ``None`` -- so that loading a recognized
extension reports the missing dependency rather than claiming the extension is unknown.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, FrozenSet, Iterable, Optional

from .._config import OPTIONAL_DEPENDENCIES, is_installed, require_dependency
from .utils import extension_of

__all__ = []


@dataclasses.dataclass(frozen=True)
class FormatSpec:
    """
    Describes one file format Heat can read and/or write.

    Attributes
    ----------
    name : str
        Short name of the format, e.g. ``"hdf5"``.
    extensions : FrozenSet[str]
        File extensions claimed by this format, including the leading dot.
    dependency : str, optional
        Import name of the third-party package the format needs, if any.
    loader : Callable, optional
        Reads a file of this format. ``None`` if the format cannot be read, or if its
        dependency is unavailable.
    saver : Callable, optional
        Writes a file of this format. ``None`` if the format cannot be written, or if its
        dependency is unavailable.
    """

    name: str
    extensions: FrozenSet[str]
    dependency: Optional[str] = None
    loader: Optional[Callable] = None
    saver: Optional[Callable] = None


_REGISTRY: Dict[str, FormatSpec] = {}


def register_format(
    name: str,
    extensions: Iterable[str],
    dependency: Optional[str] = None,
    loader: Optional[Callable] = None,
    saver: Optional[Callable] = None,
) -> FormatSpec:
    """
    Registers a file format, and returns the resulting specification.

    Parameters
    ----------
    name : str
        Short name of the format, e.g. ``"hdf5"``.
    extensions : Iterable[str]
        File extensions claimed by this format, including the leading dot.
    dependency : str, optional
        Import name of the third-party package the format needs, if any.
    loader : Callable, optional
        Reads a file of this format.
    saver : Callable, optional
        Writes a file of this format.

    Raises
    ------
    ValueError
        If one of the extensions is already claimed by a different format.
    """
    spec = FormatSpec(name, frozenset(extensions), dependency, loader, saver)
    for other in _REGISTRY.values():
        if other.name != name and other.extensions & spec.extensions:
            raise ValueError(
                f"cannot register format {name!r}: extensions "
                f"{sorted(other.extensions & spec.extensions)} are already claimed by "
                f"format {other.name!r}"
            )
    _REGISTRY[name] = spec
    return spec


def registered_formats() -> Dict[str, FormatSpec]:
    """Returns a copy of the format registry, keyed by format name."""
    return dict(_REGISTRY)


def registered_extensions() -> FrozenSet[str]:
    """Returns every file extension claimed by a registered format."""
    return (
        frozenset().union(*(spec.extensions for spec in _REGISTRY.values()))
        if _REGISTRY
        else frozenset()
    )


def supports(name: str) -> bool:
    """
    Returns ``True`` if ``name`` is usable, ``False`` otherwise.

    ``name`` is either a registered format (``"hdf5"``, ``"netcdf"``, ``"zarr"``,
    ``"csv"``, ...) or the import name of one of Heat's optional dependencies
    (``"h5py"``, ``"pandas"``, ...). This supersedes the per-format ``supports_*()``
    functions.

    Parameters
    ----------
    name : str
        Name of a registered format or of an optional dependency.

    Raises
    ------
    ValueError
        If ``name`` is neither.

    Examples
    --------
    >>> ht.io.supports("hdf5")
    True
    """
    spec = _REGISTRY.get(name)
    if spec is not None:
        return spec.loader is not None or spec.saver is not None
    if name in OPTIONAL_DEPENDENCIES:
        return is_installed(name)
    raise ValueError(
        f"unknown format or dependency {name!r}; known formats are {sorted(_REGISTRY)}"
    )


def available_formats() -> Dict[str, bool]:
    """
    Returns every registered format, mapped to whether it is currently usable.

    Examples
    --------
    >>> ht.io.available_formats()
    {'csv': True, 'hdf5': True, 'netcdf': True, 'zarr': False}
    """
    return {name: supports(name) for name in _REGISTRY}


def spec_for_path(path: str) -> FormatSpec:
    """
    Returns the format registered for the extension of ``path``.

    Parameters
    ----------
    path : str
        Path whose extension selects the format.

    Raises
    ------
    ValueError
        If no registered format claims the extension.
    """
    extension = extension_of(path)
    for spec in _REGISTRY.values():
        if extension in spec.extensions:
            return spec
    raise ValueError(f"Unsupported file extension {extension}")


def _resolve(path: str, attribute: str, verb: str) -> Callable:
    """
    Returns the loader or saver for ``path``, raising if it is unavailable.

    Parameters
    ----------
    path : str
        Path whose extension selects the format.
    attribute : str
        Either ``"loader"`` or ``"saver"``.
    verb : str
        Either ``"loading"`` or ``"saving"``, used in the error message.

    Raises
    ------
    ValueError
        If no registered format claims the extension, or the format does not support
        the requested direction.
    RuntimeError
        If the format's optional dependency is unavailable.
    """
    spec = spec_for_path(path)
    function = getattr(spec, attribute)
    if function is None:
        if spec.dependency is not None:
            # raises RuntimeError naming the package and the pip extra providing it
            require_dependency(spec.dependency)
        raise ValueError(f"the {spec.name} format does not support {verb}")
    return function


def loader_for_path(path: str) -> Callable:
    """
    Returns the function that loads files with the extension of ``path``.

    Parameters
    ----------
    path : str
        Path whose extension selects the format.
    """
    return _resolve(path, "loader", "loading")


def saver_for_path(path: str) -> Callable:
    """
    Returns the function that saves files with the extension of ``path``.

    Parameters
    ----------
    path : str
        Path whose extension selects the format.
    """
    return _resolve(path, "saver", "saving")
