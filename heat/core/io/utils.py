"""Helpers shared by the file-format specific modules of :mod:`heat.core.io`."""

from __future__ import annotations

import fnmatch
import os
from typing import Any, Callable, Iterable, Optional, Tuple

import numpy as np
import torch

from .. import devices
from .. import factories
from ..communication import Communication, MPI, MPI_WORLD, sanitize_comm
from ..devices import Device
from ..dndarray import DNDarray
from ..types import datatype

__all__ = []

#: File access modes accepted by the writing functions.
VALID_WRITE_MODES = frozenset(["w", "a", "r+"])


def size_from_slice(size: int, s: slice) -> Tuple[int, int]:
    """
    Determines the size of a slice object.

    Parameters
    ----------
    size: int
        The size of the array the slice object is applied to.
    s : slice
        The slice object to determine the size of.

    Returns
    -------
    int
        The size of the sliced object.
    int
        The start index of the slice object.
    """
    new_range = range(size)[s]
    return len(new_range), new_range.start if len(new_range) > 0 else 0


def compose_slices(outer: slice, inner: slice, length: int) -> slice:
    """
    Returns the single slice equivalent to applying ``inner`` after ``outer``.

    That is, for any sequence ``x`` of length ``length``,
    ``x[compose_slices(outer, inner, length)] == x[outer][inner]``. This lets a
    distributed read fetch only the process-local region from disk, instead of
    materializing the whole user-requested slice on every process first.

    Parameters
    ----------
    outer : slice
        The slice applied first, relative to an axis of length ``length``.
    inner : slice
        The slice applied second, relative to the result of ``outer``.
    length : int
        Length of the axis ``outer`` applies to.
    """
    start, stop, step = outer.indices(length)
    inner_start, inner_stop, inner_step = inner.indices(len(range(start, stop, step)))

    count = len(range(inner_start, inner_stop, inner_step))
    if count == 0:
        return slice(0, 0)

    composed_step = step * inner_step
    composed_start = start + inner_start * step
    composed_stop = composed_start + count * composed_step
    if composed_stop < 0:
        # a negative stop would wrap around to the end; the walk runs off the front
        composed_stop = None
    return slice(composed_start, composed_stop, composed_step)


def sanitize_path(path: Any, argname: str = "path") -> str:
    """
    Returns ``path`` unchanged, raising if it is not a string.

    Parameters
    ----------
    path : Any
        The path to validate.
    argname : str, optional
        Name of the argument to use in the error message.

    Raises
    ------
    TypeError
        If ``path`` is not a string.
    """
    if not isinstance(path, str):
        raise TypeError(f"{argname} must be str, not {type(path)}")
    return path


def extension_of(path: str) -> str:
    """
    Returns the normalized file extension of ``path``, e.g. ``".h5"``.

    Parameters
    ----------
    path : str
        The path to inspect.
    """
    return os.path.splitext(path)[-1].strip().lower()


def sanitize_extension(path: str, extensions: Iterable[str], format_name: str) -> str:
    """
    Returns ``path`` unchanged, raising if its extension is not one of ``extensions``.

    Parameters
    ----------
    path : str
        The path to validate.
    extensions : Iterable[str]
        The accepted extensions, including the leading dot.
    format_name : str
        Human-readable name of the file format, used in the error message.

    Raises
    ------
    ValueError
        If the extension of ``path`` is not accepted.
    """
    if extension_of(path) not in frozenset(extensions):
        raise ValueError(f"File has no {format_name} extension, expected one of {set(extensions)}")
    return path


def sanitize_write_mode(mode: str) -> str:
    """
    Returns ``mode`` unchanged, raising if it is not a supported write mode.

    Parameters
    ----------
    mode : str
        File access mode, one of ``'w'``, ``'a'``, ``'r+'``.

    Raises
    ------
    ValueError
        If the access mode is not understood.
    """
    if mode not in VALID_WRITE_MODES:
        raise ValueError(f"mode was {mode}, not in possible modes {VALID_WRITE_MODES}")
    return mode


def resolve_device_comm(
    device: Optional[str], comm: Optional[Communication]
) -> Tuple[Device, Communication]:
    """
    Resolves the device and communicator a loaded array will be placed on.

    Parameters
    ----------
    device : str, optional
        The device id on which to place the data, defaults to the globally set default.
    comm : Communication, optional
        The communication to use, defaults to the global default.
    """
    return devices.sanitize_device(device), sanitize_comm(comm)


def local_write_slices(data: DNDarray) -> Tuple[bool, Tuple[slice, ...]]:
    """
    Returns ``(is_split, slices)``, the region of ``data`` this process is to write.

    If ``data`` is not split, axis 0 is chunked anyway so that the writing processes can
    still share the work.

    Parameters
    ----------
    data : DNDarray
        The array about to be written.
    """
    is_split = data.split is not None
    _, _, slices = data.comm.chunk(data.gshape, data.split if is_split else 0)
    return is_split, slices


def local_write_payload(data: DNDarray, slices: Tuple[slice, ...]) -> torch.Tensor:
    """
    Returns the process-local chunk of ``data`` to write, as a CPU tensor.

    Parameters
    ----------
    data : DNDarray
        The array about to be written.
    slices : Tuple[slice, ...]
        The region assigned to this process, as returned by :func:`local_write_slices`.
    """
    is_split = data.split is not None
    return data.larray.cpu() if is_split else data.larray[slices].cpu()


def serialized_write(
    comm: Communication,
    is_split: bool,
    write_root: Callable[[], None],
    write_other: Callable[[], None],
) -> None:
    """
    Writes a file one process at a time, for backends without parallel I/O support.

    Rank 0 creates the file and writes its own chunk, then passes a token around the ring
    so that every other process appends its chunk in turn. If any process raises, the
    failure is propagated to all of them rather than leaving the ring deadlocked.

    Parameters
    ----------
    comm : Communication
        The communicator whose processes participate in the write.
    is_split : bool
        Whether the data is distributed. If it is not, only rank 0 writes anything.
    write_root : Callable[[], None]
        Creates the file and writes rank 0's chunk. Only called on rank 0.
    write_other : Callable[[], None]
        Opens the existing file and writes this process' chunk. Called on all other ranks.

    Raises
    ------
    Exception
        Whatever ``write_root`` or ``write_other`` raised, on every process.
    """
    failed = 0
    excep = None

    if comm.rank == 0:
        try:
            write_root()
        except Exception as error:  # noqa: BLE001
            failed, excep = 1, error
        finally:
            # ping the next rank if it exists, then wait for the token to come back around
            if comm.size > 1:
                comm.isend(failed, dest=1)
                comm.recv()
    else:
        # wait for the previous rank to finish writing its chunk, then write our own
        failed = comm.recv()
        try:
            if not failed and is_split:
                write_other()
        except Exception as error:  # noqa: BLE001
            failed, excep = comm.rank + 1, error
        finally:
            # ping the next rank, wrapping around to 0 to complete barrier behavior
            comm.isend(failed, dest=(comm.rank + 1) % comm.size)

    propagate_failure(comm, failed, excep)


def propagate_failure(
    comm: Communication, failed: int, error: Optional[BaseException] = None
) -> None:
    """
    Re-raises ``error`` on every process if any process reported a failure.

    Parameters
    ----------
    comm : Communication
        The communicator whose processes participate.
    failed : int
        ``0`` if this process succeeded, otherwise its rank plus one.
    error : BaseException, optional
        The exception this process caught, if it failed.

    Raises
    ------
    BaseException
        Whatever the failing process caught, on every process.
    """
    failed = comm.allreduce(failed, op=MPI.MAX)
    if not failed:
        return
    if failed - 1 == comm.rank:
        comm.bcast(error, root=failed - 1)
        raise error
    error = comm.bcast(error, root=failed - 1)
    error.args = (f"raised by process rank {failed - 1}", *error.args)
    # raise the same error but without the traceback, which is on a different process
    raise error from None


def split_file_indices(n_files: int, rank: int, size: int) -> Tuple[int, int]:
    """
    Distributes ``n_files`` files over ``size`` processes as evenly as possible.

    Returns the index of this process' first file and the number of files it is to read.

    Parameters
    ----------
    n_files : int
        Total number of files to distribute.
    rank : int
        Rank of the calling process.
    size : int
        Number of processes to distribute over.
    """
    remainder = n_files % size
    if rank < remainder:
        n_for_procs = n_files // size + 1
        return rank * n_for_procs, n_for_procs
    n_for_procs = n_files // size
    return rank * n_for_procs + remainder, n_for_procs


def load_from_folder(
    path: str,
    pattern: str,
    reader: Callable[[str], np.ndarray],
    format_name: str,
    dtype: datatype,
    split: int = 0,
    device: Optional[str] = None,
    comm: Optional[Communication] = None,
) -> DNDarray:
    """
    Loads every file in ``path`` matching ``pattern`` into a single ``DNDarray``.

    The files are distributed over the processes, read locally, and concatenated along
    ``split``.

    Parameters
    ----------
    path : str
        Path to the directory holding the files.
    pattern : str
        Filename glob selecting the files to read, e.g. ``"*.npy"``.
    reader : Callable[[str], np.ndarray]
        Reads a single file into a numpy array.
    format_name : str
        Human-readable name of the file format, used in error messages.
    dtype : datatype
        Data type of the resulting array.
    split : int
        Along which axis the loaded arrays should be concatenated.
    device : str, optional
        The device id on which to place the data, defaults to the globally set default.
    comm : Communication, optional
        The communication to use, defaults to ``heat.MPI_WORLD``.

    Raises
    ------
    TypeError
        If any of the input parameters are not of correct type.
    ValueError
        If no matching files were found.
    RuntimeError
        If there are more processes than files.
    """
    path = sanitize_path(path)
    if split is not None and not isinstance(split, int):
        raise TypeError(f"split must be None or int, not {type(split)}")

    # sort, because os.listdir does not guarantee the same order on every process
    file_list = sorted(file for file in os.listdir(path) if fnmatch.fnmatch(file, pattern))
    n_files = len(file_list)
    if n_files == 0:
        raise ValueError(f"No {format_name} files were found")
    if n_files < MPI_WORLD.size and MPI_WORLD.size > 1:
        raise RuntimeError("Number of processes can't exceed number of files")

    idx, n_for_procs = split_file_indices(n_files, MPI_WORLD.rank, MPI_WORLD.size)
    array_list = [reader(os.path.join(path, file)) for file in file_list[idx : idx + n_for_procs]]

    larray = torch.from_numpy(np.concatenate(array_list, split))
    return factories.array(larray, dtype=dtype, device=device, is_split=split, comm=comm)
