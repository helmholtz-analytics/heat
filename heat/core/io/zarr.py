"""Reading and writing Heat arrays in the Zarr format.

This module is called ``zarr``, but ``import zarr`` resolves to the third-party package:
Python 3 imports are absolute. The library is only resolved on first use anyway, through
:func:`heat.core._config.require_dependency`, so that ``import heat`` does not pay for it.
"""

from __future__ import annotations

import os
import fnmatch
import glob
import operator
import warnings
from functools import reduce
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from .. import devices
from .. import factories
from .. import types
from .._config import is_installed, require_dependency
from ..communication import Communication, MPI, MPI_WORLD, sanitize_comm
from ..dndarray import DNDarray
from ..stride_tricks import sanitize_axis
from ..types import datatype
from ._registry import register_format
from .utils import compose_slices, sanitize_extension, sanitize_path

__all__ = ["supports_zarr"]

#: File extensions handled by this module.
EXTENSIONS = frozenset([".zarr"])


def supports_zarr() -> bool:
    """
    Returns ``True`` if zarr is installed, ``False`` otherwise.
    """
    return is_installed("zarr")


if supports_zarr():
    __all__.extend(["load_zarr", "save_zarr"])

    def load_zarr(
        path: str,
        variable: str = None,
        split: int = 0,
        device: Optional[str] = None,
        comm: Optional[Communication] = None,
        slices: Union[None, slice, Iterable[Union[slice, None]]] = None,
        **kwargs,
    ) -> DNDarray:
        """
        Loads data from a zarr store into DNDarray. `path` can either point to a single zarr array or a zarr group. In the latter case, `variable` must be provided to specify which array in the group to load. If `variable` contains a wildcard pattern (e.g. `RECEIVER_*/DATA`), all matching arrays will be loaded and concatenated along the specified `split` axis.

        Parameters
        ----------
        path : str
            Path to the directory in which a .zarr-file is located.
        variable : str, optional
            If the zarr store is a group, the variable (or path to variable) to load from the group.
            Can contain a wildcard pattern to load and concatenate arrays stored in slices in different directories.
        split : int
            Along which axis the loaded arrays should be concatenated.
        device : str, optional
            The device id on which to place the data, defaults to globally set default device.
        comm : Communication, optional
            The communication to use for the data distribution, default is 'heat.MPI_WORLD'
        slices: Union[None, slice, Iterable[Union[slice, None]]]
            Load only a slice of the array instead of everything
        **kwargs : Any
            extra Arguments to pass to zarr.open
        """
        zarr = require_dependency("zarr")
        # sanitize inputs
        device = devices.sanitize_device(device)
        torch_device = device.torch_device
        comm = sanitize_comm(comm)

        if not isinstance(path, str):
            raise TypeError(f"path must be str, not {type(path)}")
        if not isinstance(slices, (slice, Iterable)) and slices is not None:
            raise TypeError(f"Slices Argument must be slice, tuple or None and not {type(slices)}")
        if isinstance(slices, Iterable):
            for elem in slices:
                if isinstance(elem, slice) or elem is None:
                    continue
                raise TypeError(f"Tuple values of slices must be slice or None, not {type(elem)}")

        path = sanitize_extension(path, EXTENSIONS, "zarr")

        store_path = os.path.join(path, variable) if variable else path

        output_dtype = kwargs.pop("dtype", None)
        torch_output_dtype = output_dtype.torch_type() if output_dtype else None

        if variable and "*" in variable:
            # `variable` contains a wildcard pattern
            # e.g. data were chunked at write-out and stored in multiple directories
            if slices is not None:
                raise NotImplementedError("Slicing is not supported when loading with a wildcard.")

            base_paths = sorted(glob.glob(store_path))

            if not base_paths:
                raise FileNotFoundError(
                    f"Zarr wildcard pattern '{variable}' did not match any arrays in store '{path}'"
                )

            variable_paths = [os.path.relpath(p, start=path) for p in base_paths]

            # each rank reads data from its assigned directories and concatenates locally

            # determine which directories to open on rank
            dummy_array = factories.empty((len(base_paths),), dtype=types.float32)
            _, _, local_dir_slice = dummy_array.comm.chunk(
                dummy_array.shape, rank=dummy_array.comm.rank, split=0
            )

            # load data to torch tensors
            local_tensors = []
            for i, var_path in enumerate(variable_paths[local_dir_slice[0]]):
                local_tensor = torch.from_numpy(zarr.open(path)[var_path][:])
                if torch_output_dtype:
                    local_tensor = local_tensor.to(torch_output_dtype)
                local_tensors.append(local_tensor)

            # Have rank 0 determine the single-store shape and broadcast it to all ranks for sanitation
            target_ndims = torch.zeros(1, dtype=torch.int32)
            if dummy_array.comm.rank == 0:
                if len(local_tensors) == 0:
                    raise ValueError(
                        f"Zarr wildcard pattern '{variable}' did not match any arrays in store '{path}'"
                    )
                # broadcast shape of first local tensor to allow sanitation on empty ranks
                target_ndims = torch.tensor(local_tensors[0].ndim, dtype=torch.int32)
            dummy_array.comm.Bcast(target_ndims, root=0)
            # sanitize split axis
            proxy_shape = (1,) * target_ndims.item()
            split = sanitize_axis(proxy_shape, axis=split)

            # prepare sorted list of all heat datatypes to convert them to integer for communication with MPI
            all_heat_type_names = [dtype.__name__ for dtype in types.__get_all_heat_types()]
            all_heat_type_names.sort()

            # concatenate locally
            if len(local_tensors) >= 1:
                if len(local_tensors) == 1:
                    local_tensor = local_tensors[0]
                else:
                    local_tensor = torch.cat(local_tensors, dim=split if split is not None else 0)
                empty_ranks = torch.tensor([0], dtype=torch.int32)
                ht_type_code = all_heat_type_names.index(
                    types.canonical_heat_type(local_tensor.dtype).__name__
                )
            else:
                # no local tensors i.e. no data assigned to rank
                local_tensor = torch.empty((0,))
                empty_ranks = torch.tensor([1], dtype=torch.int32)
                # dummy dtype code
                ht_type_code = -1
            # check for empty ranks
            dummy_array.comm.Allreduce(MPI.IN_PLACE, empty_ranks, op=MPI.SUM)
            if empty_ranks.item() > 0:
                # fix local shape and dtype of empty tensors, otherwise DNDarray construction will fail
                # Rank 0 broadcasts the info to all other ranks
                target_shape = torch.zeros(
                    (
                        1,
                        target_ndims.item() + 1,
                    ),
                    dtype=torch.int64,
                )
                if local_tensor.numel() > 0:
                    target_shape[0, :-1] = torch.tensor(local_tensor.shape, dtype=torch.int64)
                # encode dtype as last entry
                target_shape[0, -1] = ht_type_code
                # share info about target shape and dtype
                target_shapes = torch.zeros(
                    (dummy_array.comm.size, target_ndims.item() + 1), dtype=torch.int64
                )
                dummy_array.comm.Allgather(target_shape, target_shapes)
                if local_tensor.numel() == 0:
                    ht_type_code = target_shapes[0, -1].item()
                    target_shape = target_shapes[0, :-1].clone()
                    target_shape[split] = 0
                    ht_type = getattr(types, all_heat_type_names[ht_type_code])
                    local_tensor = torch.empty(
                        tuple(target_shape.tolist()), dtype=ht_type.torch_type()
                    )
                # discard dtype code column
                target_shapes = target_shapes[:, :-1]
                # calculate global array shape
                out_gshape = target_shapes[0, :].clone()
                out_gshape[split] = target_shapes[:, split].sum().item()
                # wrap local tensors in DNDarray
                dndarray = DNDarray(
                    local_tensor.to(device=torch_device),
                    gshape=tuple(out_gshape.tolist()),
                    dtype=output_dtype
                    if output_dtype
                    else types.canonical_heat_type(local_tensor.dtype),
                    split=split,
                    device=device,
                    comm=comm,
                    balanced=False,
                )
            else:
                # all ranks are populated, create DNDarray directly
                dndarray = factories.array(local_tensor, is_split=split, device=device, comm=comm)
            dndarray.balance_()
            return dndarray

        # standard single zarr array
        arr: zarr.Array = zarr.open_array(store=store_path, **kwargs)
        shape = arr.shape

        if isinstance(slices, slice) or slices is None:
            slices = [slices]

        if len(shape) < len(slices):
            raise ValueError(
                f"slices Argument has more arguments than the length of the shape of the array. {len(shape)} < {len(slices)}"
            )

        slices = [elem if elem is not None else slice(None) for elem in slices]
        slices.extend([slice(None) for _ in range(abs(len(slices) - len(shape)))])

        dtype = types.canonical_heat_type(arr.dtype)
        split = sanitize_axis(shape, axis=split)

        # slices = tuple(slice(*tslice.indices(length)) for length, tslice in zip(shape, slices))
        slices = tuple(slices)
        stored_shape = shape
        shape = [
            len(range(*tslice.indices(length))) for length, tslice in zip(stored_shape, slices)
        ]
        offset, local_shape, local_slices = comm.chunk(shape, split)

        # compose the user's slices with this process' chunk, so that only the local
        # region is read from disk rather than the whole array on every process
        read_slices = tuple(
            compose_slices(user_slice, local_slice, length)
            for user_slice, local_slice, length in zip(slices, local_slices, stored_shape)
        )

        return factories.array(
            arr[read_slices], dtype=dtype, is_split=split, device=device, comm=comm
        )

    def save_zarr(dndarray: DNDarray, path: str, overwrite: bool = False, **kwargs) -> None:
        """
        Writes the DNDArray into the zarr-format.

        Parameters
        ----------
        dndarray : DNDarray
            DNDArray to save.
        path : str
            path to save to.
        overwrite : bool
            Wether to overwrite an existing array.
        **kwargs : Any
            extra Arguments to pass to zarr.open and zarr.create

        Raises
        ------
        TypeError
            - If given parameters do not match or have conflicting information.
            - If it already exists and no overwrite is specified.

        Notes
        -----
        Zarr functions by chunking the data, were a chunk is a file inside the store.
        The problem ist that only one process writes to it at a time. Therefore when two
        processes try to write to the same chunk one will fail, unless the other finishes before
        the other starts.

        To alleviate it we can define the chunk sizes ourselves. To do this we just get the lowest size of
        the distributed axis, ex: split=0 with a (4,4) shape with a worldsize of 4 you would chunk it with (1,4).

        A problem arises when a process gets a bigger chunk and interferes with another process. Example:
        N_PROCS = 4
        SHAPE = (9,10)
        SPLIT = 0
        CHUNKS => (2,10)

        In this problem one process will have a write region of 3 rows and therefore be able to either not write
        or overwrite what another process does therefore destroying the parallel write as it would at the end load
        2 chunks to write 3 rows.
        To counter act this we just set the chunk size in the split axis to 1. This allows for no overwrites but can
        cripple write speeds and or even speed it up.

        Another Problem with this approach is that we tell zarr have full chunks, i.e if array has shape (10_000, 10_000)
        and we split it at axis=0 with 4 processes we have chunks of (2_500, 10_000). Zarr will load the whole chunk into
        memory making it memory intensive and probably inefficient. Better approach would be to have a smaller chunk size
        for example half of it but that cannot be determined at all times so the current approach is a compromise.

        Another Problem is the split=None scenario. In this case every processs has the same data, so only one needs to write
        so we ignore chunking and let zarr decide the chunk size and let only one process, aka rank=0 write.

        To avoid errors when using NumPy arrays as chunk shape, the chunks argument is only passed to zarr.create if it is
        not None. This prevents issues with ambiguous truth values or attribute errors on None.

        """
        zarr = require_dependency("zarr")
        if not isinstance(path, str):
            raise TypeError(f"path must be str, not {type(path)}")

        path = sanitize_extension(path, EXTENSIONS, "zarr")

        if os.path.exists(path) and not overwrite:
            raise RuntimeError("Given Path already exists.")

        if dndarray.comm.rank == 0:
            if dndarray.split is None or dndarray.comm.size == 1:
                chunks = None
            else:
                chunks = np.array(dndarray.gshape)
                axis = dndarray.split

                if chunks[axis] % dndarray.comm.size != 0:
                    chunks[axis] = 1
                else:
                    chunks[axis] //= dndarray.comm.size

                    CODEC_LIMIT_BYTES = 2**31 - 1  # PR#1766

                    for _ in range(
                        10
                    ):  # Use for loop instead of while true for better handling of edge cases
                        byte_size = reduce(operator.mul, chunks, 1) * dndarray.larray.element_size()
                        if byte_size > CODEC_LIMIT_BYTES:
                            if chunks[axis] % 2 == 0:
                                chunks[axis] /= 2
                                continue
                            else:
                                chunks[axis] = 1
                                break
                        else:
                            break
                    else:
                        chunks[axis] = 1
                        warnings.warn(
                            "Calculation of chunk size for zarr format unexpectadly defaulted to 1 on the split axis"
                        )

            dtype = dndarray.dtype.char()

            zarr_create_kwargs = {
                "store": path,
                "shape": dndarray.gshape,
                "dtype": dtype,
                "overwrite": overwrite,
                **kwargs,
            }

            if chunks is not None:
                zarr_create_kwargs["chunks"] = chunks.tolist()

            zarr_array = zarr.create(**zarr_create_kwargs)

        # Wait for the file creation to finish
        dndarray.comm.Barrier()
        zarr_array = zarr.open(store=path, mode="r+", **kwargs)

        if dndarray.split is not None:
            _, _, slices = dndarray.comm.chunk(dndarray.gshape, dndarray.split)

            zarr_array[slices] = (
                dndarray.larray.cpu().numpy()  # Numpy array needed as zarr can only understand numpy dtypes and infers it.
            )
        else:
            if dndarray.comm.rank == 0:
                zarr_array[:] = dndarray.larray.cpu().numpy()

        dndarray.comm.Barrier()


register_format(
    "zarr",
    EXTENSIONS,
    dependency="zarr",
    loader=load_zarr if supports_zarr() else None,
    saver=save_zarr if supports_zarr() else None,
)
