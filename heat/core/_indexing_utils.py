from __future__ import annotations

import bisect
from collections.abc import Iterable
from typing import Any, NamedTuple, Union

import numpy as np
import torch
from mpi4py import MPI

from .dndarray import DNDarray
from . import factories
from . import types
from .types import bool as ht_bool, uint8 as ht_uint8

# Type aliases
Index = Union[int, slice, type(...), None, torch.Tensor, np.ndarray, "DNDarray"]
Key = Union[Index, tuple[Index, ...], list[Index]]


class ProcessedKey(NamedTuple):
    """
    A named tuple to store the processed key information for distributed indexing operations.
    """

    key: Any
    op_type: str  # "scalar", "descending_slice", "distr_mask", "local_mask", "local", "distributed"
    output_shape: tuple
    output_split: int | None
    split_key_is_ordered: int
    key_is_mask_like: bool
    out_is_balanced: bool
    root: int | None


# ----------------------------------------------------------------------
# Predicates and type inspection
# ----------------------------------------------------------------------


def _is_boolean_array(k: Any) -> bool:
    """Return True if k is a boolean or uint8 array/tensor of any dimension."""
    return hasattr(k, "dtype") and k.dtype in (
        ht_bool,
        ht_uint8,
        torch.bool,
        torch.uint8,
        np.bool_,
        np.uint8,
    )


def _is_boolean_scalar_key(k: Any) -> bool:
    """Return True if k is a python bool or a 0-D boolean array/tensor."""
    return isinstance(k, bool) or (_is_boolean_array(k) and getattr(k, "ndim", 0) == 0)


def _is_scalar_key(k: Any) -> bool:
    """Return True if k is a non-boolean scalar or 0-D array indexer."""
    return not _is_boolean_scalar_key(k) and (np.isscalar(k) or getattr(k, "ndim", 1) == 0)


# ----------------------------------------------------------------------
# Key unwrapping and normalization
# ----------------------------------------------------------------------


def _unwrap_local_key(key: Any, device: torch.device | None = None) -> Any:
    """
    Recursively unwrap local DNDarray or numpy array keys into torch-compatible indexers on correct device.
    """
    if isinstance(key, DNDarray):
        if key.is_distributed():
            raise TypeError("Cannot use distributed DNDarray for local fast-path indexing")
        return key.larray.item() if key.ndim == 0 else key.larray
    if isinstance(key, np.ndarray):
        return torch.from_numpy(key).to(device=device)
    if isinstance(key, tuple):
        if len(key) == 1 and _is_boolean_array(key[0]):
            return _unwrap_local_key(key[0], device=device)
        else:
            return tuple(_unwrap_local_key(k, device=device) for k in key)
    if isinstance(key, list):
        return [_unwrap_local_key(k, device=device) for k in key]
    return key


def _normalize_key(key: Key, device: torch.device) -> tuple[Any, ...]:
    """
    Standardize the non-DNDarray coordinate indices to PyTorch-friendly key items.

    Returns a tuple of normalized key items.
    """
    # Normalize top-level container to a list
    if isinstance(key, tuple):
        key_list = list(key)
    elif isinstance(key, list):
        # Could be a list of indices: arr[[0, 2]] or arr[0, 2]
        # Try casting to 1D integer tensor if all elements are ints
        try:
            key_list = [torch.tensor(key, device=device)]
        except (RuntimeError, TypeError, ValueError):
            key_list = list(key)
    else:
        key_list = [key]

    normalized = []
    for k in key_list:
        # Convert numpy array to torch tensor on target device
        if isinstance(k, np.ndarray):
            normalized.append(torch.from_numpy(k).to(device=device))

        # Unwrap 0-D scalar DNDarray into a Python scalar
        elif isinstance(k, DNDarray) and k.ndim == 0:
            normalized.append(k.larray.item())

        # Unpack singleton containers like (idx,) often produced by nonzero/where
        elif isinstance(k, (tuple, list)) and len(k) == 1 and isinstance(k[0], DNDarray):
            if k[0].ndim > 0:
                normalized.append(k[0])
            else:
                normalized.append(torch.tensor([k[0].larray.item()], device=device))

        # Sequence of scalar DNDarrays -> unwrap to list of Python scalars
        elif (
            isinstance(k, (tuple, list))
            and len(k) > 0
            and all(isinstance(elem, DNDarray) and elem.ndim == 0 for elem in k)
        ):
            normalized.append(torch.tensor([elem.larray.item() for elem in k], device=device))

        # Catch invalid nested non-scalar DNDarrays early
        elif isinstance(k, (tuple, list)) and any(
            isinstance(elem, DNDarray) and elem.ndim > 0 for elem in k
        ):
            raise TypeError(
                "Nested tuple/list of non-scalar DNDarray indices is not supported. "
                "Pass them as separate indices (e.g. arr[idx0, idx1, ...]) or unwrap "
                "singleton tuples (e.g. idx = idx[0])."
            )

        # Convert non-distributed integer/indexing DNDarrays to local torch.Tensor
        elif isinstance(k, DNDarray) and k.split is None and k.dtype in (types.int32, types.int64):
            normalized.append(k.larray.to(dtype=torch.int64))

        # Ensure torch.Tensor indices are placed on the target device
        elif isinstance(k, torch.Tensor):
            normalized.append(k)

        else:
            # Leave slices, integers, None, Ellipsis,
            # and distributed DNDarrays (ndim >= 1 and split is not None) intact.
            normalized.append(k)

    return tuple(normalized)


# ----------------------------------------------------------------------
# Scalar early-out processing
# ----------------------------------------------------------------------


def _process_scalar_key(
    arr: "DNDarray",
    key: int | "DNDarray" | torch.Tensor | np.ndarray,
    indexed_axis: int,
    return_local_indices: bool | None = False,
) -> tuple[int, int | None]:
    """
    Private helper function to process a single-item scalar key used for indexing a ``DNDarray``.
    """
    # cast key to scalar if it is an array
    try:
        key = key.item()
    except AttributeError:
        pass
    if not arr.is_distributed():
        root = None
        return key, root
    if arr.split == indexed_axis:
        # adjust negative key
        if key < 0:
            key += arr.gshape[indexed_axis]
        # work out active process
        _, displs = arr.counts_displs()
        root = bisect.bisect_right(displs, key) - 1
        # correct key for rank-specific displacement
        if return_local_indices and arr.comm.rank == root:
            key -= displs[root]
    else:
        root = None
    return key, root


def _scalar_early_out(
    arr: "DNDarray",
    key: Any,
    op: str | None,
    return_local_indices: bool | None,
) -> tuple["DNDarray", ProcessedKey]:
    """Resolve early-out for scalar indices."""
    if arr.ndim == 0 and op == "get":
        raise IndexError(
            "Too many indices for DNDarray: DNDarray is 0-dimensional, but 1 were indexed"
        )

    output_shape = arr.gshape[1:]
    output_split = None if arr.split in (None, 0) else arr.split - 1
    processed_key, root = _process_scalar_key(
        arr, key, indexed_axis=0, return_local_indices=return_local_indices
    )

    return arr, ProcessedKey(
        key=processed_key,
        op_type="scalar",
        output_shape=tuple(output_shape),
        output_split=output_split,
        split_key_is_ordered=1,
        key_is_mask_like=False,
        out_is_balanced=True if output_split is None else arr.balanced,
        root=root,
    )


# ----------------------------------------------------------------------
# Key expansion and per-axis processing
# ----------------------------------------------------------------------


def _expand_dimensions_and_ellipsis(
    arr: "DNDarray",
    key: list[Any],
    output_shape: list[int],
    split_bookkeeping: list[str | None],
) -> tuple["DNDarray", list[Any], list[int], list[str | None]]:
    """
    Expands ellipses (...) into full slices and inserts singleton dimensions
    for None (newaxis) or 0-D boolean masks.
    """
    add_dims = sum(k is None or _is_boolean_scalar_key(k) for k in key)
    ellipsis = sum(isinstance(k, type(...)) for k in key)

    if ellipsis > 1:
        raise ValueError("indexing key can only contain 1 Ellipsis (...)")

    if ellipsis:
        expand_key = [slice(None)] * (arr.ndim + add_dims)
        ellipsis_index = key.index(...)
        ellipsis_dims = arr.ndim - (len(key) - ellipsis - add_dims)
        expand_key[:ellipsis_index] = key[:ellipsis_index]
        expand_key[ellipsis_index + ellipsis_dims :] = key[ellipsis_index + 1 :]
        key = expand_key

    while add_dims > 0:
        for i, k in reversed(list(enumerate(key))):
            if k is None or _is_boolean_scalar_key(k):
                if k is None:
                    key[i] = slice(None)
                else:
                    val = bool(k.item() if hasattr(k, "item") else k)
                    key[i] = slice(None) if val else slice(0, 0)

                insert_pos = i - add_dims + 1
                arr = arr.expand_dims(insert_pos)
                output_shape = output_shape[:insert_pos] + [1] + output_shape[insert_pos:]
                split_bookkeeping = (
                    split_bookkeeping[:insert_pos] + [None] + split_bookkeeping[insert_pos:]
                )
                add_dims -= 1

    return arr, key, output_shape, split_bookkeeping


def _distr_mask_fast_path(arr: "DNDarray", key: Any, op: str | None) -> bool:
    """
    Checks if the indexing operation qualifies for the distributed boolean mask fast path.
    """
    if not arr.is_distributed():
        return False

    if isinstance(key, tuple) and len(key) > arr.split:
        split_key = key[arr.split]
    elif isinstance(key, DNDarray):
        split_key = key
    else:
        split_key = None

    if (
        isinstance(split_key, DNDarray)
        and split_key.dtype in (ht_bool, ht_uint8)
        and split_key.split == arr.split
    ):
        if split_key.gshape == arr.gshape:
            # "get" flattens to 1D; if split > 0, local flattening scrambles global C-order
            return op == "set" or (op == "get" and arr.split == 0)
        elif (
            split_key.ndim == 1 and arr.split == 0 and split_key.gshape == (arr.gshape[arr.split],)
        ):
            return True

    return False


def _resolve_1d_boolean_first_dim(
    arr: "DNDarray", key: tuple[Any, ...] | Any, distr_mask_fast_path: bool
) -> tuple[Any, ...] | Any:
    """
    If key indexes axis 0 with a 1D boolean mask matching the global size of axis 0,
    convert that mask into integer coordinates via nonzero().
    """
    if distr_mask_fast_path or arr.ndim == 0:
        return key

    first = key[0] if isinstance(key, tuple) and len(key) >= 1 else key

    if not isinstance(first, (DNDarray, torch.Tensor)):
        return key

    first_dtype = getattr(first, "dtype", None)
    first_ndim = getattr(first, "ndim", 0)
    first_shape = tuple(getattr(first, "shape", ()))

    if (
        first_ndim == 1
        and first_shape == (arr.gshape[0],)
        and first_dtype in (ht_bool, ht_uint8, torch.bool, torch.uint8)
    ):
        if isinstance(first, DNDarray):
            nz = first.nonzero()
            idx0 = nz[0] if isinstance(nz, tuple) else nz
        elif isinstance(first, torch.Tensor):
            idx0 = torch.nonzero(first, as_tuple=False).flatten()

        return (idx0,) + key[1:] if isinstance(key, tuple) else (idx0,)

    return key


def _process_slice_key(
    k: slice,
    dim: int,
    is_split_axis: bool,
    displs: list[int] | None,
    counts: list[int] | None,
    rank: int | None,
    device: torch.device,
    return_local_indices: bool,
) -> tuple[Any, int, int | None, bool | None]:
    """
    Computes local slice or index tensor along an axis, determining output dimension length
    and slice monotonicity ordering.
    """
    if k.step == 0:
        raise ValueError("Slice step cannot be zero")

    start, stop, step = slice(k.start, k.stop, k.step).indices(dim)

    new_key = k
    output_dim_len = 0
    split_key_is_ordered = None
    out_is_balanced = None

    if step < 0 and start > stop:
        # total items in the global descending slice
        output_dim_len = len(range(start, stop, step))

        if is_split_axis:
            split_key_is_ordered = -1
            out_is_balanced = False

            # PyTorch cannot index with negative steps
            # we work with the values in ascending order
            s = -step
            # lowest global coordinate produced by this slice
            min_coord = start - (output_dim_len - 1) * s
            # highest global coordinate produced by this slice
            max_coord = start

            # global index interval [low, high) owned by this MPI process
            low = displs[rank]
            high = low + counts[rank]

            # overlap between the slice value bounds [min_coord, max_coord + 1)
            # and the current process's memory chunk [low, high)
            overlap_start = max(min_coord, low)
            overlap_end = min(max_coord + 1, high)

            # check if there is any overlap at all
            if overlap_start < overlap_end:
                # first item in the progression that is >= overlap_start.
                k_first = (overlap_start - min_coord + s - 1) // s
                g_first = min_coord + k_first * s

                if g_first < overlap_end:
                    # first value is still inside the rank's chunk
                    # calculate how many steps fit in [g_first, overlap_end)
                    steps_local = (overlap_end - 1 - g_first) // s + 1

                    # convert global index to a process-local offset if requested
                    start_idx = g_first - low if return_local_indices else g_first
                    stop_idx = start_idx + steps_local * s

                    # allocate local indices only
                    new_key = slice(start_idx, stop_idx, s)
                else:
                    # slicing skips local chunk completely
                    new_key = slice(0, 0)
            else:
                # local chunk is outside the slice bounds entirely
                new_key = slice(0, 0)
        else:
            # non-split axis: return the descending slice as indices
            new_key = torch.arange(start, stop, step, device=device, dtype=torch.int64)

    elif step > 0 and start < stop:
        output_dim_len = len(range(start, stop, step))

        if is_split_axis:
            split_key_is_ordered = 1
            out_is_balanced = False
            local_arr_end = displs[rank] + counts[rank]
            if stop > displs[rank] and start < local_arr_end:
                index_in_cycle = (displs[rank] - start) % step
                if start >= displs[rank]:
                    local_start = start - displs[rank]
                else:
                    local_start = 0 if index_in_cycle == 0 else step - index_in_cycle
                if stop <= local_arr_end:
                    local_stop = stop - displs[rank]
                else:
                    local_stop = counts[rank]

                new_key = slice(local_start, local_stop, step)
            else:
                new_key = slice(0, 0)
    else:
        new_key = slice(0, 0)
        output_dim_len = 0

    return new_key, output_dim_len, split_key_is_ordered, out_is_balanced


def _sanitize_int_indices(k: "DNDarray", dim: int, axis: int, comm: Any, device: Any) -> "DNDarray":
    """
    Validates integer bounds and normalizes negative coordinates for distributed/local DNDarray keys.
    """
    if k.dtype not in (types.int32, types.int64):
        if k.dtype not in (ht_bool, ht_uint8):
            raise IndexError(
                f"arrays used as indices must be of integer (or boolean) type, got {k.dtype}"
            )
        return k

    # Combine local checks into one reduced boolean tensor
    local_flags = torch.tensor(
        [
            ((k.larray < -dim) | (k.larray >= dim)).any(),
            (k.larray < 0).any(),
        ],
        dtype=torch.int32,
        device=device.torch_device,
    )

    do_reduce = comm is not None and getattr(comm, "size", 1) > 1 and k.is_distributed()
    if do_reduce:
        comm.Allreduce(MPI.IN_PLACE, local_flags, op=MPI.SUM)

    invalid_sum = local_flags[0].item()
    has_neg_sum = local_flags[1].item()

    if invalid_sum > 0:
        raise IndexError(f"index out of bounds for axis {axis} with size {dim}")

    if has_neg_sum > 0:
        k_l = k.larray.clone()
        k_l[k_l < 0] += dim
        k = factories.array(
            k_l,
            dtype=k.dtype,
            split=k.split,
            device=device,
            comm=comm,
            copy=False,
        )

    return k


# ----------------------------------------------------------------------
# Advanced indexing and routing helpers
# ----------------------------------------------------------------------


def _sanitize_advanced_keys(
    arr: "DNDarray",
    key: list[Any],
    advanced_indexing_dims: list[int],
    split_key_is_ordered: int,
    key_is_mask_like: bool,
    distr_mask_fast_path: bool,
    counts: tuple | list | None,
    displs: tuple | list | None,
    return_local_indices: bool,
) -> tuple[list[Any], bool]:
    """
    Validates key distribution alignment along the split axis and converts
    all advanced indexing key elements from DNDarrays into local torch.Tensors.
    """
    key = list(key)

    # Detect mask-like conditions (same shape for adv indexing dimensions)
    adv_keys = [key[i] for i in advanced_indexing_dims]
    key_is_mask_like = key_is_mask_like or (
        len(advanced_indexing_dims) > 1
        and all(isinstance(k, DNDarray) for k in adv_keys)
        and len(set(k.shape for k in adv_keys)) == 1
    )

    non_split_dims = [d for d in advanced_indexing_dims if d != arr.split]

    # Align distributions if mask-like
    if key_is_mask_like and arr.split is not None and arr.split in advanced_indexing_dims:
        key_splits = [k.split for k in adv_keys]
        split_pos = advanced_indexing_dims.index(arr.split)
        target_split = key_splits[split_pos]

        if key_splits.count(target_split) != len(key_splits):
            if target_split is not None and key_splits.count(None) == len(key_splits) - 1:
                for i in non_split_dims:
                    key[i] = factories.array(
                        key[i],
                        split=target_split,
                        device=arr.device,
                        comm=arr.comm,
                        copy=None,
                    )
            else:
                raise IndexError(
                    f"Indexing arrays must be distributed along the same dimension, got splits {key_splits}."
                )

    # Extract local torch.Tensors
    if arr.is_distributed() and arr.split in advanced_indexing_dims:
        if distr_mask_fast_path:
            for i in non_split_dims:
                if isinstance(key[i], DNDarray):
                    key[i] = key[i].larray
        elif split_key_is_ordered == 1:
            k = key[arr.split].larray if isinstance(key[arr.split], DNDarray) else key[arr.split]
            rank = arr.comm.rank
            low = displs[rank]
            high = low + counts[rank]

            idx_start = torch.searchsorted(k, low)
            idx_end = torch.searchsorted(k, high)
            k_local = k[idx_start:idx_end]
            if return_local_indices:
                k_local = k_local - low
            key[arr.split] = k_local

            if key_is_mask_like:
                for i in non_split_dims:
                    larr = key[i].larray if isinstance(key[i], DNDarray) else key[i]
                    key[i] = larr[idx_start:idx_end]
            else:
                for i in non_split_dims:
                    if isinstance(key[i], DNDarray):
                        key[i] = key[i].larray
        else:
            # split_key_is_ordered == 0 (unordered indexing)
            for i in advanced_indexing_dims:
                if isinstance(key[i], DNDarray):
                    key[i] = key[i].larray
    else:
        for i in advanced_indexing_dims:
            if isinstance(key[i], DNDarray):
                key[i] = key[i].larray

    return key, key_is_mask_like


def _reorder_advanced_idx_axes(
    arr: "DNDarray",
    key: list[Any],
    advanced_indexing_dims: list[int],
    advanced_indexing_shapes: list[tuple[int, ...]],
    output_shape: list[int | None],
    split_bookkeeping: list[str | None],
    key_is_mask_like: bool,
) -> tuple["DNDarray", list[Any], list[int | None], list[str | None], tuple[int, ...]]:
    """
    Broadcasts advanced indexing dimensions and rearranges non-consecutive dimensions
    to the front of the array as mandated by NumPy advanced indexing semantics.
    """
    try:
        broadcasted_shape = torch.broadcast_shapes(*advanced_indexing_shapes)
    except RuntimeError:
        raise IndexError(
            "Shape mismatch: indexing arrays could not be broadcast together with shapes: {}".format(
                advanced_indexing_shapes
            )
        )

    add_dims = len(broadcasted_shape) - len(advanced_indexing_dims)
    is_consecutive = (
        len(advanced_indexing_dims) == 1
        or list(range(advanced_indexing_dims[0], advanced_indexing_dims[-1] + 1))
        == advanced_indexing_dims
    )

    if is_consecutive:
        output_shape[
            advanced_indexing_dims[0] : advanced_indexing_dims[0] + len(advanced_indexing_dims)
        ] = broadcasted_shape

        if key_is_mask_like:
            has_split = (
                "split" in split_bookkeeping
                and split_bookkeeping.index("split") in advanced_indexing_dims
            )
            split_bookkeeping[
                advanced_indexing_dims[0] : advanced_indexing_dims[0] + len(advanced_indexing_dims)
            ] = ["split"] if has_split else [None]
        else:
            adv_sb = split_bookkeeping[advanced_indexing_dims[0] : advanced_indexing_dims[-1] + 1]
            new_adv_sb = [None] * len(broadcasted_shape)
            if "split" in adv_sb:
                new_idx = max(0, adv_sb.index("split") + add_dims)
                new_adv_sb[new_idx] = "split"

            split_bookkeeping = (
                split_bookkeeping[: advanced_indexing_dims[0]]
                + new_adv_sb
                + split_bookkeeping[advanced_indexing_dims[-1] + 1 :]
            )
    else:
        # Non-consecutive: transpose to make advanced dims leading and consecutive
        non_adv_ind_dims = [i for i in range(arr.ndim) if i not in advanced_indexing_dims]
        transpose_axes = tuple(advanced_indexing_dims + non_adv_ind_dims)
        arr = arr.transpose(transpose_axes)

        output_shape = [output_shape[i] for i in transpose_axes]
        output_shape[: len(advanced_indexing_dims)] = broadcasted_shape

        split_bookkeeping = [split_bookkeeping[i] for i in transpose_axes]
        adv_sb = split_bookkeeping[: len(advanced_indexing_dims)]
        new_adv_sb = [None] * len(broadcasted_shape)

        if "split" in adv_sb:
            new_idx = max(0, adv_sb.index("split") + add_dims)
            new_adv_sb[new_idx] = "split"

        split_bookkeeping = new_adv_sb + split_bookkeeping[len(advanced_indexing_dims) :]
        key = [key[i] for i in advanced_indexing_dims] + [key[i] for i in non_adv_ind_dims]

    return arr, key, output_shape, split_bookkeeping


def _assess_op_type(
    root: int | None,
    split_key_is_ordered: int,
    distr_mask_fast_path: bool,
    key_is_mask_like: bool,
) -> str:
    """Determine the indexing operation routing category."""
    if root is not None:
        return "scalar"
    if split_key_is_ordered == 0:
        return "distributed"
    if split_key_is_ordered == -1:
        return "descending_slice"
    if distr_mask_fast_path:
        return "distr_mask"
    if key_is_mask_like:
        return "local_mask"
    return "local"


# ----------------------------------------------------------------------
# Main orchestrator
# ----------------------------------------------------------------------


def _resolve_indexing_state(
    arr: "DNDarray",
    key: Key,
    return_local_indices: bool | None = False,
    op: str | None = None,
) -> tuple["DNDarray", ProcessedKey]:
    """
    Private helper function to align the indexing key and the array for distributed indexing operations.
    This function is used internally by both ``__getitem__`` and ``__setitem__`` pipelines.

    After processing the key, the following conditions are guaranteed:
    - Any ellipses (`...`) or newaxis (`None`) objects have been replaced with the appropriate number of slice objects.
    - ``np.ndarray`` and ``DNDarray`` objects have been converted to process-local ``torch.Tensor`` objects.
    - The dimensionality of the key perfectly matches the (potentially modified) ``DNDarray`` it indexes.
    - Negative indices have been wrapped appropriately.

    This function also manipulates ``arr`` if necessary, inserting and/or transposing dimensions as dictated
    by advanced indexing rules. Finally, it calculates the output shape, new split axis, and balanced status
    of the resulting indexed array.

    Parameters
    ----------
    arr : DNDarray
        The ``DNDarray`` to be indexed.
    key : array-like indexer
        The raw key used for indexing.
    return_local_indices : bool, optional
        Whether to map the split-axis indices from global to process-local indices. This is only applied
        when the indexing key along the split dimension is ordered (i.e., ``split_key_is_ordered == 1``).
        Default: ``False``.
    op : str, optional
        The indexing context for which the key is being processed. Can be ``"get"`` for ``__getitem__``
        or ``"set"`` for ``__setitem__``. Default: ``None``.

    Returns
    -------
    tuple
        A tuple containing two elements: ``(arr, processed_key)``.

        - arr (DNDarray):
            The array to be indexed. Its dimensions may have been transposed or expanded if advanced,
            dimensional, or broadcasted indexing was used.
        - processed_key (ProcessedKey):
            A named tuple containing the resolved state required to execute the indexing operation,
            consisting of the following fields:

            - key (tuple): The processed, Torch-compatible index. Note: Indices along the split axis
                are local if ordered indexing is used, but remain global if unordered indexing is required.
            - op_type (str): The categorized indexing routing (``"scalar"``, ``"slice"``,
                ``"descending_slice"``, ``"distr_mask"``, ``"local_mask"``, ``"local"``, or ``"distributed"``).
            - output_shape (tuple): The global shape of the resulting array.
            - output_split (int or None): The split axis of the resulting array.
            - split_key_is_ordered (int): Monotonicity of the split key (``1``: ascending, ``0``: unordered,
                ``-1``: descending).
            - key_is_mask_like (bool): Whether the key acts as a boolean mask.
            - out_is_balanced (bool): Whether the resulting ``DNDarray`` maintains load balance.
            - root (int or None): The root MPI process ID if single-element indexing along the split
                axis isolate data to one rank.
    """
    # early out for scalar key
    if _is_scalar_key(key):
        return _scalar_early_out(
            arr=arr,
            key=key,
            op=op,
            return_local_indices=return_local_indices,
        )

    # normalize key items to torch-friendly types (torch.Tensor, int, slice, None, Ellipsis)
    # NB: distributed DNDarrays are not unwrapped here, they are handled later
    normalized_key = _normalize_key(key, device=arr.device.torch_device)
    # maintain single item when raw key was not passed as a tuple
    key = normalized_key if isinstance(key, tuple) else normalized_key[0]

    # unpack single-element tuple containing a boolean mask (a[(mask,)] same as a[mask])
    if isinstance(key, tuple) and len(key) == 1 and _is_boolean_array(key[0]):
        key = key[0]

    # evaluate if this is a distributed mask aligned with the array
    distr_mask_fast_path = _distr_mask_fast_path(arr, key, op)
    if distr_mask_fast_path and not isinstance(key, tuple):
        return arr, ProcessedKey(
            key=key.larray,
            op_type="distr_mask",
            output_shape=None,
            output_split=0 if op == "get" else arr.split,
            split_key_is_ordered=0,
            key_is_mask_like=True,
            out_is_balanced=False,
            root=None,
        )

    # 1D boolean mask resolution
    key = _resolve_1d_boolean_first_dim(arr, key, distr_mask_fast_path)

    output_shape = list(arr.gshape)
    split_bookkeeping = [None] * arr.ndim
    new_split = arr.split
    arr_is_distributed = False
    if arr.split is not None:
        split_bookkeeping[arr.split] = "split"
        if arr.is_distributed():
            counts, displs = arr.counts_displs()
            arr_is_distributed = True

    advanced_indexing = False
    split_key_is_ordered = 1
    key_is_mask_like = False
    out_is_balanced = True if not arr.is_distributed() else arr.balanced
    root = None

    if isinstance(key, (DNDarray, torch.Tensor)):
        if key.dtype in (ht_bool, ht_uint8, torch.bool, torch.uint8):
            # boolean indexing: shape must be consistent with arr.shape
            key_ndim = key.ndim
            if not tuple(key.shape) == arr.shape[:key_ndim]:
                raise IndexError(
                    "Boolean index of shape {} does not match indexed array of shape {}".format(
                        tuple(key.shape), arr.shape
                    )
                )
            if key_ndim == 0:
                # 0-D boolean mask: keep as 0-D tensor, do not extract non-zero
                key = key.larray if isinstance(key, DNDarray) else key
            else:
                # extract non-zero elements
                try:
                    key = key.nonzero(as_tuple=True)
                except TypeError:
                    key = key.nonzero()

            key_is_mask_like = True
        else:
            # advanced indexing on first dimension: first dim will expand to shape of key
            output_shape = tuple(list(key.shape) + output_shape[1:])
            # adjust split axis accordingly
            if arr_is_distributed:
                if arr.split != 0:
                    # split axis is not affected
                    split_bookkeeping = [None] * key.ndim + split_bookkeeping[1:]
                    new_split = (
                        split_bookkeeping.index("split") if "split" in split_bookkeeping else None
                    )
                    out_is_balanced = arr.balanced
                else:
                    # split axis is affected
                    if key.ndim > 1:
                        key_numel = key.numel()
                        if key_numel == arr.shape[0]:
                            new_split = tuple(key.shape).index(arr.shape[0])
                        else:
                            new_split = key.ndim - 1
                    else:
                        new_split = 0

                    key_is_dist = isinstance(key, DNDarray) and key.is_distributed()
                    if isinstance(key, DNDarray):
                        out_is_balanced = key.balanced
                        key = key.larray
                    else:
                        out_is_balanced = True

                    # normalize negative indices
                    if key.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                        dim = arr.gshape[0]
                        if ((key < -dim) | (key >= dim)).any():
                            raise IndexError(f"index out of bounds for axis 0 with size {dim}")
                        key = torch.where(key < 0, key + dim, key)

                    # identify ordered key
                    if key_is_dist or key.ndim > 1:
                        split_key_is_ordered = 0
                    else:
                        split_key_is_ordered = int((key[1:] >= key[:-1]).all().item())

                    # unordered local keys
                    if not split_key_is_ordered and not key_is_dist:
                        out_is_balanced = True

                    # ordered keys
                    if split_key_is_ordered:
                        rank = arr.comm.rank
                        low = displs[rank]
                        high = low + counts[rank]
                        idx_start = torch.searchsorted(key, low)
                        idx_end = torch.searchsorted(key, high)
                        key = key[idx_start:idx_end]
                        if return_local_indices:
                            key = key - low
                        out_is_balanced = False
            else:
                try:
                    out_is_balanced = key.balanced
                    new_split = key.split
                    key = key.larray
                except AttributeError:
                    # torch key, non-distributed indexed array
                    out_is_balanced = True
                    new_split = None

            op_type = _assess_op_type(
                root=root,
                split_key_is_ordered=split_key_is_ordered,
                distr_mask_fast_path=distr_mask_fast_path,
                key_is_mask_like=key_is_mask_like,
            )
            return arr, ProcessedKey(
                key=key,
                op_type=op_type,
                output_shape=tuple(output_shape),
                output_split=new_split,
                split_key_is_ordered=split_key_is_ordered,
                key_is_mask_like=key_is_mask_like,
                out_is_balanced=out_is_balanced,
                root=root,
            )

    if isinstance(key, (tuple, list)):
        key = list(key)
    else:
        key = [key]

    # check for ellipsis, newaxis. NB: (ht.newaxis is None)==True
    arr, key, output_shape, split_bookkeeping = _expand_dimensions_and_ellipsis(
        arr, key, output_shape, split_bookkeeping
    )

    # recalculate new_split, transpose_axes after dimensions manipulation
    new_split = split_bookkeeping.index("split") if "split" in split_bookkeeping else None

    # check for advanced indexing and slices
    advanced_indexing_dims = []
    advanced_indexing_shapes = []

    for i, k in enumerate(key):
        if _is_scalar_key(k):
            try:
                output_shape[i], split_bookkeeping[i] = None, None
            except IndexError:
                raise IndexError(
                    f"Too many indices for DNDarray: DNDarray is {arr.ndim}-dimensional, but {len(key)} dimensions were indexed"
                )
            if i == arr.split:
                key[i], root = _process_scalar_key(
                    arr, k, indexed_axis=i, return_local_indices=return_local_indices
                )
            else:
                key[i], _ = _process_scalar_key(arr, k, indexed_axis=i, return_local_indices=False)
        elif isinstance(k, Iterable) or isinstance(k, DNDarray):
            advanced_indexing = True
            advanced_indexing_dims.append(i)

            is_fast_path_component = distr_mask_fast_path and i == arr.split

            if is_fast_path_component:
                key[i] = k.larray if isinstance(k, DNDarray) else k
                advanced_indexing_shapes.append(tuple(k.shape))
                # skip the rest, local boolean masking along split axis
                continue

            if not isinstance(k, DNDarray):
                k = factories.array(k, device=arr.device, comm=arr.comm, copy=None)

            # normalize negative integer indices (NumPy/PyTorch semantics) and validate bounds
            k = _sanitize_int_indices(
                k=k, dim=arr.gshape[i], axis=i, comm=arr.comm, device=arr.device
            )

            advanced_indexing_shapes.append(k.gshape)
            if arr_is_distributed and i == arr.split:
                if (
                    not k.is_distributed()
                    and k.ndim == 1
                    and (k.larray.numel() <= 1 or (k.larray[1:] >= k.larray[:-1]).all().item())
                ):
                    split_key_is_ordered = 1
                    out_is_balanced = False
                else:
                    split_key_is_ordered = 0
                    out_is_balanced = True
            key[i] = k

        elif isinstance(k, slice) and k != slice(None):
            is_split_axis = arr_is_distributed and new_split == i
            rank = arr.comm.rank if arr_is_distributed else None

            slice_key, dim_len, s_ordered, s_balanced = _process_slice_key(
                k=k,
                dim=arr.gshape[i],
                is_split_axis=is_split_axis,
                displs=displs if arr_is_distributed else None,
                counts=counts if arr_is_distributed else None,
                rank=rank,
                device=arr.device.torch_device,
                return_local_indices=return_local_indices,
            )

            key[i] = slice_key
            output_shape[i] = dim_len
            if is_split_axis and s_ordered is not None:
                split_key_is_ordered = s_ordered
                out_is_balanced = s_balanced

    if advanced_indexing:
        key, key_is_mask_like = _sanitize_advanced_keys(
            arr=arr,
            key=key,
            advanced_indexing_dims=advanced_indexing_dims,
            split_key_is_ordered=split_key_is_ordered,
            key_is_mask_like=key_is_mask_like,
            distr_mask_fast_path=distr_mask_fast_path,
            counts=counts if arr_is_distributed else None,
            displs=displs if arr_is_distributed else None,
            return_local_indices=return_local_indices,
        )

        arr, key, output_shape, split_bookkeeping = _reorder_advanced_idx_axes(
            arr=arr,
            key=key,
            advanced_indexing_dims=advanced_indexing_dims,
            advanced_indexing_shapes=advanced_indexing_shapes,
            output_shape=output_shape,
            split_bookkeeping=split_bookkeeping,
            key_is_mask_like=key_is_mask_like,
        )

    # expand key to match the number of dimensions of the DNDarray
    if arr.ndim > len(key):
        key += [slice(None)] * (arr.ndim - len(key))

    while None in output_shape:
        lost_dim = output_shape.index(None)
        output_shape.pop(lost_dim)
        split_bookkeeping.pop(lost_dim)

    output_shape = tuple(output_shape)
    new_split = split_bookkeeping.index("split") if "split" in split_bookkeeping else None

    op_type = _assess_op_type(
        root=root,
        split_key_is_ordered=split_key_is_ordered,
        distr_mask_fast_path=distr_mask_fast_path,
        key_is_mask_like=key_is_mask_like,
    )

    return arr, ProcessedKey(
        key=tuple(key),
        op_type=op_type,
        output_shape=tuple(output_shape),
        output_split=new_split,
        split_key_is_ordered=split_key_is_ordered,
        key_is_mask_like=key_is_mask_like,
        out_is_balanced=out_is_balanced,
        root=root,
    )


# ----------------------------------------------------------------------
# RHS and setitem helpers
# ----------------------------------------------------------------------


def _broadcast_value(value: Any, output_shape: tuple[int, ...]) -> tuple[Any, bool]:
    """
    Broadcasts the assignment DNDarray `value` to the shape of the indexed array `arr[key]` if necessary.
    """
    is_scalar = (
        np.isscalar(value)
        or getattr(value, "ndim", 1) == 0
        or (value.shape == (1,) and value.split is None)
    )
    if is_scalar:
        # no need to broadcast
        return value, is_scalar
    # need information on indexed array
    indexed_dims = len(output_shape)
    value_shape = value.shape
    # check if value needs to be broadcasted
    if value_shape != output_shape:
        # assess whether the shapes are compatible, starting from the trailing dimension
        for i in range(1, min(len(value_shape), len(output_shape)) + 1):
            if value_shape[-i] != output_shape[-i] and value_shape[-i] != 1:
                raise ValueError(
                    f"could not broadcast input array from shape {value_shape} into shape {output_shape}"
                )
        # value has more dimensions than indexed array
        if value.ndim > indexed_dims:
            # check if all dimensions except the indexed ones are singletons
            all_singletons = value.shape[: value.ndim - indexed_dims] == (1,) * (
                value.ndim - indexed_dims
            )
            if not all_singletons:
                raise ValueError(
                    f"could not broadcast input array from shape {value_shape} into shape {output_shape}"
                )
            # squeeze out singleton dimensions
            value = value.squeeze(tuple(range(value.ndim - indexed_dims)))
        else:
            while value.ndim < indexed_dims:
                # broadcasting
                # expand missing dimensions to align split axis
                value = value.expand_dims(0)
            value_shape = tuple(torch.broadcast_shapes(value.shape, output_shape))
    return value, is_scalar


def _resolve_duplicate_indices(
    key_in,
    rhs_in: torch.Tensor,
    target_shape: tuple[int, ...],
):
    """
    CUDA-safe handling for duplicate advanced indices:
    enforce NumPy semantics (last assignment wins) by dropping earlier duplicates.
    Works for:
        - key_in: torch.Tensor (indexes axis 0)
        - key_in: tuple/list of torch.Tensors (pure advanced indexing)
    rhs_in must match the indexing result shape.
    """
    # Scalars or single element: no need to deduplicate
    if not torch.is_tensor(rhs_in) or rhs_in.numel() <= 1:
        return key_in, rhs_in

    # Normalize key to tuple of tensors
    if torch.is_tensor(key_in):
        idx_tensors = (key_in,)
    elif (
        isinstance(key_in, (tuple, list))
        and len(key_in) > 0
        and all(torch.is_tensor(k) for k in key_in)
    ):
        idx_tensors = tuple(key_in)
    else:
        # Not pure advanced-tensor indexing -> don't touch
        return key_in, rhs_in

    device = rhs_in.device

    # Broadcast indices to common shape
    try:
        idx_b = torch.broadcast_tensors(*idx_tensors)
    except RuntimeError:
        # If broadcast fails, leave it to PyTorch (will error appropriately)
        return key_in, rhs_in

    pos_shape = idx_b[0].shape
    pos_ndim = len(pos_shape)
    n = idx_b[0].numel()

    idx_flat = [
        torch.where(t < 0, t + int(target_shape[d]), t)
        .to(device=device, dtype=torch.int64)
        .reshape(-1)
        for d, t in enumerate(idx_b)
    ]

    # Build linear index for duplicate detection
    if len(idx_flat) == 1:
        lin = idx_flat[0]
    else:
        lin = idx_flat[0]
        # linearize across the first len(idx_flat) dimensions of the target tensor
        for d in range(1, len(idx_flat)):
            lin = lin * int(target_shape[d]) + idx_flat[d]

    # Determine sorting order (stable sort preserves original order)
    order = torch.argsort(lin, stable=True)
    pos = None

    lin_s = lin[order]

    # Fast path: check adjacent elements in sorted order
    # If all adjacent elements are distinct, there are no duplicates
    if (lin_s[1:] != lin_s[:-1]).all():
        return key_in, rhs_in

    if pos is None:
        pos = torch.arange(n, device=device, dtype=torch.int64)

    pos_s = pos[order]

    is_last = torch.ones_like(lin_s, dtype=torch.bool)
    is_last[:-1] = lin_s[1:] != lin_s[:-1]
    keep_pos = pos_s[is_last]  # positions in original stream

    # Reduce RHS accordingly:
    # Flatten leading "pos_ndim" dims into one, keep trailing dims as payload
    rhs_view = rhs_in.reshape(n, *rhs_in.shape[pos_ndim:])
    rhs_u = rhs_view[keep_pos].reshape(keep_pos.numel(), *rhs_in.shape[pos_ndim:])

    # Reduce indices accordingly (use flattened 1D indices)
    if torch.is_tensor(key_in):
        key_u = idx_flat[0][keep_pos]
        return key_u, rhs_u

    key_u = tuple(t[keep_pos] for t in idx_flat)
    return key_u, rhs_u


def _setitem_advanced_unordered_local(
    x_local: torch.Tensor,
    split_key: torch.Tensor,
    value_torch: torch.Tensor,
    *,
    split_axis: int,
    value_key_start_dim: int,
    local_offset: int,
    local_size: int,
    value_is_scalar: bool,
    out_dtype: torch.dtype,
    base_index: tuple | None = None,
) -> None:
    """
    The function is a helper that updates ``x_local`` in-place according to the logical advanced
    indexing pattern encoded by ``split_key`` and the broadcasted ``value_torch``.
    This helper operates exclusively on local ``torch.Tensor`` views:
    - ``x_local`` is the local slice of the distributed array on this rank.
    - ``split_key`` contains GLOBAL indices along the split axis.
    - Only those indices that fall into ``[local_offset, local_offset + local_size)``
        are applied on this rank.
    """
    # 1) Local mask: which global indices in `split_key` belong to this rank?
    global_indices = split_key
    local_mask = (global_indices >= local_offset) & (global_indices < local_offset + local_size)

    coord = local_mask.nonzero(as_tuple=True)

    if coord[0].numel() == 0:
        # Nothing to do on this rank, exit early.
        return

    # 2) Map global → local indices along the split axis
    global_split_indices = global_indices[coord]
    local_split_indices = global_split_indices - local_offset

    # build LHS index for x_local (corresponds to self.larray)
    lhs_index = list(base_index)

    lhs_index[split_axis] = local_split_indices
    lhs_index = tuple(lhs_index)

    # build RHS index for value_torch
    if value_is_scalar:
        rhs = value_torch.to(out_dtype)
    else:
        rhs_index = [slice(None)] * value_torch.ndim
        m = split_key.ndim

        for d in range(m):
            rhs_index[value_key_start_dim + d] = coord[d]

        rhs = value_torch[tuple(rhs_index)].to(out_dtype)

    if x_local.is_cuda:
        lhs_index, rhs = _resolve_duplicate_indices(lhs_index, rhs, x_local.shape)

    x_local[lhs_index] = rhs
