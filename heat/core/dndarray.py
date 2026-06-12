"""Provides HeAT's core data structure, the DNDarray, a distributed n-dimensional array"""

from __future__ import annotations

import numpy as np
import torch
import warnings

from mpi4py import MPI
from typing import TypeVar, Any, Union
from collections.abc import Iterable

warnings.simplefilter("always", ResourceWarning)

# NOTE: heat module imports need to be placed at the very end of the file to avoid cyclic dependencies
__all__ = ["DNDarray"]

Communication = TypeVar("Communication")

# Type aliases
Index = Union[int, slice, type(...), None, torch.Tensor, np.ndarray, "DNDarray"]
Indexer = Union[Index, tuple[Index, ...], list[Index]]


class LocalIndex:
    """
    Indexing class for local operations (primarily for :func:`lloc` function)
    For docs on ``__getitem__`` and ``__setitem__`` see :func:`lloc`
    """

    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, key):
        return self.obj[key]

    def __setitem__(self, key, value):
        self.obj[key] = value


from typing import NamedTuple


class ProcessedKey(NamedTuple):
    """
    A named tuple to store the processed key information for distributed indexing operations.
    """

    key: Any
    op_type: str  # "scalar", "slice", "descending_slice", "distr_mask", "local_mask", "advanced", "distributed"
    output_shape: tuple
    output_split: int | None
    split_key_is_ordered: int
    key_is_mask_like: bool
    out_is_balanced: bool
    root: int | None
    backwards_transpose_axes: tuple


def _process_scalar_key(
    arr: "DNDarray",
    key: int | "DNDarray" | torch.Tensor | np.ndarray,
    indexed_axis: int,
    return_local_indices: bool | None = False,
) -> tuple[int, int]:
    """
    Private helper function to process a single-item scalar key used for indexing a ``DNDarray``.
    """
    device = arr.larray.device
    try:
        # is key an ndarray or DNDarray or torch.Tensor?
        key = key.item()
    except AttributeError:
        # key is already an integer, do nothing
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
        if key in displs:
            root = displs.index(key)
        else:
            displs = torch.cat(
                (
                    torch.tensor(displs, device=device),
                    torch.tensor(key, device=device).reshape(-1),
                ),
                dim=0,
            )
            _, sorted_indices = displs.unique(sorted=True, return_inverse=True)
            root = sorted_indices[-1].item() - 1
            displs = displs.tolist()
        # correct key for rank-specific displacement
        if return_local_indices:
            if arr.comm.rank == root:
                key -= displs[root]
    else:
        root = None
    return key, root


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
    if rhs_in.numel() <= 1:
        return key_in, rhs_in

    # Normalize key to either a single tensor or tuple of tensors
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

    # Broadcast indices to common shape, then flatten
    try:
        idx_b = torch.broadcast_tensors(*idx_tensors)
    except RuntimeError:
        # If broadcast fails, leave it to PyTorch (will error appropriately)
        return key_in, rhs_in

    pos_shape = idx_b[0].shape
    pos_ndim = len(pos_shape)
    n = idx_b[0].numel()

    idx_flat = [t.to(device=device, dtype=torch.int64).reshape(-1) for t in idx_b]

    # Build linear index for duplicate detection
    if len(idx_flat) == 1:
        lin = idx_flat[0]
    else:
        lin = idx_flat[0]
        # linearize across the first len(idx_flat) dimensions of the target tensor
        for d in range(1, len(idx_flat)):
            lin = lin * int(target_shape[d]) + idx_flat[d]

    # Fast path: no duplicates
    if torch.unique(lin).numel() == n:
        return key_in, rhs_in

    # Determine "last occurrence" per linear index (last wins)
    pos = torch.arange(n, device=device, dtype=torch.int64)

    # Prefer stable sort by lin if available; otherwise sort by combined key
    try:
        order = torch.argsort(lin, stable=True)
    except TypeError:
        # combined key sorts by lin, then by pos
        combined = lin.to(torch.int64) * (n + 1) + pos
        order = torch.argsort(combined)

    lin_s = lin[order]
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


def _resolve_indexing_state(
    arr: "DNDarray",
    key: Indexer,
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
                ``"descending_slice"``, ``"distr_mask"``, ``"local_mask"``, ``"advanced"``, or ``"distributed"``).
            - output_shape (tuple): The global shape of the resulting array.
            - output_split (int or None): The split axis of the resulting array.
            - split_key_is_ordered (int): Monotonicity of the split key (``1``: ascending, ``0``: unordered,
                ``-1``: descending).
            - key_is_mask_like (bool): Whether the key acts as a boolean mask.
            - out_is_balanced (bool): Whether the resulting ``DNDarray`` maintains load balance.
            - root (int or None): The root MPI process ID if single-element indexing along the split
                axis isolate data to one rank.
            - backwards_transpose_axes (tuple): The axes required to transpose ``arr`` back to its
                original shape if advanced indexing triggered a transposition.
    """
    # early out for scalar key
    is_scalar = np.isscalar(key) or getattr(key, "ndim", 1) == 0

    is_boolean = isinstance(key, bool) or (
        hasattr(key, "dtype")
        and key.dtype in (ht_bool, ht_uint8, torch.bool, torch.uint8, np.bool_, np.uint8)
    )

    if is_scalar and not is_boolean:
        if arr.ndim == 0 and op == "get":
            raise IndexError(
                "Too many indices for DNDarray: DNDarray is 0-dimensional, but 1 were indexed"
            )

        output_shape = arr.gshape[1:]
        output_split = None if arr.split in (None, 0) else arr.split - 1
        key, root = _process_scalar_key(
            arr, key, indexed_axis=0, return_local_indices=return_local_indices
        )

        return arr, ProcessedKey(
            key=key,
            op_type="scalar",
            output_shape=tuple(output_shape),
            output_split=output_split,
            split_key_is_ordered=1,
            key_is_mask_like=False,
            out_is_balanced=True,
            root=root,
            backwards_transpose_axes=tuple(range(arr.ndim)),
        )

    # evaluate if this is a distributed fast-path mask before we modify the key

    distr_mask_fast_path = False
    # mask along split axis within tuple?
    if arr.is_distributed():
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
            # exact shape match
            if split_key.gshape == arr.gshape:
                # "get" flattens to 1D
                # if split > 0, local flattening scrambles global C-order
                if op == "set" or (op == "get" and arr.split == 0):
                    distr_mask_fast_path = True
            elif (
                split_key.ndim == 1
                and arr.split == 0
                and split_key.gshape == (arr.gshape[arr.split],)
            ):
                # 1D mask on split=0
                distr_mask_fast_path = True

            # early out if mask and not tuple key
            if distr_mask_fast_path and not isinstance(key, tuple):
                return arr, ProcessedKey(
                    key=key.larray,
                    op_type="distr_mask",
                    output_shape=(),  # Dummy shape, bypassed safely in __setitem__
                    output_split=0 if op == "get" else arr.split,
                    split_key_is_ordered=0,
                    key_is_mask_like=True,
                    out_is_balanced=False,
                    root=None,
                    backwards_transpose_axes=tuple(range(arr.ndim)),
                )

    # normalize index components
    if isinstance(key, DNDarray):
        if key.dtype not in (ht_bool, ht_uint8) and key.split is None:
            key = key.larray.to(torch.int64)
    elif isinstance(key, (list, tuple)):
        key = type(key)(
            k.larray.to(torch.int64)
            if isinstance(k, DNDarray) and k.dtype not in (ht_bool, ht_uint8) and k.split is None
            else k
            for k in key
        )

    # 1D boolean mask resolution
    first = key[0] if isinstance(key, tuple) and len(key) >= 1 else key
    if isinstance(first, (DNDarray, torch.Tensor, np.ndarray)) and arr.ndim >= 1:
        first_dtype = getattr(first, "dtype", None)
        first_ndim = getattr(first, "ndim", 0)
        first_shape = tuple(getattr(first, "shape", ()))

        if (
            not distr_mask_fast_path
            and first_ndim == 1
            and first_shape == (arr.gshape[0],)
            and first_dtype in (ht_bool, ht_uint8, torch.bool, torch.uint8, np.bool_, np.uint8)
        ):
            if isinstance(first, DNDarray):
                nz = first.nonzero()
                if isinstance(nz, tuple):
                    nz = nz[0]
                if getattr(nz, "ndim", 1) > 1 and nz.shape[-1] == 1:
                    nz = nz.squeeze(-1)
                idx0 = nz
            elif isinstance(first, torch.Tensor):
                idx0 = torch.nonzero(first, as_tuple=False).flatten()
            else:  # np.ndarray
                idx0 = np.nonzero(first)[0].astype(np.int64)

            key = (idx0,) + key[1:] if isinstance(key, tuple) else (idx0,)

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
    backwards_transpose_axes = tuple(range(arr.ndim))

    if isinstance(key, list):
        try:
            key = torch.tensor(key, device=arr.larray.device)
        except RuntimeError:
            raise IndexError("Invalid indices: expected a list of integers, got {}".format(key))

    if isinstance(key, (DNDarray, torch.Tensor, np.ndarray)):
        if key.dtype in (ht_bool, ht_uint8, torch.bool, torch.uint8, np.bool_, np.uint8):
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
                        try:
                            key_numel = key.numel()
                        except AttributeError:
                            key_numel = key.size
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
                    elif not isinstance(key, torch.Tensor):
                        key = torch.as_tensor(key, device=arr.larray.device)
                        out_is_balanced = True
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
                        try:
                            sorted_k, _ = torch.sort(key, stable=True)
                        except TypeError:
                            sorted_k, _ = torch.sort(key)
                        split_key_is_ordered = int((key == sorted_k).all().item())

                    # unordered local keys
                    if not split_key_is_ordered and not key_is_dist:
                        if op == "get":
                            # prepare for distributed non-ordered indexing: distribute local key
                            key = factories.array(key, split=new_split, device=arr.device).larray
                            out_is_balanced = True
                        else:
                            out_is_balanced = True

                    # ordered keys
                    if split_key_is_ordered:
                        # extract local key
                        cond1 = key >= displs[arr.comm.rank]
                        cond2 = key < displs[arr.comm.rank] + counts[arr.comm.rank]
                        key = key[cond1 & cond2]
                        if return_local_indices:
                            key -= displs[arr.comm.rank]
                        out_is_balanced = False
            else:
                try:
                    out_is_balanced = key.balanced
                    new_split = key.split
                    key = key.larray
                except AttributeError:
                    # torch or numpy key, non-distributed indexed array
                    out_is_balanced = True
                    new_split = None

            # define indexing type
            if root is not None:
                op_type = "scalar"
            elif split_key_is_ordered == 0:
                op_type = "distributed"
            elif key_is_mask_like:
                op_type = "local_mask"
            else:
                op_type = "advanced"

            return arr, ProcessedKey(
                key=key,
                op_type=op_type,
                output_shape=tuple(output_shape),
                output_split=new_split,
                split_key_is_ordered=split_key_is_ordered,
                key_is_mask_like=key_is_mask_like,
                out_is_balanced=out_is_balanced,
                root=root,
                backwards_transpose_axes=backwards_transpose_axes,
            )

    if isinstance(key, (tuple, list)):
        key = list(key)
    else:
        key = [key]

    # check for ellipsis, newaxis. NB: (np.newaxis is None)==True
    def is_0d_bool(k):
        if isinstance(k, bool):
            return True
        if hasattr(k, "dtype") and k.dtype in (
            ht_bool,
            ht_uint8,
            torch.bool,
            torch.uint8,
            np.bool_,
            np.uint8,
        ):
            if getattr(k, "ndim", 1) == 0:
                return True
        return False

    add_dims = sum(k is None or is_0d_bool(k) for k in key)
    ellipsis = sum(isinstance(k, type(...)) for k in key)
    if ellipsis > 1:
        raise ValueError("indexing key can only contain 1 Ellipsis (...)")
    if ellipsis:
        # key contains exactly 1 ellipsis
        # replace with explicit `slice(None)` for affected dimensions
        # output_shape, split_bookkeeping not affected
        expand_key = [slice(None)] * (arr.ndim + add_dims)
        ellipsis_index = key.index(...)
        ellipsis_dims = arr.ndim - (len(key) - ellipsis - add_dims)
        expand_key[:ellipsis_index] = key[:ellipsis_index]
        expand_key[ellipsis_index + ellipsis_dims :] = key[ellipsis_index + 1 :]
        key = expand_key
    while add_dims > 0:
        # expand array dims: output_shape, split_bookkeeping to reflect newaxis
        # replace newaxis with slice(None), replace 0-D bools with a target slice
        for i, k in reversed(list(enumerate(key))):
            if k is None or is_0d_bool(k):
                if k is None:
                    key[i] = slice(None)
                else:
                    val = bool(k.item() if hasattr(k, "item") else k)
                    key[i] = slice(None) if val else slice(0, 0)

                arr = arr.expand_dims(i - add_dims + 1)
                output_shape = (
                    output_shape[: i - add_dims + 1] + [1] + output_shape[i - add_dims + 1 :]
                )
                split_bookkeeping = (
                    split_bookkeeping[: i - add_dims + 1]
                    + [None]
                    + split_bookkeeping[i - add_dims + 1 :]
                )
                add_dims -= 1

    # recalculate new_split, transpose_axes after dimensions manipulation
    new_split = split_bookkeeping.index("split") if "split" in split_bookkeeping else None
    transpose_axes, backwards_transpose_axes = tuple(range(arr.ndim)), tuple(range(arr.ndim))
    # check for advanced indexing and slices
    advanced_indexing_dims = []
    advanced_indexing_shapes = []
    lose_dims = 0
    for i, k in enumerate(key):
        if isinstance(k, DNDarray) and k.ndim == 0:
            k = k.larray.item()
            key[i] = k
        # for robustness: handle list/tuple keys that contain DNDarrays
        elif isinstance(k, (list, tuple)) and any(isinstance(kk, DNDarray) for kk in k):
            # Case 1: singleton container (common from where/nonzero): (idx,) -> idx
            if len(k) == 1 and isinstance(k[0], DNDarray):
                k = k[0]
                key[i] = k

            else:
                # Case 2: sequence of scalar DNDarrays -> unwrap to python scalars
                new_k = []
                all_scalar = True
                for kk in k:
                    if isinstance(kk, DNDarray):
                        if kk.ndim != 0:
                            all_scalar = False
                            break
                        new_k.append(kk.larray.item())
                    else:
                        new_k.append(kk)

                if all_scalar:
                    k = new_k
                    key[i] = k
                else:
                    # This is an ambiguous nested "tuple of index arrays" inside a single axis.
                    # In NumPy semantics such tuples belong at TOP LEVEL (arr[idx0, idx1, ...]),
                    # not nested as one axis key.
                    raise TypeError(
                        "Nested tuple/list of non-scalar DNDarray indices is not supported. "
                        "Pass them as separate indices (e.g. arr[idx0, idx1, ...]) or unwrap "
                        "singleton tuples (e.g. idx = idx[0])."
                    )

        if np.isscalar(k) or getattr(k, "ndim", 1) == 0:
            # single-element indexing along axis i
            try:
                output_shape[i], split_bookkeeping[i] = None, None
            except IndexError:
                raise IndexError(
                    f"Too many indices for DNDarray: DNDarray is {arr.ndim}-dimensional, but {len(key)} dimensions were indexed"
                )
            lose_dims += 1
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
            if k.dtype in (types.int32, types.int64) and k.ndim >= 1:
                dim = arr.gshape[i]

                # compute local flags even if k.larray is empty (any() on empty -> False)
                invalid_local = ((k.larray < -dim) | (k.larray >= dim)).any().item()
                has_neg_local = (k.larray < 0).any().item()

                # Decide once, then ALL ranks take the same path for collectives
                do_reduce = (
                    arr.comm is not None and getattr(arr.comm, "size", 1) > 1 and k.is_distributed()
                )

                if do_reduce:
                    invalid_sum = arr.comm.allreduce(int(invalid_local), op=MPI.SUM)
                    has_neg_sum = arr.comm.allreduce(int(has_neg_local), op=MPI.SUM)
                else:
                    invalid_sum = int(invalid_local)
                    has_neg_sum = int(has_neg_local)

                if invalid_sum > 0:
                    raise IndexError(f"index out of bounds for axis {i} with size {dim}")

                if has_neg_sum > 0:
                    k_l = k.larray.clone()
                    k_l[k_l < 0] += dim
                    k = factories.array(
                        k_l,
                        dtype=k.dtype,
                        split=k.split,
                        device=arr.device,
                        comm=arr.comm,
                        copy=False,
                    )

            advanced_indexing_shapes.append(k.gshape)
            if arr_is_distributed and i == arr.split:
                if (
                    not k.is_distributed()
                    and k.ndim == 1
                    and (k.larray == torch.sort(k.larray, stable=True)[0]).all()
                ):
                    split_key_is_ordered = 1
                    out_is_balanced = False
                else:
                    split_key_is_ordered = 0

                    # redistribute key along last axis to match split axis of indexed array
                    k = k.resplit(-1)
                    out_is_balanced = True
            key[i] = k

        elif isinstance(k, slice) and k != slice(None):
            if k.step == 0:
                raise ValueError("Slice step cannot be zero")
            start, stop, step = slice(k.start, k.stop, k.step).indices(arr.gshape[i])

            if step < 0 and start > stop:
                # PyTorch doesn't support negative step
                key[i] = torch.arange(
                    start, stop, step, device=arr.larray.device, dtype=torch.int64
                )
                output_shape[i] = len(key[i])

                if arr_is_distributed and new_split == i:
                    split_key_is_ordered = -1
                    # flip key and keep process-local indices
                    key[i] = key[i].flip(0)
                    cond1 = key[i] >= displs[arr.comm.rank]
                    cond2 = key[i] < displs[arr.comm.rank] + counts[arr.comm.rank]
                    key[i] = key[i][cond1 & cond2]
                    if return_local_indices:
                        key[i] -= displs[arr.comm.rank]
                    # slices can result in unbalanced chunks
                    out_is_balanced = False

            elif step > 0 and start < stop:
                output_shape[i] = len(range(start, stop, step))

                if arr_is_distributed and new_split == i:
                    split_key_is_ordered = 1
                    out_is_balanced = False
                    local_arr_end = displs[arr.comm.rank] + counts[arr.comm.rank]
                    if stop > displs[arr.comm.rank] and start < local_arr_end:
                        index_in_cycle = (displs[arr.comm.rank] - start) % step
                        if start >= displs[arr.comm.rank]:
                            # slice begins on current rank
                            local_start = start - displs[arr.comm.rank]
                        else:
                            local_start = 0 if index_in_cycle == 0 else step - index_in_cycle
                        if stop <= local_arr_end:
                            # slice ends on current rank
                            local_stop = stop - displs[arr.comm.rank]
                        else:
                            local_stop = counts[arr.comm.rank]

                        key[i] = slice(local_start, local_stop, step)
                    else:
                        key[i] = slice(0, 0)
            elif step == 0:
                raise ValueError("Slice step cannot be zero")
            else:
                key[i] = slice(0, 0)
                output_shape[i] = 0

    if advanced_indexing:
        # adv indexing key elements are DNDarrays: extract torch tensors
        # options: 1. key is mask-like (covers boolean mask as well), 2. adv indexing along split axis, 3. everything else
        # 1. define key as mask-like if each element of key is a DNDarray, and all elements of key are of the same shape, and the advanced-indexing dimensions are consecutive
        key_is_mask_like = key_is_mask_like or (
            len(advanced_indexing_dims) > 1
            and all(isinstance(k, DNDarray) for k in key)
            and len(set(k.shape for k in key)) == 1
            and torch.tensor(advanced_indexing_dims).diff().eq(1).all().item()
        )
        # if split axis is affected by advanced indexing, keep track of non-split dimensions for later
        if arr.is_distributed() and arr.split in advanced_indexing_dims:
            non_split_dims = list(advanced_indexing_dims).copy()
            if arr.split is not None:
                non_split_dims.remove(arr.split)
        # 1. key is mask-like
        if key_is_mask_like:
            key = list(key)
            key_splits = [k.split for k in key]
            if arr.split is not None and arr.split in advanced_indexing_dims:
                split_key_pos = advanced_indexing_dims.index(arr.split)

                if not key_splits.count(key_splits[split_key_pos]) == len(key_splits):
                    if (
                        key_splits[arr.split] is not None
                        and key_splits.count(None) == len(key_splits) - 1
                    ):
                        for i in non_split_dims:
                            key[i] = factories.array(
                                key[i],
                                split=key_splits[arr.split],
                                device=arr.device,
                                comm=arr.comm,
                                copy=None,
                            )
                    else:
                        raise IndexError(
                            f"Indexing arrays must be distributed along the same dimension, got splits {key_splits}."
                        )
                else:
                    # all key_splits must be the same, otherwise raise IndexError
                    if not key_splits.count(key_splits[0]) == len(key_splits):
                        raise IndexError(
                            f"Indexing arrays must be distributed along the same dimension, got splits {key_splits}."
                        )
            # all key elements are now DNDarrays of the same shape, same split axis
        # 2. advanced indexing along split axis
        if arr.is_distributed() and arr.split in advanced_indexing_dims:
            if distr_mask_fast_path:
                # mask is already a local tensor, just extract any other advanced indices
                for i in non_split_dims:
                    if isinstance(key[i], DNDarray):
                        key[i] = key[i].larray
            elif split_key_is_ordered == 1:
                # extract torch tensors, keep process-local indices only
                k = key[arr.split].larray
                cond1 = k >= displs[arr.comm.rank]
                cond2 = k < displs[arr.comm.rank] + counts[arr.comm.rank]
                k = k[cond1 & cond2]
                if return_local_indices:
                    k -= displs[arr.comm.rank]
                key[arr.split] = k
                for i in non_split_dims:
                    if key_is_mask_like:
                        # select the same elements along non-split dimensions
                        key[i] = key[i].larray[cond1 & cond2]
                    else:
                        key[i] = key[i].larray
            elif split_key_is_ordered == 0:
                # extract torch tensors, any other communication + mask-like case are handled in __getitem__ or __setitem__
                for i in advanced_indexing_dims:
                    key[i] = key[i].larray
            # split_key_is_ordered == -1 not treated here as it is slicing, not advanced indexing
        else:
            # advanced indexing does not affect split axis, return torch tensors
            for i in advanced_indexing_dims:
                key[i] = key[i].larray
        # all adv indexing keys are now torch tensors

        # shapes of adv indexing arrays must be broadcastable
        try:
            broadcasted_shape = torch.broadcast_shapes(*advanced_indexing_shapes)
        except RuntimeError:
            raise IndexError(
                "Shape mismatch: indexing arrays could not be broadcast together with shapes: {}".format(
                    advanced_indexing_shapes
                )
            )
        add_dims = len(broadcasted_shape) - len(advanced_indexing_dims)
        if (
            len(advanced_indexing_dims) == 1
            or list(range(advanced_indexing_dims[0], advanced_indexing_dims[-1] + 1))
            == advanced_indexing_dims
        ):
            # dimensions affected by advanced indexing are consecutive:
            output_shape[
                advanced_indexing_dims[0] : advanced_indexing_dims[0] + len(advanced_indexing_dims)
            ] = broadcasted_shape
            if key_is_mask_like:
                # advanced indexing dimensions will be collapsed into one dimension
                if (
                    "split" in split_bookkeeping
                    and split_bookkeeping.index("split") in advanced_indexing_dims
                ):
                    split_bookkeeping[
                        advanced_indexing_dims[0] : advanced_indexing_dims[0]
                        + len(advanced_indexing_dims)
                    ] = ["split"]
                else:
                    split_bookkeeping[
                        advanced_indexing_dims[0] : advanced_indexing_dims[0]
                        + len(advanced_indexing_dims)
                    ] = [None]
            else:
                split_bookkeeping = (
                    split_bookkeeping[: advanced_indexing_dims[0]]
                    + [None] * add_dims
                    + split_bookkeeping[advanced_indexing_dims[0] :]
                )
        else:
            # advanced-indexing dimensions are not consecutive:
            # transpose array to make the advanced-indexing dimensions consecutive as the first dimensions
            non_adv_ind_dims = list(i for i in range(arr.ndim) if i not in advanced_indexing_dims)
            # keep track of transpose axes order, to be able to transpose back later
            transpose_axes = tuple(advanced_indexing_dims + non_adv_ind_dims)
            arr = arr.transpose(transpose_axes)
            backwards_transpose_axes = tuple(
                torch.tensor(transpose_axes, device=arr.larray.device).argsort(stable=True).tolist()
            )
            # output shape and split bookkeeping
            output_shape = list(output_shape[i] for i in transpose_axes)
            output_shape[: len(advanced_indexing_dims)] = broadcasted_shape
            split_bookkeeping = list(split_bookkeeping[i] for i in transpose_axes)
            split_bookkeeping = [None] * add_dims + split_bookkeeping
            # modify key to match the new dimension order
            key = [key[i] for i in advanced_indexing_dims] + [key[i] for i in non_adv_ind_dims]
            # update advanced-indexing dims
            advanced_indexing_dims = list(range(len(advanced_indexing_dims)))

    # expand key to match the number of dimensions of the DNDarray
    if arr.ndim > len(key):
        key += [slice(None)] * (arr.ndim - len(key))

    key = tuple(key)
    for i in range(output_shape.count(None)):
        lost_dim = output_shape.index(None)
        output_shape.remove(None)
        split_bookkeeping = split_bookkeeping[:lost_dim] + split_bookkeeping[lost_dim + 1 :]
    output_shape = tuple(output_shape)
    new_split = split_bookkeeping.index("split") if "split" in split_bookkeeping else None

    if root is not None:
        op_type = "scalar"
    elif split_key_is_ordered == 0:
        op_type = "distributed"
    elif split_key_is_ordered == -1:
        op_type = "descending_slice"
    elif key_is_mask_like:
        op_type = "distr_mask" if distr_mask_fast_path else "local_mask"
    else:
        op_type = "advanced"

    return arr, ProcessedKey(
        key=tuple(key),
        op_type=op_type,
        output_shape=tuple(output_shape),
        output_split=new_split,
        split_key_is_ordered=split_key_is_ordered,
        key_is_mask_like=key_is_mask_like,
        out_is_balanced=out_is_balanced,
        root=root,
        backwards_transpose_axes=backwards_transpose_axes,
    )


class DNDarray:
    """
    Distributed N-Dimensional array. The core element of Heat. It is composed of
    PyTorch tensors local to each process.

    Parameters
    ----------
    array : torch.Tensor
        Local array elements
    gshape : tuple[int,...]
        The global shape of the array
    dtype : datatype
        The datatype of the array
    split : int or None
        The axis on which the array is divided between processes
    device : Device
        The device on which the local arrays are using (cpu or gpu)
    comm : Communication
        The communications object for sending and receiving data
    balanced: bool or None
        Describes whether the data are evenly distributed across processes.
        If this information is not available (``self.balanced is None``), it
        can be gathered via the :func:`is_balanced()` method (requires communication).
    """

    def __init__(
        self,
        array: torch.Tensor,
        gshape: tuple[int, ...],
        dtype: datatype,
        split: int | None,
        device: Device,
        comm: Communication,
        balanced: bool,
    ):
        self.__array = array
        self.__gshape = gshape
        self.__dtype = dtype
        self.__split = split
        self.__device = device
        self.__comm = comm
        self.__balanced: bool = balanced
        self.__ishalo = False
        self.__halo_next: torch.Tensor | None = None
        self.__halo_prev: torch.Tensor | None = None
        self.__partitions_dict__ = None
        self.__lshape_map = None

        # check for inconsistencies between torch and heat devices
        assert str(array.device) == device.torch_device

    @property
    def balanced(self) -> bool:
        """
        Boolean value indicating if the DNDarray is balanced between the MPI processes
        """
        return self.__balanced

    @property
    def comm(self) -> Communication:
        """
        The :class:`~heat.core.communication.Communication` of the ``DNDarray``
        """
        return self.__comm

    @property
    def device(self) -> Device:
        """
        The :class:`~heat.core.devices.Device` of the ``DNDarray``
        """
        return self.__device

    @property
    def dtype(self) -> datatype:
        """
        The :class:`~heat.core.types.datatype` of the ``DNDarray``
        """
        return self.__dtype

    @property
    def gshape(self) -> tuple:
        """
        Returns the global shape of the ``DNDarray`` across all processes
        """
        return self.__gshape

    @property
    def halo_next(self) -> torch.Tensor:
        """
        Returns the halo of the next process
        """
        return self.__halo_next

    @property
    def halo_prev(self) -> torch.Tensor:
        """
        Returns the halo of the previous process
        """
        return self.__halo_prev

    @property
    def larray(self) -> torch.Tensor:
        """
        Returns the underlying process-local ``torch.Tensor`` of the ``DNDarray``
        """
        return self.__array

    @larray.setter
    def larray(self, array: torch.Tensor):
        """
        Setter for ``self.larray``, the underlying local ``torch.Tensor`` of the ``DNDarray``.

        Parameters
        ----------
        array : torch.Tensor
            The new underlying local ``torch.tensor`` of the ``DNDarray``

        Warning
        -----------
        Please use this function with care, as it might corrupt/invalidate the metadata in the ``DNDarray`` instance.
        """
        # sanitize tensor input
        sanitation.sanitize_in_tensor(array)
        # verify consistency of tensor shape with global DNDarray
        sanitation.sanitize_lshape(self, array)
        # set balanced status
        split = self.split
        if split is not None and array.shape[split] != self.lshape[split]:
            self.__balanced = None
        self.__array = array

    @property
    def nbytes(self) -> int:
        """
        Returns the number of bytes consumed by the global tensor. Equivalent to property gnbytes.

        Note
        ------------
            Does not include memory consumed by non-element attributes of the ``DNDarray`` object.
        """
        return self.__array.element_size() * self.size

    @property
    def ndim(self) -> int:
        """
        Number of dimensions of the ``DNDarray``
        """
        return len(self.__gshape)

    @property
    def __partitioned__(self) -> dict:
        """
        Return a dictionary containing information useful for working with the partitioned
        data. These items include the shape of the data on each process, the starting index of the data
        that a process has, the datatype of the data, the local devices, as well as the global
        partitioning scheme.

        An example of the output and shape is shown in :func:`ht.core.DNDarray.create_partition_interface <ht.core.DNDarray.create_partition_interface>`.

        Returns
        -------
        dictionary with the partition interface
        """
        if self.__partitions_dict__ is None:
            self.__partitions_dict__ = self.create_partition_interface()
        return self.__partitions_dict__

    @property
    def size(self) -> int:
        """
        Number of total elements of the ``DNDarray``
        """
        if self.larray.is_mps:
            # MPS does not support double precision
            size = torch.prod(
                torch.tensor(self.gshape, dtype=torch.float32, device=self.device.torch_device)
            )
        else:
            size = torch.prod(
                torch.tensor(self.gshape, dtype=torch.float64, device=self.device.torch_device)
            )
        return size.long().item()

    @property
    def gnbytes(self) -> int:
        """
        Returns the number of bytes consumed by the global ``DNDarray``

        Note
        -----------
            Does not include memory consumed by non-element attributes of the ``DNDarray`` object.
        """
        return self.nbytes

    @property
    def gnumel(self) -> int:
        """
        Returns the number of total elements of the ``DNDarray``
        """
        return self.size

    @property
    def imag(self) -> DNDarray:
        """
        Return the imaginary part of the ``DNDarray``.
        """
        return complex_math.imag(self)

    @property
    def lnbytes(self) -> int:
        """
        Returns the number of bytes consumed by the local ``torch.Tensor``

        Note
        -------------------
            Does not include memory consumed by non-element attributes of the ``DNDarray`` object.
        """
        return self.__array.element_size() * self.__array.nelement()

    @property
    def lnumel(self) -> int:
        """
        Number of elements of the ``DNDarray`` on each process
        """
        return np.prod(self.__array.shape)

    @property
    def lloc(self) -> "DNDarray" | None:
        """
        Local item setter and getter. i.e. this function operates on a local
        level and only on the PyTorch tensors composing the :class:`DNDarray`.
        This function uses the LocalIndex class. As getter, it returns a ``DNDarray``
        with the indices selected at a *local* level

        Parameters
        ----------
        key : int or slice or tuple[int,...]
            Indices of the desired data.
        value : scalar, optional
            All types compatible with pytorch tensors, if none given then this is a getter function

        Examples
        --------
        >>> a = ht.zeros((4, 5), split=0)
        DNDarray([[0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.]], dtype=ht.float32, device=cpu:0, split=0)
        >>> a.lloc[1, 0:4]
        (1/2) tensor([0., 0., 0., 0.])
        (2/2) tensor([0., 0., 0., 0.])
        >>> a.lloc[1, 0:4] = torch.arange(1, 5)
        >>> a
        DNDarray([[0., 0., 0., 0., 0.],
                  [1., 2., 3., 4., 0.],
                  [0., 0., 0., 0., 0.],
                  [1., 2., 3., 4., 0.]], dtype=ht.float32, device=cpu:0, split=0)
        """
        return LocalIndex(self.__array)

    @property
    def lshape(self) -> tuple[int]:
        """
        Returns the shape of the ``DNDarray`` on each node
        """
        return tuple(self.__array.shape)

    @property
    def lshape_map(self) -> torch.Tensor:
        """
        Returns the lshape map. If it hasn't been previously created then it will be created here.
        """
        return self.create_lshape_map()

    @property
    def real(self) -> DNDarray:
        """
        Return the real part of the ``DNDarray``.
        """
        return complex_math.real(self)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the ``DNDarray`` as a whole
        """
        return self.__gshape

    @property
    def split(self) -> int | None:
        """
        Returns the axis on which the ``DNDarray`` is split
        """
        return self.__split

    def stride(self) -> tuple[int, ...]:
        """
        Returns the steps in each dimension when traversing a ``DNDarray``. torch-like usage: ``self.stride()``
        """
        return self.__array.stride()

    @property
    def strides(self) -> tuple[int, ...]:
        """
        Returns bytes to step in each dimension when traversing a ``DNDarray``. numpy-like usage: ``self.strides()``
        """
        steps = list(self.__array.stride())
        try:
            itemsize = self.__array.untyped_storage().element_size()
        except AttributeError:
            itemsize = self.__array.storage().element_size()
        strides = tuple(step * itemsize for step in steps)
        return strides

    @property
    def T(self):
        """
        Reverse the dimensions of a DNDarray.
        """
        # specialty docs for this version of transpose. The transpose function is in heat/core/linalg/basics
        return linalg.transpose(self, axes=None)

    @property
    def array_with_halos(self) -> torch.Tensor:
        """
        Fetch halos of size ``halo_size`` from neighboring ranks and save them in ``self.halo_next``/``self.halo_prev``
        in case they are not already stored. If ``halo_size`` differs from the size of already stored halos,
        the are overwritten.
        """
        return self.__cat_halo()

    def __prephalo(self, start, end) -> torch.Tensor:
        """
        Extracts the halo indexed by start, end from ``self.array`` in the direction of ``self.split``

        Parameters
        ----------
        start : int
            Start index of the halo extracted from ``self.array``
        end : int
            End index of the halo extracted from ``self.array``
        """
        ix = [slice(None, None, None)] * len(self.shape)
        try:
            ix[self.split] = slice(start, end)
        except IndexError:
            print("Indices out of bound")

        return self.__array[tuple(ix)].clone()

    def get_halo(self, halo_size: int, prev: bool = True, next: bool = True):
        """
        Fetch halos of size ``halo_size`` from neighboring ranks and save them in ``self.halo_next/self.halo_prev``.

        Parameters
        ----------
        halo_size : int
            Size of the halo.
        prev : bool, optional
            If True, fetch the halo from the previous rank. Default: True.
        next : bool, optional
            If True, fetch the halo from the next rank. Default: True.
        """
        if not isinstance(halo_size, int):
            raise TypeError(
                f"halo_size needs to be of Python type integer, {type(halo_size)} given"
            )
        if halo_size < 0:
            raise ValueError(
                f"halo_size needs to be a non-negative Python integer, {halo_size} given"
            )

        if self.is_distributed() and halo_size > 0:
            # gather lshapes
            lshape_map = self.lshape_map
            rank = self.comm.rank

            populated_ranks = torch.nonzero(lshape_map[:, self.split]).squeeze().tolist()
            if rank in populated_ranks:
                first_rank = populated_ranks[0]
                last_rank = populated_ranks[-1]
                if rank != last_rank:
                    next_rank = populated_ranks[populated_ranks.index(rank) + 1]
                if rank != first_rank:
                    prev_rank = populated_ranks[populated_ranks.index(rank) - 1]
            else:
                # if process has no data we ignore it
                return

            if (halo_size > self.lshape_map[:, self.split][populated_ranks]).any():
                # halo_size is larger than the local size on at least one process
                raise ValueError(
                    f"halo_size {halo_size} needs to be smaller than chunk-size {self.lshape[self.split]} )"
                )

            a_prev = self.__prephalo(0, halo_size)
            a_next = self.__prephalo(-halo_size, None)
            res_prev = None
            res_next = None
            req_list = []

            # exchange data with next populated process
            if prev:
                if rank != last_rank:
                    req_list.append(self.comm.Isend(a_next, next_rank))
                if rank != first_rank:
                    res_prev = torch.empty(
                        a_prev.size(), dtype=a_prev.dtype, device=self.device.torch_device
                    )
                    req_list.append(self.comm.Irecv(res_prev, source=prev_rank))

            if next:
                if rank != first_rank:
                    req_list.append(self.comm.Isend(a_prev, prev_rank))
                if rank != last_rank:
                    res_next = torch.empty(
                        a_next.size(), dtype=a_next.dtype, device=self.device.torch_device
                    )
                    req_list.append(self.comm.Irecv(res_next, source=next_rank))

            for req in req_list:
                req.Wait()

            self.__halo_next = res_next
            self.__halo_prev = res_prev
            self.__ishalo = True

    def __cat_halo(self) -> torch.Tensor:
        """
        Return local array concatenated to halos if they are available.
        """
        if not self.is_distributed():
            return self.__array
        return torch.cat(
            [_ for _ in (self.__halo_prev, self.__array, self.__halo_next) if _ is not None],
            dim=self.split,
        )

    def __array__(self) -> np.ndarray:
        """
        Returns a view of the process-local slice of the :class:`DNDarray` as a numpy ndarray, if the ``DNDarray`` resides on CPU. Otherwise, it returns a copy, on CPU, of the process-local slice of ``DNDarray`` as numpy ndarray.
        """
        return self.larray.cpu().__array__()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Override NumPy's universal functions.
        """
        import heat

        # TODO support ufunc method variants
        if method == "__call__":
            try:
                func = getattr(heat, ufunc.__name__)
            except AttributeError:
                return NotImplemented
            return func(*inputs, **kwargs)
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        """
        Augments NumPy's functions.
        """
        import heat

        try:
            ht_func = getattr(heat, func.__name__)
        except AttributeError:
            return NotImplemented
        return ht_func(*args, **kwargs)

    def astype(self, dtype, copy=True) -> DNDarray:
        """
        Returns a casted version of this array.
        Casted array is a new array of the same shape but with given type of this array. If copy is ``True``, the
        same array is returned instead.

        Parameters
        ----------
        dtype : datatype
            Heat type to which the array is cast
        copy : bool, optional
            By default the operation returns a copy of this array. If copy is set to ``False`` the cast is performed
            in-place and this array is returned

        """
        dtype = canonical_heat_type(dtype)
        if self.__array.is_mps:
            if dtype == types.float64:
                # print warning
                warnings.warn(
                    "MPS does not support float64. Casting to float32 instead.",
                    ResourceWarning,
                )
                dtype = types.float32
            elif dtype == types.complex128:
                # print warning
                warnings.warn(
                    "MPS does not support complex128. Casting to complex64 instead.",
                    ResourceWarning,
                )
                dtype = types.complex64
        casted_array = self.__array.type(dtype.torch_type())
        if copy:
            return DNDarray(
                casted_array,
                gshape=self.shape,
                dtype=dtype,
                split=self.split,
                device=self.device,
                comm=self.comm,
                balanced=self.balanced,
            )

        self.__array = casted_array
        self.__dtype = dtype

        return self

    def balance_(self) -> None:
        """
        Function for balancing a :class:`DNDarray` between all nodes. To determine if this is needed use the :func:`is_balanced()` function.
        If the ``DNDarray`` is already balanced this function will do nothing. This function modifies the ``DNDarray``
        itself and will not return anything.

        Examples
        --------
        >>> a = ht.zeros((10, 2), split=0)
        >>> a[:, 0] = ht.arange(10)
        >>> b = a[3:]
        [0/2] tensor([[3., 0.],
        [1/2] tensor([[4., 0.],
                      [5., 0.],
                      [6., 0.]])
        [2/2] tensor([[7., 0.],
                      [8., 0.],
                      [9., 0.]])
        >>> b.balance_()
        >>> print(b.gshape, b.lshape)
        [0/2] (7, 2) (1, 2)
        [1/2] (7, 2) (3, 2)
        [2/2] (7, 2) (3, 2)
        >>> b
        [0/2] tensor([[3., 0.],
                     [4., 0.],
                     [5., 0.]])
        [1/2] tensor([[6., 0.],
                      [7., 0.]])
        [2/2] tensor([[8., 0.],
                      [9., 0.]])
        >>> print(b.gshape, b.lshape)
        [0/2] (7, 2) (3, 2)
        [1/2] (7, 2) (2, 2)
        [2/2] (7, 2) (2, 2)
        """
        if not self.is_distributed():
            self.__balanced = True
        if self.is_balanced(force_check=True):
            return
        self.redistribute_()

    def __bool__(self) -> bool:
        """
        Boolean scalar casting.
        """
        return self.__cast(bool)

    def __cast(self, cast_function) -> float | int:
        """
        Implements a generic cast function for ``DNDarray`` objects.

        Parameters
        ----------
        cast_function : function
            The actual cast function, e.g. ``float`` or ``int``

        Raises
        ------
        TypeError
            If the ``DNDarray`` object cannot be converted into a scalar.

        """
        if np.prod(self.shape) == 1:
            if self.split is None:
                return cast_function(self.__array)

            is_empty = np.prod(self.__array.shape) == 0
            root = self.comm.allreduce(0 if is_empty else self.comm.rank, op=MPI.SUM)

            return self.comm.bcast(None if is_empty else cast_function(self.__array), root=root)

        raise TypeError("only size-1 arrays can be converted to Python scalars")

    def collect_(self, target_rank: int | None = 0) -> None:
        """
        A method collecting a distributed DNDarray to one MPI rank, chosen by the `target_rank` variable.
        It is a specific case of the ``redistribute_`` method.

        Parameters
        ----------
        target_rank : int, optional
            The rank to which the DNDarray will be collected. Default: 0.

        Raises
        ------
        TypeError
            If the target rank is not an integer.
        ValueError
            If the target rank is out of bounds.

        Examples
        --------
        >>> st = ht.ones((50, 81, 67), split=2)
        >>> print(st.lshape)
        [0/2] (50, 81, 23)
        [1/2] (50, 81, 22)
        [2/2] (50, 81, 22)
        >>> st.collect_()
        >>> print(st.lshape)
        [0/2] (50, 81, 67)
        [1/2] (50, 81, 0)
        [2/2] (50, 81, 0)
        >>> st.collect_(1)
        >>> print(st.lshape)
        [0/2] (50, 81, 0)
        [1/2] (50, 81, 67)
        [2/2] (50, 81, 0)
        """
        if not isinstance(target_rank, int):
            raise TypeError(f"target rank must be of type int , but was {type(target_rank)}")
        if target_rank >= self.comm.size:
            raise ValueError("target rank is out of bounds")
        if not self.is_distributed():
            return

        target_map = self.lshape_map.clone()
        target_map[:, self.split] = 0
        target_map[target_rank, self.split] = self.gshape[self.split]
        self.redistribute_(target_map=target_map)

    def __complex__(self) -> DNDarray:
        """
        Complex scalar casting.
        """
        return self.__cast(complex)

    def counts_displs(self) -> tuple[tuple[int], tuple[int]]:
        """
        Returns actual counts (number of items per process) and displacements (offsets) of the DNDarray.
        Does not assume load balance.
        """
        if self.split is not None:
            counts = self.lshape_map[:, self.split]
            displs = [0] + torch.cumsum(counts, dim=0)[:-1].tolist()
            return tuple(counts.tolist()), tuple(displs)

        raise ValueError("Non-distributed DNDarray. Cannot calculate counts and displacements.")

    def cpu(self) -> DNDarray:
        """
        Returns a copy of this object in main memory. If this object is already in main memory, then no copy is
        performed and the original object is returned.
        """
        self.__array = self.__array.cpu()
        self.__device = devices.cpu
        return self

    def create_lshape_map(self, force_check: bool = False) -> torch.Tensor:
        """
        Generate a 'map' of the lshapes of the data on all processes.
        Units are ``(process rank, lshape)``

        Parameters
        ----------
        force_check : bool, optional
            if False (default) and the lshape map has already been created, use the previous
            result. Otherwise, create the lshape_map
        """
        if not force_check and self.__lshape_map is not None:
            return self.__lshape_map.clone()

        lshape_map = torch.zeros(
            (self.comm.size, self.ndim), dtype=torch.int64, device=self.device.torch_device
        )
        if not self.is_distributed():
            lshape_map[:] = torch.tensor(self.gshape, device=self.device.torch_device)
            self.__lshape_map = lshape_map
            return lshape_map.clone()
        if self.is_balanced(force_check=True):
            for i in range(self.comm.size):
                _, lshape, _ = self.comm.chunk(self.gshape, self.split, rank=i)
                lshape_map[i, :] = torch.tensor(lshape, device=self.device.torch_device)
        else:
            lshape_map[self.comm.rank, :] = torch.tensor(
                self.lshape, device=self.device.torch_device
            )
            self.comm.Allreduce(MPI.IN_PLACE, lshape_map, MPI.SUM)

        self.__lshape_map = lshape_map
        return lshape_map.clone()

    def create_partition_interface(self):
        """
        Create a partition interface in line with the DPPY proposal. This is subject to change.
        The intention of this to facilitate the usage of a general format for the referencing of
        distributed datasets.

        An example of the output and shape is shown below.

        __partitioned__ = {
            'shape': (27, 3, 2),
            'partition_tiling': (4, 1, 1),
            'partitions': {
                (0, 0, 0): {
                    'start': (0, 0, 0),
                    'shape': (7, 3, 2),
                    'data': tensor([...], dtype=torch.int32),
                    'location': [0],
                    'dtype': torch.int32,
                    'device': 'cpu'
                },
                (1, 0, 0): {
                    'start': (7, 0, 0),
                    'shape': (7, 3, 2),
                    'data': None,
                    'location': [1],
                    'dtype': torch.int32,
                    'device': 'cpu'
                },
                (2, 0, 0): {
                    'start': (14,  0,  0),
                    'shape': (7, 3, 2),
                    'data': None,
                    'location': [2],
                    'dtype': torch.int32,
                    'device': 'cpu'
                },
                (3, 0, 0): {
                    'start': (21,  0,  0),
                    'shape': (6, 3, 2),
                    'data': None,
                    'location': [3],
                    'dtype': torch.int32,
                    'device': 'cpu'
                }
            },
            'locals': [(rank, 0, 0)],
            'get': lambda x: x,
        }

        Returns
        -------
        dictionary containing the partition interface as shown above.
        """
        lshape_map = self.create_lshape_map()
        start_idx_map = torch.zeros_like(lshape_map)

        part_tiling = [1] * self.ndim
        lcls = [0] * self.ndim

        z = torch.tensor([0], device=self.device.torch_device, dtype=torch.int64)

        if self.split is not None:
            starts = torch.cat((z, torch.cumsum(lshape_map[:, self.split], dim=0)[:-1]), dim=0)
            lcls[self.split] = self.comm.rank
            part_tiling[self.split] = self.comm.size
            start_idx_map[:, self.split] = starts
        else:
            start_idx_map[:] = 0

        partitions = {}
        base_key = [0] * self.ndim
        for r in range(self.comm.size):
            if self.split is not None:
                base_key[self.split] = r
                dat = None if r != self.comm.rank else self.larray
            else:
                dat = self.larray
            partitions[tuple(base_key)] = {
                "start": tuple(start_idx_map[r].tolist()),
                "shape": tuple(lshape_map[r].tolist()),
                "data": dat,
                "location": [r],
                "dtype": self.dtype.torch_type(),
                "device": self.device.torch_device,
            }

        partition_dict = {
            "shape": self.gshape,
            "partition_tiling": tuple(part_tiling),
            "partitions": partitions,
            "locals": [tuple(lcls)],
            "get": lambda x: x,
        }

        self.__partitions_dict__ = partition_dict

        return partition_dict

    def __float__(self) -> DNDarray:
        """
        Float scalar casting.

        See Also
        --------
        :func:`~heat.core.manipulations.flatten`
        """
        return self.__cast(float)

    def fill_diagonal(self, value: float) -> DNDarray:
        """
        Fill the main diagonal of a 2D :class:`DNDarray`.
        This function modifies the input tensor in-place, and returns the input array.

        Parameters
        ----------
        value : float
            The value to be placed in the ``DNDarrays`` main diagonal
        """
        # Todo: make this 3D/nD
        if len(self.shape) != 2:
            raise ValueError("Only 2D tensors supported at the moment")

        if self.split is not None and self.comm.is_distributed:
            counts, displ, _ = self.comm.counts_displs_shape(self.shape, self.split)
            k = min(self.shape[0], self.shape[1])
            for p in range(self.comm.size):
                if displ[p] > k:
                    break
                proc = p
            if self.comm.rank <= proc:
                indices = (
                    displ[self.comm.rank],
                    displ[self.comm.rank + 1] if (self.comm.rank + 1) != self.comm.size else k,
                )
                if self.split == 0:
                    self.larray[:, indices[0] : indices[1]] = self.larray[
                        :, indices[0] : indices[1]
                    ].fill_diagonal_(value)
                elif self.split == 1:
                    self.larray[indices[0] : indices[1], :] = self.larray[
                        indices[0] : indices[1], :
                    ].fill_diagonal_(value)

        else:
            self.larray = self.larray.fill_diagonal_(value)

        return self

    def __broadcast_value(
        self,
        key: int | tuple[int, ...] | slice,
        value: "DNDarray",
        **kwargs,
    ):
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
        output_shape = kwargs.get("output_shape", None)
        if output_shape is not None:
            indexed_dims = len(output_shape)
        else:
            if isinstance(key, (int, tuple)):
                # direct indexing, output_shape has not been calculated
                # use proxy to avoid MPI communication and limit memory usage
                indexed_proxy = self.__torch_proxy__()[key]
                indexed_dims = indexed_proxy.ndim
                output_shape = tuple(indexed_proxy.shape)
            else:
                raise RuntimeError(
                    "Not enough information to broadcast value to indexed array, please provide `output_shape`"
                )
        value_shape = value.shape
        # check if value needs to be broadcasted
        if value_shape != output_shape:
            # assess whether the shapes are compatible, starting from the trailing dimension
            for i in range(1, min(len(value_shape), len(output_shape)) + 1):
                if i == 1:
                    if value_shape[-i] != output_shape[-i] and not value_shape[-i] == 1:
                        # shapes are not compatible, raise error
                        raise ValueError(
                            f"could not broadcast input array from shape {value_shape} into shape {output_shape}"
                        )
                else:
                    if value_shape[-i] != output_shape[-i] and (not value_shape[-i] == 1):
                        # shapes are not compatible, raise error
                        raise ValueError(
                            f"could not broadcast input from shape {value_shape} into shape {output_shape}"
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
                    try:
                        value_shape = tuple(torch.broadcast_shapes(value.shape, output_shape))
                    except RuntimeError:
                        raise ValueError(
                            f"could not broadcast input array from shape {value_shape} into shape {output_shape}"
                        )
        return value, is_scalar

    def __set(
        self,
        key: int | tuple[int, ...] | list[int],
        value: float | "DNDarray" | torch.Tensor,
    ):
        """
        Setter for not advanced indexing, i.e. when arr[key] is an in-place view of arr.
        """
        # only assign values if key does not contain empty slices
        process_is_inactive = self.larray[key].numel() == 0
        if not process_is_inactive:
            rhs = value.larray.type(self.dtype.torch_type())
            key_to_use = key

            # CUDA: make advanced indexing assignment deterministic for duplicate indices
            if self.larray.is_cuda:
                key_to_use, rhs = _resolve_duplicate_indices(key_to_use, rhs, self.larray.shape)

            self.larray[key_to_use] = rhs
        return

    @staticmethod
    def __advanced_setitem_unordered_local(
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
        if base_index is None:
            lhs_index = [slice(None)] * x_local.ndim
        else:
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

    def __getitem_scalar(self, p: ProcessedKey) -> DNDarray:
        if p.root is not None:
            # Single-element indexing along split axis
            if self.comm.rank == p.root:
                indexed_arr = self.larray[p.key]
            else:
                indexed_arr = torch.zeros(
                    p.output_shape, dtype=self.larray.dtype, device=self.larray.device
                )
            self.comm.Bcast(indexed_arr, root=p.root)
        else:
            indexed_arr = self.larray[p.key]

        if self.ndim > 0:
            self = self.transpose(p.backwards_transpose_axes)

        return DNDarray(
            indexed_arr,
            gshape=p.output_shape,
            dtype=self.dtype,
            split=p.output_split,
            device=self.device,
            comm=self.comm,
            balanced=p.out_is_balanced,
        )

    def __getitem_slice(self, p: ProcessedKey) -> "DNDarray":
        indexed_arr = self.larray[p.key]
        if self.ndim > 0:
            self = self.transpose(p.backwards_transpose_axes)

        return DNDarray(
            indexed_arr,
            gshape=p.output_shape,
            dtype=self.dtype,
            split=p.output_split,
            device=self.device,
            comm=self.comm,
            balanced=p.out_is_balanced,
        )

    def __getitem_descending_slice_distributed(self, p: ProcessedKey) -> DNDarray:
        from .manipulations import flip

        # local indexing
        indexed_arr = self.larray[p.key]
        if self.ndim > 0:
            self = self.transpose(p.backwards_transpose_axes)

        # wrap the reversed local chunks into an unbalanced DNDarray
        intermediate = DNDarray(
            indexed_arr,
            gshape=p.output_shape,
            dtype=self.dtype,
            split=p.output_split,
            device=self.device,
            comm=self.comm,
            balanced=False,
        )
        # intermediate.balance_()
        # global flip to reflect the descending slice
        return flip(intermediate, axis=p.output_split)

    def __getitem_mask(self, p: ProcessedKey, original_key) -> "DNDarray":
        # local masking, then wrap into DNDarray
        local_mask = p.key
        local_result = self.larray[local_mask]

        return factories.array(
            local_result, is_split=p.output_split, device=self.device, comm=self.comm, copy=False
        )

    def __getitem_advanced_local(self, p: ProcessedKey, original_key) -> "DNDarray":
        indexed_arr = self.larray[p.key]
        if self.ndim > 0:
            self = self.transpose(p.backwards_transpose_axes)

        return DNDarray(
            indexed_arr,
            gshape=p.output_shape,
            dtype=self.dtype,
            split=p.output_split,
            device=self.device,
            comm=self.comm,
            balanced=p.out_is_balanced,
        )

    def __getitem_advanced_distributed(self, p: ProcessedKey) -> "DNDarray":
        self, indexed_arr = self.__getitem_unordered(
            key=p.key,
            output_shape=p.output_shape,
            output_split=p.output_split,
            out_is_balanced=p.out_is_balanced,
            key_is_mask_like=p.key_is_mask_like,
            backwards_transpose_axes=p.backwards_transpose_axes,
        )
        return indexed_arr

    def __getitem_unordered(
        self,
        key: tuple,
        output_shape: tuple,
        output_split: int,
        out_is_balanced: bool,
        key_is_mask_like: bool,
        backwards_transpose_axes: tuple,
    ) -> DNDarray:
        """
        Handles the MPI communication (Alltoallv) when the key along the
        split axis is unordered and indices are global.
        """
        _, displs = self.counts_displs()
        rank = self.comm.rank

        key_is_single_tensor = isinstance(key, torch.Tensor)
        split_key = key if key_is_single_tensor else key[self.split]
        split_key_flat = split_key.reshape(-1)

        # Calculate communication split axis for transposing later
        if key_is_single_tensor or key_is_mask_like:
            communication_split = 0
        else:
            communication_split = (
                output_split - (split_key.ndim - 1) if split_key.ndim > 1 else output_split
            )

        # Step 1: route and send index requests

        sort_idx, send_counts_t, send_displs_t, recv_counts_t, recv_displs_t = (
            self.__prepare_unordered_comm(split_key_flat, displs)
        )

        send_counts = send_counts_t.tolist()
        send_displs = send_displs_t.tolist()
        recv_counts = recv_counts_t.tolist()
        recv_displs = recv_displs_t.tolist()

        # Expand counts for multidimensional mask coordinates
        if key_is_mask_like:
            mask_dims = len(key)
            idx_send_counts = [c * mask_dims for c in send_counts]
            idx_send_displs = [d * mask_dims for d in send_displs]
            idx_recv_counts = [c * mask_dims for c in recv_counts]
            idx_recv_displs = [d * mask_dims for d in recv_displs]

            send_indices = torch.stack([k.flatten()[sort_idx] for k in key], dim=1).reshape(-1)
            recv_indices_flat = torch.zeros(
                sum(idx_recv_counts), dtype=split_key.dtype, device=self.larray.device
            )
        else:
            idx_send_counts, idx_send_displs = send_counts, send_displs
            idx_recv_counts, idx_recv_displs = recv_counts, recv_displs

            send_indices = split_key_flat[sort_idx]
            recv_indices_flat = torch.zeros(
                sum(idx_recv_counts), dtype=split_key.dtype, device=self.larray.device
            )

        self.comm.Alltoallv(
            (send_indices, idx_send_counts, idx_send_displs),
            (recv_indices_flat, idx_recv_counts, idx_recv_displs),
        )

        if key_is_mask_like:
            recv_indices = recv_indices_flat.reshape(sum(recv_counts), len(key))
        else:
            recv_indices = recv_indices_flat

        # Step 2: local data lookup based on received indices

        if key_is_mask_like:
            recv_indices[:, self.split] -= displs[rank]
            lookup_key = tuple(recv_indices[:, i] for i in range(len(key)))
            local_vals = self.larray[lookup_key]
        else:
            recv_indices -= displs[rank]
            if key_is_single_tensor:
                local_vals = self.larray[recv_indices]
            else:
                lookup_key = list(key)
                lookup_key[self.split] = recv_indices
                local_vals = self.larray[tuple(lookup_key)]

        # Step 3: return data to requesting processes

        # Ensure the indexed elements are aligned along axis 0
        transpose_axes = list(range(local_vals.ndim))
        transpose_axes[0], transpose_axes[communication_split] = (
            transpose_axes[communication_split],
            transpose_axes[0],
        )
        local_vals = local_vals.permute(*transpose_axes)

        feature_shape = list(local_vals.shape[1:])
        feature_size = 1
        for dim in feature_shape:
            feature_size *= dim

        return_send_counts = [c * feature_size for c in recv_counts]
        return_send_displs = [d * feature_size for d in recv_displs]
        return_recv_counts = [c * feature_size for c in send_counts]
        return_recv_displs = [d * feature_size for d in send_displs]

        send_vals = local_vals.reshape(-1)
        recv_vals_flat = torch.empty(
            sum(return_recv_counts), dtype=self.larray.dtype, device=self.larray.device
        )

        self.comm.Alltoallv(
            (send_vals, return_send_counts, return_send_displs),
            (recv_vals_flat, return_recv_counts, return_recv_displs),
        )

        # Step 4: reshape received values and reorder to match original key order

        recv_vals = recv_vals_flat.reshape(-1, *feature_shape)

        # Reverse the sorting applied in Step 1
        inv_sort_idx = torch.empty_like(sort_idx)
        inv_sort_idx[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)
        unsorted_vals = recv_vals[inv_sort_idx]

        # Restore original dimension order
        final_vals = unsorted_vals.permute(*transpose_axes)

        # Reshape to match the global output shape expectation
        if communication_split != output_split:
            original_local_shape = (
                output_shape[:communication_split]
                + split_key.shape
                + output_shape[output_split + 1 :]
            )
            final_vals = final_vals.reshape(original_local_shape)

        indexed_arr = DNDarray(
            final_vals,
            gshape=output_shape,
            dtype=self.dtype,
            split=output_split,
            device=self.device,
            comm=self.comm,
            balanced=out_is_balanced,
        )

        if self.ndim > 0:
            return self.transpose(backwards_transpose_axes), indexed_arr
        return self, indexed_arr

    def __prepare_unordered_comm(self, split_key_flat: torch.Tensor, displs: tuple) -> tuple:
        """
        Helper function for distributed unordered indexing.
        Determines destination ranks, sorts the key, and computes Alltoallv parameters.
        """
        displs_t = torch.tensor(displs, device=self.device.torch_device)

        # map global indices to destination ranks
        dest_ranks = torch.searchsorted(displs_t[1:], split_key_flat, right=True).to(torch.int64)

        # sort by destination rank to pack memory contiguously
        sort_idx = torch.argsort(dest_ranks)
        dest_ranks_sorted = dest_ranks[sort_idx]

        # calculate send_counts and send_displs
        send_counts = torch.bincount(dest_ranks_sorted, minlength=self.comm.size).to(torch.int64)
        send_displs = torch.zeros_like(send_counts)
        send_displs[1:] = torch.cumsum(send_counts, dim=0)[:-1]

        # compose communication matrix, i.e. share `send_counts` information with all processes
        comm_matrix = torch.zeros(
            (self.comm.size, self.comm.size),
            dtype=torch.int64,
            device=self.device.torch_device,
        )
        self.comm.Allgather(send_counts, comm_matrix)

        # comm_matrix columns contain recv_counts for each process
        recv_counts = comm_matrix[:, self.comm.rank].squeeze(0)
        recv_displs = torch.zeros_like(recv_counts)
        recv_displs[1:] = recv_counts.cumsum(0)[:-1]

        return (
            sort_idx,
            send_counts,
            send_displs,
            recv_counts,
            recv_displs,
        )

    def __getitem__(self, key: Indexer) -> DNDarray:
        """
        Global getter function for DNDarrays.

        Returns a new DNDarray corresponding to the selection of values from the original DNDarray as specified by `key`.
        The `key` can be a variety of indexers, including integers, slices, lists, boolean masks, DNDarrays, ndarrays,
        torch tensors, and a combination thereof. The function will determine the appropriate method to retrieve the
        requested data based on the type and structure of `key`, setting up necessary MPI communication if the indexing
        pattern requires data from multiple processes.

        Notes
        -----
        The returned DNDarray will have its shape, split, and balanced status determined according to the indexing operation performed.
        For more details see INDEXING.md. #TODO: link to documentation

        Parameters
        ----------
        key : array-like indexer
            Indices to get from the tensor.

        Examples
        --------
        >>> a = ht.arange(10, split=0)
        (1/2) >>> tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        (2/2) >>> tensor([5, 6, 7, 8, 9], dtype=torch.int32)
        >>> a[1:6]
        (1/2) >>> tensor([1, 2, 3, 4], dtype=torch.int32)
        (2/2) >>> tensor([5], dtype=torch.int32)
        >>> a = ht.zeros((4, 5), split=0)
        (1/2) >>> tensor([[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]])
        (2/2) >>> tensor([[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]])
        >>> a[1:4, 1]
        (1/2) >>> tensor([0.])
        (2/2) >>> tensor([0., 0.])
        """
        if key is None:
            return self.expand_dims(0)
        if (
            key is ...
            or (isinstance(key, slice) and key == slice(None))
            or (isinstance(key, tuple) and key == ())
        ):
            return self

        # key processing returns a ProcessedKey namedtuple
        self, processed_key = _resolve_indexing_state(
            self, key, return_local_indices=True, op="get"
        )
        print(f"DEBUGGING: Processed key: {processed_key}")

        # dispatch to appropriate getitem method
        op = processed_key.op_type
        # print("DEBUGGING: Operation type:", op)

        if op == "scalar":
            return self.__getitem_scalar(processed_key)
        elif op == "distr_mask":
            return self.__getitem_mask(processed_key, key)
        elif op == "distributed":
            return self.__getitem_advanced_distributed(processed_key)
        elif op == "slice":
            return self.__getitem_slice(processed_key)
        elif op == "descending_slice":
            return self.__getitem_descending_slice_distributed(processed_key)
        elif op in ("local_mask", "advanced"):
            return self.__getitem_advanced_local(processed_key, key)

    if torch.cuda.device_count() > 0:

        def gpu(self) -> DNDarray:
            """
            Returns a copy of this object in GPU memory. If this object is already in GPU memory, then no copy is
            performed and the original object is returned.

            """
            self.__array = self.__array.cuda(devices.gpu.torch_device)
            self.__device = devices.gpu
            return self

    def __int__(self) -> DNDarray:
        """
        Integer scalar casting.
        """
        return self.__cast(int)

    def is_balanced(self, force_check: bool = False) -> bool:
        """
        Determine if ``self`` is balanced evenly (or as evenly as possible) across all nodes
        distributed evenly (or as evenly as possible) across all processes.
        This is equivalent to returning ``self.balanced``. If no information
        is available (``self.balanced = None``), the balanced status will be
        assessed via collective communication.

        Parameters
        ----------
        force_check : bool, optional
            If True, the balanced status of the ``DNDarray`` will be assessed via
            collective communication in any case.
        """
        if not force_check and self.balanced is not None:
            return self.balanced

        _, _, chk = self.comm.chunk(self.shape, self.split)
        test_lshape = tuple([x.stop - x.start for x in chk])
        balanced = 1 if test_lshape == self.lshape else 0

        out = self.comm.allreduce(balanced, MPI.SUM)
        balanced = True if out == self.comm.size else False
        return balanced

    def is_distributed(self) -> bool:
        """
        Determines whether the data of this ``DNDarray`` is distributed across multiple processes.
        """
        return self.split is not None and self.comm.is_distributed()

    def item(self):
        """
        Returns the only element of a 1-element :class:`DNDarray`.
        Mirror of the pytorch command by the same name. If size of ``DNDarray`` is >1 element, then a ``ValueError`` is
        raised (by pytorch)

        Examples
        --------
        >>> import heat as ht
        >>> x = ht.zeros((1))
        >>> x.item()
        0.0
        """
        if self.size > 1:
            raise ValueError("only one-element DNDarrays can be converted to Python scalars")
        # make sure the element is on every process
        self.resplit_(None)
        return self.__array.item()

    def __len__(self) -> int:
        """
        The length of the ``DNDarray``, i.e. the number of items in the first dimension.
        """
        try:
            len = self.shape[0]
            return len
        except IndexError:
            raise TypeError("len() of unsized DNDarray")

    def numpy(self) -> np.typing.NDArray[Any]:
        """
        Returns a copy of the :class:`DNDarray` as numpy ndarray. If the ``DNDarray`` resides on the GPU, the underlying data will be copied to the CPU first.

        If the ``DNDarray`` is distributed, an MPI Allgather operation will be performed before converting to np.ndarray, i.e. each MPI process will end up holding a copy of the entire array in memory.  Make sure process memory is sufficient!

        Examples
        --------
        >>> import heat as ht
        T1 = ht.random.randn((10,8))
        T1.numpy()
        """
        dist = self.copy().resplit_(axis=None)
        return dist.larray.cpu().numpy()

    def _repr_pretty_(self, p, cycle):
        """
        Pretty print for IPython.
        """
        if cycle:
            p.text(printing.__str__(self))
        else:
            p.text(printing.__str__(self))

    def __repr__(self) -> str:
        """
        Returns a printable representation of the passed DNDarray, targeting developers.
        """
        return printing.__repr__(self)

    def ravel(self) -> "DNDarray":
        """
        Flattens the ``DNDarray``.

        See Also
        --------
        :func:`~heat.core.manipulations.ravel`

        Examples
        --------
        >>> a = ht.ones((2, 3), split=0)
        >>> b = a.ravel()
        >>> a[0, 0] = 4
        >>> b
        DNDarray([4., 1., 1., 1., 1., 1.], dtype=ht.float32, device=cpu:0, split=0)
        """
        return manipulations.ravel(self)

    def redistribute_(
        self, lshape_map: torch.Tensor | None = None, target_map: torch.Tensor | None = None
    ) -> None:
        """
        Redistributes the data of the :class:`DNDarray` *along the split axis* to match the given target map.
        This function does not modify the non-split dimensions of the ``DNDarray``.
        This is an abstraction and extension of the balance function.

        Parameters
        ----------
        lshape_map : torch.Tensor, optional
            The current lshape of processes.
            Units are ``[rank, lshape]``.
        target_map : torch.Tensor, optional
            The desired distribution across the processes.
            Units are ``[rank, target lshape]``.
            Note: the only important parts of the target map are the values along the split axis,
            values which are not along this axis are there to mimic the shape of the ``lshape_map``.

        Examples
        --------
        >>> st = ht.ones((50, 81, 67), split=2)
        >>> target_map = torch.zeros((st.comm.size, 3), dtype=torch.int64)
        >>> target_map[0, 2] = 67
        >>> print(target_map)
        [0/2] tensor([[ 0,  0, 67],
        [0/2]         [ 0,  0,  0],
        [0/2]         [ 0,  0,  0]], dtype=torch.int32)
        [1/2] tensor([[ 0,  0, 67],
        [1/2]         [ 0,  0,  0],
        [1/2]         [ 0,  0,  0]], dtype=torch.int32)
        [2/2] tensor([[ 0,  0, 67],
        [2/2]         [ 0,  0,  0],
        [2/2]         [ 0,  0,  0]], dtype=torch.int32)
        >>> print(st.lshape)
        [0/2] (50, 81, 23)
        [1/2] (50, 81, 22)
        [2/2] (50, 81, 22)
        >>> st.redistribute_(target_map=target_map)
        >>> print(st.lshape)
        [0/2] (50, 81, 67)
        [1/2] (50, 81, 0)
        [2/2] (50, 81, 0)
        """
        if not self.is_distributed():
            return
        snd_dtype = self.dtype.torch_type()
        # units -> {pr, 1st index, 2nd index}
        if lshape_map is None:
            # NOTE: giving an lshape map which is incorrect will result in an incorrect distribution
            lshape_map = self.create_lshape_map(force_check=True)
        else:
            if not isinstance(lshape_map, torch.Tensor):
                raise TypeError(f"lshape_map must be a torch.Tensor, currently {type(lshape_map)}")
            if lshape_map.shape != (self.comm.size, len(self.gshape)):
                raise ValueError(
                    f"lshape_map must have the shape ({self.comm.size}, {len(self.gshape)}), currently {lshape_map.shape}"
                )
        if target_map is None:  # if no target map is given then it will balance the tensor
            _, _, chk = self.comm.chunk(self.shape, self.split)
            target_map = lshape_map.clone()
            target_map[..., self.split] = 0
            for pr in range(self.comm.size):
                target_map[pr, self.split] = self.comm.chunk(self.shape, self.split, rank=pr)[1][
                    self.split
                ]
            self.__balanced = True
        else:
            sanitation.sanitize_in_tensor(target_map)
            if target_map[..., self.split].sum() != self.shape[self.split]:
                raise ValueError(
                    f"Sum along the split axis of the target map must be equal to the shape in that dimension, currently {target_map[..., self.split]}"
                )
            if target_map.shape != (self.comm.size, len(self.gshape)):
                raise ValueError(
                    f"target_map must have the shape {(self.comm.size, len(self.gshape))}, currently {target_map.shape}"
                )
            # no info on balanced status
            self.__balanced = False
        lshape_cumsum = torch.cumsum(lshape_map[..., self.split], dim=0)
        chunk_cumsum = torch.cat(
            (
                torch.tensor([0], device=self.device.torch_device),
                torch.cumsum(target_map[..., self.split], dim=0),
            ),
            dim=0,
        )
        # need the data start as well for process 0
        for rcv_pr in range(self.comm.size - 1):
            st = chunk_cumsum[rcv_pr].item()
            sp = chunk_cumsum[rcv_pr + 1].item()
            # start pr should be the next process with data
            if lshape_map[rcv_pr, self.split] >= target_map[rcv_pr, self.split]:
                # if there is more data on the process than the start process than start == stop
                st_pr = rcv_pr
                sp_pr = rcv_pr
            else:
                # if there is less data on the process than need to get the data from the next data
                # with data
                # need processes > rcv_pr with lshape > 0
                st_pr = (
                    torch.nonzero(input=lshape_map[rcv_pr:, self.split] > 0, as_tuple=False)[
                        0
                    ].item()
                    + rcv_pr
                )
                hld = (
                    torch.nonzero(input=sp <= lshape_cumsum[rcv_pr:], as_tuple=False).flatten()
                    + rcv_pr
                )
                sp_pr = hld[0].item() if hld.numel() > 0 else self.comm.size

            # st_pr and sp_pr are the processes on which the data sits at the beginning
            # need to loop from st_pr to sp_pr + 1 and send the pr
            for snd_pr in range(st_pr, sp_pr + 1):
                if snd_pr == self.comm.size:
                    break
                data_required = abs(sp - st - lshape_map[rcv_pr, self.split].item())
                send_amt = (
                    data_required
                    if data_required <= lshape_map[snd_pr, self.split]
                    else lshape_map[snd_pr, self.split]
                )
                if (sp - st) <= lshape_map[rcv_pr, self.split].item() or snd_pr == rcv_pr:
                    send_amt = 0
                # send amount is the data still needed by recv if that is available on the snd
                if send_amt != 0:
                    self.__redistribute_shuffle(
                        snd_pr=snd_pr, send_amt=send_amt, rcv_pr=rcv_pr, snd_dtype=snd_dtype
                    )
                lshape_cumsum[snd_pr] -= send_amt
                lshape_cumsum[rcv_pr] += send_amt
                lshape_map[rcv_pr, self.split] += send_amt
                lshape_map[snd_pr, self.split] -= send_amt
            if lshape_map[rcv_pr, self.split] > target_map[rcv_pr, self.split]:
                # if there is any data left on the process then send it to the next one
                send_amt = lshape_map[rcv_pr, self.split] - target_map[rcv_pr, self.split]
                self.__redistribute_shuffle(
                    snd_pr=rcv_pr, send_amt=send_amt.item(), rcv_pr=rcv_pr + 1, snd_dtype=snd_dtype
                )
                lshape_cumsum[rcv_pr] -= send_amt
                lshape_cumsum[rcv_pr + 1] += send_amt
                lshape_map[rcv_pr, self.split] -= send_amt
                lshape_map[rcv_pr + 1, self.split] += send_amt

        if any(lshape_map[..., self.split] != target_map[..., self.split]):
            # sometimes need to call the redistribute once more,
            # (in the case that the second to last processes needs to get data from +1 and -1)
            self.redistribute_(lshape_map=lshape_map, target_map=target_map)

        self.__lshape_map = target_map

    def __redistribute_shuffle(
        self,
        snd_pr: int | torch.Tensor,
        send_amt: int | torch.Tensor,
        rcv_pr: int | torch.Tensor,
        snd_dtype: torch.dtype,
    ):
        """
        Function to abstract the function used during redistribute for shuffling data between
        processes along the split axis

        Parameters
        ----------
        snd_pr : int or torch.Tensor
            Sending process
        send_amt : int or torch.Tensor
            Amount of data to be sent by the sending process
        rcv_pr : int or torch.Tensor
            Receiving process
        snd_dtype : torch.dtype
            Torch type of the data in question
        """
        rank = self.comm.rank
        send_slice = [slice(None)] * self.ndim
        keep_slice = [slice(None)] * self.ndim
        if rank == snd_pr:
            if snd_pr < rcv_pr:  # data passed to a higher rank (off the bottom)
                send_slice[self.split] = slice(
                    self.lshape[self.split] - send_amt, self.lshape[self.split]
                )
                keep_slice[self.split] = slice(0, self.lshape[self.split] - send_amt)
            if snd_pr > rcv_pr:  # data passed to a lower rank (off the top)
                send_slice[self.split] = slice(0, send_amt)
                keep_slice[self.split] = slice(send_amt, self.lshape[self.split])
            data = self.__array[tuple(send_slice)].clone()
            self.comm.Send(data, dest=rcv_pr, tag=685)
            self.__array = self.__array[tuple(keep_slice)]
        if rank == rcv_pr:
            shp = list(self.gshape)
            shp[self.split] = send_amt
            data = torch.zeros(shp, dtype=snd_dtype, device=self.device.torch_device)
            self.comm.Recv(data, source=snd_pr, tag=685)
            if snd_pr < rcv_pr:  # data passed from a lower rank (append to top)
                self.__array = torch.cat((data, self.__array), dim=self.split)
            if snd_pr > rcv_pr:  # data passed from a higher rank (append to bottom)
                self.__array = torch.cat((self.__array, data), dim=self.split)

    def resplit_(self, axis: int = None):
        """
        In-place option for resplitting a :class:`DNDarray`.

        Parameters
        ----------
        axis : int
            The new split axis, ``None`` denotes gathering, an int will set the new split axis

        Examples
        --------
        >>> a = ht.zeros(
        ...     (
        ...         4,
        ...         5,
        ...     ),
        ...     split=0,
        ... )
        >>> a.lshape
        (0/2) (2, 5)
        (1/2) (2, 5)
        >>> ht.resplit_(a, None)
        >>> a.split
        None
        >>> a.lshape
        (0/2) (4, 5)
        (1/2) (4, 5)
        >>> a = ht.zeros(
        ...     (
        ...         4,
        ...         5,
        ...     ),
        ...     split=0,
        ... )
        >>> a.lshape
        (0/2) (2, 5)
        (1/2) (2, 5)
        >>> ht.resplit_(a, 1)
        >>> a.split
        1
        >>> a.lshape
        (0/2) (4, 3)
        (1/2) (4, 2)
        """
        # sanitize the axis to check whether it is in range
        axis = sanitize_axis(self.shape, axis)

        self.__partitions_dict__ = None

        # early out for unchanged content
        if self.comm.size == 1:
            self.__split = axis
        if axis == self.split:
            return self

        if axis is None:
            gathered = torch.empty(
                self.shape, dtype=self.dtype.torch_type(), device=self.device.torch_device
            )
            counts, displs = self.counts_displs()
            self.comm.Allgatherv(self.__array, (gathered, counts, displs), recv_axis=self.split)
            self.__array = gathered
            self.__split = axis
            self.__lshape_map = None
            return self
        # tensor needs be split/sliced locally
        if self.split is None:
            # new_arr = self
            _, _, slices = self.comm.chunk(self.shape, axis)
            temp = self.__array[slices]
            self.__array = torch.empty((1,), device=self.device.torch_device)
            # necessary to clear storage of local __array
            self.__array = temp.clone().detach()
            self.__split = axis
            self.__lshape_map = None
            return self

        arr_tiles = tiling.SplitTiles(self)
        new_tiles = tiling.SplitTiles(self)

        gshape = self.shape
        new_lshape = list(gshape)
        new_lshape[axis] = int(arr_tiles.tile_dimensions[axis][self.comm.rank].item())

        recv_buffer = torch.empty(
            tuple(new_lshape), dtype=self.dtype.torch_type(), device=self.device.torch_device
        )

        self._axis2axisResplit(
            self.larray, self.split, arr_tiles, recv_buffer, axis, new_tiles, self.comm
        )

        self.__array = recv_buffer
        self.__split = axis
        self.__lshape_map = None

        return self

    def __setitem_scalar(self, p: ProcessedKey, value: "DNDarray", value_is_scalar: bool) -> None:
        if p.root is not None:
            if self.comm.rank == p.root:
                indexed_proxy = self.__torch_proxy__()[p.key]
                if indexed_proxy.names.count("split") != 0:
                    indexed_lshape_map = self.lshape_map[:, 1:]
                    if value.lshape_map != indexed_lshape_map:
                        try:
                            value.redistribute_(target_map=indexed_lshape_map)
                        except ValueError:
                            raise ValueError(
                                f"cannot assign value to indexed DNDarray because "
                                f"distribution schemes do not match: "
                                f"{value.lshape_map} vs. {indexed_lshape_map}"
                            )
                self.__set(p.key, value)
        else:
            if not value_is_scalar:
                value = sanitation.sanitize_distribution(value, target=self[p.key])
            self.__set(p.key, value)

    def __setitem_slice(self, p: ProcessedKey, value: "DNDarray", value_is_scalar: bool) -> None:
        if not self.is_distributed() and not value.is_distributed():
            self.__set(p.key, value)
            return

        if self.is_distributed() and not value_is_scalar:
            if not value.is_distributed():
                value = factories.array(
                    value.larray,
                    dtype=value.dtype,
                    split=p.output_split,
                    device=self.device,
                    comm=self.comm,
                )
            else:
                if value.split != p.output_split:
                    raise RuntimeError(
                        f"Cannot assign distributed `value` with split axis {value.split} "
                        f"to indexed DNDarray with split axis {p.output_split}."
                    )
            target_shape = torch.tensor(
                tuple(self.larray[p.key].shape), device=self.device.torch_device
            )
            target_map = torch.zeros(
                (self.comm.size, len(target_shape)),
                dtype=torch.int64,
                device=self.device.torch_device,
            )
            self.comm.Allgather(target_shape, target_map)
            value.redistribute_(target_map=target_map)

        self.__set(p.key, value)

    def __setitem_advanced_local(
        self, p: ProcessedKey, original_key, value: "DNDarray", value_is_scalar: bool
    ) -> None:
        self.__setitem_slice(p, value, value_is_scalar)

    def __setitem_descending_slice_distributed(
        self, p: ProcessedKey, value: "DNDarray", value_is_scalar: bool
    ) -> None:
        flipped_value = manipulations.flip(value, axis=p.output_split)
        split_key = factories.array(
            p.key[self.split], is_split=0, device=self.device, comm=self.comm
        )
        if not flipped_value.is_distributed():
            flipped_value = factories.array(
                flipped_value.larray,
                dtype=flipped_value.dtype,
                split=p.output_split,
                device=self.device,
                comm=self.comm,
            )
        target_map = flipped_value.lshape_map
        target_map[:, p.output_split] = split_key.lshape_map[:, 0]
        flipped_value.redistribute_(target_map=target_map)
        self.__set(p.key, flipped_value)

    def __setitem_mask(
        self, p: ProcessedKey, original_key, value: "DNDarray", value_is_scalar: bool
    ) -> None:
        pytorch_key = p.key

        if isinstance(pytorch_key, tuple):
            for k in pytorch_key:
                if isinstance(k, torch.Tensor) and k.dtype in (torch.bool, torch.uint8):
                    local_mask = k
                    break
        else:
            local_mask = pytorch_key

        if value_is_scalar:
            if hasattr(value, "larray"):
                scalar_torch = value.larray
            else:
                scalar_torch = torch.as_tensor(value, device=self.device.torch_device)
            scalar_torch = scalar_torch.type(self.dtype.torch_type())
            self.larray[pytorch_key] = scalar_torch
        else:
            if isinstance(value, DNDarray) and value.is_distributed():
                expected_elements = int(local_mask.sum().item())
                if value.lshape[0] != expected_elements:
                    raise ValueError(
                        f"Shape mismatch: Cannot assign distributed array with local shape {value.lshape} "
                        f"to a mask requiring {expected_elements} elements on rank {self.comm.rank}."
                    )

                # value perfectly aligns
                value_torch = value.larray
                self.larray[pytorch_key] = value_torch.type(self.dtype.torch_type())

            else:
                # Value is a non-distributed array -> MPI prefix sum needed
                if hasattr(value, "larray"):
                    value_torch = value.larray
                else:
                    value_torch = torch.as_tensor(value, device=self.device.torch_device)

                # distinguish between exact-shape masks and 1D row-filtering masks
                is_row_mask = local_mask.ndim == 1 and self.ndim > 1

                if not is_row_mask and value_torch.ndim == 1:
                    # N-D mask on N-D array -> flattens into 1D sequence, requires MPI prefix sum
                    local_mask_flat = local_mask.flatten()
                    local_true = int(local_mask_flat.sum().item())

                    if self.comm.size > 1:
                        if self.comm.rank == 0:
                            offset = 0
                            _ = self.comm.exscan(local_true)
                        else:
                            offset = self.comm.exscan(local_true)
                    else:
                        offset = 0

                    rhs_local = value_torch[offset : offset + local_true].type(
                        self.dtype.torch_type()
                    )

                    x_flat = self.larray.view(-1)
                    x_flat[local_mask_flat] = rhs_local
                else:
                    # PyTorch assigns and broadcasts natively
                    self.larray[pytorch_key] = value_torch.type(self.dtype.torch_type())

    def __setitem_advanced_distributed(
        self, p: ProcessedKey, original_key, value: "DNDarray", value_is_scalar: bool
    ) -> None:
        # check distribution status of the indexing key
        split_key_orig = (
            original_key[self.split] if isinstance(original_key, tuple) else original_key
        )
        key_is_distributed = (
            isinstance(split_key_orig, DNDarray) and split_key_orig.is_distributed()
        )

        # reject implicit cross-distribution assignments
        if key_is_distributed and not value.is_distributed() and not value_is_scalar:
            raise ValueError(
                f"Distribution mismatch: index distributed={key_is_distributed}, value distributed={value.is_distributed()}. "
                "Cannot assign a non-distributed value array using a distributed index. "
                "Please distribute the value array or use a non-distributed index."
            )

        if value.is_distributed():
            self.__setitem_unordered(
                key=p.key,
                key_is_mask_like=p.key_is_mask_like,
                value=value,
                key_is_single_tensor=isinstance(p.key, torch.Tensor),
                counts=self.counts_displs()[0],
                displs=self.counts_displs()[1],
                rank=self.comm.rank,
                backwards_transpose_axes=p.backwards_transpose_axes,
            )
            return

        counts, displs = self.counts_displs()
        rank = self.comm.rank
        key_is_single_tensor = isinstance(p.key, torch.Tensor)

        if (
            value_is_scalar
            and isinstance(original_key, tuple)
            and len(original_key) == self.ndim
            and all(
                isinstance(k, DNDarray) and k.ndim == 1 and k.dtype in (types.int32, types.int64)
                for k in original_key
            )
        ):
            global_indices = []
            for k in original_key:
                k_full = k.copy()
                k_full.resplit_(None)
                global_indices.append(k_full.larray)

            idx_split_global = global_indices[self.split]
            local_offset = displs[rank]
            local_size = counts[rank]

            mask = (idx_split_global >= local_offset) & (
                idx_split_global < local_offset + local_size
            )
            if not mask.any():
                return

            lhs_index = []
            for dim, gind in enumerate(global_indices):
                sel = gind[mask]
                if dim == self.split:
                    sel = sel - local_offset
                lhs_index.append(sel)
            lhs_index = tuple(lhs_index)

            if hasattr(value, "larray"):
                scalar_torch = value.larray
            else:
                scalar_torch = torch.as_tensor(value, device=self.device.torch_device)
            scalar_torch = scalar_torch.type(self.dtype.torch_type())

            self.larray[lhs_index] = scalar_torch
            return

        if key_is_single_tensor:
            split_key = p.key
            split_key_flat = split_key.reshape(-1)
            local_indices = torch.nonzero(
                (split_key_flat >= displs[rank]) & (split_key_flat < displs[rank] + counts[rank])
            ).flatten()

            if local_indices.numel() > 0:
                key_local = split_key_flat[local_indices] - displs[rank]

                if value_is_scalar:
                    rhs = value.larray.type(self.dtype.torch_type())
                else:
                    # flatten leading dimensions of value.larray that correspond to the multi-dimensional key
                    rhs_view = value.larray.reshape(-1, *value.larray.shape[split_key.ndim :])
                    rhs = rhs_view[local_indices].type(self.dtype.torch_type())

                if self.larray.is_cuda:
                    key_local, rhs = _resolve_duplicate_indices(key_local, rhs, self.larray.shape)

                self.larray[key_local] = rhs
            return

        if isinstance(original_key, tuple):
            original_split_axis = p.backwards_transpose_axes[self.split]
            raw_split_part = original_key[original_split_axis]
        else:
            raw_split_part = original_key

        if isinstance(raw_split_part, DNDarray):
            split_key = raw_split_part.larray
        elif isinstance(raw_split_part, torch.Tensor):
            split_key = raw_split_part
        else:
            split_key = p.key[self.split]

        if isinstance(split_key, DNDarray):
            split_key = split_key.larray

        if split_key.dtype == torch.bool:
            split_key = torch.nonzero(split_key, as_tuple=False).flatten()

        local_offset = displs[rank]
        local_size = counts[rank]

        if hasattr(value, "larray"):
            value_torch = value.larray
        else:
            value_torch = torch.as_tensor(value, device=self.device.torch_device)

        feature_dims = self.larray.ndim - (self.split + 1)

        if value_is_scalar:
            value_key_start_dim = 0
        else:
            value_key_start_dim = value_torch.ndim - split_key.ndim - feature_dims
            if value_key_start_dim < 0:
                raise RuntimeError("value_key_start_dim < 0 – inconsistent shapes")

        local_split_axis = self.split

        base_index = [slice(None)] * self.larray.ndim
        if isinstance(original_key, tuple):
            for dim, k_part in enumerate(original_key):
                if dim == self.split:
                    continue
                if isinstance(k_part, DNDarray):
                    base_index[dim] = k_part.larray
                else:
                    base_index[dim] = k_part

        self.__advanced_setitem_unordered_local(
            x_local=self.larray,
            split_key=split_key,
            value_torch=value_torch,
            split_axis=local_split_axis,
            value_key_start_dim=value_key_start_dim,
            local_offset=local_offset,
            local_size=local_size,
            value_is_scalar=value_is_scalar,
            out_dtype=self.dtype.torch_type(),
            base_index=tuple(base_index),
        )

    def __setitem_unordered(
        self,
        key: tuple | list | torch.Tensor,
        key_is_mask_like: bool,
        value: "DNDarray",
        key_is_single_tensor: bool,
        counts: tuple,
        displs: tuple,
        rank: int,
        backwards_transpose_axes: tuple,
    ) -> DNDarray:
        """
        Handles the MPI communication when assigning a distributed
        value to a distributed array with unordered global indices.
        """
        # distribution of `key` and `value` must be aligned
        if key_is_mask_like:
            # redistribute `value` to match distribution of `key` in one pass
            split_key = key[self.split]
            global_split_key = factories.array(
                split_key, is_split=0, device=self.device, comm=self.comm, copy=False
            )
            target_map = value.lshape_map
            target_map[:, value.split] = global_split_key.lshape_map[:, 0]
            value.redistribute_(target_map=target_map)
        else:
            # redistribute split-axis `key` to match distribution of `value` in one pass
            if key_is_single_tensor:
                # key is a single torch.Tensor
                split_key = key
            else:
                split_key = key[self.split]
            global_split_key = factories.array(
                split_key, is_split=0, device=self.device, comm=self.comm, copy=False
            )
            target_map = global_split_key.lshape_map
            target_map[:, 0] = value.lshape_map[:, value.split]
            global_split_key.redistribute_(target_map=target_map)
            split_key = global_split_key.larray

        # key and value are now aligned

        # prepare for `value` Alltoallv:
        # work along axis 0, transpose if necessary
        transpose_axes = list(range(value.ndim))
        transpose_axes[0], transpose_axes[value.split] = (
            transpose_axes[value.split],
            transpose_axes[0],
        )
        value = value.transpose(transpose_axes)

        split_key_flat = split_key.reshape(-1)
        sort_idx, send_counts, send_displs, recv_counts, recv_displs = (
            self.__prepare_unordered_comm(split_key_flat, displs)
        )

        # allocate send buffer: add 1 column to store sent indices
        send_buf_shape = list(value.lshape)
        if value.ndim < 2:
            send_buf_shape.append(1)
        if key_is_mask_like:
            send_buf_shape[-1] += len(key)
        else:
            send_buf_shape[-1] += 1
        send_buf = torch.zeros(
            send_buf_shape, dtype=value.dtype.torch_type(), device=self.device.torch_device
        )

        # pack the send_buf
        if sort_idx.numel() > 0:
            if value.ndim < 2:
                send_buf[:, :-1] = value.larray[sort_idx].unsqueeze(1)
            else:
                send_buf[:, :-1] = value.larray[sort_idx]

            if key_is_mask_like:
                for i in range(-len(key), 0):
                    send_buf[:, i] = key[i + len(key)][sort_idx]
            else:
                send_buf[:, -1] = split_key_flat[sort_idx].to(send_buf.dtype)

        # allocate receive buffer, with 1 extra column for incoming indices
        recv_buf_shape = value.lshape_map[self.comm.rank]
        recv_buf_shape[value.split] = recv_counts.sum()
        recv_buf_shape = recv_buf_shape.tolist()
        if value.ndim < 2:
            recv_buf_shape.append(1)
        if key_is_mask_like:
            recv_buf_shape[-1] += len(key)
        else:
            recv_buf_shape[-1] += 1
        recv_buf_shape = tuple(recv_buf_shape)
        recv_buf = torch.zeros(
            recv_buf_shape, dtype=value.dtype.torch_type(), device=self.device.torch_device
        )
        # perform Alltoallv along the 0 axis
        send_counts, send_displs, recv_counts, recv_displs = (
            send_counts.tolist(),
            send_displs.tolist(),
            recv_counts.tolist(),
            recv_displs.tolist(),
        )
        self.comm.Alltoallv(
            (send_buf, send_counts, send_displs), (recv_buf, recv_counts, recv_displs)
        )
        del send_buf

        if key_is_mask_like:
            key = list(key)
            # extract incoming indices from recv_buf
            recv_indices = recv_buf[..., -len(key) :]
            # correct split-axis indices for rank offset
            recv_indices[:, 0] -= displs[rank]
            key = recv_indices.split(1, dim=1)
            key = [key[i].squeeze_(1) for i in range(len(key))]
            # remove indices from recv_buf
            recv_buf = recv_buf[..., : -len(key)]
        else:
            # store incoming indices in int 1-D tensor and correct for rank offset
            recv_indices = recv_buf[..., -1].type(torch.int64) - displs[rank]
            # remove last column from recv_buf
            recv_buf = recv_buf[..., :-1]

            # replace split-axis key with incoming local indices
            if key_is_single_tensor:
                key = recv_indices
            else:
                key = list(key)
                key[self.split] = recv_indices
                key = tuple(key)

        # transpose back value and recv_buf if necessary, wrap recv_buf in DNDarray
        value = value.transpose(transpose_axes)
        if value.ndim < 2:
            recv_buf.squeeze_(1)
        recv_buf = DNDarray(
            recv_buf.permute(*transpose_axes),
            gshape=value.gshape,
            dtype=value.dtype,
            split=value.split,
            device=value.device,
            comm=value.comm,
            balanced=value.balanced,
        )
        # set local elements of `self` to corresponding elements of `value`
        self.__set(key, recv_buf)
        if self.ndim > 0:
            return self.transpose(backwards_transpose_axes)
        return self

    def __setitem__(
        self,
        key: Indexer,
        value: float | "DNDarray" | torch.Tensor,
    ):
        """
        Global item setter

        Parameters
        ----------
        key : array-like indexer
            Index/indices to be set
        value: float | "DNDarray" | torch.Tensor
            Value to be set to the specified positions in the DNDarray (self)

        Notes
        -----
        If a ``DNDarray`` is given as the value to be set then the split axes are assumed to be equal.
        If they are not, PyTorch will raise an error when the values are attempted to be set on the local array

        Examples
        --------
        >>> a = ht.zeros((4, 5), split=0)
        (1/2) >>> tensor([[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]])
        (2/2) >>> tensor([[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]])
        >>> a[1:4, 1] = 1
        >>> a
        (1/2) >>> tensor([[0., 0., 0., 0., 0.],
                          [0., 1., 0., 0., 0.]])
        (2/2) >>> tensor([[0., 1., 0., 0., 0.],
                          [0., 1., 0., 0., 0.]])
        """
        try:
            value = factories.array(value)
        except TypeError:
            raise TypeError(f"Cannot assign object of type {type(value)} to DNDarray.")

        original_key = key

        self, processed_key = _resolve_indexing_state(
            self, key, return_local_indices=True, op="set"
        )
        # print(f"DEBUGGING: Processed key: {processed_key}")

        op = processed_key.op_type

        # match dimensions (except for distr_mask as it perfectly aligns)
        if op == "distr_mask":
            value_is_scalar = (
                np.isscalar(value)
                or getattr(value, "ndim", 1) == 0
                or (getattr(value, "shape", None) == (1,) and getattr(value, "split", 0) is None)
            )
        else:
            value, value_is_scalar = self.__broadcast_value(
                key, value, output_shape=processed_key.output_shape
            )

        # dispatch to the appropriate setter
        if op == "distr_mask":
            self.__setitem_mask(processed_key, original_key, value, value_is_scalar)
        elif op == "scalar":
            self.__setitem_scalar(processed_key, value, value_is_scalar)
        elif op == "distributed":
            self.__setitem_advanced_distributed(processed_key, original_key, value, value_is_scalar)
        elif op == "slice":
            self.__setitem_slice(processed_key, value, value_is_scalar)
        elif op == "descending_slice":
            self.__setitem_descending_slice_distributed(processed_key, value, value_is_scalar)
        elif op in ("local_mask", "advanced"):
            self.__setitem_advanced_local(processed_key, original_key, value, value_is_scalar)

    def __str__(self) -> str:
        """
        Computes a string representation of the passed ``DNDarray``.
        """
        return printing.__str__(self)

    def tolist(self, keepsplit: bool = False) -> list:
        """
        Return a copy of the local array data as a (nested) Python list. For scalars, a standard Python number is returned.

        Parameters
        ----------
        keepsplit: bool
            Whether the list should be returned locally or globally.

        Examples
        --------
        >>> a = ht.array([[0, 1], [2, 3]])
        >>> a.tolist()
        [[0, 1], [2, 3]]

        >>> a = ht.array([[0, 1], [2, 3]], split=0)
        >>> a.tolist()
        [[0, 1], [2, 3]]

        >>> a = ht.array([[0, 1], [2, 3]], split=1)
        >>> a.tolist(keepsplit=True)
        (1/2) [[0], [2]]
        (2/2) [[1], [3]]
        """
        if not keepsplit:
            return self.resplit(axis=None).__array.tolist()

        return self.__array.tolist()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Supports PyTorch's dispatch mechanism.
        """
        import heat

        if kwargs is None:
            kwargs = {}
        try:
            ht_func = getattr(heat, func.__name__)
        except AttributeError:
            return NotImplemented
        return ht_func(*args, **kwargs)

    def __torch_proxy__(self) -> torch.Tensor:
        """
        Return a 1-element `torch.Tensor` strided as the global `self` shape. The split axis of the initial DNDarray is stored in the `names` attribute of the returned tensor.
        Used internally to lower memory footprint of sanitation.
        """
        names = [None] * self.ndim
        if self.split is not None:
            names[self.split] = "split"
        return (
            torch.ones((1,), dtype=torch.int8, device=self.larray.device)
            .as_strided(self.gshape, [0] * self.ndim)
            .refine_names(*names)
        )


# Heat imports at the end to break cyclic dependencies
from . import complex_math
from . import devices
from . import factories
from . import indexing
from . import linalg
from . import manipulations
from . import printing
from . import rounding
from . import sanitation
from . import statistics
from . import stride_tricks
from . import tiling
from . import types

from .devices import Device
from .stride_tricks import sanitize_axis
from .types import datatype, canonical_heat_type
from .types import bool as ht_bool, uint8 as ht_uint8
