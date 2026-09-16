
# Indexing on DNDarrays

Heat v1.9 introduces fully distributed indexing for DNDarrays. While the indexing behaviour is designed to be highly compatible with the NumPy API, the memory-distributed nature of DNDarrays introduces unique considerations regarding performance and communication overhead. In the following sections, we will cover the basics plus some of these Heat-specific indexing features.

*Note: This guide is heavily inspired by the official [NumPy indexing documentation](https://numpy.org/doc/stable/user/basics.indexing.html).*

## Distributed indexing

We work under the assumption that Heat users process data in very large, memory-distributed arrays. In the following, we will refer to `array`, `key`, and `value` as the DNDarray, the index/combination of indices, and (if present) the value to be assigned to the index, respectively. Examples:

- item getting: `array[key]`
- item setting: `array[key] = value`

We assume that not only `array`,  but also `key` and `value` may be very large and distributed across MPI processes if the use case requires.

The following table shows the distribution semantics of the DNDarray indexing operations.

| Array is distributed | Operation | Key is distributed | Value is distributed | Result is distributed | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **No** | `array[key]` | **No** | -- | **No** | Standard local indexing directly on underlying torch tensor. |
| **No** | `array[key]` | **Yes** | -- | **Yes** | For a 1D distributed key, the output inherits `split` and balanced status from the key. |
| **Yes** | `array[key]` | **No** | -- | **Yes** / **No** | Scalar `key` on split axis collapses that dimension, output is replicated on each process (`split=None`). For all other key types distribution is maintained. |
| **Yes** | `array[key]` | **Yes** | -- | **Yes** | **Local path:** Aligned boolean mask flattens locally with 0 communication.<br>**Communication path:** Unordered distributed integer indices trigger `__getitem_unordered` with `Alltoallv` exchange. |
| **No** | `array[key] = val` | **No** | **No** | **No** (In-place) | In-place assignment directly on underlying tensor. |
| **Yes** | `array[key] = val` | **No** | **No** | **Yes** (In-place) | **Scalars:** Assigned directly with 0 communication (PyTorch broadcasts locally).<br>**Local arrays:** Converted to a distributed array matching the target split axis and aligned via `redistribute_`. |
| **Yes** | `array[key] = val` | **No** | **Yes** | **Yes** (In-place) | **Split axis match required:** If `value.split != target.split`, raises a `RuntimeError`.  |
| **Yes** | `array[key] = val` | **Yes** | **No, scalar** | **Yes** (In-place) | Python scalars and 0-D tensors assign directly to all local masked/indexed positions. |
| **Yes** | `array[key] = val` | **Yes** | **No, array** | **ERROR** / **Yes** | **Supported** only for boolean mask key, otherwise **ValueError** is raised. |
| **Yes** | `array[key] = val` | **Yes** | **Yes** | **Yes** (In-place) | **Aligned boolean mask key:** Local assignment with 0 communication.<br>**Unordered integer indices:** `key` is redistributed to match `value`, followed by a dual `Alltoallv` shuffle (indices and data payload). |

*Note: Extracting a single element along the split axis will collapse that dimension, resulting in `split=None`.*

---

## Basic slicing and indexing

Basic slicing extends Python's basic concept of slicing to N dimensions. It occurs when `key` is a `slice` object (constructed by `start:stop:step` notation inside brackets), an integer, or a tuple of slice objects and integers.

### Single element indexing
When indexing a single element or a specific slice that reduces the dimensionality of the array, the `split` axis is dynamically updated. If the array is indexed with an integer along the dimension it is split on, that dimension is collapsed and the resulting slice is no longer distributed along that axis.

```python
import heat as ht

# 1D array distributed across processes
x = ht.arange(10, split=0)
# indexing collapses the 0th dimension; the result is no longer distributed
result = x[2]
# result.split is None
```

If the array is multi-dimensional and split on an axis that is not the one being collapsed, the split axis shifts to account for the removed dimension.

```python
# 2D array distributed along axis 1 (columns)
x = ht.arange(10).reshape(2, 5)
x_split1 = ht.array(x, split=1)

# selecting a specific row collapses axis 0
result = x_split1[0]
# result.split is 0, because the old axis 1 is now the new axis 0
```

### Slicing and striding
Standard slicing `start:stop:step` preserves the dimensions of the array. The array remains distributed along the original split axis. Negative steps are supported and will reverse the elements locally while executing collective communication to reverse the chunks globally.

```python
x = ht.arange(20, split=0)
# slice with a step
result = x[1:11:3]
# result.split remains 0
```

### Dimensional indexing

You can manipulate the dimensionality of a DNDarray directly inside the brackets using ht.newaxis (or None) and ... (Ellipsis).

- `None` or `np.newaxis` inserts a new axis of size 1 into the array's shape. If the array is distributed, inserting an axis before the split axis will cause the split axis index to shift by +1.

- `...` expands to the number of `:` objects needed to make a selection tuple of the same length as the array dimensions.

```python
x = ht.array([[[1], [2], [3]], [[4], [5], [6]]], split=1)

# adds a new dimension at axis 1
x_newaxis = x[:, None, :2, :]
# original split was 1; new split is 2
```

## Advanced indexing

Advanced indexing is triggered when the selection object key is a non-tuple sequence object, a DNDarray (of integer or boolean data type), a torch.Tensor, or a tuple with at least one sequence object or multi-dimensional array.

Advanced indexing always returns a copy of the data (contrast with basic slicing that returns a view).

### Integer array indexing

You can use DNDarray objects containing integers to select arbitrary items. The resulting array will take on the distribution map of the indexing key.


```python
# array split along axis 0
x = ht.arange(60, split=0).reshape(5, 3, 4)

# using multiple non-distributed indices
k1 = ht.array([0, 4, 1, 0])
k2 = ht.array([0, 2, 1, 0])
k3 = ht.array([1, 2, 3, 1])

# standard advanced indexing
result = x[k1, k2, k3]
```

### Boolean array indexing

Boolean arrays used as indices are treated as a mask. The result is a 1-D array containing the elements that correspond to True in the boolean array.

```python
arr = ht.arange(60, split=0).reshape(3, 4, 5)
mask = arr > 30

# returns a 1D array of all elements > 30, split along axis 0
result = arr[mask]
```

Row-selection optimization: Heat implements a highly optimized fast-path for the common data science pattern of row-filtering. If you index a 2D array split along axis 0 with a 1D boolean mask that is also split along axis 0, Heat skips the heavy distributed indexing machinery. It applies the mask locally and resolves the global shape via a fast metadata exchange. The output remains a 2D array split along axis 0.


```python
arr_2d = ht.arange(20, split=0).reshape((10, 2))
mask_1d = ht.array([True, False, True, False, True, False, True, False, True, False], split=0)

# the result remains a 2D array (shape: 5, 2) and retains split=0
result = arr_2d[mask_1d]
```

### In-place assignment (setitem)

Advanced indexing can be used to assign values. If the assignment value is itself a distributed DNDarray, Heat will automatically execute a distributed routing protocol (via Alltoallv) to align the spatial memory distribution of the values with the target indices before executing the local assignments.

```python
x = ht.arange(10 * 20 * 30, split=1).reshape(10, 20, 30)

# boolean mask assignment
mask = x > 100
x[mask] = 99.0

# advanced integer assignment with an aligned distributed value
# (assigning 10 elements along axis 1 on a 1D slice across all other dimensions)
indices = ht.array([2, 5, 8, 11], dtype=ht.int64, split=0)
value = ht.ones((10, 4, 30), split=1)

x[:, indices, :] = value
```

## Combining advanced and basic indexing

When you mix advanced indexing (like integer arrays or lists) with basic slicing (like `:`), the shape of the resulting `DNDarray` depends on whether the advanced indices are positioned next to each other.

Heat follows NumPy's standard transposition rules for mixed indexing, while automatically managing the distributed memory alignment internally. The array's nominal `split` axis will track the new dimensional layout.

### Advanced indexing on consecutive dimensions
If the advanced indices are adjacent to each other (not separated by a slice), the resulting broadcasted shape of the advanced indices is inserted directly into the output shape at the position of the first advanced index.

If the original array's `split` axis is untouched by the advanced indexing, it will simply shift to account for the collapsed dimensions.

```python
import heat as ht

# arr shape: (10, 20, 30, 40), distributed along axis 3
arr = ht.zeros((10, 20, 30, 40), split=3)
a1 = ht.array([1, 2])
a2 = ht.array([3, 4])

# Advanced indices are consecutive on axes 1 and 2
result = arr[:, a1, a2, :]

# The advanced indices on axes 1 and 2 broadcast to a single shape (2,)
# Result shape: (10, 2, 40)

# The original split axis 3 is now the last dimension in the new shape.
# result.split is 2
```

### Advanced indexing on non-consecutive dimensions

If the advanced indices are separated by a basic slice, the resulting layout becomes ambiguous. To resolve this, the advanced-indexing dimensions are grouped together and transposed to the very front of the resulting array's shape.

Any remaining basic slices follow behind them. The split axis is tracked through this transposition and assigned its new relative index.

```python
import heat as ht

# arr shape: (10, 20, 30, 40), distributed along axis 3
arr = ht.zeros((10, 20, 30, 40), split=3)
a1 = ht.array([1, 2])
a2 = ht.array([3, 4])

# Advanced indices (axes 0 and 2) are separated by a slice (axis 1)
result = arr[a1, :, a2, :]

# The advanced indices broadcast to shape (2,) and are moved to the front.
# The untouched basic slices (from axes 1 and 3) are appended to the back.
# Result shape: (2, 20, 40)

# The original split axis 3 is still the last dimension in the new array.
# result.split is 2
```

## Communication overhead

The indexing operations evaluate the state of the indexing key to determine the most efficient network routing strategy. The communication overhead ranges from completely zero (purely local execution) to heavy all-to-all exchanges for non-sequential advanced indexing.

Here are the different possible configurations, categorized and ordered from the lowest communication overhead to the highest within each category.

### Summary of Communication Overhead

| Category | Configuration (Operation & State) | Communication Overhead (MPI Calls) |
| :--- | :--- | :--- |
| **Single Element Indexing** | `array[key]` (key is an integer on a *non-split* axis) | **None** |
| | `array[key] = local_value` (key is an int on the *split* axis) | **None** (Only the root rank executes the local set) |
| | `array[key]` (key is an int on the *split* axis) | **1 `Bcast`** (Root extracts value and broadcasts to all ranks) |
| **Slicing & Striding** | `array[slice]` or `array[slice] = local_value` | **None** |
| | `array[::-1]` (Descending slice along split axis) | **None** (Executes local slice followed by a global `flip` operation) |
| | `array[::-1] = distributed_value` (Descending slice write) | **Multiple `Send`/`Recv`** (Executes `redistribute_` using point-to-point transfers if array slice and value are misaligned) |
| **Dimensional Indexing** | `array[..., None]` or `array[:, np.newaxis]` | **None** |
| **Advanced Indexing** | `array[mask]` (1D or full bool mask, split=0) | **1 `Allreduce`** (Applies mask locally, reduces element counts to compute `gshape`) |
|                   | `array[non_seq_key] = local_value` | **1 `Allreduce`** (Batched validation for bounds and negative coordinates) |
|                   | `array[non_seq_key]` (Unstructured read) | **1 `Alltoall` + 2 `Alltoallv`** (Exchanges counts, requests indices, returns data) |
| **Slicing & Striding**| `array[::-1]` (Descending slice along split axis) | **Point-to-point / Redistribution** (Local slice followed by distributed `flip`) |
