
# Indexing on DNDarrays

Heat v2.0 introduces fully distributed indexing for DNDarrays. While the indexing behaviour is designed to be highly compatible with the NumPy API, the memory-distributed nature of DNDarrays introduces unique considerations regarding performance and communication overhead. In the following sections, we will cover the basics plus some of these Heat-specific indexing features.

*Note: This guide is heavily inspired by the official [NumPy indexing documentation](https://numpy.org/doc/stable/user/basics.indexing.html).*

## Distributed indexing

We work under the assumption that Heat users process data in very large, memory-distributed arrays. In the following, we will refer to `array`, `key`, and `value` as the DNDarray, the index/combination of indices, and (if present) the value to be assigned to the index, respectively. Examples:

- item getting: `array[key]`
- item setting: `array[key] = value`

We assume that not only `array`,  but also `key` and `value` may be very large and distributed across MPI processes if the use case requires.

The following table shows the distribution semantics of the DNDarray indexing operations.

| Array is distributed | Operation | Key is distributed | Value is distributed | Result is distributed | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **No** | `array[key]` | **No** | -- | **No** | Standard local indexing. |
| **No** | `array[key]` | **Yes** | -- | **Yes** | The resulting array inherits the `split` axis and balanced status directly from the distributed key. |
| **Yes** | `array[key]` | **No** | -- | **Yes** / **No** | **No** if the key is a pure scalar along the split axis (the split dimension is lost and the result is broadcasted).<br>**Yes** for slices/masks. Non-sequential local advanced indices are automatically distributed across the split axis under the hood. |
| **Yes** | `array[key]` | **Yes** | -- | **Yes** | Split axis is retained or shifted. Evaluated as a `distr_mask` fast-path or triggers `__getitem_unordered` for cross-node MPI collective fetching. |
| **No** | `array[key] = val` | **No** | **No** | **No** (In-place) | Standard local assignment.  |
| **Yes** | `array[key] = val` | **No** | **No** | **Yes** (In-place) | The local value is automatically converted into a distributed array and broadcasted to align with the array's distribution constraints. |
| **Yes** | `array[key] = val` | **No** | **Yes** | **Yes** (In-place) | **Split axis match required:** If the `value`'s split axis doesn't match the target's split axis, a `RuntimeError` is raised. If they do match, `value` is dynamically load-balanced (`redistribute_`) to match the target's chunk sizes before assignment. |
| **Yes** | `array[key] = val` | **Yes** | **No, scalar** | **Yes** (In-place) | A pure scalar value is correctly assigned to all masked/indexed elements across all MPI ranks natively. |
| **Yes** | `array[key] = val` | **Yes** | **No, array** | **ERROR** / **Yes** | **Exception raised** for integer indices. **Supported** for boolean masks via MPI prefix sums to dynamically slice the non-distributed array. |
| **Yes** | `array[key] = val` | **Yes** | **Yes** | **Yes** (In-place) | **Communication-heavy:** For masks, `value` is redistributed to match `key`. For integer arrays, `key` is redistributed to match `value`. Both are followed by an `Alltoallv` shuffle. |

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

# advanced integer assignment with a distributed value
indices = ht.random.randint(0, 20, (2, 3, 4), dtype=ht.int64, split=0)
value = ht.ones((1, 2, 3, 4, 1), split=0)

# value is automatically broadcasted and redistributed to match 'x[..., indices, :]'
x[..., indices, :] = value
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

The indexing operations dynamically evaluate the state of the indexing key to determine the most efficient network routing strategy. The communication overhead ranges from completely zero (purely local execution) to heavy all-to-all exchanges for non-sequential advanced indexing.

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
| **Advanced Indexing** | `array[mask] = local_value` (Boolean mask assignment) | **None / 1 `exscan`** (Zero for scalars; requires prefix sum for 1D local arrays) |
| | `array[mask]` (1D bool mask on 2D array, both split=0) | **None** (Locally applied, delegates global shape resolution to `factories.array`) |
| | `array[non_seq_key] = local_value` (Integer array assignment) | **2 `Allreduce`** (Evaluates global key bounds, then applies locally) |
| | `array[non_seq_key]` (Standard unstructured advanced read) | **1 `Allgather` + 2 `Alltoallv`** (Builds comm matrix, requests indices, returns data) |
| | `array[non_seq_key] = distributed_value` | **2 `Allreduce` + 1 `Allgather` + 1 `Alltoallv`** (+ hidden P2P in `redistribute_` if necessary) |

---

### Detailed Breakdown by Category

#### 1. Slicing and striding
* **Zero Overhead (Fast Path):** If the key consists solely of basic components (slices with positive steps, integers, `None`, or `...`), the `_resolve_indexing_state` method dynamically assigns an `op_type` (like `"slice"` or `"scalar"`) that bypasses state-checking `Allreduce` calls entirely, resulting in zero MPI overhead.
* **Negative Slicing / Descending Strides (Low to Moderate Overhead):** Because PyTorch does not natively support negative slice steps, descending slices (e.g., `[::-1]`) are caught during key processing and explicitly converted into integer tensors (`torch.arange`).
  * For **Reads** (`__getitem__`), the `op_type` is evaluated as `"descending_slice"`. This bypasses non-sequential routing and triggers `__getitem_descending_slice_distributed`, which performs a local slice and wraps the result in an unbalanced array before executing a global `flip` operation.
  * For **Writes** (`__setitem__`), a specific handler for `"descending_slice"` flips the right-hand value and dynamically matches its distribution map to the key, triggering point-to-point `Send`/`Recv` exchanges via the `redistribute_` method.

#### 2. Dimensional indexing
* **Zero Overhead:** The use of `None`, `np.newaxis`, or `...` (Ellipsis) is handled during the initial `__process_key` phase. These simply manipulate the local array dimensions and update the split axis bookkeeping without requiring cross-rank data movement. Because they evaluate as basic components, they hit the zero-overhead Fast Path.

#### 3. Single element indexing
* **Zero Overhead (Non-Split Axis Get/Set):** If the scalar index is applied to any dimension other than the split dimension, the operation evaluates locally on all ranks.
* **Zero Overhead (Split Axis Set):** In `__setitem__`, assigning a local or scalar value to a single index on the split axis identifies a `root` process. Only the `root` process performs the assignment; no broadcast is performed.
* **Low Overhead (Split Axis Get):** In `__getitem__`, if a single element is requested along the split axis, the `root` process extracts the local tensor and uses a single `MPI.Bcast` to share the result with all other ranks.

#### 4. Advanced indexing (integer array and boolean array)
Advanced indexing covers the most complex routing logic, where overhead scales based on the operation and the nature of the value being assigned.
* **Zero to low (local mask assignment):** When assigning a scalar using a boolean mask, the operation evaluates locally with zero MPI overhead. If assigning a 1D non-distributed tensor to an N-D mask, it uses a single `MPI.exscan` to compute sequence offsets.
* **Very low (local integer assignment):** If a local array is assigned using integer arrays, the system performs two `MPI.Allreduce` calls to securely validate index bounds and check for negative indices across ranks. Afterwards, it isolates the assignment using `_advanced_setitem_unordered_local`, avoiding heavy payload exchanges.
* **Low (row-selection optimization):** A dedicated fast path exists for 2D arrays split along axis 0 when indexed by a 1D boolean mask. It avoids full advanced indexing by applying the mask locally.
* **High (non-sequential read):** Standard advanced reads rely on a collective MPI approach. Ranks first execute an `MPI.Allgather` to build a communication matrix. Active ranks then distribute their requested indices via `MPI.Alltoallv`, execute local lookups, and return the resulting data via a second `MPI.Alltoallv` exchange.
* **Very high (distributed assignment):** If `__setitem__` is called with a distributed value, the engine must align the distributions using point-to-point communications (`redistribute_`), use an `Allgather` to construct the communication matrix, and shuffle the payload concurrently via an `MPI.Alltoallv` operation.
