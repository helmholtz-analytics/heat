# Heat architecture guide

Heat distributes NumPy-, scikit-learn-, and PyTorch-like scientific computing and machine learning pipelines across multi-CPU and multi-GPU systems. Built on PyTorch and MPI (`mpi4py`), every process runs the same Python program in a Single Program, Multiple Data (SPMD) model—executing PyTorch kernels on local tensor shards while coordinating across ranks via MPI.

## Core design rule

Heat is built to process massive datasets whose global memory footprint (`A.nbytes`) exceeds the VRAM of a single GPU. Always design algorithms under the assumption that input arrays have **$10^9+$ elements (tens to hundreds of gigabytes)** and cannot fit in the memory of a single rank.

**Never implement a distributed algorithm by gathering or replicating the raw distributed input operands onto every rank (e.g., via `resplit_(None)`, `Allgather`, or `Gather`) just to call a single-process PyTorch function.**

- **Memory scaling:** Per-rank memory allocations must scale with the local shard size (`O(prod(A.lshape))`), not the global input size (`O(prod(A.shape))`).
- **When gathering is acceptable:** Gathering or replicating (`split=None`) is valid *after* local processing, reduction, or sketching has reduced the data to a provably small size independent of the large distributed dimension (for example, scalars, 1D histograms/bincounts, small sketches, or small `(k, k)` factors such as `R` in tall-skinny QR).
- **Role of PyTorch:** PyTorch (`torch.*`) is used as the single-process compute kernel on local `torch.Tensor` shards (`DNDarray.larray`), never as a shortcut on gathered global inputs.
- **Role of MPI:** MPI (`DNDarray.comm`) exchanges only the minimal data required between shards (reduced partials, halo slices, panel blocks, or ring-passed chunks).

---

## The `DNDarray` data model (`heat/core/dndarray.py`)

A `DNDarray` combines a local `torch.Tensor` with global distribution metadata:

| Attribute / property | Meaning |
| :--- | :--- |
| `larray` | Process-local `torch.Tensor` holding this rank's shard. |
| `shape` / `gshape` | Global tuple shape across all MPI ranks. |
| `lshape` | Local tuple shape (`tuple(larray.shape)`) on the calling rank. |
| `split` | `int` axis along which the array is partitioned across ranks, or `None` if replicated on every rank. |
| `comm` | `Communication` instance (`MPICommunication`, defaulting to `MPI_WORLD`). |
| `device` | `ht.Device` (`cpu` or `gpu`, covering CUDA and ROCm). |
| `dtype` | Canonical Heat `datatype` (`ht.float32`, `ht.float64`, `ht.int64`, `ht.complex64`, etc.). |
| `balanced` | `bool` or `None` indicating whether chunks follow `comm.chunk(gshape, split)` evenly. |
| `lshape_map` | `(comm.size, ndim)` `torch.Tensor` mapping each rank to its `lshape`. |

### Critical invariants

1. **Single split axis (1D decomposition only):** A `DNDarray` is partitioned across ranks along **at most one dimension** (`split` is a single `int` or `None`; multi-axis or 2D block-cyclic process-grid distributions are not supported). Every non-split dimension is completely local on every rank (`A.lshape[i] == A.shape[i]` for all `i != A.split`).
2. **Unbalanced and empty shards:** When `gshape[split]` is not divisible by `comm.size` (or when `gshape[split] < comm.size`), ranks have different `lshape[split]`, and trailing ranks can hold **0 elements** (`0 in A.lshape`). Never assume `A.lshape` is identical across ranks without checking `A.is_balanced()`.
3. **Global vs. local indexing:** Slices and indices computed from `comm.chunk(...)` or `A.counts_displs()` are in **global** coordinates (`0 .. gshape[split]`). Subtract the rank's starting offset before indexing `A.larray`.
4. **Batched dimensions:** Matrix operations in `heat.core.linalg` must support leading batch dimensions (`A.ndim > 2`). When `A.split < A.ndim - 2`, the matrix dimensions `(M, N)` are completely local on every rank and no MPI communication is needed.
5. **Constructing outputs:**
   - Use `DNDarray(local_tensor, gshape=..., dtype=..., split=..., device=..., comm=..., balanced=...)` when global shape, split axis, and balance state are already known locally (zero MPI overhead).
   - Use `factories.array(local_tensor, is_split=axis, device=..., comm=...)` when each rank holds a local shard and `gshape` / `balanced` must be inferred across `comm`.
   - Beware of `factories.zeros_like(a)` / `empty_like(a)`: they always allocate a **balanced** `DNDarray`. If `a.balanced` is `False`, `zeros_like(a).lshape` will differ from `a.lshape`.

---

## Communication layer (`heat/core/communication.py`)

`MPICommunication` wraps an `mpi4py` communicator and handles PyTorch interoperability automatically:

- **Direct tensor buffers:** Pass `DNDarray` or `torch.Tensor` directly to `Bcast`, `Ibcast`, `Allreduce`, `Iallreduce`, `Reduce`, `Scan`, `Exscan`, `Gather`, `Gatherv`, `Allgather`, `Allgatherv`, `Scatter`, `Scatterv`, `Alltoall`, `Alltoallv`, `Send`, `Isend`, `Recv`, and `Irecv`.
- **Non-contiguous tensors and GPU staging:** `MPICommunication` automatically builds derived MPI vector datatypes for strided/permuted tensors and stages CUDA/ROCm tensors through host memory when `GPU_AWARE_MPI` is `False` (on `Wait()` for non-blocking `MPIRequest` calls).
- **Axis-aware collectives:** Collectives such as `Allgatherv(..., recv_axis=...)`, `Gatherv(..., axis=...)`, `Scatterv(..., axis=...)`, and `Alltoallv(..., send_axis=..., recv_axis=...)` pack and unpack along arbitrary axes.
- **Communicator and group cleanup:** Whenever creating temporary sub-communicators or groups (`comm.Split`, `comm.Create_group`, `comm.group.Incl`), explicitly call `.Free()` on them before returning so iterative algorithms do not exhaust MPI communicator handles.

---

## Canonical distributed patterns

Match new implementations to the closest existing pattern in the codebase:

1. **Elementwise local operations (`heat/core/_operations.py` -> `__local_op`):**
   - Apply the PyTorch kernel directly to `x.larray` with zero MPI communication and preserve `gshape`, `split`, and `balanced`.
2. **Elementwise binary operations (`heat/core/_operations.py` -> `__binary_op`):**
   - Validate and align operand distributions with `sanitation.sanitize_distribution` (which redistributes out-of-place only when `split` axes match but `lshape_map`s differ), then execute the PyTorch kernel on `t1.larray` and `t2.larray`.
3. **Reductions and scans (`heat/core/_operations.py` -> `__reduce_op`, `__cum_op`; `heat/core/statistics.py`):**
   - **Reduction axis != `x.split`:** Reduce `x.larray` locally with PyTorch; no MPI communication occurs, and `output_split` shifts down by 1 for each reduced axis `< x.split` (unless `keepdims=True`).
   - **Reduction axis == `x.split` or `axis is None`:** Supply the operation's neutral element when `0 in x.lshape`, compute the local partial reduction on `x.larray`, and combine partials via `x.comm.Allreduce` (or `x.comm.Exscan` for cumulative operations), returning `split=None`.
4. **All-pairs and matrix multiplication (`heat/core/linalg/basics.py` -> `matmul`, `outer`; `heat/spatial/distance.py`):**
   - Keep one operand stationary and circulate shards of the other using point-to-point ring exchanges (`Send`/`Recv`) or pipelined non-blocking broadcasts (`Ibcast` with a small prefetch window), accumulating into a distributed output `DNDarray`.
5. **Block matrix decompositions (`heat/core/linalg/qr.py`):**
   - For tall-skinny row-split matrices (`split = -2`), compute local QR on `A.larray`, reduce small `(N, N)` upper-triangular `R` factors along a communication tree (`Gatherv`/`Scatterv`), accumulate orthogonal tree factors, and apply a single local matrix multiplication at the end.
   - For column-split (`split = -1`) or non-tall-skinny row-split matrices, use column-block Gram-Schmidt with broadcast/allreduce of panel blocks.

---

## Distributed machine learning algorithms (scikit-learn API)

Heat provides distributed classical machine learning estimators, transformers, and graph builders in `heat/cluster`, `heat/classification`, `heat/regression`, `heat/naive_bayes`, `heat/decomposition`, `heat/preprocessing`, and `heat/graph`. Estimators inherit from `ht.BaseEstimator` and mixins (`ClusterMixin`, `ClassificationMixin`, `RegressorMixin`, `TransformerMixin`), mirroring scikit-learn's `fit`, `predict`, `transform`, `fit_predict`, and `fit_transform` interfaces on 2D `(n_samples, n_features)` `DNDarray` inputs (typically `split=0` or `split=None`).

- **No `scikit-learn` wrappers on gathered data:** Never implement an estimator or transformer by calling `resplit_(None)` on `X` or `y` and delegating to `sklearn`.
- **Compose distributed Heat primitives:** Build ML algorithms primarily out of Heat's distributed tensor and linear algebra operations (`ht.sum`, `ht.mean`, `ht.matmul`, `ht.spatial.distance.cdist` / `rbf`, `ht.linalg.qr`, `ht.linalg.reigh`, `ht.lanczos`) so MPI communication is handled automatically under the hood, dropping down to local `larray` kernels and `X.comm` collectives only when no higher-level distributed primitive fits.
- **Distribution of fitted attributes and outputs:**
  - **Model parameters:** Compact learned parameters (such as `cluster_centers_` in `KMeans`, class statistics, scaler parameters, or linear weights) may be stored as small `DNDarray`s. Large learned representations (such as spectral `eigenvectors` or principal components) must remain distributed.
  - **Sample-wise outputs:** `labels_`, `predict(X)`, and `transform(X)` must preserve the sample distribution (`split=0`) of `X` without gathering the full output onto every rank.

---

## Distributed deep learning and data loading

Heat bridges distributed `DNDarray` datasets with standard single-process PyTorch models (`torch.nn.Module`) and optimizers (`torch.optim.Optimizer`) over MPI, without initializing `torch.distributed` process groups:

- **Dataset splitting (`heat/utils/data/datatools.py` -> `create_train_val_split`):** Shuffles aligned `DNDarray`s along `sample_axis` using a shared permutation seed (broadcast from rank 0 when `seed=None`) and extracts splits via `take`, preserving each array's `split` axis, `dtype`, `device`, and `comm`.
- **Distributed batch loading (`heat/utils/data/datatools.py` -> `DistributedDataLoader`):** Partitions a shared global sample order (`seed + epoch`) across ranks, retrieves complete samples asynchronously via non-blocking `Itake` (`TakeRequest`) calls (redistributing feature-split batches along axis 0 when needed), and yields local `torch.Tensor` batches on `device`.
  - When `drop_last=False`, global indices are padded so every rank executes the same number of batches.
  - Because iterators hold a prefetch queue of pending `Itake` requests, always iterate inside a context manager (`with iter(loader) as batches:`) or call `.close()` consistently across all ranks when stopping early.
- **PyTorch `DataLoader` interoperability (`heat/utils/data/datatools.py` -> `DistributedDataset`):** Wraps one or more `DNDarray`s as a map-style dataset for `torch.utils.data.DataLoader` (must be used with `num_workers=0` and an identity `collate_fn`), gathering requested batch indices via `Allgather` and assembling local batches via `take`.
- **Data-parallel model and optimizer wrappers (`heat/nn`, `heat/optim`):** `ht.nn.DataParallel` and `ht.optim.DataParallelOptimizer` replicate model parameters across `comm` and synchronize/average gradients over MPI during `backward()` and `step()`. Every rank must execute the same forward, backward, and optimizer step sequence; never guard training steps behind `if comm.rank == 0:`.
- **Distributed evaluation:** For aligned `split=0` validation or test arrays, run inference directly on `DNDarray.larray` under `torch.no_grad()` and aggregate local metrics (such as correct prediction counts and sample counts) via `comm.allreduce(...)`.

---

## Validation, device, and testing requirements

- **Sanitation helpers (`heat/core/sanitation.py`, `heat/core/stride_tricks.py`):** Always validate inputs with `sanitize_in`, normalize axes with `sanitize_axis`, and validate `out=` buffers with `sanitize_out`.
- **Device portability:** Never hardcode `"cpu"` or `"cuda"`; always allocate tensors on `x.device.torch_device`. Note that Apple MPS does not support `float64` or `complex128` (downcast to `float32` / `complex64` when `larray.is_mps`).
- **Parallel verification:** Unit tests must run across multiple MPI process counts (including prime counts such as 3 and 5 to exercise unbalanced shards and empty ranks) and must assert **both** numerical correctness against NumPy/SciPy/PyTorch **and** the output's `split` attribute and `lshape`.
- **In-place naming and `out=` buffers:** Methods that mutate an array in place must end with a trailing underscore (`resplit_`, `balance_`, `collect_`). Functions accepting an `out=` buffer must validate its shape, `split`, device, and `comm` with `sanitation.sanitize_out(...)`.
- **`DNDarray` method bindings:** Core array operations (reductions, norms, linear algebra routines) are typically also bound as methods or properties on `DNDarray` (e.g., `DNDarray.mean`, `DNDarray.mT`) at the bottom of their defining module.
