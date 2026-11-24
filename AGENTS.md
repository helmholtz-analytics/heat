# Heat development

## Before writing code
- Read `ARCHITECTURE.md` first.
- **Scale assumption:** Heat is built to process massive arrays whose global size (`A.nbytes`) exceeds the VRAM of a single GPU. Always design algorithms under the assumption that input arrays have **$10^9+$ elements (tens to hundreds of gigabytes)** and cannot fit in the memory of a single rank. Never gather or replicate raw distributed inputs (`resplit_(None)`, `Allgather`, `Gather`) onto all ranks just to call a single-process PyTorch function.

## Environment setup
- Python 3.12+
- MPI: OpenMPI, MPICH, or Intel MPI
- Dependencies: `mpi4py >= 3.1`, `pytorch >= 2.6`
- Create conda environment: `conda env create -f scripts/heat_dev.yml && conda activate heat_dev`
- Create pip environment: `python -m venv heat_venv && source heat_venv/bin/activate`
- Install: `pip install -e '.[dev]'`
- Use `git remote add upstream https://github.com/helmholtz-analytics/heat.git` to add the main repository for synchronizing from a fork.

## Distributed computing rules (mandatory)
- **No gathering of raw distributed inputs:** Per-rank memory allocations must scale with `A.lshape` (the local shard), never `A.shape` (the global input). Gathering (`split=None`) is only acceptable *after* local processing or reduction has shrunk the data to a provably small size (such as scalars, 1D reduction/histogram vectors, small sketches, or small `k x k` factors).
- **PyTorch as local kernel only:** Use PyTorch (`torch.*`) on `DNDarray.larray` shards inside a proper distributed algorithm (local ops, `Allreduce`/`Exscan`, point-to-point ring exchanges, tree reductions, or block algorithms). Do not reimplement low-level kernels from scratch.
- **Single split axis & batched dimensions:** Heat uses 1D slab partitioning—a `DNDarray` can be split along **at most one axis** (`split` is a single `int` or `None`, never a tuple of axes or a 2D process grid). Support all valid single `split` axes (`None`, `0`, `1`, negative axes, and batch axes when `A.ndim > 2`), preserving the input `split` layout in the output whenever mathematically possible.
- **Handle unbalanced and empty shards:** Do not assume `A.lshape` is equal across ranks or non-empty (`0 in A.lshape` occurs when `A.shape[A.split] < A.comm.size`).
- **MPI resource hygiene:** Explicitly call `.Free()` on any temporary MPI communicators or `MPI.Group` handles created during execution.
- **Deep learning over MPI, not `torch.distributed`:** Do not initialize `torch.distributed` process groups or use PyTorch `DDP`. Use Heat's MPI-backed `ht.nn.DataParallel`, `ht.optim.DataParallelOptimizer`, and `ht.utils.data.DistributedDataLoader` / `DistributedDataset`, and always iterate `DistributedDataLoader` inside `with iter(loader) as batches:` so pending non-blocking `Itake` requests are drained on all ranks.

## Code style and quality
- Use `pre-commit install` to enforce coding standards.
- Python: Follow PEP 8 and use type hints (mypy strict mode).
- Formatting: Ruff (line length 100).
- Docstyle: NumPy convention (see Heat docstring standard).
- Imports: Use `import heat as ht` and `import numpy as np`.
- Incremental changes: When addressing PR review feedback, make targeted edits rather than rewriting entire functions from scratch.

## Testing instructions
- Use `mpirun -n <N> pytest tests/` to run the unit tests where `<N>` is the number of MPI processes. Test with 1, 2, 3, and 5 processes (3 and 5 exercise unbalanced chunks and empty shards).
- If GPUs are available, also test with `HEAT_TEST_USE_DEVICE="gpu" mpirun -n <N> pytest`.
- Every unit test for a distributed operation must verify numerical results against NumPy/SciPy/PyTorch **and** assert output `split` and `lshape` across ranks.
- Do not install missing dependencies and do not edit any files by yourself. First check whether an environment is active. If not, load the environment and try the tests again. Otherwise ask for permission.

## Project structure
- `heat/classification` - Classification models like kneighbours
- `heat/cluster` - Clustering models like kmeans, kmedians, kmediods, and spectral clustering
- `heat/core` - DNDarray class and numeric functions
- `heat/core/linalg` - Linear algebra functions
- `heat/datasets` - Datasets used for testing
- `heat/decomposition` - Matrix decomposition like DMD and PCA
- `heat/fft` - Discrete Fast Fourier Transforms
- `heat/graph` - Graph-based classes like graph Laplacian
- `heat/naive_bayes` - Naive-Bayes classifier
- `heat/nn` - Neural Network classes
- `heat/optim` - Optimization algorithms
- `heat/preprocessing` - Data preprocessing techniques
- `heat/regression` - Regression techniques
- `heat/sparse` - Sparse arrays
- `heat/spatial` - Distance functions
- `heat/testing` - Class for testing infrastructure
- `heat/utils` - Helper functions and distributed deep learning data tools (`heat/utils/data`)
- `tests` - Unit tests with the same structure as `heat/`

## Git and PR instructions
- General guidelines in `CONTRIBUTING.md`.
- Sign off all commits with DCO (`git commit -s -m "..."`).
- Title format: `[component] Brief description`.
- Fill out the template `.github/PULL_REQUEST_TEMPLATE.md` for the description. If a part is unsure, ask the user. If an operation cannot be implemented without gathering the raw global input array onto every rank, stop and discuss it with maintainers instead of submitting a gather-based workaround.
- Write `AI Support 🦾 (<exact model and version>)` at the end of the description after a blank line, stating the exact model and version that was used.
