# Heat Development

## Environment Setup
- Python 3.11+
- MPI: OpenMPI, MPICH, or Intel MP
- Dependencies: mpi4py >= 3.1, pytorch >= 2.4
- create conda environment: `conda env create -f scripts/heat_dev.yml && conda activate heat_dev`
- create pip environment: `python -m venv heat_venv && source heat_venv/bin/activate`
- install: `pip install -e '.[dev]`
- Use `git remote add upstream https://github.com/helmholtz-analytics/heat.git` to add the main repository for synchronizing from a fork.

## Code Style
- Use `pre-commit install` to enforce coding standard
- Python: Follow PEP 8, use type hints (mypy strict mode)
- Formatting: Ruff (line length 100)
- Docstyle: NumPy convention

## Code Quality
- Prefer PyTorch over implementation from scratch.
- Add unit tests for your changes.
- Use `import heat as ht` when importing the library.

## Testing instructions
- Use `mpirun -n <N> pytest tests/` to run the unit tests where <N> is the number of MPI processes. Test with 1, 2, 3 and 4 processes.
- Do not install missing dependencies and do not edit any files by yourself. First check whether an environment is active. If not load the environment and try the tests again. Otherwise ask for permission.

## Project Structure
- `heat/array_api` - Additional module following Python array API standard. Only look when asked for the module explicitly.
- `heat/classification` - Classification models like kneighbours
- `heat/cluster` - Clustering models like kmeans, kmedians, kmediods, and spectral clustering
- `heat/core` - DNDarray class and numeric functions
- `heat/core/linalg` - Linear algebra functions
- `heat/datasets`- Datasets used for testing
- `heat/decomposition`- Matrix decomposition like DMD and PCA
- `heat/fft` - Discrete Fast Fourier Transforms
- `heat/graph` - Graph-based classes like graph Laplacian
- `heat/naive_bayes` - Naive-Bayes classifier
- `heat/nn` - Neural Network classes
- `heat/optim` - Optimization algorithms
- `heat/preprocessing` - Data preprocessing techniques
- `heat/regression` - Regression techniques
- `heat/sparse` - Sparse arrays
- `heat/spatial`- Distance functions
- `heat/testing` - Class for testing setup
- `heat/utils` - Helper functions
- `testing`- Unit tests with the same structure as `heat/`

## PR Instructions
- Title format: [component] Brief description
- Fill out the template `.github/PULL_REQUEST_TEMPLATE.md` for the description. If a part is unsure, ask the user.
- Write `AI Support 🦾` at the end of the description after a newline
