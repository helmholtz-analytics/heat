# Welcome to the Heat Tutorials

Heat builds on PyTorch and mpi4py to provide a high-performance computing infrastructure for memory-intensive applications within the NumPy/SciPy ecosystem. If you are looking to scale single-CPU code to a multi-node cluster to exploit its cumulative RAM for massive datasets, these interactive notebooks will guide you through the core concepts.

## Environment Setup

Before diving into the tutorials, you need to set up your Python environment and start an IPython cluster. We provide scripts and guides for both local machines and High-Performance Computing (HPC) environments.

**Local Setup (Laptop/Workstation):**
1. Run either [`setup_pip.sh`](https://github.com/helmholtz-analytics/heat/blob/main/doc/source/tutorials/notebooks/setup/setup_pip.sh) or [`setup_conda.sh`](https://github.com/helmholtz-analytics/heat/blob/main/doc/source/tutorials/notebooks/setup/setup_conda.sh) to install OpenMPI, create a virtual environment, install Heat and its dependencies (like `ipyparallel`), and launch Jupyter.
2. Open and run [`setup_local.ipynb`](https://github.com/helmholtz-analytics/heat/blob/main/doc/source/tutorials/notebooks/setup/setup_local.ipynb) to verify your IPyParallel cluster is running correctly.

**HPC Setup (SLURM / JSC):**
* If you are running on Jupyter-JSC at the Jülich Supercomputing Centre, start with [`setup_jsc.ipynb`](https://github.com/helmholtz-analytics/heat/blob/main/doc/source/tutorials/notebooks/setup/setup_jsc.ipynb).

---

## Training materials

Once your cluster is running, we recommend following the tutorials in this order:

### The Fundamentals
* **[Basics.ipynb](Basics.ipynb)**: Start here! Learn how to create `DNDarrays`, understand the `split` parameter, and perform basic operations across multiple processes.
* **[Internals.ipynb](Internals.ipynb)**: A look behind the curtain for potential contributors and power users. Understand how Heat manages local shapes (`lshape`) and learn how to use `redistribute_` to manually balance data across your cluster.

### Mathematics & Machine Learning
See Heat in action on complex mathematical operations and large-scale datasets.
* **[Loading_preprocessing.ipynb](Loading_preprocessing.ipynb)**: Learn how to parallel-load and preprocess large datasets using Heat's distributed data structures and operations.
* **[Linear_algebra.ipynb](Linear_algebra.ipynb)**: Explore matrix-matrix multiplications, Randomized SVD, and Hierarchical SVD.
* **[Clustering_and_PCA.ipynb](Clustering_and_PCA.ipynb)**: Learn how to scale K-Means clustering and Principal Component Analysis (PCA) using a real-world dataset of ~1.4 million asteroids from the JPL Small Body Database.
* **[DMD.ipynb](DMD.ipynb)**: Dive into Dynamic Mode Decomposition (DMD) using roughly a year of 6-hourly global windspeed data from ERA5.

### Profiling & Optimization
* **[Profiling_with_perun.ipynb](Profiling_with_perun.ipynb)**: Learn how to track performance and measure the energy consumption (power draw) of your distributed applications using the `perun` library and its `@monitor` decorators.

### Distributed Deep Learning
* Finally, the **distributed training scripts** from our Nov. 2025 training are available in the [examples/tutorial2025_dl](https://github.com/helmholtz-analytics/heat/tree/main/examples/tutorial2025_dl) directory.

---

## Problems?

We rely on your feedback!

- Did you encounter any issues?
- Do you want to request a tutorial on a specific topic?
- Do you have suggestions for improving the existing notebooks?

Please open an issue in our GitHub repository: [https://github.com/helmholtz-analytics/heat/issues](https://github.com/helmholtz-analytics/heat/issues). Thanks!
