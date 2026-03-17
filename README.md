<div align="center">
  <img src="https://raw.githubusercontent.com/helmholtz-analytics/heat/main/doc/source/_static/images/logo.png">
</div>

---

# Project Status

[![CPU/CUDA/ROCm tests](https://codebase.helmholtz.cloud/helmholtz-analytics/ci/badges/heat/base/pipeline.svg)](https://codebase.helmholtz.cloud/helmholtz-analytics/ci/-/commits/heat/base)
[![Documentation Status](https://readthedocs.org/projects/heat/badge/?version=latest)](https://heat.readthedocs.io/en/latest/?badge=latest)
[![coverage](https://codecov.io/gh/helmholtz-analytics/heat/branch/main/graph/badge.svg)](https://codecov.io/gh/helmholtz-analytics/heat)
[![license: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI Version](https://img.shields.io/pypi/v/heat)](https://pypi.org/project/heat/)
[![Downloads](https://pepy.tech/badge/heat)](https://pepy.tech/project/heat)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/heat.svg)](https://anaconda.org/channels/conda-forge/packages/heat/overview)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/helmholtz-analytics/heat/badge)](https://securityscorecards.dev/viewer/?uri=github.com/helmholtz-analytics/heat)
[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/7688/badge)](https://bestpractices.coreinfrastructure.org/projects/7688)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2531472.svg)](https://doi.org/10.5281/zenodo.2531472)
[![Benchmarks](https://img.shields.io/badge/Grafana-Benchmarks-2ea44f)](https://930000e0-e69a-4939-912e-89a92316b420.ka.bw-cloud-instance.org/grafana)

# Heat
**High-performance data analytics in Python, at scale.**

[Getting Started](quick_start.md) | [Tutorials](https://github.com/helmholtz-analytics/heat/tree/main/doc/source/tutorials/notebooks) | [Docs](https://heat.readthedocs.io/) | [Contributing](contributing.md)

---

### Why Heat?
Heat is a distributed tensor framework built on **PyTorch** and **mpi4py**. It provides highly optimized algorithms and data structures for tensor computations using CPUs, GPUs (CUDA/ROCm), and distributed cluster systems. It is designed to handle **massive arrays** that exceed the memory and computational limits of a single machine.

* **Seamless integration:** Port existing NumPy/SciPy code to multi-node clusters with minimal effort.
* **Hardware-agnostic:** Supports CPUs and GPUs (CUDA, ROCm, Apple MPS).
* **Efficient scaling:** Exploit the entire, cumulative RAM of your cluster for memory-intensive operations.

### Requirements
* **Python:** >= 3.11
* **MPI:** OpenMPI, MPICH, or Intel MPI
* **Frameworks:** mpi4py >= 3.1, pytorch >= 2.3

### Installation
```bash
# Via pip (with optional I/O support)
pip install heat[hdf5,netcdf,zarr]

# Via conda-forge
conda install -c conda-forge heat

# Via easybuild (for HPC systems)
eb heat-<version>.eb --robot

# Via spack (for HPC systems)
spack install py-heat
```

### Distributed Example
Heat handles inter-node communication automatically. Define how your data is partitioned across the cluster using the [`DNDarray.split`](https://heat.readthedocs.io/en/stable/autoapi/heat/core/dndarray/index.html) attribute. Push computations to your GPUs with the [`DNDarray.device`](https://heat.readthedocs.io/en/stable/autoapi/heat/core/dndarray/index.html) attribute. Heat will take care of the rest, ensuring efficient data movement and synchronization across nodes.

Here an example from our [Linear Algebra tutorial](https://github.com/helmholtz-analytics/heat/tree/main/doc/source/tutorials/notebooks/Linear_Algebra.ipynb):

<details>
<summary><b>View Distributed Example (mpirun / srun)</b></summary>

**1. Create your script (`my_script.py`):**

```python
import heat as ht

split_A=0
split_B=1
M = 10000
N = 10000
K = 10000
A = ht.random.randn(M, N, split=split_A, device="gpu")
B = ht.random.randn(N, K, split=split_B, device="gpu")
C = ht.matmul(A, B)
print(C)

```
**2. Run with MPI:**

On your laptop, e.g. with OpenMPI:
```bash
mpirun -np 4 python my_script.py
```
On an HPC cluster with SLURM:
```bash
srun --nodes=2 --ntasks-per-node=2 --gpus-per-node=2 python my_script.py
```


</details>


### Contributing
We welcome contributions from the community. Please see our [Contribution Guidelines](contributing.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).

For bug reports, feature requests, or general questions, please use [GitHub Issues](https://github.com/helmholtz-analytics/heat/issues) or [Discussions](https://github.com/helmholtz-analytics/heat/discussions).

### Citations
Citations are essential for the sustainability of this project. If Heat supports your work, please cite our main paper:

Götz, M., et al. (2020). HeAT - a Distributed and GPU-accelerated Tensor Framework for Data Analytics. In *2020 IEEE International Conference on Big Data (Big Data)* (pp. 276-287). IEEE. DOI: 10.1109/BigData50022.2020.9378050.

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{heatBigData2020,
  author={Götz, Markus and Debus, Charlotte and Coquelin, Daniel and Krajsek, Kai and Comito, Claudia and Knechtges, Philipp and Hagemeier, Björn and Tarnawa, Michael and Hanselmann, Simon and Siggel, Martin and Basermann, Achim and Streit, Achim},
  booktitle={2020 IEEE International Conference on Big Data (Big Data)},
  title={HeAT – a Distributed and GPU-accelerated Tensor Framework for Data Analytics},
  year={2020},
  volume={},
  number={},
  pages={276-287},
  keywords={Heating systems;Industries;Data analysis;Big Data;Parallel processing;Libraries;Arrays;HeAT;Tensor Framework;High-performance Computing;PyTorch;NumPy;Message Passing Interface;GPU;Big Data Analytics;Machine Learning;Dask;Model Parallelism;Parallel Application Frameworks},
  doi={10.1109/BigData50022.2020.9378050}}
```
</details>

### Acknowledgments
Support for this work was provided by the **Helmholtz Association Initiative and Networking Fund** (Project **ZT-I-0003**), the **Helmholtz AI** platform grant, the **European Space Agency (ESA)** (Programme [4000144045](https://activities.esa.int/index.php/4000144045)), the **Helmholtz Impulse and Networking Fund** (Project [HeatHub](https://hifis.net/announcement/2026/01/08/scienceserve-awardees/)).

### License
Heat is distributed under the **MIT license**. See the [LICENSE](LICENSE) file for details.
