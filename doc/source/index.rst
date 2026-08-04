.. title:: Heat

.. meta::
   :description: Scale NumPy-based data analysis to HPC
   :keywords: data analysis, HPC, MPI, GPU, multi-GPU, distributed computing, parallel processing, data science, data analytics, machine learning, scientific computing, high-performance computing, Python libraries, NumPy API, PyTorch


.. div:: text-center mt-5 mb-5

   .. image:: _static/images/logo.png
      :alt: Heat Framework Logo
      :align: center
      :width: 70%

.. div:: fs-5 text-center mt-4 mb-5

   **High-performance data analytics in Python, at scale.**

.. grid:: 1 2 4 4
   :gutter: 3
   :class-container: text-center mb-5

   .. grid-item::
      .. div:: fs-5 fw-bold  mb-2

         Distributed

      .. div:: small text-muted

         Multi-node data-parallel processing via optimized MPI communication.

   .. grid-item::
      .. div:: fs-5 fw-bold  mb-2

         Accelerated

      .. div:: small text-muted

         Native, out-of-the-box, multi-GPU hardware acceleration via PyTorch.

   .. grid-item::
      .. div:: fs-5 fw-bold  mb-2

         Scalable

      .. div:: small text-muted

         Scale effortlessly beyond single-node RAM limits.

   .. grid-item::
      .. div:: fs-5 fw-bold  mb-2

         Interoperable

      .. div:: small text-muted

         Plug & play compatibility with the Python array ecosystem.

-----

In a nutshell
===============

.. grid:: 1 1 2 2
   :gutter: 4
   :class-container: mt-4 mb-4

   .. grid-item::
      :columns: 12 12 5 5

      Heat builds on **PyTorch** and **mpi4py** to process **massive arrays** - huge collections of images, high-dimensional climate simulation grids, or massive machine learning feature matrices - that exceed the memory and computational limits of a single machine.

      Define your data distribution axis via the ``split`` parameter, assign hardware using the ``device`` attribute, and let Heat orchestrate the parallel computation.

      **Prototype locally, execute on any cluster.**

   .. grid-item::
      :columns: 12 12 7 7

      .. code-block:: python
         :caption: my_script.py

         import heat as ht

         # Distributed random matrix generation
         A = ht.random.randn(40000, 10000, split=0, device="gpu")
         B = ht.random.randn(10000, 40000, split=1, device="gpu")

         # Multi-GPU-accelerated matrix multiplication
         C = ht.matmul(A, B)

      .. code-block:: bash
         :caption: Run locally or scale across cluster nodes via MPI

         mpirun -np 4 python my_script.py

-----

Getting started
===============

.. grid:: 1 1 2 2
   :gutter: 4
   :class-container: mt-4 mb-5

   .. grid-item::
      :columns: 12 12 6 6

      **Quick Install**

      .. tab-set::

         .. tab-item:: pip

            .. code-block:: bash

               pip install heat

         .. tab-item:: conda

            .. code-block:: bash

               conda install -c conda-forge heat

   .. grid-item::
      :columns: 12 12 6 6

      **HPC & multi-GPU deployments**

      For Spack, EasyBuild, and containerized setups, refer to our comprehensive deployment guide.

      .. button-link:: /quick_start.html
         :color: primary
         :class: sd-btn-primary sd-btn-block mt-3

         View full installation guide

-----

Latest news
===========

.. container:: frontpage-news

   .. include:: _timeline.rst

.. button-ref:: news
   :color: primary
   :outline:
   :class: mt-3

   View full news history

-----

Tutorials & courses
===================

.. grid:: 1 2 2 2
   :gutter: 3
   :class-container: text-center mt-4 mb-4

   .. grid-item-card::
      :class-card: sd-card
      :link: https://hub.nfdi-jupyter.de/v2/gh/helmholtz-analytics/heat/jupyter4nfdi?labpath=tutorials%2FJupyter4NFDI_landing_notebook.ipynb&system=deNBI-Cloud&flavor=l1&localstoragepath=%2Fhome%2Fjovyan%2Fwork

      .. image:: _static/images/jupyter.png
         :alt: Jupyter4NFDI Interactive Course
         :align: center
         :height: 140px

      .. div:: mt-3 **Take the Course in the Cloud**

      .. div:: text-muted small mt-1

         Run our tutorials on |j4nfdi_badge|

      .. |j4nfdi_badge| image:: https://nfdi-jupyter.de/images/jupyter4nfdi_badge.svg
         :alt: Jupyter4NFDI
         :height: 22px

   .. grid-item-card::
      :class-card: sd-card
      :link: /tutorials/notebooks/README
      :link-type: doc

      .. image:: _static/images/tutorial_split_dndarray.svg
         :alt: Download Course
         :align: center
         :height: 140px

      .. div:: mt-3 **Run the Course Locally**

      .. div:: text-muted small mt-1

         Download the full suite of interactive Jupyter notebooks to run on your own hardware or cluster.

   .. grid-item-card::
      :class-card: sd-card
      :link: /tutorials/tutorial_30_minutes
      :link-type: doc

      .. image:: _static/images/logo_emblem.svg
         :alt: welcome tutorial
         :align: center
         :height: 140px

      .. div:: mt-3 **30-minute tutorial**

      .. div:: text-muted small mt-1

         **A 30-minute welcome to Heat:** DNDarrays and basic operations.

   .. grid-item-card::
      :class-card: sd-card
      :link: /tutorials/tutorial_parallel_computation
      :link-type: doc

      .. image:: _static/images/logo_emblem.svg
         :alt: Parallel computation
         :align: center
         :height: 140px

      .. div:: mt-3 **Parallel computation**

      .. div:: text-muted small mt-1

         **Parallel computing:** distributed MPI computation and (multi-)GPU acceleration.


-----

How-to guides
=============

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: text-center mt-4 mb-4

   .. grid-item-card::
      :class-card: sd-card
      :link: /tutorials/notebooks/Loading_preprocessing
      :link-type: doc

      .. image:: _static/images/tutorial_split_dndarray.svg
         :alt: Parallel I/O & Preprocessing Example
         :align: center
         :height: 140px

      .. div:: mt-3 **Parallel I/O & Preprocessing**

      .. div:: text-muted small mt-1

         **Parallel I/O:** ingest HDF5, Zarr, and NetCDF formats directly into distributed memory.


   .. grid-item-card::
      :class-card: sd-card
      :link: /tutorials/tutorial_clustering
      :link-type: doc

      .. image:: _static/images/tutorial_clustering.svg
         :alt: Clustering
         :align: center
         :height: 140px

      .. div:: mt-3 **Clustering**

      .. div:: text-muted small mt-1

         **Clustering analysis:** Automatically identify groups of similar data points in massive distributed datasets via unsupervised clustering methods.

   .. grid-item-card::
      :class-card: sd-card
      :link: /tutorials/notebooks/Linear_algebra
      :link-type: doc

      .. image:: _static/images/hSVD_bench_rank5.png
         :alt: Linear algebra
         :align: center
         :height: 140px

      .. div:: mt-3 **Linear algebra**

      .. div:: text-muted small mt-1

         **Linear algebra:** Matrix-matrix multiplications, Singular Value Decomposition across multi-GPU.

   .. grid-item-card::
      :class-card: sd-card
      :link: /tutorials/notebooks/Clustering_and_PCA
      :link-type: doc

      .. image::
         :alt: clustering and PCA
         :align: center
         :height: 140px

      .. div:: mt-3 **Clustering and PCA**

      .. div:: text-muted small mt-1

         **Dimensionality reduction:** Distributed clustering algorithms and Principal Component Analysis multi-node.

   .. grid-item-card::
      :class-card: sd-card
      :link: /tutorials/notebooks/Profiling_with_perun
      :link-type: doc

      .. image:: _static/images/perun_logo.svg
         :alt: Performance Profiling Example
         :align: center
         :height: 140px

      .. div:: mt-3 **Performance Profiling**

      .. div:: text-muted small mt-1

         **Profiling:** Track cluster memory consumption, execution efficiency, and resource utilization using Perun.

-----

Reference material
==================

.. grid:: 1 2 2 2
   :gutter: 3
   :class-container: text-center mt-4 mb-4

   .. grid-item-card::
      :class-card: sd-card
      :link: /coverage_tables
      :link-type: doc

      .. image:: _static/images/tutorial_split_dndarray.svg
         :alt: Installation
         :align: center
         :height: 140px

      .. div:: mt-3 **NumPy API**

      .. div:: text-muted mt-1

         **Numerical data processing**: NumPy/SciPy API compatibility tracking

   .. grid-item-card::
      :class-card: sd-card
      :link: /autoapi/index
      :link-type: doc

      .. image:: _static/images/api_ref_graphics.png
         :alt: API reference
         :align: center
         :height: 140px

      .. div:: mt-3 **API reference**

      .. div:: text-muted mt-1

         **API reference:**  all numerical functions and machine learning algorithms.


-----

Get in touch
============

.. grid:: 1 1 3 3
   :gutter: 4
   :class-container: text-center mt-4 mb-5

   .. grid-item::

      .. raw:: html

         <a href="https://github.com/helmholtz-analytics/heat/discussions" style="text-decoration: none;">
            <i class="fa-solid fa-comments fa-4x mb-3" style="color: var(--pst-color-primary);"></i>
         </a>

      **GitHub Discussions**

   .. grid-item::

      .. raw:: html

         <a href="https://matrix.to/#/#heat:helmholtz.cloud" style="text-decoration: none;">
            <img src="_static/images/matrix-icon.svg" height="64px" alt="Matrix Space" class="mb-3" />
         </a>

      **Heat Matrix Space**

   .. grid-item::

      .. raw:: html

         <a href="https://www.linkedin.com/company/heat-framework" style="text-decoration: none;">
            <i class="fa-brands fa-linkedin fa-4x mb-3" style="color: var(--pst-color-primary);"></i>
         </a>

      **LinkedIn**



.. toctree::
   :caption: Getting Started
   :hidden:
   :maxdepth: 1

   quick_start

.. toctree::
   :caption: Main Documentation
   :hidden:
   :maxdepth: 1

   usage
   /autoapi/index

.. toctree::
   :caption: Community & Development
   :hidden:
   :maxdepth: 1

   CONTRIBUTING
   news
