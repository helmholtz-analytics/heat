.. meta::
   :description: Scale NumPy-based data analysis to HPC
   :keywords: data analysis, HPC, MPI, GPU, multi-GPU, distributed computing, parallel processing, data science, data analytics, machine learning, scientific computing, high-performance computing, Python libraries, NumPy API, PyTorch

.. ================================================================================
.. Heat
.. ================================================================================

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

      **HPC & Multi-GPU Deployments**

      For Spack, EasyBuild, and containerized setups, refer to our comprehensive deployment guide.

      .. button-link:: /quick_start.html
         :color: primary
         :class: sd-btn-primary sd-btn-block mt-3

         View Full Installation Guide

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

Deep dive & architecture
========================

.. grid:: 1 1 1 1
   :gutter: 3
   :class-container: text-center mt-4 mb-4

   .. grid-item-card::
      :class-card: sd-card
      :link: /tutorials/notebooks/Internals
      :link-type: doc

      .. image:: _static/images/internals.png
         :alt: Performance Profiling Example
         :align: center
         :height: 140px

      .. div:: mt-3 **Heat internal functions**

      .. div:: text-muted small mt-1

         Heat internal functions for contributors and power users.

-----

Community & support
===================

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: mt-4 mb-5

   .. grid-item-card:: GitHub Discussions
      :class-card: sd-card
      :link: https://github.com/helmholtz-analytics/heat/discussions

      .. div:: mt-2 **Community Forum**

      .. div:: text-muted small mt-1

         Ask questions, share your HPC workflows, and discuss feature requests asynchronously.

   .. grid-item-card:: Matrix Space
      :class-card: sd-card
      :link: https://matrix.to/#/#heat:helmholtz.cloud

      .. div:: mt-2 **Real-time Chat**

      .. div:: text-muted small mt-1

         Join our Matrix space to chat directly with core developers and other Heat users.

   .. grid-item-card:: Issue Tracker
      :class-card: sd-card
      :link: https://github.com/helmholtz-analytics/heat/issues

      .. div:: mt-2 **Report a Bug**

      .. div:: text-muted small mt-1

         Encountered an issue with multi-GPU scaling or MPI communication? Let us know so we can fix it.

-----

Partner with us
===============

.. div:: mb-4

   Whether you are a motivated student looking for opportunities in computational research, or an institution interested in funding parallel computing infrastructure, we invite you to collaborate with the Heat core development team.

.. tab-set::

   .. tab-item:: Student Projects
      :sync: student

      .. grid:: 1 1 2 2
         :gutter: 3
         :padding: 0
         :class-container: text-left

         .. grid-item::
            :columns: 12 12 7 7

            **Kickstart Your Research in HPC & AI**

            We are actively looking for motivated BSc, MSc, and student workers to tackle open challenges in memory-distributed data science.

            * **Core Topic Tracks:** Massively parallel tensor operations, communication backends, algorithm development, user-requested features.
            * **What We Provide:** Direct mentorship from the core maintainers, computing time on top-tier HPC cluster environments, and clear paths to academic publication.

         .. grid-item::
            :columns: 12 12 5 5
            :class: text-center

            .. div:: mt-4

               .. button-link:: https://github.com/helmholtz-analytics/heat/discussions/categories/student-projects
                  :color: primary
                  :class: sd-btn-primary sd-btn-block mb-2

                  Explore Open Theses & Issues

               .. button-link:: mailto:contact@example.com
                  :color: secondary
                  :outline:
                  :class: sd-btn-block

                  Contact Core Maintainers

   .. tab-item:: Funding Calls
      :sync: funding

      .. grid:: 1 1 2 2
         :gutter: 3
         :padding: 0
         :class-container: text-left

         .. grid-item::
            :columns: 12 12 6 6

            **Institutional Backing & Joint Proposals**

            If you are assembling a domain-specific grant or third-party funding proposal (e.g., BMBFTR, EU, or Helmholtz transfer tracks) and need to scale data workflows to massive cluster environments, let's team up.

            We partner with both scientific institutions and industrial application partners during the proposal stage. If the project is successful, our core engineering team provides **ad-hoc support** to integrate Heat's distributed tensor operations directly into the project's pipeline.

         .. grid-item::
            :columns: 12 12 6 6

            **How We Integrate Into Project Consortia:**

            * **Technical Work Packages:** We help design dedicated work packages focused on software scaling, parallel acceleration, and performance optimization.
            * **HPC Middleware Expertise:** We act as the foundational parallel computing layer, helping translate domain-specific algorithms into cluster-ready architectures.
            * **Transfer Use-Case Co-Design:** We work side-by-side with your researchers or R&D engineers to implement custom distributed tensor operations unique to your scientific or technical discipline.

-----

Latest news
===========

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: mt-4

   .. grid-item-card:: Workshop registration open
      :class-card: sd-card
      :link: https://indico3-jsc.fz-juelich.de/event/327/

      *May 2026*

      Registration is open for our upcoming virtual workshop on high-performance data analytics. Join us for hands-on sessions, expert talks, and live demos of Heat in action.

   .. grid-item-card:: NumFOCUS Affiliation
      :class-card: sd-card
      :link: https://www.fz-juelich.de/en/rse/the_latest/the-heat-library-becomes-a-numfocus-affiliated-project

      *April 2026*

      Heat is officially a NumFOCUS affiliated project! A big milestone for our domain-agnostic library. Check out the announcement details.

   .. grid-item-card:: Version 1.8 Released
      :class-card: sd-card
      :link: https://github.com/helmholtz-analytics/heat/releases/tag/v1.8.0

      *March 2026*

      Our latest featured update is officially live. Check out the updated repository for full performance metrics.

   .. grid-item-card:: HeatHub
      :class-card: sd-card
      :link: https://www.fz-juelich.de/en/rse/the_latest/congratulations-scienceserve-recipients

      .. image:: _static/images/ScienceServe_banner.png
         :alt: ScienceServe Funding Banner
         :align: center

      ^^^

      *October 2025*

      Our project HeatHub has been awarded Helmholtz ScienceServe funding for 2026!

Roadmap
=======

.. div:: mb-4

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: mb-2

   .. grid-item-card:: Active Milestones
      :class-card: sd-card
      :link: https://github.com/helmholtz-analytics/heat/milestones
      :text-align: center

      Track our progress toward upcoming minor and major framework releases.

   .. grid-item-card:: Live Project Board
      :class-card: sd-card
      :link: https://github.com/orgs/helmholtz-analytics/projects/13/views/6
      :text-align: center

      See active pull requests, bug fixes, and feature implementations in real time.

   .. grid-item-card:: Contribute Code
      :class-card: sd-card
      :link: https://github.com/helmholtz-analytics/heat/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22
      :text-align: center

      Explore "good first issues" and feature requests for new contributors.

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
   documentation_howto
   CODE_OF_CONDUCT
