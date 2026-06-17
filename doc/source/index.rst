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

.. Why Heat?
.. =========

.. Heat is a distributed tensor framework built on **PyTorch** and **mpi4py**. It provides highly optimized algorithms and data structures for tensor computations using CPUs, GPUs (CUDA/ROCm), and distributed cluster systems. It is designed to handle **massive arrays** that exceed the memory and computational limits of a single machine.

.. * **Seamless integration:** Port existing NumPy/SciPy code to multi-node clusters with minimal effort.
.. * **Hardware-agnostic:** Supports CPUs and GPUs (CUDA, ROCm, Apple MPS).
.. * **Efficient scaling:** Exploit the entire, cumulative RAM of your cluster for memory-intensive operations.

In a nutshell
===============

.. grid:: 1 1 2 2
   :gutter: 4
   :class-container: mt-4 mb-4

   .. grid-item::
      :columns: 12 12 5 5

      .. **Prototype on your laptop, execute on any cluster**

      Heat builds on **PyTorch** and **mpi4py** to process **massive arrays** - huge collections of images, MORE EXAMPLES - that exceed the memory and computational limits of a single machine.

      Define your data distribution axis via the ``split`` parameter, assign hardware using the ``device`` attribute, and let Heat orchestrate data movement and cross-node communication.

      **Prototype locally, execute on any cluster.**

      You got the expensive compute, but multi-GPU runs seem out of reach?
      ``pip install heat``

   .. grid-item::
      :columns: 12 12 7 7

      .. code-block:: python
         :caption: my_script.py

         import heat as ht

         # Distributed random matrix generation
         A = ht.random.randn(40000, 10000, split=0, device="gpu")
         B = ht.random.randn(10000, 40000, split=1, device="gpu")

         # Hardware-accelerated matrix multiplication
         C = ht.matmul(A, B)

      .. code-block:: bash
         :caption: Run locally or scale across cluster nodes via MPI

         mpirun -np 4 python my_script.py

-----

Getting started
================

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: text-center mt-4 mb-4

   .. grid-item-card::
      :class-card: sd-card
      :link: /quick_start
      :link-type: doc

      .. image:: _static/images/install_graphics.png
         :alt: Installation
         :align: center
         :height: 140px

      .. div:: mt-3 **Installation**

      .. div:: text-muted mt-1

         **Install guide** from your laptop to HPC systems

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

Tutorials
=========

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
      :link: /tutorials/notebooks/Internals
      :link-type: doc

      .. image:: _static/images/internals.png
         :alt: Performance Profiling Example
         :align: center
         :height: 140px

      .. div:: mt-3 **Heat internal functions**

      .. div:: text-muted small mt-1

         Heat internal functions for contributors and power users.

   .. grid-item-card::
      :class-card: sd-card
      :link: /tutorials/notebooks/README
      :link-type: doc

      .. image:: _static/images/internals.png
         :alt: All tutorials
         :align: center
         :height: 140px

      .. div:: mt-3 **All tutorials**

      .. div:: text-muted small mt-1

         All of our available tutorials are linked here.



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


.. -----


.. Documentation Portal
.. ====================

.. .. grid:: 1 1 2 4
..    :gutter: 3
..    :padding: 0
..    :class-container: mt-4 mb-4

..    .. grid-item-card:: Tutorials
..       :class-card: sd-card
..       :link: tutorials/notebooks/1_basics
..       :link-type: doc

..       **Learning-oriented** pages to guide you through your very first steps, configuration workflows, and basic cluster operations.

..    .. grid-item-card:: How-To Guides
..       :class-card: sd-card
..       :link: usage
..       :link-type: doc

..       **Task-oriented** recipes showing you how to solve specific analytical problems, load custom data files, and scale specific operations.

..    .. grid-item-card:: Explanations
..       :class-card: sd-card
..       :link: tutorials/notebooks/2_internals
..       :link-type: doc

..       **Understanding-oriented** deep dives into cluster architecture, the inner mechanics of ``DNDarray`` splitting, and parallelization theory.

..    .. grid-item-card:: API Reference
..       :class-card: sd-card
..       :link: api
..       :link-type: doc

..       **Information-oriented** technical specs covering function definitions, parameters, return types, and class structures.

.. .. toctree::
..    :hidden:

..    getting_started
..    usage
..    api

-----

Partner with Us
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
            * **What We Provide:** Direct mentorship from core open-source maintainers, computing time on top-tier HPC cluster environments, and clear paths to academic publication.

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

Latest News
===========

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: mt-4

   .. grid-item-card:: Save the Date
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
=======================

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

      Explore "good first issues" and open feature requests for new contributors.



.. toctree::
   :caption: Getting Started
   :hidden:
   :maxdepth: 1

   quick_start
   .. tutorials/tutorials
   .. case_studies

.. toctree::
   :caption: Main Documentation
   :hidden:
   :maxdepth: 1

   usage
   api
   documentation_howto

.. toctree::
   :caption: Community & Development
   :hidden:
   :maxdepth: 1

   CONTRIBUTING
   CODE_OF_CONDUCT
