
.. meta::
   :description: Scale NumPy-based data analysis to HPC
   :keywords: data analysis, HPC, MPI, GPU, multi-GPU, distributed computing, parallel processing, data science, data analytics, machine learning, scientific computing, high-performance computing, Python libraries, NumPy API, PyTorch

========================
Heat
========================

.. ----------------=====================================================
.. MAIN HERO BANNER PLACEHOLDER (TBD)
.. Replace this block with banner choice
.. ----------------=====================================================
.. image:: _static/images/tbd.png
   :alt: Heat  Banner
   :align: center
   :width: 100%

.. div:: text-center
   :class: mt-4 mb-4

   **High-performance data analytics in Python, at scale.**

.. grid:: 1 2 4 4
   :gutter: 3
   :class-container: text-center

   .. grid-item::
      **Distributed**

      .. div:: mt-2

         Multi-node data processing. MPI-based performance.

         .. image:: _static/images/mpi_logo_icon.png
            :width: 40px
            :alt: MPI Compatible

   .. grid-item::
      **Accelerated**

      .. div:: mt-2

         Native multi-GPU support. Out of the box.

         .. image:: _static/images/pytorch_logo_icon.png
            :width: 35px
            :alt: PyTorch Engine

   .. grid-item::
      **Scalable**

      .. div:: mt-2

         Scale beyond single-node memory limits. Effortlessly.

         .. image:: _static/images/cluster_logo_icon.png
            :width: 35px
            :alt: Multi-Node Memory Scaling

   .. grid-item::
      **Interoperable**

      .. div:: mt-2

         Plug & play with the Numpy ecosystem.

         .. image:: _static/images/numpy_logo_icon.png
            :width: 35px
            :alt: NumPy API Mirror

.. div:: text-center
   :class: mt-4 mb-4

   .. button-link:: https://github.com/helmholtz-analytics/heat
      :color: primary
      :class: sd-btn-primary

      Get Started on GitHub

-----

.. grid:: 1 1 2 2
   :gutter: 4
   :padding: 2

   .. grid-item::
      :columns: 12 12 6 6

      Why Heat?
      =========

      * **Seamless Integration:** Port existing NumPy/SciPy code to multi-node clusters with minimal effort.
      * **Hardware-Agnostic:** Built on PyTorch and ``mpi4py`` to support native execution on CPUs and cluster-wide GPUs (CUDA, ROCm, Apple MPS).
      * **Efficient Scaling:** Exploit the entire, cumulative RAM of your cluster to run memory-intensive operations effortlessly.

   .. grid-item::
      :columns: 12 12 6 6
      :class: text-center

      .. image:: _static/images/tutorial_split_dndarray.svg
         :alt: Heat Cluster Data Distribution Architecture
         :align: center
         :width: 90%

-----

Examples
========

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item::
      :columns: 12 12 5 5

      **Write Local Code, Execute Across Clusters**

      Define your data distribution format via the ``split`` parameter , assign processing hardware using the ``device`` attribute, and let Heat automatically orchestrate data movement and cross-node communication.

      Launch the exact same script on your personal workstation using standard Python routines or distributed via infrastructure systems like SLURM.

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
   :caption: Execution

   # Run locally on 4 CPU/GPU workers
   mpirun -np 4 python my_script.py

-----

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: text-center

   .. grid-item-card::
      :class-card: sd-card
      :link: tutorials/notebooks/4_matrix_factorizations
      :link-type: doc

      .. image:: _static/images/hSVD_bench_rank5.png
         :alt: Matrix Factorizations Example
         :align: center
         :height: 140px

      .. div:: mt-3 **Matrix Factorizations**

      .. div:: text-muted small mt-1

         Distributed SVD and PCA pipelines for high-dimensional feature decomposition.

   .. grid-item-card::
      :class-card: sd-card
      :link: tutorials/notebooks/2_internals
      :link-type: doc

      .. image:: _static/images/tutorial_split_dndarray.svg
         :alt: Massively Parallel Clustering Example
         :align: center
         :height: 140px

      .. div:: mt-3 **Massively Parallel Clustering**

      .. div:: text-muted small mt-1

         K-Means and spectral data partitioning across multi-node GPU memories.

   .. grid-item-card::
      :class-card: sd-card
      :link: tutorials/notebooks/6_profiling
      :link-type: doc

      .. image:: _static/images/perun_logo.svg
         :alt: Performance Profiling Example
         :align: center
         :height: 140px

      .. div:: mt-3 **Performance Profiling**

      .. div:: text-muted small mt-1

         Tracking cluster memory consumption and execution efficiency using Perun.

-----

Documentation Portal
====================

.. grid:: 1 1 2 4
   :gutter: 3
   :padding: 0

   .. grid-item-card:: Tutorials
      :class-card: sd-card
      :link: tutorials/notebooks/1_basics
      :link-type: doc

      **Learning-oriented** pages to guide you through your very first steps, configuration workflows, and basic cluster operations.

   .. grid-item-card:: How-To Guides
      :class-card: sd-card
      :link: usage
      :link-type: doc

      **Task-oriented** recipes showing you how to solve specific analytical problems, load custom data files, and scale specific operations.

   .. grid-item-card:: Explanations
      :class-card: sd-card
      :link: tutorials/notebooks/2_internals
      :link-type: doc

      **Understanding-oriented** deep dives into cluster architecture, the inner mechanics of ``DNDarray`` splitting, and parallelization theory.

   .. grid-item-card:: API Reference
      :class-card: sd-card
      :link: api
      :link-type: doc

      **Information-oriented** technical specs covering function definitions, parameters, return types, and class structures.

.. toctree::
   :hidden:

   getting_started
   usage
   api

-----

Latest News
===========

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Version 1.8 Released
      :class-card: sd-card
      :link: https://github.com/helmholtz-analytics/heat/releases/tag/v1.8.0

      *March 2026*

      Our latest featured update is officially live. Check out the updated repository for full performance metrics.

   .. grid-item-card:: NumFOCUS Affiliation
      :class-card: sd-card

      *April 2026*

      Heat is officially a NumFOCUS affiliated project! We are proud to join this open-source ecosystem.

   .. grid-item-card:: Save the Date
      :class-card: sd-card

      *November 2026*

      Registration opens soon for our upcoming virtual workshop on high-performance data analytics.

   .. grid-item-card:: Bogus News
      :class-card: sd-card

      *December 2026*

      Registration opens soon for our upcoming virtual workshop on high-performance data analytics.


.. .. meta::
..    :description: Scale NumPy based data analysis to HPC
..    :keywords: data analysis, HPC, MPI, GPU

.. ========================
.. Heat
.. ========================

.. .. image:: _static/images/logo.png
..    :alt: Heat
..    :align: center
..    :width: 60%

.. .. div:: text-center

..    **High-performance data analytics in Python, at scale.**

..    `Getting Started <https://heat.readthedocs.io/>`_ | `Tutorials <https://github.com/helmholtz-analytics/heat>`_ | `Contributing <https://heat.readthedocs.io/>`_

.. -----

.. Features
.. ========

.. .. grid:: 1 1 3 3
..    :gutter: 3
..    :padding: 2

..    .. grid-item-card:: 🔗 Seamless Integration
..       :class-card: sd-card

..       Port existing NumPy/SciPy code to multi-node clusters with minimal effort.

..    .. grid-item-card:: 💻 Hardware-Agnostic
..       :class-card: sd-card

..       Supports CPUs and distributed GPUs (CUDA, ROCm, Apple MPS) out of the box.

..    .. grid-item-card:: 🚀 Efficient Scaling
..       :class-card: sd-card

..       Exploit the entire, cumulative RAM of your cluster for memory-intensive operations.

.. Why Heat?
.. =========

.. * **No Learning Curve:** Mirroring the NumPy interface (except for ``split`` and ``device`` attributes) means you can write cluster code instantly.
.. * **Complex Parallelism Made Easy:** Implements non-trivially parallelized functions like matrix factorizations or PCA seamlessly.
.. * **Massive Datasets:** Excellent weak scaling capabilities enable you to process data that exceeds single-machine limits.

.. Quick Example
.. =============

.. .. grid:: 1 1 2 2
..    :gutter: 4

..    .. grid-item::
..       :columns: 12 12 5 5

..       **Write Distributed Code Effortlessly**

..       Define how your data is partitioned across the cluster using the ``split`` argument and specify the device target via ``device``.

..       Heat handles inter-node communication automatically. Run it seamlessly via ``mpirun`` on your laptop or ``srun`` on your cluster.

..       .. button-link:: https://github.com/helmholtz-analytics/heat
..          :color: primary
..          :class: sd-btn-primary
..          :expand:

..          🚀 Get Started on GitHub

..    .. grid-item::
..       :columns: 12 12 7 7

..       .. code-block:: python
..          :caption: my_script.py

..          import heat as ht

..          # In-memory distributed random matrices
..          A = ht.random.randn(40000, 10000, split=0, device="gpu")
..          B = ht.random.randn(10000, 40000, split=1, device="gpu")

..          # Automatic parallel matrix multiplication
..          C = ht.matmul(A, B)

.. .. toctree::
..    :hidden:

..    getting_started
..    usage
..    api

.. Latest News
.. ===========

.. .. grid:: 1 1 3 3
..    :gutter: 3

..    .. grid-item-card:: 🚀 Version 1.8 Released
..       :class-card: sd-card
..       :link: https://github.com/helmholtz-analytics/heat/releases/tag/v1.8.0

..       *March 2026*

..       Our latest featured update is live. Check out the release notes to learn about performance upgrades and changes.

..    .. grid-item-card:: 📅 Save the Date
..       :class-card: sd-card

..       *November 2026*

..       Our upcoming hands-on workshop is just around the corner. Stay tuned for registration info!

..    .. grid-item-card:: 🤝 NumFOCUS Affiliation
..       :class-card: sd-card

..       *April 2026*

..       We are thrilled to announce that Heat is now officially a NumFOCUS affiliated project.
