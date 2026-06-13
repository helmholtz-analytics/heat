.. meta::
   :description: Scale NumPy based data analysis to HPC
   :keywords: data analysis, HPC, MPI, GPU

========================
Heat
========================

.. image:: _static/images/logo.png
   :alt: Heat Logo
   :align: center
   :width: 55%

.. div:: text-center

   **High-performance data analytics in Python, at scale.**

   Heat is a flexible, distributed tensor framework that lets you scale NumPy code effortlessly across multi-node clusters.

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

Quick Example
=============

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
