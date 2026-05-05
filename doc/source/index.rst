.. meta::
   :description: Scale NumPy based data analysis to HPC
   :keywords: data analysis, HPC, MPI, GPU

========================
Heat
========================

.. image:: _static/images/logo.png
   :alt: Heat
   :align: center
   :width: 80%

-----

Features
========

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Seamless integration
      :class-card: sd-shadow-sm

      Port existing NumPy/SciPy code to multi-node clusters with minimal effort

   .. grid-item-card:: Hardware-agnostic
      :class-card: sd-shadow-sm

      Supports CPUs and GPUs (CUDA, ROCm, Apple MPS)

   .. grid-item-card:: Efficient scaling
      :class-card: sd-shadow-sm

      Exploit the entire, cumulative RAM of your cluster for memory-intensive operations

-----

Quick Example
=============

Define distribution via the ``split`` argument and specify the device via the ``device`` arguments:

.. code-block:: python

   import heat as ht

   A = ht.random.randn(40000, 10000, split=0, device="gpu")
   B = ht.random.randn(10000, 40000, split=1, device="gpu")
   C = ht.matmul(A, B)

Then, run with ``mpirun`` on your laptop or ``srun`` on your cluster

.. code-block:: bash

   mpirun -np 4 python my_script.py


-----

Why Heat?
=========

- Mirroring of NumPy interface except for ``split`` and ``device`` attributes removes the learning curve.
- Implements many functions in parallel that are non-trivially parallelized such as matrix factorizations or PCA.
- Excellent weak scaling capabilities enable processing of huge datasets.

-----

Get Started
===========

.. button-link:: https://github.com/helmholtz-analytics/heat
   :color: primary
   :expand:

   🚀 Get started on GitHub

-----

.. toctree::
   :hidden:

   getting_started
   usage
   api


-----

Latest News
===========

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item::
      :columns: 12 12 7 7

      .. grid-item-card::
         :class-card: sd-shadow-md
         :link: https://github.com/helmholtz-analytics/heat/releases/tag/v1.8.0
         :link-type: url

         **🚀 Version 1.8 released**

         *March 2026*

         This is your **featured update**. Use a slightly longer description here
         to highlight the most important change or announcement.


   .. grid-item::
      :columns: 12 12 5 5

      .. grid:: 1
         :gutter: 2

         .. grid-item-card::
            :class-card: sd-shadow-sm

            **Save the date**

            *November 2026*

            Upcoming workshop

         .. grid-item-card::
            :class-card: sd-shadow-sm

            **NumFOCUS affiliation**

            *April 2026*

            Heat is now a NumFOCUS affiliated project.
