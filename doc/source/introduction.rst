Introduction
============

Goal
----

The goal of HeAT is to fill the gap between machine learning libraries that have
a strong focus on exploiting GPUs for performance, and traditional, distributed
high-performance computing (HPC). The basic idea is to provide a dtype,
distributed tensor library with machine learning methods based on it.

Among other things, the implementation will allow us to tackle use cases that
would otherwise exceed memory limits of a single node.

Features
--------

  * high-performance n-dimensional tensors
  * CPU, GPU and distributed computation using MPI
  * powerful machine learning methods using above mentioned tensors
