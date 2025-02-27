v1.5.1
======

Changes
-------

Compatibility
~~~~~~~~~~~~~

-  #1706 Support PyTorch 2.5.1 (#1701) (by @mtar)
-  #1775 Support PyTorch 2.6.0 (#1775) (by @mrfh92)

Bug Fixes
~~~~~~~~~

-  #1791 ``heat.eq``, ``heat.ne`` now allow non-array operands (by
   @\ `github-actions[bot] <https://github.com/apps/github-actions>`__)
-  #1790 Fixed precision loss in several functions when dtype is float64
   (by
   @\ `github-actions[bot] <https://github.com/apps/github-actions>`__)
-  #1764 Printing non-distributed data (by
   @\ `github-actions[bot] <https://github.com/apps/github-actions>`__)

CI
~~

-  #1750 Linters will no longer format tutorials (by
   @\ `github-actions[bot] <https://github.com/apps/github-actions>`__)
-  #1708 Documentation updates after new release (by
   @\ `github-actions[bot] <https://github.com/apps/github-actions>`__)
-  #1743 Modernise setup.py configuration (by
   @\ `github-actions[bot] <https://github.com/apps/github-actions>`__)

Docs
~~~~

-  #1698 Added Dalcin et al. 2018 reference to
   ``manipulations._axis2axisResplit`` (by
   @\ `github-actions[bot] <https://github.com/apps/github-actions>`__)
-  #1745 Easier access to GitHub from the docs (by
   @\ `github-actions[bot] <https://github.com/apps/github-actions>`__)

Contributors
------------

@ClaudiaComito, @JuanPedroGHM, @github-actions [bot], @joernhees,
@mrfh92, @mtar and
`github-actions[bot] <https://github.com/apps/github-actions>`__ v1.5.1
- Heat 1.5.1

v1.5.0
======

.. _changes-1:

Changes
-------

Cluster
~~~~~~~

-  #1593 Improved Batch Parallelization. (by @mrfh92)

Data
~~~~

-  #1529 Make ``dataset.ishuffle`` optional.

IO
~~

-  #1602 Improved load balancing when loading .npy files from path. (by
   @Reisii)
-  #1551 Improved load balancing when loading .csv files from path. (by
   @Reisii)

Linear Algebra
~~~~~~~~~~~~~~

-  #1261 Batched matrix multiplication. (by @FOsterfeld)
-  #1504 Add solver for triangular systems. (by @FOsterfeld)

Manipulations
~~~~~~~~~~~~~

-  #1419 Implement distributed ``unfold`` operation. (by @FOsterfeld)

Random
~~~~~~

-  #1508 Introduce Batchparallel for RNG as default. (by @mrfh92)

Signal
~~~~~~

-  #1515 Support batch 1-d convolution in ``ht.signal.convolve``. (by
   @ClaudiaComito)

Statistics
~~~~~~~~~~

-  #1510 Support multiple axes for ``ht.percentile``. (by
   @ClaudiaComito)

Sparse
~~~~~~

-  #1377 Distributed Compressed Sparse Column Matrix. (by @Mystic-Slice)

Other
~~~~~

-  #1618 Support mpi4py 4.x.x (by @JuanPedroGHM)

.. _contributors-1:

Contributors
------------

@ClaudiaComito, @FOsterfeld, @JuanPedroGHM, @Reisii, @mrfh92, @mtar and
Hoppe

v1.4.2 - Maintenance release
============================

.. _changes-2:

Changes
-------

Maintenance
~~~~~~~~~~~

-  `#1467 <https://github.com/helmholtz-analytics/heat/pull/1467>`__,
   `#1525 <https://github.com/helmholtz-analytics/heat/pull/1525>`__
   Support PyTorch 2.3.1 (by @mtar)
-  `#1535 <https://github.com/helmholtz-analytics/heat/pull/1535>`__
   Address test failures after netCDF4 1.7.1, numpy 2 releases (by
   @ClaudiaComito)

v1.4.1 - Bug fix release
========================

.. _changes-3:

Changes
-------

.. _bug-fixes-1:

Bug fixes
~~~~~~~~~

-  #1472 DNDarrays returned by ``_like`` functions default to same
   device as input DNDarray (by @mrfh92, @ClaudiaComito)

.. _maintenance-1:

Maintenance
~~~~~~~~~~~

-  #1441 added names of non-core members in citation file (by @mrfh92)

v1.4.0 - Interactive HPC tutorials, distributed FFT, batch-parallel clustering, support PyTorch 2.2.2
=====================================================================================================

.. _changes-4:

Changes
-------

Documentation
~~~~~~~~~~~~~

-  #1406 New tutorials for interactive parallel mode for both HPC and
   local usage (by @ClaudiaComito)

🔥 Features
~~~~~~~~~~~

-  #1288 Batch-parallel K-means and K-medians (by @mrfh92)
-  #1228 Introduce in-place-operators for ``arithmetics.py`` (by
   @LScheib)
-  #1218 Distributed Fast Fourier Transforms (by @ClaudiaComito)

.. _bug-fixes-2:

Bug fixes
~~~~~~~~~

-  #1363 ``ht.array`` constructor respects implicit torch device when
   copy is set to false (by @JuanPedroGHM)
-  #1216 Avoid unnecessary gathering of distributed operand (by
   @samadpls)
-  #1329 Refactoring of QR: stabilized Gram-Schmidt for split=1 and
   TS-QR for split=0 (by @mrfh92)

Interoperability
~~~~~~~~~~~~~~~~

-  #1418 and #1290: Support PyTorch 2.2.2 (by @mtar)
-  #1315 and #1337: Fix some NumPy deprecations in the core and
   statistics tests (by @FOsterfeld)

v1.3.1 - Bug fixes, Docker documentation update
===============================================

.. _bug-fixes-3:

Bug fixes
---------

-  #1259 Bug-fix for ``ht.regression.Lasso()`` on GPU (by @mrfh92)
-  #1201 Fix ``ht.diff`` for 1-element-axis edge case (by @mtar)

.. _changes-5:

Changes
-------

.. _interoperability-1:

Interoperability
~~~~~~~~~~~~~~~~

-  #1257 Docker release 1.3.x update (by @JuanPedroGHM)

.. _maintenance-2:

Maintenance
~~~~~~~~~~~

-  #1274 Update version before release (by @ClaudiaComito)
-  #1267 Unit tests: Increase tolerance for ``ht.allclose`` on
   ``ht.inv`` operations for all torch versions (by @ClaudiaComito)
-  #1266 Sync ``pre-commit`` configuration with ``main`` branch (by
   @ClaudiaComito)
-  #1264 Fix Pytorch release tracking workflows (by @mtar)
-  #1234 Update sphinx package requirements (by @mtar)
-  #1187 Create configuration file for Read the Docs (by @mtar)

v1.3.0 - Scalable SVD, GSoC`22 contributions, Docker image, PyTorch 2 support, AMD GPUs acceleration
====================================================================================================

This release includes many important updates (see below). We
particularly would like to thank our enthusiastic
`GSoC2022 <https://summerofcode.withgoogle.com/programs/2022>`__ /
tentative GSoC2023 contributors @Mystic-Slice @neosunhan @Sai-Suraj-27
@shahpratham @AsRaNi1 @Ishaan-Chandak 🙏🏼 Thank you so much!

Highlights
----------

-  #1155 Support PyTorch 2.0.1 (by @ClaudiaComito)
-  #1152 Support AMD GPUs (by @mtar)
-  #1126 `Distributed hierarchical
   SVD <https://helmholtz-analytics.github.io/heat/2023/06/16/new-feature-hsvd.html>`__
   (by @mrfh92)
-  #1028 Introducing the ``sparse`` module: Distributed Compressed
   Sparse Row Matrix (by @Mystic-Slice)
-  Performance improvements:

   -  #1125 distributed ``heat.reshape()`` speed-up (by @ClaudiaComito)
   -  #1141 ``heat.pow()`` speed-up when exponent is ``int`` (by
      @ClaudiaComito @coquelin77 )
   -  #1119 ``heat.array()`` default to ``copy=None`` (e.g., only if
      necessary) (by @ClaudiaComito @neosunhan )

-  #970 `Dockerfile and accompanying
   documentation <https://github.com/helmholtz-analytics/heat/tree/release/1.3.x/docker>`__
   (by @bhagemeier)

Changelog
---------

Array-API compliance / Interoperability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  #1154 Introduce ``DNDarray.__array__()`` method for interoperability
   with ``numpy``, ``xarray`` (by @ClaudiaComito)
-  #1147 Adopt
   `NEP29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`__,
   drop support for PyTorch 1.7, Python 3.6 (by @mtar)
-  #1119 ``ht.array()`` default to ``copy=None`` (e.g., only if
   necessary) (by @ClaudiaComito)
-  #1020 Implement ``broadcast_arrays``, ``broadcast_to`` (by
   @neosunhan)
-  #1008 API: Rename ``keepdim`` kwarg to ``keepdims`` (by @neosunhan)
-  #788 Interface for
   `DPPY <https://github.com/IntelPython/DPPY-Spec>`__ interoperability
   (by @coquelin77 @fschlimb )

New Features
~~~~~~~~~~~~

-  #1126 `Distributed hierarchical
   SVD <https://helmholtz-analytics.github.io/heat/2023/06/16/new-feature-hsvd.html>`__
   (by @mrfh92)
-  #1020 Implement ``broadcast_arrays``, ``broadcast_to`` (by
   @neosunhan)
-  #983 Signal processing: fully distributed 1D convolution (by
   @shahpratham)
-  #1063 add **eq** to Device (by @mtar)

.. _bug-fixes-4:

Bug Fixes
~~~~~~~~~

-  #1141 ``heat.pow()`` speed-up when exponent is ``int`` (by
   @ClaudiaComito)
-  #1136 Fixed PyTorch version check in ``sparse`` module (by
   @Mystic-Slice)
-  #1098 Validates number of dimensions in input to
   ``ht.sparse.sparse_csr_matrix`` (by @Ishaan-Chandak)
-  #1095 Convolve with distributed kernel on multiple GPUs (by
   @shahpratham)
-  #1094 Fix division precision error in ``random`` module (by
   @Mystic-Slice)
-  #1075 Fixed initialization of DNDarrays communicator in some routines
   (by @AsRaNi1)
-  #1066 Verify input object type and layout + Supporting tests (by
   @Mystic-Slice)
-  #1037 Distributed weighted ``average()`` along tuple of axes: shape
   of ``weights`` to match shape of input (by @Mystic-Slice)

Benchmarking
~~~~~~~~~~~~

-  #1137 Continous Benchmarking of runtime (by @JuanPedroGHM)

.. _documentation-1:

Documentation
~~~~~~~~~~~~~

-  #1150 Refactoring for efficiency and readability (by @Sai-Suraj-27)
-  #1130 Reintroduce Quick Start (by @ClaudiaComito)
-  #1079 A better README file (by @Sai-Suraj-27)

.. _linear-algebra-1:

Linear Algebra
~~~~~~~~~~~~~~

-  #1126, #1160 `Distributed hierarchical
   SVD <https://helmholtz-analytics.github.io/heat/2023/06/16/new-feature-hsvd.html>`__
   (by @mrfh92 @ClaudiaComito )

v1.2.2 - Bug fixes, support OpenMPI>=4.1.2, support PyTorch 1.13.1
==================================================================

.. _changes-6:

Changes
-------

Communication
-------------

-  #1058 Fix edge-case contiguity mismatch for Allgatherv (by
   @ClaudiaComito)

.. _contributors-2:

Contributors
------------

@ClaudiaComito, @JuanPedroGHM

v1.2.1
======

.. _changes-7:

Changes
-------

-  #1048 Support PyTorch 1.13.0 on branch release/1.2.x (by
   @github-actions)

.. _bug-fixes-5:

🐛 Bug Fixes
------------

-  #1038 Lanczos decomposition ``linalg.solver.lanczos``: Support double
   precision, complex data types (by @ClaudiaComito)
-  #1034 ``ht.array``, closed loophole allowing ``DNDarray``
   construction with incompatible shapes of local arrays (by
   @Mystic-Slice)

.. _linear-algebra-2:

Linear Algebra
--------------

-  #1038 Lanczos decomposition ``linalg.solver.lanczos``: Support double
   precision, complex data types (by @ClaudiaComito)

🧪 Testing
----------

-  #1025 mirror repository on gitlab + ci (by @mtar)
-  #1014 fix: set cuda rng state on gpu tests for test_random.py (by
   @JuanPedroGHM)

v1.2.0
======

.. _highlights-1:

Highlights
----------

-  `#906 <https://github.com/helmholtz-analytics/heat/pull/906>`__
   PyTorch 1.11 support
-  `#595 <https://github.com/helmholtz-analytics/heat/pull/595>`__
   Distributed 1-D convolution: ``ht.convolve``
-  `#941 <https://github.com/helmholtz-analytics/heat/pull/941>`__
   Parallel I/O: write to CSV file with ``ht.save_csv``.
-  `#887 <https://github.com/helmholtz-analytics/heat/pull/887>`__
   Binary operations between operands of equal shapes, equal ``split``
   axes, but different distribution maps.
-  Expanded memory-distributed `linear algebra <#linalg>`__,
   `manipulations <#manipulations>`__ modules.

.. _bug-fixes-6:

Bug Fixes
---------

-  `#826 <https://github.com/helmholtz-analytics/heat/pull/826>`__ Fixed
   ``__setitem__`` handling of distributed ``DNDarray`` values which
   have a different shape in the split dimension
-  `#846 <https://github.com/helmholtz-analytics/heat/pull/846>`__ Fixed
   an issue in ``_reduce_op`` when axis and keepdim were set.
-  `#846 <https://github.com/helmholtz-analytics/heat/pull/846>`__ Fixed
   an issue in ``min``, ``max`` where DNDarrays with empty processes
   can’t be computed.
-  `#868 <https://github.com/helmholtz-analytics/heat/pull/868>`__ Fixed
   an issue in ``__binary_op`` where data was falsely distributed if a
   DNDarray has single element.
-  `#916 <https://github.com/helmholtz-analytics/heat/pull/916>`__ Fixed
   an issue in ``random.randint`` where the size parameter does not
   accept ints.

.. _new-features-1:

New features
------------

Arithmetics
~~~~~~~~~~~

-  `#945 <https://github.com/helmholtz-analytics/heat/pull/945>`__
   ``ht.divide`` now supports ``out`` and ``where`` kwargs ###
   Communication
-  `#868 <https://github.com/helmholtz-analytics/heat/pull/868>`__ New
   ``MPICommunication`` method ``Split``
-  `#940 <https://github.com/helmholtz-analytics/heat/pull/940>`__ and
   `#967 <https://github.com/helmholtz-analytics/heat/pull/967>`__
   Duplicate ``MPI.COMM_WORLD`` and ``MPI_SELF`` to make library more
   independent.

DNDarray
~~~~~~~~

-  `#856 <https://github.com/helmholtz-analytics/heat/pull/856>`__ New
   ``DNDarray`` method ``__torch_proxy__``
-  `#885 <https://github.com/helmholtz-analytics/heat/pull/885>`__ New
   ``DNDarray`` method ``conj`` ### Linear Algebra
-  `#840 <https://github.com/helmholtz-analytics/heat/pull/840>`__ New
   feature: ``vecdot()``
-  `#842 <https://github.com/helmholtz-analytics/heat/pull/842>`__ New
   feature: ``vdot``
-  `#846 <https://github.com/helmholtz-analytics/heat/pull/846>`__ New
   features ``norm``, ``vector_norm``, ``matrix_norm``
-  `#850 <https://github.com/helmholtz-analytics/heat/pull/850>`__ New
   Feature ``cross``
-  `#877 <https://github.com/helmholtz-analytics/heat/pull/877>`__ New
   feature ``det``
-  `#875 <https://github.com/helmholtz-analytics/heat/pull/875>`__ New
   feature ``inv`` ### Logical
-  `#862 <https://github.com/helmholtz-analytics/heat/pull/862>`__ New
   feature ``signbit`` ### Manipulations
-  `#829 <https://github.com/helmholtz-analytics/heat/pull/829>`__ New
   feature: ``roll``
-  `#853 <https://github.com/helmholtz-analytics/heat/pull/853>`__ New
   Feature: ``swapaxes``
-  `#854 <https://github.com/helmholtz-analytics/heat/pull/854>`__ New
   Feature: ``moveaxis`` ### Printing
-  `#816 <https://github.com/helmholtz-analytics/heat/pull/816>`__ New
   feature: Local printing (``ht.local_printing()``) and global printing
   options
-  `#816 <https://github.com/helmholtz-analytics/heat/pull/816>`__ New
   feature: print only on process 0 with ``print0(...)`` and
   ``ht.print0(...)`` ### Random
-  `#858 <https://github.com/helmholtz-analytics/heat/pull/858>`__ New
   Feature: ``standard_normal``, ``normal`` ### Rounding
-  `#827 <https://github.com/helmholtz-analytics/heat/pull/827>`__ New
   feature: ``sign``, ``sgn`` ### Statistics
-  `#928 <https://github.com/helmholtz-analytics/heat/pull/928>`__ New
   feature: ``bucketize``, ``digitize`` ### General
-  `#876 <https://github.com/helmholtz-analytics/heat/pull/876>`__ Fix
   examples (Lasso and kNN)
-  `#894 <https://github.com/helmholtz-analytics/heat/pull/894>`__
   Change inclusion of license file
-  `#948 <https://github.com/helmholtz-analytics/heat/pull/948>`__
   Improve CSV write performance.
-  `#960 <https://github.com/helmholtz-analytics/heat/pull/960>`__
   Bypass unnecessary communication by replacing ``factories.array``
   with\ ``DNDarray`` contruct in random.py

v1.1.1
======

-  `#864 <https://github.com/helmholtz-analytics/heat/pull/864>`__
   Dependencies: constrain ``torchvision`` version range to match
   supported ``pytorch`` version range.

.. _highlights-2:

Highlights
----------

-  Slicing/indexing overhaul for a more NumPy-like user experience.
   Warning for distributed arrays: `breaking
   change! <#breaking-changes>`__ Indexing one element along the
   distribution axis now implies the indexed element is communicated to
   all processes.
-  More flexibility in handling non-load-balanced distributed arrays.
-  More distributed operations,
   incl. `meshgrid <https://github.com/helmholtz-analytics/heat/pull/794>`__.

Breaking Changes
----------------

-  `#758 <https://github.com/helmholtz-analytics/heat/pull/758>`__
   Indexing a distributed ``DNDarray`` along the ``DNDarray.split``
   dimension now returns a non-distributed ``DNDarray``, i.e. the
   indexed element is MPI-broadcasted. Example on 2 processes:

   .. code:: python

      a = ht.arange(5 * 5, split=0).reshape((5, 5))
      print(a.larray)
      >>> [0] tensor([[ 0,  1,  2,  3,  4],
      >>> [0]         [ 5,  6,  7,  8,  9],
      >>> [0]         [10, 11, 12, 13, 14]], dtype=torch.int32)
      >>> [1] tensor([[15, 16, 17, 18, 19],
      >>> [1]         [20, 21, 22, 23, 24]], dtype=torch.int32)
      b = a[:, 2]
      print(b.larray)
      >>> [0] tensor([ 2,  7, 12], dtype=torch.int32)
      >>> [1] tensor([17, 22], dtype=torch.int32)
      print(b.shape)
      >>> [0] (5,)
      >>> [1] (5,)
      print(b.split)
      >>> [0] 0
      >>> [1] 0
      c = a[4]
      print(c.larray)
      >>> [0] tensor([20, 21, 22, 23, 24], dtype=torch.int32)
      >>> [1] tensor([20, 21, 22, 23, 24], dtype=torch.int32)
      print(c.shape)
      >>> [0] (5,)
      >>> [1] (5,)
      print(c.split)
      >>> [0] None
      >>> [1] None

.. _bug-fixes-7:

Bug Fixes
---------

-  `#758 <https://github.com/helmholtz-analytics/heat/pull/758>`__ Fix
   indexing inconsistencies in ``DNDarray.__getitem__()``
-  `#768 <https://github.com/helmholtz-analytics/heat/pull/768>`__ Fixed
   an issue where ``deg2rad`` and ``rad2deg``\ are not working with the
   ‘out’ parameter.
-  `#785 <https://github.com/helmholtz-analytics/heat/pull/785>`__
   Removed ``storage_offset`` when finding the mpi buffer
   (``communication. MPICommunication.as_mpi_memory()``).
-  `#785 <https://github.com/helmholtz-analytics/heat/pull/785>`__ added
   allowance for 1 dimensional non-contiguous local tensors in
   ``communication. MPICommunication.mpi_type_and_elements_of()``
-  `#787 <https://github.com/helmholtz-analytics/heat/pull/787>`__ Fixed
   an issue where Heat cannot be imported when some optional
   dependencies are not available.
-  `#790 <https://github.com/helmholtz-analytics/heat/pull/790>`__ catch
   incorrect device after ``bcast`` in ``DNDarray.__getitem__``
-  `#796 <https://github.com/helmholtz-analytics/heat/pull/796>`__
   ``heat.reshape(a, shape, new_split)`` now always returns a
   distributed ``DNDarray`` if ``new_split is not None`` (inlcuding when
   the original input ``a`` is not distributed)
-  `#811 <https://github.com/helmholtz-analytics/heat/pull/811>`__ Fixed
   memory leak in ``DNDarray.larray``
-  `#820 <https://github.com/helmholtz-analytics/heat/pull/820>`__
   ``randn`` values are pushed away from 0 by the minimum value the
   given dtype before being transformed into the Gaussian shape
-  `#821 <https://github.com/helmholtz-analytics/heat/pull/821>`__ Fixed
   ``__getitem__`` handling of distributed ``DNDarray`` key element
-  `#831 <https://github.com/helmholtz-analytics/heat/pull/831>`__
   ``__getitem__`` handling of ``array-like`` 1-element key

Feature additions
-----------------

Exponential
~~~~~~~~~~~

-  `#812 <https://github.com/helmholtz-analytics/heat/pull/712>`__ New
   feature: ``logaddexp``, ``logaddexp2``

.. _linear-algebra-3:

Linear Algebra
~~~~~~~~~~~~~~

-  `#718 <https://github.com/helmholtz-analytics/heat/pull/718>`__ New
   feature: ``trace()``
-  `#768 <https://github.com/helmholtz-analytics/heat/pull/768>`__ New
   feature: unary positive and negative operations
-  `#820 <https://github.com/helmholtz-analytics/heat/pull/820>`__
   ``dot`` can handle matrix-vector operation now

.. _manipulations-1:

Manipulations
~~~~~~~~~~~~~

-  `#796 <https://github.com/helmholtz-analytics/heat/pull/796>`__
   ``DNDarray.reshape(shape)``: method now allows shape elements to be
   passed in as single arguments.

Trigonometrics / Arithmetic
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  `#806 <https://github.com/helmholtz-analytics/heat/pull/809>`__ New
   feature: ``square``
-  `#809 <https://github.com/helmholtz-analytics/heat/pull/809>`__ New
   feature: ``acosh``, ``asinh``, ``atanh``

Misc.
~~~~~

-  `#761 <https://github.com/helmholtz-analytics/heat/pull/761>`__ New
   feature: ``result_type``
-  `#788 <https://github.com/helmholtz-analytics/heat/pull/788>`__ Added
   the partition interface ``DNDarray`` for use with DPPY
-  `#794 <https://github.com/helmholtz-analytics/heat/pull/794>`__ New
   feature: ``meshgrid``
-  `#821 <https://github.com/helmholtz-analytics/heat/pull/821>`__
   Enhancement: it is no longer necessary to load-balance an imbalanced
   ``DNDarray`` before gathering it onto all processes. In short:
   ``ht.resplit(array, None)`` now works on imbalanced arrays as well.

v1.0.0
======

New features / Highlights
-------------------------

-  `#660 <https://github.com/helmholtz-analytics/heat/pull/660>`__ NN
   module for data parallel neural networks
-  `#699 <https://github.com/helmholtz-analytics/heat/pull/699>`__
   Support for complex numbers; New functions: ``angle``, ``real``,
   ``imag``, ``conjugate``
-  `#702 <https://github.com/helmholtz-analytics/heat/pull/702>`__
   Support channel stackoverflow
-  `#728 <https://github.com/helmholtz-analytics/heat/pull/728>`__
   ``DASO`` optimizer
-  `#757 <https://github.com/helmholtz-analytics/heat/pull/757>`__ Major
   documentation overhaul, custom docstrings formatting

.. _bug-fixes-8:

Bug fixes
~~~~~~~~~

-  `#706 <https://github.com/helmholtz-analytics/heat/pull/706>`__ Bug
   fix: prevent ``__setitem__``, ``__getitem__`` from modifying key in
   place
-  `#709 <https://github.com/helmholtz-analytics/heat/pull/709>`__ Set
   the encoding for README.md in setup.py explicitly.
-  `#716 <https://github.com/helmholtz-analytics/heat/pull/716>`__
   Bugfix: Finding clusters by spectral gap fails when multiple diffs
   identical
-  `#732 <https://github.com/helmholtz-analytics/heat/pull/732>`__
   Corrected logic in ``DNDarray.__getitem__`` to produce the correct
   split axis
-  `#734 <https://github.com/helmholtz-analytics/heat/pull/734>`__ Fix
   division by zero error in ``__local_op`` with out != None on empty
   local arrays.
-  `#735 <https://github.com/helmholtz-analytics/heat/pull/735>`__ Set
   return type to bool in relational functions.
-  `#744 <https://github.com/helmholtz-analytics/heat/pull/744>`__ Fix
   split semantics for reduction operations
-  `#756 <https://github.com/helmholtz-analytics/heat/pull/756>`__ Keep
   track of sent items while balancing within ``sort()``
-  `#764 <https://github.com/helmholtz-analytics/heat/pull/764>`__ Fixed
   an issue where ``repr`` was giving the wrong output.
-  `#767 <https://github.com/helmholtz-analytics/heat/pull/767>`__
   Corrected ``std`` to not use numpy

.. _dndarray-1:

DNDarray
~~~~~~~~

-  `#680 <https://github.com/helmholtz-analytics/heat/pull/680>`__ New
   property: ``larray``: extract local torch.Tensor
-  `#683 <https://github.com/helmholtz-analytics/heat/pull/683>`__ New
   properties: ``nbytes``, ``gnbytes``, ``lnbytes``
-  `#687 <https://github.com/helmholtz-analytics/heat/pull/687>`__ New
   property: ``balanced``

Factories
~~~~~~~~~

-  `#707 <https://github.com/helmholtz-analytics/heat/pull/707>`__ New
   feature: ``asarray()``

.. _io-1:

I/O
~~~

-  `#559 <https://github.com/helmholtz-analytics/heat/pull/559>`__
   Enhancement: ``save_netcdf`` allows naming dimensions, creating
   unlimited dimensions, using existing dimensions and variables,
   slicing

.. _linear-algebra-4:

Linear Algebra
~~~~~~~~~~~~~~

-  `#658 <https://github.com/helmholtz-analytics/heat/pull/658>`__
   Bugfix: ``matmul`` on GPU will cast away from ``int``\ s to
   ``float``\ s for the operation and cast back upon its completion.
   This may result in numerical inaccuracies for very large ``int64``
   DNDarrays

Logical
~~~~~~~

-  `#711 <https://github.com/helmholtz-analytics/heat/pull/711>`__
   ``isfinite()``, ``isinf()``, ``isnan()``
-  `#743 <https://github.com/helmholtz-analytics/heat/pull/743>`__
   ``isneginf()``, ``isposinf()``

.. _manipulations-2:

Manipulations
~~~~~~~~~~~~~

-  `#677 <https://github.com/helmholtz-analytics/heat/pull/677>`__ New
   features: ``split``, ``vsplit``, ``dsplit``, ``hsplit``
-  `#690 <https://github.com/helmholtz-analytics/heat/pull/690>`__ New
   feature: ``ravel``
-  `#690 <https://github.com/helmholtz-analytics/heat/pull/690>`__
   Enhancement: ``reshape`` accepts shape arguments with one unknown
   dimension
-  `#690 <https://github.com/helmholtz-analytics/heat/pull/690>`__
   Enhancement: reshape accepts shape arguments with one unknown
   dimension.
-  `#706 <https://github.com/helmholtz-analytics/heat/pull/706>`__ Bug
   fix: prevent ``__setitem__``, ``__getitem__`` from modifying key in
   place

Neural Networks
~~~~~~~~~~~~~~~

-  `#660 <https://github.com/helmholtz-analytics/heat/pull/660>`__ New
   submodule: ``nn.DataParallel`` for creating and training data
   parallel neural networks
-  `#660 <https://github.com/helmholtz-analytics/heat/pull/660>`__ New
   feature: Synchronous and Asynchronous gradient updates availble for
   ``ht.nn.DataParallel``
-  `#660 <https://github.com/helmholtz-analytics/heat/pull/660>`__ New
   feature: ``utils.data.datatools.DataLoader`` for created a local
   ``torch.utils.data.Dataloader`` for use with ``ht.nn.DataParallel``
-  `#660 <https://github.com/helmholtz-analytics/heat/pull/660>`__ New
   feature: ``utils.data.datatools.Dataset`` for created a local
   ``torch.utils.data.Dataset`` for use with ``ht.nn.DataParallel``
-  `#660 <https://github.com/helmholtz-analytics/heat/pull/660>`__ Added
   MNIST example to ``example/nn`` to show the use of
   ``ht.nn.DataParallel``. The ``MNISTDataset`` can be found in
   ``ht.utils.data.mnist.py``
-  `#660 <https://github.com/helmholtz-analytics/heat/pull/660>`__ New
   feature: Data loader for H5 datasets which shuffles data in the
   background during training
   (``utils.data.partial_dataset.PartialH5Dataset``)
-  `#728 <https://github.com/helmholtz-analytics/heat/pull/728>`__ New
   feature: ``nn.DataParallelMultiGPU`` which uses ``torch.distributed``
   for local communication (for use with ``optim.DASO``)
-  `#728 <https://github.com/helmholtz-analytics/heat/pull/728>`__ New
   feature: ``optim.DetectMetricPlateau`` detects when a given metric
   plateaus.

Relational
~~~~~~~~~~

-  `#792 <https://github.com/helmholtz-analytics/heat/pull/792>`__ API
   extension (aliases): ``greater``,\ ``greater_equal``, ``less``,
   ``less_equal``, ``not_equal``

Statistical Functions
~~~~~~~~~~~~~~~~~~~~~

-  `#679 <https://github.com/helmholtz-analytics/heat/pull/679>`__ New
   feature: ``histc()`` and ``histogram()``

Types
~~~~~

-  `#712 <https://github.com/helmholtz-analytics/heat/pull/712>`__ New
   function: ``issubdtype``
-  `#738 <https://github.com/helmholtz-analytics/heat/pull/738>`__
   ``iscomplex()``, ``isreal()``

.. _bug-fixes-9:

Bug fixes
---------

-  `#709 <https://github.com/helmholtz-analytics/heat/pull/709>`__ Set
   the encoding for README.md in setup.py explicitly.
-  `#716 <https://github.com/helmholtz-analytics/heat/pull/716>`__
   Bugfix: Finding clusters by spectral gap fails when multiple diffs
   identical
-  `#732 <https://github.com/helmholtz-analytics/heat/pull/732>`__
   Corrected logic in ``DNDarray.__getitem__`` to produce the correct
   split axis
-  `#734 <https://github.com/helmholtz-analytics/heat/pull/734>`__ Fix
   division by zero error in ``__local_op`` with out != None on empty
   local arrays.
-  `#735 <https://github.com/helmholtz-analytics/heat/pull/735>`__ Set
   return type to bool in relational functions.
-  `#744 <https://github.com/helmholtz-analytics/heat/pull/744>`__ Fix
   split semantics for reduction operations
-  `#756 <https://github.com/helmholtz-analytics/heat/pull/756>`__ Keep
   track of sent items while balancing within ``sort()``
-  `#764 <https://github.com/helmholtz-analytics/heat/pull/764>`__ Fixed
   an issue where ``repr`` was giving the wrong output.

Enhancements
------------

.. _manipulations-3:

Manipulations
~~~~~~~~~~~~~

-  `#690 <https://github.com/helmholtz-analytics/heat/pull/690>`__
   Enhancement: reshape accepts shape arguments with one unknown
   dimension.
-  `#706 <https://github.com/helmholtz-analytics/heat/pull/706>`__ Bug
   fix: prevent ``__setitem__``, ``__getitem__`` from modifying key in
   place ### Unit testing / CI
-  `#717 <https://github.com/helmholtz-analytics/heat/pull/717>`__
   Switch CPU CI over to Jenkins and pre-commit to GitHub action.
-  `#720 <https://github.com/helmholtz-analytics/heat/pull/720>`__
   Ignore test files in codecov report and allow drops in code coverage.
-  `#725 <https://github.com/helmholtz-analytics/heat/pull/725>`__ Add
   tests for expected warnings.
-  `#736 <https://github.com/helmholtz-analytics/heat/pull/736>`__
   Reference Jenkins CI tests and set development status to Beta.

v0.5.1
======

-  `#678 <https://github.com/helmholtz-analytics/heat/pull/678>`__
   Bugfix: Internal functions now use explicit device parameters for
   DNDarray and torch.Tensor initializations.
-  `#684 <https://github.com/helmholtz-analytics/heat/pull/684>`__ Bug
   fix: distributed ``reshape`` now works on booleans as well.

v0.5.0
======

-  `#488 <https://github.com/helmholtz-analytics/heat/pull/488>`__
   Enhancement: Rework of the test device selection.
-  `#569 <https://github.com/helmholtz-analytics/heat/pull/569>`__ New
   feature: distributed ``percentile()`` and ``median()``
-  `#572 <https://github.com/helmholtz-analytics/heat/pull/572>`__ New
   feature: distributed ``pad()``
-  `#573 <https://github.com/helmholtz-analytics/heat/pull/573>`__
   Bugfix: matmul fixes: early out for 2 vectors, remainders not added
   if inner block is 1 for split 10 case
-  `#575 <https://github.com/helmholtz-analytics/heat/pull/558>`__
   Bugfix: Binary operations use proper type casting
-  `#575 <https://github.com/helmholtz-analytics/heat/pull/558>`__
   Bugfix: ``where()`` and ``cov()`` convert ints to floats when given
   as parameters
-  `#577 <https://github.com/helmholtz-analytics/heat/pull/577>`__ Add
   ``DNDarray.ndim`` property
-  `#578 <https://github.com/helmholtz-analytics/heat/pull/578>`__
   Bugfix: Bad variable in ``reshape()``
-  `#580 <https://github.com/helmholtz-analytics/heat/pull/580>`__ New
   feature: distributed ``fliplr()``
-  `#581 <https://github.com/helmholtz-analytics/heat/pull/581>`__ New
   Feature: ``DNDarray.tolist()``
-  `#583 <https://github.com/helmholtz-analytics/heat/pull/583>`__ New
   feature: distributed ``rot90()``
-  `#593 <https://github.com/helmholtz-analytics/heat/pull/593>`__ New
   feature distributed ``arctan2()``
-  `#594 <https://github.com/helmholtz-analytics/heat/pull/594>`__ New
   feature: Advanced indexing
-  `#594 <https://github.com/helmholtz-analytics/heat/pull/594>`__
   Bugfix: distributed ``__getitem__`` and ``__setitem__`` memory
   consumption heavily reduced
-  `#596 <https://github.com/helmholtz-analytics/heat/pull/596>`__ New
   feature: distributed ``outer()``
-  `#598 <https://github.com/helmholtz-analytics/heat/pull/598>`__ Type
   casting changed to PyTorch style casting (i.e. intuitive casting)
   instead of safe casting
-  `#600 <https://github.com/helmholtz-analytics/heat/pull/600>`__ New
   feature: ``shape()``
-  `#608 <https://github.com/helmholtz-analytics/heat/pull/608>`__ New
   features: distributed ``stack()``, ``column_stack()``,
   ``row_stack()``
-  `#614 <https://github.com/helmholtz-analytics/heat/pull/614>`__ New
   feature: printing of DNDarrays and ``__repr__`` and ``__str__``
   functions
-  `#615 <https://github.com/helmholtz-analytics/heat/pull/615>`__ New
   feature: distributed ``skew()``
-  `#615 <https://github.com/helmholtz-analytics/heat/pull/615>`__ New
   feature: distributed ``kurtosis()``
-  `#618 <https://github.com/helmholtz-analytics/heat/pull/618>`__
   Printing of unbalanced DNDarrays added
-  `#620 <https://github.com/helmholtz-analytics/heat/pull/620>`__ New
   feature: distributed ``knn``
-  `#624 <https://github.com/helmholtz-analytics/heat/pull/624>`__
   Bugfix: distributed ``median()`` indexing and casting
-  `#629 <https://github.com/helmholtz-analytics/heat/pull/629>`__ New
   features: distributed ``asin``, ``acos``, ``atan``, ``atan2``
-  `#631 <https://github.com/helmholtz-analytics/heat/pull/631>`__
   Bugfix: get_halo behaviour when rank has no data.
-  `#634 <https://github.com/helmholtz-analytics/heat/pull/634>`__ New
   features: distributed ``kmedians``, ``kmedoids``, ``manhattan``
-  `#633 <https://github.com/helmholtz-analytics/heat/pull/633>`__
   Documentation: updated contributing.md
-  `#635 <https://github.com/helmholtz-analytics/heat/pull/635>`__
   ``DNDarray.__getitem__`` balances and resplits the given key to None
   if the key is a DNDarray
-  `#638 <https://github.com/helmholtz-analytics/heat/pull/638>`__ Fix:
   arange returns float32 with single input of type float & update
   skipped device tests
-  `#639 <https://github.com/helmholtz-analytics/heat/pull/639>`__
   Bugfix: balanced array in demo_knn, changed behaviour of knn
-  `#648 <https://github.com/helmholtz-analytics/heat/pull/648>`__
   Bugfix: tensor printing with PyTorch 1.6.0
-  `#651 <https://github.com/helmholtz-analytics/heat/pull/651>`__
   Bugfix: ``NotImplemented`` is now ``NotImplementedError`` in
   ``core.communication.Communication`` base class
-  `#652 <https://github.com/helmholtz-analytics/heat/pull/652>`__
   Feature: benchmark scripts and jobscript generation
-  `#653 <https://github.com/helmholtz-analytics/heat/pull/653>`__
   Printing above threshold gathers the data without a buffer now
-  `#653 <https://github.com/helmholtz-analytics/heat/pull/653>`__
   Bugfixes: Update unittests argmax & argmin + force index order in
   mpi_argmax & mpi_argmin. Add device parameter for tensor creation in
   dndarray.get_halo().
-  `#659 <https://github.com/helmholtz-analytics/heat/pull/659>`__ New
   feature: distributed ``random.permutation`` + ``random.randperm``
-  `#662 <https://github.com/helmholtz-analytics/heat/pull/662>`__
   Bugfixes: ``minimum()`` and ``maximum()`` split semantics, scalar
   input, different input dtype
-  `#664 <https://github.com/helmholtz-analytics/heat/pull/664>`__ New
   feature / enhancement: distributed ``random.random_sample``,
   ``random.random``, ``random.sample``, ``random.ranf``,
   ``random.random_integer``
-  `#666 <https://github.com/helmholtz-analytics/heat/pull/666>`__ New
   feature: distributed prepend/append for ``diff()``.
-  `#667 <https://github.com/helmholtz-analytics/heat/pull/667>`__
   Enhancement ``reshape``: rename axis parameter
-  `#678 <https://github.com/helmholtz-analytics/heat/pull/678>`__ New
   feature: distributed ``tile``
-  `#670 <https://github.com/helmholtz-analytics/heat/pull/670>`__ New
   Feature: ``bincount()``
-  `#674 <https://github.com/helmholtz-analytics/heat/pull/674>`__ New
   feature: ``repeat``
-  `#670 <https://github.com/helmholtz-analytics/heat/pull/670>`__ New
   Feature: distributed ``bincount()``
-  `#672 <https://github.com/helmholtz-analytics/heat/pull/672>`__ Bug /
   Enhancement: Remove ``MPIRequest.wait()``, rewrite calls with capital
   letters. lower case ``wait()`` now falls back to the ``mpi4py``
   function

v0.4.0
======

-  Update documentation theme to “Read the Docs”
-  `#429 <https://github.com/helmholtz-analytics/heat/pull/429>`__
   Create submodule for Linear Algebra functions
-  `#429 <https://github.com/helmholtz-analytics/heat/pull/429>`__
   Implemented QR
-  `#429 <https://github.com/helmholtz-analytics/heat/pull/429>`__
   Implemented a tiling class to create Square tiles along the diagonal
   of a 2D matrix
-  `#429 <https://github.com/helmholtz-analytics/heat/pull/429>`__ Added
   PyTorch Jitter to inner function of matmul for increased speed
-  `#483 <https://github.com/helmholtz-analytics/heat/pull/483>`__
   Bugfix: Underlying torch tensor moves to the right device on array
   initialisation
-  `#483 <https://github.com/helmholtz-analytics/heat/pull/483>`__
   Bugfix: DNDarray.cpu() changes heat device to cpu
-  `#496 <https://github.com/helmholtz-analytics/heat/pull/496>`__ New
   feature: flipud()
-  `#498 <https://github.com/helmholtz-analytics/heat/pull/498>`__
   Feature: flip()
-  `#499 <https://github.com/helmholtz-analytics/heat/pull/499>`__
   Bugfix: MPI datatype mapping: ``torch.int16`` now maps to
   ``MPI.SHORT`` instead of ``MPI.SHORT_INT``
-  `#501 <https://github.com/helmholtz-analytics/heat/pull/501>`__ New
   Feature: flatten
-  `#506 <https://github.com/helmholtz-analytics/heat/pull/506>`__
   Bugfix: setup.py has correct version parsing
-  `#507 <https://github.com/helmholtz-analytics/heat/pull/507>`__
   Bugfix: sanitize_axis changes axis of 0-dim scalars to None
-  `#511 <https://github.com/helmholtz-analytics/heat/pull/511>`__ New
   feature: reshape
-  `#515 <https://github.com/helmholtz-analytics/heat/pull/515>`__
   ht.var() now returns the unadjusted sample variance by default,
   Bessel’s correction can be applied by setting ddof=1.
-  `#518 <https://github.com/helmholtz-analytics/heat/pull/518>`__
   Implementation of Spectral Clustering.
-  `#519 <https://github.com/helmholtz-analytics/heat/pull/519>`__
   Bugfix: distributed slicing with empty list or scalar as input;
   distributed nonzero() of empty (local) tensor.
-  `#520 <https://github.com/helmholtz-analytics/heat/pull/520>`__
   Bugfix: Resplit returns correct values now.
-  `#520 <https://github.com/helmholtz-analytics/heat/pull/520>`__
   Feature: SplitTiles class, used in new resplit, tiles with
   theoretical and actual split axes
-  `#521 <https://github.com/helmholtz-analytics/heat/pull/521>`__ Add
   documentation for the dtype reduce_op in Heat’s core
-  `#522 <https://github.com/helmholtz-analytics/heat/pull/522>`__ Added
   CUDA-aware MPI detection for MVAPICH, MPICH and ParaStation.
-  `#524 <https://github.com/helmholtz-analytics/heat/pull/524>`__ New
   Feature: cumsum & cumprod
-  `#526 <https://github.com/helmholtz-analytics/heat/pull/526>`__
   float32 is now consistent default dtype for factories.
-  `#531 <https://github.com/helmholtz-analytics/heat/pull/531>`__
   Tiling objects are not separate from the DNDarray
-  `#534 <https://github.com/helmholtz-analytics/heat/pull/534>`__
   ``eye()`` supports all 2D split combinations and matrix
   configurations.
-  `#535 <https://github.com/helmholtz-analytics/heat/pull/535>`__
   Introduction of BaseEstimator and clustering, classification and
   regression mixins.
-  `#536 <https://github.com/helmholtz-analytics/heat/pull/536>`__
   Getting rid of the docs folder
-  `#541 <https://github.com/helmholtz-analytics/heat/pull/541>`__
   Introduction of basic halo scheme for inter-rank operations
-  `#558 <https://github.com/helmholtz-analytics/heat/pull/558>`__
   ``sanitize_memory_layout`` assumes default memory layout of the input
   tensor
-  `#558 <https://github.com/helmholtz-analytics/heat/pull/558>`__
   Support for PyTorch 1.5.0 added
-  `#562 <https://github.com/helmholtz-analytics/heat/pull/562>`__
   Bugfix: split semantics of ht.squeeze()
-  `#567 <https://github.com/helmholtz-analytics/heat/pull/567>`__
   Bugfix: split differences for setitem are now assumed to be correctly
   given, error will come from torch upon the setting of the value

v0.3.0
======

-  `#454 <https://github.com/helmholtz-analytics/heat/issues/454>`__
   Update lasso example
-  `#474 <https://github.com/helmholtz-analytics/heat/pull/474>`__ New
   feature: distributed Gaussian Naive Bayes classifier
-  `#473 <https://github.com/helmholtz-analytics/heat/issues/473>`__
   Matmul now will not split any of the input matrices if both have
   ``split=None``. To toggle splitting of one input for increased speed
   use the allow_resplit flag.
-  `#473 <https://github.com/helmholtz-analytics/heat/issues/473>`__
   ``dot`` handles 2 split None vectors correctly now
-  `#470 <https://github.com/helmholtz-analytics/heat/pull/470>`__
   Enhancement: Accelerate distance calculations in kmeans clustering by
   introduction of new module spatial.distance
-  `#478 <https://github.com/helmholtz-analytics/heat/pull/478>`__
   ``ht.array`` now typecasts the local torch tensors if the torch
   tensors given are not the torch version of the specified dtype + unit
   test updates
-  `#479 <https://github.com/helmholtz-analytics/heat/pull/479>`__
   Completion of spatial.distance module to support 2D input arrays of
   different splittings (None or 0) and different datatypes, also if
   second input argument is None

v0.2.2
======

This version adds support for PyTorch 1.4.0. There are also several
minor feature improvements and bug fixes listed below. -
`#443 <https://github.com/helmholtz-analytics/heat/pull/443>`__ added
option for neutral elements to be used in the place of empty tensors in
reduction operations (``operations.__reduce_op``)
(cf. `#369 <https://github.com/helmholtz-analytics/heat/issues/369>`__
and `#444 <https://github.com/helmholtz-analytics/heat/issues/444>`__) -
`#445 <https://github.com/helmholtz-analytics/heat/pull/445>`__ ``var``
and ``std`` both now support iterable axis arguments -
`#452 <https://github.com/helmholtz-analytics/heat/pull/452>`__ updated
pull request template -
`#465 <https://github.com/helmholtz-analytics/heat/pull/465>`__ bug fix:
``x.unique()`` returns a DNDarray both in distributed and
non-distributed mode (cf. [#464]) -
`#463 <https://github.com/helmholtz-analytics/heat/pull/463>`__ Bugfix:
Lasso tests now run with both GPUs and CPUs

v0.2.1
======

This version fixes the packaging, such that installed versions of HeAT
contain all required Python packages.

v0.2.0
======

This version varies greatly from the previous version (0.1.0). This
version includes a great increase in functionality and there are many
changes. Many functions which were working previously now behave more
closely to their numpy counterparts. Although a large amount of progress
has been made, work is still ongoing. We appreciate everyone who uses
this package and we work hard to solve the issues which you report to
us. Thank you!

Updated Package Requirements
----------------------------

-  python >= 3.5
-  mpi4py >= 3.0.0
-  numpy >= 1.13.0
-  torch >= 1.3.0

Optional Packages
^^^^^^^^^^^^^^^^^

-  h5py >= 2.8.0
-  netCDF4 >= 1.4.0, <= 1.5.2
-  pre-commit >= 1.18.3 (development requirement)

Additions
---------

GPU Support
~~~~~~~~~~~

`#415 <https://github.com/helmholtz-analytics/heat/pull/415>`__ GPU
support was added for this release. To set the default device use
``ht.use_device(dev)`` where ``dev`` can be either “gpu” or “cpu”. Make
sure to specify the device when creating DNDarrays if the desired device
is different than the default. If no device is specified then that
device is assumed to be “cpu”.

Basic Operations
^^^^^^^^^^^^^^^^

-  `#308 <https://github.com/helmholtz-analytics/heat/pull/308>`__
   balance
-  `#308 <https://github.com/helmholtz-analytics/heat/pull/308>`__
   convert DNDarray to numpy NDarray
-  `#412 <https://github.com/helmholtz-analytics/heat/pull/412>`__ diag
   and diagonal
-  `#388 <https://github.com/helmholtz-analytics/heat/pull/388>`__ diff
-  `#362 <https://github.com/helmholtz-analytics/heat/pull/362>`__
   distributed random numbers
-  `#327 <https://github.com/helmholtz-analytics/heat/pull/327>`__
   exponents and logarithms
-  `#423 <https://github.com/helmholtz-analytics/heat/pull/423>`__
   Fortran memory layout
-  `#330 <https://github.com/helmholtz-analytics/heat/pull/330>`__ load
   csv
-  `#326 <https://github.com/helmholtz-analytics/heat/pull/326>`__
   maximum
-  `#324 <https://github.com/helmholtz-analytics/heat/pull/324>`__
   minimum
-  `#304 <https://github.com/helmholtz-analytics/heat/pull/304>`__
   nonzero
-  `#367 <https://github.com/helmholtz-analytics/heat/pull/367>`__ prod
-  `#402 <https://github.com/helmholtz-analytics/heat/pull/402>`__ modf
-  `#428 <https://github.com/helmholtz-analytics/heat/pull/428>`__
   redistribute
-  `#345 <https://github.com/helmholtz-analytics/heat/pull/345>`__
   resplit out of place
-  `#402 <https://github.com/helmholtz-analytics/heat/pull/402>`__ round
-  `#312 <https://github.com/helmholtz-analytics/heat/pull/312>`__ sort
-  `#423 <https://github.com/helmholtz-analytics/heat/pull/423>`__
   strides
-  `#304 <https://github.com/helmholtz-analytics/heat/pull/304>`__ where

Basic Multi-DNDarray Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  `#366 <https://github.com/helmholtz-analytics/heat/pull/366>`__
   bit-wise AND, OR, and XOR
-  `#319 <https://github.com/helmholtz-analytics/heat/pull/319>`__
   concatenate
-  `#387 <https://github.com/helmholtz-analytics/heat/pull/387>`__
   hstack
-  `#387 <https://github.com/helmholtz-analytics/heat/pull/387>`__
   vstack

Developmental
^^^^^^^^^^^^^

-  Code of conduct
-  Contribution guidelines

   -  pre-commit and black checks added to Pull Requests to ensure
      proper formatting

-  Issue templates
-  `#357 <https://github.com/helmholtz-analytics/heat/pull/357>`__
   Logspace factory
-  `#428 <https://github.com/helmholtz-analytics/heat/pull/428>`__
   lshape map creation
-  Pull Request Template
-  Removal of the ml folder in favor of regression and clustering
   folders
-  `#365 <https://github.com/helmholtz-analytics/heat/pull/365>`__ Test
   suite

Linear Algebra and Statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  `#352 <https://github.com/helmholtz-analytics/heat/pull/352>`__
   average
-  `#322 <https://github.com/helmholtz-analytics/heat/pull/322>`__ dot
-  `#322 <https://github.com/helmholtz-analytics/heat/pull/322>`__ cov
-  `#286 <https://github.com/helmholtz-analytics/heat/pull/286>`__
   matmul
-  `#350 <https://github.com/helmholtz-analytics/heat/pull/350>`__ mean
   for all numerical HeAT dtypes available

Regression, Clustering, and Misc.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  `#307 <https://github.com/helmholtz-analytics/heat/pull/307>`__ lasso
   regression example
-  `#308 <https://github.com/helmholtz-analytics/heat/pull/308>`__
   kmeans scikit feature completeness
-  `#435 <https://github.com/helmholtz-analytics/heat/pull/435>`__
   Parter matrix

.. _bug-fixes-10:

Bug Fixes
=========

-  KMeans bug fixes

   -  Working in distributed mode
   -  Fixed shape cluster centers for ``init='kmeans++'``

-  \__local_op now returns proper gshape
-  allgatherv fix -> elements now sorted in the correct order
-  getitiem fixes and improvements
-  unique now returns a distributed result if the input was distributed
-  AllToAll on single process now functioning properly
-  optional packages are truly optional for running the unit tests
-  the output of mean and var (and std) now set the correct split axis
   for the returned DNDarray
