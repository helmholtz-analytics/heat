# Pending additions

- [#867](https://github.com/helmholtz-analytics/heat/pull/867) Upgraded to support torch 1.9.0
- [#876](https://github.com/helmholtz-analytics/heat/pull/876) Make examples work (Lasso and kNN)
- [#894](https://github.com/helmholtz-analytics/heat/pull/894) Change inclusion of license file
- [#884](https://github.com/helmholtz-analytics/heat/pull/884) Added capabilities for PyTorch 1.10.0, this is now the recommended version to use.

## Bug Fixes
- [#826](https://github.com/helmholtz-analytics/heat/pull/826) Fixed `__setitem__` handling of distributed `DNDarray` values which have a different shape in the split dimension
- [#846](https://github.com/helmholtz-analytics/heat/pull/846) Fixed an issue in `_reduce_op` when axis and keepdim were set.
- [#846](https://github.com/helmholtz-analytics/heat/pull/846) Fixed an issue in `min`, `max` where DNDarrays with empty processes can't be computed.
- [#868](https://github.com/helmholtz-analytics/heat/pull/868) Fixed an issue in `__binary_op` where data was falsely distributed if a DNDarray has single element.
- [#916](https://github.com/helmholtz-analytics/heat/pull/916) Fixed an issue in `random.randint` where the size parameter does not accept ints.

## Feature Additions

### Arithmetics
 - - [#887](https://github.com/helmholtz-analytics/heat/pull/887) Binary operations now support operands of equal shapes, equal `split` axes, but different distribution maps.

## Feature additions
### Communication
- [#868](https://github.com/helmholtz-analytics/heat/pull/868) New `MPICommunication` method `Split`
### DNDarray
- [#856](https://github.com/helmholtz-analytics/heat/pull/856) New `DNDarray` method `__torch_proxy__`
- [#885](https://github.com/helmholtz-analytics/heat/pull/885) New `DNDarray` method `conj`

# Feature additions
### Linear Algebra
- [#840](https://github.com/helmholtz-analytics/heat/pull/840) New feature: `vecdot()`
- [#842](https://github.com/helmholtz-analytics/heat/pull/842) New feature: `vdot`
- [#846](https://github.com/helmholtz-analytics/heat/pull/846) New features `norm`, `vector_norm`, `matrix_norm`
- [#850](https://github.com/helmholtz-analytics/heat/pull/850) New Feature `cross`
- [#877](https://github.com/helmholtz-analytics/heat/pull/877) New feature `det`
- [#875](https://github.com/helmholtz-analytics/heat/pull/875) New feature `inv`
### Logical
- [#862](https://github.com/helmholtz-analytics/heat/pull/862) New feature `signbit`
### Manipulations
- [#829](https://github.com/helmholtz-analytics/heat/pull/829) New feature: `roll`
- [#853](https://github.com/helmholtz-analytics/heat/pull/853) New Feature: `swapaxes`
- [#854](https://github.com/helmholtz-analytics/heat/pull/854) New Feature: `moveaxis`
### Printing
- [#816](https://github.com/helmholtz-analytics/heat/pull/816) New feature: Local printing (`ht.local_printing()`) and global printing options
- [#816](https://github.com/helmholtz-analytics/heat/pull/816) New feature: print only on process 0 with `print0(...)` and `ht.print0(...)`
### Random
- [#858](https://github.com/helmholtz-analytics/heat/pull/858) New Feature: `standard_normal`, `normal`
### Rounding
- [#827](https://github.com/helmholtz-analytics/heat/pull/827) New feature: `sign`, `sgn`

# v1.1.1
- [#864](https://github.com/helmholtz-analytics/heat/pull/864) Dependencies: constrain `torchvision` version range to match supported `pytorch` version range.

## Highlights
- Slicing/indexing overhaul for a more NumPy-like user experience. Warning for distributed arrays: [breaking change!](#breaking-changes) Indexing one element along the distribution axis now implies the indexed element is communicated to all processes.
- More flexibility in handling non-load-balanced distributed arrays.
- More distributed operations, incl. [meshgrid](https://github.com/helmholtz-analytics/heat/pull/794).

## Breaking Changes
- [#758](https://github.com/helmholtz-analytics/heat/pull/758) Indexing a distributed `DNDarray` along the `DNDarray.split` dimension now returns a non-distributed `DNDarray`, i.e. the indexed element is MPI-broadcasted.
Example on 2 processes:
  ```python
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
  ```

## Bug Fixes
- [#758](https://github.com/helmholtz-analytics/heat/pull/758) Fix indexing inconsistencies in `DNDarray.__getitem__()`
- [#768](https://github.com/helmholtz-analytics/heat/pull/768) Fixed an issue where `deg2rad` and `rad2deg`are not working with the 'out' parameter.
- [#785](https://github.com/helmholtz-analytics/heat/pull/785) Removed `storage_offset` when finding the mpi buffer (`communication. MPICommunication.as_mpi_memory()`).
- [#785](https://github.com/helmholtz-analytics/heat/pull/785) added allowance for 1 dimensional non-contiguous local tensors in `communication. MPICommunication.mpi_type_and_elements_of()`
- [#787](https://github.com/helmholtz-analytics/heat/pull/787) Fixed an issue where Heat cannot be imported when some optional dependencies are not available.
- [#790](https://github.com/helmholtz-analytics/heat/pull/790) catch incorrect device after `bcast` in `DNDarray.__getitem__`
- [#796](https://github.com/helmholtz-analytics/heat/pull/796) `heat.reshape(a, shape, new_split)` now always returns a distributed `DNDarray` if `new_split is not None` (inlcuding when the original input `a` is not distributed)
- [#811](https://github.com/helmholtz-analytics/heat/pull/811) Fixed memory leak in `DNDarray.larray`
- [#820](https://github.com/helmholtz-analytics/heat/pull/820) `randn` values are pushed away from 0 by the minimum value the given dtype before being transformed into the Gaussian shape
- [#821](https://github.com/helmholtz-analytics/heat/pull/821) Fixed `__getitem__` handling of distributed `DNDarray` key element
- [#831](https://github.com/helmholtz-analytics/heat/pull/831) `__getitem__` handling of `array-like` 1-element key

## Feature additions
### Exponential
- [#812](https://github.com/helmholtz-analytics/heat/pull/712) New feature: `logaddexp`, `logaddexp2`

### Linear Algebra
- [#718](https://github.com/helmholtz-analytics/heat/pull/718) New feature: `trace()`
- [#768](https://github.com/helmholtz-analytics/heat/pull/768) New feature: unary positive and negative operations
- [#820](https://github.com/helmholtz-analytics/heat/pull/820) `dot` can handle matrix-vector operation now

### Manipulations
- [#796](https://github.com/helmholtz-analytics/heat/pull/796) `DNDarray.reshape(shape)`: method now allows shape elements to be passed in as single arguments.

### Trigonometrics / Arithmetic
- [#806](https://github.com/helmholtz-analytics/heat/pull/809) New feature: `square`
- [#809](https://github.com/helmholtz-analytics/heat/pull/809) New feature: `acosh`, `asinh`, `atanh`

### Misc.
- [#761](https://github.com/helmholtz-analytics/heat/pull/761) New feature: `result_type`
- [#794](https://github.com/helmholtz-analytics/heat/pull/794) New feature: `meshgrid`
- [#821](https://github.com/helmholtz-analytics/heat/pull/821) Enhancement: it is no longer necessary to load-balance an imbalanced `DNDarray` before gathering it onto all processes. In short: `ht.resplit(array, None)` now works on imbalanced arrays as well.

# v1.0.0

## New features / Highlights
- [#660](https://github.com/helmholtz-analytics/heat/pull/660) NN module for data parallel neural networks
- [#699](https://github.com/helmholtz-analytics/heat/pull/699) Support for complex numbers; New functions: `angle`, `real`, `imag`, `conjugate`
- [#702](https://github.com/helmholtz-analytics/heat/pull/702) Support channel stackoverflow
- [#728](https://github.com/helmholtz-analytics/heat/pull/728) `DASO` optimizer
- [#757](https://github.com/helmholtz-analytics/heat/pull/757) Major documentation overhaul, custom docstrings formatting

### Bug fixes
- [#706](https://github.com/helmholtz-analytics/heat/pull/706) Bug fix: prevent `__setitem__`, `__getitem__` from modifying key in place
- [#709](https://github.com/helmholtz-analytics/heat/pull/709) Set the encoding for README.md in setup.py explicitly.
- [#716](https://github.com/helmholtz-analytics/heat/pull/716) Bugfix: Finding clusters by spectral gap fails when multiple diffs identical
- [#732](https://github.com/helmholtz-analytics/heat/pull/732) Corrected logic in `DNDarray.__getitem__` to produce the correct split axis
- [#734](https://github.com/helmholtz-analytics/heat/pull/734) Fix division by zero error in `__local_op` with out != None on empty local arrays.
- [#735](https://github.com/helmholtz-analytics/heat/pull/735) Set return type to bool in relational functions.
- [#744](https://github.com/helmholtz-analytics/heat/pull/744) Fix split semantics for reduction operations
- [#756](https://github.com/helmholtz-analytics/heat/pull/756) Keep track of sent items while balancing within `sort()`
- [#764](https://github.com/helmholtz-analytics/heat/pull/764) Fixed an issue where `repr` was giving the wrong output.
- [#767](https://github.com/helmholtz-analytics/heat/pull/767) Corrected `std` to not use numpy

### DNDarray
- [#680](https://github.com/helmholtz-analytics/heat/pull/680) New property: `larray`: extract local torch.Tensor
- [#683](https://github.com/helmholtz-analytics/heat/pull/683) New properties: `nbytes`, `gnbytes`, `lnbytes`
- [#687](https://github.com/helmholtz-analytics/heat/pull/687) New property: `balanced`

### Factories
- [#707](https://github.com/helmholtz-analytics/heat/pull/707) New feature: `asarray()`

### I/O
- [#559](https://github.com/helmholtz-analytics/heat/pull/559) Enhancement: `save_netcdf` allows naming dimensions, creating unlimited dimensions, using existing dimensions and variables, slicing

### Linear Algebra
- [#658](https://github.com/helmholtz-analytics/heat/pull/658) Bugfix: `matmul` on GPU will cast away from `int`s to `float`s for the operation and cast back upon its completion. This may result in numerical inaccuracies for very large `int64` DNDarrays

### Logical
- [#711](https://github.com/helmholtz-analytics/heat/pull/711) `isfinite()`, `isinf()`, `isnan()`
- [#743](https://github.com/helmholtz-analytics/heat/pull/743) `isneginf()`, `isposinf()`

### Manipulations
- [#677](https://github.com/helmholtz-analytics/heat/pull/677) New features: `split`, `vsplit`, `dsplit`, `hsplit`
- [#690](https://github.com/helmholtz-analytics/heat/pull/690) New feature: `ravel`
- [#690](https://github.com/helmholtz-analytics/heat/pull/690) Enhancement: `reshape` accepts shape arguments with one unknown dimension
- [#690](https://github.com/helmholtz-analytics/heat/pull/690) Enhancement: reshape accepts shape arguments with one unknown dimension.
- [#706](https://github.com/helmholtz-analytics/heat/pull/706) Bug fix: prevent `__setitem__`, `__getitem__` from modifying key in place

### Neural Networks
- [#660](https://github.com/helmholtz-analytics/heat/pull/660) New submodule: `nn.DataParallel` for creating and training data parallel neural networks
- [#660](https://github.com/helmholtz-analytics/heat/pull/660) New feature: Synchronous and Asynchronous gradient updates availble for `ht.nn.DataParallel`
- [#660](https://github.com/helmholtz-analytics/heat/pull/660) New feature: `utils.data.datatools.DataLoader` for created a local `torch.utils.data.Dataloader` for use with `ht.nn.DataParallel`
- [#660](https://github.com/helmholtz-analytics/heat/pull/660) New feature: `utils.data.datatools.Dataset` for created a local `torch.utils.data.Dataset` for use with `ht.nn.DataParallel`
- [#660](https://github.com/helmholtz-analytics/heat/pull/660) Added MNIST example to `example/nn` to show the use of `ht.nn.DataParallel`. The `MNISTDataset` can be found in `ht.utils.data.mnist.py`
- [#660](https://github.com/helmholtz-analytics/heat/pull/660) New feature: Data loader for H5 datasets which shuffles data in the background during training (`utils.data.partial_dataset.PartialH5Dataset`)
- [#728](https://github.com/helmholtz-analytics/heat/pull/728) New feature: `nn.DataParallelMultiGPU` which uses `torch.distributed` for local communication (for use with `optim.DASO`)
- [#728](https://github.com/helmholtz-analytics/heat/pull/728) New feature: `optim.DetectMetricPlateau` detects when a given metric plateaus.

### Relational
- [#792](https://github.com/helmholtz-analytics/heat/pull/792) API extension (aliases): `greater`,`greater_equal`, `less`, `less_equal`, `not_equal`

### Statistical Functions
- [#679](https://github.com/helmholtz-analytics/heat/pull/679) New feature: ``histc()`` and ``histogram()``

### Types
- [#712](https://github.com/helmholtz-analytics/heat/pull/712) New function: `issubdtype`
- [#738](https://github.com/helmholtz-analytics/heat/pull/738) `iscomplex()`, `isreal()`


## Bug fixes
- [#709](https://github.com/helmholtz-analytics/heat/pull/709) Set the encoding for README.md in setup.py explicitly.
- [#716](https://github.com/helmholtz-analytics/heat/pull/716) Bugfix: Finding clusters by spectral gap fails when multiple diffs identical
- [#732](https://github.com/helmholtz-analytics/heat/pull/732) Corrected logic in `DNDarray.__getitem__` to produce the correct split axis
- [#734](https://github.com/helmholtz-analytics/heat/pull/734) Fix division by zero error in `__local_op` with out != None on empty local arrays.
- [#735](https://github.com/helmholtz-analytics/heat/pull/735) Set return type to bool in relational functions.
- [#744](https://github.com/helmholtz-analytics/heat/pull/744) Fix split semantics for reduction operations
- [#756](https://github.com/helmholtz-analytics/heat/pull/756) Keep track of sent items while balancing within `sort()`
- [#764](https://github.com/helmholtz-analytics/heat/pull/764) Fixed an issue where `repr` was giving the wrong output.

## Enhancements
### Manipulations
- [#690](https://github.com/helmholtz-analytics/heat/pull/690) Enhancement: reshape accepts shape arguments with one unknown dimension.
- [#706](https://github.com/helmholtz-analytics/heat/pull/706) Bug fix: prevent `__setitem__`, `__getitem__` from modifying key in place
### Unit testing / CI
- [#717](https://github.com/helmholtz-analytics/heat/pull/717) Switch CPU CI over to Jenkins and pre-commit to GitHub action.
- [#720](https://github.com/helmholtz-analytics/heat/pull/720) Ignore test files in codecov report and allow drops in code coverage.
- [#725](https://github.com/helmholtz-analytics/heat/pull/725) Add tests for expected warnings.
- [#736](https://github.com/helmholtz-analytics/heat/pull/736) Reference Jenkins CI tests and set development status to Beta.

# v0.5.1

- [#678](https://github.com/helmholtz-analytics/heat/pull/678) Bugfix: Internal functions now use explicit device parameters for DNDarray and torch.Tensor initializations.
- [#684](https://github.com/helmholtz-analytics/heat/pull/684) Bug fix: distributed `reshape` now works on booleans as well.

# v0.5.0

- [#488](https://github.com/helmholtz-analytics/heat/pull/488) Enhancement: Rework of the test device selection.
- [#569](https://github.com/helmholtz-analytics/heat/pull/569) New feature: distributed `percentile()` and `median()`
- [#572](https://github.com/helmholtz-analytics/heat/pull/572) New feature: distributed `pad()`
- [#573](https://github.com/helmholtz-analytics/heat/pull/573) Bugfix: matmul fixes: early out for 2 vectors, remainders not added if inner block is 1 for split 10 case
- [#575](https://github.com/helmholtz-analytics/heat/pull/558) Bugfix: Binary operations use proper type casting
- [#575](https://github.com/helmholtz-analytics/heat/pull/558) Bugfix: ``where()`` and ``cov()`` convert ints to floats when given as parameters
- [#577](https://github.com/helmholtz-analytics/heat/pull/577) Add ``DNDarray.ndim`` property
- [#578](https://github.com/helmholtz-analytics/heat/pull/578) Bugfix: Bad variable in ``reshape()``
- [#580](https://github.com/helmholtz-analytics/heat/pull/580) New feature: distributed ``fliplr()``
- [#581](https://github.com/helmholtz-analytics/heat/pull/581) New Feature: ``DNDarray.tolist()``
- [#583](https://github.com/helmholtz-analytics/heat/pull/583) New feature: distributed ``rot90()``
- [#593](https://github.com/helmholtz-analytics/heat/pull/593) New feature distributed ``arctan2()``
- [#594](https://github.com/helmholtz-analytics/heat/pull/594) New feature: Advanced indexing
- [#594](https://github.com/helmholtz-analytics/heat/pull/594) Bugfix: distributed ``__getitem__`` and ``__setitem__`` memory consumption heavily reduced
- [#596](https://github.com/helmholtz-analytics/heat/pull/596) New feature: distributed ``outer()``
- [#598](https://github.com/helmholtz-analytics/heat/pull/598) Type casting changed to PyTorch style casting (i.e. intuitive casting) instead of safe casting
- [#600](https://github.com/helmholtz-analytics/heat/pull/600) New feature: ``shape()``
- [#608](https://github.com/helmholtz-analytics/heat/pull/608) New features: distributed ``stack()``, ``column_stack()``, ``row_stack()``
- [#614](https://github.com/helmholtz-analytics/heat/pull/614) New feature: printing of DNDarrays and ``__repr__`` and ``__str__`` functions
- [#615](https://github.com/helmholtz-analytics/heat/pull/615) New feature: distributed `skew()`
- [#615](https://github.com/helmholtz-analytics/heat/pull/615) New feature: distributed `kurtosis()`
- [#618](https://github.com/helmholtz-analytics/heat/pull/618) Printing of unbalanced DNDarrays added
- [#620](https://github.com/helmholtz-analytics/heat/pull/620) New feature: distributed `knn`
- [#624](https://github.com/helmholtz-analytics/heat/pull/624) Bugfix: distributed `median()` indexing and casting
- [#629](https://github.com/helmholtz-analytics/heat/pull/629) New features: distributed `asin`, `acos`, `atan`, `atan2`
- [#631](https://github.com/helmholtz-analytics/heat/pull/631) Bugfix: get_halo behaviour when rank has no data.
- [#634](https://github.com/helmholtz-analytics/heat/pull/634) New features: distributed `kmedians`, `kmedoids`, `manhattan`
- [#633](https://github.com/helmholtz-analytics/heat/pull/633) Documentation: updated contributing.md
- [#635](https://github.com/helmholtz-analytics/heat/pull/635) `DNDarray.__getitem__` balances and resplits the given key to None if the key is a DNDarray
- [#638](https://github.com/helmholtz-analytics/heat/pull/638) Fix: arange returns float32 with single input of type float & update skipped device tests
- [#639](https://github.com/helmholtz-analytics/heat/pull/639) Bugfix: balanced array in demo_knn, changed behaviour of knn
- [#648](https://github.com/helmholtz-analytics/heat/pull/648) Bugfix: tensor printing with PyTorch 1.6.0
- [#651](https://github.com/helmholtz-analytics/heat/pull/651) Bugfix: `NotImplemented` is now `NotImplementedError` in `core.communication.Communication` base class
- [#652](https://github.com/helmholtz-analytics/heat/pull/652) Feature: benchmark scripts and jobscript generation
- [#653](https://github.com/helmholtz-analytics/heat/pull/653) Printing above threshold gathers the data without a buffer now
- [#653](https://github.com/helmholtz-analytics/heat/pull/653) Bugfixes: Update unittests argmax & argmin + force index order in mpi_argmax & mpi_argmin. Add device parameter for tensor creation in dndarray.get_halo().
- [#659](https://github.com/helmholtz-analytics/heat/pull/659) New feature: distributed `random.permutation` + `random.randperm`
- [#662](https://github.com/helmholtz-analytics/heat/pull/662) Bugfixes: `minimum()` and `maximum()` split semantics, scalar input, different input dtype
- [#664](https://github.com/helmholtz-analytics/heat/pull/664) New feature / enhancement: distributed `random.random_sample`, `random.random`, `random.sample`, `random.ranf`, `random.random_integer`
- [#666](https://github.com/helmholtz-analytics/heat/pull/666) New feature: distributed prepend/append for `diff()`.
- [#667](https://github.com/helmholtz-analytics/heat/pull/667) Enhancement `reshape`: rename axis parameter
- [#678](https://github.com/helmholtz-analytics/heat/pull/678) New feature: distributed `tile`
- [#670](https://github.com/helmholtz-analytics/heat/pull/670) New Feature: `bincount()`
- [#674](https://github.com/helmholtz-analytics/heat/pull/674) New feature: `repeat`
- [#670](https://github.com/helmholtz-analytics/heat/pull/670) New Feature: distributed `bincount()`
- [#672](https://github.com/helmholtz-analytics/heat/pull/672) Bug / Enhancement: Remove `MPIRequest.wait()`, rewrite calls with capital letters. lower case `wait()` now falls back to the `mpi4py` function

# v0.4.0

- Update documentation theme to "Read the Docs"
- [#429](https://github.com/helmholtz-analytics/heat/pull/429) Create submodule for Linear Algebra functions
- [#429](https://github.com/helmholtz-analytics/heat/pull/429) Implemented QR
- [#429](https://github.com/helmholtz-analytics/heat/pull/429) Implemented a tiling class to create Square tiles along the diagonal of a 2D matrix
- [#429](https://github.com/helmholtz-analytics/heat/pull/429) Added PyTorch Jitter to inner function of matmul for increased speed
- [#483](https://github.com/helmholtz-analytics/heat/pull/483) Bugfix: Underlying torch tensor moves to the right device on array initialisation
- [#483](https://github.com/helmholtz-analytics/heat/pull/483) Bugfix: DNDarray.cpu() changes heat device to cpu
- [#496](https://github.com/helmholtz-analytics/heat/pull/496) New feature: flipud()
- [#498](https://github.com/helmholtz-analytics/heat/pull/498) Feature: flip()
- [#499](https://github.com/helmholtz-analytics/heat/pull/499) Bugfix: MPI datatype mapping: `torch.int16` now maps to `MPI.SHORT` instead of `MPI.SHORT_INT`
- [#501](https://github.com/helmholtz-analytics/heat/pull/501) New Feature: flatten
- [#506](https://github.com/helmholtz-analytics/heat/pull/506) Bugfix: setup.py has correct version parsing
- [#507](https://github.com/helmholtz-analytics/heat/pull/507) Bugfix: sanitize_axis changes axis of 0-dim scalars to None
- [#511](https://github.com/helmholtz-analytics/heat/pull/511) New feature: reshape
- [#515](https://github.com/helmholtz-analytics/heat/pull/515) ht.var() now returns the unadjusted sample variance by default, Bessel's correction can be applied by setting ddof=1.
- [#518](https://github.com/helmholtz-analytics/heat/pull/518) Implementation of Spectral Clustering.
- [#519](https://github.com/helmholtz-analytics/heat/pull/519) Bugfix: distributed slicing with empty list or scalar as input; distributed nonzero() of empty (local) tensor.
- [#520](https://github.com/helmholtz-analytics/heat/pull/520) Bugfix: Resplit returns correct values now.
- [#520](https://github.com/helmholtz-analytics/heat/pull/520) Feature: SplitTiles class, used in new resplit, tiles with theoretical and actual split axes
- [#521](https://github.com/helmholtz-analytics/heat/pull/521) Add documentation for the dtype reduce_op in Heat's core
- [#522](https://github.com/helmholtz-analytics/heat/pull/522) Added CUDA-aware MPI detection for MVAPICH, MPICH and ParaStation.
- [#524](https://github.com/helmholtz-analytics/heat/pull/524) New Feature: cumsum & cumprod
- [#526](https://github.com/helmholtz-analytics/heat/pull/526) float32 is now consistent default dtype for factories.
- [#531](https://github.com/helmholtz-analytics/heat/pull/531) Tiling objects are not separate from the DNDarray
- [#534](https://github.com/helmholtz-analytics/heat/pull/534) `eye()` supports all 2D split combinations and matrix configurations.
- [#535](https://github.com/helmholtz-analytics/heat/pull/535) Introduction of BaseEstimator and clustering, classification and regression mixins.
- [#536](https://github.com/helmholtz-analytics/heat/pull/536) Getting rid of the docs folder
- [#541](https://github.com/helmholtz-analytics/heat/pull/541) Introduction of basic halo scheme for inter-rank operations
- [#558](https://github.com/helmholtz-analytics/heat/pull/558) `sanitize_memory_layout` assumes default memory layout of the input tensor
- [#558](https://github.com/helmholtz-analytics/heat/pull/558) Support for PyTorch 1.5.0 added
- [#562](https://github.com/helmholtz-analytics/heat/pull/562) Bugfix: split semantics of ht.squeeze()
- [#567](https://github.com/helmholtz-analytics/heat/pull/567) Bugfix: split differences for setitem are now assumed to be correctly given, error will come from torch upon the setting of the value

# v0.3.0

- [#454](https://github.com/helmholtz-analytics/heat/issues/454) Update lasso example
- [#474](https://github.com/helmholtz-analytics/heat/pull/474) New feature: distributed Gaussian Naive Bayes classifier
- [#473](https://github.com/helmholtz-analytics/heat/issues/473) Matmul now will not split any of the input matrices if both have `split=None`. To toggle splitting of one input for increased speed use the allow_resplit flag.
- [#473](https://github.com/helmholtz-analytics/heat/issues/473) `dot` handles 2 split None vectors correctly now
- [#470](https://github.com/helmholtz-analytics/heat/pull/470) Enhancement: Accelerate distance calculations in kmeans clustering by introduction of new module spatial.distance
- [#478](https://github.com/helmholtz-analytics/heat/pull/478) `ht.array` now typecasts the local torch tensors if the torch tensors given are not the torch version of the specified dtype + unit test updates
- [#479](https://github.com/helmholtz-analytics/heat/pull/479) Completion of spatial.distance module to support 2D input arrays of different splittings (None or 0) and different datatypes, also if second input argument is None

# v0.2.2

This version adds support for PyTorch 1.4.0. There are also several minor feature improvements and bug fixes listed below.
- [#443](https://github.com/helmholtz-analytics/heat/pull/443) added option for neutral elements to be used in the place of empty tensors in reduction operations (`operations.__reduce_op`) (cf. [#369](https://github.com/helmholtz-analytics/heat/issues/369) and [#444](https://github.com/helmholtz-analytics/heat/issues/444))
- [#445](https://github.com/helmholtz-analytics/heat/pull/445) `var` and `std` both now support iterable axis arguments
- [#452](https://github.com/helmholtz-analytics/heat/pull/452) updated pull request template
- [#465](https://github.com/helmholtz-analytics/heat/pull/465) bug fix: `x.unique()` returns a DNDarray both in distributed and non-distributed mode (cf. [#464])
- [#463](https://github.com/helmholtz-analytics/heat/pull/463) Bugfix: Lasso tests now run with both GPUs and CPUs

# v0.2.1

This version fixes the packaging, such that installed versions of HeAT contain all required Python packages.

# v0.2.0

This version varies greatly from the previous version (0.1.0). This version includes a great increase in
functionality and there are many changes. Many functions which were working previously now behave more closely
to their numpy counterparts. Although a large amount of progress has been made, work is still ongoing. We
appreciate everyone who uses this package and we work hard to solve the issues which you report to us. Thank you!

## Updated Package Requirements
- python >= 3.5
- mpi4py >= 3.0.0
- numpy >= 1.13.0
- torch >= 1.3.0

#### Optional Packages
- h5py >= 2.8.0
- netCDF4 >= 1.4.0, <= 1.5.2
- pre-commit >= 1.18.3 (development requirement)

## Additions

### GPU Support
[#415](https://github.com/helmholtz-analytics/heat/pull/415) GPU support was added for this release. To set the default device use `ht.use_device(dev)` where `dev` can be either
"gpu" or "cpu". Make sure to specify the device when creating DNDarrays if the desired device is different than the
default. If no device is specified then that device is assumed to be "cpu".

#### Basic Operations
- [#308](https://github.com/helmholtz-analytics/heat/pull/308) balance
- [#308](https://github.com/helmholtz-analytics/heat/pull/308) convert DNDarray to numpy NDarray
- [#412](https://github.com/helmholtz-analytics/heat/pull/412) diag and diagonal
- [#388](https://github.com/helmholtz-analytics/heat/pull/388) diff
- [#362](https://github.com/helmholtz-analytics/heat/pull/362) distributed random numbers
- [#327](https://github.com/helmholtz-analytics/heat/pull/327) exponents and logarithms
- [#423](https://github.com/helmholtz-analytics/heat/pull/423) Fortran memory layout
- [#330](https://github.com/helmholtz-analytics/heat/pull/330) load csv
- [#326](https://github.com/helmholtz-analytics/heat/pull/326) maximum
- [#324](https://github.com/helmholtz-analytics/heat/pull/324) minimum
- [#304](https://github.com/helmholtz-analytics/heat/pull/304) nonzero
- [#367](https://github.com/helmholtz-analytics/heat/pull/367) prod
- [#402](https://github.com/helmholtz-analytics/heat/pull/402) modf
- [#428](https://github.com/helmholtz-analytics/heat/pull/428) redistribute
- [#345](https://github.com/helmholtz-analytics/heat/pull/345) resplit out of place
- [#402](https://github.com/helmholtz-analytics/heat/pull/402) round
- [#312](https://github.com/helmholtz-analytics/heat/pull/312) sort
- [#423](https://github.com/helmholtz-analytics/heat/pull/423) strides
- [#304](https://github.com/helmholtz-analytics/heat/pull/304) where

#### Basic Multi-DNDarray Operations
- [#366](https://github.com/helmholtz-analytics/heat/pull/366) bit-wise AND, OR, and XOR
- [#319](https://github.com/helmholtz-analytics/heat/pull/319) concatenate
- [#387](https://github.com/helmholtz-analytics/heat/pull/387) hstack
- [#387](https://github.com/helmholtz-analytics/heat/pull/387) vstack

#### Developmental
- Code of conduct
- Contribution guidelines
    * pre-commit and black checks added to Pull Requests to ensure proper formatting
- Issue templates
- [#357](https://github.com/helmholtz-analytics/heat/pull/357) Logspace factory
- [#428](https://github.com/helmholtz-analytics/heat/pull/428) lshape map creation
- Pull Request Template
- Removal of the ml folder in favor of regression and clustering folders
- [#365](https://github.com/helmholtz-analytics/heat/pull/365) Test suite

#### Linear Algebra and Statistics
- [#352](https://github.com/helmholtz-analytics/heat/pull/352) average
- [#322](https://github.com/helmholtz-analytics/heat/pull/322) dot
- [#322](https://github.com/helmholtz-analytics/heat/pull/322) cov
- [#286](https://github.com/helmholtz-analytics/heat/pull/286) matmul
- [#350](https://github.com/helmholtz-analytics/heat/pull/350) mean for all numerical HeAT dtypes available

#### Regression, Clustering, and Misc.
- [#307](https://github.com/helmholtz-analytics/heat/pull/307) lasso regression example
- [#308](https://github.com/helmholtz-analytics/heat/pull/308) kmeans scikit feature completeness
- [#435](https://github.com/helmholtz-analytics/heat/pull/435) Parter matrix

# Bug Fixes

- KMeans bug fixes
    * Working in distributed mode
    * Fixed shape cluster centers for `init='kmeans++'`
- __local_op now returns proper gshape
- allgatherv fix -> elements now sorted in the correct order
- getitiem fixes and improvements
- unique now returns a distributed result if the input was distributed
- AllToAll on single process now functioning properly
- optional packages are truly optional for running the unit tests
- the output of mean and var (and std) now set the correct split axis for the returned DNDarray
