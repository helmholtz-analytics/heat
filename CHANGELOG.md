# Pending additions

## New features
- [#680](https://github.com/helmholtz-analytics/heat/pull/680) New property: larray
- [#683](https://github.com/helmholtz-analytics/heat/pull/683) New properties: nbytes, gnbytes, lnbytes
- [#687](https://github.com/helmholtz-analytics/heat/pull/687) New DNDarray property: balanced
### Manipulations
- [#677](https://github.com/helmholtz-analytics/heat/pull/677) split, vsplit, dsplit, hsplit
### Statistical Functions
- [#679](https://github.com/helmholtz-analytics/heat/pull/679) New feature: ``histc()`` and ``histogram()``
### Linear Algebra
### ...

## Bug fixes
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
- [#521](https://github.com/helmholtz-analytics/heat/pull/521) Add documentation for the generic reduce_op in Heat's core
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
