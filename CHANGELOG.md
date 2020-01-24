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
