# # Loading and Preprocessing
#
# ### Refresher
#
# Using PyTorch as compute engine and mpi4py for communication, Heat implements a number of array operations and algorithms that are optimized for memory-distributed data volumes. This allows you to tackle datasets that are too large for single-node (or worse, single-GPU) processing.
#
# As opposed to task-parallel frameworks, Heat takes a data-parallel approach, meaning that each "worker" or MPI process performs the same tasks on different slices of the data. Many operations and algorithms are not embarassingly parallel, and involve data exchange between processes. Heat operations and algorithms are designed to minimize this communication overhead, and to make it transparent to the user.
#
# In other words:
# - you don't have to worry about optimizing data chunk sizes;
# - you don't have to make sure your research problem is embarassingly parallel, or artificially make your dataset smaller so your RAM is sufficient;
# - you do have to make sure that you have sufficient **overall** RAM to run your global task (e.g. number of nodes / GPUs).

# The following shows some I/O and preprocessing examples. We'll use small datasets here as each of us only has access to one node only.

# ### I/O
#
# Let's start with loading a data set. Heat supports reading and writing from/into shared memory for a number of formats, including HDF5, NetCDF, and because we love scientists, csv. Check out the `ht.load` and `ht.save` functions for more details. Here we will load data in [HDF5 format](https://en.wikipedia.org/wiki/Hierarchical_Data_Format).
#
# This particular example data set (generated from all Asteroids from the [JPL Small Body Database](https://ssd.jpl.nasa.gov/sb/)) is really small, but it allows to demonstrate the basic functionality of Heat.
#

# The above cell should return [0, 1, 2, 3].
#
# Now let's import `heat` and load the data set.

import heat as ht

# X = ht.load_hdf5("../data/sbdb_asteroids.h5",dtype=ht.float64,dataset="data",split=0)

# Some random data for small scale tests
X = ht.random.randn(1000, 3, split=0)

# We have loaded the entire data onto 4 MPI processes, each with 12 cores. We have created `X` with `split=0`, so each process stores evenly-sized slices of the data along dimension 0.

# ### Data exploration
#
# Let's get an idea of the size of the data.


# print global metadata once only
if X.comm.rank == 0:
    print(f"X is a {X.ndim}-dimensional array with shape{X.shape}")
    print(f"X takes up {X.nbytes/1e6} MB of memory.")

# X is a matrix of shape *(datapoints, features)*.
#
# To get a first overview, we can print the data and determine its feature-wise mean, variance, min, max etc. These are reduction operations along the datapoints dimension, which is also the `split` dimension. You don't have to implement [`MPI.Allreduce`](https://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/) operations yourself, communication is handled by Heat operations.


features_mean = ht.mean(X, axis=0)
features_var = ht.var(X, axis=0)
features_max = ht.max(X, axis=0)
features_min = ht.min(X, axis=0)
# ht.percentile is buggy, see #1389, we'll leave it out for now
# features_median = ht.percentile(X,50.,axis=0)


if ht.MPI_WORLD.rank == 0:
    print(f"Mean: {features_mean}")
    print(f"Var: {features_var}")
    print(f"Max: {features_max}")
    print(f"Min: {features_min}")


# Note that the `features_...` DNDarrays are no longer distributed, i.e. a copy of these results exists on each GPU, as the split dimension of the input data has been lost in the reduction operations.

# ### Preprocessing/scaling
#
# Next, we can preprocess the data, e.g., by standardizing and/or normalizing. Heat offers several preprocessing routines for doing so, the API is similar to [`sklearn.preprocessing`](https://scikit-learn.org/stable/modules/preprocessing.html) so adapting existing code shouldn't be too complicated.
#
# Again, please let us know if you're missing any features.


# Standard Scaler
scaler = ht.preprocessing.StandardScaler()
X_standardized = scaler.fit_transform(X)
standardized_mean = ht.mean(X_standardized, axis=0)
standardized_var = ht.var(X_standardized, axis=0)

if ht.MPI_WORLD.rank == 0:
    print(f"Standard Scaler Mean: {standardized_mean}")
    print(f"Standard Scaler Var: {standardized_var}")

# Robust Scaler
scaler = ht.preprocessing.RobustScaler()
X_robust = scaler.fit_transform(X)
robust_mean = ht.mean(X_robust, axis=0)
robust_var = ht.var(X_robust, axis=0)

if ht.MPI_WORLD.rank == 0:
    print(f"Robust Scaler Mean: {robust_mean}")
    print(f"Robust Scaler Median: {robust_var}")


# Within Heat, you have several options to apply memory-distributed machine learning algorithms on your data.
#
# Is the algorithm you're looking for not yet implemented? [Let us know](https://github.com/helmholtz-analytics/heat/issues/new/choose)!
