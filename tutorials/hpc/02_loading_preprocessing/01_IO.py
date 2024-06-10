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
# Now let's import `heat` and load a data set.

import heat as ht

# Some random data for small scale tests
iris = ht.load("iris.csv", sep=";", split=0)
print(iris)

# We have loaded the entire data onto 4 MPI processes, each with 12 cores. We have created `X` with `split=0`, so each process stores evenly-sized slices of the data along dimension 0.

# similar for HDF5
X = ht.load_hdf5("path_to_data/sbdb_asteroids.h5", device="gpu", dataset="data", split=0)
print(X.shape)
