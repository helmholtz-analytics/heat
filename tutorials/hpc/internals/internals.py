import heat as ht
import torch

# # Heat as infrastructure for MPI applications
#
# In this section, we'll go through some Heat-specific functionalities that simplify the implementation of a data-parallel application in Python. We'll demonstrate them on small arrays and 4 processes on a single cluster node, but the functionalities are indeed meant for a multi-node set up with huge arrays that cannot be processed on a single node.


# We already mentioned that the DNDarray object is "MPI-aware". Each DNDarray is associated to an MPI communicator, it is aware of the number of processes in the communicator, and it knows the rank of the process that owns it.
#

a = ht.random.randn(7, 4, 3, split=0)
if a.comm.rank == 0:
    print(f"a.com gets the communicator {a.comm} associated with DNDarray a")

# MPI size = total number of processes
size = a.comm.size

if a.comm.rank == 0:
    print(f"a is distributed over {size} processes")
    print(f"a is a distributed {a.ndim}-dimensional array with global shape {a.shape}")


# MPI rank = rank of each process
rank = a.comm.rank
# Local shape = shape of the data on each process
local_shape = a.lshape
print(f"Rank {rank} holds a slice of a with local shape {local_shape}")


# ### Distribution map
#
# In many occasions, when building a memory-distributed pipeline it will be convenient for each rank to have information on what ranks holds which slice of the distributed array.
#
# The `lshape_map` attribute of a DNDarray gathers (or, if possible, calculates) this info from all processes and stores it as metadata of the DNDarray. Because it is meant for internal use, it is stored in a torch tensor, not a DNDarray.
#
# The `lshape_map` tensor is a 2D tensor, where the first dimension is the number of processes and the second dimension is the number of dimensions of the array. Each row of the tensor contains the local shape of the array on a process.


lshape_map = a.lshape_map
if a.comm.rank == 0:
    print(f"lshape_map available on any process: {lshape_map}")

# Go back to where we created the DNDarray and and create `a` with a different split axis. See how the `lshape_map` changes.

# ### Modifying the DNDarray distribution
#
# In a distributed pipeline, it is sometimes necessary to change the distribution of a DNDarray, when the array is not distributed in the most convenient way for the next operation / algorithm.
#
# Depending on your needs, you can choose between:
# - `DNDarray.redistribute_()`: This method keeps the original split axis, but redistributes the data of the DNDarray according to a "target map".
# - `DNDarray.resplit_()`: This method changes the split axis of the DNDarray. This is a more expensive operation, and should be used only when absolutely necessary. Depending on your needs and available resources, in some cases it might be wiser to keep a copy of the DNDarray with a different split axis.
#
# Let's see some examples.


# redistribute
target_map = a.lshape_map
target_map[:, a.split] = torch.tensor([1, 2, 2, 2])
# in-place redistribution (see ht.redistribute for out-of-place)
a.redistribute_(target_map=target_map)

# new lshape map after redistribution
a.lshape_map

# local arrays after redistribution
a.larray


# resplit
a.resplit_(axis=1)

a.lshape_map


# You can use the `resplit_` method (in-place), or `ht.resplit` (out-of-place) to change the distribution axis, but also to set the distribution axis to None. The latter corresponds to an MPI.Allgather operation that gathers the entire array on each process. This is useful when you've achieved a small enough data size that can be processed on a single device, and you want to avoid communication overhead.


# "un-split" distributed array
a.resplit_(axis=None)
# each process now holds a copy of the entire array


# The opposite is not true, i.e. you cannot use `resplit_` to distribute an array with split=None. In that case, you must use the `ht.array()` factory function:


# make `a` split again
a = ht.array(a, split=0)


# ### Making disjoint data into a global DNDarray
#
# Another common occurrence in a data-parallel pipeline: you have addressed the embarassingly-parallel part of your algorithm with any array framework, each process working independently from the others. You now want to perform a non-embarassingly-parallel operation on the entire dataset, with Heat as a backend.
#
# You can use the `ht.array` factory function with the `is_split` argument to create a DNDarray from a disjoint (on each MPI process) set of arrays. The `is_split` argument indicates the axis along which the disjoint data is to be "joined" into a global, distributed DNDarray.


# create some random local arrays on each process
import numpy as np

local_array = np.random.rand(3, 4)

# join them into a distributed array
a_0 = ht.array(local_array, is_split=0)
a_0.shape


# Change the cell above and join the arrays along a different axis. Note that the shapes of the local arrays must be consistent along the non-split axes. They can differ along the split axis.

# The `ht.array` function takes any data object as an input that can be converted to a torch tensor.

# Once you've made your disjoint data into a DNDarray, you can apply any Heat operation or algorithm to it and exploit the cumulative RAM of all the processes in the communicator.

# You can access the MPI communication functionalities of the DNDarray through the `comm` attribute, i.e.:
#
# ```python
# # these are just examples, this cell won't do anything
# a.comm.Allreduce(a, b, op=MPI.SUM)
#
# a.comm.Allgather(a, b)
# a.comm.Isend(a, dest=1, tag=0)
# ```
#
# etc.

# In the next notebooks, we'll show you how we use Heat's distributed-array infrastructure to scale complex data analysis workflows to large datasets and high-performance computing resources.
#
# - [Data loading and preprocessing](4_loading_preprocessing.ipynb)
# - [Matrix factorization algorithms](5_matrix_factorizations.ipynb)
# - [Clustering algorithms](6_clustering.ipynb)
