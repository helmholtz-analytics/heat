import heat as ht
import torch

# ### Modifying the DNDarray distribution
#
# In a distributed pipeline, it is sometimes necessary to change the distribution of a DNDarray, when the array is not distributed in the most convenient way for the next operation / algorithm.
#
# Depending on your needs, you can choose between:
# - `DNDarray.redistribute_()`: This method keeps the original split axis, but redistributes the data of the DNDarray according to a "target map".
# - `DNDarray.resplit_()`: This method changes the split axis of the DNDarray. This is a more expensive operation, and should be used only when absolutely necessary. Depending on your needs and available resources, in some cases it might be wiser to keep a copy of the DNDarray with a different split axis.
#
# Let's see some examples.

a = ht.random.randn(7, 4, 3, split=1)

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
print(a_0.shape)


# Change the cell above and join the arrays along a different axis. Note that the shapes of the local arrays must be consistent along the non-split axes. They can differ along the split axis.

# The `ht.array` function takes any data object as an input that can be converted to a torch tensor.

# Once you've made your disjoint data into a DNDarray, you can apply any Heat operation or algorithm to it and exploit the cumulative RAM of all the processes in the communicator.
