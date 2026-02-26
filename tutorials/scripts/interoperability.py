# %% [markdown]
# Interoperability of Heat DNDarray with other python array libraries
# ===================================================================

# This tutorial will show how to use data generated in heat with other libraries and vice versa.
# Do have a look if you plan on migrating your code from one of the discussed libraries to heat!

# This tutorial assumes that you understand the use of the `split` attribute in heat, which determines the distribution of the data.
# If you don't feel comfortable with this yet, please check out a basic tutorial on heat first.

# Before discussing how to interoperate with other libraries, we import heat and set up some data.
# Some things are different when the data is distributed, which is why we generate split and unsplit data.
# Note that you need to run this tutorial with multiple MPI tasks for the split attribute to matter as otherwise the data will not be distributed regardless of the split attribute.

# %%
import heat as ht

data_heat = ht.arange(4 * 3).reshape(4, 3).resplit(None)
data_heat_split = ht.arange(4 * 3).reshape(4, 3).resplit(0)

print(f'Running with {ht.comm.size} tasks')
print(f'`data_heat_split` is{' not' if not data_heat_split.is_distributed() else ''} distributed')

# %% [markdown]
# NumPy
# -----
# We begin with unsplit data, which we can simply cast back and forth between numpy and heat:

# %%
import numpy as np

data_numpy = np.arange(4 * 3).reshape(4, 3)

assert np.allclose(data_numpy, np.array(data_heat))  # convert heat to numpy
assert ht.allclose(data_heat, ht.array(data_numpy))  # convert numpy to heat

# %% [markdown]
# With the split data, we need to be more careful, because while casting from heat to numpy, we get only the local data on each process.
# We can use the `chunk` method of heat communicators to compute the shape and slices of local data given an array of any shape and split.

# Note that you should use this slice on the numpy data in order to extract the process local data.
# You should not use this on the heat data, because you never want to use different slices on each task in heat, as this would conflict with heat taking care of distributing the data itself.

# %%
global_shape, split = data_heat_split.shape, data_heat_split.split
offset, local_shape, slices = data_heat_split.comm.chunk(global_shape, split)
assert np.allclose(data_numpy[*slices], np.array(data_heat_split))  # convert from heat to numpy

# %% [markdown]
# Going from process-local numpy data to a heat array is a bit more complicated.
# We need to access the `larray` attribute of the heat array, which is a torch tensor, so we cast to torch and then assign this to the local heat data.

# %%
import torch
_data_heat_split = ht.empty(global_shape, split=split, dtype=ht.int)
_data_heat_split.larray[...] = torch.from_numpy(data_numpy[*slices])
assert ht.allclose(_data_heat_split, data_heat_split)  # convert from numpy to heat
