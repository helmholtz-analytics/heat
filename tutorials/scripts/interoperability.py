# %% [markdown]
# Interoperability of Heat DNDarray with other python array libraries
# ===================================================================

# %%
import heat as ht
import torch
data_heat = ht.arange(4 * 3).reshape(4, 3).resplit(None)
data_heat_split = ht.arange(4 * 3).reshape(4, 3).resplit(0)

print(f'Running with {ht.comm.size} tasks')

# %% [markdown]
# NumPy
# -----
# We begin with unsplit data, which we can simply cast back and forth between numpy and heat:

# %%
import numpy as np
data_numpy = np.arange(4 * 3).reshape(4, 3)
assert np.allclose(data_numpy, np.array(data_heat))
assert ht.allclose(data_heat, ht.array(data_numpy))

# %% [markdown]
# With the split data, we need to be more careful, because if while casting from heat to numpy, we get only the local data on each process.
# We can use the `chunk` method of heat communicators to compute the shape and slices of local data given a heat array of any shape and split.
# Note that you should use this slice on the numpy data in order to extract the process local data. You should not use this on the heat data, because you never want to use different slices on each task in heat, as heat takes care of distributing the data among processes itself.

# %%
global_shape, split = data_heat_split.shape, data_heat_split.split
offset, local_shape, slices = data_heat_split.comm.chunk(global_shape, split)
assert np.allclose(data_numpy[*slices], np.array(data_heat_split))

# %% [markdown]
# Going from process-local numpy data to a heat array is a bit more complicated.
# We need to access the `larray` attribute of the heat array, which is a torch tensor, so we cast to torch and then assign this to the local heat data.

# %%
_data_heat_split = ht.empty(global_shape, split=split, dtype=ht.int)
_data_heat_split.larray[...] = torch.from_numpy(data_numpy[*slices])
assert ht.allclose(_data_heat_split, data_heat_split)
