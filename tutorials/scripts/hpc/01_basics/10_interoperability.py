# ### Interoperability
#
# We can easily create DNDarrays from PyTorch tensors and numpy ndarrays. We can also convert DNDarrays to PyTorch tensors and numpy ndarrays. This makes it easy to integrate Heat into existing PyTorch and numpy workflows.
#

# Heat will try to reuse the memory of the original array as much as possible. If you would prefer a copy with different memory, the ```copy``` keyword argument can be used when creating a DNDArray from other libraries.

import heat as ht
import torch
import numpy as np

torch_array = torch.arange(ht.MPI_WORLD.rank, ht.MPI_WORLD.rank + 5)
heat_array = ht.array(torch_array, copy=False, is_split=0)
heat_array[0] = -1
print(torch_array)

torch_array = torch.arange(ht.MPI_WORLD.rank, ht.MPI_WORLD.rank + 5)
heat_array = ht.array(torch_array, copy=True, is_split=0)
heat_array[0] = -1
print(torch_array)

np_array = heat_array.numpy()
print(np_array)


# Interoperability is a key feature of Heat, and we are constantly working to increase Heat's compliance to the [Python array API standard](https://data-apis.org/array-api/latest/). As usual, please [let us know](tinyurl.com/demoissues) if you encounter any issues or have any feature requests.
