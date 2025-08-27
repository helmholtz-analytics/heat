import heat as ht
import numpy as np
import torch

# ### Data Types
#
# Heat supports various data types and operations to retrieve and manipulate the type of a Heat array. However, in contrast to NumPy, Heat is limited to logical (bool) and numerical types (uint8, int16/32/64, float32/64, and complex64/128).
#
# **NOTE:** by default, Heat will allocate floating-point values in single precision, due to a much higher processing performance on GPUs. This is one of the main differences between Heat and NumPy.

a = ht.zeros((3, 4))
print(f"floating-point values in single precision is default: {a.dtype}")

b = torch.zeros(3, 4)
print(f"like in PyTorch: {b.dtype}")


b = np.zeros((3, 4))
print(f"whereas floating-point values in double  precision is default in numpy: {b.dtype}")

b = a.astype(ht.int64)
print(f"casting to int64: {b}")
