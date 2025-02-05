"""Tests during the implementation of the Local Outlier Factor (LOF) algorithm"""

import heat as ht
import torch

a = ht.array([10, 20, 2, 17, 8], split=0)
# print(f"a={a}, \n b={b}, \n c={c}")

y = ht.array([[2, 3, 1], [5, 6, 4], [7, 8, 9]], split=0)
o = ht.zeros([y.shape[0], y.shape[1]], split=0)

x = y.larray
buffer = torch.zeros_like(x)
o.larray[:] = x

print(f"process= {ht.MPI_WORLD.rank}\n o={o}\n buffer={buffer}")
