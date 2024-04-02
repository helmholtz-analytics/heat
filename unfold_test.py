"""
Test module for DNDarray.unfold
"""

import heat as ht
from mpi4py import MPI
import torch

from heat import factories
from heat import DNDarray


# tests
n = 20

# x = torch.arange(0.0, n)
# y = 2 * x.unsqueeze(1) + x.unsqueeze(0)
# y = factories.array(y)
# y.resplit_(1)

# print(y)
# print(unfold(y, 0, 3, 1))

x = torch.arange(0, n * n).reshape((n, n))
# print(f"x: {x}")
y = factories.array(x)
y.resplit_(0)

u = x.unfold(0, 3, 3)
u = u.unfold(1, 3, 3)
u = factories.array(u)
v = ht.unfold(y, 0, 3, 3)
v.resplit_(1)
v = ht.unfold(v, 1, 3, 3)

comm = u.comm
# print(v)
equal = ht.equal(u, v)

u_shape = u.shape
v_shape = v.shape

if comm.rank == 0:
    print(f"u.shape: {u_shape}")
    print(f"v.shape: {v_shape}")
    print(f"torch and heat unfold equal: {equal}")
    # print(f"u: {u}")

# print(f"v: {v}")
