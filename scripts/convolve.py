import torch
import numpy as np
import sys

sys.path.append('../')

import heat as ht
from heat import manipulations

print("-------------------------")

a = ht.arange(0, 128, split=0).astype(ht.float)
v = ht.ones(7).astype(ht.float)
rank, size = [a.comm.rank, a.comm.size]
print("[{0}/{1}], a shape: {2}".format(rank, size, a.lshape))
# c = ht.vstack((a, b))
conv = ht.convolve1D(a, v, mode='full')

# rebalance is necessary for gather

print("[{0}/{1}], shape: {2}".format(rank, size, conv.shape))
print("[{0}/{1}], gshape: {2}".format(rank, size, conv.gshape))
print("[{0}/{1}], lshape: {2}".format(rank, size, conv.lshape))
print("[{0}/{1}], dndarray: {2}".format(rank, size, conv))
print("[{0}/{1}], _DNDarray__array: {2}".format(rank, size, conv._DNDarray__array))
print("[{0}/{1}], is_balanced: {2}".format(rank, size, conv.is_balanced()))

conv.balance_()

counts, displs, _ = conv.comm.counts_displs_shape(conv.shape, conv.split)

print("[{0}/{1}], counts: {2}, displs: {3}".format(rank, size, counts, displs))


gathered = manipulations.resplit(conv, axis=None)

print("[{0}/{1}], gathered: {2}".format(rank, size, gathered))

exit()
'''
t = torch.rand((1, 1, 16, 16), dtype=torch.float32)
filter1d = torch.tensor([[
    [1, 2, 1]
]])
filter2d = torch.tensor([[
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]], dtype=torch.float32)

result = torch.nn.functional.conv1d(
    t,
    filter2d
)

print(t)
print(filter2d)
print(result)
print(result.shape)
'''
