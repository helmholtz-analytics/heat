import heat as ht
import torch
import numpy as np
from copy import copy as _copy
from heat.core.tensor import tensor
from heat.core.communicator import mpi, MPICommunicator, NoneCommunicator
from heat.core import types

#filters = torch.randn(33, 16, 3)
#inputs = ht.randn(20, 16, 50)
#ht.conv1d(inputs, filters)


"""
a = np.ones(10)
v = np.ones(3)
res1 = np.convolve(a,v, mode='full')
res2 = np.convolve(a,v, mode='valid')
res3 = np.convolve(a,v, mode='same')
print('pes1: ', res1)
print('pes2: ', res2)
print('pes3: ', res3)
"""

a = ht.ones(10).astype(ht.int)
print('a: ',type(a.sum().split)

"""
b = ht.ones(10)
print('b: ',b.sum(axis=0))

c = ht.ones(10, split=0)
print('c: ',a.sum())
b = ht.ones(10, split=0)
print('b: ',a.sum(axis=0))




#b = ht.ones(1)

#print(b._tensor__array)
#print('b: ', b.sum(), b.shape)

#v = ht.arange(3).astype(ht.float)
#print(type(ht.convolve(a,v, mode='full')))
#res2 = ht.convolve(a,v, mode='valid')
#res3 = ht.convolve(a,v, mode='same')
#print('res1: ', np.array(res1.array))
#print('res2: ', np.array(res2.array))
#print('res3: ', np.array(res3.array))

#print('res1len: ', res1.shape)
#print('res2len: ', res2.shape)
#print('res3len: ', res3.shape)


#aa = ht.ones((3,3)) * 3
#bb = ht.ones((1,3))

#dtype = types.canonical_heat_type(torch.float64)

#print(dtype)

# print(aa+bb, aa, bb)


#na = np.ones(6)
#nv = np.ones(2)
#print(np.convolve(na, nv, mode = 'valid'))
filters = torch.ones(1, 1, 2)
inputs = torch.ones(1, 1, 6)
print(torch.conv1d(inputs, filters))


self.array = array
self.gshape = gshape
self.dtype = dtype
self.split = split
self.comm = comm
self.halo_next = halo_next
self.halo_prev = halo_prev
self.halo_size = halo_size

a = ht.ones(6)
b = tensor(a.array, a.gshape, a.dtype, a.split, a.comm)
#b.array[2] = 99
#b.array[2] = 99
c = a.astype(ht.float32)

nv = ht.ones(3)
aa = ht.ones(6, split=0)
bb = ht.ones(6, split=0)
"""
#na = ht.convolve(aa, nv, mode = 'valid')
#nb = ht.convolve(bb, nv, mode = 'valid')
#print(aa.exp(bb), bb)
#nc = aa + bb

#print(nc.halo_next)


#v = ht.ones(3)
#qq1 = ht.convolve(a,v,mode = 'valid')
#print('qq1: ', qq1)

#print()

# torch_tensor, tuple(gshape), self.dtype, split, self.comm

#nw = np.concatenate((np.zeros(2),na,np.zeros(2)))
#print(np.convolve(na, nv, mode = 'full'))
#print(np.convolve(na, nv, mode = 'full'))
#print(np.convolve(nw, nv, mode = 'valid'))
#print(np.convolve(na, nv, mode = 'same'))

#aa = ht.ones((1,1,9), split=2) * 3
#dd = ht.ones((1,1,9), split=2) * 3

#ww = ht.ones((10,10,10), split = 2)

#print(ww.lshape)
#print(ww.gshape[:ww.split], ww.gshape[ww.split + 1:], ww.gshape)


# qq=ww.unsqueeze(0)
#qq=ww.unsqueeze(2)
#print('qqas: ', qq.gshape)


"""
aa = ht.ones((33, 16, 50), split=1) * 3
print('lshape: ', aa.split)
#print('tensordata: ', aa._tensor__array)

#filters = torch.randn(33, 16, 3)
#inputs = torch.randn(20, 16, 50)
#print('split: ', aa.split)
ww = ht.ones((20, 16, 3))

bb = ht.conv1d(aa, ww)
#print('aa: ', aa.halo_next)
bb=aa.exp()
#print('bb: ', bb)
bb = ht.conv1d(aa, ww)
#print('qq2: ', aa.halo_size)
#gg = ht.conv1d(dd, weight, stride=1)
#print(gg.halo_size)
"""
"""
aa = ht.arange(4, split=0)
print(aa)
print('sum: ', aa.sum().item())
# print(aa.array)
aa.gethalo(1)
print('halo: ', aa.halo_next)
aa.exp()
# aa.delhalo()
#print(aa.halo_next, aa.halo_prev)
#print(aa.halorize())
"""
