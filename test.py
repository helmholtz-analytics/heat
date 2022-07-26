"""Testing file for coo sparse array class"""
import heat as ht
from heat.core.factories import sparse_coo_matrix
import torch
import numpy as np
from scipy.sparse import coo_matrix

# TEST SPARSE COO
rank = ht.communication.MPI_WORLD.rank
size = ht.communication.MPI_WORLD.size
comm = ht.communication.MPI_WORLD
# comm = MPI.COMM_WORLD

i = [[0, 1, 1], [1, 0, 1], [2, 0, 2], [1, 0, 1]]
v = [3, 4, 5]
s = torch.sparse_coo_tensor(i, v, (2, 2, 3, 2))
split = 3
# output_shape = (s.shape[split]*size,) + tuple(s.shape[:3])

# Have to change the shape to list to be able to change the value of shape[split]
output_shape = tuple(s.shape)
output_shape = list(output_shape)
output_shape[split] = output_shape[split] * size
# print(output_shape)
output_shape = tuple(output_shape)
ht_s = ht.dcoo_array.Dcoo_array(
    s,
    gshape=output_shape,
    split=split,
    dtype=ht.int64,
    gnnz=6,
    device="cpu",
    balanced=True,
    comm=comm,
)
print(ht_s.indices)

scipy_coo = coo_matrix(([3, 4, 5], ([0, 1, 1], [2, 0, 2])), shape=(2, 3))
ht_s = ht.sparse_coo_matrix(scipy_coo, is_split=1)
# print(ht_s.indices)
# print(ht_s.lindices)
# print(ht_s.gnnz)
# print(ht_s.lnnz)
# row  = np.array([0, 0, 1, 3, 1, 0, 0])
# col  = np.array([0, 2, 1, 3, 1, 0, 0])
# data = np.array([1, 1, 1, 1, 1, 1, 1])
# coo = coo_matrix((data, (row, col)), shape=(4, 4))
# print(type(coo.col))
# new =  sparse_coo_matrix(coo, split=0)

# torch coo sparse tensor
# i = [[0, 7, 18], [2, 0, 8]]
# v = [3, 4, 5]
# s = torch.sparse_coo_tensor(i, v, (20, 13))
# ht_s = ht.sparse_coo_matrix(s, is_split=0)
# print(ht_s.indices)
# print(ht_s.lindices)
# print(ht_s.gnnz)
# print(ht_s.lnnz)

from heat import factories
