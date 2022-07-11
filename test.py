import heat as ht
import torch

# TEST SPARSE COO
rank = ht.communication.MPI_WORLD.rank
size = ht.communication.MPI_WORLD.size
comm = ht.communication.MPI_WORLD
# comm = MPI.COMM_WORLD

i = [[0, 1, 1], [2, 0, 2], [1, 1, 0]]
v =  [3, 4, 5]
s = torch.sparse_coo_tensor(i, v, (2, 3,2))

output_shape = (s.shape[0]*size,) + tuple(s.shape[1:])
# print(size)
ht_s = ht.coo_matrix.coo_matrix(s, gshape = output_shape, split=2, dtype=ht.int64, gnnz=6, device="cpu", balanced=True, comm=comm)
print(ht_s.indices)