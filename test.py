import heat as ht
import torch

# TEST SPARSE COO
rank = ht.communication.MPI_WORLD.rank
size = ht.communication.MPI_WORLD.size
comm = ht.communication.MPI_WORLD
# comm = MPI.COMM_WORLD

i = [[0, 1, 1], [1, 0, 1],[1, 0, 1],[1, 0, 1]]
v =  [3, 4, 5]
s = torch.sparse_coo_tensor(i, v, (2, 2,2,2))
split=3

#output_shape = (s.shape[split]*size,) + tuple(s.shape[:3])

# Have to change the shape to list to be able to change the value of shape[split]
output_shape = tuple(s.shape)
output_shape = list(output_shape)
output_shape[split] = output_shape[split]*size
print(output_shape)
output_shape = tuple(output_shape)
ht_s = ht.coo_matrix.coo_matrix(s, gshape = output_shape, split=split, dtype=ht.int64, gnnz=6, device="cpu", balanced=True, comm=comm)
print(ht_s.indices)