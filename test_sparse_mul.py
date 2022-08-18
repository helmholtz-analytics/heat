import heat as ht
import torch
import numpy as np
from scipy.sparse import csr_matrix
from mpi4py import MPI

from heat.core.tests.test_suites.basic_test import TestCase

# Ignore warning from torch
import warnings

warnings.filterwarnings("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

indptr = np.array([0, 1, 3, 4])
indices = np.array([1,0,1,2])
data = np.array([2,1,2,3])

t1 = torch.sparse_csr_tensor(
    torch.tensor(indptr, dtype=torch.int64),
    torch.tensor(indices, dtype=torch.int64),
    torch.tensor(data),
)
t1_sparse = ht.sparse.sparse_csr_matrix(t1, split=0)


indptr = np.array([0, 1, 2, 4])
indices = np.array([0,0,0,2])
data = np.array([1,1,2,4])

t2 = torch.sparse_csr_tensor(
    torch.tensor(indptr, dtype=torch.int64),
    torch.tensor(indices, dtype=torch.int64),
    torch.tensor(data),
)
t2_sparse = ht.sparse.sparse_csr_matrix(t2, split=0)

multi = t1_sparse + t2_sparse

print(f'''Rank: {rank}
        data: {multi.ldata}
        indptr: {multi.lindptr}
        global_indptr: {multi.global_indptr()}
        indices: {multi.lindices}''')