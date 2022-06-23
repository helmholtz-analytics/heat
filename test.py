import heat as ht
import torch
import numpy as np

crow_indices = [0, 2, 4, 4, 5, 6]
col_indices = [0, 1, 0, 1, 3, 4]
values = [1, 2, 3, 4, 5, 6]

t = torch.sparse_csr_tensor(torch.tensor(crow_indices, dtype=torch.int64),
                        torch.tensor(col_indices, dtype=torch.int64),
                        torch.tensor(values), dtype=torch.double)

ht.sparse_csr_array(t, split=1)
