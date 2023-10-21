from ..dcsr_matrix import DCSR_matrix
from ..factories import sparse_csr_matrix
from ...core import devices
import torch


def matmul(t1: DCSR_matrix, t2: DCSR_matrix) -> DCSR_matrix:
    if t1.shape[1] != t2.shape[0]:
        raise ValueError("Incompatible dimensions for matrix multiplication")

    out_split = 0 if t1.split == 0 or t2.split == 0 else None

    def collect(t):
        return torch.sparse_csr_tensor(
            t.indptr,
            t.indices,
            t.data,
            device=t.device.torch_device
            if t.device is not None
            else devices.get_device().torch_device,
            size=t.shape,
        )

    collected_t1 = collect(t1)
    collected_t2 = collect(t2)

    matmul_res = collected_t1 @ collected_t2

    return sparse_csr_matrix(
        matmul_res, dtype=t1.dtype, device=t1.device, comm=t1.comm, split=out_split
    )


DCSR_matrix.matmul = lambda self, other: matmul(self, other)
