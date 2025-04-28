"""
Basic linear algebra operations on distributed ``DCSR_matrix``
"""

from ..dcsr_matrix import DCSR_matrix
from ..factories import sparse_csr_matrix
from ...core import devices
import torch


def matmul(A: DCSR_matrix, B: DCSR_matrix) -> DCSR_matrix:
    """
    Matrix multiplication of two DCSR matrices.
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible dimensions for matrix multiplication")

    out_split = 0 if A.split == 0 or B.split == 0 else None

    collected_B = torch.sparse_csr_tensor(
        B.indptr,
        B.indices,
        B.data,
        device=B.device.torch_device if B.device is not None else devices.get_device().torch_device,
        size=B.shape,
    )

    matmul_res = A.larray @ collected_B

    return sparse_csr_matrix(
        matmul_res, dtype=A.dtype, device=A.device, comm=A.comm, is_split=out_split
    )


DCSR_matrix.matmul = lambda self, other: matmul(self, other)
