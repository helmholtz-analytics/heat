"""
file for future "full" SVD implementation
"""
from typing import Tuple
from ..dndarray import DNDarray

__all__ = ["svd"]


def svd(A: DNDarray) -> Tuple[DNDarray, DNDarray, DNDarray]:
    """
    The intended functionality is similar to `numpy.linalg.svd`, but of-course allowing for distributed-memory parallelization and GPU-support.
    """
    raise NotImplementedError(
        " Memory-distributed 'full' (i.e. non-trucated and non-approximate) SVD not implemented yet. Consider using `heat.linalg.hsvd` for an approximate, truncated  SVD instead."
    )
