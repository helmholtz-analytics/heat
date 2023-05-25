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
        "Currently, a 'full' (i.e. non-trucated and non-approximate) SVD is not available in Heat. Please consider using `heat.linalg.hsvd_rank` or `heat.linalg.hsvd_rtol` for computing an approximate, truncated (w.r.t. rank or relative tolerance) SVD instead. Alternatively, you may also check on github whether the implementation of the full SVD is meanwhile available in a development branch of Heat."
    )
