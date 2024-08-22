"""
Module implementing the Dynamic Mode Decomposition (DMD) algorithm.
"""

import heat as ht
from typing import Optional, Tuple, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class DMD(ht.RegressionMixin, ht.BaseEstimator):
    """
    Dynamic Mode Decomposition (DMD), plain vanilla version with SVD-based implementation.

    The time series of which DMD shall be computed must be given as a 2-D DNDarray of shape (n_features, n_timesteps).

    Parameters
    ----------
    svd_algorithm : str, optional
        Specifies the algorithm to use for the singular value decomposition (SVD). Options are 'full' (default), 'hierarchical', and 'randomized'.
    svd_rank : int, optional
        The rank to which SVD shall be truncated. For `'full'` SVD, `svd_rank = None` together with `svd_tol = None` (default) will result in no truncation.
        For `svd_algorithm='full'`, at most one of `svd_rank` or `svd_tol` may be specified.
        For `svd_algorithm='hierarchical'`, either `svd_rank` (rank to truncate to) or `svd_tol` (tolerance to truncate to) must be specified.
        For `svd_algorithm='randomized'`, `svd_rank` must be specified and determines the the rank to truncate to.
    svd_tol : float, optional
        The tolerance to which SVD shall be truncated. For `'full'` SVD, `svd_tol = None` together with `svd_rank = None` (default) will result in no truncation.
        For `svd_algorithm='hierarchical'`, either `svd_tol` (accuracy to truncate to) or `svd_rank` (rank to truncate to) must be specified.
        For `svd_algorithm='randomized'`, `svd_tol` is meaningless.

    Attributes
    ----------
    svd_algorithm : str
        The algorithm used for the singular value decomposition (SVD).
    svd_rank : int
        The rank to which SVD shall be truncated.
    svd_tol : float
        The tolerance to which SVD shall be truncated.

    Notes
    ----------
    ...
    """

    def __init__(
        self,
        svd_algorithm: Optional[str] = "full",
        svd_rank: Optional[int] = None,
        svd_tol: Optional[float] = None,
    ):
        self.svd_rank = svd_rank
        self.svd_tol = svd_tol
        self.svd_algorithm = svd_algorithm
