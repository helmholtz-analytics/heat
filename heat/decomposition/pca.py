"""
Module implementing decomposition techniques, such as PCA.
"""

import heat as ht
from typing import Optional, Tuple, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

"""
The implementation is heavily inspired by the corresponding routines in scikit-learn (https://scikit-learn.org/stable/modules/decomposition.html).
"""


class StandardScaler(ht.TransformMixin, ht.BaseEstimator):
    """
    todo
    """

    pass
