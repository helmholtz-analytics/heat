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

    def __init__(
        self,
        n_components=None,
        copy=True,
        whiten=False,
        svd_solver="hsvd",
        tol=None,
        iterated_power="auto",
        n_oversamples=10,
        power_iteration_normalizer="qr",
        random_state=None,
    ):
        # check correctness of inputs
        if not copy:
            raise ValueError("In-place PCA computation is not yet supported. Please set copy=True.")
        if whiten:
            raise ValueError("Whitening is not yet supported. Please set whiten=False.")
        if not (svd_solver == "full" or svd_solver == "hierarchical"):
            raise ValueError(
                "At the moment, only svd_solver='full' (for tall-skinny or short-fat data) and svd_solver='hierarchical' are supported. \n An implementation of the 'full' option for arbitrarily shaped data as well as the option 'randomized' are already planned."
            )
        if iterated_power != "auto" and not isinstance(iterated_power, int):
            raise ValueError("iterated_power must be 'auto' or an integer.")
        if isinstance(iterated_power, int) and iterated_power < 0:
            raise ValueError("if an integer, iterated_power must be greater or equal to 0.")
        if power_iteration_normalizer != "qr":
            raise ValueError("Only power_iteration_normalizer='qr' is supported yet.")
        if not isinstance(n_oversamples, int) or n_oversamples < 0:
            raise ValueError("n_oversamples must be a non-negative integer.")
        if tol is not None:
            raise ValueError(
                "Argument tol is not yet necessary as iterative methods for PCA are not yet implemented. Please set tol=None."
            )
        if random_state is None:
            random_state = 0
        if not isinstance(random_state, int):
            raise ValueError("random_state must be None or an integer.")
        if (
            n_components is not None
            and not (isinstance(n_components, int) and n_components >= 1)
            and not (isinstance(n_components, float) and n_components > 0.0 and n_components < 1.0)
        ):
            raise ValueError(
                "n_components must be None, in integer greater or equal to 1 or a float in (0,1). Option 'mle' is not supported at the momemnt."
            )

        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state
