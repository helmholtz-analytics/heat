Module heat.spatial.distance
============================
Module for (pairwise) distance functions

Functions
---------

`cdist(X: heat.core.dndarray.DNDarray, Y: heat.core.dndarray.DNDarray = None, quadratic_expansion: bool = False) ‑> heat.core.dndarray.DNDarray`
:   Calculate Euclidian distance between two DNDarrays:

    .. math:: d(x,y) = \sqrt{(|x-y|^2)}

    Returns 2D DNDarray of size :math: `m \times n`

    Parameters
    ----------
    X : DNDarray
        2D array of size :math: `m \times f`
    Y : DNDarray
        2D array of size :math: `n \times f`
    quadratic_expansion : bool
        Whether to use quadratic expansion for :math:`\sqrt{(|x-y|^2)}` (Might yield speed-up)

`manhattan(X: heat.core.dndarray.DNDarray, Y: heat.core.dndarray.DNDarray = None, expand: bool = False)`
:   Calculate Manhattan distance between two DNDarrays:

    .. math:: d(x,y) = \sum{|x_i-y_i|}

    Returns 2D DNDarray of size :math: `m \times n`

    Parameters
    ----------
    X : DNDarray
        2D array of size :math: `m \times f`
    Y : DNDarray
        2D array of size :math: `n \times f`
    expand : bool
        Whether to use dimension expansion (Might yield speed-up)

`rbf(X: heat.core.dndarray.DNDarray, Y: heat.core.dndarray.DNDarray = None, sigma: float = 1.0, quadratic_expansion: bool = False) ‑> heat.core.dndarray.DNDarray`
:   Calculate Gaussian distance between two DNDarrays:

    .. math:: d(x,y) = exp(-(|x-y|^2/2\sigma^2)

    Returns 2D DNDarray of size :math: `m \times n`

    Parameters
    ----------
    X : DNDarray
        2D array of size :math: `m \times f`
    Y : DNDarray
        2D array of size `n \times f`
    sigma: float
        Scaling factor for gaussian kernel
    quadratic_expansion : bool
        Whether to use quadratic expansion for :math:`\sqrt{(|x-y|^2)}` (Might yield speed-up)
