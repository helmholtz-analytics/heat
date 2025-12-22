Module heat.graph.laplacian
===========================
Module for graph-based classes

Classes
-------

`Laplacian(similarity: Callable, weighted: bool = True, definition: str = 'norm_sym', mode: str = 'fully_connected', threshold_key: str = 'upper', threshold_value: float = 1.0, neighbours: int = 10)`
:   Graph Laplacian from a dataset

    Parameters
    ----------
    similarity : Callable
        Metric function that defines similarity between vertices. Should accept a data matrix :math:`n \times f` as input and
        return an :math:`n\times n` similarity matrix. Additional required parameters can be passed via a lambda function.
    definition : str
        Type of Laplacian

            - ``'simple'``: Laplacian matrix for simple graphs :math:`L = D - A`

            - ``'norm_sym'``: Symmetric normalized Laplacian :math:`L^{sym} = I - D^{-1/2} A D^{-1/2}`

            - ``'norm_rw'``: Random walk normalized Laplacian :math:`L^{rw} = D^{-1} L = I - D^{-1}`

    mode : str
        How to calculate adjacency from the similarity matrix

            - ``'fully_connected'`` is fully-connected, so :math:`A = S`

            - ``'eNeighbour'`` is the epsilon neighbourhood, with :math:`A_{ji} = 0` if :math:`S_{ij} > upper` or
            :math:`S_{ij} < lower`; for eNeighbour an upper or lower boundary needs to be set

    threshold_key : str
        ``'upper'`` or ``'lower'``, defining the type of threshold for the epsilon-neighborhood
    threshold_value : float
        Boundary value for the epsilon-neighborhood
    neighbours : int
        Number of nearest neighbors to be considered for adjacency definition. Currently not implemented

    ### Methods

    `construct(self, X: DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Callable to get the Laplacian matrix from the dataset ``X`` according to the specified Laplacian

        Parameters
        ----------
        X : DNDarray
            The data matrix, Shape = (n_samples, n_features)
