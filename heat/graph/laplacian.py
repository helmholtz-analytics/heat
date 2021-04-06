"""
Module for graph-based classes
"""
from __future__ import annotations

from typing import Callable
import torch
import heat as ht
from heat.core.dndarray import DNDarray


class Laplacian:
    """
    Graph Laplacian from a dataset

    Parameters
    ----------
    similarity : Callable
        Metric function that defines similarity between vertices. Should accept a data matrix :math:`n \\times f` as input and
        return an :math:`n\\times n` similarity matrix. Additional required parameters can be passed via a lambda function.
    definition : str
        Type of Laplacian \n
            - ``'simple'``: Laplacian matrix for simple graphs :math:`L = D - A` \n
            - ``'norm_sym'``: Symmetric normalized Laplacian :math:`L^{sym} = I - D^{-1/2} A D^{-1/2}` \n
            - ``'norm_rw'``: Random walk normalized Laplacian :math:`L^{rw} = D^{-1} L = I - D^{-1}` \n
    mode : str
        How to calculate adjacency from the similarity matrix \n
            - ``'fully_connected'`` is fully-connected, so :math:`A = S` \n
            - ``'eNeighbour'`` is the epsilon neighbourhood, with :math:`A_{ji} = 0` if :math:`S_{ij} > upper` or
            :math:`S_{ij} < lower`; for eNeighbour an upper or lower boundary needs to be set \n
    threshold_key : str
        ``'upper'`` or ``'lower'``, defining the type of threshold for the epsilon-neighborhood
    threshold_value : float
        Boundary value for the epsilon-neighborhood
    neighbours : int
        Number of nearest neighbors to be considered for adjacency definition. Currently not implemented
    """

    def __init__(
        self,
        similarity: Callable,
        weighted: bool = True,
        definition: str = "norm_sym",
        mode: str = "fully_connected",
        threshold_key: str = "upper",
        threshold_value: float = 1.0,
        neighbours: int = 10,
    ) -> DNDarray:
        self.similarity_metric = similarity
        self.weighted = weighted
        if definition not in ["simple", "norm_sym"]:
            raise NotImplementedError(
                "Currently only simple and normalized symmetric graph laplacians are supported"
            )
        else:
            self.definition = definition
        if mode not in ["eNeighbour", "fully_connected"]:
            raise NotImplementedError(
                "Only eNeighborhood and fully-connected graphs supported at the moment."
            )
        else:
            self.mode = mode

        if threshold_key not in ["upper", "lower"]:
            raise ValueError(
                "Only 'upper' and 'lower' threshold types supported for eNeighbouhood graph construction"
            )
        else:
            self.epsilon = (threshold_key, threshold_value)

        self.neighbours = neighbours

    def _normalized_symmetric_L(self, A: DNDarray) -> DNDarray:
        """
        Helper function to calculate the normalized symmetric Laplacian

        .. math:: L^{sym} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}

        Parameters
        ----------
        A : DNDarray
            The adjacency matrix of the graph
        """
        degree = ht.sum(A, axis=1)
        degree.resplit_(axis=None)
        # Find stand-alone vertices with no connections
        temp = torch.ones(
            degree.shape, dtype=degree.larray.dtype, device=degree.device.torch_device
        )
        degree.larray = torch.where(degree.larray == 0, temp, degree.larray)
        L = A / ht.sqrt(ht.expand_dims(degree, axis=1))
        L = L / ht.sqrt(ht.expand_dims(degree, axis=0))
        L = L * (-1.0)
        L.fill_diagonal(1.0)
        return L

    def _simple_L(self, A: DNDarray):
        """
        Helper function to calculate the simple graph Laplacian

        .. math:: L = D - A

        Parameters
        ----------
        A : DNDarray
            The Adjacency Matrix of the graph
        """
        degree = ht.sum(A, axis=1)
        L = ht.diag(degree) - A
        return L

    def construct(self, X: DNDarray) -> DNDarray:
        """
        Callable to get the Laplacian matrix from the dataset ``X`` according to the specified Laplacian

        Parameters
        ----------
        X : DNDarray
            The data matrix, Shape = (n_samples, n_features)
        """
        S = self.similarity_metric(X)
        S.fill_diagonal(0.0)

        if self.mode == "eNeighbour":
            if self.epsilon[0] == "upper":
                if self.weighted:
                    S = ht.where(S < self.epsilon[1], S, 0)
                else:
                    S = ht.int(S < self.epsilon[1])
            else:
                if self.weighted:
                    S = ht.where(S > self.epsilon[1], S, 0)
                else:
                    S = ht.int(S > self.epsilon[1])

        if self.definition == "simple":
            L = self._simple_L(S)
        elif self.definition == "norm_sym":
            L = self._normalized_symmetric_L(S)

        return L
