import torch
import heat as ht


class Laplacian:
    def __init__(
        self,
        similarity,
        weighted=True,
        definition="norm_sym",
        mode="fully_connected",
        threshold_key="upper",
        threshold_value=1.0,
        neighbours=10,
    ):
        """
        Graph Laplacians from a dataset

        Parameters
        ----------
        similarity : function f(X) --> similarity matrix
            Metric function that defines similarity between vertices. Should accept a data matrix (n,f) as input and return an (n,n) similarity matrix.
            Additional required parameters can be passed via a lambda function.
        definition : string
            Type of Laplacian
            'simple': Laplacian matrix for simple graphs L = D - A
            'norm_sym': Symmetric normalized Laplacian L^sym = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}
            'norm_rw': L^rw = D^{-1} L = I - D^{-1} A
        mode : "fc", "eNeighbour"
            How to calculate adjacency from the similarity matrix
            "fully_connected" is fully-connected, so A = S
            "eNeighbour" is the epsilon neighbourhood, with A_ji = 0 if S_ij </> lower/upper; for eNeighbour an upper or lower boundary needs to be set
        threshold_key : string
            "upper" or "lower", defining the type of threshold for the epsilon-neighrborhood
        threshold_value : float
            Boundary value for the epsilon-neighrborhood
        neighbours : int
            Number of neirest neighbors to be considered for adjacency definition. Currently not implemented
        Returns
        -------
        L : ht.DNDarray

        """
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

    def _normalized_symmetric_L(self, A):
        degree = ht.sum(A, axis=1)
        degree.resplit_(axis=None)
        # Find stand-alone vertices with no connections
        temp = torch.ones(
            degree.shape, dtype=degree._DNDarray__array.dtype, device=degree.device.torch_device
        )
        degree._DNDarray__array = torch.where(
            degree._DNDarray__array == 0, temp, degree._DNDarray__array
        )
        L = A / ht.sqrt(ht.expand_dims(degree, axis=1))
        L = L / ht.sqrt(ht.expand_dims(degree, axis=0))
        L = L * (-1.0)
        L.fill_diagonal(1.0)
        return L

    def _simple_L(self, A):
        degree = ht.sum(A, axis=1)
        L = ht.diag(degree) - A
        return L

    def construct(self, X):
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
