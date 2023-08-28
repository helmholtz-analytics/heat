"""
Module Implementing the Kmedians Algorithm
"""
import heat as ht
from heat.cluster._kcluster import _KCluster
from heat.core.dndarray import DNDarray
from typing import Optional, Union, TypeVar


class KMedians(_KCluster):
    """
    K-Medians clustering algorithm [1].
    Uses the Manhattan (City-block, :math:`L_1`) metric for distance calculations

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of centroids to generate.
    init : str or DNDarray, default: ‘random’
        Method for initialization:

             - ‘k-medians++’ : selects initial cluster centers for the clustering in a smart way to speed up convergence [2].
             - ‘random’: choose k observations (rows) at random from data for the initial centroids.
             - DNDarray: gives the initial centers, should be of Shape = (n_clusters, n_features)
    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence.
    random_state : int
        Determines random number generation for centroid initialization.

    References
    -------------
    [1] Hakimi, S., and O. Kariv. "An algorithmic approach to network location problems II: The p-medians." SIAM Journal on Applied Mathematics 37.3 (1979): 539-560.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: Union[str, DNDarray] = "random",
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = None,
    ):
        if init == "kmedians++":
            init = "probability_based"

        super().__init__(
            metric=lambda x, y: ht.spatial.distance.manhattan(x, y, expand=True),
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )

    def _update_centroids(self, x: DNDarray, matching_centroids: DNDarray):
        """
        Compute coordinates of new centroid as median of the data points in ``x`` that are assigned to it

        Parameters
        ----------
        x :  DNDarray
            Input data
        matching_centroids : DNDarray
            Array filled with indeces ``i`` indicating to which cluster ``ci`` each sample point in x is assigned

        """
        new_cluster_centers = self._cluster_centers.copy()
        for i in range(self.n_clusters):
            # points in current cluster
            selection = (matching_centroids == i).astype(ht.int64)
            # Remove 0-element lines to avoid spoiling of median
            assigned_points = x * selection
            rows = assigned_points.abs().sum(axis=1) != 0
            local = assigned_points.larray[rows.larray]
            clean = ht.array(local, is_split=x.split)
            clean.balance_()
            # failsafe in case no point is assigned to this cluster
            # draw a random datapoint to continue/restart
            if clean.shape[0] == 0:
                _, displ, _ = x.comm.counts_displs_shape(shape=x.shape, axis=0)
                sample = ht.random.randint(0, x.shape[0]).item()
                proc = 0
                for p in range(x.comm.size):
                    if displ[p] > sample:
                        break
                    proc = p
                xi = ht.zeros(x.shape[1], dtype=x.dtype)
                if x.comm.rank == proc:
                    idx = sample - displ[proc]
                    xi = ht.array(x.lloc[idx, :], device=x.device, comm=x.comm)
                xi.comm.Bcast(xi, root=proc)
                new_cluster_centers[i, :] = xi
            else:
                if clean.shape[0] <= ht.MPI_WORLD.size:
                    clean.resplit_(axis=None)
                median = ht.median(clean, axis=0, keepdims=True)
                new_cluster_centers[i : i + 1, :] = median

        return new_cluster_centers

    def fit(self, x: DNDarray):
        """
        Computes the centroid of a k-medians clustering.

        Parameters
        ----------
        x : DNDarray
            Training instances to cluster. Shape = (n_samples, n_features)
        """
        # input sanitation
        if not isinstance(x, ht.DNDarray):
            raise ValueError(f"input needs to be a ht.DNDarray, but was {type(x)}")

        # initialize the clustering
        self._initialize_cluster_centers(x)
        self._n_iter = 0
        matching_centroids = ht.zeros((x.shape[0]), split=x.split, device=x.device, comm=x.comm)
        # iteratively fit the points to the centroids
        for epoch in range(self.max_iter):
            # increment the iteration count
            self._n_iter += 1
            # determine the centroids
            matching_centroids = self._assign_to_cluster(x)

            # update the centroids
            new_cluster_centers = self._update_centroids(x, matching_centroids)
            # check whether centroid movement has converged
            self._inertia = ((self._cluster_centers - new_cluster_centers) ** 2).sum()
            self._cluster_centers = new_cluster_centers.copy()
            if self.tol is not None and self._inertia <= self.tol:
                break

        self._labels = matching_centroids

        return self
