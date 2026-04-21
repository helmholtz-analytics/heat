"""
Cluster metrics for the Heat library.
"""

import heat as ht
import numpy as np


def _validate_input(X, labels, metric="euclidean"):
    """
    Input validation for clustering metrics

    Parameters
    ----------
    X : {DNDarray, list}
        Input data.
    labels : {DNDarray, list}
        Labels.
    metric : str, optional
        The metric to use for validation. Default is "euclidean".

    Returns
    -------
    X : DNDarray
        The converted and validated X.
    labels : DNDarray
        The converted and validated labels.

    Examples
    --------
    >>> import heat as ht
    >>> X = ht.array([[1, 2], [3, 4]], dtype=ht.float)
    >>> labels = ht.array([0, 1])
    >>> _validate_input(X, labels)
    (DNDarray([[1., 2.], [3., 4.]], dtype=ht.float32, device=cpu:0, split=None),
     DNDarray([0, 1], dtype=ht.int64, device=cpu:0, split=None))
    """
    ht.sanitize_in(X)
    ht.sanitize_in(labels)

    # check precomputed distance matrix
    if metric == "precomputed":
        if X.ndim != 2 or X.shape[-1] != X.shape[-2]:
            raise ValueError(
                f"Precomputed distance matrix needs to be 2D and square but has shape {X.shape}"
            )

        diag_elements = ht.diag(X)

        if ht.any(ht.abs(diag_elements) > ht.finfo(X.dtype).eps * 100):
            raise ValueError(
                "The precomputed distance matrix contains non-zero elements on the diagonal"
            )

    # flatten labels
    if len(labels.shape) > 1:
        if np.prod(labels.shape) == X.shape[0]:
            labels = labels.flatten()
        else:
            raise ValueError(
                f"labels should be a 1D array or a column vector but has shape {labels.shape}."
            )

    # check for consistency between X and labels
    if X.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Found input variables with inconsistent number of samples and labels! Got: {X.shape[0]} samples and {labels.shape[0]} labels"
        )

    # ensure split axis is the same
    if X.split != labels.split:
        import warnings

        warnings.warn(
            f"labels are resplit from {labels.split} to match {X.split} to match split of data"
        )
        labels.resplit_(X.split)

    # Check if local chunks match
    if X.split is not None:
        if X.lshape[X.split] != labels.lshape[X.split]:
            raise ValueError(
                "Local shapes of X and labels do not match. Ensure they are partitioned identically."
            )

    return X, labels


def silhouette_samples(X, labels, *, metric="euclidean", **kwds):
    """
    Compute the Silhouette Coefficient for each sample.

    The Silhouette Coefficient is a measure of how close an object is to its own cluster
    (cohesion) compared to other clusters (separation).
    The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.

    * The score is 0 for clusters with only a single sample.
    * The calculation involves computing the mean intra-cluster distance (a) and
      the mean nearest-cluster distance (b) for each sample.

    Parameters
    ----------
    X : DNDarray
        An array of pairwise distances between samples, or a feature array.
        If `metric='precomputed'`, X is assumed to be a distance matrix and a feature array otherwise.
    labels : DNDarray
        Labels for each sample.
    metric : str, optional
        The metric to use when calculating distance between instances in a feature array.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Default is "euclidean".
    **kwds : optional
        Additional keyword arguments are ignored (reserved for compatibility).

    Notes
    -----
    The Silhouette Coefficient $s(i)$ for a single sample is defined as:
    $$s(i) = \frac{b(i) - a(i)}{\\max(a(i), b(i))}$$
    where $a(i)$ is the mean distance to other samples in the same cluster and $b(i)$
    is the mean distance to samples in the nearest neighbor cluster.

    Raises
    ------
    ValueError
        If the number of labels is 1 or equal to the number of samples.
        If `metric='precomputed'` and the diagonal contains non-zero elements.

    See Also
    --------
    silhouette_score : Average silhouette coefficient over all samples.

    Examples
    --------
    >>> import heat as ht
    >>> X = ht.array([[1, 2], [1, 1], [4, 4], [4, 5]], split=0)
    >>> labels = ht.array([0, 0, 1, 1], split=0)
    >>> ht.cluster.silhouette_samples(X, labels)
    DNDarray([0.7452, 0.7836, 0.7452, 0.7836], dtype=ht.float64, device=cpu:0, split=0)
    """
    # Sanitation and checks
    X, labels = _validate_input(X, labels, metric=metric)

    unique_labels, labels_encoded = ht.unique(
        labels, return_inverse=True
    )  # f.e. labels = [10, 30, 20, 10], then unique_labels = [10,20,30] and labels_encoded = [0, 2, 1, 0]
    unique_labels.resplit_(None)

    labels_freqs = ht.bincount(labels_encoded)

    if metric == "precomputed":
        D = X
    elif metric == "euclidean":
        D = ht.spatial.cdist(X, X)
    else:
        raise NotImplementedError(
            f"{metric=} not implemented, choose from 'precomputed' and 'euclidean'."
        )

    # a(i) calculation
    a_mask = ht.reshape(labels, (1, -1)) == ht.reshape(
        labels, (-1, 1)
    )  # reshape((-1,1)) transposes labels_encoded
    a_mask = a_mask

    a_clust_dists = ht.sum(ht.mul(D, a_mask), axis=1)
    denominator_a = labels_freqs[labels_encoded] - 1
    denominator_a = ht.where(
        denominator_a > 0,
        denominator_a,
        1,
    )

    a = ht.div(a_clust_dists, denominator_a)

    # b(i) calculation
    b_mask = ht.reshape(labels, (-1, 1)) == ht.reshape(unique_labels, (1, -1))
    full_b_mask = labels.reshape((-1, 1)) == unique_labels.reshape((1, -1))

    b_clust_dists = ht.matmul(D, full_b_mask)

    denominator_b = labels_freqs
    b_clust_means = ht.div(b_clust_dists, denominator_b)

    # we want neighboring clusters, so set the distance to points in own cluster to infinity
    b_clust_means.larray[b_mask.larray > 0] = float("inf")
    b = ht.min(b_clust_means, axis=1)

    sil_samples = ht.div(ht.sub(b, a), ht.maximum(a, b))

    sil_samples = ht.where(labels_freqs[labels_encoded] > 1, sil_samples, 0)
    sil_samples = ht.nan_to_num(sil_samples)

    return sil_samples


def silhouette_score(X, labels, *, metric="euclidean", sample_size=None, random_state=None, **kwds):
    r"""
    Compute the mean Silhouette Coefficient of all samples.

    The Silhouette Coefficient is calculated using the mean intra-cluster distance (a)
    and the mean nearest-cluster distance (b) for each sample. The Silhouette
    Coefficient for a sample is $(b - a) / \max(a, b)$.

    * This function returns the average of `silhouette_samples`.
    * To clarify, $b$ is the distance between a sample and the nearest cluster that
      the sample is not a part of.

    Parameters
    ----------
    X : DNDarray
        An array of pairwise distances between samples, or a feature array.
    labels : DNDarray
        Labels for each sample.
    metric : str, optional
        The metric to use when calculating distance between instances in a feature array.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Default is "euclidean".
    sample_size : int, optional
        The size of the sample to use when computing the Silhouette Coefficient on a random subset of the data.
        If ``sample_size is None``, no sampling is used.
    random_state : int, optional
        Determines random number generation for selecting a subset of samples.
        Used when `sample_size` is not `None`.
    **kwds : optional
        Additional keyword arguments passed to `silhouette_samples`.

    Notes
    -----
    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters.

    See Also
    --------
    silhouette_samples : Silhouette Coefficient for each individual sample.

    Examples
    --------
    >>> import heat as ht
    >>> X = ht.array([[1, 2], [1, 1], [4, 4], [4, 5]], split=0)
    >>> labels = ht.array([0, 0, 1, 1], split=0)
    >>> ht.cluster.silhouette_score(X, labels)
    0.76439
    """
    if sample_size is not None:
        X, labels = _validate_input(X, labels)  # same as silhouette_samples
        if random_state is not None:
            ht.random.seed(random_state)
        indices = ht.random.permutation(X.shape[0])[
            :sample_size
        ]  # selects a subset of random samples, but all ranks need same index array

        labels = labels[indices]
        if metric == "precomputed":  # input is distance matrix
            raise NotImplementedError(
                "Random sampling in silhouette with precomputed distance matrix is not currently supported. Please open an issue on GitHub if you need this feature."
            )
            X = X[indices].T[indices].T
        else:
            X = X[indices]

        X.balance_()
        labels.balance_()

    return float(ht.mean(silhouette_samples(X, labels, metric=metric, **kwds)))
