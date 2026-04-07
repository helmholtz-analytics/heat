"""
Cluster metrics for the HeAT library.
"""

import heat as ht
import numpy as np


BORDER_LENGTH = 1e6  # value determines when array will be transformed to heat array


# Checks
def check_number_of_labels(n_labels, n_samples):
    """
    Check that number of labels are valid.

    Parameters
    ----------
    n_labels : int
        Number of labels.
    n_samples : int
        Number of samples.

    Examples
    --------
    >>> check_number_of_labels(2, 5)
    >>> check_number_of_labels(1, 5)
    ValueError: Number of labels is 1. Valid values are 2 to n_samples - 1 (inclusive)
    """
    if not 1 < n_labels < n_samples:
        raise ValueError(
            "Number of labels is %d. Valid values are 2 to n_samples - 1 (inclusive)" % n_labels
        )


def check_X_y(X, y, accept_sparse=False, metric="euclidean"):
    """
    Input validation for standard estimators.

    Parameters
    ----------
    X : {DNDarray, list, sparse matrix}
        Input data.
    y : {DNDarray, list, sparse matrix}
        Labels.
    accept_sparse : bool, optional
        Whether to accept sparse matrix input. Default is False.
    metric : str, optional
        The metric to use for validation. Default is "euclidean".

    Returns
    -------
    X : DNDarray
        The converted and validated X.
    y : DNDarray
        The converted and validated y.

    Examples
    --------
    >>> import heat as ht
    >>> X = ht.array([[1, 2], [3, 4]], dtype=ht.float)
    >>> y = ht.array([0, 1])
    >>> check_X_y(X, y)
    (DNDarray([[1., 2.], [3., 4.]], dtype=ht.float32, device=cpu:0, split=None),
     DNDarray([0, 1], dtype=ht.int64, device=cpu:0, split=None))
    """
    X = check_array(X, accept_sparse=accept_sparse, input_name="X", metric=metric)

    y = _check_y(y)

    check_consistent_length(X, y)

    return X, y


def check_array(X, accept_sparse=False, input_name="X", metric="euclidean"):
    """
    Input validation for a single array-like object.
    Converts input to a distributed HeAT DNDarray if dimensions exceed BORDER_LENGTH.

    Parameters
    ----------
    X : {DNDarray, array-like, sparse matrix}
        The input data to check.
    accept_sparse : bool, optional
        Whether to accept sparse matrix input. Default is False.
    input_name : str, optional
        The name of the input variable to use in error messages. Default is "X".
    metric : str, optional
        The metric to use. Default is "euclidean".

    Returns
    -------
    X : DNDarray
        The validated and potentially converted HeAT DNDarray.

    Raises
    ------
    TypeError
        If the input is sparse but `accept_sparse` is False.
    ValueError
        If `metric="precomputed"` and the diagonal contains non-zero elements.

    Examples
    --------
    >>> import heat as ht
    >>> X = [[0, 1], [1, 0]]
    >>> check_array(X, metric="precomputed")
    DNDarray([[0., 1.], [1., 0.]], dtype=ht.float32, device=cpu:0, split=0)
    """
    # Convert to heat array if input big enough --> overhead acceptable if array big enough
    if X.shape[0] > BORDER_LENGTH or X.shape[1] > BORDER_LENGTH:
        if not isinstance(X, ht.DNDarray):
            X = ht.array(X, split=0)

    if not accept_sparse and X.is_sparse:
        raise TypeError(f"{input_name} is sparse, but sparse input is not accepted.")

    if metric == "precomputed":
        error_msg = ValueError(
            "The precomputed distance matrix contains non-zero elements on the diagonal"
        )
        # mb write a function to fill diag with 0, like np.fill_diagonal(X, 0)
        diag_elements = ht.diag(X)

        atol = ht.finfo(X.dtype).eps * 100  # tolerance based on machine accuracy

        if ht.any(ht.abs(diag_elements) > atol):
            raise error_msg
        elif ht.any(diag_elements != 0):  # integral dtype
            raise error_msg
    return X


def _check_y(y):
    """
    Standard validation of labels
    Ensures y is a 1D DNDarray and handles reshaping of column vectors.

    Parameters
    ----------
    y : {DNDarray, array-like}
        The target values (labels).

    Returns
    -------
    y : DNDarray
        The validated and potentially converted HeAT DNDarray in 1D

    Raises
    ------
    ValueError
        If y is None or cannot be squeezed into a 1D representation.

    Examples
    --------
    >>> import heat as ht
    >>> y = ht.array([[1], [2], [3]])
    >>> _check_y(y)
    DNDarray([1, 2, 3], dtype=ht.int64, device=cpu:0, split=0)
    """
    if y.shape[0] > BORDER_LENGTH:
        if not isinstance(y, ht.DNDarray):
            y = ht.array(y, split=0)

    # need 1D labels
    if len(y.shape) > 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            raise ValueError("y should be a 1D array or a column vector.")

    if y is None:
        raise ValueError(" requires y to be passed, but the target y is None")

    return y


def check_consistent_length(X, y):
    """
    Check that X and y have a consistent number of samples (rows).
    For distributed DNDarrays, it ensures identical split axes and local chunk sizes.

    Parameters
    ----------
    X : DNDarray
        Input data.
    y : DNDarray
        Labels

    Raises
    ------
    ValueError
        If the global sample counts differ or if local distributed chunks
        are not aligned.

    Examples
    --------
    >>> import heat as ht
    >>> X = ht.zeros((10, 2), split=0)
    >>> y = ht.ones((10,), split=None)
    >>> check_consistent_length(X, y)
    # y is silently resplit to match X.split=0
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: [{X.shape[0]}, {y.shape[0]}]"
        )

    if isinstance(X, ht.DNDarray) and isinstance(y, ht.DNDarray):
        # ensure split axis is the same
        if X.split != y.split:
            y.resplit_(X.split)
        # Check if local chunks match
        if X.lshape[0] != y.lshape[0]:
            raise ValueError(
                "Local shapes of X and y do not match. Ensure they are partitioned identically."
            )


# Functions
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
        If `metric='precomputed'`, X is assumed to be a distance matrix.
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
    X, labels = check_X_y(X, labels, accept_sparse=["csr"], metric=metric)

    ht.sanitize_in(X)
    ht.sanitize_in(labels)

    unique_labels, labels_encoded = ht.unique(
        labels, return_inverse=True
    )  # f.e. labels = [10, 30, 20, 10], then unique_labels = [10,20,30] and labels_encoded = [0, 2, 1, 0]
    unique_labels.resplit_(None)
    # labels_encoded.resplit(None)
    labels_freqs = ht.bincount(labels_encoded)
    # labels_freqs.resplit_(None)
    n_samples = labels.shape[0]
    n_labels = unique_labels.shape[0]
    check_number_of_labels(n_labels, n_samples)

    if metric == "precomputed":
        D = X
    else:
        D = ht.spatial.cdist(X, X)

    # a(i) calculation
    a_mask = ht.reshape(labels_encoded, (1, -1)) == ht.reshape(
        labels, (-1, 1)
    )  # reshape((-1,1)) transposes labels_encoded
    a_mask = a_mask.astype(ht.float32)

    a_clust_dists = ht.sum(ht.mul(D, a_mask), axis=1)
    denominator_a = labels_freqs.larray[labels_encoded.larray] - 1
    denominator_a = ht.array(denominator_a, split=0)
    denominator_a = ht.where(
        denominator_a.astype(ht.float32) > 0,
        denominator_a,
        1.0,
    )

    a = ht.div(a_clust_dists, denominator_a)

    # b(i) calculation
    b_mask = ht.reshape(labels, (-1, 1)) == ht.reshape(unique_labels, (1, -1))
    full_b_mask = (labels_encoded.reshape((-1, 1)) == unique_labels.reshape((1, -1))).astype(
        ht.float32
    )

    b_clust_dists = ht.matmul(D, full_b_mask)

    # labels_freqs.resplit_(None)
    denominator_b = labels_freqs.astype(ht.float32)
    # denominator_b.resplit_(None)
    b_clust_means = b_clust_dists / denominator_b

    # we want neighbor cluster, so set the distance to points in own cluster to infinity
    b_clust_means.larray[b_mask.larray > 0] = float("inf")
    b = ht.min(b_clust_means, axis=1)

    sil_samples = ht.div(ht.sub(b, a), ht.maximum(a, b))

    sil_samples = ht.where(labels_freqs[labels_encoded] > 1, sil_samples, 0.0)
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
        X, labels = check_X_y(X, labels, accept_sparse=["csc", "csr"])  # same as silhouette_samples
        if random_state is not None:
            ht.random.seed(random_state)
        indices = random_state.permutation(X.shape[0])[
            :sample_size
        ]  # selects a subset of random samples, but all ranks need same indices

        if metric == "precomputed":  # input is distance matrix
            X, labels = X[indices].T[indices].T, labels[indices]
        else:
            X, labels = X[indices], labels[indices]
    return float(ht.mean(silhouette_samples(X, labels, metric=metric, **kwds)))
