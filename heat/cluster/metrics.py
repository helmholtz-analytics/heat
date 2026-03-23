import heat as ht
import numpy as np


# Checks
def check_number_of_labels(n_labels, n_samples):
    """Check that number of labels are valid.

    Parameters
    ----------
    n_labels : int
        Number of labels.

    n_samples : int
        Number of samples.
    """
    if not 1 < n_labels < n_samples:
        raise ValueError(
            "Number of labels is %d. Valid values are 2 to n_samples - 1 (inclusive)" % n_labels
        )


def check_X_y(X, y, accept_sparse=False):
    """Input validation for standard estimators.

    Parameters
    ----------
    X : {DNDarray, list, sparse matrix}
        Input data.

    y : {DNDarray, list, sparse matrix}
        Labels.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    y_converted : object
        The converted and validated y.
    """
    X = check_array(
        X,
        accept_sparse=accept_sparse,
        input_name="X",
    )

    y = _check_y(y)

    check_consistent_length(X, y)

    return X, y


def check_array(X, accept_sparse=False, input_name="X"):
    if not isinstance(X, ht.DNDarray):
        # Convert to heat array
        X = ht.array(X, split=0)

    if not accept_sparse and X.is_sparse:
        raise TypeError(f"{input_name} is sparse, but sparse input is not accepted.")

    return X


def _check_y(y):
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
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: [{X.shape[0]}, {y.shape[0]}]"
        )

    # ensure split axis is the same
    if X.split != y.split:
        y.resplit_(X.split)

    # Check if local chunks match
    if X.lshape[0] != y.lshape[0]:
        raise ValueError(
            "Local shapes of X and y do not match. Ensure they are partitioned identically."
        )


def check_random_state(seed):
    if seed is None:
        return ht.random

    if isinstance(seed, (int, np.integer)):
        ht.random.seed(int(seed))
        return ht.random

    raise ValueError(f"Type {type(seed)} cannot be used to seed ht.random")


# Functions
def silhouette_samples(X, labels, *, metric="euclidean", **kwds):
    X, labels = check_X_y(
        X, labels, accept_sparse=["csr"]
    )  # think about accept_sparse, i have no idea what it is and what csr means

    ht.sanitize_in(X)
    ht.sanitize_in(labels)

    #X_distributed = ht.array(X, split=0)
    #labels_distributed = ht.array(labels, split=0)




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

    unique_labels, labels_encoded = ht.unique(
        labels, return_inverse=True
    )  # f.e. labels = [10, 30, 20, 10], then unique_labels = [10,20,30] and labels_encoded = [0, 2, 1, 0]
    unique_labels.resplit_(None)
    labels_encoded.resplit(None)
    labels_freqs = ht.bincount(labels_encoded)
    labels_freqs.resplit_(None)
    n_samples = labels.shape[0]
    n_labels = unique_labels.shape[0]
    check_number_of_labels(n_labels, n_samples)

    rank = X.comm.rank
    """
    print(f"[{rank}] labels_encoded (Local RAM): {labels_encoded.larray}")
    print(f"[{rank}] labels_distributed (Local RAM): {labels_distributed.larray}")
    print(f"[{rank}] unique_labels (Local RAM): {unique_labels.larray}")
    print(f"[{rank}] n_labels: {n_labels}")
    print(f"[{rank}] n_samples: {n_samples}")
    print(f"[{rank}] labels_freqs (Local RAM): {labels_freqs.larray}")
    """

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
    denominator_a = ht.where(
        ht.array(denominator_a, split=0).astype(ht.float32) > 0,
        ht.array(denominator_a, split=0),
        1.0,
    )

    a = ht.div(a_clust_dists, ht.array(denominator_a, split=0))

    # b(i) calculation
    b_mask = ht.reshape(labels, (-1, 1)) == ht.reshape(unique_labels, (1, -1))
    full_b_mask = (labels_encoded.reshape((-1, 1)) == unique_labels.reshape((1, -1))).astype(
        ht.float32
    )

    b_clust_dists = ht.matmul(D, full_b_mask)
    labels_freqs.resplit_(None)
    denominator_b = labels_freqs.astype(ht.float32)
    denominator_b.resplit_(None)
    b_clust_means = b_clust_dists / denominator_b

    # we want neighbor cluster, so set the distance to points in own cluster to infinity
    b_clust_means.larray[b_mask.larray > 0] = float("inf")
    b = ht.min(b_clust_means, axis=1)

    sil_samples = ht.div(ht.sub(b, a), ht.maximum(a, b))

    sil_samples = ht.where(labels_freqs[labels_encoded] > 1, sil_samples, 0.0)
    sil_samples = ht.nan_to_num(sil_samples)

    return sil_samples


def silhouette_score(X, labels, *, metric="euclidean", sample_size=None, random_state=None, **kwds):
    if sample_size is not None:
        X, labels = check_X_y(X, labels, accept_sparse=["csc", "csr"])  # same as silhouette_samples
        random_state = check_random_state(random_state)
        indices = random_state.permutation(X.shape[0])[
            :sample_size
        ]  # selecs a subset of random samples, but all ranks need same indices

        if metric == "precomputed":  # precomputed means here distance matrix for some reason?
            X, labels = X[indices].T[indices].T, labels[indices]
        else:
            X, labels = X[indices], labels[indices]
    return float(ht.mean(silhouette_samples(X, labels, metric=metric, **kwds)))
