import heat as ht


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


def silhouette_samples(X, labels, *, metric="euclidean", **kwds):
    X_distributed = ht.array(X, split=0)
    labels_distributed = ht.array(labels, split=0)

    if metric == "precomputed":
        error_msg = ValueError(
            "The precomputed distance matrix contains non-zero elements on the diagonal"
        )
        # mb write a function to fill diag with 0, like np.fill_diagonal(X, 0)
        diag_elements = ht.diag(X_distributed)

        if X_distributed.dtype.kind == "f":
            atol = ht.finfo(X_distributed.dtype).eps * 100  # tolerance based on machine accuracy

            if ht.any(ht.abs(diag_elements) > atol):
                raise error_msg
        elif ht.any(diag_elements != 0):  # integral dtype
            raise error_msg

    unique_labels, labels_encoded = ht.unique(
        labels_distributed, return_inverse=True
    )  # f.e. labels = [10, 30, 20, 10], then unique_labels = [10,20,30] and labels_encoded = [0, 2, 1, 0]
    unique_labels.resplit_(None)
    labels_encoded.resplit(None)
    labels_freqs = ht.bincount(labels_encoded)
    labels_freqs.resplit_(None)
    n_samples = labels_distributed.shape[0]
    n_labels = unique_labels.shape[0]
    check_number_of_labels(n_labels, n_samples)

    rank = X_distributed.comm.rank
    # print(f"unique labels: {unique_labels} of Rank {rank}")
    # print(f"labels encoded: {labels_encoded} of Rank {rank}")
    print(f"[{rank}] labels_encoded (Local RAM): {labels_encoded.larray}")
    print(f"[{rank}] labels_distributed (Local RAM): {labels_distributed.larray}")
    print(f"[{rank}] unique_labels (Local RAM): {unique_labels.larray}")
    print(f"[{rank}] n_labels: {n_labels}")
    print(f"[{rank}] n_samples: {n_samples}")
    print(f"[{rank}] labels_freqs (Local RAM): {labels_freqs.larray}")
    # print(f"label_freqs: {labels_freqs} of Rank {rank}")

    if metric == "precomputed":
        D = X_distributed
    else:
        D = ht.spatial.cdist(X_distributed, X_distributed)

    print(f"[{rank}] D: \n{D.larray}")

    # a(i) calculation
    print(" ")
    # print("a calculation")
    a_mask = ht.reshape(labels_encoded, (1, -1)) == ht.reshape(
        labels_distributed, (-1, 1)
    )  # reshape((-1,1)) transposes labels_encoded
    a_mask = a_mask.astype(ht.float32)
    # print(f"  [{rank}] a_mask local rows: \n{a_mask.larray}")

    a_clust_dists = ht.sum(ht.mul(D, a_mask), axis=1)
    # print(f"  [{rank}] D*a_mask: \n{ht.mul(D,a_mask).larray}")
    denominator_a = labels_freqs.larray[labels_encoded.larray] - 1  # 0 check needed i guess
    denominator_a = ht.where(
        ht.array(denominator_a, split=0).astype(ht.float32) > 0,
        ht.array(denominator_a, split=0),
        1.0,
    )
    # print(f"  [{rank}] a_clust_dists rows: \n{a_clust_dists.larray}")
    # print(f"  [{rank}] denominator_a rows: \n{denominator_a}")

    a = ht.div(a_clust_dists, ht.array(denominator_a, split=0))
    print(f"  [{rank}] a: \n{a}")
    # print(" ")

    # b(i) calculation
    print(" ")
    print("b calculation")
    b_mask = ht.reshape(labels_distributed, (-1, 1)) == ht.reshape(unique_labels, (1, -1))
    # b_mask = b_mask.astype(ht.float32)
    full_b_mask = (labels_encoded.reshape((-1, 1)) == unique_labels.reshape((1, -1))).astype(
        ht.float32
    )

    print(f"  [{rank}] b_mask (Local RAM): \n{full_b_mask.larray}")

    b_clust_dists = ht.matmul(D, full_b_mask)
    print(f"  [{rank}] b_clust_dists (Local RAM): \n{b_clust_dists.larray}")
    # b_clust_means = ht.div(b_clust_dists, ht.bincount(labels_encoded))
    # denominator_b = labels_freqs.larray.reshape((1, -1))

    # Bug here
    labels_freqs.resplit_(None)
    # print(f"label_freqs: {labels_freqs} of Rank {rank}")
    print(f"label_freqs: {labels_freqs.larray} of Rank {rank}")
    denominator_b = labels_freqs.astype(ht.float32)
    denominator_b.resplit_(None)
    # b_clust_means = ht.div(b_clust_dists, denominator_b)
    b_clust_means = b_clust_dists / denominator_b

    print(f"  [{rank}] denominator_b: \n{denominator_b.larray}")
    # print(f"  [{rank}] denominator_b: \n{ht.array(denominator_b.larray, split=0)}")
    print(f"  [{rank}] b_clust_means (Local RAM) before inf: \n{b_clust_means.larray}")

    # we want neighbor cluster, so set the distance to points in own cluster to infinity
    # b_clust_means += b_mask*ht.inf

    # 2. Apply it using ht.where to avoid larray shape mismatches
    b_clust_means.larray[b_mask.larray > 0] = float("inf")
    print(f"  [{rank}] b_clust_means (Local RAM) after inf: \n{b_clust_means.larray}")
    b = ht.min(b_clust_means, axis=1)
    print(f"  [{rank}] b : \n{b}")

    sil_samples = ht.div(ht.sub(b, a), ht.maximum(a, b))

    sil_samples = ht.where(labels_freqs[labels_encoded] > 1, sil_samples, 0.0)
    sil_samples = ht.nan_to_num(sil_samples)

    return sil_samples
