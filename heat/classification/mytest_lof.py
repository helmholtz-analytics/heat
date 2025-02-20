"""Tests during the implementation of the Local Outlier Factor (LOF) algorithm"""

import heat as ht
import torch
from heat.spatial import distance

# a = ht.array([10, 20, 2, 17, 8], split=0)
# print(f"a={a}, \n b={b}, \n c={c}")

# y = ht.array([[2, 3, 1, 4], [5, 6, 4, 2], [7, 8, 9, 1]], split=0)
# o = ht.zeros([y.shape[0], y.shape[1]], split=0)

# values, indices=torch.topk(y.larray, 3)
# values, ydispl, _ = y.comm.counts_displs_shape(y.shape, y.split)
# process=ht.MPI_WORLD.rank
# global_idx=indices+ydispl[process]
# print(f"indices={indices}\n ydispl={ydispl}")
# print(f"indices+ydispl={global_idx}\n process= {process}")

# x = y.larray
# buffer = torch.zeros_like(x)
# o.larray[:] = x


# print(f"y.shape[0]={y.shape[0]}\n y.shape[1]={y.shape[1]}")
# print(f"process= {ht.MPI_WORLD.rank}\n o={o}\n buffer={buffer}")

# Create toy data
X = ht.array([[1.0, 1.0], [19.0, 19.0], [3.0, 3.0]], split=0)
# Y = ht.array([[0.0, 1.0], [0.0, 2.0], [100.0, 10.0], [100.0, 10.0]], split=0)
Y = ht.array(
    [
        [0.0, 1.0],
        [100.0, 100.0],
        [200.0, 200.0],
        [30.0, 30.0],
        [20.0, 20.0],
        [20.0, 0.0],
        [30.0, 30.0],
        [20.0, 20.0],
        [2.0, 1.0],
    ],
    split=0,
)


def test_cdist_small():
    """
    Testfunction for the cdist_small function.
    """
    # Compute pairwise distances with n_smallest = 2
    # print("execute cdist_small...\n")
    n_smallest = 4
    dist, indices = distance.cdist_small(X, Y, n_smallest=n_smallest)
    # print("finish executing cdist_small...\n")

    # print("Distances:\n", dist_np)
    # print("Indices:\n", indices_np)

    dist = dist.resplit_(None)

    # Manually compute expected distances
    # print("computing expected distances...\n")
    expected_distances = ht.spatial.cdist(X, Y)
    # print("computing expected indices...\n")
    expected_dist, expected_idx = ht.topk(expected_distances, n_smallest, largest=False)

    # print("validating results...\n")
    # Validate results
    print(f"process: {ht.MPI_WORLD.rank}, dist={dist}\n expected_dist={expected_dist}")
    print(f"process: {ht.MPI_WORLD.rank}, indices={indices}\n expected_idx={expected_idx}")

    assert ht.allclose(dist, expected_dist), "Distance matrix incorrect!"
    assert ht.equal(indices, expected_idx), "Index matrix incorrect!"
    print("Test passed successfully!")


# Run the test
# test_cdist_small()

# a = ht.array([0,10, 0], split=0)
# b = ht.array([[1,1,1], [2,2,2], [3,3,3], [4,4,4]], split=0)
# max=ht.maximum(a,b)
# print(f"process: {ht.MPI_WORLD.rank}, max={max}")

Y = ht.array(
    [
        [0.0, 1.0],
        [100.0, 100.0],
        [200.0, 200.0],
        [30.0, 30.0],
        [20.0, 20.0],
        [21.0, 0],
        [31.0, 0],
        [40.0, 40.0],
        [2.0, 1.0],
    ],
    split=0,
)
dist, indices = distance.cdist_small(Y, Y, n_smallest=3)


X = ht.array([[0], [4], [2]], split=0)  # Punkt 0  # Punkt 1  # Punkt 2

Y = ht.array(
    [[0], [3], [1], [100], [100], [100], [100], [100], [100]],  # Punkt 0  # Punkt 1  # Punkt 2
    split=0,
)
dist, indices = distance.cdist_small(X, Y, n_smallest=3, metric=distance._manhattan)
# print(f"process: {ht.MPI_WORLD.rank}, dist={dist}\n indices={indices}")


# k_dist=dist[:, -1]
# idx_k_dist=indices[:, -1]

# rank = X.comm.Get_rank()
# _, displ, _ = X.comm.counts_displs_shape(dist.shape, dist.split)

# idx_test=idx_k_dist-displ[rank]

# rd=ht.maximum(k_dist, dist[idx_k_dist,-1])

# k_dist=ht.array((3,4,2,5,4),split=0)
# idx_k_dist=ht.array((1,0,0,3,2),split=0)
# rd=ht.maximum(k_dist, k_dist[idx_k_dist])

# rank = k_dist.comm.Get_rank()
# _, displ, _ = k_dist.comm.counts_displs_shape(k_dist.shape, k_dist.split)
# idx_k_dist-=displ[rank]
# rd=ht.where(idx_k_dist<0,0,ht.maximum(k_dist, k_dist[idx_k_dist]))

# print(f"process: {ht.MPI_WORLD.rank} \n k_dist.larray={k_dist.larray}, \n  rd.larray={rd.larray}\n")

k_dist = ht.array((3, 4, 2, 5, 4, 1), split=0)
idx_k_dist = ht.array((1, 0, 0, 3, 2, 0), split=0)
k_dist_gathered = k_dist.resplit_(None)
k_dist_indexed = k_dist_gathered[idx_k_dist]
k_dist_indexed = k_dist_indexed.resplit_(0)
rd = ht.maximum(k_dist, k_dist[idx_k_dist])
print(f"process: {ht.MPI_WORLD.rank} \n  k_dist_indexed={k_dist_indexed}\n rd={rd}\n")

rd = ht.maximum(k_dist, k_dist_indexed)
