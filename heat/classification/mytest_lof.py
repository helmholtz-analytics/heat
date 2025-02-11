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


def test_cdist_small():
    """
    Testfunction for the cdist_small function.
    """
    # Create toy data
    X = ht.array([[1.0, 1.0], [19.0, 19.0], [3.0, 3.0]], split=0)
    # Y = ht.array([[0.0, 1.0], [0.0, 2.0], [100.0, 10.0], [100.0, 10.0]], split=0)
    Y = ht.array(
        [[0.0, 1.0], [100.0, 100.0], [200.0, 200.0], [30.0, 30.0], [20.0, 20.0], [2.0, 0.0]],
        split=0,
    )

    # Compute pairwise distances with n_smallest = 2
    # print("execute cdist_small...\n")
    n_smallest = 2
    dist, indices = distance.cdist_small(X, Y, n_smallest=n_smallest)
    # print("finish executing cdist_small...\n")

    # print("Distances:\n", dist_np)
    # print("Indices:\n", indices_np)

    dist = dist.resplit_(None)

    # Manually compute expected distances
    # print("computing expected distances...\n")
    expected_distances = ht.spatial.cdist(X, Y)
    # print("computing expected indices...\n")
    expected_dist, expected_idx = ht.topk(
        expected_distances, n_smallest, largest=False, sorted=False
    )

    # print("validating results...\n")
    # Validate results
    print(f"process: {ht.MPI_WORLD.rank}, dist={dist}\n expected_dist={expected_dist}")
    print(f"process: {ht.MPI_WORLD.rank}, indices={indices}\n expected_idx={expected_idx}")

    assert ht.allclose(dist, expected_dist), "Distance matrix incorrect!"
    assert ht.equal(indices, expected_idx), "Index matrix incorrect!"
    print("Test passed successfully!")


# Run the test
test_cdist_small()

# Y = ht.array([[0.0, 1.0], [100.0, 100.0], [200.0, 200.0], [30.0, 30.0], [20.0, 20.0]], split=0)
# lshap=Y.lshape_map[ht.MPI_WORLD.rank,0]
# print(f"process: {ht.MPI_WORLD.rank}, lshape={lshap}")
