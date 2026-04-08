import heat as ht
import numpy as np
from sklearn.metrics import silhouette_samples as sk_silhouette
from sklearn.metrics import silhouette_score as sk_silhouette_score
from sklearn.datasets import make_blobs
from heat.cluster.metrics import silhouette_samples, silhouette_score


def test_silhouette_implementation():
    n_samples = 10
    n_features = 5
    centers = 3

    X_np, labels_np = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=42)

    # Reference values from Scikit-Learn
    sk_results = sk_silhouette(X_np, labels_np)

    # Heat values
    X_ht = ht.array(X_np, split=0)
    labels_ht = ht.array(labels_np, split=0)

    ht_results = silhouette_samples(X_ht, labels_ht)

    # Comparison

    ht_results_np = ht_results.resplit(None).numpy()
    #print(f"Labels: {labels_np}")
    #print(f"HeAT Results: {ht_results}")
    #print(f"SK Results: {sk_results}")

    assert np.allclose(sk_results, ht_results_np, atol=1e-9), f'Max diff between Heat and scipy: np.max(np.abs(sk_results - ht_results_np))'



    # Single sample in a cluster
    labels_edge = np.array([0, 0, 0, 1])
    X_edge = np.random.rand(4, n_features)

    res_edge = silhouette_samples(ht.array(X_edge), ht.array(labels_edge))
    assert res_edge[3] == 0


def test_minimal_silhouette():
    X_np = np.array([[0, 0], [10, 10], [20, 20], [1, 1]], dtype=np.float32)
    labels_np = np.array([0, 2, 1, 0], dtype=np.int32)

    # HeAT values
    X_ht = ht.array(X_np, split=0)
    labels_ht = ht.array(labels_np, split=0)

    ht_res = silhouette_samples(X_ht, labels_ht)

    res_np = ht_res.numpy()

    sk_results = sk_silhouette(X_np, labels_np)

    #print(f"Labels: {labels_np}")
    #print(f"HeAT Results: {res_np}")
    #print(f"SK Results: {sk_results}")

        # Expected value for i=0 (Cluster 0)
        # a = dist((0,0), (1,1)) = 1.414
        # b = dist((0,0), (10,10)) = 14.14
        # sil = (14.14 - 1.414) / 14.14 = 0.9

    assert res_np[0] > 0.8, f"Point 0 is {res_np[0]:.4f}"


def test_silhouette_score_basic():
    X = ht.array([[1, 2], [1, 1], [4, 4], [4, 5]], split=0)
    labels = ht.array([0, 0, 1, 1], split=0)

    score = silhouette_score(X, labels)

    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0
    assert np.allclose(score,0.76439, atol=1e-9)


def test_silhouette_sampling_determinism():
    n_samples = 1000
    n_features = 2
    centers = 3

    # Generate stable data
    X_np, labels_np = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        random_state=42
    )

    # Convert to distributed Heat arrays
    X = ht.array(X_np, split=0)
    labels = ht.array(labels_np, split=0)

    # Run deterministic tests
    score1 = silhouette_score(X, labels, sample_size=20, random_state=42)
    score2 = silhouette_score(X, labels, sample_size=20, random_state=42)

    assert score1 == score2
    assert -1.0 <= score1 <= 1.0


def test_silhouette_sampling_stochasticity():
    # Test that random_state=None produces different scores (usually)
    n_samples = 1000
    n_features = 2
    centers = 3

    X_np, labels_np = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        random_state=None
    )

    # Convert to distributed Heat arrays
    X = ht.array(X_np, split=0)
    labels = ht.array(labels_np, split=0)

    score1 = silhouette_score(X, labels, sample_size=20, random_state=None)
    score2 = silhouette_score(X, labels, sample_size=20, random_state=None)


    assert score1 != score2


def test_silhouette_precomputed_metric():
    # X as a distance matrix
    X_dist = ht.array([
        [0.0, 1.0, 5.0, 5.0],
        [1.0, 0.0, 5.0, 5.0],
        [5.0, 5.0, 0.0, 1.0],
        [5.0, 5.0, 1.0, 0.0]
    ], split=0)
    labels = ht.array([0, 0, 1, 1], split=0)

    score = silhouette_score(X_dist, labels, metric="precomputed")
    assert score > 0.5  # Well separated clusters

def run_error_test(name, func, expected_exception, match_text):
    """
    Helper to run a test and verify it raises the correct error.
    """
    rank = ht.communication.MPI_WORLD.rank
    try:
        func()
        if rank == 0:
            print(f"FAILED: {name} (No exception raised)")
    except expected_exception as e:
        if match_text in str(e):
            if rank == 0:
                print(f"PASSED: {name}")
        else:
            if rank == 0:
                print(f"FAILED: {name} (Wrong error message: {e})")
    except Exception as e:
        if rank == 0:
            print(f"FAILED: {name} (Wrong exception type: {type(e).__name__}: {e})")

def test_suite():
    rank = ht.communication.MPI_WORLD.rank
    if rank == 0:
        print(f"--- Starting Validation Tests on {ht.communication.MPI_WORLD.size} Ranks ---")

    # 1. Number of labels too small (n_labels = 1)
    def test_single_label():
        X = ht.zeros((10, 2), split=0)
        labels = ht.zeros((10,), split=0)
        silhouette_score(X, labels)
    run_error_test("Single Label Error", test_single_label, ValueError, "Number of labels is 1")

    # 2. Number of labels too large (n_labels = n_samples)
    def test_too_many_labels():
        X = ht.zeros((10, 2), split=0)
        labels = ht.arange(10, split=0)
        silhouette_score(X, labels)
    run_error_test("N-Labels Error", test_too_many_labels, ValueError, "Valid values are 2 to n_samples - 1")

    # 3. Inconsistent lengths
    def test_mismatched_lengths():
        X = ht.zeros((10, 2), split=0)
        labels = ht.zeros((5,), split=0)
        silhouette_score(X, labels)
    run_error_test("Consistent Length Error", test_mismatched_lengths, ValueError, "inconsistent numbers")

    # 4. Precomputed Diagonal Check (Floats)
    def test_nonzero_diagonal():
        # Diagonal is 0.5, which is > atol
        X = ht.eye(4, split=0) * 0.5 #creates DNDarray with non-zero diagonal and zeros elsewhere
        labels = ht.array([0, 0, 1, 1], split=0)
        silhouette_score(X, labels, metric="precomputed")
    run_error_test("Float Diagonal Error", test_nonzero_diagonal, ValueError, "non-zero elements on the diagonal")

    # 5. Invalid Label Shape
    def test_label_shape():
        X = ht.zeros((4, 2), split=0)
        labels = ht.zeros((4, 2), split=0)
        silhouette_score(X, labels)
    run_error_test("Label Shape Error", test_label_shape, ValueError, "y should be a 1D array")

if __name__ == "__main__":
    test_silhouette_implementation()
    test_silhouette_score_basic()
    test_silhouette_sampling_determinism()
    test_silhouette_sampling_stochasticity()
    test_silhouette_precomputed_metric()
    #test_minimal_silhouette()
    test_suite()
