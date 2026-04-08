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
    #score1 = silhouette_score(X, labels, sample_size=20, random_state=42)
    #score2 = silhouette_score(X, labels, sample_size=20, random_state=42)
    score1 = silhouette_score(X, labels, sample_size=5, random_state=42)
    score2 = silhouette_score(X, labels, sample_size=5, random_state=42)

    assert score1 == score2
    assert -1.0 <= score1 <= 1.0


def test_silhouette_sampling_stochasticity():
    # Test that random_state=None produces different scores (usually)
    X = ht.random.randn(100, 2, split=0)
    labels = ht.array([0] * 50 + [1] * 50, split=0)

    score1 = silhouette_score(X, labels, sample_size=10, random_state=None)
    score2 = silhouette_score(X, labels, sample_size=10, random_state=None)

    # While statistically possible to be equal, it's highly unlikely with enough data
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

if __name__ == "__main__":
    test_silhouette_implementation()
    test_silhouette_score_basic()
    test_silhouette_sampling_determinism()
    test_silhouette_sampling_stochasticity()
    test_silhouette_precomputed_metric()
    #test_minimal_silhouette()
