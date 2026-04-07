import heat as ht
import numpy as np
from sklearn.metrics import silhouette_samples as sk_silhouette
from sklearn.metrics import calinski_harabasz_score as sk_calinski_harabasz_score
from sklearn.metrics import silhouette_score as sk_silhouette_score
from sklearn.datasets import make_blobs
from heat.cluster.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score


def test_silhouette_implementation():
    n_samples = 100
    n_features = 5
    centers = 3
    X_np, labels_np = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=42)

    # Reference values from Scikit-Learn
    sk_results = sk_silhouette(X_np, labels_np)

    # HeAT values
    X_ht = ht.array(X_np, split=0)
    labels_ht = ht.array(labels_np, split=0)

    ht_results = silhouette_samples(X_ht, labels_ht)

    # Comparison

    ht_results_np = ht_results.resplit(None).numpy()

    assert np.allclose(sk_results, ht_results_np, atol=1e-5), f'Max diff between Heat and scipy: np.max(np.abs(sk_results - ht_results_np))'

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

    print(f"Labels: {labels_np}")
    print(f"HeAT Results: {res_np}")
    print(f"SK Results: {sk_results}")

        # Expected value for i=0 (Cluster 0)
        # a = dist((0,0), (1,1)) = 1.414
        # b = dist((0,0), (10,10)) = 14.14
        # sil = (14.14 - 1.414) / 14.14 = 0.9

    assert res_np[0] > 0.8, f"Point 0 is {res_np[0]:.4f}"

def test_calinski_harabasz_implementation():
    n_samples = 1000
    n_features = 5
    k = 3

    X_np, labels_np = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=k,
        random_state=42
    )

    X_ht = ht.array(X_np, split=0)
    labels_ht = ht.array(labels_np, split=0)

    # Reference values from Scikit-Learn
    sk_result = sk_calinski_harabasz_score(X_np, labels_np)

    # Heat values

    ht_result = calinski_harabasz_score(X_ht, labels_ht, k)

    # Comparison
    if isinstance(ht_result, ht.DNDarray):
        ht_result = ht_result.item()

    assert np.allclose(sk_result, ht_result, atol=1e-5), \
        f"Mismatch with Scikit-Learn! SK: {sk_result}, HT: {ht_result}"

    print(f"Test Passed: SK={sk_result:.4f}, HT={ht_result:.4f}")

    # Edge Case: 1 Cluster
    labels_single = ht.zeros((n_samples,), dtype=ht.int32)
    res_single = calinski_harabasz_score(X_ht, labels_single, 1)
    assert res_single == 0, "CH Score should be 0 for a single cluster."


def test_calinski_harabasz_spherical(): # works only with mpirun -n 1
    n= 1000
    n_features = 5
    k = 4
    seed = 1

    X_ht = ht.utils.data.spherical.create_spherical_dataset(
        num_samples_cluster=n,
        radius=1.0,
        offset=4.0,
        dtype=ht.float32,
        random_state=seed
    )
    km = ht.cluster.KMeans(n_clusters=4)
    km.fit(X_ht)
    labels_ht = km.labels_

    X_np = X_ht.numpy()
    labels_np = labels_ht.numpy().flatten()

    #Sklearn
    sk_result = sk_calinski_harabasz_score(X_np, labels_np)

    # Heat
    ht_result = calinski_harabasz_score(X_ht, labels_ht, k)

    # Comparison
    if isinstance(ht_result, ht.DNDarray):
        ht_result = ht_result.item()

    assert np.allclose(sk_result, ht_result, atol=1e-5), \
        f"Mismatch! SK: {sk_result}, HT: {ht_result}"

    print(f"Test Passed: SK={sk_result:.4f}, HT={ht_result:.4f}")

    # Edge Case: 1 Cluster
    labels_single = ht.zeros((n*k,), dtype=ht.int32, split=0)
    res_single = calinski_harabasz_score(X_ht, labels_single, 1)
    if isinstance(res_single, ht.DNDarray):
        res_single = res_single.item()

    assert res_single == 0, "CH Score should be 0 for a single cluster."




if __name__ == "__main__":
    #test_silhouette_implementation()
    #test_minimal_silhouette()
    test_calinski_harabasz_implementation()
    #test_calinski_harabasz_spherical()
