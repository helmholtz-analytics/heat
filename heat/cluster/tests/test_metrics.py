import heat as ht
import numpy as np
from sklearn.metrics import silhouette_samples as sk_silhouette
from sklearn.datasets import make_blobs
from heat.cluster.metrics import silhouette_samples


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

    ht_results_np = ht_results.numpy()

    if np.allclose(sk_results, ht_results_np, atol=1e-5):
        print("✅ Test Passed: HeAT matches Scikit-Learn results.")
    else:
        max_diff = np.max(np.abs(sk_results - ht_results_np))
        print(f"❌ Test Failed: Results differ. Max diff: {max_diff}")
        #print(f"sk_results are {np.abs(sk_results)}; ht_results are {np.abs(ht_results_np)}")

    # Single sample in a cluster
    labels_edge = np.array([0, 0, 0, 1])
    X_edge = np.random.rand(4, n_features)

    res_edge = silhouette_samples(ht.array(X_edge), ht.array(labels_edge))
    if res_edge[3] == 0:
        print("✅ Edge Case Passed: Single-sample cluster correctly assigned 0.0")


def test_minimal_silhouette():
    X_np = np.array([[0, 0], [10, 10], [20, 20], [1, 1]], dtype=np.float32)
    labels_np = np.array([0, 2, 1, 0], dtype=np.int32)

    ht_res = silhouette_samples(X_np, labels_np)

    res_np = ht_res.numpy()


    print(f"Labels: {labels_np}")
    print(f"HeAT Results: {res_np}")

        # Expected value for i=0 (Cluster 0)
        # a = dist((0,0), (1,1)) = 1.414
        # b = dist((0,0), (10,10)) = 14.14
        # sil = (14.14 - 1.414) / 14.14 = 0.9

    if res_np[0] > 0.8:
        print(f"✅ Success! Point 0 is {res_np[0]:.4f}")
    else:
        print(f"❌ Failure! Point 0 is {res_np[0]:.4f}")


if __name__ == "__main__":
    test_silhouette_implementation()
    #test_minimal_silhouette()
