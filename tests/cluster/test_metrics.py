import heat as ht
import numpy as np
from sklearn.metrics import silhouette_samples as sk_silhouette
from sklearn.metrics import silhouette_score as sk_silhouette_score
from sklearn.datasets import make_blobs
from heat.cluster.metrics import silhouette_samples, silhouette_score
from heat.testing.basic_test import TestCase


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


class TestSilhouette(TestCase):

    def test_silhouette_samples(self):
        n_points = [8, 17]
        n_cluster = [3, 5]
        dims = [1, 2, 3]
        splits=[None, 0]
        metric=['precomputed', 'euclidean']

        for metric in metric:
            for split in splits:
                for n_p in n_points:
                    for n_c in n_cluster:
                        for d in dims:
                            with self.subTest(f'{split=}, {n_p=}, {n_c=}, {d=}, {metric=}'):
                                data = ht.random.random((n_p, d), split=split)
                                labels = ht.random.randint(low=0, high=n_c, size=n_p, split=split)

                                if metric == 'precomputed':
                                    data = ht.spatial.cdist(data, data)

                                # Compute silhouette of all samples with sklearn and with heat
                                sk_results = sk_silhouette(data.resplit(None).numpy(), labels.resplit(None).numpy(), metric=metric)
                                ht_results = silhouette_samples(data, labels, metric=metric).resplit(None).numpy()

                                assert np.allclose(sk_results, ht_results,
                                                   atol=1e-6), f'Max diff between Heat and scipy: {np.max(np.abs(sk_results - ht_results))}'


    def test_special_cases(self):
        n = ht.comm.size * 4
        d = 3
        data = ht.random.random((n, d), split=0)
        labels = ht.arange(data.shape[0])

        # all elements in distinct clusters
        labels = ht.arange(data.shape[0])
        score = ht.cluster.silhouette_score(data, labels)
        assert np.isclose(score, 0), f'Non-zero {score=} even though all clusters have only a single element'

        # all elements in the same cluster
        labels[...] = 0
        score = ht.cluster.silhouette_score(data, labels)
        assert np.isclose(score, 0), f'Non-zero {score=} even though there is only one cluster'

        # perfect clustering
        data[...] = 0
        data[:n//2,0] = 0
        data[n//2:,0] = 1
        labels[:n//2] = 0
        labels[n//2:] = 1
        score = ht.cluster.silhouette_score(data, labels)
        assert np.isclose(score, 1), f'Non-one {score=} even though the clustering is perfect'

        # perfect, but strange clustering
        data[...] = 0
        data[:n//2,0] = 0
        data[n//2:,0] = 1
        labels[:n//2] = 0
        labels[n//2:] = 2
        score = ht.cluster.silhouette_score(data, labels)
        assert np.isclose(score, 1), f'Non-one {score=} even though the clustering is perfect'

        # worst possible clustering
        data[...] = 0
        data[:n//2,0] = 0
        data[n//2:,0] = 1
        labels = ht.arange(data.shape[0]) % 2
        score = ht.cluster.silhouette_score(data, labels)
        mean_inner_distance = (n/24)/(n/2-1)
        mean_outter_distance = (n/24)/(n/2)
        expect_score = mean_outter_distance / mean_inner_distance - 1
        assert np.isclose(score, expect_score), f'Unexpected {score=}!={expect_score} even though the clustering is wrong'
        assert score < 0, f'Unexpected {score=}>=0 even though the clustering is wrong'


    def test_minimal_silhouette(self):
        if self.comm.size > 3:
            self.skipTest('Matrix multiplication bug #2093')
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


    def test_silhouette_score_basic(self):
        if self.comm.size > 2:
            self.skipTest('Matrix multiplication bug #2093')
        X = ht.array([[1, 2], [1, 1], [4, 4], [4, 5]], split=0)
        labels = ht.array([0, 0, 1, 1], split=0)

        score = silhouette_score(X, labels)

        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0
        assert np.allclose(score,0.76439, atol=1e-9)


    def test_silhouette_sampling_determinism(self):
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


    def test_silhouette_sampling_stochasticity(self):
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


    def test_silhouette_precomputed_metric(self):
        if self.comm.size > 3:
            self.skipTest('Matrix multiplication bug #2093')
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

    def test_suite(self):
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
