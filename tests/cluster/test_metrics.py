import pytest
pytest.importorskip("sklearn")

import heat as ht
import numpy as np
from sklearn.metrics import silhouette_samples as sk_silhouette
from sklearn.metrics import silhouette_score as sk_silhouette_score
from sklearn.datasets import make_blobs
from heat.cluster.metrics import silhouette_samples, silhouette_score
from heat.testing.basic_test import TestCase


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
                    if n_p / ht.comm.size < 2:
                        self.skipTest('Matrix multiplication bug #2093')
                    for n_c in n_cluster:
                        for d in dims:
                            with self.subTest(f'{split=}, {n_p=}, {n_c=}, {d=}, {metric=}'):
                                data = ht.random.random((n_p, d), split=split)
                                labels = ht.random.randint(low=0, high=n_c, size=n_p, split=split)

                                if metric == 'precomputed':
                                    data = ht.spatial.cdist(data, data)

                                # Compute silhouette of all samples with sklearn and with heat
                                sk_results = sk_silhouette(data.numpy(), labels.numpy(), metric=metric)
                                ht_results = silhouette_samples(data, labels, metric=metric).numpy()

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


    def test_minimal_silhouette_example(self):
        if self.comm.size > 3:
            self.skipTest('Matrix multiplication bug #2093')
        X = ht.array([[0, 0], [10, 10], [20, 20], [1, 1]], dtype=np.float32, split=0)
        labels = ht.array([0, 2, 1, 0], dtype=np.int32, split=0)

        sil = silhouette_samples(X, labels)

        # Expected value for i=0 (Cluster 0)
        # a = dist((0,0), (1,1)) = 1.414
        # b = dist((0,0), (10,10)) = 14.14
        # sil = (14.14 - 1.414) / 14.14 = 0.9

        if self.comm.rank == 0:
            assert sil.larray[0] == pytest.approx(0.9), f"Point 0 is {sil.larray[0]:.4f}"


    def test_minimal_silhouette_score_example(self):
        if self.comm.size > 2:
            self.skipTest('Matrix multiplication bug #2093')
        X = ht.array([[1, 2], [1, 1], [4, 4], [4, 5]], split=0)
        labels = ht.array([0, 0, 1, 1], split=0)

        score = silhouette_score(X, labels)

        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0
        assert np.allclose(score,0.76439, atol=1e-9)

    def test_silhouette_random_sampling(self):
        n_points = 64
        n_samples = [8, 17, 64]
        n_cluster = [3, 5]
        dims = [1, 2, 3]
        splits=[None, 0]
        metric=['euclidean']

        for metric in metric:
            for split in splits:
                for n_s in n_samples:
                    for n_c in n_cluster:
                        for d in dims:
                            with self.subTest(f'{split=}, {n_s=}, {n_c=}, {d=}, {metric=}'):
                                data = ht.random.random((n_points, d), split=split)
                                labels = ht.random.randint(low=0, high=n_c, size=n_points, split=split)

                                if metric == 'precomputed':
                                    data = ht.spatial.cdist(data, data)

                                # Compute silhouette of all samples with sklearn and with heat
                                score1 = silhouette_score(data, labels, metric=metric, sample_size=n_s, random_state=42)
                                score2 = silhouette_score(data, labels, metric=metric, sample_size=n_s, random_state=42)
                                score3 = silhouette_score(data, labels, metric=metric, sample_size=n_s, random_state=43)

                                assert -1 <= score1 <= 1
                                assert np.isclose(score1, score2)

                                if n_s == n_points:
                                    score_all = silhouette_score(data, labels, metric=metric)
                                    assert np.isclose(score1, score_all)
                                else:
                                    assert not np.isclose(score1, score3)


    def test_input_validation(self):
        # inconsistent number of labels and samples
        for n_labels in [1, 11]:
            X = ht.zeros((10, 2), split=None)
            labels = ht.zeros((1,), split=None)
            with self.assertRaisesRegex(ValueError, "inconsistent number of samples and labels"):
                silhouette_score(X, labels)

        # invalid label shape
        X = ht.zeros((4, 2), split=None)
        labels = ht.zeros((4, 2), split=None)
        with self.assertRaisesRegex(ValueError, "labels should be a 1D array"):
            silhouette_score(X, labels)

        # distance matrix with non-zero diagonal elements
        X = ht.eye(4, split=0) * 0.5 #creates DNDarray with non-zero diagonal and zeros elsewhere
        labels = ht.array([0, 0, 1, 1], split=0)
        with self.assertRaisesRegex(ValueError, "non-zero elements on the diagonal"):
            silhouette_score(X, labels, metric="precomputed")

        for shape in [(2, 2, 2), (4, 6)]:
            X = ht.zeros((4, 6), split=0)
            labels = ht.array([0, 0, 1, 1], split=None)
            with self.assertRaisesRegex(ValueError, "Precomputed distance matrix needs to be 2D and square"):
                silhouette_score(X, labels, metric="precomputed")
