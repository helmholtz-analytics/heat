import os
import unittest

import heat as ht
import torch

from pathlib import Path
from heat.testing.basic_test import TestCase

try:
    import sklearn
    from sklearn.manifold import spectral_embedding as sklearn_spectral_embedding
except ImportError:
    sklearn = None


class TestSpectral(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.CSV_PATH = str(Path(ht.__file__).parent / "datasets" / "iris.csv")

    def test_clusterer(self):
        spectral = ht.cluster.SpectralClustering()
        self.assertTrue(ht.is_estimator(spectral))
        self.assertTrue(ht.is_clusterer(spectral))

    def test_get_and_set_params(self):
        spectral = ht.cluster.SpectralClustering()
        params = spectral.get_params()

        self.assertEqual(
            params,
            {
                "n_clusters": None,
                "eigen_solver": "randomized",
                "n_components": None,
                "random_state": None,
                "gamma": 1.0,
                "affinity": "rbf",
                "laplacian": "fully_connected",
                "threshold": 1.0,
                "boundary": "upper",
                "lanczos_n_iter": 300,
                "assign_labels": "kmeans",
                "reigh_rank": 100,
                "reigh_n_oversamples": 10,
                "reigh_power_iter": 0,
            },
        )

        params["n_clusters"] = 10
        spectral.set_params(**params)
        self.assertEqual(10, spectral.n_clusters)

    def test_fit_iris(self):
        if not self.is_mps:
            # get some test data
            iris = ht.load(self.CSV_PATH, sep=";", split=0)
            lanczosniter = 10
            reighrank = 10
            # fit the clusters
            spectral = ht.cluster.SpectralClustering(
                n_clusters=3, random_state=0, gamma=1.0, affinity="rbf", laplacian="fully_connected", eigen_solver="lanczos", lanczos_n_iter=lanczosniter
            )
            spectral.fit(iris)
            self.assertIsInstance(spectral.labels_, ht.DNDarray)

            spectral = ht.cluster.SpectralClustering(
                eigen_solver="randomized",
                affinity="euclidean",
                laplacian="eNeighbour",
                threshold=0.5,
                boundary="upper",
                reigh_rank=reighrank,
            )
            labels = spectral.fit_predict(iris)
            self.assertIsInstance(labels, ht.DNDarray)

            spectral = ht.cluster.SpectralClustering(
                n_clusters=3, gamma=1.0, init="kmeans++", max_iter=30, tol=0.0
            )
            labels = spectral.fit_predict(iris)
            self.assertIsInstance(labels, ht.DNDarray)

            spectral = ht.cluster.SpectralClustering(
                gamma=0.1,
                affinity="rbf",
                laplacian="eNeighbour",
                threshold=0.5,
                boundary="upper",
                eigen_solver="randomized",
                reigh_rank=5,
                reigh_n_oversamples=2,
                reigh_power_iter=1,
            )
            labels = spectral.fit_predict(iris)
            self.assertIsInstance(labels, ht.DNDarray)

            # Errors
            with self.assertRaises(NotImplementedError):
                spectral = ht.cluster.SpectralClustering(affinity="mahalanobis", lanczos_n_iter=2)

            with self.assertRaises(NotImplementedError):
                spectral = ht.cluster.SpectralClustering(eigen_solver="unknown")

            with self.assertRaises(ValueError):
                spectral = ht.cluster.SpectralClustering(eigen_solver="randomized", reigh_rank=2, n_clusters=5)

            iris_split = ht.load(self.CSV_PATH, sep=";", split=1)
            spectral = ht.cluster.SpectralClustering(eigen_solver="lanczos", lanczos_n_iter=20)
            with self.assertRaises(NotImplementedError):
                spectral.fit(iris_split)


    @unittest.skipIf(sklearn is None, "scikit-learn is not installed")
    def test_against_sklearn(self):
        if self.is_mps:
            return

        # 3 connected, asymmetric 2D clusters so the smallest eigenvalues are distinct
        torch.manual_seed(42)
        c0 = torch.randn(20, 2, dtype=torch.float64) + torch.tensor([0.0, 0.0])
        c1 = torch.randn(25, 2, dtype=torch.float64) + torch.tensor([3.0, 0.0])
        c2 = torch.randn(30, 2, dtype=torch.float64) + torch.tensor([0.0, 3.5])
        x_torch = torch.cat([c0, c1, c2], dim=0)
        x_ht = ht.array(x_torch, split=0)

        # Compare spectral embeddings on the same zero-diagonal affinity graph
        aff_ht = ht.spatial.rbf(x_ht, sigma=1.0, quadratic_expansion=True)
        aff_ht.fill_diagonal(0.0)
        aff_np = aff_ht.numpy()

        spectral = ht.cluster.SpectralClustering(
            n_clusters=3,
            affinity="precomputed",
            eigen_solver="randomized",
            reigh_rank=30,
            reigh_power_iter=20,
            random_state=0,
        )
        emb_ht = spectral._SpectralClustering__spectral_embedding(aff_ht, n_components=3)
        emb_sk = sklearn_spectral_embedding(
            aff_np, n_components=3, eigen_solver="arpack", random_state=0, drop_first=True
        )
        self.assertTrue(torch.allclose(emb_ht.resplit_(None).larray, torch.tensor(emb_sk), atol=1e-4))

        # Compare lanczos solver embedding against scikit-learn
        spectral_lanczos = ht.cluster.SpectralClustering(
            n_clusters=3,
            affinity="precomputed",
            eigen_solver="lanczos",
            lanczos_n_iter=60,
            random_state=0,
        )
        emb_lanczos = spectral_lanczos._SpectralClustering__spectral_embedding(
            aff_ht, n_components=3, eigen_solver="lanczos"
        )
        self.assertTrue(
            torch.allclose(emb_lanczos.resplit_(None).larray, torch.tensor(emb_sk), atol=1e-4)
        )

    def test_sklearn_api_compatibility(self):
        if self.is_mps:
            return

        iris = ht.load(self.CSV_PATH, sep=";", split=0)
        n_samples, n_features = iris.shape

        # __init__ should not mutate n_components when None
        spectral = ht.cluster.SpectralClustering(n_clusters=3, random_state=0)
        self.assertIsNone(spectral.get_params()["n_components"])

        # fit and fit_predict accept y and set n_features_in_ and affinity_matrix_
        dummy_y = ht.zeros((n_samples,), split=0)
        spectral.fit(iris, y=dummy_y)
        self.assertEqual(spectral.n_features_in_, n_features)
        self.assertIsInstance(spectral.affinity_matrix_, ht.DNDarray)
        self.assertEqual(spectral.affinity_matrix_.shape, (n_samples, n_samples))
        self.assertEqual(ht.sum(ht.abs(ht.diag(spectral.affinity_matrix_))).item(), 0.0)

        # Repeated fit calls with the same random_state produce identical labels
        labels_first = spectral.labels_.copy()
        labels_second = spectral.fit_predict(iris, y=dummy_y)
        self.assertTrue(ht.equal(labels_first, labels_second))

        # predict returns labels of expected length
        pred = spectral.predict(iris)
        self.assertIsInstance(pred, ht.DNDarray)
        self.assertEqual(pred.shape[0], n_samples)

        # Callable affinity matches equivalent built-in 'rbf' affinity
        gamma = 0.5
        sig = (1.0 / (2.0 * gamma)) ** 0.5
        spectral_rbf = ht.cluster.SpectralClustering(
            n_clusters=3, gamma=gamma, affinity="rbf", random_state=42
        ).fit(iris)
        spectral_call = ht.cluster.SpectralClustering(
            n_clusters=3,
            affinity=lambda x: ht.spatial.rbf(x, sigma=sig, quadratic_expansion=True),
            random_state=42,
        ).fit(iris)
        self.assertTrue(ht.allclose(spectral_rbf.affinity_matrix_, spectral_call.affinity_matrix_))
        self.assertTrue(ht.equal(spectral_rbf.labels_, spectral_call.labels_))

        # Precomputed affinity does not set n_features_in_
        spectral_pre = ht.cluster.SpectralClustering(
            n_clusters=3, affinity="precomputed", random_state=0
        )
        spectral_pre.fit(spectral_rbf.affinity_matrix_)
        self.assertFalse(hasattr(spectral_pre, "n_features_in_"))
        self.assertTrue(ht.allclose(spectral_pre.affinity_matrix_, spectral_rbf.affinity_matrix_))

    def test_input_validation(self):
        spectral = ht.cluster.SpectralClustering(n_clusters=3)

        with self.assertRaises(ValueError):
            spectral.fit(torch.randn(10, 2))

        with self.assertRaises(ValueError):
            spectral.fit(ht.zeros((10,), split=0))

        with self.assertRaises(ValueError):
            spectral.fit(ht.zeros((5, 4, 2), split=0))

        spectral_pre = ht.cluster.SpectralClustering(n_clusters=3, affinity="precomputed")
        with self.assertRaises(ValueError):
            spectral_pre.fit(ht.zeros((10, 8), split=0))

    def test_laplacian_return_affinity_and_isolated_node(self):
        if self.is_mps:
            return

        # Graph with 4 nodes where node 3 is isolated (zero similarity to all others)
        aff_torch = torch.tensor(
            [
                [1.0, 0.8, 0.2, 0.0],
                [0.8, 1.0, 0.5, 0.0],
                [0.2, 0.5, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        aff_ht = ht.array(aff_torch, split=0)

        lap = ht.graph.Laplacian(lambda x: x, definition="norm_sym")
        L, dd, S = lap.construct(aff_ht, return_diag=True, return_affinity=True)
        self.assertEqual(S.shape, (4, 4))
        self.assertEqual(ht.sum(ht.abs(ht.diag(S))).item(), 0.0)
        self.assertFalse(ht.isnan(L).any().item())
        self.assertAlmostEqual(dd.larray[3].item(), 1.0)
        self.assertAlmostEqual(L[3, 3].item(), 1.0)

        lap_simple = ht.graph.Laplacian(lambda x: x, definition="simple")
        L_simple, S_simple = lap_simple.construct(aff_ht, return_diag=False, return_affinity=True)
        self.assertEqual(L_simple.shape, (4, 4))
        self.assertTrue(ht.allclose(S, S_simple))
