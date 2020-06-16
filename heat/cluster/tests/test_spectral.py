import os
import unittest

import heat as ht

if os.environ.get("DEVICE") == "gpu" and ht.torch.cuda.is_available():
    ht.use_device("gpu")
    ht.torch.cuda.set_device(ht.torch.device(ht.get_device().torch_device))
else:
    ht.use_device("cpu")
device = ht.get_device().torch_device
ht_device = None
if os.environ.get("DEVICE") == "lgpu" and ht.torch.cuda.is_available():
    device = ht.gpu.torch_device
    ht_device = ht.gpu
    ht.torch.cuda.set_device(device)


class TestSpectral(unittest.TestCase):
    def test_clusterer(self):
        spectral = ht.cluster.Spectral()
        self.assertTrue(ht.is_estimator(spectral))
        self.assertTrue(ht.is_clusterer(spectral))

    def test_get_and_set_params(self):
        spectral = ht.cluster.Spectral()
        params = spectral.get_params()

        self.assertEqual(
            params,
            {
                "n_clusters": None,
                "gamma": 1.0,
                "metric": "rbf",
                "laplacian": "fully_connected",
                "threshold": 1.0,
                "boundary": "upper",
                "n_lanczos": 300,
                "assign_labels": "kmeans",
            },
        )

        params["n_clusters"] = 10
        spectral.set_params(**params)
        self.assertEqual(10, spectral.n_clusters)

    def test_fit_iris(self):
        # get some test data
        iris = ht.load("heat/datasets/data/iris.csv", sep=";", split=0)
        m = 10

        # fit the clusters
        spectral = ht.cluster.Spectral(
            n_clusters=3, gamma=1.0, metric="rbf", laplacian="fully_connected", n_lanczos=m
        )
        spectral.fit(iris)
        self.assertIsInstance(spectral.labels_, ht.DNDarray)

        spectral = ht.cluster.Spectral(
            metric="euclidean", laplacian="eNeighbour", threshold=0.5, boundary="upper", n_lanczos=m
        )
        labels = spectral.fit_predict(iris)
        self.assertIsInstance(labels, ht.DNDarray)

        spectral = ht.cluster.Spectral(
            gamma=0.1,
            metric="rbf",
            laplacian="eNeighbour",
            threshold=0.5,
            boundary="upper",
            n_lanczos=m,
        )
        labels = spectral.fit_predict(iris)
        self.assertIsInstance(labels, ht.DNDarray)

        kmeans = {"kmeans++": "kmeans++", "max_iter": 30, "tol": -1}
        spectral = ht.cluster.Spectral(
            n_clusters=3, gamma=1.0, normalize=True, n_lanczos=m, params=kmeans
        )
        labels = spectral.fit_predict(iris)
        self.assertIsInstance(labels, ht.DNDarray)

        # Errors
        with self.assertRaises(NotImplementedError):
            spectral = ht.cluster.Spectral(metric="ahalanobis", n_lanczos=m)

        iris_split = ht.load("heat/datasets/data/iris.csv", sep=";", split=1)
        spectral = ht.cluster.Spectral(n_lanczos=20)
        with self.assertRaises(NotImplementedError):
            spectral.fit(iris_split)
