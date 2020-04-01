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


class TestKMeans(unittest.TestCase):
    def test_laplacian(self):
        S = ht.array(
            [
                [0, 0.6, 0.8, 0.2, 0.2, 0.1, 0.4, 0.2, 0.6, 0.7],
                [0.6, 0, 0.6, 0.3, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2],
                [0.8, 0.6, 0, 0.2, 0.1, 0.3, 0.4, 0.3, 0.2, 0.2],
                [0.2, 0.3, 0.2, 0, 0.9, 0.9, 0.4, 0.3, 0.3, 0.4],
                [0.2, 0.2, 0.1, 0.9, 0, 0.8, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.3, 0.9, 0.8, 0, 0.7, 0.7, 0.2, 0.2],
                [0.4, 0.1, 0.4, 0.4, 0.1, 0.7, 0, 0.7, 0.3, 0.1],
                [0.2, 0.2, 0.3, 0.3, 0.1, 0.7, 0.7, 0, 0.4, 0.3],
                [0.6, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.4, 0, 0.8],
                [0.7, 0.2, 0.2, 0.4, 0.1, 0.2, 0.1, 0.3, 0.8, 0],
            ],
            split=0,
            device=ht_device,
        )

        L = ht.cluster.spectral.laplacian(S, norm=True, mode="fc")
        self.assertIsInstance(L, ht.DNDarray)
        self.assertEqual(L.split, S.split)

        L = ht.cluster.spectral.laplacian(S, norm=True, mode="eNeighbour", upper=0.5)
        self.assertIsInstance(L, ht.DNDarray)
        self.assertEqual(L.split, S.split)

        L = ht.cluster.spectral.laplacian(S, norm=False, mode="eNeighbour", lower=0.5)
        self.assertIsInstance(L, ht.DNDarray)
        self.assertEqual(L.split, S.split)
        L_true = ht.array(
            [
                [4, -1, -1, 0, 0, 0, 0, 0, -1, -1],
                [-1, 2, -1, 0, 0, 0, 0, 0, 0, 0],
                [-1, -1, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, -1, -1, 0, 0, 0, 0],
                [0, 0, 0, -1, 2, -1, 0, 0, 0, 0],
                [0, 0, 0, -1, -1, 4, -1, -1, 0, 0],
                [0, 0, 0, 0, 0, -1, 2, -1, 0, 0],
                [0, 0, 0, 0, 0, -1, -1, 2, 0, 0],
                [-1, 0, 0, 0, 0, 0, 0, 0, 2, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, -1, 2],
            ],
            dtype=ht.float32,
            split=0,
            device=ht_device,
        )
        self.assertTrue(ht.equal(L, L_true))

        with self.assertRaises(ValueError):
            L = ht.cluster.spectral.laplacian(S, norm=True, mode="eNeighbour")
        with self.assertRaises(ValueError):
            L = ht.cluster.spectral.laplacian(S, norm=True, mode="eNeighbour", upper=0.3, lower=0.7)
        with self.assertRaises(NotImplementedError):
            L = ht.cluster.spectral.laplacian(S, norm=True, mode="knearest")

    def test_fit_iris(self):
        # get some test data
        iris = ht.load("heat/datasets/data/iris.csv", sep=";")

        # fit the clusters
        m = 20
        spectral = ht.cluster.Spectral(
            gamma=1.0, metric="rbf", laplacian="fully_connected", normalize=True, n_lanczos=m
        )
        spectral.fit(iris)

        self.assertIsInstance(spectral.labels_, ht.DNDarray)
        self.assertIsInstance(spectral._components, ht.DNDarray)

        with self.assertRaises(NotImplementedError):
            spectral = ht.cluster.Spectral(
                gamma=1.0,
                metric="rbf",
                laplacian="fully_connected",
                normalize=True,
                n_lanczos=20,
                assign_labels="discretize",
            )
        with self.assertRaises(NotImplementedError):
            spectral = ht.cluster.Spectral(
                gamma=1.0,
                metric="nearest_neighbors",
                laplacian="fully_connected",
                normalize=True,
                n_lanczos=m,
            )
            spectral.fit(iris)

        iris_split = ht.load("heat/datasets/data/iris.csv", sep=";", split=1)
        spectral = ht.cluster.Spectral(n_lanczos=20)

        with self.assertRaises(NotImplementedError):
            spectral.fit(iris_split)
