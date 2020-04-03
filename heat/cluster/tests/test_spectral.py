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
    def test_fit_iris(self):
        # get some test data
        iris = ht.load("heat/datasets/data/iris.csv", sep=";")

        # fit the clusters
        m = 4
        spectral = ht.cluster.Spectral(
            gamma=1.0, metric="rbf", laplacian="fully_connected", normalize=True, n_lanczos=m
        )
        spectral.fit(iris)

        self.assertIsInstance(spectral.labels_, ht.DNDarray)

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
