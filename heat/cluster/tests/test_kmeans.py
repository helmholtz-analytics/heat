import os
import unittest

import heat as ht

envar = os.getenv("HEAT_USE_DEVICE", "cpu")

if envar == 'cpu':
    ht.use_device("cpu")
    ht.torch_device = ht.get_device().torch_device
    heat_device = None
elif envar == 'gpu' and ht.torch.cuda.is_available():
    ht.use_device("gpu")
    ht.torch.cuda.set_device(ht.torch.device(ht.get_device().torch_device))
    torch_device = ht.get_device().torch_device
    heat_device = None
elif envar == 'lcpu' and ht.torch.cuda.is_available():
    ht.use_device("gpu")
    ht.torch.cuda.set_device(ht.torch.device(ht.get_device().torch_device))
    torch_device = ht.cpu.torch_device
    heat_device = ht.cpu
elif envar == 'lgpu' and ht.torch.cuda.is_available():
    ht.use_device("cpu")
    ht.torch.cuda.set_device(ht.torch.device(ht.get_device().torch_device))
    torch_device = ht.cpu.torch_device
    heat_device = ht.cpu


class TestKMeans(unittest.TestCase):
    def test_fit_iris(self):
        # get some test data
        iris = ht.load("heat/datasets/data/iris.csv", sep=";", device=heat_device)

        # fit the clusters
        k = 3
        kmeans = ht.cluster.KMeans(n_clusters=k)
        kmeans.fit(iris)

        # check whether the results are correct
        self.assertIsInstance(kmeans.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmeans.cluster_centers_.shape, (k, iris.shape[1]))
        # same test with init=kmeans++
        kmeans = ht.cluster.KMeans(n_clusters=k, init="kmeans++")
        kmeans.fit(iris)

        # check whether the results are correct
        self.assertIsInstance(kmeans.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmeans.cluster_centers_.shape, (k, iris.shape[1]))

        iris_split = ht.load("heat/datasets/data/iris.csv", sep=";", split=1, device=heat_device)
        kmeans = ht.cluster.KMeans(n_clusters=k)

        with self.assertRaises(NotImplementedError):
            kmeans.fit(iris_split)

        kmeans = ht.cluster.KMeans(n_clusters=k, init="random_number")
        with self.assertRaises(ValueError):
            kmeans.fit(iris_split)
