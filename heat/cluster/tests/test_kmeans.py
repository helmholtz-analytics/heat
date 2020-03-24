import os
import unittest

import heat as ht

ht_device, torch_device, _ = ht.devices._use_envar_device()


class TestKMeans(unittest.TestCase):
    def test_fit_iris(self):
        # get some test data
        iris = ht.load("heat/datasets/data/iris.csv", sep=";", device=ht_device)

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

        iris_split = ht.load("heat/datasets/data/iris.csv", sep=";", split=1, device=ht_device)
        kmeans = ht.cluster.KMeans(n_clusters=k)

        with self.assertRaises(NotImplementedError):
            kmeans.fit(iris_split)

        kmeans = ht.cluster.KMeans(n_clusters=k, init="random_number")
        with self.assertRaises(ValueError):
            kmeans.fit(iris_split)
