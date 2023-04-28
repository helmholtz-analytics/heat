import os
import unittest
import numpy as np
import torch
import heat as ht
from heat.utils.data.spherical import create_spherical_dataset

from ...core.tests.test_suites.basic_test import TestCase


class TestKMeans(TestCase):
    def test_clusterer(self):
        kmeans = ht.cluster.KMeans()
        self.assertTrue(ht.is_estimator(kmeans))
        self.assertTrue(ht.is_clusterer(kmeans))

    def test_get_and_set_params(self):
        kmeans = ht.cluster.KMeans()
        params = kmeans.get_params()

        self.assertEqual(
            params,
            {"n_clusters": 8, "init": "random", "max_iter": 300, "tol": 1e-4, "random_state": None},
        )

        params["n_clusters"] = 10
        kmeans.set_params(**params)
        self.assertEqual(10, kmeans.n_clusters)

    def test_fit_iris_unsplit(self):
        for split in [None, 0]:
            # get some test data
            iris = ht.load("heat/datasets/iris.csv", sep=";", split=split)

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

    def test_exceptions(self):
        # get some test data
        iris_split = ht.load("heat/datasets/iris.csv", sep=";", split=1)

        # build a clusterer
        k = 3
        kmeans = ht.cluster.KMeans(n_clusters=k)

        with self.assertRaises(NotImplementedError):
            kmeans.fit(iris_split)
        with self.assertRaises(ValueError):
            kmeans.set_params(foo="bar")
        with self.assertRaises(ValueError):
            kmeans = ht.cluster.KMeans(n_clusters=k, init="random_number")
            kmeans.fit(iris_split)

    def test_spherical_clusters(self):
        seed = 1
        n = 20 * ht.MPI_WORLD.size
        data = create_spherical_dataset(
            num_samples_cluster=n, radius=1.0, offset=4.0, dtype=ht.float32, random_state=seed
        )
        kmeans = ht.cluster.KMeans(n_clusters=4, init="kmeans++")
        kmeans.fit(data)
        self.assertIsInstance(kmeans.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmeans.cluster_centers_.shape, (4, 3))

        # More Samples
        n = 100 * ht.MPI_WORLD.size
        data = create_spherical_dataset(
            num_samples_cluster=n, radius=1.0, offset=4.0, dtype=ht.float32, random_state=seed
        )
        kmeans = ht.cluster.KMeans(n_clusters=4, init="kmeans++")
        kmeans.fit(data)
        self.assertIsInstance(kmeans.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmeans.cluster_centers_.shape, (4, 3))

        # different datatype
        n = 20 * ht.MPI_WORLD.size
        data = create_spherical_dataset(
            num_samples_cluster=n, radius=1.0, offset=4.0, dtype=ht.float64, random_state=seed
        )
        kmeans = ht.cluster.KMeans(n_clusters=4, init="kmeans++")
        kmeans.fit(data)
        self.assertIsInstance(kmeans.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmeans.cluster_centers_.shape, (4, 3))

        # on Ints (different radius, offset and datatype
        data = create_spherical_dataset(
            num_samples_cluster=n, radius=10.0, offset=40.0, dtype=ht.int32, random_state=seed
        )
        kmeans = ht.cluster.KMeans(n_clusters=4, init="kmeans++")
        kmeans.fit(data)
        self.assertIsInstance(kmeans.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmeans.cluster_centers_.shape, (4, 3))
