import os
import unittest

import heat as ht
import numpy as np
import torch

from ...core.tests.test_suites.basic_test import TestCase


class TestKMeans(TestCase):
    def create_spherical_dataset(
        self, num_samples_cluster, radius=1.0, offset=4.0, dtype=ht.float32, random_state=1
    ):
        """
        Creates k=4 sperical clusters in 3D space along the space-diagonal

        Parameters
        ----------
        num_samples_cluster: int
            Number of samples per cluster. Each process will create n // MPI_WORLD.size elements for each cluster
        radius: float
            Radius of the sphere
        offset: float
            Shift of the clusters along the axes. The 4 clusters will be positioned centered around c1=(offset, offset,offset),
            c2=(2*offset,2*offset,2*offset), c3=(-offset, -offset, -offset) and c4=(2*offset, -2*offset, -2*offset)
        dtype: ht.datatype
        random_state: int
            seed of the torch random number generator
        """
        # contains num_samples

        p = ht.MPI_WORLD.size
        # create k sperical clusters with each n elements per cluster. Each process creates k * n/p elements
        num_ele = num_samples_cluster // p
        ht.random.seed(random_state)
        # radius between 0 and 1
        r = ht.random.rand(num_ele, split=0) * radius
        # theta between 0 and pi
        theta = ht.random.rand(num_ele, split=0) * 3.1415
        # phi between 0 and 2pi
        phi = ht.random.rand(num_ele, split=0) * 2 * 3.1415
        # Cartesian coordinates
        x = r * ht.sin(theta) * ht.cos(phi)
        x.astype(dtype, copy=False)
        y = r * ht.sin(theta) * ht.sin(phi)
        y.astype(dtype, copy=False)
        z = r * ht.cos(theta)
        z.astype(dtype, copy=False)

        cluster1 = ht.stack((x + offset, y + offset, z + offset), axis=1)
        cluster2 = ht.stack((x + 2 * offset, y + 2 * offset, z + 2 * offset), axis=1)
        cluster3 = ht.stack((x - offset, y - offset, z - offset), axis=1)
        cluster4 = ht.stack((x - 2 * offset, y - 2 * offset, z - 2 * offset), axis=1)

        data = ht.concatenate((cluster1, cluster2, cluster3, cluster4), axis=0)
        # Note: enhance when shuffel is available
        return data

    def test_clusterer(self):
        kmedoid = ht.cluster.KMedoids()
        self.assertTrue(ht.is_estimator(kmedoid))
        self.assertTrue(ht.is_clusterer(kmedoid))

    def test_get_and_set_params(self):
        kmedoid = ht.cluster.KMedoids()
        params = kmedoid.get_params()

        self.assertEqual(
            params, {"n_clusters": 8, "init": "random", "max_iter": 300, "random_state": None}
        )

        params["n_clusters"] = 10
        kmedoid.set_params(**params)
        self.assertEqual(10, kmedoid.n_clusters)

    def test_fit_iris_unsplit(self):
        split = 0
        # get some test data
        iris = ht.load("heat/datasets/data/iris.csv", sep=";", split=split)

        # fit the clusters
        k = 3
        kmedoid = ht.cluster.KMedoids(n_clusters=k)
        kmedoid.fit(iris)

        # check whether the results are correct
        self.assertIsInstance(kmedoid.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmedoid.cluster_centers_.shape, (k, iris.shape[1]))
        # same test with init=kmedoids++
        kmedoid = ht.cluster.KMedoids(n_clusters=k, init="kmedoids++")
        kmedoid.fit(iris)

        # check whether the results are correct
        self.assertIsInstance(kmedoid.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmedoid.cluster_centers_.shape, (k, iris.shape[1]))

        # check whether result is actually a datapoint
        for i in range(kmedoid.cluster_centers_.shape[0]):
            self.assertTrue(
                ht.any(ht.sum(ht.abs(kmedoid.cluster_centers_[i, :] - iris), axis=1) == 0)
            )

    def test_exceptions(self):
        # get some test data
        iris_split = ht.load("heat/datasets/data/iris.csv", sep=";", split=1)

        # build a clusterer
        k = 3
        kmedoid = ht.cluster.KMedoids(n_clusters=k)

        with self.assertRaises(NotImplementedError):
            kmedoid.fit(iris_split)
        with self.assertRaises(ValueError):
            kmedoid.set_params(foo="bar")
        with self.assertRaises(ValueError):
            kmedoid = ht.cluster.KMedoids(n_clusters=k, init="random_number")
            kmedoid.fit(iris_split)

    def test_spherical_clusters(self):
        seed = 1
        n = 20 * ht.MPI_WORLD.size
        data = self.create_spherical_dataset(
            num_samples_cluster=n, radius=1.0, offset=4.0, dtype=ht.float32, random_state=seed
        )
        kmedoid = ht.cluster.KMedoids(n_clusters=4, init="kmedoids++")
        kmedoid.fit(data)
        self.assertIsInstance(kmedoid.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmedoid.cluster_centers_.shape, (4, 3))

        # More Samples
        n = 100 * ht.MPI_WORLD.size
        data = self.create_spherical_dataset(
            num_samples_cluster=n, radius=1.0, offset=4.0, dtype=ht.float32, random_state=seed
        )
        kmedoid = ht.cluster.KMedoids(n_clusters=4, init="kmedoids++")
        kmedoid.fit(data)
        self.assertIsInstance(kmedoid.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmedoid.cluster_centers_.shape, (4, 3))
        # check whether result is actually a datapoint
        for i in range(kmedoid.cluster_centers_.shape[0]):
            self.assertTrue(
                ht.any(ht.sum(ht.abs(kmedoid.cluster_centers_[i, :] - data), axis=1) == 0)
            )

        # different datatype
        n = 20 * ht.MPI_WORLD.size
        data = self.create_spherical_dataset(
            num_samples_cluster=n, radius=1.0, offset=4.0, dtype=ht.float64, random_state=seed
        )
        kmedoid = ht.cluster.KMedoids(n_clusters=4, init="kmedoids++")
        kmedoid.fit(data)
        self.assertIsInstance(kmedoid.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmedoid.cluster_centers_.shape, (4, 3))
        for i in range(kmedoid.cluster_centers_.shape[0]):
            self.assertTrue(
                ht.any(ht.sum(ht.abs(kmedoid.cluster_centers_[i, :] - data), axis=1) == 0)
            )

        # on Ints (different radius, offset and datatype
        data = self.create_spherical_dataset(
            num_samples_cluster=n, radius=10.0, offset=40.0, dtype=ht.int32, random_state=seed
        )
        kmedoid = ht.cluster.KMedoids(n_clusters=4, init="kmedoids++")
        kmedoid.fit(data)
        self.assertIsInstance(kmedoid.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmedoid.cluster_centers_.shape, (4, 3))
        for i in range(kmedoid.cluster_centers_.shape[0]):
            self.assertTrue(
                ht.any(ht.sum(ht.abs(kmedoid.cluster_centers_[i, :] - data), axis=1) == 0)
            )
