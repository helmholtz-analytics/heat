import unittest

import heat as ht
from heat.utils.data.spherical import create_spherical_dataset

from ...core.tests.test_suites.basic_test import TestCase


class TestKMeans(TestCase):
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
        iris = ht.load("heat/datasets/iris.csv", sep=";", split=split)
        ht.random.seed(1)
        # fit the clusters
        k = 3
        kmedoid = ht.cluster.KMedoids(n_clusters=k, random_state=1)
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
        iris_split = ht.load("heat/datasets/iris.csv", sep=";", split=1)

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
        data = create_spherical_dataset(
            num_samples_cluster=n, radius=1.0, offset=4.0, dtype=ht.float32, random_state=seed
        )
        kmedoid = ht.cluster.KMedoids(n_clusters=4, init="kmedoids++")
        kmedoid.fit(data)
        self.assertIsInstance(kmedoid.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmedoid.cluster_centers_.shape, (4, 3))
        for i in range(kmedoid.cluster_centers_.shape[0]):
            self.assertTrue(
                ht.any(ht.sum(ht.abs(kmedoid.cluster_centers_[i, :] - data), axis=1) == 0)
            )

        # More Samples
        n = 100 * ht.MPI_WORLD.size
        data = create_spherical_dataset(
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
        data = create_spherical_dataset(
            num_samples_cluster=n, radius=1.0, offset=4.0, dtype=ht.float64, random_state=seed
        )
        kmedoid = ht.cluster.KMedoids(n_clusters=4, init="kmedoids++")
        kmedoid.fit(data)
        self.assertIsInstance(kmedoid.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmedoid.cluster_centers_.shape, (4, 3))
        for i in range(kmedoid.cluster_centers_.shape[0]):
            self.assertTrue(
                ht.any(
                    ht.sum(ht.abs(kmedoid.cluster_centers_[i, :] - data.astype(ht.float32)), axis=1)
                    == 0
                )
            )

        # on Ints (different radius, offset and datatype
        data = create_spherical_dataset(
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
