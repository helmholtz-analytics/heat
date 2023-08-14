import os
import unittest

import numpy as np
import torch

import heat as ht
from heat.utils.data.spherical import create_spherical_dataset

from ...core.tests.test_suites.basic_test import TestCase


class TestKMedians(TestCase):
    def test_clusterer(self):
        kmedian = ht.cluster.KMedians()
        self.assertTrue(ht.is_estimator(kmedian))
        self.assertTrue(ht.is_clusterer(kmedian))

    def test_get_and_set_params(self):
        kmedian = ht.cluster.KMedians()
        params = kmedian.get_params()

        self.assertEqual(
            params,
            {"n_clusters": 8, "init": "random", "max_iter": 300, "tol": 1e-4, "random_state": None},
        )

        params["n_clusters"] = 10
        kmedian.set_params(**params)
        self.assertEqual(10, kmedian.n_clusters)

    def test_fit_iris_unsplit(self):
        split = 0
        # get some test data
        iris = ht.load("heat/datasets/iris.csv", sep=";", split=split)

        # fit the clusters
        k = 3
        kmedian = ht.cluster.KMedians(n_clusters=k)
        kmedian.fit(iris)

        # check whether the results are correct
        self.assertIsInstance(kmedian.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmedian.cluster_centers_.shape, (k, iris.shape[1]))
        # same test with init=kmedians++
        kmedian = ht.cluster.KMedians(n_clusters=k, init="kmedians++")
        kmedian.fit(iris)

        # check whether the results are correct
        self.assertIsInstance(kmedian.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmedian.cluster_centers_.shape, (k, iris.shape[1]))

    def test_exceptions(self):
        # get some test data
        iris_split = ht.load("heat/datasets/iris.csv", sep=";", split=1)

        # build a clusterer
        k = 3
        kmedian = ht.cluster.KMedians(n_clusters=k)

        with self.assertRaises(NotImplementedError):
            kmedian.fit(iris_split)
        with self.assertRaises(ValueError):
            kmedian.set_params(foo="bar")
        with self.assertRaises(ValueError):
            kmedian = ht.cluster.KMedians(n_clusters=k, init="random_number")
            kmedian.fit(iris_split)

    def test_spherical_clusters(self):
        seed = 1
        n = 20 * ht.MPI_WORLD.size
        data = create_spherical_dataset(
            num_samples_cluster=n, radius=1.0, offset=4.0, dtype=ht.float32, random_state=seed
        )
        kmedians = ht.cluster.KMedians(n_clusters=4, init="kmedians++")
        kmedians.fit(data)
        self.assertIsInstance(kmedians.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmedians.cluster_centers_.shape, (4, 3))

        # More Samples
        n = 100 * ht.MPI_WORLD.size
        data = create_spherical_dataset(
            num_samples_cluster=n, radius=1.0, offset=4.0, dtype=ht.float32, random_state=seed
        )
        kmedians = ht.cluster.KMedians(n_clusters=4, init="kmedians++")
        kmedians.fit(data)
        self.assertIsInstance(kmedians.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmedians.cluster_centers_.shape, (4, 3))

        # different datatype
        n = 20 * ht.MPI_WORLD.size
        data = create_spherical_dataset(
            num_samples_cluster=n, radius=1.0, offset=4.0, dtype=ht.float64, random_state=seed
        )
        kmedians = ht.cluster.KMedians(n_clusters=4, init="kmedians++")
        kmedians.fit(data)
        self.assertIsInstance(kmedians.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmedians.cluster_centers_.shape, (4, 3))

        # on Ints (different radius, offset and datatype
        data = create_spherical_dataset(
            num_samples_cluster=n, radius=10.0, offset=40.0, dtype=ht.int32, random_state=seed
        )
        kmedians = ht.cluster.KMedians(n_clusters=4, init="kmedians++")
        kmedians.fit(data)
        self.assertIsInstance(kmedians.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmedians.cluster_centers_.shape, (4, 3))
