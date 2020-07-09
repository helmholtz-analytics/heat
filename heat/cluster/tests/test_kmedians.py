import os
import unittest

import heat as ht

from ...core.tests.test_suites.basic_test import TestCase


class TestKMeans(TestCase):
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
        for split in [None, 0]:
            # get some test data
            iris = ht.load("heat/datasets/data/iris.csv", sep=";", split=split)

            # fit the clusters
            k = 3
            kmedian = ht.cluster.KMedians(n_clusters=k)
            kmedian.fit(iris)

            # check whether the results are correct
            self.assertIsInstance(kmedian.cluster_centers_, ht.DNDarray)
            self.assertEqual(kmedian.cluster_centers_.shape, (k, iris.shape[1]))
            # same test with init=kmeans++
            kmedian = ht.cluster.KMedians(n_clusters=k, init="kmedians++")
            kmedian.fit(iris)

            # check whether the results are correct
            self.assertIsInstance(kmedian.cluster_centers_, ht.DNDarray)
            self.assertEqual(kmedian.cluster_centers_.shape, (k, iris.shape[1]))

    def test_exceptions(self):
        # get some test data
        iris_split = ht.load("heat/datasets/data/iris.csv", sep=";", split=1)

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
