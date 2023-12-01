import os
import unittest
import numpy as np
import torch
import heat as ht
from heat.utils.data.spherical import create_spherical_dataset
from mpi4py import MPI

from ...core.tests.test_suites.basic_test import TestCase

# test BatchParallelKCluster base class and auxiliary functions


# test BatchParallelKMeans and BatchParallelKMedians
class TestBatchParallelKCluster(TestCase):
    def test_clusterer(self):
        for ParallelClusterer in [ht.cluster.BatchParallelKMeans, ht.cluster.BatchParallelKMedians]:
            parallelclusterer = ParallelClusterer()
            self.assertTrue(ht.is_estimator(parallelclusterer))
            self.assertTrue(ht.is_clusterer(parallelclusterer))

    def test_get_and_set_params(self):
        for ParallelClusterer in [ht.cluster.BatchParallelKMeans, ht.cluster.BatchParallelKMedians]:
            if ParallelClusterer is ht.cluster.BatchParallelKMeans:
                ppinitkw = "k-means++"
            elif ParallelClusterer is ht.cluster.BatchParallelKMedians:
                ppinitkw = "k-medians++"
            parallelclusterer = ParallelClusterer()
            params = parallelclusterer.get_params()

            self.assertEqual(
                params,
                {
                    "n_clusters": 8,
                    "init": ppinitkw,
                    "max_iter": 300,
                    "tol": 1e-4,
                    "random_state": None,
                },
            )

            params["n_clusters"] = 10
            parallelclusterer.set_params(**params)
            self.assertEqual(10, parallelclusterer.n_clusters)

    def test_spherical_clusters(self):
        for ParallelClusterer in [ht.cluster.BatchParallelKMeans, ht.cluster.BatchParallelKMedians]:
            if ParallelClusterer is ht.cluster.BatchParallelKMeans:
                ppinitkw = "k-means++"
            elif ParallelClusterer is ht.cluster.BatchParallelKMedians:
                ppinitkw = "k-medians++"
            for seed in [1, None]:
                n = 20 * ht.MPI_WORLD.size
                for dtype in [ht.float32, ht.float64]:
                    data = create_spherical_dataset(
                        num_samples_cluster=n,
                        radius=1.0,
                        offset=4.0,
                        dtype=dtype,
                        random_state=seed,
                    )
                    for n_clusters in [4, 5]:
                        parallelclusterer = ParallelClusterer(
                            n_clusters=n_clusters, init=ppinitkw, random_state=seed
                        )
                        parallelclusterer.fit(data)
                        self.assertIsInstance(parallelclusterer.cluster_centers_, ht.DNDarray)
                        self.assertEqual(parallelclusterer.cluster_centers_.split, None)
                        self.assertEqual(parallelclusterer.cluster_centers_.shape, (n_clusters, 3))
                        self.assertEqual(parallelclusterer.cluster_centers_.dtype, dtype)
                        labels = parallelclusterer.predict(data)
                        self.assertIsInstance(labels, ht.DNDarray)
                        self.assertEqual(labels.split, 0)
                        self.assertEqual(labels.shape, (data.shape[0], 1))
                        self.assertEqual(labels.dtype, ht.int32)
                        self.assertEqual(labels.max(), n_clusters - 1)
                        self.assertEqual(labels.min(), 0)

    def catch_all_errors(self):
        for ParallelClusterer in [ht.cluster.BatchParallelKMeans, ht.cluster.BatchParallelKMedians]:
            # wrong split dimension for fit
            parallelclusterer = ParallelClusterer()
            X = ht.random.randn(4, ht.MPI_WORLD.size * 10, split=1)
            with self.assertRaises(ValueError):
                parallelclusterer.fit(X)
            # predict is called before fit
            X = ht.random.randn(ht.MPI_WORLD.size * 10, 2, split=0)
            with self.assertRaises(RuntimeError):
                parallelclusterer.predict(X)
            # wrong split dimension for predict
            parallelclusterer = ParallelClusterer()
            X = ht.random.randn(ht.MPI_WORLD.size * 10, 2, split=0)
            parallelclusterer.fit_predict(X)
            X = ht.random.randn(4, ht.MPI_WORLD.size * 10, split=1)
            with self.assertRaises(ValueError):
                parallelclusterer.predict(X)
            # wrong shape for predict
            X = ht.random.randn(ht.MPI_WORLD.size * 10, 3, split=0)
            with self.assertRaises(ValueError):
                parallelclusterer.predict(X)
            # wrong inputs for constructor
            with self.assertRaises(ValueError):
                parallelclusterer = ParallelClusterer(n_clusters=-1)
            with self.assertRaises(TypeError):
                parallelclusterer = ParallelClusterer(n_clusters="abc")
            with self.assertRaises(ValueError):
                parallelclusterer = ParallelClusterer(max_iter=-10)
            with self.assertRaises(TypeError):
                parallelclusterer = ParallelClusterer(max_iter=0.1)
            with self.assertRaises(ValueError):
                parallelclusterer = ParallelClusterer(tol=-0.5)
            with self.assertRaises(TypeError):
                parallelclusterer = ParallelClusterer(tol="abc")
            with self.assertRaises(ValueError):
                parallelclusterer = ParallelClusterer(init="abc")
            with self.assertRaises(TypeError):
                parallelclusterer = ParallelClusterer(init=3.14)
            with self.assertRaises(NotImplementedError):
                parallelclusterer = ParallelClusterer(init="random")
