import os
import platform
import unittest
import numpy as np
import torch
import heat as ht
from heat.utils.data.spherical import create_spherical_dataset
from mpi4py import MPI

from ...core.tests.test_suites.basic_test import TestCase
from ..batchparallelclustering import _kmex, _initialize_plus_plus, _BatchParallelKCluster

# test BatchParallelKCluster base class and auxiliary functions

# skip on MPS
envar = os.getenv("HEAT_TEST_USE_DEVICE", "cpu")
is_mps = envar == "gpu" and platform.system() == "Darwin"


@unittest.skipIf(is_mps, "Batchparallelclustering fit() fails on MPS")
class TestAuxiliaryFunctions(TestCase):
    def test_kmex(self):
        X = torch.rand(10, 3)
        max_iter = 10
        tol = 1e-2
        init = "random"
        # test wrong shape of init in _kmex
        wrong_init = torch.rand(3, 3)
        with self.assertRaises(ValueError):
            _kmex(X, 2, 2, wrong_init, max_iter, tol)
        # test wrong dtype of init in _kmex
        wrong_init = "abc"
        with self.assertRaises(ValueError):
            _kmex(X, 2, 2, wrong_init, max_iter, tol)
        # test initialization "random"
        _kmex(X, 2, 2, init, max_iter, tol)
        # test initialization with prescribed centers
        init = torch.rand(2, 3)
        _kmex(X, 2, 2, init, max_iter, tol)

    def test_initialize_plus_plus(self):
        with self.subTest("subsampling"):
            X = torch.rand(100, 3)
            centers = _initialize_plus_plus(X, 3, 2, random_state=0, max_samples=50)
            self.assertEqual(centers.shape, (3, 3))

        # 2) probs.sum() <= 0 because weights are all zero -> fallback to dist -> multinomial runs
        with self.subTest("weights_zero_fallback_to_dist"):
            X = torch.rand(30, 3)
            weights = torch.zeros(X.shape[0], dtype=X.dtype)
            centers = _initialize_plus_plus(X, 3, 2, random_state=0, weights=weights)
            self.assertEqual(centers.shape, (3, 3))

        # 3) fully degenerate distances (all points identical) -> probs.sum() <= 0 twice -> candidate selection branch
        with self.subTest("all_distances_zero_candidate_selection"):
            X = torch.ones(10, 3)
            weights = torch.ones(X.shape[0], dtype=X.dtype)
            centers = _initialize_plus_plus(X, 3, 2, random_state=0, weights=weights)
            self.assertEqual(centers.shape, (3, 3))

        # 4) extreme degenerate case: only one sample, n_clusters>1 -> candidates empty branch
        with self.subTest("single_sample_candidates_empty"):
            X = torch.ones(1, 3)
            centers = _initialize_plus_plus(X, 2, 2, random_state=0)
            self.assertEqual(centers.shape, (2, 3))

        # 5) NaN-handling path -> nan_to_num is exercised (should not crash)
        with self.subTest("nan_to_num_path"):
            X = torch.tensor(
                [[0.0, 0.0, 0.0],
                 [float("nan"), 0.0, 0.0],
                 [1.0, 0.0, 0.0]],
                dtype=torch.float32,
            )
            # seed chosen so first centroid is deterministic (helps avoid flakiness)
            centers = _initialize_plus_plus(X, 2, 2, random_state=2)
            self.assertEqual(centers.shape, (2, 3))

    def test_BatchParallelKClustering(self):
        with self.assertRaises(TypeError):
            _BatchParallelKCluster(2, 10, "++", 100, 1e-2, random_state=3.14, n_procs_to_merge=None)
        with self.assertWarns(UserWarning):
            _BatchParallelKCluster(3, 10, "++", 100, 1e-2, random_state=None, n_procs_to_merge=None)
        with self.assertRaises(TypeError):
            _BatchParallelKCluster(
                2, 10, "++", 100, 1e-2, random_state=None, n_procs_to_merge=2.7183
            )
        with self.assertRaises(ValueError):
            _BatchParallelKCluster(2, 10, "++", 100, 1e-2, random_state=None, n_procs_to_merge=1)


# test BatchParallelKMeans and BatchParallelKMedians
@unittest.skipIf(is_mps, "Batchparallelclustering fit() fails on MPS")
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
                    "n_procs_to_merge": None,
                },
            )

            params["n_clusters"] = 10
            params["n_procs_to_merge"] = 4
            parallelclusterer.set_params(**params)
            self.assertEqual(10, parallelclusterer.n_clusters)

    def test_spherical_clusters(self):
        if self.is_mps:
            dtypes = [ht.float32]
        else:
            dtypes = [ht.float32, ht.float64]

        for ParallelClusterer in [ht.cluster.BatchParallelKMeans, ht.cluster.BatchParallelKMedians]:
            if ParallelClusterer is ht.cluster.BatchParallelKMeans:
                ppinitkws = ["k-means++"]
            elif ParallelClusterer is ht.cluster.BatchParallelKMedians:
                ppinitkws = ["k-medians++"]
            for seed in [1, None]:
                n = 20 * ht.MPI_WORLD.size
                for dtype in dtypes:
                    data = create_spherical_dataset(
                        num_samples_cluster=n,
                        radius=1.0,
                        offset=4.0,
                        dtype=dtype,
                        random_state=seed,
                    )
                    for n_clusters in [4, 5]:
                        for n_procs_to_merge in [2, 3, None]:
                            for ppinitkw in ppinitkws:
                                parallelclusterer = ParallelClusterer(
                                    n_clusters=n_clusters,
                                    init=ppinitkws[0],
                                    random_state=seed,
                                    n_procs_to_merge=n_procs_to_merge,
                                )
                                parallelclusterer.fit(data)
                                self.assertIsInstance(
                                    parallelclusterer.cluster_centers_, ht.DNDarray
                                )
                                self.assertEqual(parallelclusterer.cluster_centers_.split, None)
                                self.assertEqual(
                                    parallelclusterer.cluster_centers_.shape, (n_clusters, 3)
                                )
                                self.assertEqual(parallelclusterer.cluster_centers_.dtype, dtype)
                                self.assertIsInstance(parallelclusterer.n_iter_, int)
                                labels = parallelclusterer.predict(data)
                                functional_value = parallelclusterer.functional_value_
                                self.assertIsInstance(functional_value, float)
                                self.assertIsInstance(labels, ht.DNDarray)
                                self.assertEqual(labels.split, 0)
                                self.assertEqual(labels.shape, (data.shape[0], 1))
                                self.assertEqual(labels.dtype, ht.int32)
                                self.assertTrue(labels.max() <= n_clusters - 1)
                                self.assertTrue(labels.min() >= 0)

    def test_if_errors_thrown(self):
        for ParallelClusterer in [ht.cluster.BatchParallelKMeans, ht.cluster.BatchParallelKMedians]:
            parallelclusterer = ParallelClusterer()
            # wrong dtype for fit
            with self.assertRaises(TypeError):
                parallelclusterer.fit("abc")
            # wrong dimension for fit
            X = ht.random.randn(4, 2, 2, split=0)
            with self.assertRaises(ValueError):
                parallelclusterer.fit(X)
            # wrong split dimension for fit
            X = ht.random.randn(4, ht.MPI_WORLD.size * 10, split=1)
            with self.assertRaises(ValueError):
                parallelclusterer.fit(X)
            # now comes predict:
            # predict is called before fit
            X = ht.random.randn(ht.MPI_WORLD.size * 10, 2, split=0)
            with self.assertRaises(RuntimeError):
                parallelclusterer.predict(X)
            parallelclusterer = ParallelClusterer()
            X = ht.random.randn(ht.MPI_WORLD.size * 10, 2, split=0)
            parallelclusterer.fit_predict(X)
            # wrong dtype for predict
            with self.assertRaises(TypeError):
                parallelclusterer.predict("abc")
            # wrong dimension for predict
            X = ht.random.randn(4, 2, 2, split=0)
            with self.assertRaises(ValueError):
                parallelclusterer.predict(X)
            # wrong split dimension for predict
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
