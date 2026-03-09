import os
import unittest

import heat as ht
import torch

from ...core.tests.test_suites.basic_test import TestCase


class TestSpectral(TestCase):
    def test_clusterer(self):
        spectral = ht.cluster.SpectralClustering()
        self.assertTrue(ht.is_estimator(spectral))
        self.assertTrue(ht.is_clusterer(spectral))

    def test_get_and_set_params(self):
        spectral = ht.cluster.SpectralClustering()
        params = spectral.get_params()

        self.assertEqual(
            params,
            {
                "n_clusters": None,
                "eigen_solver": "randomized",
                "n_components": None,
                "random_state": None,
                "gamma": 1.0,
                "affinity": "rbf",
                "laplacian": "fully_connected",
                "threshold": 1.0,
                "boundary": "upper",
                "lanczos_n_iter": 300,
                "assign_labels": "kmeans",
                "reigh_rank": 100,
                "reigh_n_oversamples": 10,
                "reigh_power_iter": 0,
            },
        )

        params["n_clusters"] = 10
        spectral.set_params(**params)
        self.assertEqual(10, spectral.n_clusters)

    def test_fit_iris(self):
        if not self.is_mps:
            # get some test data
            iris = ht.load("heat/datasets/iris.csv", sep=";", split=0)
            lanczosniter = 10
            reighrank = 10
            # fit the clusters
            spectral = ht.cluster.SpectralClustering(
                n_clusters=3, random_state=0, gamma=1.0, affinity="rbf", laplacian="fully_connected", eigen_solver="lanczos", lanczos_n_iter=lanczosniter
            )
            spectral.fit(iris)
            self.assertIsInstance(spectral.labels_, ht.DNDarray)

            spectral = ht.cluster.SpectralClustering(
                eigen_solver="randomized",
                affinity="euclidean",
                laplacian="eNeighbour",
                threshold=0.5,
                boundary="upper",
                reigh_rank=reighrank,
            )
            labels = spectral.fit_predict(iris)
            self.assertIsInstance(labels, ht.DNDarray)

            kmeans = {"kmeans++": "kmeans++", "max_iter": 30, "tol": -1}
            spectral = ht.cluster.SpectralClustering(
                n_clusters=3, gamma=1.0, normalize=True, params=kmeans
            )
            labels = spectral.fit_predict(iris)
            self.assertIsInstance(labels, ht.DNDarray)

            spectral = ht.cluster.SpectralClustering(
                gamma=0.1,
                affinity="rbf",
                laplacian="eNeighbour",
                threshold=0.5,
                boundary="upper",
                eigen_solver="randomized",
                reigh_rank=5,
                reigh_n_oversamples=2,
                reigh_power_iter=1,
            )
            labels = spectral.fit_predict(iris)
            self.assertIsInstance(labels, ht.DNDarray)

            # Errors
            with self.assertRaises(NotImplementedError):
                spectral = ht.cluster.SpectralClustering(affinity="mahalanobis", lanczos_n_iter=2)

            with self.assertRaises(NotImplementedError):
                spectral = ht.cluster.SpectralClustering(eigen_solver="unknown")

            with self.assertRaises(ValueError):
                spectral = ht.cluster.SpectralClustering(eigen_solver="randomized", reigh_rank=2, n_clusters=5)

            iris_split = ht.load("heat/datasets/iris.csv", sep=";", split=1)
            spectral = ht.cluster.SpectralClustering(eigen_solver="lanczos", lanczos_n_iter=20)
            with self.assertRaises(NotImplementedError):
                spectral.fit(iris_split)
