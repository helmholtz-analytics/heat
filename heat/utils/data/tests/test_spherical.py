import heat as ht
import unittest
import torch
from heat.core.tests.test_suites.basic_test import TestCase


class TestCreateClusters(TestCase):
    def test_create_cluster(self):
        n_samples = ht.MPI_WORLD.size * 10 + 3
        n_features = 3
        n_clusters = ht.MPI_WORLD.size
        cluster_mean = torch.arange(n_clusters, dtype=torch.float32).repeat(n_features, 1).T

        # test case with uneven distribution of clusters over processes and variances given as vector
        cluster_weight = torch.zeros(n_clusters)
        cluster_weight[ht.MPI_WORLD.rank] += 0.5
        cluster_weight[0] += 0.5
        cluster_std = 0.01 * torch.ones(n_clusters)
        data = ht.utils.data.spherical.create_clusters(
            n_samples, n_features, n_clusters, cluster_mean, cluster_std, cluster_weight
        )
        self.assertEqual(data.shape, (n_samples, n_features))
        self.assertEqual(data.dtype, ht.float32)

        # test case with even distribution of clusters over processes and variances given as matrix
        cluster_weight = None
        cluster_std = 0.01 * torch.rand(n_clusters, n_features, n_features)
        cluster_std = torch.transpose(cluster_std, 1, 2) @ cluster_std
        data = ht.utils.data.spherical.create_clusters(
            n_samples, n_features, n_clusters, cluster_mean, cluster_std, cluster_weight
        )
        self.assertEqual(data.shape, (n_samples, n_features))
        self.assertEqual(data.dtype, ht.float32)

    def test_if_errors_are_catched(self):
        n_samples = ht.MPI_WORLD.size * 10 + 3
        n_features = 3
        n_clusters = ht.MPI_WORLD.size
        cluster_mean = torch.arange(n_clusters, dtype=torch.float32).repeat(n_features, 1).T
        cluster_std = 0.01 * torch.ones(n_clusters)

        with self.assertRaises(TypeError):
            ht.utils.data.spherical.create_clusters(
                n_samples, n_features, n_clusters, "abc", cluster_std
            )
        with self.assertRaises(ValueError):
            ht.utils.data.spherical.create_clusters(
                n_samples, n_features, n_clusters, torch.zeros(2, 2), cluster_std
            )
        with self.assertRaises(TypeError):
            ht.utils.data.spherical.create_clusters(
                n_samples, n_features, n_clusters, cluster_mean, "abc"
            )
        with self.assertRaises(ValueError):
            ht.utils.data.spherical.create_clusters(
                n_samples, n_features, n_clusters, cluster_mean, torch.zeros(2, 2)
            )
        with self.assertRaises(TypeError):
            ht.utils.data.spherical.create_clusters(
                n_samples, n_features, n_clusters, cluster_mean, cluster_std, "abc"
            )
        with self.assertRaises(ValueError):
            ht.utils.data.spherical.create_clusters(
                n_samples,
                n_features,
                n_clusters,
                cluster_mean,
                cluster_std,
                torch.ones(
                    n_clusters + 1,
                ),
            )
        with self.assertRaises(ValueError):
            ht.utils.data.spherical.create_clusters(
                n_samples,
                n_features,
                n_clusters,
                cluster_mean,
                cluster_std,
                2
                * torch.ones(
                    n_clusters,
                ),
            )
