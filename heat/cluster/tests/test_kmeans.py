import os
import unittest
import numpy as np
import torch
import heat as ht

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
        torch.manual_seed(random_state)
        # radius between 0 and 1
        r = torch.rand((num_ele,)) * radius
        # theta between 0 and pi
        theta = torch.rand((num_ele,)) * 3.1415
        # phi between 0 and 2pi
        phi = torch.rand((num_ele,)) * 2 * 3.1415
        # Cartesian coordinates
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
        cluster1 = torch.cat(
            (x.unsqueeze(1) + offset, y.unsqueeze(1) + offset, z.unsqueeze(1) + offset), dim=1
        )
        cluster2 = torch.cat(
            (x.unsqueeze(1) + 2 * offset, y.unsqueeze(1) + 2 * offset, z.unsqueeze(1) + 2 * offset),
            dim=1,
        )
        cluster3 = torch.cat(
            (x.unsqueeze(1) - offset, y.unsqueeze(1) - offset, z.unsqueeze(1) - offset), dim=1
        )
        cluster4 = torch.cat(
            (x.unsqueeze(1) - 2 * offset, y.unsqueeze(1) - 2 * offset, z.unsqueeze(1) - 2 * offset),
            dim=1,
        )

        # cluster centers are (k,k,k)*5 --> centroid1 = (0,0,0), centroid2 = (5,5,5), etc
        local_data = torch.cat((cluster1, cluster2, cluster3, cluster4), dim=0)
        local_data = local_data.type(dtype.torch_type())

        data = ht.array(local_data[torch.randperm(local_data.size()[0])], is_split=0)
        return data

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
            iris = ht.load("heat/datasets/data/iris.csv", sep=";", split=split)

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
        iris_split = ht.load("heat/datasets/data/iris.csv", sep=";", split=1)

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
        data = self.create_spherical_dataset(
            num_samples_cluster=n, radius=1.0, offset=4.0, dtype=ht.float32, random_state=seed
        )
        reference = ht.array([[-8, -8, -8], [-4, -4, -4], [4, 4, 4], [8, 8, 8]], dtype=ht.float32)
        kmeans = ht.cluster.KMeans(n_clusters=4, init="kmeans++")
        idx = 0
        for i in range(10):
            kmeans.fit(data)
            result, _ = ht.sort(kmeans.cluster_centers_, axis=0)
            if ht.allclose(result, reference, atol=0.5):
                idx += 1
        # At least one of the runs must yield accurate results
        self.assertTrue(idx > 0)

        # More Samples
        n = 100 * ht.MPI_WORLD.size
        data = self.create_spherical_dataset(
            num_samples_cluster=n, radius=1.0, offset=4.0, dtype=ht.float32, random_state=seed
        )
        reference = ht.array([[-8, -8, -8], [-4, -4, -4], [4, 4, 4], [8, 8, 8]], dtype=ht.float32)
        kmeans = ht.cluster.KMeans(n_clusters=4, init="kmeans++")
        idx = 0
        for i in range(10):
            kmeans.fit(data)
            result, _ = ht.sort(kmeans.cluster_centers_, axis=0)
            if ht.allclose(result, reference, atol=0.5):
                idx += 1
        self.assertTrue(idx > 0)

        # different datatype
        n = 20 * ht.MPI_WORLD.size
        data = self.create_spherical_dataset(
            num_samples_cluster=n, radius=1.0, offset=4.0, dtype=ht.float64, random_state=seed
        )
        reference = ht.array([[-8, -8, -8], [-4, -4, -4], [4, 4, 4], [8, 8, 8]], dtype=ht.float64)
        kmeans = ht.cluster.KMeans(n_clusters=4, init="kmeans++")
        idx = 0
        for i in range(10):
            kmeans.fit(data)
            result, _ = ht.sort(kmeans.cluster_centers_, axis=0)
            if ht.allclose(result, reference, atol=0.5):
                idx += 1
        self.assertTrue(idx > 0)

        # on Ints (different radius, offset and datatype
        data = self.create_spherical_dataset(
            num_samples_cluster=n, radius=10.0, offset=40.0, dtype=ht.int32, random_state=seed
        )
        reference = ht.array(
            [[-80, -80, -80], [-40, -40, -40], [40, 40, 40], [80, 80, 80]], dtype=ht.float32
        )
        kmeans = ht.cluster.KMeans(n_clusters=4, init="kmeans++")
        idx = 0
        for i in range(10):
            kmeans.fit(data)
            result, _ = ht.sort(kmeans.cluster_centers_, axis=0)
            if ht.allclose(result, reference, atol=5):
                idx += 1
        self.assertTrue(idx > 0)
