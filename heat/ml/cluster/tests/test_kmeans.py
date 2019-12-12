import os
import unittest

import heat as ht

if ht.io.supports_hdf5():

    class TestKMeans(unittest.TestCase):
        def test_fit_iris(self):
            # get some test data
            iris = ht.load_hdf5(os.path.join(os.getcwd(), "heat/datasets/data/iris.h5"), "data")

            # fit the clusters
            k = 3
            kmeans = ht.ml.cluster.KMeans(n_clusters=k)
            kmeans.fit(iris)

            # check whether the results are correct
            self.assertIsInstance(kmeans.cluster_centers_, ht.DNDarray)
            self.assertEqual(kmeans.cluster_centers_.shape, (k, iris.shape[1]))
            # same test with init=kmeans++
            kmeans = ht.ml.cluster.KMeans(n_clusters=k, init="kmeans++")
            kmeans.fit(iris)

            # check whether the results are correct
            self.assertIsInstance(kmeans.cluster_centers_, ht.DNDarray)
            self.assertEqual(kmeans.cluster_centers_.shape, (k, iris.shape[1]))
