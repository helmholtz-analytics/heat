import os
import unittest

import heat as ht


class TestKMeans(unittest.TestCase):
    def test_fit_iris(self):
        # get some test data
        iris = ht.load_hdf5(os.path.join(os.getcwd(), 'heat/datasets/data/iris.h5'), 'data')

        # fit the clusters
        k = 3
        kmeans = ht.ml.cluster.KMeans(n_clusters=k)
        centroids = kmeans.fit(iris)

        # check whether the results are correct
        self.assertIsInstance(centroids, ht.DNDarray)
        self.assertEqual(centroids.shape, (1, iris.shape[1], k))
