import os
import unittest

import heat as ht

if os.environ.get("DEVICE") == "gpu" and ht.torch.cuda.is_available():
    ht.use_device("gpu")
    ht.torch.cuda.set_device(ht.torch.device(ht.get_device().torch_device))
else:
    ht.use_device("cpu")
device = ht.get_device().torch_device
ht_device = None
if os.environ.get("DEVICE") == "lgpu" and ht.torch.cuda.is_available():
    device = ht.gpu.torch_device
    ht_device = ht.gpu
    ht.torch.cuda.set_device(device)


class TestKMeans(unittest.TestCase):
    def test_fit_iris(self):
        # get some test data
        iris = ht.load_hdf5(
            os.path.join(os.getcwd(), "heat/datasets/data/iris.h5"), "data", device=ht_device
        )

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
