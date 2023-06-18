import unittest
import heat as ht

from heat.classification.kneighborsclassifier import KNeighborsClassifier
from heat.core.tests.test_suites.basic_test import TestCase


class TestKNN(TestCase):
    @unittest.skipUnless(ht.supports_hdf5(), "Requires HDF5")
    def test_split_none(self):
        x = ht.load_hdf5("heat/datasets/iris.h5", dataset="data")

        # generate keys for the iris.h5 dataset
        labels = ht.zeros(150, dtype=ht.int32)
        labels[50:100] = 1
        labels[100:] = 2

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x, labels)
        result = knn.predict(x)

        self.assertTrue(ht.is_estimator(knn))
        self.assertTrue(ht.is_classifier(knn))
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, labels.shape)

    @unittest.skipUnless(ht.supports_hdf5(), "Requires HDF5")
    def test_split_zero(self):
        x = ht.load_hdf5("heat/datasets/iris.h5", dataset="data", split=0)

        # generate keys for the iris.h5 dataset
        labels = ht.zeros(150, dtype=ht.int32)
        labels[50:100] = 1
        labels[100:] = 2

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x, labels)
        result = knn.predict(x)

        self.assertTrue(ht.is_estimator(knn))
        self.assertTrue(ht.is_classifier(knn))
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, labels.shape)

    def test_exception(self):
        a = ht.zeros((3,))
        b = ht.zeros((3, 2))
        c = ht.zeros((2, 2, 2))

        with self.assertRaises(ValueError):
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(a, b)

        with self.assertRaises(ValueError):
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(b, c)

        with self.assertRaises(ValueError):
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(c, a)

    def test_utility(
        self,
    ):
        a = ht.array([1, 2, 3, 4])
        b = ht.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

        one_hot = KNeighborsClassifier.one_hot_encoding(a)
        self.assertTrue((one_hot == b).all())

    @unittest.skipUnless(ht.supports_hdf5(), "Requires HDF5")
    def test_fit_one_hot(
        self,
    ):
        x = ht.load_hdf5("heat/datasets/iris.h5", dataset="data")

        labels = ht.zeros(150, dtype=ht.int32, split=0)
        labels[50:100] = 1
        labels[100:] = 2

        # keys as one_hot
        keys = ht.zeros((150, 3), dtype=ht.int32)
        keys[50:100, 1] = 1
        keys[100:, 2] = 1
        y = ht.array(keys)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x, y)
        result = knn.predict(x)

        self.assertTrue(ht.is_estimator(knn))
        self.assertTrue(ht.is_classifier(knn))
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, labels.shape)
