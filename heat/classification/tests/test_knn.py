import os
import unittest

import heat as ht

from heat.core.tests.test_suites.basic_test import TestCase

from heat.classification.knn import KNN


class TestKNN(TestCase):
    def test_classifier(self):
        X = ht.load_hdf5("heat/datasets/data/iris.h5", dataset="data", split=0)

        # Generate keys for the iris.h5 dataset
        keys = []
        for i in range(50):
            keys.append([1, 0, 0])
        for i in range(50, 100):
            keys.append([0, 1, 0])
        for i in range(100, 150):
            keys.append([0, 0, 1])
        Y = ht.array(keys, split=0)
        knn = KNN(X, Y, 5)
        self.assertTrue(ht.is_estimator(knn))
        self.assertTrue(ht.is_classifier(knn))

    def test_split_none(self):
        X = ht.load_hdf5("heat/datasets/data/iris.h5", dataset="data")

        # Generate keys for the iris.h5 dataset
        keys = []
        for i in range(50):
            keys.append([1, 0, 0])
        for i in range(50, 100):
            keys.append([0, 1, 0])
        for i in range(100, 150):
            keys.append([0, 0, 1])
        Y = ht.array(keys)

        knn = KNN(X, Y, 5)

        result = knn.predict(X)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, Y.shape)

    def test_split_zero(self):
        X = ht.load_hdf5("heat/datasets/data/iris.h5", dataset="data", split=0)

        # Generate keys for the iris.h5 dataset
        keys = []
        for i in range(50):
            keys.append([1, 0, 0])
        for i in range(50, 100):
            keys.append([0, 1, 0])
        for i in range(100, 150):
            keys.append([0, 0, 1])
        Y = ht.array(keys, split=0)

        knn = KNN(X, Y, 5)

        result = knn.predict(X)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, Y.shape)

    def test_exception(self,):
        a = ht.array([1, 2, 3])
        b = ht.array([1])
        with self.assertRaises(ValueError):
            knn = KNN(a, b, 1)
        c = ht.array([1, 2, 3])
        knn = KNN(a, c, 1)
        with self.assertRaises(ValueError):
            knn.fit(a, b)
