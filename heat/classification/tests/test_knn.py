import os
import unittest

import heat as ht

from heat.core.tests.test_suites.basic_test import TestCase

from heat.classification.knn import KNN


class TestKNN(TestCase):
    def test_split_none(self):
        X = ht.load_hdf5("heat/datasets/iris.h5", dataset="data")

        # Generate keys for the iris.h5 dataset
        keys = []
        for i in range(50):
            keys.append(0)
        for i in range(50, 100):
            keys.append(1)
        for i in range(100, 150):
            keys.append(2)
        Y = ht.array(keys)

        knn = KNN(X, Y, 5)

        result = knn.predict(X)

        self.assertTrue(ht.is_estimator(knn))
        self.assertTrue(ht.is_classifier(knn))
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, Y.shape)

    def test_split_zero(self):
        X = ht.load_hdf5("heat/datasets/iris.h5", dataset="data", split=0)

        # Generate keys for the iris.h5 dataset
        keys = []
        for i in range(50):
            keys.append(0)
        for i in range(50, 100):
            keys.append(1)
        for i in range(100, 150):
            keys.append(2)
        Y = ht.array(keys, split=0)

        knn = KNN(X, Y, 5)

        result = knn.predict(X)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, Y.shape)

    def test_exception(self,):
        a = ht.zeros((3,))
        b = ht.zeros((1,))
        c = ht.zeros((3,))
        d = ht.zeros((2, 2, 2))

        with self.assertRaises(ValueError):
            knn = KNN(a, b, 1)

        with self.assertRaises(ValueError):
            knn = KNN(a, b, 1)

        knn = KNN(a, c, 1)
        with self.assertRaises(ValueError):
            knn.fit(a, b)

        knn = KNN(a, c, 1)
        with self.assertRaises(ValueError):
            knn.fit(a, d)

        with self.assertRaises(ValueError):
            knn = KNN(a, d, 1)

    def test_utility(self,):
        a = ht.array([1, 2, 3, 4])
        b = ht.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

        one_hot = KNN.label_to_one_hot(a)

        self.assertTrue((one_hot == b).all())

    def test_fit_one_hot(self,):
        X = ht.load_hdf5("heat/datasets/iris.h5", dataset="data")

        # Keys as label array
        keys = []
        for i in range(50):
            keys.append(0)
        for i in range(50, 100):
            keys.append(1)
        for i in range(100, 150):
            keys.append(2)
        labels = ht.array(keys, split=0)

        # Keys as one_hot
        keys = []
        for i in range(50):
            keys.append([1, 0, 0])
        for i in range(50, 100):
            keys.append([0, 1, 0])
        for i in range(100, 150):
            keys.append([0, 0, 1])

        Y = ht.array(keys)

        knn = KNN(X, Y, 5)

        knn.fit(X, Y)

        result = knn.predict(X)

        self.assertTrue(ht.is_estimator(knn))
        self.assertTrue(ht.is_classifier(knn))
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, labels.shape)
