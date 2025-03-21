import unittest
import heat as ht

from heat.classification.localoutlierfactor import LocalOutlierFactor
from heat.core.tests.test_suites.basic_test import TestCase


class TestLOF(TestCase):
    def test_exception(self):
        with self.assertRaises(ValueError):
            LocalOutlierFactor(binary_decision=None)

        with self.assertRaises(ValueError):
            LocalOutlierFactor(binary_decision="top_n", top_n=None)

        with self.assertRaises(ValueError):
            LocalOutlierFactor(binary_decision="top_n", top_n=-1)

        with self.assertRaises(ValueError):
            LocalOutlierFactor(threshold=0.5)

        with self.assertRaises(ValueError):
            LocalOutlierFactor(n_neighbors=0)

        with self.assertRaises(ValueError):
            LocalOutlierFactor(metric=None)

    def test_utility(self):
        # Generate toy data, with 2 clusters
        X_inliers = ht.random.randn(100, 2, split=0)
        X_inliers = ht.concatenate((X_inliers + 2, X_inliers - 2), axis=0)

        # Add outliers
        X_outliers = ht.array(
            [[10, 10], [4, 7], [8, 3], [-2, 6], [5, -9], [-1, -10], [7, -2], [-6, 4], [-5, -8]],
            split=0,
        )
        X = ht.concatenate((X_inliers, X_outliers), axis=0)

        # Test lof with threshold
        lof = LocalOutlierFactor(n_neighbors=10, threshold=3)
        lof.fit(X)
        anomaly = lof.anomaly.numpy()
        condition = anomaly[-X_outliers.shape[0] :] == 1
        self.assertTrue(condition.all())

        # Test lof with top_n
        lof = LocalOutlierFactor(n_neighbors=10, binary_decision="top_n", top_n=X_outliers.shape[0])
        lof.fit(X)
        anomaly = lof.anomaly.numpy()
        condition = anomaly[-X_outliers.shape[0] :] == 1
        self.assertTrue(condition.all())
