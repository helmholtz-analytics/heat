import unittest
import heat as ht
import numpy as np

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
        ht.random.seed(42)  # For reproducibility
        X_inliers = ht.random.randn(100, 2, split=0)
        X_inliers = ht.concatenate((X_inliers + 2, X_inliers - 2), axis=0)
        n_neighbors = 10

        # Add outliers
        X_outliers = ht.array(
            [[6, 9], [4, 7], [8, 3], [-2, 6], [5, -9], [-1, -10], [7, -2], [-6, 4], [-5, -8]],
            split=0,
        )
        X = ht.concatenate((X_inliers, X_outliers), axis=0)

        # Test lof with threshold
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, threshold=3)
        lof.fit(X)
        anomaly = lof.anomaly.numpy()
        condition = anomaly[-X_outliers.shape[0] :] == 1
        self.assertTrue(condition.all())

        # Test lof with top_n
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors, binary_decision="top_n", top_n=X_outliers.shape[0]
        )
        lof.fit(X)
        anomaly = lof.anomaly.numpy()
        condition = anomaly[-X_outliers.shape[0] :] == 1
        self.assertTrue(condition.all())

        # Compare with scikit-learn's LocalOutlierFactor
        # (hard-coded for reusability without sklearn installation)
        X_inliers = ht.array(
            [
                [0.1, 0.2],
                [0.2, 0.1],
                [0.15, 0.25],
                [0.3, 0.1],
                [0.25, 0.15],
                [0.05, 0.05],
                [-0.1, 0.0],
                [0.0, -0.1],
                [-0.2, -0.2],
                [-0.15, 0.1],
                [0.1, -0.15],
                [0.05, 0.2],
                [-0.25, 0.05],
                [0.2, -0.2],
                [-0.2, 0.2],
                [0.1, 0.0],
                [0.0, 0.1],
                [-0.1, -0.1],
                [0.15, -0.05],
            ],
            split=0,
            dtype=ht.float64,
        )

        X_inliers = ht.concatenate((X_inliers + 2, X_inliers - 2), axis=0)
        X = ht.concatenate((X_inliers, X_outliers), axis=0)

        # Following sklearn results can be reproduced using
        #   >>> X= X.resplit_(None).larray
        #   >>> skLOF = sklearn.neighbors.LocalOutlierFactor(n_neighbors, metric='euclidean', algorithm='brute')
        #   >>> skLOF.fit(X)
        #   >>> sklearn_result = - skLOF.negative_outlier_factor_
        sklearn_result = np.array(
            [
                0.99108349,
                1.00418816,
                1.03426844,
                1.06724007,
                1.01458797,
                0.94845131,
                0.99696432,
                0.99032559,
                1.17582066,
                0.98378393,
                0.99078099,
                1.01103704,
                1.11724802,
                1.10750862,
                1.09542395,
                0.97165935,
                0.95689391,
                0.99475836,
                1.00595599,
                0.99057196,
                1.00366992,
                1.03373667,
                1.06668784,
                1.01406486,
                0.94796281,
                0.99696432,
                0.98980558,
                1.17521084,
                0.98378393,
                0.99026041,
                1.01103704,
                1.11724802,
                1.10693752,
                1.09542395,
                0.97115928,
                0.95640055,
                0.99423509,
                1.01355282,
                22.03163408,
                18.250704,
                17.44611921,
                18.85830019,
                27.50529293,
                23.25407642,
                20.78187176,
                22.96233196,
                21.68260391,
            ]
        )
        sklearn_result = ht.array(sklearn_result, split=0)

        # test with run-time-efficient implementation
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, fully_distributed=False)
        lof.fit(X)
        lof_scores = lof.lof_scores
        condition = ht.allclose(lof_scores, sklearn_result, atol=1e-2)
        self.assertTrue(condition)

        # test with memory-efficient implementation
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, fully_distributed=True)
        lof.fit(X)
        lof_scores = lof.lof_scores
        condition = ht.allclose(lof_scores, sklearn_result, atol=1e-2)
        self.assertTrue(condition)
