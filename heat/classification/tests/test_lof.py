import unittest
import heat as ht
import numpy as np

from heat.classification.localoutlierfactor import LocalOutlierFactor
from heat.core.tests.test_suites.basic_test import TestCase


class TestLOF(TestCase):
    def test_exception(self):
        """
        Tests for exceptions in LocalOutlierFactor.
        """
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

    def _setup_lof_dataset(self):
        """
        Helper method to construct dataset for LOF tests.

        Returns:
            X (DNDarray): Combined dataset (50 inliers + 5 outliers)
            n_outliers (int): Number of outliers (5)
            sklearn_result (DNDarray): Expected LOF scores from scikit-learn
        """
        # ------------------------------------------------------------------
        # Construct data set
        #    - 50 inliers: Gaussian cluster around (0, 0)
        #    - 5 clearly separated outliers far away from the cluster
        # ------------------------------------------------------------------
        rng = np.random.RandomState(123)

        # Inliers: single dense cluster, low chance of exact duplicate distances
        X_inliers_np = rng.normal(loc=0.0, scale=0.8, size=(50, 2))

        # Outliers: well-separated, not forming a dense counter-cluster
        X_outliers_np = np.array(
            [
                [6.0, 6.0],
                [-6.5, 6.0],
                [6.5, -6.0],
                [-6.0, -6.5],
                [0.0, 8.0],
            ],
            dtype=np.float64,
        )

        X_np = np.vstack([X_inliers_np, X_outliers_np])
        n_outliers = X_outliers_np.shape[0]

        X = ht.array(X_np, split=0, dtype=ht.float64, device=self.device)

        # ------------------------------------------------------------------
        # Construct data set for consistency check with the scikit-learn implementation:
        #    The following scikit-learn results can be reproduced using
        #    >>> X= X.resplit_(None).larray
        #    >>> skLOF = sklearn.neighbors.LocalOutlierFactor(n_neighbors, metric='euclidean', algorithm='brute')
        #    >>> skLOF.fit(X)
        #    >>> sklearn_result = - skLOF.negative_outlier_factor_
        # ------------------------------------------------------------------
        sklearn_result=np.array([1.0451677 , 0.97246276, 1.05081738, 1.41589941, 1.00463741,
           0.94233711, 1.01496385, 0.97546921, 1.29098113, 1.02392189,
           1.03969391, 0.99881874, 1.03134108, 1.01905314, 0.96573209,
           1.49743089, 1.1818625 , 0.98563474, 0.97014285, 0.9746302 ,
           1.10869988, 0.99776567, 0.9553028 , 1.19799836, 1.19699439,
           1.06447612, 1.0516235 , 0.99328519, 1.11292566, 1.09032844,
           1.02628087, 0.96525917, 1.06084697, 0.95882729, 0.97700327,
           1.00376853, 1.0174526 , 1.35802438, 0.97794061, 1.0535402 ,
           0.99089245, 1.08928467, 1.0049388 , 1.01353299, 1.08469539,
           1.01231012, 1.00256663, 1.00926798, 1.06179548, 0.96298944,
           5.55093291, 7.60215346, 7.99742319, 7.75727456, 5.6316978 ])

        sklearn_result = ht.array(sklearn_result, split=0)

        return X, n_outliers, sklearn_result



    def _test_utility(self, fully_distributed, n_neighbors=10):
        """
        Helper function and consistency tests for LocalOutlierFactor.
        """
        X, n_outliers, sklearn_result = self._setup_lof_dataset()

        # ------------------------------------------------------------------
        # 1) LOF with threshold-based decision
        #    Threshold chosen safely above typical inlier-LOF values
        # ------------------------------------------------------------------
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            binary_decision="threshold",
            threshold=3.0,
            fully_distributed=fully_distributed,
        )
        lof.fit(X)
        anomaly = lof.anomaly.numpy()

        # All inliers should be classified as inliers (-1),
        # the 5 explicit outliers as outliers (+1)
        self.assertTrue(np.all(anomaly[:-n_outliers] == -1))
        self.assertTrue(np.all(anomaly[-n_outliers:] == 1))

        # ------------------------------------------------------------------
        # 2) LOF with top_n-based decision
        #    Select the last n_outliers points (the far-away ones)
        # ------------------------------------------------------------------
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            binary_decision="top_n",
            top_n=n_outliers,
            fully_distributed=fully_distributed,
        )
        lof.fit(X)
        anomaly = lof.anomaly.numpy()

        # The last n_outliers samples must be flagged as outliers
        self.assertTrue(np.all(anomaly[-n_outliers:] == 1))

        # ------------------------------------------------------------------
        # 3) Consistency check:
        #    compare the lof scores with the scikit-learn implementation
        # ------------------------------------------------------------------
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, fully_distributed=fully_distributed)
        lof.fit(X)
        lof_scores = lof.lof_scores

        condition = ht.allclose(lof_scores, sklearn_result, atol=1e-6, rtol=1e-6)
        self.assertTrue(condition)


    def _test_utility_runtime_efficient(self):
        """
        Tests LocalOutlierFactor with a runtime efficient implementation.
        """
        n_neighbors = 10
        self._test_utility(self, fully_distributed=False, n_neighbors=n_neighbors)


    def _test_utility_memory_efficient(self):
        """
        Tests LocalOutlierFactor with a memory efficient implementation.
        """
        n_neighbors = 10
        self._test_utility(self, fully_distributed=True, n_neighbors=n_neighbors)

    def test_map_idx_to_proc(self):
        lof = LocalOutlierFactor()
        comm = ht.communication.MPI_WORLD
        size = comm.Get_size()

        # Pick an array length that is usually not divisible by number of ranks
        n = size * 3 + 1

        # --- 1D test case ---------------------------------------------------
        idx_1d = ht.arange(n, split=0, dtype=ht.int64)
        mapped_1d = lof._map_idx_to_proc(idx_1d, comm)

        # Expected rank assignment according to block distribution
        _, displ, _ = comm.counts_displs_shape(idx_1d.shape, idx_1d.split)
        expected_1d = np.empty(n, dtype=np.int64)
        for rank in range(size):
            lower = int(displ[rank])
            upper = n if rank == size - 1 else int(displ[rank + 1])
            expected_1d[lower:upper] = rank

        self.assertEqual(mapped_1d.shape, idx_1d.shape)
        self.assertEqual(mapped_1d.split, idx_1d.split)
        self.assertEqual(mapped_1d.dtype, idx_1d.dtype)
        self.assertTrue(np.array_equal(mapped_1d.numpy(), expected_1d))

        # --- 2D test case ---------------------------------------------------
        rng = np.random.RandomState(123)
        idx_np = rng.randint(0, n, size=(n, 4)).astype(np.int64)
        idx_2d = ht.array(idx_np, split=0, dtype=ht.int64)
        mapped_2d = lof._map_idx_to_proc(idx_2d, comm)

        # Expected mapping via searchsorted on displacement boundaries
        displ_np = np.asarray(displ, dtype=np.int64)
        expected_2d = np.searchsorted(displ_np[1:], idx_np, side="right")

        self.assertEqual(mapped_2d.shape, idx_2d.shape)
        self.assertEqual(mapped_2d.split, idx_2d.split)
        self.assertEqual(mapped_2d.dtype, idx_2d.dtype)
        self.assertTrue(np.array_equal(mapped_2d.numpy(), expected_2d))
