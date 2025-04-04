import heat as ht
import unittest
import numpy as np

from ...tests.test_suites.basic_test import TestCase


class TestEigh(TestCase):
    def _check_eigh_result(self, X, Lambda, H):
        dtypetol = 1e-3 if X.dtype == ht.float32 else 1e-3
        self.assertEqual(Lambda.shape, (X.shape[0],))
        self.assertEqual(H.shape, X.shape)
        self.assertEqual(H.split, X.split)
        self.assertEqual(Lambda.split, 0)
        self.assertEqual(H.dtype, X.dtype)
        self.assertEqual(Lambda.dtype, X.dtype)
        X_rec = H @ ht.diag(Lambda) @ H.T
        self.assertTrue(ht.norm(X - X_rec) / ht.norm(X) < dtypetol)
        HtH = H.T @ H
        eye_size_H = ht.eye(100, split=HtH.split, dtype=X.dtype)
        self.assertTrue(ht.norm(HtH - eye_size_H) / ht.norm(eye_size_H) < dtypetol)

    def test_eigh(self):
        # test with default values
        splits = [0, None]
        dtypes = [ht.float32, ht.float64]
        for split in splits:
            for dtype in dtypes:
                with self.subTest(split=split, dtype=dtype):
                    X = ht.random.randn(100, 100, split=split, dtype=dtype)
                    X = X + X.T.resplit_(X.split)
                    Lambda, H = ht.linalg.eigh(X)
                    self._check_eigh_result(X, Lambda, H)

            # test non-default options
            X = ht.random.randn(100, 100, split=0, dtype=ht.float32)
            X = X @ X.T
            Lambda, H = ht.linalg.eigh(X, r_max_zolopd=1, silent=False)
            self._check_eigh_result(X, Lambda, H)

    def test_eigh_catch_wrong_inputs(self):
        X = ht.random.rand(100, 101, split=0, dtype=ht.float32)
        with self.assertRaises(ValueError):
            ht.linalg.eigh(X)
