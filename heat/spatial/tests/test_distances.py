import unittest
import os

import torch

import heat as ht
import numpy as np
import math

from heat.core.tests.test_suites.basic_test import TestCase


class TestDistances(TestCase):
    def test_cdist(self):
        n = ht.communication.MPI_WORLD.size
        X = ht.ones((n * 2, 4), dtype=ht.float32, split=None)
        Y = ht.zeros((n * 2, 4), dtype=ht.float32, split=None)
        res_XX_cdist = ht.zeros((n * 2, n * 2), dtype=ht.float32, split=None)
        res_XX_rbf = ht.ones((n * 2, n * 2), dtype=ht.float32, split=None)
        res_XX_manhattan = ht.zeros((n * 2, n * 2), dtype=ht.float32, split=None)
        res_XY_cdist = ht.ones((n * 2, n * 2), dtype=ht.float32, split=None) * 2
        res_XY_rbf = ht.ones((n * 2, n * 2), dtype=ht.float32, split=None) * math.exp(-1.0)
        res_XY_manhattan = ht.ones((n * 2, n * 2), dtype=ht.float32, split=None) * 4

        # Case 1a: X.split == None, Y == None
        d = ht.spatial.cdist(X, quadratic_expansion=False)
        self.assertTrue(ht.allclose(d, res_XX_cdist))
        self.assertEqual(d.split, None)

        d = ht.spatial.cdist(X, quadratic_expansion=True)
        self.assertTrue(ht.allclose(d, res_XX_cdist))
        self.assertEqual(d.split, None)

        d = ht.spatial.rbf(X, quadratic_expansion=False)
        self.assertTrue(ht.allclose(d, res_XX_rbf))
        self.assertEqual(d.split, None)

        d = ht.spatial.rbf(X, quadratic_expansion=True)
        self.assertTrue(ht.allclose(d, res_XX_rbf))
        self.assertEqual(d.split, None)

        d = ht.spatial.manhattan(X, expand=False)
        self.assertTrue(ht.allclose(d, res_XX_manhattan))
        self.assertEqual(d.split, None)

        d = ht.spatial.manhattan(X, expand=True)
        self.assertTrue(ht.allclose(d, res_XX_manhattan))
        self.assertEqual(d.split, None)

        # Case 1b: X.split == None, Y != None, Y.split == None
        d = ht.spatial.cdist(X, Y, quadratic_expansion=False)
        self.assertTrue(ht.allclose(d, d.dtype(res_XY_cdist)))
        self.assertEqual(d.split, None)

        d = ht.spatial.cdist(X, Y, quadratic_expansion=True)
        self.assertTrue(ht.allclose(d, d.dtype(res_XY_cdist)))
        self.assertEqual(d.split, None)

        d = ht.spatial.rbf(X, Y, sigma=math.sqrt(2.0), quadratic_expansion=False)
        self.assertTrue(ht.allclose(d, res_XY_rbf))
        self.assertEqual(d.split, None)

        d = ht.spatial.rbf(X, Y, sigma=math.sqrt(2.0), quadratic_expansion=True)
        self.assertTrue(ht.allclose(d, res_XY_rbf))
        self.assertEqual(d.split, None)

        d = ht.spatial.manhattan(X, Y, expand=False)
        self.assertTrue(ht.allclose(d, d.dtype(res_XY_manhattan)))
        self.assertEqual(d.split, None)

        d = ht.spatial.manhattan(X, Y, expand=True)
        self.assertTrue(ht.allclose(d, d.dtype(res_XY_manhattan)))
        self.assertEqual(d.split, None)

        # Case 1c: X.split == None, Y != None, Y.split == 0
        Y = ht.zeros((n * 2, 4), dtype=ht.float32, split=0)
        res_XX_cdist = ht.zeros((n * 2, n * 2), dtype=ht.float32, split=1)
        res_XX_rbf = ht.ones((n * 2, n * 2), dtype=ht.float32, split=1)
        res_XY_cdist = ht.ones((n * 2, n * 2), dtype=ht.float32, split=1) * 2
        res_XY_rbf = ht.ones((n * 2, n * 2), dtype=ht.float32, split=1) * math.exp(-1.0)

        d = ht.spatial.cdist(X, Y, quadratic_expansion=False)
        self.assertTrue(ht.allclose(d, d.dtype(res_XY_cdist)))
        self.assertEqual(d.split, 1)

        d = ht.spatial.cdist(X, Y, quadratic_expansion=True)
        self.assertTrue(ht.allclose(d, d.dtype(res_XY_cdist)))
        self.assertEqual(d.split, 1)

        d = ht.spatial.rbf(X, Y, sigma=math.sqrt(2.0), quadratic_expansion=False)
        self.assertTrue(ht.allclose(d, res_XY_rbf))
        self.assertEqual(d.split, 1)

        d = ht.spatial.rbf(X, Y, sigma=math.sqrt(2.0), quadratic_expansion=True)
        self.assertTrue(ht.allclose(d, res_XY_rbf))
        self.assertEqual(d.split, 1)

        d = ht.spatial.manhattan(X, Y, expand=False)
        self.assertTrue(ht.allclose(d, d.dtype(res_XY_manhattan)))
        self.assertEqual(d.split, 1)

        d = ht.spatial.manhattan(X, Y, expand=True)
        self.assertTrue(ht.allclose(d, d.dtype(res_XY_manhattan)))
        self.assertEqual(d.split, 1)

        # Case 2a: X.split == 0, Y == None
        X = ht.ones((n * 2, 4), dtype=ht.float32, split=0)
        Y = ht.zeros((n * 2, 4), dtype=ht.float32, split=None)
        res_XX_cdist = ht.zeros((n * 2, n * 2), dtype=ht.float32, split=0)
        res_XX_rbf = ht.ones((n * 2, n * 2), dtype=ht.float32, split=0)
        res_XY_cdist = ht.ones((n * 2, n * 2), dtype=ht.float32, split=0) * 2
        res_XY_rbf = ht.ones((n * 2, n * 2), dtype=ht.float32, split=0) * math.exp(-1.0)

        d = ht.spatial.cdist(X, quadratic_expansion=False)
        self.assertTrue(ht.allclose(d, res_XX_cdist))
        self.assertEqual(d.split, 0)

        d = ht.spatial.cdist(X, quadratic_expansion=True)
        self.assertTrue(ht.allclose(d, res_XX_cdist))
        self.assertEqual(d.split, 0)

        d = ht.spatial.rbf(X, quadratic_expansion=False)
        self.assertTrue(ht.allclose(d, res_XX_rbf))
        self.assertEqual(d.split, 0)

        d = ht.spatial.rbf(X, quadratic_expansion=True)
        self.assertTrue(ht.allclose(d, res_XX_rbf))
        self.assertEqual(d.split, 0)

        d = ht.spatial.manhattan(X, expand=False)
        self.assertTrue(ht.allclose(d, res_XX_manhattan))
        self.assertEqual(d.split, 0)

        d = ht.spatial.manhattan(X, expand=True)
        self.assertTrue(ht.allclose(d, res_XX_manhattan))
        self.assertEqual(d.split, 0)

        # Case 2b: X.split == 0, Y != None, Y.split == None
        d = ht.spatial.cdist(X, Y, quadratic_expansion=False)
        self.assertTrue(ht.allclose(d, d.dtype(res_XY_cdist)))
        self.assertEqual(d.split, 0)

        d = ht.spatial.cdist(X, Y, quadratic_expansion=True)
        self.assertTrue(ht.allclose(d, d.dtype(res_XY_cdist)))
        self.assertEqual(d.split, 0)

        d = ht.spatial.rbf(X, Y, sigma=math.sqrt(2.0), quadratic_expansion=False)
        self.assertTrue(ht.allclose(d, res_XY_rbf))
        self.assertEqual(d.split, 0)

        d = ht.spatial.rbf(X, Y, sigma=math.sqrt(2.0), quadratic_expansion=True)
        self.assertTrue(ht.allclose(d, res_XY_rbf))
        self.assertEqual(d.split, 0)

        d = ht.spatial.manhattan(X, Y, expand=False)
        self.assertTrue(ht.allclose(d, d.dtype(res_XY_manhattan)))
        self.assertEqual(d.split, 0)

        d = ht.spatial.manhattan(X, Y, expand=True)
        self.assertTrue(ht.allclose(d, d.dtype(res_XY_manhattan)))
        self.assertEqual(d.split, 0)

        # Case 2c: X.split == 0, Y != None, Y.split == 0
        Y = ht.zeros((n * 2, 4), dtype=ht.float32, split=0)

        d = ht.spatial.cdist(X, Y, quadratic_expansion=False)
        self.assertTrue(ht.allclose(d, d.dtype(res_XY_cdist)))
        self.assertEqual(d.split, 0)

        d = ht.spatial.cdist(X, Y, quadratic_expansion=True)
        self.assertTrue(ht.allclose(d, d.dtype(res_XY_cdist)))
        self.assertEqual(d.split, 0)

        d = ht.spatial.rbf(X, Y, sigma=math.sqrt(2.0), quadratic_expansion=False)
        self.assertTrue(ht.allclose(d, res_XY_rbf))
        self.assertEqual(d.split, 0)

        d = ht.spatial.rbf(X, Y, sigma=math.sqrt(2.0), quadratic_expansion=True)
        self.assertTrue(ht.allclose(d, res_XY_rbf))
        self.assertEqual(d.split, 0)

        d = ht.spatial.manhattan(X, Y, expand=False)
        self.assertTrue(ht.allclose(d, d.dtype(res_XY_manhattan)))
        self.assertEqual(d.split, 0)

        d = ht.spatial.manhattan(X, Y, expand=True)
        self.assertTrue(ht.allclose(d, d.dtype(res_XY_manhattan)))
        self.assertEqual(d.split, 0)

        # Case 3 X.split == 1
        X = ht.ones((n * 2, 4), dtype=ht.float32, split=1)
        with self.assertRaises(NotImplementedError):
            ht.spatial.cdist(X)
        with self.assertRaises(NotImplementedError):
            ht.spatial.cdist(X, Y, quadratic_expansion=False)
        X = ht.ones((n * 2, 4), dtype=ht.float32, split=None)
        Y = ht.zeros((n * 2, 4), dtype=ht.float32, split=1)
        with self.assertRaises(NotImplementedError):
            ht.spatial.cdist(X, Y, quadratic_expansion=False)

        Z = ht.ones((n * 2, 6, 3), dtype=ht.float32, split=None)
        with self.assertRaises(NotImplementedError):
            ht.spatial.cdist(Z, quadratic_expansion=False)
        with self.assertRaises(NotImplementedError):
            ht.spatial.cdist(X, Z, quadratic_expansion=False)

        n = ht.communication.MPI_WORLD.size
        A = ht.ones((n * 2, 6), dtype=ht.float32, split=None)
        for i in range(n):
            A[2 * i, :] = A[2 * i, :] * (2 * i)
            A[2 * i + 1, :] = A[2 * i + 1, :] * (2 * i + 1)
        res = torch.cdist(A.larray, A.larray)

        A = ht.ones((n * 2, 6), dtype=ht.float32, split=0)
        for i in range(n):
            A[2 * i, :] = A[2 * i, :] * (2 * i)
            A[2 * i + 1, :] = A[2 * i + 1, :] * (2 * i + 1)
        B = A.astype(ht.int32)

        d = ht.spatial.cdist(A, B, quadratic_expansion=False)
        result = ht.array(res, dtype=ht.float32, split=0)
        self.assertTrue(ht.allclose(d, result, atol=1e-5))

        n = ht.communication.MPI_WORLD.size
        A = ht.ones((n * 2, 6), dtype=ht.float32, split=None)
        for i in range(n):
            A[2 * i, :] = A[2 * i, :] * (2 * i)
            A[2 * i + 1, :] = A[2 * i + 1, :] * (2 * i + 1)
        res = torch.cdist(A.larray, A.larray)

        A = ht.ones((n * 2, 6), dtype=ht.float32, split=0)
        for i in range(n):
            A[2 * i, :] = A[2 * i, :] * (2 * i)
            A[2 * i + 1, :] = A[2 * i + 1, :] * (2 * i + 1)
        B = A.astype(ht.int32)

        d = ht.spatial.cdist(A, B, quadratic_expansion=False)
        result = ht.array(res, dtype=ht.float32, split=0)
        self.assertTrue(ht.allclose(d, result, atol=1e-8))

        if not self.is_mps:
            B = A.astype(ht.float64)
            d = ht.spatial.cdist(A, B, quadratic_expansion=False)
            result = ht.array(res, dtype=ht.float64, split=0)
            self.assertTrue(ht.allclose(d, result, atol=1e-8))

        B = A.astype(ht.int16)
        d = ht.spatial.cdist(A, B, quadratic_expansion=False)
        result = ht.array(res, dtype=ht.float32, split=0)
        self.assertTrue(ht.allclose(d, result, atol=1e-8))

        d = ht.spatial.cdist(B, quadratic_expansion=False)
        result = ht.array(res, dtype=ht.float32, split=0)
        self.assertTrue(ht.allclose(d, result, atol=1e-8))

        B = A.astype(ht.int32)
        d = ht.spatial.cdist(B, quadratic_expansion=False)
        result = ht.array(res, dtype=ht.float32, split=0)
        self.assertTrue(ht.allclose(d, result, atol=1e-8))

        if not self.is_mps:
            B = A.astype(ht.float64)
            d = ht.spatial.cdist(B, quadratic_expansion=False)
            result = ht.array(res, dtype=ht.float64, split=0)
            self.assertTrue(ht.allclose(d, result, atol=1e-8))

    def test_cdist_small(self):
        ht.random.seed(10)
        n_neighbors = 10
        X = ht.random.rand(1000, 100, dtype=ht.float32, split=0)
        Y = ht.random.rand(1500, 100, dtype=ht.float32, split=0)

        # Test functionality
        d = ht.spatial.cdist(X, Y, quadratic_expansion=False)
        std_dist, std_idx = ht.topk(d, k=n_neighbors, dim=1, largest=False)
        dist, idx = ht.spatial.cdist_small(X, Y, n_smallest=n_neighbors)
        self.assertTrue(ht.allclose(std_dist, dist, atol=1e-8))
        print(std_dist - dist)
        # Note: if some distances in the same row of the distance matrix are the same,
        # the respective indices in this comarison may differ (randomly ordered)
        self.assertTrue(ht.allclose(std_idx, idx, atol=1e-8))

        # Test functionality with chunk-wise computation
        dist_chunked, idx_chunked = ht.spatial.cdist_small(X, Y, chunks=1, n_smallest=n_neighbors)
        self.assertTrue(ht.allclose(std_dist, dist_chunked, atol=1e-8))
        self.assertTrue(ht.allclose(std_idx, idx_chunked, atol=1e-8))

        # Splitting
        X = ht.random.rand(1000, 100, dtype=ht.float32, split=None)
        Y = ht.random.rand(1500, 100, dtype=ht.float32, split=0)
        Z = ht.random.rand(2000, 100, dtype=ht.float32, split=1)
        with self.assertRaises(NotImplementedError):
            ht.spatial.cdist_small(X, Y)
        with self.assertRaises(NotImplementedError):
            ht.spatial.cdist_small(Y, X)
        with self.assertRaises(NotImplementedError):
            ht.spatial.cdist_small(X, Z)
        with self.assertRaises(NotImplementedError):
            ht.spatial.cdist_small(Z, X)
        with self.assertRaises(NotImplementedError):
            ht.spatial.cdist_small(Y, Z)
        with self.assertRaises(NotImplementedError):
            ht.spatial.cdist_small(Z, Y)

        # Non-matching shape[1]
        X = ht.random.rand(1000, 100, dtype=ht.float32, split=0)
        Y = ht.random.rand(1500, 150, dtype=ht.float32, split=0)
        with self.assertRaises(ValueError):
            ht.spatial.cdist_small(X, Y)

        # More neighbors than points
        n_smallest = 2000
        X = ht.random.rand(1000, 100, dtype=ht.float32, split=0)
        Y = ht.random.rand(1500, 100, dtype=ht.float32, split=0)
        with self.assertRaises(ValueError):
            ht.spatial.cdist_small(X, Y, n_smallest=n_smallest)
