import os
import unittest

import heat as ht

from heat.core.tests.test_suites.basic_test import TestCase


class TestLaplacian(TestCase):
    def test_laplacian(self):
        size = ht.communication.MPI_WORLD.size
        rank = ht.communication.MPI_WORLD.rank
        X = ht.ones((size * 2, 4), split=0)
        X._DNDarray__array[0, :] *= rank
        X._DNDarray__array[1, :] *= rank + 0.5

        L = ht.graph.Laplacian(lambda x: ht.spatial.cdist(x, quadratic_expansion=True))
        res = L.construct(X)
        self.assertIsInstance(res, ht.DNDarray)
        self.assertEqual(res.shape, (size * 2, size * 2))
        self.assertEqual(res.split, 0)

        L = ht.graph.Laplacian(lambda x: ht.spatial.rbf(x, sigma=1.0, quadratic_expansion=True))
        res = L.construct(X)
        self.assertIsInstance(res, ht.DNDarray)
        self.assertEqual(res.shape, (size * 2, size * 2))
        self.assertEqual(res.split, 0)

        L = ht.graph.Laplacian(
            lambda x: ht.spatial.cdist(x, quadratic_expansion=True), weighted=False
        )
        res = L.construct(X)
        self.assertIsInstance(res, ht.DNDarray)
        self.assertEqual(res.shape, (size * 2, size * 2))
        self.assertEqual(res.split, 0)

        L = ht.graph.Laplacian(
            lambda x: ht.spatial.cdist(x, quadratic_expansion=True), definition="simple"
        )
        res = L.construct(X)
        self.assertIsInstance(res, ht.DNDarray)
        self.assertEqual(res.shape, (size * 2, size * 2))
        self.assertEqual(res.split, 0)

        L = ht.graph.Laplacian(
            lambda x: ht.spatial.cdist(x, quadratic_expansion=True), mode="eNeighbour"
        )
        res = L.construct(X)
        self.assertIsInstance(res, ht.DNDarray)
        self.assertEqual(res.shape, (size * 2, size * 2))
        self.assertEqual(res.split, 0)

        L = ht.graph.Laplacian(
            lambda x: ht.spatial.cdist(x, quadratic_expansion=True),
            mode="eNeighbour",
            threshold_key="lower",
            threshold_value=3.0,
        )
        res = L.construct(X)
        self.assertIsInstance(res, ht.DNDarray)
        self.assertEqual(res.shape, (size * 2, size * 2))
        self.assertEqual(res.split, 0)

        with self.assertRaises(ValueError):
            L = ht.graph.Laplacian(
                lambda x: ht.spatial.cdist(x, quadratic_expansion=True), threshold_key="both"
            )
        with self.assertRaises(NotImplementedError):
            L = ht.graph.Laplacian(
                lambda x: ht.spatial.cdist(x, quadratic_expansion=True), mode="kNN"
            )
        with self.assertRaises(NotImplementedError):
            L = ht.graph.Laplacian(
                lambda x: ht.spatial.cdist(x, quadratic_expansion=True), definition="norm_rw"
            )
