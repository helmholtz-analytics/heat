import os
import unittest
import numpy as np
import torch
import heat as ht

from ...core.tests.test_suites.basic_test import TestCase


class TestPCA(TestCase):
    def test_pca_setup(self):
        pca = ht.decomposition.PCA()

        # check correct base classes
        self.assertTrue(ht.is_estimator(pca))
        self.assertTrue(ht.is_transformer(pca))

        # check correct default values
        self.assertEqual(pca.n_components, None)
        self.assertEqual(pca.whiten, False)
        self.assertEqual(pca.svd_solver, "hierarchical")
        self.assertEqual(pca.tol, None)
        self.assertEqual(pca.iterated_power, "auto")
        self.assertEqual(pca.n_oversamples, 10)
        self.assertEqual(pca.power_iteration_normalizer, "qr")
        self.assertEqual(pca.random_state, 0)

        # check catching of invalid parameters
        # wrong withening
        with self.assertRaises(ValueError):
            ht.decomposition.PCA(whiten=True)
        # wrong iterated power
        with self.assertRaises(ValueError):
            ht.decomposition.PCA(iterated_power=0.5)
        with self.assertRaises(ValueError):
            ht.decomposition.PCA(iterated_power=-1)
        # wrong power_iteration_normalizer
        with self.assertRaises(ValueError):
            ht.decomposition.PCA(power_iteration_normalizer="LU")
        # wrong n_oversamples
        with self.assertRaises(ValueError):
            ht.decomposition.PCA(n_oversamples=-1)
        # wrong tol
        with self.assertRaises(ValueError):
            ht.decomposition.PCA(tol=1e-10)
        # wrong random_state
        with self.assertRaises(ValueError):
            ht.decomposition.PCA(random_state=0.1234)
        # wrong n_components
        with self.assertRaises(ValueError):
            ht.decomposition.PCA(n_components="mle")
        # wrong svd_solver
        with self.assertRaises(ValueError):
            ht.decomposition.PCA(svd_solver="arpack")

    def test_pca_with_hierarchical_rank(self):
        split = 1
        data = ht.random.randn(100, 5, split=split)
        pca = ht.decomposition.PCA(n_components=3, svd_solver="hierarchical")
        pca.fit(data)
        self.assertEqual(pca.components_.shape, (3, 5))
        self.assertEqual(pca.explained_variance_.shape, (3,))
        self.assertEqual(pca.explained_variance_ratio_.shape, (3,))
        self.assertEqual(pca.singular_values_.shape, (3,))
        self.assertEqual(pca.mean_.shape, (5,))
        self.assertEqual(pca.noise_variance_, 0)

    def test_pca_with_hiearchical_rtol(self):
        pass

    def test_pca_with_full(self):
        pass
