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
        # test fit
        rank = 3
        data = ht.random.randn(15 * ht.MPI_WORLD.size, 5, split=0)
        pca = ht.decomposition.PCA(n_components=rank, svd_solver="hierarchical")
        pca.fit(data)
        self.assertEqual(pca.components_.shape, (rank, 5))
        self.assertEqual(pca.n_components_, rank)
        self.assertEqual(pca.explained_variance_, None)
        self.assertEqual(pca.explained_variance_ratio_, None)
        self.assertEqual(pca.singular_values_, None)
        self.assertEqual(pca.mean_.shape, (5,))
        self.assertTrue(
            isinstance(pca.total_explained_variance_ratio_, float)
        )  # and pca.total_explained_variance_ratio_ > 0)
        self.assertEqual(pca.noise_variance_, None)

        # test transform
        y0 = pca.transform(ht.random.randn(5 * ht.MPI_WORLD.size, 5, split=0))
        y1 = pca.transform(ht.random.randn(10, 5, split=1))
        self.assertEqual(y0.shape, (5 * ht.MPI_WORLD.size, rank))
        self.assertEqual(y1.shape, (10, rank))

        with self.assertRaises(ValueError):
            pca.transform(ht.random.randn(5 * ht.MPI_WORLD.size, 6, split=0))

        # test fit transform and inverse transform
        y = pca.fit_transform(data)
        self.assertEqual(y.shape, (data.shape[0], rank))
        x = pca.inverse_transform(y)
        self.assertEqual(x.shape, data.shape)
        y.resplit_(1)
        x = pca.inverse_transform(y)
        self.assertEqual(x.shape, data.shape)

        # catch split=1 as wrong input
        with self.assertRaises(ValueError):
            pca.inverse_transform(ht.random.randn(5 * ht.MPI_WORLD.size, 5, split=1))

    def test_pca_with_hiearchical_rtol(self):
        # test fit
        ratio = 0.95
        data = ht.random.randn(15 * ht.MPI_WORLD.size, 7, split=0)
        pca = ht.decomposition.PCA(n_components=ratio, svd_solver="hierarchical")
        pca.fit(data)
        self.assertEqual(pca.components_.shape[1], 7)
        self.assertTrue(0 < pca.components_.shape[0] <= 7)
        self.assertEqual(pca.explained_variance_, None)
        self.assertEqual(pca.explained_variance_ratio_, None)
        self.assertEqual(pca.singular_values_, None)
        self.assertEqual(pca.mean_.shape, (7,))
        self.assertEqual(pca.n_components_, pca.components_.shape[0])
        self.assertTrue(
            isinstance(pca.total_explained_variance_ratio_, float)
        )  # and pca.total_explained_variance_ratio_ > 0)
        # self.assertTrue(pca.total_explained_variance_ratio_ >= ratio)
        self.assertEqual(pca.noise_variance_, None)

        # rest has already been tested for hierarchical SVD with fixed rank

    def test_pca_with_full_rank(self):
        # test fit with tall skinny data, including check for wrong inputs
        rank = 2
        data = ht.random.randn(15 * ht.MPI_WORLD.size, 5, split=0)
        pca = ht.decomposition.PCA(n_components=rank, svd_solver="full")
        pca.fit(data)
        self.assertEqual(pca.components_.shape, (2, 5))
        self.assertEqual(pca.n_components_, 2)
        self.assertEqual(pca.explained_variance_.shape, (2,))
        self.assertEqual(pca.explained_variance_ratio_.shape, (2,))
        self.assertEqual(pca.singular_values_.shape, (2,))
        self.assertEqual(pca.mean_.shape, (5,))
        self.assertTrue(0 < pca.total_explained_variance_ratio_ <= 1)

        with self.assertRaises(ValueError):
            pca.fit(torch.randn(5 * ht.MPI_WORLD.size, 5))
        with self.assertRaises(ValueError):
            pca.fit(data, data)

        # test transform, including check for wrong inputs
        y0 = pca.transform(ht.random.randn(5 * ht.MPI_WORLD.size, 5, split=0))
        y1 = pca.transform(ht.random.randn(10, 5, split=1))
        self.assertEqual(y0.shape, (5 * ht.MPI_WORLD.size, rank))
        self.assertEqual(y1.shape, (10, rank))

        with self.assertRaises(ValueError):
            pca.transform(ht.random.randn(5 * ht.MPI_WORLD.size, 6, split=0))
        with self.assertRaises(ValueError):
            pca.transform("abc")

        # test fit transform and inverse transform, including check for wrong inputs
        y = pca.fit_transform(data)
        self.assertEqual(y.shape, (data.shape[0], rank))
        x = pca.inverse_transform(y)
        self.assertEqual(x.shape, data.shape)
        y.resplit_(1)
        x = pca.inverse_transform(y)
        self.assertEqual(x.shape, data.shape)

        with self.assertRaises(ValueError):
            pca.inverse_transform("abc")

    def test_pca_with_full_rtol(self):
        # test fit
        ratio = 0.5
        data = ht.random.randn(15 * ht.MPI_WORLD.size, 7, split=0)
        pca = ht.decomposition.PCA(n_components=ratio, svd_solver="full")
        pca.fit(data)
        self.assertEqual(pca.components_.shape[1], 7)
        self.assertTrue(0 < pca.components_.shape[0] <= 7)
        self.assertEqual(pca.explained_variance_.shape, (pca.n_components_,))
        self.assertEqual(pca.explained_variance_ratio_.shape, (pca.n_components_,))
        self.assertEqual(pca.singular_values_.shape, (pca.n_components_,))
        self.assertEqual(pca.mean_.shape, (7,))
        self.assertEqual(pca.n_components_, pca.components_.shape[0])
        self.assertTrue(
            isinstance(pca.total_explained_variance_ratio_, float)
            and pca.total_explained_variance_ratio_ > 0
        )
        self.assertTrue(pca.total_explained_variance_ratio_ >= ratio)
        self.assertEqual(pca.noise_variance_, None)
