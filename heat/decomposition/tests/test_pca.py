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
        self.assertEqual(pca.iterated_power, 0)
        self.assertEqual(pca.n_oversamples, 10)
        self.assertEqual(pca.power_iteration_normalizer, "qr")
        self.assertEqual(pca.random_state, None)

        # check catching of invalid parameters
        # wrong withening
        with self.assertRaises(NotImplementedError):
            ht.decomposition.PCA(whiten=True)
        # wrong iterated power
        with self.assertRaises(TypeError):
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
        # in-place not yet supported
        with self.assertRaises(NotImplementedError):
            ht.decomposition.PCA(copy=False)

    def test_pca_with_hierarchical_rank(self):
        # test fit
        rank = 3
        data = ht.random.randn(15 * ht.MPI_WORLD.size, 5, split=0)
        pca = ht.decomposition.PCA(n_components=rank, svd_solver="hierarchical")
        pca.fit(data)
        self.assertEqual(pca.components_.shape, (rank, 5))
        self.assertEqual(pca.n_components_, rank)

        self.assertEqual(pca.mean_.shape, (5,))
        self.assertTrue(
            isinstance(pca.total_explained_variance_ratio_, float)
            and pca.total_explained_variance_ratio_ > 0.0
            and pca.total_explained_variance_ratio_ <= 1.0
        )
        if ht.MPI_WORLD.size > 1:
            self.assertEqual(pca.noise_variance_, None)
            self.assertEqual(pca.explained_variance_, None)
            self.assertEqual(pca.explained_variance_ratio_, None)
            self.assertEqual(pca.singular_values_, None)

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
        if ht.MPI_WORLD.size > 1:
            with self.assertRaises(ValueError):
                pca.fit(ht.random.randn(5 * ht.MPI_WORLD.size, ht.MPI_WORLD.size, split=1))

    def test_pca_with_hiearchical_rtol(self):
        # test fit
        ratio = 0.9
        data = ht.random.randn(15 * ht.MPI_WORLD.size, 7, split=0)
        pca = ht.decomposition.PCA(n_components=ratio, svd_solver="hierarchical")
        pca.fit(data)
        self.assertEqual(pca.components_.shape[1], 7)
        self.assertTrue(0 < pca.components_.shape[0] <= 7)
        self.assertEqual(pca.mean_.shape, (7,))
        self.assertEqual(pca.n_components_, pca.components_.shape[0])
        self.assertTrue(
            isinstance(pca.total_explained_variance_ratio_, float)
            and pca.total_explained_variance_ratio_ >= 0.0
            and pca.total_explained_variance_ratio_ <= 1.0
        )
        self.assertTrue(pca.total_explained_variance_ratio_ >= ratio)
        if ht.MPI_WORLD.size > 1:
            self.assertEqual(pca.explained_variance_, None)
            self.assertEqual(pca.explained_variance_ratio_, None)
            self.assertEqual(pca.singular_values_, None)
            self.assertEqual(pca.noise_variance_, None)

        # rest has already been tested for hierarchical SVD with fixed rank

    def test_pca_with_full_rank(self):
        # test fit with tall skinny data, including check for wrong inputs
        data = ht.random.randn(15 * ht.MPI_WORLD.size, 5, split=0)
        pca = ht.decomposition.PCA(n_components=None, svd_solver="full")
        pca.fit(data)
        self.assertEqual(pca.components_.shape, (5, 5))
        self.assertEqual(pca.n_components_, 5)
        self.assertEqual(pca.explained_variance_.shape, (5,))
        self.assertEqual(pca.explained_variance_ratio_.shape, (5,))
        self.assertEqual(pca.singular_values_.shape, (5,))
        self.assertEqual(pca.mean_.shape, (5,))
        self.assertTrue(
            0.0 <= pca.total_explained_variance_ratio_ <= 1.0 + 1e-6
        )  # required due to numerical inaccuracies

        with self.assertRaises(TypeError):
            pca.fit(torch.randn(5 * ht.MPI_WORLD.size, 5))
        with self.assertRaises(ValueError):
            pca.fit(data, data)

        # test transform, including check for wrong inputs
        y0 = pca.transform(ht.random.randn(5 * ht.MPI_WORLD.size, 5, split=0))
        y1 = pca.transform(ht.random.randn(10, 5, split=1))
        self.assertEqual(y0.shape, (5 * ht.MPI_WORLD.size, 5))
        self.assertEqual(y1.shape, (10, 5))

        with self.assertRaises(ValueError):
            pca.transform(ht.random.randn(5 * ht.MPI_WORLD.size, 6, split=0))
        with self.assertRaises(TypeError):
            pca.transform("abc")

        # test fit transform and inverse transform, including check for wrong inputs
        y = pca.fit_transform(data)
        self.assertEqual(y.shape, (data.shape[0], 5))
        x = pca.inverse_transform(y)
        self.assertEqual(x.shape, data.shape)
        y.resplit_(1)
        x = pca.inverse_transform(y)
        self.assertEqual(x.shape, data.shape)

        with self.assertRaises(TypeError):
            pca.inverse_transform("abc")
        with self.assertRaises(ValueError):
            pca.inverse_transform(ht.random.randn(ht.MPI_WORLD.size, 6, split=0))

    def test_pca_with_full_rtol(self):
        # test fit
        ratio = 0.85
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
            and pca.total_explained_variance_ratio_ > 0.0
            and pca.total_explained_variance_ratio_ <= 1.0
        )
        self.assertTrue(pca.total_explained_variance_ratio_ >= ratio)
        self.assertEqual(pca.noise_variance_, None)

    def test_pca_randomized(self):
        rank = 2
        pca = ht.decomposition.PCA(n_components=rank, svd_solver="randomized")
        data = ht.random.randn(15 * ht.MPI_WORLD.size, 5, split=0)

        pca.fit(data)
        self.assertEqual(pca.components_.shape, (rank, 5))
        self.assertEqual(pca.n_components_, rank)
        self.assertEqual(pca.mean_.shape, (5,))

        if ht.MPI_WORLD.size > 1:
            self.assertEqual(pca.total_explained_variance_ratio_, None)
            self.assertEqual(pca.noise_variance_, None)
            self.assertEqual(pca.explained_variance_, None)
            self.assertEqual(pca.explained_variance_ratio_, None)
            self.assertEqual(pca.singular_values_, None)

        pca = ht.decomposition.PCA(n_components=None, svd_solver="randomized", random_state=1234)
        self.assertEqual(ht.random.get_state()[1], 1234)


class TestIncrementalPCA(TestCase):
    def test_incrementalpca_setup(self):
        pca = ht.decomposition.IncrementalPCA(n_components=2)

        # check correct base classes
        self.assertTrue(ht.is_estimator(pca))
        self.assertTrue(ht.is_transformer(pca))

        # check correct default values
        self.assertEqual(pca.n_components, 2)
        self.assertEqual(pca.whiten, False)
        self.assertEqual(pca.batch_size, None)
        self.assertEqual(pca.components_, None)
        self.assertEqual(pca.singular_values_, None)
        self.assertEqual(pca.mean_, None)
        self.assertEqual(pca.n_components_, None)
        self.assertEqual(pca.batch_size_, None)
        self.assertEqual(pca.n_samples_seen_, 0)

        # check catching of invalid parameters
        # whitening and in-place are not yet supported
        with self.assertRaises(NotImplementedError):
            ht.decomposition.IncrementalPCA(whiten=True)
        with self.assertRaises(NotImplementedError):
            ht.decomposition.IncrementalPCA(copy=False)
        # wrong n_components
        with self.assertRaises(TypeError):
            ht.decomposition.IncrementalPCA(n_components=0.9)
        with self.assertRaises(ValueError):
            ht.decomposition.IncrementalPCA(n_components=0)

    def test_incrementalpca_part1(self):
        # full rank is reached, split = 0
        # dtype float32
        pca = ht.decomposition.IncrementalPCA()
        data0 = ht.random.randn(150 * ht.MPI_WORLD.size, 2 * ht.MPI_WORLD.size + 1, split=0)
        data1 = 1.0 + ht.random.rand(50 * ht.MPI_WORLD.size, 2 * ht.MPI_WORLD.size + 1, split=0)
        data = ht.vstack([data0, data1])
        data0_np = data0.numpy()
        data_np = data.numpy()

        # fit is not yet implemented
        with self.assertRaises(NotImplementedError):
            pca.fit(data0)
        # wrong input for partial_fit
        with self.assertRaises(ValueError):
            pca.partial_fit(data0, y="Why can't we get rid of this argument?")

        # test partial_fit, step 0
        pca.partial_fit(data0)
        self.assertEqual(
            pca.components_.shape, (2 * ht.MPI_WORLD.size + 1, 2 * ht.MPI_WORLD.size + 1)
        )
        self.assertEqual(pca.n_components_, 2 * ht.MPI_WORLD.size + 1)
        self.assertEqual(pca.mean_.shape, (2 * ht.MPI_WORLD.size + 1,))
        self.assertEqual(pca.singular_values_.shape, (2 * ht.MPI_WORLD.size + 1,))
        self.assertEqual(pca.n_samples_seen_, 150 * ht.MPI_WORLD.size)
        s0_np = np.linalg.svd(data0_np - data0_np.mean(axis=0), compute_uv=False, hermitian=False)
        self.assertTrue(np.allclose(s0_np, pca.singular_values_.numpy()))

        # test partial_fit, step 1
        pca.partial_fit(data1)
        self.assertEqual(
            pca.components_.shape, (2 * ht.MPI_WORLD.size + 1, 2 * ht.MPI_WORLD.size + 1)
        )
        self.assertEqual(pca.n_components_, 2 * ht.MPI_WORLD.size + 1)
        self.assertTrue(ht.allclose(pca.mean_, ht.mean(data, axis=0)))
        self.assertEqual(pca.singular_values_.shape, (2 * ht.MPI_WORLD.size + 1,))
        self.assertEqual(pca.n_samples_seen_, 200 * ht.MPI_WORLD.size)
        s_np = np.linalg.svd(data_np - data_np.mean(axis=0), compute_uv=False, hermitian=False)
        self.assertTrue(np.allclose(s_np, pca.singular_values_.numpy()))

        # test transform (only possible here, as in the next test truncation happens)
        new_data = ht.random.rand(100, 2 * ht.MPI_WORLD.size + 1, split=1)
        Y = pca.transform(new_data)
        Z = pca.inverse_transform(Y)
        self.assertTrue(ht.allclose(new_data, Z, atol=1e-4, rtol=1e-4))

        # wrong inputs for transform and inverse transform
        with self.assertRaises(ValueError):
            pca.transform(ht.zeros((200, 2 * ht.MPI_WORLD.size + 2), split=0))
        with self.assertRaises(ValueError):
            pca.inverse_transform(ht.zeros((200, 2 * ht.MPI_WORLD.size + 2), split=0))

    def test_incrementalpca_part2(self):
        # full rank not reached, but truncation happens, split = 1
        # dtype float64
        pca = ht.decomposition.IncrementalPCA(n_components=15)
        data0 = ht.random.randn(9, 100 * ht.MPI_WORLD.size + 1, split=1, dtype=ht.float64)
        data1 = 1.0 + ht.random.rand(11, 100 * ht.MPI_WORLD.size + 1, split=1, dtype=ht.float64)
        data = ht.vstack([data0, data1])
        data0_np = data0.numpy()
        data_np = data.numpy()

        # test partial_fit, step 0
        pca.partial_fit(data0)
        self.assertEqual(pca.components_.shape, (9, 100 * ht.MPI_WORLD.size + 1))
        self.assertEqual(pca.components_.dtype, ht.float64)
        self.assertEqual(pca.n_components_, 9)
        self.assertEqual(pca.mean_.shape, (100 * ht.MPI_WORLD.size + 1,))
        self.assertEqual(pca.mean_.dtype, ht.float64)
        self.assertEqual(pca.singular_values_.shape, (9,))
        self.assertEqual(pca.singular_values_.dtype, ht.float64)
        self.assertEqual(pca.n_samples_seen_, 9)
        s0_np = np.linalg.svd(data0_np - data0_np.mean(axis=0), compute_uv=False, hermitian=False)
        self.assertTrue(np.allclose(s0_np, pca.singular_values_.numpy(), atol=1e-12))

        # test partial_fit, step 1
        # here actually truncation happens as we have rank 20 but n_components=15
        pca.partial_fit(data1)
        self.assertEqual(pca.components_.shape, (15, 100 * ht.MPI_WORLD.size + 1))
        self.assertEqual(pca.n_components_, 15)
        self.assertEqual(pca.mean_.shape, (100 * ht.MPI_WORLD.size + 1,))
        self.assertEqual(pca.singular_values_.shape, (15,))
        self.assertEqual(pca.n_samples_seen_, 20)
        s_np = np.linalg.svd(data_np - data_np.mean(axis=0), compute_uv=False, hermitian=False)
        self.assertTrue(np.allclose(s_np[:15], pca.singular_values_.numpy()))
