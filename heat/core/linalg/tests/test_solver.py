import torch
import os
import unittest
import heat as ht
import numpy as np

from ...tests.test_suites.basic_test import TestCase


class TestSolver(TestCase):
    def test_cg(self):
        size = ht.communication.MPI_WORLD.size * 3
        b = ht.arange(1, size + 1, dtype=ht.float32, split=0)
        A = ht.manipulations.diag(b)
        x0 = ht.random.rand(size, dtype=b.dtype, split=b.split)

        x = ht.ones(b.shape, dtype=b.dtype, split=b.split)

        res = ht.linalg.cg(A, b, x0)
        self.assertTrue(ht.allclose(x, res, atol=1e-3))

        b_np = np.arange(1, size + 1)
        with self.assertRaises(TypeError):
            ht.linalg.cg(A, b_np, x0)

        with self.assertRaises(RuntimeError):
            ht.linalg.cg(A, A, x0)
        with self.assertRaises(RuntimeError):
            ht.linalg.cg(b, b, x0)
        with self.assertRaises(RuntimeError):
            ht.linalg.cg(A, b, A)

    def test_lanczos(self):
        # define time and space domains
        x = np.linspace(-10, 10, 100)
        t = np.linspace(0, 6 * np.pi, 800)
        # dt = t[2] - t[1]
        Xm, Tm = np.meshgrid(x, t)

        # create three spatiotemporal patterns
        r = 3
        f1 = np.multiply(20 - 0.2 * np.power(Xm, 2), np.exp((2.3j) * Tm))
        f2 = np.multiply(Xm, np.exp(0.6j * Tm))
        f3 = np.multiply(
            5 * np.multiply(1 / np.cosh(Xm / 2), np.tanh(Xm / 2)), 2 * np.exp((0.1 + 2.8j) * Tm)
        )

        # combine signals and make data matrix
        D = (f1 + f2 + f3).T

        # generarte correlation matrix
        CorrMatrix = D.real @ D.real.T

        # Manual SVD on Correlation Matrix
        SigSquare, U = np.linalg.eig(CorrMatrix)
        idxs = SigSquare.argsort()[::-1]
        SigSquare_, U_ = SigSquare[idxs][:r], U[:, idxs][:, :r]

        # with Lanczos
        ht_CorrMatrix = ht.array(CorrMatrix, dtype=ht.float64, split=0)
        ht_V, ht_T = ht.lanczos(ht_CorrMatrix, m=6)
        np_T = ht_T.numpy()
        np_V = ht_V.numpy()
        lanczos_vals, lanczos_vecs = np.linalg.eig(np_T)
        lanczos_idxs = lanczos_vals.argsort()[::-1]
        lanczos_vals_ = lanczos_vals[lanczos_idxs][:r]
        lanczos_vecs_ = lanczos_vecs[:, lanczos_idxs][:, :r]

        eigen_vecs_ = np_V @ lanczos_vecs_
        eigen_vecs_ = eigen_vecs_[:, :r]

        self.assertTrue(np.allclose(SigSquare_, lanczos_vals_, atol=1e-7))
        self.assertTrue(np.allclose(U_.real, eigen_vecs_, atol=1e-7))
