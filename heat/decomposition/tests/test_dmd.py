import os
import unittest
import numpy as np
import torch
import heat as ht

from ...core.tests.test_suites.basic_test import TestCase


class TestDMD(TestCase):
    def test_dmd_setup_and_catch_wrong(self):
        # catch wrong inputs
        with self.assertRaises(TypeError):
            ht.decomposition.DMD(svd_solver=0)
        with self.assertRaises(ValueError):
            ht.decomposition.DMD(svd_solver="Gramian")
        with self.assertRaises(ValueError):
            ht.decomposition.DMD(svd_solver="full", svd_rank=3, svd_tol=1e-1)
        with self.assertRaises(ValueError):
            ht.decomposition.DMD(svd_solver="full", svd_tol=-0.031415926)
        with self.assertRaises(ValueError):
            ht.decomposition.DMD(svd_solver="hierarchical")
        with self.assertRaises(ValueError):
            ht.decomposition.DMD(svd_solver="hierarchical", svd_rank=3, svd_tol=1e-1)
        with self.assertRaises(ValueError):
            ht.decomposition.DMD(svd_solver="randomized")
        with self.assertRaises(ValueError):
            ht.decomposition.DMD(svd_solver="randomized", svd_rank=2, svd_tol=1e-1)
        with self.assertRaises(TypeError):
            ht.decomposition.DMD(svd_solver="full", svd_rank=0.1)
        with self.assertRaises(ValueError):
            ht.decomposition.DMD(svd_solver="hierarchical", svd_rank=0)
        with self.assertRaises(TypeError):
            ht.decomposition.DMD(svd_solver="hierarchical", svd_tol="auto")
        with self.assertRaises(ValueError):
            ht.decomposition.DMD(svd_solver="randomized", svd_rank=0)

        dmd = ht.decomposition.DMD(svd_solver="full")
        with self.assertRaises(ValueError):
            dmd.fit(ht.zeros((5 * ht.MPI_WORLD.size, 2, 2), split=0))
        with self.assertRaises(ValueError):
            dmd.fit(ht.zeros((5 * ht.MPI_WORLD.size, 1), split=0))
        with self.assertRaises(RuntimeError):
            dmd.predict_next(ht.zeros(10))
        with self.assertRaises(RuntimeError):
            dmd.predict(ht.zeros(10), 10)

    def test_dmd_functionality_split0(self):
        # check whether the everything works with split=0, various checks are scattered over the different cases
        X = ht.random.randn(10 * ht.MPI_WORLD.size, 10, split=0)
        dmd = ht.decomposition.DMD(svd_solver="full")
        dmd.fit(X)
        self.assertTrue(dmd.rom_eigenmodes_.dtype == ht.complex64)
        self.assertEqual(dmd.rom_eigenmodes_.shape, (dmd.n_modes_, dmd.n_modes_))
        dmd = ht.decomposition.DMD(svd_solver="full", svd_tol=1e-1)
        dmd.fit(X)
        self.assertTrue(dmd.rom_basis_.shape[0] == 10 * ht.MPI_WORLD.size)
        dmd = ht.decomposition.DMD(svd_solver="full", svd_rank=3)
        dmd.fit(X)
        self.assertTrue(dmd.rom_basis_.shape[1] == 3)
        self.assertTrue(dmd.dmdmodes_.shape == (10 * ht.MPI_WORLD.size, 3))
        dmd = ht.decomposition.DMD(svd_solver="hierarchical", svd_rank=3)
        dmd.fit(X)
        self.assertTrue(dmd.rom_eigenvalues_.shape == (3,))
        dmd = ht.decomposition.DMD(svd_solver="hierarchical", svd_tol=1e-1)
        dmd.fit(X)
        Y = ht.random.randn(10 * ht.MPI_WORLD.size, split=0)
        Z = dmd.predict_next(Y)
        self.assertTrue(Z.shape == (10 * ht.MPI_WORLD.size,))
        self.assertTrue(dmd.rom_eigenvalues_.dtype == ht.complex64)
        self.assertTrue(dmd.dmdmodes_.dtype == ht.complex64)

        X = ht.random.randn(1000, 10 * ht.MPI_WORLD.size, split=0, dtype=ht.float32)
        dmd = ht.decomposition.DMD(svd_solver="randomized", svd_rank=4)
        dmd.fit(X)
        Y = ht.random.rand(1000, 2 * ht.MPI_WORLD.size, split=1, dtype=ht.float32)
        Z = dmd.predict_next(Y, 2)
        self.assertTrue(Z.dtype == ht.float32)
        self.assertEqual(Z.shape, Y.shape)

        # wrong shape of input for prediction
        with self.assertRaises(ValueError):
            dmd.predict_next(ht.zeros((100, 4), split=0))
        with self.assertRaises(ValueError):
            dmd.predict(ht.zeros((100, 4), split=0), 10)
        # wrong input for steps in predict
        with self.assertRaises(TypeError):
            dmd.predict(
                ht.zeros((1000, 5), split=0),
                "this is clearly neither an integer nor a list of integers",
            )

    def test_dmd_functionality_split1(self):
        # check whether everything works with split=1, various checks are scattered over the different cases
        X = ht.random.randn(10, 10 * ht.MPI_WORLD.size, split=1, dtype=ht.float64)
        dmd = ht.decomposition.DMD(svd_solver="full")
        dmd.fit(X)
        self.assertTrue(dmd.dmdmodes_.shape[0] == 10)
        dmd = ht.decomposition.DMD(svd_solver="full", svd_tol=1e-1)
        dmd.fit(X)
        dmd = ht.decomposition.DMD(svd_solver="full", svd_rank=3)
        dmd.fit(X)
        self.assertTrue(dmd.dmdmodes_.shape[1] == 3)
        dmd = ht.decomposition.DMD(svd_solver="hierarchical", svd_rank=3)
        dmd.fit(X)
        self.assertTrue(dmd.rom_transfer_matrix_.shape == (3, 3))
        self.assertTrue(dmd.rom_transfer_matrix_.dtype == ht.float64)
        dmd = ht.decomposition.DMD(svd_solver="hierarchical", svd_tol=1e-1)
        dmd.fit(X)
        self.assertTrue(dmd.rom_eigenvalues_.dtype == ht.complex128)
        Y = ht.random.randn(10, 2 * ht.MPI_WORLD.size, split=1)
        Z = dmd.predict_next(Y)
        self.assertTrue(Z.shape == Y.shape)

        X = ht.random.randn(1000, 10 * ht.MPI_WORLD.size, split=0)
        dmd = ht.decomposition.DMD(svd_solver="randomized", svd_rank=4)
        dmd.fit(X)
        self.assertTrue(dmd.rom_eigenmodes_.shape == (4, 4))
        self.assertTrue(dmd.n_modes_ == 4)
        Y = ht.random.randn(1000, 2, split=0, dtype=ht.float64)
        Z = dmd.predict_next(Y)
        self.assertTrue(Z.dtype == Y.dtype)
        self.assertEqual(Z.shape, Y.shape)

    def test_dmd_correctness(self):
        ht.random.seed(25032025)
        # test correctness on behalf of a constructed example with known solution
        # to do so we need to use the exact SVD, i.e., the "full" solver

        # ----------------- first case: split = 0 -----------------
        # dtype if float32, random transfer matrix
        r = 6
        A_red = ht.array(
            [
                [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.5, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
            ],
            split=None,
            dtype=ht.float32,
        )
        x0_red = ht.random.randn(r, 1, split=None)
        m, n = 25 * ht.MPI_WORLD.size, 15
        X = ht.hstack(
            [
                (ht.array(torch.linalg.matrix_power(A_red.larray, i) @ x0_red.larray))
                for i in range(n + 1)
            ]
        )
        U = ht.random.randn(m, r, split=0)
        U, _ = ht.linalg.qr(U)
        X = U @ X

        dmd = ht.decomposition.DMD(svd_solver="full", svd_rank=r)
        dmd.fit(X)

        # check whether the DMD-modes are correct
        sorted_ev_1 = np.sort_complex(dmd.rom_eigenvalues_.numpy())
        sorted_ev_2 = np.sort_complex(np.linalg.eigvals(A_red.numpy()))
        self.assertTrue(np.allclose(sorted_ev_1, sorted_ev_2, atol=1e-3, rtol=1e-3))

        # check prediction of next states
        Y = dmd.predict_next(X)
        self.assertTrue(ht.allclose(Y[:, :n], X[:, 1:], atol=1e-3, rtol=1e-3))

        # check prediction of previous states
        Y = dmd.predict_next(X, -1)
        self.assertTrue(ht.allclose(Y[:, 1:], X[:, :n], atol=1e-3, rtol=1e-3))

        # check catching wrong n_steps argument
        with self.assertRaises(TypeError):
            dmd.predict_next(X, "this is clearly not an integer")

        # ----------------- second case: split = 1 -----------------
        # dtype is float64, transfer matrix with nontrivial kernel
        r = 3
        A_red = ht.array(
            [[0.0, 0.0, 1.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], split=None, dtype=ht.float64
        )
        x0_red = ht.random.randn(r, 1, split=None, dtype=ht.float64)
        m, n = 10, 15 * ht.MPI_WORLD.size + 2
        X = ht.hstack(
            [
                (ht.array(torch.linalg.matrix_power(A_red.larray, i) @ x0_red.larray))
                for i in range(n + 1)
            ]
        )
        U = ht.random.randn(m, r, split=None, dtype=ht.float64)
        U, _ = ht.linalg.qr(U)
        X = U @ X
        X = X.resplit_(1)

        dmd = ht.decomposition.DMD(svd_solver="hierarchical", svd_rank=r)
        dmd.fit(X)

        # check whether the DMD-modes are correct
        sorted_ev_1 = np.sort_complex(dmd.rom_eigenvalues_.numpy())
        sorted_ev_2 = np.sort_complex(np.linalg.eigvals(A_red.numpy()))
        self.assertTrue(np.allclose(sorted_ev_1, sorted_ev_2, atol=1e-12, rtol=1e-12))

        # check prediction of third-next step
        Y = dmd.predict_next(X, 3)
        self.assertTrue(ht.allclose(Y[:, : n - 2], X[:, 3:], atol=1e-12, rtol=1e-12))
        # note: checking previous steps doesn't make sense here, as kernel of A_red is nontrivial

        # check batch prediction (split = 1)
        X_batch = X[:, : 5 * ht.MPI_WORLD.size]
        X_batch.balance_()
        Y = dmd.predict(X_batch, 5)
        Y_np = Y.numpy()
        X_np = X.numpy()
        for i in range(5):
            self.assertTrue(np.allclose(Y_np[i, :, :5], X_np[:, i : i + 5], atol=1e-12, rtol=1e-12))

        # check batch prediction (split = None)
        X_batch = ht.random.rand(10, 2 * ht.MPI_WORLD.size, split=None)
        Y = dmd.predict(X_batch, [-1, 1, 3])
