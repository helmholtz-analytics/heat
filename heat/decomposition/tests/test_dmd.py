import os
import unittest
import platform
import numpy as np
import torch
import heat as ht

from ...core.tests.test_suites.basic_test import TestCase

# MPS does not support non-float matrix multiplication
envar = os.getenv("HEAT_TEST_USE_DEVICE", "cpu")
is_mps = envar == "gpu" and platform.system() == "Darwin"


@unittest.skipIf(is_mps, "MPS does not support non-float matrix multiplication")
class TestDMD(TestCase):
    def test_dmd_setup_catch_wrong(self):
        # catch wrong inputs during setup
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

    def test_dmd_fit_catch_wrong(self):
        dmd = ht.decomposition.DMD(svd_solver="full")
        with self.assertRaises(ValueError):
            dmd.fit(ht.zeros((5 * ht.MPI_WORLD.size, 2, 2), split=0))
        with self.assertRaises(ValueError):
            dmd.fit(ht.zeros((5 * ht.MPI_WORLD.size, 1), split=0))

    def test_dmd_predict_catch_wrong(self):
        # not yet fitted
        dmd = ht.decomposition.DMD(svd_solver="full")
        with self.assertRaises(RuntimeError):
            dmd.predict_next(ht.zeros(10))
        with self.assertRaises(RuntimeError):
            dmd.predict(ht.zeros(10), 10)

        X = ht.random.randn(1000, 10 * ht.MPI_WORLD.size, split=0, dtype=ht.float32)
        dmd = ht.decomposition.DMD(svd_solver="randomized", svd_rank=4)
        dmd.fit(X)
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
        # check catching wrong n_steps argument
        with self.assertRaises(TypeError):
            dmd.predict_next(X, "this is clearly not an integer")
        # what not has been implemented so far
        with self.assertRaises(NotImplementedError):
            dmd.predict(ht.zeros((1000, 5), split=0), 10)

    def test_dmd_functionality_split0_full(self):
        # split=0, full SVD
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

    def test_dmd_functionality_split0_hierarchical(self):
        # split=0, hierarchical SVD
        X = ht.random.randn(10 * ht.MPI_WORLD.size, 10, split=0)
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

    def test_dmd_functionality_split0_randomized(self):
        # split=0, randomized SVD
        X = ht.random.randn(1000, 10 * ht.MPI_WORLD.size, split=0, dtype=ht.float32)
        dmd = ht.decomposition.DMD(svd_solver="randomized", svd_rank=4)
        dmd.fit(X)
        Y = ht.random.rand(1000, 2 * ht.MPI_WORLD.size, split=1, dtype=ht.float32)
        Z = dmd.predict_next(Y, 2)
        self.assertTrue(Z.dtype == ht.float32)
        self.assertEqual(Z.shape, Y.shape)
        Y = ht.random.rand(1000, split=0, dtype=ht.float32)
        Z = dmd.predict_next(Y, 2)
        self.assertTrue(Z.dtype == ht.float32)
        self.assertEqual(Z.shape, Y.shape)

    def test_dmd_functionality_split1_full(self):
        # split=1, full SVD
        X = ht.random.randn(10, 10 * ht.MPI_WORLD.size, split=1, dtype=ht.float64)
        dmd = ht.decomposition.DMD(svd_solver="full")
        print(dmd)
        dmd.fit(X)
        print(dmd)
        self.assertTrue(dmd.dmdmodes_.shape[0] == 10)
        dmd = ht.decomposition.DMD(svd_solver="full", svd_tol=1e-1)
        dmd.fit(X)
        dmd = ht.decomposition.DMD(svd_solver="full", svd_rank=3)
        dmd.fit(X)
        self.assertTrue(dmd.dmdmodes_.shape[1] == 3)

    def test_dmd_functionality_split1_hierarchical(self):
        # split=1, hierarchical SVD
        X = ht.random.randn(10, 10 * ht.MPI_WORLD.size, split=1, dtype=ht.float64)
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

    def test_dmd_functionality_split1_randomized(self):
        # split=1, randomized SVD
        X = ht.random.randn(1000, 10 * ht.MPI_WORLD.size, split=0)
        dmd = ht.decomposition.DMD(svd_solver="randomized", svd_rank=4)
        dmd.fit(X)
        self.assertTrue(dmd.rom_eigenmodes_.shape == (4, 4))
        self.assertTrue(dmd.n_modes_ == 4)
        Y = ht.random.randn(1000, 2, split=0, dtype=ht.float64)
        Z = dmd.predict_next(Y)
        self.assertTrue(Z.dtype == Y.dtype)
        self.assertEqual(Z.shape, Y.shape)

    def test_dmd_correctness_split0(self):
        ht.random.seed(25032025)
        # test correctness on behalf of a constructed example with known solution
        # to do so we need to use the exact SVD, i.e., the "full" solver
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

    def test_dmd_correctness_split1(self):
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


class TestDMDc(TestCase):
    def test_dmdc_setup_catch_wrong(self):
        # catch wrong inputs
        with self.assertRaises(TypeError):
            ht.decomposition.DMDc(svd_solver=0)
        with self.assertRaises(ValueError):
            ht.decomposition.DMDc(svd_solver="Gramian")
        with self.assertRaises(ValueError):
            ht.decomposition.DMDc(svd_solver="full", svd_rank=3, svd_tol=1e-1)
        with self.assertRaises(ValueError):
            ht.decomposition.DMDc(svd_solver="full", svd_tol=-0.031415926)
        with self.assertRaises(ValueError):
            ht.decomposition.DMDc(svd_solver="hierarchical")
        with self.assertRaises(ValueError):
            ht.decomposition.DMDc(svd_solver="hierarchical", svd_rank=3, svd_tol=1e-1)
        with self.assertRaises(ValueError):
            ht.decomposition.DMDc(svd_solver="randomized")
        with self.assertRaises(ValueError):
            ht.decomposition.DMDc(svd_solver="randomized", svd_rank=2, svd_tol=1e-1)
        with self.assertRaises(TypeError):
            ht.decomposition.DMDc(svd_solver="full", svd_rank=0.1)
        with self.assertRaises(ValueError):
            ht.decomposition.DMDc(svd_solver="hierarchical", svd_rank=0)
        with self.assertRaises(TypeError):
            ht.decomposition.DMDc(svd_solver="hierarchical", svd_tol="auto")
        with self.assertRaises(ValueError):
            ht.decomposition.DMDc(svd_solver="randomized", svd_rank=0)

    def test_dmdc_fit_catch_wrong(self):
        dmd = ht.decomposition.DMDc(svd_solver="full")
        # wrong dimensions of input
        with self.assertRaises(ValueError):
            dmd.fit(ht.zeros((5 * ht.MPI_WORLD.size, 2, 2), split=0), ht.zeros((2, 4), split=0))
        with self.assertRaises(ValueError):
            dmd.fit(ht.zeros((2, 4), split=0), ht.zeros((5 * ht.MPI_WORLD.size, 2, 2), split=0))
        # less than two timesteps
        with self.assertRaises(ValueError):
            dmd.fit(ht.zeros((5 * ht.MPI_WORLD.size, 1), split=0), ht.zeros((2, 4), split=0))
        with self.assertRaises(ValueError):
            dmd.fit(ht.zeros((2, 4), split=0), ht.zeros((5 * ht.MPI_WORLD.size, 1), split=0))
        # inconsistent number of timesteps
        with self.assertRaises(ValueError):
            dmd.fit(ht.zeros((5 * ht.MPI_WORLD.size, 3), split=0), ht.zeros((2, 4), split=0))
        # predict for fit
        with self.assertRaises(RuntimeError):
            dmd.predict(ht.zeros((5 * ht.MPI_WORLD.size, 3), split=0), ht.zeros((2, 4), split=0))
        # split mismatch for X and C
        X = ht.random.randn(1000, 10 * ht.MPI_WORLD.size, split=0, dtype=ht.float32)
        dmd = ht.decomposition.DMDc(svd_solver="randomized", svd_rank=4)
        # split mismatch for X and C
        C = ht.random.randn(10, 10 * ht.MPI_WORLD.size, split=1)
        with self.assertRaises(ValueError):
            dmd.fit(X, C)

    def test_dmdc_predict_catch_wrong(self):
        X = ht.random.randn(1000, 10 * ht.MPI_WORLD.size, split=0, dtype=ht.float32)
        dmd = ht.decomposition.DMDc(svd_solver="randomized", svd_rank=4)
        C = ht.random.randn(10, 10 * ht.MPI_WORLD.size, split=None)
        dmd.fit(X, C)
        Y = ht.random.randn(3, 10 * ht.MPI_WORLD.size, split=1)
        # wrong dimensions of input for prediction
        with self.assertRaises(ValueError):
            dmd.predict(Y, ht.zeros((5, 5, 5), split=0))
        with self.assertRaises(ValueError):
            dmd.predict(ht.zeros((5, 5, 5), split=0), C)
        # wrong sizes for inputs in predict
        with self.assertRaises(ValueError):
            dmd.predict(Y, ht.zeros((10, 5), split=0))
        with self.assertRaises(ValueError):
            dmd.predict(ht.zeros((1000, 5), split=0), C)
        # wrong split for C
        with self.assertRaises(ValueError):
            dmd.predict(Y, ht.zeros((10, 5), split=1))
        # wrong shape for C
        with self.assertRaises(ValueError):
            dmd.predict(Y, ht.zeros((5, 5), split=None))

    def test_dmdc_functionality_split0_full(self):
        # split=0, full SVD
        X = ht.random.randn(10 * ht.MPI_WORLD.size, 10, split=0)
        C = ht.random.randn(10, 10, split=0)
        dmd = ht.decomposition.DMDc(svd_solver="full")
        print(dmd)
        dmd.fit(X, C)
        print(dmd)
        self.assertTrue(dmd.rom_eigenmodes_.dtype == ht.complex64)
        self.assertEqual(dmd.rom_eigenmodes_.shape, (dmd.n_modes_, dmd.n_modes_))
        dmd = ht.decomposition.DMDc(svd_solver="full", svd_tol=1e-1)
        dmd.fit(X, C)
        self.assertTrue(dmd.rom_basis_.shape[0] == 10 * ht.MPI_WORLD.size)
        dmd = ht.decomposition.DMDc(svd_solver="full", svd_rank=3)
        dmd.fit(X, C)
        self.assertTrue(dmd.rom_basis_.shape[1] == 3)
        self.assertTrue(dmd.dmdmodes_.shape == (10 * ht.MPI_WORLD.size, 3))

    def test_dmdc_functionality_split0_hierarchical(self):
        # split=0, hierarchical SVD
        X = ht.random.randn(10 * ht.MPI_WORLD.size, 10, split=0)
        C = ht.random.randn(10, 10, split=0)
        dmd = ht.decomposition.DMDc(svd_solver="hierarchical", svd_rank=3)
        dmd.fit(X, C)
        self.assertTrue(dmd.rom_eigenvalues_.shape == (3,))
        dmd = ht.decomposition.DMDc(svd_solver="hierarchical", svd_tol=1e-1)
        dmd.fit(X, C)
        Y = ht.random.randn(3, 10 * ht.MPI_WORLD.size, split=1)
        C = ht.random.randn(10, 5, split=None)
        Z = dmd.predict(Y, C)
        self.assertTrue(Z.shape == (3, 10 * ht.MPI_WORLD.size, 5))
        self.assertTrue(dmd.rom_eigenvalues_.dtype == ht.complex64)
        self.assertTrue(dmd.dmdmodes_.dtype == ht.complex64)

    def test_dmdc_functionality_split0_randomized(self):
        # split=0, randomized SVD
        X = ht.random.randn(1000, 10 * ht.MPI_WORLD.size, split=0, dtype=ht.float32)
        dmd = ht.decomposition.DMDc(svd_solver="randomized", svd_rank=4)
        C = ht.random.randn(10, 10 * ht.MPI_WORLD.size, split=None)
        dmd.fit(X, C)
        Y = ht.random.rand(2 * ht.MPI_WORLD.size, 1000, split=0, dtype=ht.float32)
        C = ht.random.rand(10, 5, split=None)
        Z = dmd.predict(Y, C)
        self.assertTrue(Z.dtype == ht.float32)
        self.assertEqual(Z.shape, (2 * ht.MPI_WORLD.size, 1000, 5))

    def test_dmdc_functionality_split1_full(self):
        # split=1, full SVD
        X = ht.random.randn(10, 15 * ht.MPI_WORLD.size, split=1, dtype=ht.float64)
        C = ht.random.randn(2, 15 * ht.MPI_WORLD.size, split=1, dtype=ht.float64)
        dmd = ht.decomposition.DMDc(svd_solver="full")
        dmd.fit(X, C)
        self.assertTrue(dmd.dmdmodes_.shape[0] == 10)
        dmd = ht.decomposition.DMDc(svd_solver="full", svd_tol=1e-1)
        dmd.fit(X, C)
        dmd = ht.decomposition.DMDc(svd_solver="full", svd_rank=3)
        dmd.fit(X, C)
        self.assertTrue(dmd.dmdmodes_.shape[1] == 3)

    def test_dmdc_functionality_split1_hierarchical(self):
        # split=1, hierarchical SVD
        X = ht.random.randn(10, 15 * ht.MPI_WORLD.size, split=1, dtype=ht.float64)
        C = ht.random.randn(2, 15 * ht.MPI_WORLD.size, split=1, dtype=ht.float64)
        dmd = ht.decomposition.DMDc(svd_solver="hierarchical", svd_rank=3)
        dmd.fit(X, C)
        self.assertTrue(dmd.rom_transfer_matrix_.shape == (3, 3))
        self.assertTrue(dmd.rom_transfer_matrix_.dtype == ht.float64)
        dmd = ht.decomposition.DMDc(svd_solver="hierarchical", svd_tol=1e-1)
        dmd.fit(X, C)
        self.assertTrue(dmd.rom_eigenvalues_.dtype == ht.complex128)
        Y = ht.random.randn(10 * ht.MPI_WORLD.size, 10, split=0)
        C = ht.random.randn(2, split=None)
        Z = dmd.predict(Y, C)
        self.assertTrue(Z.shape == (10 * ht.MPI_WORLD.size, 10, 1))

    def test_dmdc_functionality_split1_randomized(self):
        # split=1, randomized SVD
        X = ht.random.randn(1000, 10 * ht.MPI_WORLD.size, split=0)
        C = ht.random.randn(10, 10 * ht.MPI_WORLD.size, split=None)
        dmd = ht.decomposition.DMDc(svd_solver="randomized", svd_rank=8)
        dmd.fit(X, C)
        self.assertTrue(dmd.rom_eigenmodes_.shape == (8, 8))
        self.assertTrue(dmd.n_modes_ == 8)
        Y = ht.random.randn(1000, split=0, dtype=ht.float64)
        Z = dmd.predict(Y, C)
        self.assertTrue(Z.dtype == Y.dtype)
        self.assertEqual(Z.shape, (1, 1000, 10 * ht.MPI_WORLD.size))

    def test_dmdc_correctness_split0(self):
        # check correctness on behalf of a constructed example with known solution,
        # thus only the "full" solver is used
        r = 3
        A_red = ht.array(
            [
                [0.0, 1, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.1],
            ],
            split=None,
            dtype=ht.float64,
        )
        B_red = ht.array(
            [
                [1.0, 0.0],
                [0.0, -1.0],
                [0.0, 1.0],
            ],
            split=None,
            dtype=ht.float64,
        )
        x0_red = ht.array(
            [
                [
                    10.0,
                ],
                [
                    5.0,
                ],
                [
                    -10.0,
                ],
            ],
            split=None,
            dtype=ht.float64,
        )
        m, n = 10 * ht.MPI_WORLD.size, 10
        C = 0.1 * ht.ones((2, n), split=None, dtype=ht.float64)
        X_red = [x0_red]
        for k in range(n - 1):
            X_red.append(A_red @ X_red[-1] + B_red @ C[:, k].reshape(-1, 1))
        X = ht.stack(X_red, axis=1).squeeze()
        U = ht.random.randn(m, r, split=0, dtype=ht.float64)
        U, _ = ht.linalg.qr(U)
        X = U @ X

        dmd = ht.decomposition.DMDc(svd_solver="full", svd_rank=3)
        dmd.fit(X, C)

        # check whether the DMD-modes are correct
        sorted_ev_1 = np.sort_complex(dmd.rom_eigenvalues_.numpy())
        sorted_ev_2 = np.sort_complex(np.linalg.eigvals(A_red.numpy()))
        self.assertTrue(np.allclose(sorted_ev_1, sorted_ev_2, atol=1e-12, rtol=1e-12))

        # check if DMD fits the data correctly
        X_red = dmd.rom_basis_.T @ X
        X_res = (
            X_red[:, 1:]
            - dmd.rom_transfer_matrix_ @ X_red[:, :-1]
            - dmd.rom_control_matrix_ @ C[:, :-1]
        )
        self.assertTrue(ht.max(ht.abs(X_res)) < 1e-10)

        # check predict
        Y = dmd.predict(X[:, 0], C[:, :10]).squeeze()

        # check prediction of next states
        Y_red = dmd.rom_basis_.T @ Y
        Y_res = (
            Y_red[:, 1:]
            - dmd.rom_transfer_matrix_ @ Y_red[:, :-1]
            - dmd.rom_control_matrix_ @ C[:, :-1]
        )
        self.assertTrue(ht.max(ht.abs(Y_res)) < 1e-10)
        self.assertTrue(ht.allclose(Y[:, :], X[:, :10], atol=1e-10, rtol=1e-10))

    def test_dmdc_correctness_split1(self):
        # check correctness on behalf of a constructed example with known solution,
        # thus only the "full" solver is used
        A_red = ht.array(
            [
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.05,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    -0.1,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.5,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.5,
                    0.0,
                ],
            ],
            split=None,
            dtype=ht.float32,
        )
        B_red = ht.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            split=None,
            dtype=ht.float32,
        )
        x0_red = ht.ones((5, 1), split=None, dtype=ht.float32)
        n = 20 * ht.MPI_WORLD.size
        C = 0.1 * ht.random.randn(2, n, split=None, dtype=ht.float32)
        X_red = [x0_red]
        for k in range(n - 1):
            X_red.append(A_red @ X_red[-1] + B_red @ C[:, k].reshape(-1, 1))
        X = ht.stack(X_red, axis=1).squeeze()
        X.resplit_(1)

        dmd = ht.decomposition.DMDc(svd_solver="full")
        dmd.fit(X, C)

        # check whether the DMD-modes are correct
        sorted_ev_1 = np.sort_complex(dmd.rom_eigenvalues_.numpy())
        sorted_ev_2 = np.sort_complex(np.linalg.eigvals(A_red.numpy()))
        self.assertTrue(np.allclose(sorted_ev_1, sorted_ev_2, atol=1e-4, rtol=1e-4))

        # check if DMD fits the data correctly
        X_red = dmd.rom_basis_.T @ X
        X_red.resplit_(None)
        X_res = (
            X_red[:, 1:]
            - dmd.rom_transfer_matrix_ @ X_red[:, :-1]
            - dmd.rom_control_matrix_ @ C[:, :-1]
        )
        self.assertTrue(ht.max(ht.abs(X_res)) < 1e-2)

        # # check predict
        Y = dmd.predict(X[:, 0], C).squeeze()

        # check prediction of next states
        Y_red = dmd.rom_basis_.T @ Y
        Y_res = (
            Y_red[:, 1:]
            - dmd.rom_transfer_matrix_ @ Y_red[:, :-1]
            - dmd.rom_control_matrix_ @ C[:, :-1]
        )
        self.assertTrue(ht.max(ht.abs(Y_res)) < 1e-2)
        self.assertTrue(ht.allclose(Y[:, :], X[:, :], atol=1e-2, rtol=1e-2))
