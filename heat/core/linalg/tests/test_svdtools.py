import torch
import os
import unittest
import heat as ht
import numpy as np
from mpi4py import MPI

from ...tests.test_suites.basic_test import TestCase


class TestHSVD(TestCase):
    def test_hsvd_rank_part1(self):
        # not testing on MPS for now as torch.norm() is unstable
        if not self.is_mps:
            nprocs = MPI.COMM_WORLD.Get_size()
            test_matrices = [
                ht.random.randn(50, 15 * nprocs, dtype=ht.float32, split=1),
                ht.random.randn(50, 15 * nprocs, dtype=ht.float64, split=1),
                ht.random.randn(15 * nprocs, 50, dtype=ht.float32, split=0),
                ht.random.randn(15 * nprocs, 50, dtype=ht.float64, split=0),
                ht.random.randn(15 * nprocs, 50, dtype=ht.float32, split=None),
                ht.random.randn(50, 15 * nprocs, dtype=ht.float64, split=None),
                ht.zeros((50, 15 * nprocs), dtype=ht.float32, split=1),
            ]
            rtols = [1e-1, 1e-2, 1e-3]
            ranks = [5, 10, 15]

            # check if hsvd yields "reasonable" results for random matrices, i.e.
            #    U (resp. V) is orthogonal for split=1 (resp. split=0)
            #    hsvd_rank yields the correct rank
            #    the true reconstruction error is <= error estimate
            #    for hsvd_rtol: true reconstruction error <= rtol (provided no further options)

            for i, A in enumerate(test_matrices):
                print("Testing hsvd for matrix {} of {}".format(i + 1, len(test_matrices)))
                if A.dtype == ht.float64:
                    dtype_tol = 1e-8
                if A.dtype == ht.float32:
                    dtype_tol = 1e-3

                for r in ranks:
                    U, sigma, Vt, err_est = ht.linalg.hsvd_rank(A, r, compute_sv=True, silent=True)
                    V = Vt.T
                    hsvd_rk = U.shape[1]

                    if ht.norm(A) > 0:
                        self.assertEqual(hsvd_rk, r)
                        if A.split == 1:
                            U_orth_err = (
                                ht.norm(
                                    U.T @ U
                                    - ht.eye(
                                        hsvd_rk, dtype=U.dtype, split=U.T.split, device=U.device
                                    )
                                )
                                / hsvd_rk**0.5
                            )
                            self.assertTrue(U_orth_err <= dtype_tol)
                        if A.split == 0:
                            V_orth_err = (
                                ht.norm(
                                    V.T @ V
                                    - ht.eye(
                                        hsvd_rk, dtype=V.dtype, split=V.T.split, device=V.device
                                    )
                                )
                                / hsvd_rk**0.5
                            )
                            self.assertTrue(V_orth_err <= dtype_tol)
                        true_rel_err = ht.norm(U @ ht.diag(sigma) @ V.T - A) / ht.norm(A)
                        self.assertTrue(true_rel_err <= err_est or true_rel_err < dtype_tol)
                    else:
                        self.assertEqual(hsvd_rk, 1)
                        self.assertEqual(ht.norm(U), 0)
                        self.assertEqual(ht.norm(sigma), 0)
                        self.assertEqual(ht.norm(V), 0)

                    # check if wrong parameter choice is caught
                    with self.assertRaises(RuntimeError):
                        ht.linalg.hsvd_rank(A, r, maxmergedim=4)

                for tol in rtols:
                    U, sigma, Vt, err_est = ht.linalg.hsvd_rtol(A, tol, compute_sv=True, silent=True)
                    V = Vt.T
                    hsvd_rk = U.shape[1]

                    if ht.norm(A) > 0:
                        if A.split == 1:
                            U_orth_err = (
                                ht.norm(
                                    U.T @ U
                                    - ht.eye(
                                        hsvd_rk, dtype=U.dtype, split=U.T.split, device=U.device
                                    )
                                )
                                / hsvd_rk**0.5
                            )
                            # print(U_orth_err)
                            self.assertTrue(U_orth_err <= dtype_tol)
                        if A.split == 0:
                            V_orth_err = (
                                ht.norm(
                                    V.T @ V
                                    - ht.eye(
                                        hsvd_rk, dtype=V.dtype, split=V.T.split, device=V.device
                                    )
                                )
                                / hsvd_rk**0.5
                            )
                            self.assertTrue(V_orth_err <= dtype_tol)
                        true_rel_err = ht.norm(U @ ht.diag(sigma) @ V.T - A) / ht.norm(A)
                        self.assertTrue(true_rel_err <= err_est or true_rel_err < dtype_tol)
                        self.assertTrue(true_rel_err <= tol)
                    else:
                        self.assertEqual(hsvd_rk, 1)
                        self.assertEqual(ht.norm(U), 0)
                        self.assertEqual(ht.norm(sigma), 0)
                        self.assertEqual(ht.norm(V), 0)

                    # check if wrong parameter choices are catched
                    with self.assertRaises(ValueError):
                        ht.linalg.hsvd_rtol(A, tol, maxmergedim=4)
                    with self.assertRaises(ValueError):
                        ht.linalg.hsvd_rtol(A, tol, maxmergedim=10, maxrank=11)
                    with self.assertRaises(ValueError):
                        ht.linalg.hsvd_rtol(A, tol, no_of_merges=1)

                # check if wrong input arrays are caught
                wrong_test_matrices = [
                    0,
                    ht.ones((50, 15 * nprocs), dtype=ht.int8, split=1),
                    ht.ones((50, 15 * nprocs), dtype=ht.int16, split=1),
                    ht.ones((50, 15 * nprocs), dtype=ht.int32, split=1),
                    ht.ones((50, 15 * nprocs), dtype=ht.int64, split=1),
                    ht.ones((50, 15 * nprocs), dtype=ht.complex64, split=1),
                    ht.ones((50, 15 * nprocs), dtype=ht.complex128, split=1),
                ]

                for A in wrong_test_matrices:
                    with self.assertRaises(TypeError):
                        ht.linalg.hsvd_rank(A, 5)
                    with self.assertRaises(TypeError):
                        ht.linalg.hsvd_rank(A, 1e-1)

                wrong_test_matrices = [
                    ht.ones((15, 15 * nprocs, 15), split=1, dtype=ht.float64),
                    ht.ones(15 * nprocs, split=0, dtype=ht.float64),
                ]
                for wrong_arr in wrong_test_matrices:
                    with self.assertRaises(ValueError):
                        ht.linalg.hsvd_rank(wrong_arr, 5)
                    with self.assertRaises(ValueError):
                        ht.linalg.hsvd_rtol(wrong_arr, 1e-1)

                # check if compute_sv=False yields the correct number of outputs (=1)
                self.assertEqual(len(ht.linalg.hsvd_rank(test_matrices[0], 5)), 2)
                self.assertEqual(len(ht.linalg.hsvd_rtol(test_matrices[0], 5e-1)), 2)

    def test_hsvd_rank_part2(self):
        # check if hsvd_rank yields correct results for maxrank <= truerank
        nprocs = MPI.COMM_WORLD.Get_size()
        true_rk = max(10, nprocs)
        if self.is_mps:
            test_matrices_low_rank = [
                ht.utils.data.matrixgallery.random_known_rank(
                    50, 15 * nprocs, true_rk, split=1, dtype=ht.float32
                ),
                ht.utils.data.matrixgallery.random_known_rank(
                    50, 15 * nprocs, true_rk, split=1, dtype=ht.float32
                ),
            ]
        else:
            test_matrices_low_rank = [
                ht.utils.data.matrixgallery.random_known_rank(
                    50, 15 * nprocs, true_rk, split=1, dtype=ht.float32
                ),
                ht.utils.data.matrixgallery.random_known_rank(
                    50, 15 * nprocs, true_rk, split=1, dtype=ht.float32
                ),
                ht.utils.data.matrixgallery.random_known_rank(
                    15 * nprocs, 50, true_rk, split=0, dtype=ht.float64
                ),
                ht.utils.data.matrixgallery.random_known_rank(
                    15 * nprocs, 50, true_rk, split=0, dtype=ht.float64
                ),
            ]

        for mat in test_matrices_low_rank:
            A = mat[0]
            if A.dtype == ht.float64:
                dtype_tol = 1e-8
            if A.dtype == ht.float32:
                dtype_tol = 1e-3

            for r in [true_rk, true_rk + 2]:
                U, s, Vt, _ = ht.linalg.hsvd_rank(A, r, compute_sv=True)
                V = Vt.T
                V = V[:, :true_rk].resplit(V.split)
                U = U[:, :true_rk].resplit(U.split)
                s = s[:true_rk]

                U_orth_err = (
                    ht.norm(
                        U.T @ U - ht.eye(true_rk, dtype=U.dtype, split=U.T.split, device=U.device)
                    )
                    / true_rk**0.5
                )
                V_orth_err = (
                    ht.norm(
                        V.T @ V - ht.eye(true_rk, dtype=V.dtype, split=V.T.split, device=V.device)
                    )
                    / true_rk**0.5
                )
                true_rel_err = ht.norm(U @ ht.diag(s) @ V.T - A) / ht.norm(A)

                self.assertTrue(ht.norm(s - mat[1][1]) / ht.norm(mat[1][1]) <= dtype_tol)
                self.assertTrue(U_orth_err <= dtype_tol)
                self.assertTrue(V_orth_err <= dtype_tol)
                self.assertTrue(true_rel_err <= dtype_tol)


class TestRSVD(TestCase):
    def test_rsvd(self):
        if self.is_mps:
            dtypes = [ht.float32]
        else:
            dtypes = [ht.float32, ht.float64]
        for dtype in dtypes:
            dtype_tol = 1e-4 if dtype == ht.float32 else 1e-10
            for split in [0, 1, None]:
                X = ht.random.randn(200, 200, dtype=dtype, split=split)
                for rank in [ht.MPI_WORLD.size, 10]:
                    for n_oversamples in [5, 10]:
                        for power_iter in [0, 1, 2, 3]:
                            U, S, Vt = ht.linalg.rsvd(
                                X, rank, n_oversamples=n_oversamples, power_iter=power_iter
                            )
                            V = Vt.T
                            self.assertEqual(U.shape, (X.shape[0], rank))
                            self.assertEqual(S.shape, (rank,))
                            self.assertEqual(V.shape, (X.shape[1], rank))
                            self.assertTrue(ht.all(S >= 0))
                            self.assertTrue(
                                ht.allclose(
                                    U.T @ U,
                                    ht.eye(rank, dtype=U.dtype, split=U.split),
                                    rtol=dtype_tol,
                                    atol=dtype_tol,
                                )
                            )
                            self.assertTrue(
                                ht.allclose(
                                    V.T @ V,
                                    ht.eye(rank, dtype=V.dtype, split=V.split),
                                    rtol=dtype_tol,
                                    atol=dtype_tol,
                                )
                            )

    def test_rsvd_catch_wrong_inputs(self):
        X = ht.random.randn(10, 10)
        # wrong dtype for rank
        with self.assertRaises(TypeError):
            ht.linalg.rsvd(X, "a")
        # rank zero
        with self.assertRaises(ValueError):
            ht.linalg.rsvd(X, 0)
        # wrong dtype for n_oversamples
        with self.assertRaises(TypeError):
            ht.linalg.rsvd(X, 10, n_oversamples="a")
        # n_oversamples negative
        with self.assertRaises(ValueError):
            ht.linalg.rsvd(X, 10, n_oversamples=-1)
        # wrong dtype for power_iter
        with self.assertRaises(TypeError):
            ht.linalg.rsvd(X, 10, power_iter="a")
        # power_iter negative
        with self.assertRaises(ValueError):
            ht.linalg.rsvd(X, 10, power_iter=-1)


class TestISVD(TestCase):
    def test_isvd(self):
        if self.is_mps:
            dtypes = [ht.float32]
        else:
            dtypes = [ht.float32, ht.float64]
        for dtype in dtypes:
            dtypetol = 1e-5 if dtype == ht.float32 else 1e-10
            for old_split in [0, 1, None]:
                X_old, SVD_old = ht.utils.data.matrixgallery.random_known_rank(
                    250, 25, 3 * ht.MPI_WORLD.size, split=old_split, dtype=dtype
                )
                U_old, S_old, V_old = SVD_old
                Vt_old = V_old.T
                for new_split in [0, 1, None]:
                    new_data = ht.random.randn(
                        250, 2 * ht.MPI_WORLD.size, split=new_split, dtype=dtype
                    )
                    U_new, S_new, Vt_new = ht.linalg.isvd(new_data, U_old, S_old, Vt_old)
                    V_new = Vt_new.T
                    # check if U_new, V_new are orthogonal
                    self.assertTrue(
                        ht.allclose(
                            U_new.T @ U_new,
                            ht.eye(U_new.shape[1], dtype=U_new.dtype, split=U_new.split),
                            atol=dtypetol,
                            rtol=dtypetol,
                        )
                    )
                    self.assertTrue(
                        ht.allclose(
                            V_new.T @ V_new,
                            ht.eye(V_new.shape[1], dtype=V_new.dtype, split=V_new.split),
                            atol=dtypetol,
                            rtol=dtypetol,
                        )
                    )
                    # check if entries of S_new are positive
                    self.assertTrue(ht.all(S_new >= 0))
                    # check if the reconstruction error is small
                    X_new = ht.hstack([X_old, new_data.resplit_(X_old.split)])
                    X_rec = U_new @ ht.diag(S_new) @ V_new.T
                    self.assertTrue(ht.allclose(X_rec, X_new, atol=dtypetol, rtol=dtypetol))

    def test_isvd_catch_wrong_inputs(self):
        u_old = ht.zeros((10, 2))
        s_old = ht.zeros((3,))
        v_old = ht.zeros((5, 3))
        new_data = ht.zeros((11, 5))
        with self.assertRaises(ValueError):
            ht.linalg.isvd(new_data, u_old, s_old, v_old)
        s_old = ht.zeros((2,))
        with self.assertRaises(ValueError):
            ht.linalg.isvd(new_data, u_old, s_old, v_old)
        v_old = ht.zeros((5, 2))
        with self.assertRaises(ValueError):
            ht.linalg.isvd(new_data, u_old, s_old, v_old)
