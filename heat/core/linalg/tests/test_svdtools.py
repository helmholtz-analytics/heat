import torch
import os
import unittest
import heat as ht
import numpy as np
from mpi4py import MPI

from ...tests.test_suites.basic_test import TestCase


class TestHSVD(TestCase):
    # the following skipIf doesn't work
    # @unittest.skipIf(ht.get_device().device_type.startswith("gpu") and torch.backends.mps.is_built() and torch.backends.mps.is_available(), "MPS unstable for now")
    # not testing on MPS for now as torch.norm() is unstable
    def test_hsvd_rank_part1(self):
        is_mps = (
            ht.get_device().device_type.startswith("gpu")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        )
        if not is_mps:
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
                    U, sigma, V, err_est = ht.linalg.hsvd_rank(A, r, compute_sv=True, silent=True)
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
                        self.assertTrue(true_rel_err <= err_est)
                    else:
                        self.assertEqual(hsvd_rk, 1)
                        self.assertEqual(ht.norm(U), 0)
                        self.assertEqual(ht.norm(sigma), 0)
                        self.assertEqual(ht.norm(V), 0)

                    # check if wrong parameter choice is caught
                    with self.assertRaises(RuntimeError):
                        ht.linalg.hsvd_rank(A, r, maxmergedim=4)

                for tol in rtols:
                    U, sigma, V, err_est = ht.linalg.hsvd_rtol(A, tol, compute_sv=True, silent=True)
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
                        self.assertTrue(true_rel_err <= err_est)
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

                # check if wrong input arrays are catched
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

    @unittest.skipIf(torch.cuda.is_available() and torch.version.hip, "not supported for HIP")
    def test_hsvd_rank_part2(self):
        is_mps = (
            ht.get_device().device_type.startswith("gpu")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        )
        if not is_mps:
            # check if hsvd_rank yields correct results for maxrank <= truerank
            # this needs to be skipped on AMD because generation of test data relies on QR...
            nprocs = MPI.COMM_WORLD.Get_size()
            true_rk = max(10, nprocs)
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
                    U, s, V, _ = ht.linalg.hsvd_rank(A, r, compute_sv=True)
                    V = V[:, :true_rk].resplit(V.split)
                    U = U[:, :true_rk].resplit(U.split)
                    s = s[:true_rk]

                    U_orth_err = (
                        ht.norm(
                            U.T @ U
                            - ht.eye(true_rk, dtype=U.dtype, split=U.T.split, device=U.device)
                        )
                        / true_rk**0.5
                    )
                    V_orth_err = (
                        ht.norm(
                            V.T @ V
                            - ht.eye(true_rk, dtype=V.dtype, split=V.T.split, device=V.device)
                        )
                        / true_rk**0.5
                    )
                    true_rel_err = ht.norm(U @ ht.diag(s) @ V.T - A) / ht.norm(A)

                    self.assertTrue(ht.norm(s - mat[1][1]) / ht.norm(mat[1][1]) <= dtype_tol)
                    self.assertTrue(U_orth_err <= dtype_tol)
                    self.assertTrue(V_orth_err <= dtype_tol)
                    self.assertTrue(true_rel_err <= dtype_tol)
