import heat as ht
import unittest
import torch
from heat.core.tests.test_suites.basic_test import TestCase


class TestMatrixgallery(TestCase):
    def __check_parter(self, parter):
        self.assertEqual(parter.shape, (20, 20))
        # TODO: check for singular values of the parter matrix

    def __check_orthogonality(self, U):
        U_orth_err = (
            ht.norm(U.T @ U - ht.eye(U.shape[1], dtype=U.dtype, split=U.T.split, device=U.device))
            / U.shape[1] ** 0.5
        )
        if U.dtype == ht.float64:
            dtype_tol = 1e-12
        if U.dtype == ht.float32:
            dtype_tol = 1e-6
        self.assertTrue(U_orth_err <= dtype_tol)

    def test_hermitian(self):
        with self.assertRaises(ValueError):
            ht.utils.data.matrixgallery.hermitian(10, 20)
        with self.assertRaises(ValueError):
            ht.utils.data.matrixgallery.hermitian(20, split=0, dtype=ht.int32)

        # test default: complex single precision, not positive definite
        A = ht.utils.data.matrixgallery.hermitian(20, split=1)
        A_err = ht.norm(A - A.T.conj().resplit_(A.split)) / ht.norm(A)
        self.assertTrue(A_err <= 1e-6)

        for posdef in [True, False]:
            # test complex double precision
            A = ht.utils.data.matrixgallery.hermitian(
                20, dtype=ht.complex128, split=0, positive_definite=posdef
            )
            A_err = ht.norm(A - A.T.conj().resplit_(A.split)) / ht.norm(A)
            self.assertTrue(A.dtype == ht.complex128)
            self.assertTrue(A_err <= 1e-12)

            # test real datatype
            A = ht.utils.data.matrixgallery.hermitian(
                20, dtype=ht.float32, split=0, positive_definite=posdef
            )
            A_err = ht.norm(A - A.T.conj().resplit_(A.split)) / ht.norm(A)
            self.assertTrue(A_err <= 1e-6)
            self.assertTrue(A.dtype == ht.float32)

    def test_parter(self):
        parter = ht.utils.data.matrixgallery.parter(20)
        self.__check_parter(parter)

        parters0 = ht.utils.data.matrixgallery.parter(20, split=0, comm=ht.MPI_WORLD)
        self.__check_parter(parters0)

        parters1 = ht.utils.data.matrixgallery.parter(20, split=1, comm=ht.MPI_WORLD)
        self.__check_parter(parters1)

        with self.assertRaises(ValueError):
            ht.utils.data.matrixgallery.parter(20, split=2, comm=ht.MPI_WORLD)

    def test_random_orthogonal(self):
        with self.assertRaises(RuntimeError):
            ht.utils.data.matrixgallery.random_orthogonal(10, 20)

        Q = ht.utils.data.matrixgallery.random_orthogonal(20, 15)
        # Q_orth_err = ht.norm(
        #                     Q.T @ Q
        #                     - ht.eye(Q.shape[1], dtype=Q.dtype, split=Q.T.split, device=Q.device)
        #                 )
        # self.assertTrue(Q_orth_err <= 1e-6)
        self.__check_orthogonality(Q)

    def test_random_known_singularvalues(self):
        with self.assertRaises(RuntimeError):
            ht.utils.data.matrixgallery.random_known_singularvalues(30, 20, "abc", split=1)
        with self.assertRaises(RuntimeError):
            ht.utils.data.matrixgallery.random_known_singularvalues(30, 20, ht.eye(20), split=1)
        with self.assertRaises(RuntimeError):
            ht.utils.data.matrixgallery.random_known_singularvalues(30, 20, ht.ones(50), split=1)

        svals_input = ht.ones(15)
        A, SVD = ht.utils.data.matrixgallery.random_known_singularvalues(
            30, 20, svals_input, split=1
        )
        U = SVD[0]
        S = SVD[1]
        V = SVD[2]
        if A.dtype == ht.float64:
            dtype_tol = 1e-12
        if A.dtype == ht.float32:
            dtype_tol = 1e-6
        self.__check_orthogonality(U)
        self.__check_orthogonality(V)
        self.assertTrue(ht.allclose(S, svals_input, rtol=dtype_tol))
        A_err = ht.norm(A - U @ ht.diag(S) @ V.T) / ht.norm(A)
        self.assertTrue(A_err <= dtype_tol)

    def test_random_known_rank(self):
        with self.assertRaises(RuntimeError):
            ht.utils.data.matrixgallery.random_known_rank(30, 20, 25, split=1)
        rkinput = 15
        A, SVD = ht.utils.data.matrixgallery.random_known_rank(30, 20, rkinput, split=1)
        U = SVD[0]
        S = SVD[1]
        V = SVD[2]
        if A.dtype == ht.float64:
            dtype_tol = 1e-12
        if A.dtype == ht.float32:
            dtype_tol = 1e-6
        self.__check_orthogonality(U)
        self.__check_orthogonality(V)
        self.assertTrue(S.shape[0] == rkinput)
        A_err = ht.norm(A - U @ ht.diag(S) @ V.T) / ht.norm(A)
        self.assertTrue(A_err <= dtype_tol)
