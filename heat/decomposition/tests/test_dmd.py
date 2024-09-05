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

    def test_dmd_functionality_split0(self):
        # check whether the everything works with split=0
        X = ht.random.randn(10 * ht.MPI_WORLD.size, 10, split=0)
        dmd = ht.decomposition.DMD(svd_solver="full")
        dmd.fit(X)
        dmd = ht.decomposition.DMD(svd_solver="full", svd_tol=1e-1)
        dmd.fit(X)
        dmd = ht.decomposition.DMD(svd_solver="full", svd_rank=3)
        dmd.fit(X)
        dmd = ht.decomposition.DMD(svd_solver="hierarchical", svd_rank=3)
        dmd.fit(X)
        dmd = ht.decomposition.DMD(svd_solver="hierarchical", svd_tol=1e-1)
        dmd.fit(X)
        Y = ht.random.randn(10 * ht.MPI_WORLD.size, split=0)
        dmd.predict_next(Y)

        X = ht.random.randn(1000, 10 * ht.MPI_WORLD.size, split=0)
        dmd = ht.decomposition.DMD(svd_solver="randomized", svd_rank=4)
        dmd.fit(X)
        Y = ht.random.rand(4, 1000, split=1)
        dmd.predict_next(Y)

    def test_dmd_functionality_split1(self):
        # check whether everything works with split=1
        X = ht.random.randn(10, 10 * ht.MPI_WORLD.size, split=1)
        dmd = ht.decomposition.DMD(svd_solver="full")
        dmd.fit(X)
        dmd = ht.decomposition.DMD(svd_solver="full", svd_tol=1e-1)
        dmd.fit(X)
        dmd = ht.decomposition.DMD(svd_solver="full", svd_rank=3)
        dmd.fit(X)
        dmd = ht.decomposition.DMD(svd_solver="hierarchical", svd_rank=3)
        dmd.fit(X)
        dmd = ht.decomposition.DMD(svd_solver="hierarchical", svd_tol=1e-1)
        dmd.fit(X)
        Y = ht.random.randn(2 * ht.MPI_WORLD.size, 10, split=0)
        dmd.predict_next(Y)

        X = ht.random.randn(1000, 10 * ht.MPI_WORLD.size, split=0)
        dmd = ht.decomposition.DMD(svd_solver="randomized", svd_rank=4)
        dmd.fit(X)
        Y = ht.random.randn(2, 1000, split=1)
        dmd.predict_next(Y)

    ###############################################
    # WORK IN PROGRESS
    ###############################################
    def test_dmd_correctness(self):
        r = 3
        A_red = ht.random.randn(r, r, split=None)
        A_red /= ht.linalg.norm(A_red)
        x0_red = ht.random.randn(r, 1, split=None)
        m, n = 10 * ht.MPI_WORLD.size, ht.MPI_WORLD.size
        X = ht.hstack(
            [
                (ht.array(torch.linalg.matrix_power(A_red.larray, i) @ x0_red.larray))
                for i in range(n)
            ]
        )
        U = ht.random.randn(m, r, split=0)
        U, _ = ht.linalg.qr(U)
        X = U @ X

        dmd = ht.decomposition.DMD(svd_solver="full", svd_tol=1 - 1e-6)
        dmd.fit(X)
        print(dmd.rom_eigenvalues_)
        print(torch.linalg.eigvals(A_red.larray))
