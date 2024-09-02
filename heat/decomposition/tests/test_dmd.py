import os
import unittest
import numpy as np
import torch
import heat as ht

from ...core.tests.test_suites.basic_test import TestCase


class TestDMD(TestCase):
    def test_dmd(self):
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
        X = ht.random.randn(1000, 10 * ht.MPI_WORLD.size, split=0)
        dmd = ht.decomposition.DMD(svd_solver="randomized", svd_rank=4)
        dmd.fit(X)
