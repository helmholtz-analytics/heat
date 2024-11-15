import heat as ht
import unittest
import torch
import numpy as np

from ...tests.test_suites.basic_test import TestCase


class TestZoloPD(TestCase):
    def catch_wrong_inputs(self):
        # if A is not a DNDarray
        with self.assertRaises(TypeError):
            ht.pd("I am clearly not a DNDarray. Do you mind?")
        # test wrong input dimension
        with self.assertRaises(ValueError):
            ht.pd(ht.zeros(10, 10, 10), dtype=ht.float32)
        # test wrong input shape
        with self.assertRaises(ValueError):
            ht.pd(ht.random.rand((10, 11)), dtype=ht.float32)
        # test wrong input dtype
        with self.assertRaises(TypeError):
            ht.pd(ht.ones((10, 10), dtype=ht.int32))
        # wrong input for r
        with self.assertRaises(TypeError):
            ht.pd(ht.ones((11, 10)), r=1.0)
        # wrong input for tol
        with self.assertRaises(TypeError):
            ht.pd(ht.ones((11, 10)), r=2, condition_estimate=1)

    def test_pd_split1(self):
        A = ht.random.randn(100, 100, split=1, dtype=ht.float32)
        ht.pd(A, silent=False)
