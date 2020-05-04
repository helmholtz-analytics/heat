import heat as ht
import numpy as np
import os
import torch
import unittest
import warnings

from ...tests.test_suites.basic_test import TestCase

if os.environ.get("EXTENDED_TESTS"):
    extended_tests = True
    warnings.warn("Extended Tests will take roughly 100x longer than the standard tests")
else:
    extended_tests = False


class TestQR(TestCase):
    @unittest.skipIf(not extended_tests, "extended tests")
    def test_qr_sp0_ext(self):
        st_whole = torch.randn(70, 70, device=self.device.torch_device)
        sp = 0
        for m in range(50, st_whole.shape[0] + 1, 1):
            for n in range(50, st_whole.shape[1] + 1, 1):
                for t in range(1, 3):
                    st = st_whole[:m, :n].clone()
                    a_comp = ht.array(st, split=0)
                    a = ht.array(st, split=sp)
                    qr = a.qr(tiles_per_proc=t)
                    self.assertTrue(ht.allclose(a_comp, qr.Q @ qr.R, rtol=1e-5, atol=1e-5))
                    self.assertTrue(ht.allclose(qr.Q.T @ qr.Q, ht.eye(m), rtol=1e-5, atol=1e-5))
                    self.assertTrue(ht.allclose(ht.eye(m), qr.Q @ qr.Q.T, rtol=1e-5, atol=1e-5))

    @unittest.skipIf(not extended_tests, "extended tests")
    def test_qr_sp1_ext(self):
        st_whole = torch.randn(70, 70, device=self.device.torch_device)
        sp = 1
        for m in range(50, st_whole.shape[0] + 1, 1):
            for n in range(50, st_whole.shape[1] + 1, 1):
                for t in range(1, 3):
                    st = st_whole[:m, :n].clone()
                    a_comp = ht.array(st, split=0)
                    a = ht.array(st, split=sp)
                    qr = a.qr(tiles_per_proc=t)
                    self.assertTrue(ht.allclose(a_comp, qr.Q @ qr.R, rtol=1e-5, atol=1e-5))
                    self.assertTrue(ht.allclose(qr.Q.T @ qr.Q, ht.eye(m), rtol=1e-5, atol=1e-5))
                    self.assertTrue(ht.allclose(ht.eye(m), qr.Q @ qr.Q.T, rtol=1e-5, atol=1e-5))

    def test_qr(self):
        m, n = 20, 40
        st = torch.randn(m, n, device=self.device.torch_device, dtype=torch.float)
        a_comp = ht.array(st, split=0)
        for t in range(1, 3):
            for sp in range(2):
                a = ht.array(st, split=sp, dtype=torch.float)
                qr = a.qr(tiles_per_proc=t)
                self.assertTrue(ht.allclose((a_comp - (qr.Q @ qr.R)), 0, rtol=1e-5, atol=1e-5))
                self.assertTrue(ht.allclose(qr.Q.T @ qr.Q, ht.eye(m), rtol=1e-5, atol=1e-5))
                self.assertTrue(ht.allclose(ht.eye(m), qr.Q @ qr.Q.T, rtol=1e-5, atol=1e-5))
        m, n = 40, 40
        st1 = torch.randn(m, n, device=self.device.torch_device)
        a_comp1 = ht.array(st1, split=0)
        for t in range(1, 3):
            for sp in range(2):
                a1 = ht.array(st1, split=sp)
                qr1 = a1.qr(tiles_per_proc=t)
                self.assertTrue(ht.allclose((a_comp1 - (qr1.Q @ qr1.R)), 0, rtol=1e-5, atol=1e-5))
                self.assertTrue(ht.allclose(qr1.Q.T @ qr1.Q, ht.eye(m), rtol=1e-5, atol=1e-5))
                self.assertTrue(ht.allclose(ht.eye(m), qr1.Q @ qr1.Q.T, rtol=1e-5, atol=1e-5))
        m, n = 40, 20
        st2 = torch.randn(m, n, dtype=torch.double, device=self.device.torch_device)
        a_comp2 = ht.array(st2, split=0, dtype=ht.double)
        for t in range(1, 3):
            for sp in range(2):
                a2 = ht.array(st2, split=sp)
                qr2 = a2.qr(tiles_per_proc=t)
                self.assertTrue(ht.allclose(a_comp2, qr2.Q @ qr2.R, rtol=1e-5, atol=1e-5))
                self.assertTrue(
                    ht.allclose(qr2.Q.T @ qr2.Q, ht.eye(m, dtype=ht.double), rtol=1e-5, atol=1e-5)
                )
                self.assertTrue(
                    ht.allclose(ht.eye(m, dtype=ht.double), qr2.Q @ qr2.Q.T, rtol=1e-5, atol=1e-5)
                )
                # test if calc R alone works
                qr = ht.qr(a2, calc_q=False, overwrite_a=True)
                self.assertTrue(qr.Q is None)

        m, n = 40, 20
        st = torch.randn(m, n, device=self.device.torch_device)
        a_comp = ht.array(st, split=None)
        a = ht.array(st, split=None)
        qr = a.qr()
        self.assertTrue(ht.allclose(a_comp, qr.Q @ qr.R, rtol=1e-5, atol=1e-5))
        self.assertTrue(ht.allclose(qr.Q.T @ qr.Q, ht.eye(m), rtol=1e-5, atol=1e-5))
        self.assertTrue(ht.allclose(ht.eye(m), qr.Q @ qr.Q.T, rtol=1e-5, atol=1e-5))

        # raises
        with self.assertRaises(TypeError):
            ht.qr(np.zeros((10, 10)))
        with self.assertRaises(TypeError):
            ht.qr(a_comp, tiles_per_proc="ls")
        with self.assertRaises(TypeError):
            ht.qr(a_comp, tiles_per_proc=1, calc_q=30)
        with self.assertRaises(TypeError):
            ht.qr(a_comp, tiles_per_proc=1, overwrite_a=30)
        with self.assertRaises(ValueError):
            ht.qr(a_comp, tiles_per_proc=torch.tensor([1, 2, 3]))
        with self.assertRaises(ValueError):
            ht.qr(ht.zeros((3, 4, 5)))
