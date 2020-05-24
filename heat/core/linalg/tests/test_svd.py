import heat as ht
import numpy as np
import os
import torch
import unittest
import warnings

if os.environ.get("DEVICE") == "gpu" and torch.cuda.is_available():
    ht.use_device("gpu")
    torch.cuda.set_device(torch.device(ht.get_device().torch_device))
else:
    ht.use_device("cpu")
device = ht.get_device().torch_device
ht_device = None
if os.environ.get("DEVICE") == "lgpu" and torch.cuda.is_available():
    device = ht.gpu.torch_device
    ht_device = ht.gpu
    torch.cuda.set_device(device)

if os.environ.get("EXTENDED_TESTS"):
    extended_tests = True
    warnings.warn("Extended Tests will take roughly 100x longer than the standard tests")
else:
    extended_tests = False

sz = ht.MPI_WORLD.size


class TestSVD(unittest.TestCase):
    @unittest.skipIf(not extended_tests, "extended tests")
    def test_svd_sp0_ext(self):
        st_whole = torch.randn(sz * 8, sz * 8, device=device)
        sp = 0
        for m in range(sz * 6, st_whole.shape[0] + 1, 1):
            for n in range(sz * 6, st_whole.shape[1] + 1, 1):
                st = st_whole[:m, :n].clone()
                a_comp = ht.array(st, split=0, device=ht_device)
                m_comp = ht.eye(m, device=ht_device)
                n_comp = ht.eye(n, device=ht_device)
                a = ht.array(st, split=sp, device=ht_device)
                u, b, v = ht.block_diagonalize(a)
                self.assertTrue(ht.allclose(((u @ b @ v - a_comp) * 10000).round(), 0))
                self.assertTrue(ht.allclose(u.T @ u, m_comp, rtol=1e-5, atol=1e-5))
                self.assertTrue(ht.allclose(u @ u.T, m_comp, rtol=1e-5, atol=1e-5))
                self.assertTrue(ht.allclose(v.T @ v, n_comp, rtol=1e-5, atol=1e-5))
                self.assertTrue(ht.allclose(v @ v.T, n_comp, rtol=1e-5, atol=1e-5))

    @unittest.skipIf(not extended_tests, "extended tests")
    def test_svd_sp1_ext(self):
        st_whole = torch.randn(sz * 8, sz * 8, device=device)
        sp = 1
        for m in range(sz * 6, st_whole.shape[0] + 1, 1):
            for n in range(sz * 6, st_whole.shape[1] + 1, 1):
                st = st_whole[:m, :n].clone()
                a_comp = ht.array(st, split=0, device=ht_device)
                m_comp = ht.eye(m, device=ht_device)
                n_comp = ht.eye(n, device=ht_device)
                a = ht.array(st, split=sp, device=ht_device)
                u, b, v = ht.block_diagonalize(a)
                self.assertTrue(ht.allclose(((u @ b @ v - a_comp) * 10000).round(), 0))
                self.assertTrue(ht.allclose(u.T @ u, m_comp, rtol=1e-5, atol=1e-5))
                self.assertTrue(ht.allclose(u @ u.T, m_comp, rtol=1e-5, atol=1e-5))
                self.assertTrue(ht.allclose(v.T @ v, n_comp, rtol=1e-5, atol=1e-5))
                self.assertTrue(ht.allclose(v @ v.T, n_comp, rtol=1e-5, atol=1e-5))

    def test_svd(self):
        sz = ht.MPI_WORLD.size
        m, n = 5 * sz, 10 * sz
        st = torch.randn(m, n, device=device, dtype=torch.float)
        a_comp = ht.array(st, split=0, device=ht_device)
        m_comp = ht.eye(m, device=ht_device)
        n_comp = ht.eye(n, device=ht_device)
        for sp in range(2):
            a = ht.array(st, split=sp, device=ht_device)
            u, b, v = ht.block_diagonalize(a)
            self.assertTrue(ht.allclose(u @ b @ v, a_comp, rtol=1e-5, atol=1e-5))
            self.assertTrue(ht.allclose(u.T @ u, m_comp, rtol=1e-5, atol=1e-5))
            self.assertTrue(ht.allclose(u @ u.T, m_comp, rtol=1e-5, atol=1e-5))
            self.assertTrue(ht.allclose(v.T @ v, n_comp, rtol=1e-5, atol=1e-5))
            self.assertTrue(ht.allclose(v @ v.T, n_comp, rtol=1e-5, atol=1e-5))

        sz = ht.MPI_WORLD.size
        m, n = 10 * sz, 10 * sz
        st1 = torch.randn(m, n, device=device)
        a_comp1 = ht.array(st1, split=0, device=ht_device)
        m_comp = ht.eye(m, device=ht_device)
        n_comp = ht.eye(n, device=ht_device)
        for sp in range(2):
            a = ht.array(st1, split=sp, device=ht_device)
            u, b, v = ht.block_diagonalize(a)
            self.assertTrue(ht.allclose(u @ b @ v, a_comp1, rtol=1e-5, atol=1e-5))
            self.assertTrue(ht.allclose(u.T @ u, m_comp, rtol=1e-5, atol=1e-5))
            self.assertTrue(ht.allclose(u @ u.T, m_comp, rtol=1e-5, atol=1e-5))
            self.assertTrue(ht.allclose(v.T @ v, n_comp, rtol=1e-5, atol=1e-5))
            self.assertTrue(ht.allclose(v @ v.T, n_comp, rtol=1e-5, atol=1e-5))
        m, n = sz * 10, sz * 6
        st2 = torch.randn(m, n, dtype=torch.double, device=device)
        a_comp2 = ht.array(st2, split=0, dtype=ht.double, device=ht_device)
        m_comp = ht.eye(m, device=ht_device, dtype=ht.double)
        n_comp = ht.eye(n, device=ht_device, dtype=ht.double)
        for sp in range(2):
            a = ht.array(st2, split=sp, device=ht_device)
            u, b, v = ht.block_diagonalize(a)
            self.assertTrue(ht.allclose(u @ b @ v, a_comp2, rtol=1e-5, atol=1e-5))
            self.assertTrue(ht.allclose(u.T @ u, m_comp, rtol=1e-5, atol=1e-5))
            self.assertTrue(ht.allclose(u @ u.T, m_comp, rtol=1e-5, atol=1e-5))
            self.assertTrue(ht.allclose(v.T @ v, n_comp, rtol=1e-5, atol=1e-5))
            self.assertTrue(ht.allclose(v @ v.T, n_comp, rtol=1e-5, atol=1e-5))

        # raises
        with self.assertRaises(TypeError):
            ht.block_diagonalize(np.zeros((10, 10)))
        with self.assertRaises(RuntimeError):
            ht.block_diagonalize(ht.zeros((3, 4, 5)))
