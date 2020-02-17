import heat as ht
import numpy as np
import os
import torch
import unittest

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

extended_tests = True if os.environ.get("EXTENDED_TESTS") else False


class TestQR(unittest.TestCase):
    def test_qr(self):
        if not extended_tests:
            m, n = 40, 40
            st = torch.randn(m, n, device=device)
            a_comp = ht.array(st, split=0, device=ht_device)
            for t in range(1, 3):
                for sp in range(2):
                    a = ht.array(st, split=sp, device=ht_device)
                    qr = a.qr(tiles_per_proc=t)
                    self.assertTrue(ht.allclose(a_comp, qr.Q @ qr.R, rtol=1e-5, atol=1e-5))
                    self.assertTrue(
                        ht.allclose(
                            qr.Q.T @ qr.Q, ht.eye(m, device=ht_device), rtol=1e-5, atol=1e-5
                        )
                    )
                    self.assertTrue(
                        ht.allclose(
                            ht.eye(m, device=ht_device), qr.Q @ qr.Q.T, rtol=1e-5, atol=1e-5
                        )
                    )
            m, n = 20, 40
            st = torch.randn(m, n, device=device)
            a_comp = ht.array(st, split=0, device=ht_device)
            for t in range(1, 3):
                for sp in range(2):
                    a = ht.array(st, split=sp, device=ht_device)
                    qr = a.qr(tiles_per_proc=t)
                    self.assertTrue(ht.allclose(a_comp, qr.Q @ qr.R, rtol=1e-5, atol=1e-5))
                    self.assertTrue(
                        ht.allclose(
                            qr.Q.T @ qr.Q, ht.eye(m, device=ht_device), rtol=1e-5, atol=1e-5
                        )
                    )
                    self.assertTrue(
                        ht.allclose(
                            ht.eye(m, device=ht_device), qr.Q @ qr.Q.T, rtol=1e-5, atol=1e-5
                        )
                    )
            m, n = 40, 20
            st = torch.randn(m, n, dtype=torch.double, device=device)
            a_comp = ht.array(st, split=0, dtype=ht.double, device=ht_device)
            for t in range(1, 3):
                for sp in range(2):
                    a = ht.array(st, split=sp, device=ht_device)
                    qr = a.qr(tiles_per_proc=t)
                    self.assertTrue(ht.allclose(a_comp, qr.Q @ qr.R, rtol=1e-5, atol=1e-5))
                    self.assertTrue(
                        ht.allclose(
                            qr.Q.T @ qr.Q,
                            ht.eye(m, dtype=ht.double, device=ht_device),
                            rtol=1e-5,
                            atol=1e-5,
                        )
                    )
                    self.assertTrue(
                        ht.allclose(
                            ht.eye(m, dtype=ht.double, device=ht_device),
                            qr.Q @ qr.Q.T,
                            rtol=1e-5,
                            atol=1e-5,
                        )
                    )
        else:
            st_whole = torch.randn(100, 100, device=device)
            for m in range(30, st_whole.shape[0] + 1, 1):
                for n in range(30, st_whole.shape[1] + 1, 1):
                    for t in range(1, 3):
                        for sp in range(0, 2):
                            st = st_whole[:m, :n].clone()
                            a_comp = ht.array(st, split=0, device=ht_device)
                            a = ht.array(st, split=sp, device=ht_device)
                            qr = a.qr(tiles_per_proc=t)
                            self.assertTrue(ht.allclose(a_comp, qr.Q @ qr.R, rtol=1e-5, atol=1e-5))
                            self.assertTrue(
                                ht.allclose(
                                    qr.Q.T @ qr.Q, ht.eye(m, device=ht_device), rtol=1e-5, atol=1e-5
                                )
                            )
                            self.assertTrue(
                                ht.allclose(
                                    ht.eye(m, device=ht_device), qr.Q @ qr.Q.T, rtol=1e-5, atol=1e-5
                                )
                            )

        m, n = 40, 20
        st = torch.randn(m, n, device=device)
        a_comp = ht.array(st, split=None, device=ht_device)
        a = ht.array(st, split=None, device=ht_device)
        qr = a.qr()
        self.assertTrue(ht.allclose(a_comp, qr.Q @ qr.R, rtol=1e-5, atol=1e-5))
        self.assertTrue(
            ht.allclose(qr.Q.T @ qr.Q, ht.eye(m, device=ht_device), rtol=1e-5, atol=1e-5)
        )
        self.assertTrue(
            ht.allclose(ht.eye(m, device=ht_device), qr.Q @ qr.Q.T, rtol=1e-5, atol=1e-5)
        )

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
