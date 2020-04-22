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


class TestQR(unittest.TestCase):
    @unittest.skipIf(not extended_tests, "extended tests")
    def test_qr_sp0_ext(self):
        st_whole = torch.randn(70, 70, device=device)
        sp = 0
        for m in range(50, st_whole.shape[0] + 1, 1):
            for n in range(50, st_whole.shape[1] + 1, 1):
                for t in range(1, 3):
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

    @unittest.skipIf(not extended_tests, "extended tests")
    def test_qr_sp1_ext(self):
        st_whole = torch.randn(70, 70, device=device)
        sp = 1
        for m in range(50, st_whole.shape[0] + 1, 1):
            for n in range(50, st_whole.shape[1] + 1, 1):
                for t in range(1, 3):
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

    def test_qr(self):
        m, n = 20, 40
        st = torch.randn(m, n, device=device, dtype=torch.float)
        a_comp = ht.array(st, split=0, device=ht_device)
        for t in range(1, 3):
            for sp in range(2):
                a = ht.array(st, split=sp, device=ht_device, dtype=torch.float)
                qr = a.qr(tiles_per_proc=t)
                self.assertTrue(ht.allclose((a_comp - (qr.Q @ qr.R)), 0, rtol=1e-5, atol=1e-5))
                self.assertTrue(
                    ht.allclose(qr.Q.T @ qr.Q, ht.eye(m, device=ht_device), rtol=1e-5, atol=1e-5)
                )
                self.assertTrue(
                    ht.allclose(ht.eye(m, device=ht_device), qr.Q @ qr.Q.T, rtol=1e-5, atol=1e-5)
                )
        # m, n = 40, 40
        # st1 = torch.randn(m, n, device=device)
        # a_comp1 = ht.array(st1, split=0, device=ht_device)
        # for t in range(1, 3):
        #     for sp in range(2):
        #         a1 = ht.array(st1, split=sp, device=ht_device)
        #         qr1 = a1.qr(tiles_per_proc=t)
        #         self.assertTrue(ht.allclose((a_comp1 - (qr1.Q @ qr1.R)), 0, rtol=1e-5, atol=1e-5))
        #         self.assertTrue(
        #             ht.allclose(qr1.Q.T @ qr1.Q, ht.eye(m, device=ht_device), rtol=1e-5, atol=1e-5)
        #         )
        #         self.assertTrue(
        #             ht.allclose(ht.eye(m, device=ht_device), qr1.Q @ qr1.Q.T, rtol=1e-5, atol=1e-5)
        #         )
        # m, n = 40, 20
        # st2 = torch.randn(m, n, dtype=torch.double, device=device)
        # a_comp2 = ht.array(st2, split=0, dtype=ht.double, device=ht_device)
        # for t in range(1, 3):
        #     for sp in range(2):
        #         a2 = ht.array(st2, split=sp, device=ht_device)
        #         qr2 = a2.qr(tiles_per_proc=t)
        #         self.assertTrue(ht.allclose(a_comp2, qr2.Q @ qr2.R, rtol=1e-5, atol=1e-5))
        #         self.assertTrue(
        #             ht.allclose(
        #                 qr2.Q.T @ qr2.Q,
        #                 ht.eye(m, dtype=ht.double, device=ht_device),
        #                 rtol=1e-5,
        #                 atol=1e-5,
        #             )
        #         )
        #         self.assertTrue(
        #             ht.allclose(
        #                 ht.eye(m, dtype=ht.double, device=ht_device),
        #                 qr2.Q @ qr2.Q.T,
        #                 rtol=1e-5,
        #                 atol=1e-5,
        #             )
        #         )
        #         # test if calc R alone works
        #         qr = ht.qr(a2, calc_q=False, overwrite_a=True)
        #         self.assertTrue(qr.Q is None)
        #
        # m, n = 40, 20
        # st = torch.randn(m, n, device=device)
        # a_comp = ht.array(st, split=None, device=ht_device)
        # a = ht.array(st, split=None, device=ht_device)
        # qr = a.qr()
        # self.assertTrue(ht.allclose(a_comp, qr.Q @ qr.R, rtol=1e-5, atol=1e-5))
        # self.assertTrue(
        #     ht.allclose(qr.Q.T @ qr.Q, ht.eye(m, device=ht_device), rtol=1e-5, atol=1e-5)
        # )
        # self.assertTrue(
        #     ht.allclose(ht.eye(m, device=ht_device), qr.Q @ qr.Q.T, rtol=1e-5, atol=1e-5)
        # )
        #
        # # raises
        # with self.assertRaises(TypeError):
        #     ht.qr(np.zeros((10, 10)))
        # with self.assertRaises(TypeError):
        #     ht.qr(a_comp, tiles_per_proc="ls")
        # with self.assertRaises(TypeError):
        #     ht.qr(a_comp, tiles_per_proc=1, calc_q=30)
        # with self.assertRaises(TypeError):
        #     ht.qr(a_comp, tiles_per_proc=1, overwrite_a=30)
        # with self.assertRaises(ValueError):
        #     ht.qr(a_comp, tiles_per_proc=torch.tensor([1, 2, 3]))
        # with self.assertRaises(ValueError):
        #     ht.qr(ht.zeros((3, 4, 5)))
