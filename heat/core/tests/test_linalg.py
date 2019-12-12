import torch
import unittest
import os
import heat as ht
import numpy as np

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


class TestLinalg(unittest.TestCase):
    def test_dot(self):
        # ONLY TESTING CORRECTNESS! ALL CALLS IN DOT ARE PREVIOUSLY TESTED
        # cases to test:
        data2d = np.ones((10, 10))
        data3d = np.ones((10, 10, 10))
        data1d = np.arange(10)

        a1d = ht.array(data1d, dtype=ht.float32, split=0, device=ht_device)
        b1d = ht.array(data1d, dtype=ht.float32, split=0, device=ht_device)

        # 2 1D arrays,
        self.assertEqual(ht.dot(a1d, b1d), np.dot(data1d, data1d))
        ret = []
        self.assertEqual(ht.dot(a1d, b1d, out=ret), np.dot(data1d, data1d))

        a1d = ht.array(data1d, dtype=ht.float32, split=None, device=ht_device)
        b1d = ht.array(data1d, dtype=ht.float32, split=0, device=ht_device)
        # 2 1D arrays,
        self.assertEqual(ht.dot(a1d, b1d), np.dot(data1d, data1d))

        a2d = ht.array(data2d, split=1, device=ht_device)
        b2d = ht.array(data2d, split=1, device=ht_device)
        # 2 2D arrays,
        res = ht.dot(a2d, b2d) - ht.array(np.dot(data2d, data2d), device=ht_device)
        self.assertEqual(ht.equal(res, ht.zeros(res.shape, device=ht_device)), 1)
        ret = ht.array(data2d, split=1, device=ht_device)
        ht.dot(a2d, b2d, out=ret)
        # print(ht.dot(a2d, b2d, out=ret))
        res = ret - ht.array(np.dot(data2d, data2d), device=ht_device)
        self.assertEqual(ht.equal(res, ht.zeros(res.shape, device=ht_device)), 1)

        const1 = 5
        const2 = 6
        # a is const,
        res = ht.dot(const1, b2d) - ht.array(np.dot(const1, data2d), device=ht_device)
        ret = 0
        ht.dot(const1, b2d, out=ret)
        self.assertEqual(ht.equal(res, ht.zeros(res.shape, device=ht_device)), 1)

        # b is const,
        res = ht.dot(a2d, const2) - ht.array(np.dot(data2d, const2), device=ht_device)
        self.assertEqual(ht.equal(res, ht.zeros(res.shape, device=ht_device)), 1)
        # a and b and const
        self.assertEqual(ht.dot(const2, const1), 5 * 6)

        with self.assertRaises(NotImplementedError):
            ht.dot(ht.array(data3d, device=ht_device), ht.array(data1d, device=ht_device))

    def test_matmul(self):
        with self.assertRaises(ValueError):
            ht.matmul(ht.ones((25, 25), device=ht_device), ht.ones((42, 42), device=ht_device))

        # cases to test:
        n, m = 21, 31
        j, k = m, 45
        a_torch = torch.ones((n, m), device=device)
        a_torch[0] = torch.arange(1, m + 1, device=device)
        a_torch[:, -1] = torch.arange(1, n + 1, device=device)
        b_torch = torch.ones((j, k), device=device)
        b_torch[0] = torch.arange(1, k + 1, device=device)
        b_torch[:, 0] = torch.arange(1, j + 1, device=device)

        # splits None None
        a = ht.ones((n, m), split=None, device=ht_device)
        b = ht.ones((j, k), split=None, device=ht_device)
        a[0] = ht.arange(1, m + 1, device=ht_device)
        a[:, -1] = ht.arange(1, n + 1, device=ht_device)
        b[0] = ht.arange(1, k + 1, device=ht_device)
        b[:, 0] = ht.arange(1, j + 1, device=ht_device)
        ret00 = ht.matmul(a, b)

        self.assertEqual(ht.all(ret00 == ht.array(a_torch @ b_torch, device=ht_device)), 1)
        self.assertIsInstance(ret00, ht.DNDarray)
        self.assertEqual(ret00.shape, (n, k))
        self.assertEqual(ret00.dtype, ht.float)
        self.assertEqual(ret00.split, None)

        if a.comm.size > 1:
            # splits 00
            a = ht.ones((n, m), split=0, dtype=ht.float64, device=ht_device)
            b = ht.ones((j, k), split=0, device=ht_device)
            a[0] = ht.arange(1, m + 1, device=ht_device)
            a[:, -1] = ht.arange(1, n + 1, device=ht_device)
            b[0] = ht.arange(1, k + 1, device=ht_device)
            b[:, 0] = ht.arange(1, j + 1, device=ht_device)
            ret00 = a @ b

            ret_comp00 = ht.array(a_torch @ b_torch, split=0, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp00))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n, k))
            self.assertEqual(ret00.dtype, ht.float64)
            self.assertEqual(ret00.split, 0)

            # splits 00 (numpy)
            a = ht.array(np.ones((n, m)), split=0, device=ht_device)
            b = ht.array(np.ones((j, k)), split=0, device=ht_device)
            a[0] = ht.arange(1, m + 1, device=ht_device)
            a[:, -1] = ht.arange(1, n + 1, device=ht_device)
            b[0] = ht.arange(1, k + 1, device=ht_device)
            b[:, 0] = ht.arange(1, j + 1, device=ht_device)
            ret00 = a @ b

            ret_comp00 = ht.array(a_torch @ b_torch, split=0, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp00))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n, k))
            self.assertEqual(ret00.dtype, ht.float64)
            self.assertEqual(ret00.split, 0)

            # splits 01
            a = ht.ones((n, m), split=0, device=ht_device)
            b = ht.ones((j, k), split=1, dtype=ht.float64, device=ht_device)
            a[0] = ht.arange(1, m + 1, device=ht_device)
            a[:, -1] = ht.arange(1, n + 1, device=ht_device)
            b[0] = ht.arange(1, k + 1, device=ht_device)
            b[:, 0] = ht.arange(1, j + 1, device=ht_device)
            ret00 = ht.matmul(a, b)

            ret_comp01 = ht.array(a_torch @ b_torch, split=0, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp01))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n, k))
            self.assertEqual(ret00.dtype, ht.float64)
            self.assertEqual(ret00.split, 0)

            # splits 10
            a = ht.ones((n, m), split=1, device=ht_device)
            b = ht.ones((j, k), split=0, device=ht_device)
            a[0] = ht.arange(1, m + 1, device=ht_device)
            a[:, -1] = ht.arange(1, n + 1, device=ht_device)
            b[0] = ht.arange(1, k + 1, device=ht_device)
            b[:, 0] = ht.arange(1, j + 1, device=ht_device)
            ret00 = ht.matmul(a, b)

            ret_comp10 = ht.array(a_torch @ b_torch, split=1, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp10))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n, k))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 1)

            # splits 11
            a = ht.ones((n, m), split=1, device=ht_device)
            b = ht.ones((j, k), split=1, device=ht_device)
            a[0] = ht.arange(1, m + 1, device=ht_device)
            a[:, -1] = ht.arange(1, n + 1, device=ht_device)
            b[0] = ht.arange(1, k + 1, device=ht_device)
            b[:, 0] = ht.arange(1, j + 1, device=ht_device)
            ret00 = ht.matmul(a, b)

            ret_comp11 = ht.array(a_torch @ b_torch, split=1, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp11))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n, k))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 1)

            # splits 11 (torch)
            a = ht.array(torch.ones((n, m), device=device), split=1, device=ht_device)
            b = ht.array(torch.ones((j, k), device=device), split=1, device=ht_device)
            a[0] = ht.arange(1, m + 1, device=ht_device)
            a[:, -1] = ht.arange(1, n + 1, device=ht_device)
            b[0] = ht.arange(1, k + 1, device=ht_device)
            b[:, 0] = ht.arange(1, j + 1, device=ht_device)
            ret00 = ht.matmul(a, b)

            ret_comp11 = ht.array(a_torch @ b_torch, split=1, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp11))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n, k))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 1)

            # splits 0 None
            a = ht.ones((n, m), split=0, device=ht_device)
            b = ht.ones((j, k), split=None, device=ht_device)
            a[0] = ht.arange(1, m + 1, device=ht_device)
            a[:, -1] = ht.arange(1, n + 1, device=ht_device)
            b[0] = ht.arange(1, k + 1, device=ht_device)
            b[:, 0] = ht.arange(1, j + 1, device=ht_device)
            ret00 = ht.matmul(a, b)

            ret_comp0 = ht.array(a_torch @ b_torch, split=0, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp0))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n, k))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # splits 1 None
            a = ht.ones((n, m), split=1, device=ht_device)
            b = ht.ones((j, k), split=None, device=ht_device)
            a[0] = ht.arange(1, m + 1, device=ht_device)
            a[:, -1] = ht.arange(1, n + 1, device=ht_device)
            b[0] = ht.arange(1, k + 1, device=ht_device)
            b[:, 0] = ht.arange(1, j + 1, device=ht_device)
            ret00 = ht.matmul(a, b)

            ret_comp1 = ht.array(a_torch @ b_torch, split=1, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp1))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n, k))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 1)

            # splits None 0
            a = ht.ones((n, m), split=None, device=ht_device)
            b = ht.ones((j, k), split=0, device=ht_device)
            a[0] = ht.arange(1, m + 1, device=ht_device)
            a[:, -1] = ht.arange(1, n + 1, device=ht_device)
            b[0] = ht.arange(1, k + 1, device=ht_device)
            b[:, 0] = ht.arange(1, j + 1, device=ht_device)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array(a_torch @ b_torch, split=0, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n, k))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # splits None 1
            a = ht.ones((n, m), split=None, device=ht_device)
            b = ht.ones((j, k), split=1, device=ht_device)
            a[0] = ht.arange(1, m + 1, device=ht_device)
            a[:, -1] = ht.arange(1, n + 1, device=ht_device)
            b[0] = ht.arange(1, k + 1, device=ht_device)
            b[:, 0] = ht.arange(1, j + 1, device=ht_device)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array(a_torch @ b_torch, split=1, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n, k))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 1)

            # vector matrix mult:
            # a -> vector
            a_torch = torch.ones((m), device=device)
            b_torch = torch.ones((j, k), device=device)
            b_torch[0] = torch.arange(1, k + 1, device=device)
            b_torch[:, 0] = torch.arange(1, j + 1, device=device)
            # splits None None
            a = ht.ones((m), split=None, device=ht_device)
            b = ht.ones((j, k), split=None, device=ht_device)
            b[0] = ht.arange(1, k + 1, device=ht_device)
            b[:, 0] = ht.arange(1, j + 1, device=ht_device)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array(a_torch @ b_torch, split=None, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (k,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, None)

            # splits None 0
            a = ht.ones((m), split=None, device=ht_device)
            b = ht.ones((j, k), split=0, device=ht_device)
            b[0] = ht.arange(1, k + 1, device=ht_device)
            b[:, 0] = ht.arange(1, j + 1, device=ht_device)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array(a_torch @ b_torch, split=None, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (k,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # splits None 1
            a = ht.ones((m), split=None, device=ht_device)
            b = ht.ones((j, k), split=1, device=ht_device)
            b[0] = ht.arange(1, k + 1, device=ht_device)
            b[:, 0] = ht.arange(1, j + 1, device=ht_device)
            ret00 = ht.matmul(a, b)
            ret_comp = ht.array(a_torch @ b_torch, split=0, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (k,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # splits 0 None
            a = ht.ones((m), split=None, device=ht_device)
            b = ht.ones((j, k), split=0, device=ht_device)
            b[0] = ht.arange(1, k + 1, device=ht_device)
            b[:, 0] = ht.arange(1, j + 1, device=ht_device)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array(a_torch @ b_torch, split=None, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (k,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # splits 0 0
            a = ht.ones((m), split=0, device=ht_device)
            b = ht.ones((j, k), split=0, device=ht_device)
            b[0] = ht.arange(1, k + 1, device=ht_device)
            b[:, 0] = ht.arange(1, j + 1, device=ht_device)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array(a_torch @ b_torch, split=None, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (k,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # splits 0 1
            a = ht.ones((m), split=0, device=ht_device)
            b = ht.ones((j, k), split=1, device=ht_device)
            b[0] = ht.arange(1, k + 1, device=ht_device)
            b[:, 0] = ht.arange(1, j + 1, device=ht_device)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array(a_torch @ b_torch, split=None, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (k,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # b -> vector
            a_torch = torch.ones((n, m), device=device)
            a_torch[0] = torch.arange(1, m + 1, device=device)
            a_torch[:, -1] = torch.arange(1, n + 1, device=device)
            b_torch = torch.ones((j), device=device)
            # splits None None
            a = ht.ones((n, m), split=None, device=ht_device)
            b = ht.ones((j), split=None, device=ht_device)
            a[0] = ht.arange(1, m + 1, device=ht_device)
            a[:, -1] = ht.arange(1, n + 1, device=ht_device)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array(a_torch @ b_torch, split=None, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, None)

            # splits 0 None
            a = ht.ones((n, m), split=0, device=ht_device)
            b = ht.ones((j), split=None, device=ht_device)
            a[0] = ht.arange(1, m + 1, device=ht_device)
            a[:, -1] = ht.arange(1, n + 1, device=ht_device)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array((a_torch @ b_torch), split=None, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # splits 1 None
            a = ht.ones((n, m), split=1, device=ht_device)
            b = ht.ones((j), split=None, device=ht_device)
            a[0] = ht.arange(1, m + 1, device=ht_device)
            a[:, -1] = ht.arange(1, n + 1, device=ht_device)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array((a_torch @ b_torch), split=None, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # splits None 0
            a = ht.ones((n, m), split=None, device=ht_device)
            b = ht.ones((j), split=0, device=ht_device)
            a[0] = ht.arange(1, m + 1, device=ht_device)
            a[:, -1] = ht.arange(1, n + 1, device=ht_device)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array((a_torch @ b_torch), split=None, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # splits 0 0
            a = ht.ones((n, m), split=0, device=ht_device)
            b = ht.ones((j), split=0, device=ht_device)
            a[0] = ht.arange(1, m + 1, device=ht_device)
            a[:, -1] = ht.arange(1, n + 1, device=ht_device)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array((a_torch @ b_torch), split=None, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # splits 1 0
            a = ht.ones((n, m), split=1, device=ht_device)
            b = ht.ones((j), split=0, device=ht_device)
            a[0] = ht.arange(1, m + 1, device=ht_device)
            a[:, -1] = ht.arange(1, n + 1, device=ht_device)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array((a_torch @ b_torch), split=None, device=ht_device)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            with self.assertRaises(NotImplementedError):
                a = ht.zeros((3, 3, 3), split=2)
                b = a.copy()
                a @ b

    def test_qr(self):
        m, n = 40, 40
        st = torch.randn(m, n)
        a_comp = ht.array(st, split=0)
        for t in range(1, 3):
            for sp in range(2):
                a = ht.array(st, split=sp)
                qr = a.qr(tiles_per_proc=t)
                self.assertTrue(ht.allclose(a_comp, qr.Q @ qr.R, rtol=1e-5, atol=1e-5))
                self.assertTrue(ht.allclose(qr.Q.T @ qr.Q, ht.eye(m), rtol=1e-5, atol=1e-5))
                self.assertTrue(ht.allclose(ht.eye(m), qr.Q @ qr.Q.T, rtol=1e-5, atol=1e-5))
        m, n = 20, 40
        st = torch.randn(m, n)
        a_comp = ht.array(st, split=0)
        for t in range(1, 3):
            for sp in range(2):
                a = ht.array(st, split=sp)
                qr = a.qr(tiles_per_proc=t)
                self.assertTrue(ht.allclose(a_comp, qr.Q @ qr.R, rtol=1e-5, atol=1e-5))
                self.assertTrue(ht.allclose(qr.Q.T @ qr.Q, ht.eye(m), rtol=1e-5, atol=1e-5))
                self.assertTrue(ht.allclose(ht.eye(m), qr.Q @ qr.Q.T, rtol=1e-5, atol=1e-5))
        m, n = 40, 20
        st = torch.randn(m, n, dtype=torch.double)
        a_comp = ht.array(st, split=0, dtype=ht.double)
        for t in range(1, 3):
            for sp in range(2):
                a = ht.array(st, split=sp)
                qr = a.qr(tiles_per_proc=t)
                self.assertTrue(ht.allclose(a_comp, qr.Q @ qr.R, rtol=1e-5, atol=1e-5))
                self.assertTrue(
                    ht.allclose(qr.Q.T @ qr.Q, ht.eye(m, dtype=ht.double), rtol=1e-5, atol=1e-5)
                )
                self.assertTrue(
                    ht.allclose(ht.eye(m, dtype=ht.double), qr.Q @ qr.Q.T, rtol=1e-5, atol=1e-5)
                )

        m, n = 40, 20
        st = torch.randn(m, n)
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

    def test_transpose(self):
        # vector transpose, not distributed
        vector = ht.arange(10, device=ht_device)
        vector_t = vector.T
        self.assertIsInstance(vector_t, ht.DNDarray)
        self.assertEqual(vector_t.dtype, ht.int32)
        self.assertEqual(vector_t.split, None)
        self.assertEqual(vector_t.shape, (10,))

        # simple matrix transpose, not distributed
        simple_matrix = ht.zeros((2, 4), device=ht_device)
        simple_matrix_t = simple_matrix.transpose()
        self.assertIsInstance(simple_matrix_t, ht.DNDarray)
        self.assertEqual(simple_matrix_t.dtype, ht.float32)
        self.assertEqual(simple_matrix_t.split, None)
        self.assertEqual(simple_matrix_t.shape, (4, 2))
        self.assertEqual(simple_matrix_t._DNDarray__array.shape, (4, 2))

        # 4D array, not distributed, with given axis
        array_4d = ht.zeros((2, 3, 4, 5), device=ht_device)
        array_4d_t = ht.transpose(array_4d, axes=(-1, 0, 2, 1))
        self.assertIsInstance(array_4d_t, ht.DNDarray)
        self.assertEqual(array_4d_t.dtype, ht.float32)
        self.assertEqual(array_4d_t.split, None)
        self.assertEqual(array_4d_t.shape, (5, 2, 4, 3))
        self.assertEqual(array_4d_t._DNDarray__array.shape, (5, 2, 4, 3))

        # vector transpose, distributed
        vector_split = ht.arange(10, split=0, device=ht_device)
        vector_split_t = vector_split.T
        self.assertIsInstance(vector_split_t, ht.DNDarray)
        self.assertEqual(vector_split_t.dtype, ht.int32)
        self.assertEqual(vector_split_t.split, 0)
        self.assertEqual(vector_split_t.shape, (10,))
        self.assertLessEqual(vector_split_t.lshape[0], 10)

        # matrix transpose, distributed
        matrix_split = ht.ones((10, 20), split=1, device=ht_device)
        matrix_split_t = matrix_split.transpose()
        self.assertIsInstance(matrix_split_t, ht.DNDarray)
        self.assertEqual(matrix_split_t.dtype, ht.float32)
        self.assertEqual(matrix_split_t.split, 0)
        self.assertEqual(matrix_split_t.shape, (20, 10))
        self.assertLessEqual(matrix_split_t.lshape[0], 20)
        self.assertEqual(matrix_split_t.lshape[1], 10)

        # 4D array, distributed
        array_4d_split = ht.ones((3, 4, 5, 6), split=3, device=ht_device)
        array_4d_split_t = ht.transpose(array_4d_split, axes=(1, 0, 3, 2))
        self.assertIsInstance(array_4d_t, ht.DNDarray)
        self.assertEqual(array_4d_split_t.dtype, ht.float32)
        self.assertEqual(array_4d_split_t.split, 2)
        self.assertEqual(array_4d_split_t.shape, (4, 3, 6, 5))

        self.assertEqual(array_4d_split_t.lshape[0], 4)
        self.assertEqual(array_4d_split_t.lshape[1], 3)
        self.assertLessEqual(array_4d_split_t.lshape[2], 6)
        self.assertEqual(array_4d_split_t.lshape[3], 5)

        # exceptions
        with self.assertRaises(TypeError):
            ht.transpose(1)
        with self.assertRaises(ValueError):
            ht.transpose(ht.zeros((2, 3), device=ht_device), axes=1.0)
        with self.assertRaises(ValueError):
            ht.transpose(ht.zeros((2, 3), device=ht_device), axes=(-1,))
        with self.assertRaises(TypeError):
            ht.zeros((2, 3), device=ht_device).transpose(axes="01")
        with self.assertRaises(TypeError):
            ht.zeros((2, 3), device=ht_device).transpose(axes=(0, 1.0))
        with self.assertRaises((ValueError, IndexError)):
            ht.zeros((2, 3), device=ht_device).transpose(axes=(0, 3))

    def test_tril(self):
        local_ones = ht.ones((5,), device=ht_device)

        # 1D case, no offset, data is not split, module-level call
        result = ht.tril(local_ones)
        comparison = torch.ones((5, 5), device=device).tril()
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.lshape, (5, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        # 1D case, positive offset, data is not split, module-level call
        result = ht.tril(local_ones, k=2)
        comparison = torch.ones((5, 5), device=device).tril(diagonal=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.lshape, (5, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        # 1D case, negative offset, data is not split, module-level call
        result = ht.tril(local_ones, k=-2)
        comparison = torch.ones((5, 5), device=device).tril(diagonal=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.lshape, (5, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        local_ones = ht.ones((4, 5), device=ht_device)

        # 2D case, no offset, data is not split, method
        result = local_ones.tril()
        comparison = torch.ones((4, 5), device=device).tril()
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        # 2D case, positive offset, data is not split, method
        result = local_ones.tril(k=2)
        comparison = torch.ones((4, 5), device=device).tril(diagonal=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        # 2D case, negative offset, data is not split, method
        result = local_ones.tril(k=-2)
        comparison = torch.ones((4, 5), device=device).tril(diagonal=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        local_ones = ht.ones((3, 4, 5, 6), device=ht_device)

        # 2D+ case, no offset, data is not split, module-level call
        result = local_ones.tril()
        comparison = torch.ones((5, 6), device=device).tril()
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (3, 4, 5, 6))
        self.assertEqual(result.lshape, (3, 4, 5, 6))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._DNDarray__array[i, j] == comparison).all())

        # 2D+ case, positive offset, data is not split, module-level call
        result = local_ones.tril(k=2)
        comparison = torch.ones((5, 6), device=device).tril(diagonal=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (3, 4, 5, 6))
        self.assertEqual(result.lshape, (3, 4, 5, 6))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._DNDarray__array[i, j] == comparison).all())

        # # 2D+ case, negative offset, data is not split, module-level call
        result = local_ones.tril(k=-2)
        comparison = torch.ones((5, 6), device=device).tril(diagonal=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (3, 4, 5, 6))
        self.assertEqual(result.lshape, (3, 4, 5, 6))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._DNDarray__array[i, j] == comparison).all())

        distributed_ones = ht.ones((5,), split=0, device=ht_device)

        # 1D case, no offset, data is split, method
        result = distributed_ones.tril()
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.split, 1)
        self.assertTrue(result.lshape[0] == 5 or result.lshape[0] == 0)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertTrue(result.sum(), 15)
        if result.comm.rank == 0:
            self.assertTrue(result._DNDarray__array[-1, 0] == 1)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result._DNDarray__array[0, -1] == 0)

        # 1D case, positive offset, data is split, method
        result = distributed_ones.tril(k=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 22)
        if result.comm.rank == 0:
            self.assertTrue(result._DNDarray__array[-1, 0] == 1)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result._DNDarray__array[0, -1] == 0)

        # 1D case, negative offset, data is split, method
        result = distributed_ones.tril(k=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 6)
        if result.comm.rank == 0:
            self.assertTrue(result._DNDarray__array[-1, 0] == 1)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result._DNDarray__array[0, -1] == 0)

        distributed_ones = ht.ones((4, 5), split=0, device=ht_device)

        # 2D case, no offset, data is horizontally split, method
        result = distributed_ones.tril()
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 10)
        if result.comm.rank == 0:
            self.assertTrue(result._DNDarray__array[0, -1] == 0)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result._DNDarray__array[-1, 0] == 1)

        # 2D case, positive offset, data is horizontally split, method
        result = distributed_ones.tril(k=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 17)
        if result.comm.rank == 0:
            self.assertTrue(result._DNDarray__array[0, -1] == 0)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result._DNDarray__array[-1, 0] == 1)

        # 2D case, negative offset, data is horizontally split, method
        result = distributed_ones.tril(k=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 3)
        if result.comm.rank == 0:
            self.assertTrue(result._DNDarray__array[0, -1] == 0)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result._DNDarray__array[-1, 0] == 1)

        distributed_ones = ht.ones((4, 5), split=1, device=ht_device)

        # 2D case, no offset, data is vertically split, method
        result = distributed_ones.tril()
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 10)
        if result.comm.rank == 0:
            self.assertTrue(result._DNDarray__array[-1, 0] == 1)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result._DNDarray__array[0, -1] == 0)

        # 2D case, positive offset, data is horizontally split, method
        result = distributed_ones.tril(k=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 17)
        if result.comm.rank == 0:
            self.assertTrue(result._DNDarray__array[-1, 0] == 1)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result._DNDarray__array[0, -1] == 0)

        # 2D case, negative offset, data is horizontally split, method
        result = distributed_ones.tril(k=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 3)
        if result.comm.rank == 0:
            self.assertTrue(result._DNDarray__array[-1, 0] == 1)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result._DNDarray__array[0, -1] == 0)

        with self.assertRaises(TypeError):
            ht.tril("asdf")
        with self.assertRaises(TypeError):
            ht.tril(distributed_ones, m=["sdf", "sf"])

    def test_triu(self):
        local_ones = ht.ones((5,), device=ht_device)

        # 1D case, no offset, data is not split, module-level call
        result = ht.triu(local_ones)
        comparison = torch.ones((5, 5), device=device).triu()
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.lshape, (5, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        # 1D case, positive offset, data is not split, module-level call
        result = ht.triu(local_ones, k=2)
        comparison = torch.ones((5, 5), device=device).triu(diagonal=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.lshape, (5, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        # 1D case, negative offset, data is not split, module-level call
        result = ht.triu(local_ones, k=-2)
        comparison = torch.ones((5, 5), device=device).triu(diagonal=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.lshape, (5, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        local_ones = ht.ones((4, 5), device=ht_device)

        # 2D case, no offset, data is not split, method
        result = local_ones.triu()
        comparison = torch.ones((4, 5), device=device).triu()
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        # 2D case, positive offset, data is not split, method
        result = local_ones.triu(k=2)
        comparison = torch.ones((4, 5), device=device).triu(diagonal=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        # 2D case, negative offset, data is not split, method
        result = local_ones.triu(k=-2)
        comparison = torch.ones((4, 5), device=device).triu(diagonal=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        local_ones = ht.ones((3, 4, 5, 6), device=ht_device)

        # 2D+ case, no offset, data is not split, module-level call
        result = local_ones.triu()
        comparison = torch.ones((5, 6), device=device).triu()
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (3, 4, 5, 6))
        self.assertEqual(result.lshape, (3, 4, 5, 6))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._DNDarray__array[i, j] == comparison).all())

        # 2D+ case, positive offset, data is not split, module-level call
        result = local_ones.triu(k=2)
        comparison = torch.ones((5, 6), device=device).triu(diagonal=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (3, 4, 5, 6))
        self.assertEqual(result.lshape, (3, 4, 5, 6))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._DNDarray__array[i, j] == comparison).all())

        # # 2D+ case, negative offset, data is not split, module-level call
        result = local_ones.triu(k=-2)
        comparison = torch.ones((5, 6), device=device).triu(diagonal=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (3, 4, 5, 6))
        self.assertEqual(result.lshape, (3, 4, 5, 6))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._DNDarray__array[i, j] == comparison).all())

        distributed_ones = ht.ones((5,), split=0, device=ht_device)

        # 1D case, no offset, data is split, method
        result = distributed_ones.triu()
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertTrue(result.sum(), 15)
        if result.comm.rank == 0:
            self.assertTrue(result._DNDarray__array[-1, 0] == 0)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result._DNDarray__array[0, -1] == 1)

        # 1D case, positive offset, data is split, method
        result = distributed_ones.triu(k=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 6)
        if result.comm.rank == 0:
            self.assertTrue(result._DNDarray__array[-1, 0] == 0)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result._DNDarray__array[0, -1] == 1)

        # 1D case, negative offset, data is split, method
        result = distributed_ones.triu(k=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 22)
        if result.comm.rank == 0:
            self.assertTrue(result._DNDarray__array[-1, 0] == 0)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result._DNDarray__array[0, -1] == 1)

        distributed_ones = ht.ones((4, 5), split=0, device=ht_device)

        # 2D case, no offset, data is horizontally split, method
        result = distributed_ones.triu()
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 14)
        if result.comm.rank == 0:
            self.assertTrue(result._DNDarray__array[0, -1] == 1)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result._DNDarray__array[-1, 0] == 0)

        # # 2D case, positive offset, data is horizontally split, method
        result = distributed_ones.triu(k=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 6)
        if result.comm.rank == 0:
            self.assertTrue(result._DNDarray__array[0, -1] == 1)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result._DNDarray__array[-1, 0] == 0)

        # # 2D case, negative offset, data is horizontally split, method
        result = distributed_ones.triu(k=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 19)
        if result.comm.rank == 0:
            self.assertTrue(result._DNDarray__array[0, -1] == 1)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result._DNDarray__array[-1, 0] == 0)

        distributed_ones = ht.ones((4, 5), split=1, device=ht_device)

        # 2D case, no offset, data is vertically split, method
        result = distributed_ones.triu()
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 14)
        if result.comm.rank == 0:
            self.assertTrue(result._DNDarray__array[-1, 0] == 0)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result._DNDarray__array[0, -1] == 1)

        # 2D case, positive offset, data is horizontally split, method
        result = distributed_ones.triu(k=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 6)
        if result.comm.rank == 0:
            self.assertTrue(result._DNDarray__array[-1, 0] == 0)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result._DNDarray__array[0, -1] == 1)

        # 2D case, negative offset, data is horizontally split, method
        result = distributed_ones.triu(k=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 19)
        if result.comm.rank == 0:
            self.assertTrue(result._DNDarray__array[-1, 0] == 0)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result._DNDarray__array[0, -1] == 1)
