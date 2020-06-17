import torch
import os
import unittest
import heat as ht
import numpy as np

from ...tests.test_suites.basic_test import TestCase


class TestLinalgBasics(TestCase):
    def test_dot(self):
        # ONLY TESTING CORRECTNESS! ALL CALLS IN DOT ARE PREVIOUSLY TESTED
        # cases to test:
        data2d = np.ones((10, 10))
        data3d = np.ones((10, 10, 10))
        data1d = np.arange(10)

        a1d = ht.array(data1d, dtype=ht.float32, split=0)
        b1d = ht.array(data1d, dtype=ht.float32, split=0)

        # 2 1D arrays,
        self.assertEqual(ht.dot(a1d, b1d), np.dot(data1d, data1d))
        ret = []
        self.assertEqual(ht.dot(a1d, b1d, out=ret), np.dot(data1d, data1d))

        a1d = ht.array(data1d, dtype=ht.float32, split=None)
        b1d = ht.array(data1d, dtype=ht.float32, split=0)
        self.assertEqual(ht.dot(a1d, b1d), np.dot(data1d, data1d))

        a1d = ht.array(data1d, dtype=ht.float32, split=None)
        b1d = ht.array(data1d, dtype=ht.float32, split=None)
        self.assertEqual(ht.dot(a1d, b1d), np.dot(data1d, data1d))

        a1d = ht.array(data1d, dtype=ht.float32, split=0)
        b1d = ht.array(data1d, dtype=ht.float32, split=0)
        self.assertEqual(ht.dot(a1d, b1d), np.dot(data1d, data1d))
        # 2 1D arrays,

        a2d = ht.array(data2d, split=1)
        b2d = ht.array(data2d, split=1)
        # 2 2D arrays,
        res = ht.dot(a2d, b2d) - ht.array(np.dot(data2d, data2d))
        self.assertEqual(ht.equal(res, ht.zeros(res.shape)), 1)
        ret = ht.array(data2d, split=1)
        ht.dot(a2d, b2d, out=ret)

        res = ret - ht.array(np.dot(data2d, data2d))
        self.assertEqual(ht.equal(res, ht.zeros(res.shape)), 1)

        const1 = 5
        const2 = 6
        # a is const
        res = ht.dot(const1, b2d) - ht.array(np.dot(const1, data2d))
        ret = 0
        ht.dot(const1, b2d, out=ret)
        self.assertEqual(ht.equal(res, ht.zeros(res.shape)), 1)

        # b is const
        res = ht.dot(a2d, const2) - ht.array(np.dot(data2d, const2))
        self.assertEqual(ht.equal(res, ht.zeros(res.shape)), 1)
        # a and b and const
        self.assertEqual(ht.dot(const2, const1), 5 * 6)

        with self.assertRaises(NotImplementedError):
            ht.dot(ht.array(data3d), ht.array(data1d))

    def test_matmul(self):
        with self.assertRaises(ValueError):
            ht.matmul(ht.ones((25, 25)), ht.ones((42, 42)))

        # cases to test:
        n, m = 21, 31
        j, k = m, 45
        a_torch = torch.ones((n, m), device=self.device.torch_device)
        a_torch[0] = torch.arange(1, m + 1, device=self.device.torch_device)
        a_torch[:, -1] = torch.arange(1, n + 1, device=self.device.torch_device)
        b_torch = torch.ones((j, k), device=self.device.torch_device)
        b_torch[0] = torch.arange(1, k + 1, device=self.device.torch_device)
        b_torch[:, 0] = torch.arange(1, j + 1, device=self.device.torch_device)

        # splits None None
        a = ht.ones((n, m), split=None)
        b = ht.ones((j, k), split=None)
        a[0] = ht.arange(1, m + 1)
        a[:, -1] = ht.arange(1, n + 1)
        b[0] = ht.arange(1, k + 1)
        b[:, 0] = ht.arange(1, j + 1)
        ret00 = ht.matmul(a, b)

        self.assertEqual(ht.all(ret00 == ht.array(a_torch @ b_torch)), 1)
        self.assertIsInstance(ret00, ht.DNDarray)
        self.assertEqual(ret00.shape, (n, k))
        self.assertEqual(ret00.dtype, ht.float)
        self.assertEqual(ret00.split, None)
        self.assertEqual(a.split, None)
        self.assertEqual(b.split, None)

        # splits None None
        a = ht.ones((n, m), split=None)
        b = ht.ones((j, k), split=None)
        a[0] = ht.arange(1, m + 1)
        a[:, -1] = ht.arange(1, n + 1)
        b[0] = ht.arange(1, k + 1)
        b[:, 0] = ht.arange(1, j + 1)
        ret00 = ht.matmul(a, b, allow_resplit=True)

        self.assertEqual(ht.all(ret00 == ht.array(a_torch @ b_torch)), 1)
        self.assertIsInstance(ret00, ht.DNDarray)
        self.assertEqual(ret00.shape, (n, k))
        self.assertEqual(ret00.dtype, ht.float)
        self.assertEqual(ret00.split, None)
        self.assertEqual(a.split, 0)
        self.assertEqual(b.split, None)

        if a.comm.size > 1:
            # splits 00
            a = ht.ones((n, m), split=0, dtype=ht.float64)
            b = ht.ones((j, k), split=0)
            a[0] = ht.arange(1, m + 1)
            a[:, -1] = ht.arange(1, n + 1)
            b[0] = ht.arange(1, k + 1)
            b[:, 0] = ht.arange(1, j + 1)
            ret00 = a @ b

            ret_comp00 = ht.array(a_torch @ b_torch, split=0)
            self.assertTrue(ht.equal(ret00, ret_comp00))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n, k))
            self.assertEqual(ret00.dtype, ht.float64)
            self.assertEqual(ret00.split, 0)

            # splits 00 (numpy)
            a = ht.array(np.ones((n, m)), split=0)
            b = ht.array(np.ones((j, k)), split=0)
            a[0] = ht.arange(1, m + 1)
            a[:, -1] = ht.arange(1, n + 1)
            b[0] = ht.arange(1, k + 1)
            b[:, 0] = ht.arange(1, j + 1)
            ret00 = a @ b

            ret_comp00 = ht.array(a_torch @ b_torch, split=0)
            self.assertTrue(ht.equal(ret00, ret_comp00))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n, k))
            self.assertEqual(ret00.dtype, ht.float64)
            self.assertEqual(ret00.split, 0)

            # splits 01
            a = ht.ones((n, m), split=0)
            b = ht.ones((j, k), split=1, dtype=ht.float64)
            a[0] = ht.arange(1, m + 1)
            a[:, -1] = ht.arange(1, n + 1)
            b[0] = ht.arange(1, k + 1)
            b[:, 0] = ht.arange(1, j + 1)
            ret00 = ht.matmul(a, b)

            ret_comp01 = ht.array(a_torch @ b_torch, split=0)
            self.assertTrue(ht.equal(ret00, ret_comp01))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n, k))
            self.assertEqual(ret00.dtype, ht.float64)
            self.assertEqual(ret00.split, 0)

            # splits 10
            a = ht.ones((n, m), split=1)
            b = ht.ones((j, k), split=0)
            a[0] = ht.arange(1, m + 1)
            a[:, -1] = ht.arange(1, n + 1)
            b[0] = ht.arange(1, k + 1)
            b[:, 0] = ht.arange(1, j + 1)
            ret00 = ht.matmul(a, b)

            ret_comp10 = ht.array(a_torch @ b_torch, split=1)
            self.assertTrue(ht.equal(ret00, ret_comp10))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n, k))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 1)

            # splits 11
            a = ht.ones((n, m), split=1)
            b = ht.ones((j, k), split=1)
            a[0] = ht.arange(1, m + 1)
            a[:, -1] = ht.arange(1, n + 1)
            b[0] = ht.arange(1, k + 1)
            b[:, 0] = ht.arange(1, j + 1)
            ret00 = ht.matmul(a, b)

            ret_comp11 = ht.array(a_torch @ b_torch, split=1)
            self.assertTrue(ht.equal(ret00, ret_comp11))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n, k))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 1)

            # splits 11 (torch)
            a = ht.array(torch.ones((n, m), device=self.device.torch_device), split=1)
            b = ht.array(torch.ones((j, k), device=self.device.torch_device), split=1)
            a[0] = ht.arange(1, m + 1)
            a[:, -1] = ht.arange(1, n + 1)
            b[0] = ht.arange(1, k + 1)
            b[:, 0] = ht.arange(1, j + 1)
            ret00 = ht.matmul(a, b)

            ret_comp11 = ht.array(a_torch @ b_torch, split=1)
            self.assertTrue(ht.equal(ret00, ret_comp11))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n, k))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 1)

            # splits 0 None
            a = ht.ones((n, m), split=0)
            b = ht.ones((j, k), split=None)
            a[0] = ht.arange(1, m + 1)
            a[:, -1] = ht.arange(1, n + 1)
            b[0] = ht.arange(1, k + 1)
            b[:, 0] = ht.arange(1, j + 1)
            ret00 = ht.matmul(a, b)

            ret_comp0 = ht.array(a_torch @ b_torch, split=0)
            self.assertTrue(ht.equal(ret00, ret_comp0))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n, k))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # splits 1 None
            a = ht.ones((n, m), split=1)
            b = ht.ones((j, k), split=None)
            a[0] = ht.arange(1, m + 1)
            a[:, -1] = ht.arange(1, n + 1)
            b[0] = ht.arange(1, k + 1)
            b[:, 0] = ht.arange(1, j + 1)
            ret00 = ht.matmul(a, b)

            ret_comp1 = ht.array(a_torch @ b_torch, split=1)
            self.assertTrue(ht.equal(ret00, ret_comp1))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n, k))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 1)

            # splits None 0
            a = ht.ones((n, m), split=None)
            b = ht.ones((j, k), split=0)
            a[0] = ht.arange(1, m + 1)
            a[:, -1] = ht.arange(1, n + 1)
            b[0] = ht.arange(1, k + 1)
            b[:, 0] = ht.arange(1, j + 1)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array(a_torch @ b_torch, split=0)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n, k))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # splits None 1
            a = ht.ones((n, m), split=None)
            b = ht.ones((j, k), split=1)
            a[0] = ht.arange(1, m + 1)
            a[:, -1] = ht.arange(1, n + 1)
            b[0] = ht.arange(1, k + 1)
            b[:, 0] = ht.arange(1, j + 1)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array(a_torch @ b_torch, split=1)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n, k))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 1)

            # vector matrix mult:
            # a -> vector
            a_torch = torch.ones((m), device=self.device.torch_device)
            b_torch = torch.ones((j, k), device=self.device.torch_device)
            b_torch[0] = torch.arange(1, k + 1, device=self.device.torch_device)
            b_torch[:, 0] = torch.arange(1, j + 1, device=self.device.torch_device)
            # splits None None
            a = ht.ones((m), split=None)
            b = ht.ones((j, k), split=None)
            b[0] = ht.arange(1, k + 1)
            b[:, 0] = ht.arange(1, j + 1)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array(a_torch @ b_torch, split=None)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (k,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, None)

            # splits None 0
            a = ht.ones((m), split=None)
            b = ht.ones((j, k), split=0)
            b[0] = ht.arange(1, k + 1)
            b[:, 0] = ht.arange(1, j + 1)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array(a_torch @ b_torch, split=None)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (k,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # splits None 1
            a = ht.ones((m), split=None)
            b = ht.ones((j, k), split=1)
            b[0] = ht.arange(1, k + 1)
            b[:, 0] = ht.arange(1, j + 1)
            ret00 = ht.matmul(a, b)
            ret_comp = ht.array(a_torch @ b_torch, split=0)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (k,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # splits 0 None
            a = ht.ones((m), split=None)
            b = ht.ones((j, k), split=0)
            b[0] = ht.arange(1, k + 1)
            b[:, 0] = ht.arange(1, j + 1)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array(a_torch @ b_torch, split=None)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (k,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # splits 0 0
            a = ht.ones((m), split=0)
            b = ht.ones((j, k), split=0)
            b[0] = ht.arange(1, k + 1)
            b[:, 0] = ht.arange(1, j + 1)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array(a_torch @ b_torch, split=None)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (k,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # splits 0 1
            a = ht.ones((m), split=0)
            b = ht.ones((j, k), split=1)
            b[0] = ht.arange(1, k + 1)
            b[:, 0] = ht.arange(1, j + 1)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array(a_torch @ b_torch, split=None)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (k,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # b -> vector
            a_torch = torch.ones((n, m), device=self.device.torch_device)
            a_torch[0] = torch.arange(1, m + 1, device=self.device.torch_device)
            a_torch[:, -1] = torch.arange(1, n + 1, device=self.device.torch_device)
            b_torch = torch.ones((j), device=self.device.torch_device)
            # splits None None
            a = ht.ones((n, m), split=None)
            b = ht.ones((j), split=None)
            a[0] = ht.arange(1, m + 1)
            a[:, -1] = ht.arange(1, n + 1)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array(a_torch @ b_torch, split=None)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, None)

            # splits 0 None
            a = ht.ones((n, m), split=0)
            b = ht.ones((j), split=None)
            a[0] = ht.arange(1, m + 1)
            a[:, -1] = ht.arange(1, n + 1)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array((a_torch @ b_torch), split=None)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # splits 1 None
            a = ht.ones((n, m), split=1)
            b = ht.ones((j), split=None)
            a[0] = ht.arange(1, m + 1)
            a[:, -1] = ht.arange(1, n + 1)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array((a_torch @ b_torch), split=None)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # splits None 0
            a = ht.ones((n, m), split=None)
            b = ht.ones((j), split=0)
            a[0] = ht.arange(1, m + 1)
            a[:, -1] = ht.arange(1, n + 1)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array((a_torch @ b_torch), split=None)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # splits 0 0
            a = ht.ones((n, m), split=0)
            b = ht.ones((j), split=0)
            a[0] = ht.arange(1, m + 1)
            a[:, -1] = ht.arange(1, n + 1)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array((a_torch @ b_torch), split=None)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            # splits 1 0
            a = ht.ones((n, m), split=1)
            b = ht.ones((j), split=0)
            a[0] = ht.arange(1, m + 1)
            a[:, -1] = ht.arange(1, n + 1)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array((a_torch @ b_torch), split=None)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n,))
            self.assertEqual(ret00.dtype, ht.float)
            self.assertEqual(ret00.split, 0)

            with self.assertRaises(NotImplementedError):
                a = ht.zeros((3, 3, 3), split=2)
                b = a.copy()
                a @ b

    def test_norm(self):
        a = ht.arange(9, dtype=ht.float32, split=0) - 4
        self.assertTrue(
            ht.allclose(ht.linalg.norm(a), ht.float32(np.linalg.norm(a.numpy())).item(), atol=1e-5)
        )
        a.resplit_(axis=None)
        self.assertTrue(
            ht.allclose(ht.linalg.norm(a), ht.float32(np.linalg.norm(a.numpy())).item(), atol=1e-5)
        )

        b = ht.array([[-4.0, -3.0, -2.0], [-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], split=0)
        self.assertTrue(
            ht.allclose(ht.linalg.norm(b), ht.float32(np.linalg.norm(b.numpy())).item(), atol=1e-5)
        )
        b.resplit_(axis=1)
        self.assertTrue(
            ht.allclose(ht.linalg.norm(b), ht.float32(np.linalg.norm(b.numpy())).item(), atol=1e-5)
        )

        with self.assertRaises(TypeError):
            c = np.arange(9) - 4
            ht.linalg.norm(c)

    def test_outer(self):
        # test outer, a and b local, different dtypes
        a = ht.arange(3, dtype=ht.int32)
        b = ht.arange(8, dtype=ht.float32)
        ht_outer = ht.outer(a, b, split=None)
        np_outer = np.outer(a.numpy(), b.numpy())
        t_outer = torch.einsum("i,j->ij", a._DNDarray__array, b._DNDarray__array)
        self.assertTrue((ht_outer.numpy() == np_outer).all())
        self.assertTrue(ht_outer._DNDarray__array.dtype is t_outer.dtype)

        # test outer, a and b distributed, no data on some ranks
        a_split = ht.arange(3, dtype=ht.float32, split=0)
        b_split = ht.arange(8, dtype=ht.float32, split=0)
        ht_outer_split = ht.outer(a_split, b_split, split=None)

        # a and b split 0, outer split 1
        ht_outer_split = ht.outer(a_split, b_split, split=1)
        self.assertTrue((ht_outer_split.numpy() == np_outer).all())
        self.assertTrue(ht_outer_split.split == 1)

        # a and b distributed, outer split unspecified
        ht_outer_split = ht.outer(a_split, b_split, split=None)
        self.assertTrue((ht_outer_split.numpy() == np_outer).all())
        self.assertTrue(ht_outer_split.split == 0)

        # a not distributed, outer.split = 1
        ht_outer_split = ht.outer(a, b_split, split=1)
        self.assertTrue((ht_outer_split.numpy() == np_outer).all())
        self.assertTrue(ht_outer_split.split == 1)

        # b not distributed, outer.split = 0
        ht_outer_split = ht.outer(a_split, b, split=0)
        self.assertTrue((ht_outer_split.numpy() == np_outer).all())
        self.assertTrue(ht_outer_split.split == 0)

        # a_split.ndim > 1 and a.split != 0
        a_split_3d = ht.random.randn(3, 3, 3, dtype=ht.float64, split=2)
        ht_outer_split = ht.outer(a_split_3d, b_split)
        np_outer_3d = np.outer(a_split_3d.numpy(), b_split.numpy())
        self.assertTrue((ht_outer_split.numpy() == np_outer_3d).all())
        self.assertTrue(ht_outer_split.split == 0)

        # write to out buffer
        ht_out = ht.empty((a.gshape[0], b.gshape[0]), dtype=ht.float32)
        ht.outer(a, b, out=ht_out)
        self.assertTrue((ht_out.numpy() == np_outer).all())
        ht_out_split = ht.empty((a_split.gshape[0], b_split.gshape[0]), dtype=ht.float32, split=1)
        ht.outer(a_split, b_split, out=ht_out_split, split=1)
        self.assertTrue((ht_out_split.numpy() == np_outer).all())

        # test exceptions
        t_a = torch.arange(3)
        with self.assertRaises(TypeError):
            ht.outer(t_a, b)
        np_b = np.arange(8)
        with self.assertRaises(TypeError):
            ht.outer(a, np_b)
        a_0d = ht.array(2.3)
        with self.assertRaises(RuntimeError):
            ht.outer(a_0d, b)
        t_out = torch.empty((a.gshape[0], b.gshape[0]), dtype=torch.float32)
        with self.assertRaises(TypeError):
            ht.outer(a, b, out=t_out)
        ht_out_wrong_dtype = ht.empty((a.gshape[0], b.gshape[0]), dtype=ht.float64)
        with self.assertRaises(TypeError):
            ht.outer(a, b, out=ht_out_wrong_dtype)
        ht_out_wrong_shape = ht.empty((7, b.gshape[0]), dtype=ht.float32)
        with self.assertRaises(ValueError):
            ht.outer(a, b, out=ht_out_wrong_shape)
        ht_out_wrong_split = ht.empty(
            (a_split.gshape[0], b_split.gshape[0]), dtype=ht.float32, split=1
        )
        with self.assertRaises(ValueError):
            ht.outer(a_split, b_split, out=ht_out_wrong_split, split=0)

    def test_projection(self):
        a = ht.arange(1, 4, dtype=ht.float32, split=None)
        e1 = ht.array([1, 0, 0], dtype=ht.float32, split=None)
        self.assertTrue(ht.equal(ht.linalg.projection(a, e1), e1))

        a.resplit_(axis=0)
        self.assertTrue(ht.equal(ht.linalg.projection(a, e1), e1))

        e2 = ht.array([0, 1, 0], dtype=ht.float32, split=0)
        self.assertTrue(ht.equal(ht.linalg.projection(a, e2), e2 * 2))

        a = ht.arange(1, 4, dtype=ht.float32, split=None)
        e3 = ht.array([0, 0, 1], dtype=ht.float32, split=0)
        self.assertTrue(ht.equal(ht.linalg.projection(a, e3), e3 * 3))

        a = np.arange(1, 4)
        with self.assertRaises(TypeError):
            ht.linalg.projection(a, e1)

        a = ht.array([[1], [2], [3]], dtype=ht.float32, split=None)
        with self.assertRaises(RuntimeError):
            ht.linalg.projection(a, e1)

    def test_transpose(self):
        # vector transpose, not distributed
        vector = ht.arange(10)
        vector_t = vector.T
        self.assertIsInstance(vector_t, ht.DNDarray)
        self.assertEqual(vector_t.dtype, ht.int32)
        self.assertEqual(vector_t.split, None)
        self.assertEqual(vector_t.shape, (10,))

        # simple matrix transpose, not distributed
        simple_matrix = ht.zeros((2, 4))
        simple_matrix_t = simple_matrix.transpose()
        self.assertIsInstance(simple_matrix_t, ht.DNDarray)
        self.assertEqual(simple_matrix_t.dtype, ht.float32)
        self.assertEqual(simple_matrix_t.split, None)
        self.assertEqual(simple_matrix_t.shape, (4, 2))
        self.assertEqual(simple_matrix_t._DNDarray__array.shape, (4, 2))

        # 4D array, not distributed, with given axis
        array_4d = ht.zeros((2, 3, 4, 5))
        array_4d_t = ht.transpose(array_4d, axes=(-1, 0, 2, 1))
        self.assertIsInstance(array_4d_t, ht.DNDarray)
        self.assertEqual(array_4d_t.dtype, ht.float32)
        self.assertEqual(array_4d_t.split, None)
        self.assertEqual(array_4d_t.shape, (5, 2, 4, 3))
        self.assertEqual(array_4d_t._DNDarray__array.shape, (5, 2, 4, 3))

        # vector transpose, distributed
        vector_split = ht.arange(10, split=0)
        vector_split_t = vector_split.T
        self.assertIsInstance(vector_split_t, ht.DNDarray)
        self.assertEqual(vector_split_t.dtype, ht.int32)
        self.assertEqual(vector_split_t.split, 0)
        self.assertEqual(vector_split_t.shape, (10,))
        self.assertLessEqual(vector_split_t.lshape[0], 10)

        # matrix transpose, distributed
        matrix_split = ht.ones((10, 20), split=1)
        matrix_split_t = matrix_split.transpose()
        self.assertIsInstance(matrix_split_t, ht.DNDarray)
        self.assertEqual(matrix_split_t.dtype, ht.float32)
        self.assertEqual(matrix_split_t.split, 0)
        self.assertEqual(matrix_split_t.shape, (20, 10))
        self.assertLessEqual(matrix_split_t.lshape[0], 20)
        self.assertEqual(matrix_split_t.lshape[1], 10)

        # 4D array, distributed
        array_4d_split = ht.ones((3, 4, 5, 6), split=3)
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
            ht.transpose(ht.zeros((2, 3)), axes=1.0)
        with self.assertRaises(ValueError):
            ht.transpose(ht.zeros((2, 3)), axes=(-1,))
        with self.assertRaises(TypeError):
            ht.zeros((2, 3)).transpose(axes="01")
        with self.assertRaises(TypeError):
            ht.zeros((2, 3)).transpose(axes=(0, 1.0))
        with self.assertRaises((ValueError, IndexError)):
            ht.zeros((2, 3)).transpose(axes=(0, 3))

    def test_tril(self):
        local_ones = ht.ones((5,))

        # 1D case, no offset, data is not split, module-level call
        result = ht.tril(local_ones)
        comparison = torch.ones((5, 5), device=self.device.torch_device).tril()
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.lshape, (5, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        # 1D case, positive offset, data is not split, module-level call
        result = ht.tril(local_ones, k=2)
        comparison = torch.ones((5, 5), device=self.device.torch_device).tril(diagonal=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.lshape, (5, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        # 1D case, negative offset, data is not split, module-level call
        result = ht.tril(local_ones, k=-2)
        comparison = torch.ones((5, 5), device=self.device.torch_device).tril(diagonal=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.lshape, (5, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        local_ones = ht.ones((4, 5))

        # 2D case, no offset, data is not split, method
        result = local_ones.tril()
        comparison = torch.ones((4, 5), device=self.device.torch_device).tril()
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        # 2D case, positive offset, data is not split, method
        result = local_ones.tril(k=2)
        comparison = torch.ones((4, 5), device=self.device.torch_device).tril(diagonal=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        # 2D case, negative offset, data is not split, method
        result = local_ones.tril(k=-2)
        comparison = torch.ones((4, 5), device=self.device.torch_device).tril(diagonal=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        local_ones = ht.ones((3, 4, 5, 6))

        # 2D+ case, no offset, data is not split, module-level call
        result = local_ones.tril()
        comparison = torch.ones((5, 6), device=self.device.torch_device).tril()
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (3, 4, 5, 6))
        self.assertEqual(result.lshape, (3, 4, 5, 6))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._DNDarray__array[i, j] == comparison).all())

        # 2D+ case, positive offset, data is not split, module-level call
        result = local_ones.tril(k=2)
        comparison = torch.ones((5, 6), device=self.device.torch_device).tril(diagonal=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (3, 4, 5, 6))
        self.assertEqual(result.lshape, (3, 4, 5, 6))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._DNDarray__array[i, j] == comparison).all())

        # # 2D+ case, negative offset, data is not split, module-level call
        result = local_ones.tril(k=-2)
        comparison = torch.ones((5, 6), device=self.device.torch_device).tril(diagonal=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (3, 4, 5, 6))
        self.assertEqual(result.lshape, (3, 4, 5, 6))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._DNDarray__array[i, j] == comparison).all())

        distributed_ones = ht.ones((5,), split=0)

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

        distributed_ones = ht.ones((4, 5), split=0)

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

        distributed_ones = ht.ones((4, 5), split=1)

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
        local_ones = ht.ones((5,))

        # 1D case, no offset, data is not split, module-level call
        result = ht.triu(local_ones)
        comparison = torch.ones((5, 5), device=self.device.torch_device).triu()
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.lshape, (5, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        # 1D case, positive offset, data is not split, module-level call
        result = ht.triu(local_ones, k=2)
        comparison = torch.ones((5, 5), device=self.device.torch_device).triu(diagonal=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.lshape, (5, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        # 1D case, negative offset, data is not split, module-level call
        result = ht.triu(local_ones, k=-2)
        comparison = torch.ones((5, 5), device=self.device.torch_device).triu(diagonal=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.lshape, (5, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        local_ones = ht.ones((4, 5))

        # 2D case, no offset, data is not split, method
        result = local_ones.triu()
        comparison = torch.ones((4, 5), device=self.device.torch_device).triu()
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        # 2D case, positive offset, data is not split, method
        result = local_ones.triu(k=2)
        comparison = torch.ones((4, 5), device=self.device.torch_device).triu(diagonal=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        # 2D case, negative offset, data is not split, method
        result = local_ones.triu(k=-2)
        comparison = torch.ones((4, 5), device=self.device.torch_device).triu(diagonal=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == comparison).all())

        local_ones = ht.ones((3, 4, 5, 6))

        # 2D+ case, no offset, data is not split, module-level call
        result = local_ones.triu()
        comparison = torch.ones((5, 6), device=self.device.torch_device).triu()
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (3, 4, 5, 6))
        self.assertEqual(result.lshape, (3, 4, 5, 6))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._DNDarray__array[i, j] == comparison).all())

        # 2D+ case, positive offset, data is not split, module-level call
        result = local_ones.triu(k=2)
        comparison = torch.ones((5, 6), device=self.device.torch_device).triu(diagonal=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (3, 4, 5, 6))
        self.assertEqual(result.lshape, (3, 4, 5, 6))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._DNDarray__array[i, j] == comparison).all())

        # # 2D+ case, negative offset, data is not split, module-level call
        result = local_ones.triu(k=-2)
        comparison = torch.ones((5, 6), device=self.device.torch_device).triu(diagonal=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (3, 4, 5, 6))
        self.assertEqual(result.lshape, (3, 4, 5, 6))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._DNDarray__array[i, j] == comparison).all())

        distributed_ones = ht.ones((5,), split=0)

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

        distributed_ones = ht.ones((4, 5), split=0)

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

        distributed_ones = ht.ones((4, 5), split=1)

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
