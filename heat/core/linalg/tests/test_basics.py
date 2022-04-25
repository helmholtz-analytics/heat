from typing import Type
import torch
import os
import unittest
import heat as ht
import numpy as np

from ...tests.test_suites.basic_test import TestCase


class TestLinalgBasics(TestCase):
    def test_cholesky(self):
        # batch of symmetric positive-definite matrices
        A = torch.randn(3, 2, 2, dtype=torch.float64, device=self.device.torch_device)
        A = A @ A.transpose(-2, -1) + torch.eye(2, device=self.device.torch_device)
        if int(torch.__version__.split(".")[1]) < 8:
            L = torch.cholesky(A)
        else:
            L = torch.linalg.cholesky(A)

        A_ht = ht.asarray(A)
        L_ht = ht.linalg.cholesky(A_ht)
        self.assertTrue(torch.allclose(L_ht.larray, L))

        A_ht0 = ht.resplit(A_ht, 0)
        L_ht0 = ht.linalg.cholesky(A_ht0)
        self.assertTrue(ht.allclose(L_ht0, ht.resplit(L_ht, 0)))

        A_ht1 = ht.resplit(A_ht, 1)
        L_ht1 = ht.linalg.cholesky(A_ht1)
        self.assertTrue(ht.allclose(L_ht1, ht.resplit(L_ht, 1)))

        A_ht2 = ht.resplit(A_ht, 2)
        L_ht2 = ht.linalg.cholesky(A_ht2)
        self.assertTrue(ht.allclose(L_ht2, ht.resplit(L_ht, 2)))

        # single matrix
        A = torch.randn(8, 8, dtype=torch.float32, device=self.device.torch_device)
        A = A @ A.transpose(-2, -1) + torch.eye(8, device=self.device.torch_device)
        if int(torch.__version__.split(".")[1]) < 10:
            if int(torch.__version__.split(".")[1]) < 8:
                L = torch.cholesky(A)
            else:
                L = torch.linalg.cholesky(A)
            L = L.T.conj()
        else:
            L = torch.linalg.cholesky(A, upper=True)

        A_ht = ht.asarray(A)
        L_ht = ht.linalg.cholesky(A_ht, True)
        self.assertTrue(torch.allclose(L_ht.larray, L))

        A_ht0 = ht.resplit(A_ht, 0)
        L_ht0 = ht.linalg.cholesky(A_ht0, True)
        self.assertTrue(ht.allclose(L_ht0, ht.resplit(L_ht, 0)))

        A_ht1 = ht.resplit(A_ht, 1)
        L_ht1 = ht.linalg.cholesky(A_ht1, True)
        self.assertTrue(ht.allclose(L_ht1, ht.resplit(L_ht, 1)))

        with self.assertRaises(RuntimeError):
            ht.linalg.cholesky(ht.array([1, 2, 3], split=0))
        with self.assertRaises(RuntimeError):
            ht.linalg.cholesky(ht.zeros((2, 3, 2), split=1))
        with self.assertRaises(NotImplementedError):
            ht.linalg.cholesky(ht.zeros((2, 2), dtype=ht.int, split=0))
        with self.assertRaises(NotImplementedError):
            ht.linalg.cholesky(ht.zeros((2, 2), dtype=ht.complex64, split=0))

    def test_cross(self):
        a = ht.eye(3)
        b = ht.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

        # different types
        cross = ht.cross(a, b)
        self.assertEqual(cross.shape, a.shape)
        self.assertEqual(cross.dtype, a.dtype)
        self.assertEqual(cross.split, a.split)
        self.assertEqual(cross.comm, a.comm)
        self.assertEqual(cross.device, a.device)
        self.assertTrue(ht.equal(cross, ht.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])))

        # axis
        a = ht.eye(3, split=0)
        b = ht.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=ht.float, split=0)

        cross = ht.cross(a, b)
        self.assertEqual(cross.shape, a.shape)
        self.assertEqual(cross.dtype, a.dtype)
        self.assertEqual(cross.split, a.split)
        self.assertEqual(cross.comm, a.comm)
        self.assertEqual(cross.device, a.device)
        self.assertTrue(ht.equal(cross, ht.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])))

        a = ht.eye(3, dtype=ht.int8, split=1)
        b = ht.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=ht.int8, split=1)

        cross = ht.cross(a, b, axis=0)
        self.assertEqual(cross.shape, a.shape)
        self.assertEqual(cross.dtype, a.dtype)
        self.assertEqual(cross.split, a.split)
        self.assertEqual(cross.comm, a.comm)
        self.assertEqual(cross.device, a.device)
        self.assertTrue(ht.equal(cross, ht.array([[0, 0, -1], [-1, 0, 0], [0, -1, 0]])))

        # test axisa, axisb, axisc
        np.random.seed(42)
        np_a = np.random.randn(40, 3, 50)
        np_b = np.random.randn(3, 40, 50)
        np_cross = np.cross(np_a, np_b, axisa=1, axisb=0)

        a = ht.array(np_a, split=0)
        b = ht.array(np_b, split=1)
        cross = ht.cross(a, b, axisa=1, axisb=0)
        self.assert_array_equal(cross, np_cross)

        cross_axisc = ht.cross(a, b, axisa=1, axisb=0, axisc=1)
        np_cross_axisc = np.cross(np_a, np_b, axisa=1, axisb=0, axisc=1)
        self.assert_array_equal(cross_axisc, np_cross_axisc)

        # test vector axes with 2 elements
        b_2d = ht.array(np_b[:-1, :, :], split=1)
        cross_3d_2d = ht.cross(a, b_2d, axisa=1, axisb=0)
        np_cross_3d_2d = np.cross(np_a, np_b[:-1, :, :], axisa=1, axisb=0)
        self.assert_array_equal(cross_3d_2d, np_cross_3d_2d)

        a_2d = ht.array(np_a[:, :-1, :], split=0)
        cross_2d_3d = ht.cross(a_2d, b, axisa=1, axisb=0)
        np_cross_2d_3d = np.cross(np_a[:, :-1, :], np_b, axisa=1, axisb=0)
        self.assert_array_equal(cross_2d_3d, np_cross_2d_3d)

        cross_z_comp = ht.cross(a_2d, b_2d, axisa=1, axisb=0)
        np_cross_z_comp = np.cross(np_a[:, :-1, :], np_b[:-1, :, :], axisa=1, axisb=0)
        self.assert_array_equal(cross_z_comp, np_cross_z_comp)

        a_wrong_split = ht.array(np_a[:, :-1, :], split=2)
        with self.assertRaises(ValueError):
            ht.cross(a_wrong_split, b, axisa=1, axisb=0)
        with self.assertRaises(ValueError):
            ht.cross(ht.eye(3), ht.eye(4))
        with self.assertRaises(ValueError):
            ht.cross(ht.eye(3, split=0), ht.eye(3, split=1))
        if torch.cuda.is_available():
            with self.assertRaises(ValueError):
                ht.cross(ht.eye(3, device="gpu"), ht.eye(3, device="cpu"))
        with self.assertRaises(TypeError):
            ht.cross(ht.eye(3), ht.eye(3), axis="wasd")
        with self.assertRaises(ValueError):
            ht.cross(ht.eye(3, split=0), ht.eye(3, split=0), axis=0)

    def test_det(self):
        # (3,3) with pivoting
        ares = ht.array(54.0)
        a = ht.array([[-2.0, -1, 2], [2, 1, 4], [-3, 3, -1]], split=0, dtype=ht.double)
        adet = ht.linalg.det(a)

        self.assertTupleEqual(adet.shape, ares.shape)
        self.assertIsNone(adet.split)
        self.assertEqual(adet.dtype, a.dtype)
        self.assertEqual(adet.device, a.device)
        self.assertTrue(ht.equal(adet, ares))

        a = ht.array([[-2.0, -1, 2], [2, 1, 4], [-3, 3, -1]], split=1, dtype=ht.double)
        adet = ht.linalg.det(a)

        self.assertTupleEqual(adet.shape, ares.shape)
        self.assertIsNone(adet.split)
        self.assertEqual(adet.dtype, a.dtype)
        self.assertEqual(adet.device, a.device)
        self.assertTrue(ht.equal(adet, ares))

        # det==0
        ares = ht.array(0.0)
        a = ht.array([[0, 0, 0], [2, 1, 4], [-3, 3, -1]], dtype=ht.float64, split=0)
        adet = ht.linalg.det(a)

        self.assertTupleEqual(adet.shape, ares.shape)
        self.assertIsNone(adet.split)
        self.assertEqual(adet.dtype, a.dtype)
        self.assertEqual(adet.device, a.device)
        self.assertTrue(ht.equal(adet, ares))

        # (3,2,2)
        ares = ht.array([-2.0, -3.0, -8.0])

        a = ht.array([[[1.0, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]])
        adet = ht.linalg.det(a)

        self.assertTupleEqual(adet.shape, ares.shape)
        self.assertIsNone(adet.split)
        self.assertEqual(adet.dtype, a.dtype)
        self.assertEqual(adet.device, a.device)
        self.assertTrue(ht.allclose(adet, ares))

        a = ht.array([[[1.0, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]], split=0)
        adet = ht.linalg.det(a)

        self.assertTupleEqual(adet.shape, ares.shape)
        self.assertEqual(adet.split, a.split if a.is_distributed() else None)
        self.assertEqual(adet.dtype, a.dtype)
        self.assertEqual(adet.device, a.device)
        self.assertTrue(ht.allclose(adet, ares))

        a = ht.array([[[1.0, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]], split=1)
        adet = ht.linalg.det(a)

        self.assertTupleEqual(adet.shape, ares.shape)
        self.assertIsNone(adet.split)
        self.assertEqual(adet.dtype, a.dtype)
        self.assertEqual(adet.device, a.device)
        self.assertTrue(ht.allclose(adet, ares))

        a = ht.array([[[1.0, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]], split=2)
        adet = ht.linalg.det(a)

        self.assertTupleEqual(adet.shape, ares.shape)
        self.assertIsNone(adet.split)
        self.assertEqual(adet.dtype, a.dtype)
        self.assertEqual(adet.device, a.device)
        self.assertTrue(ht.allclose(adet, ares))

        with self.assertRaises(RuntimeError):
            ht.linalg.det(ht.array([1, 2, 3], split=0))
        with self.assertRaises(RuntimeError):
            ht.linalg.det(ht.zeros((2, 2, 3), split=1))
        with self.assertRaises(RuntimeError):
            ht.linalg.det(ht.zeros((2, 2), dtype=ht.int, split=0))

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

    def test_inv(self):
        # single 2D array
        # pytorch
        ares = ht.array([[2.0, 2, 1], [3, 4, 1], [0, 1, -1]])

        a = ht.array([[5.0, -3, 2], [-3, 2, -1], [-3, 2, -2]])
        ainv = ht.linalg.inv(a)

        self.assertEqual(ainv.split, a.split)
        self.assertEqual(ainv.device, a.device)
        self.assertTupleEqual(ainv.shape, a.shape)
        self.assertTrue(ht.allclose(ainv, ares))

        # distributed
        a = ht.array([[5.0, -3, 2], [-3, 2, -1], [-3, 2, -2]], split=0)
        ainv = ht.linalg.inv(a)
        self.assertEqual(ainv.split, a.split)
        self.assertEqual(ainv.device, a.device)
        self.assertTupleEqual(ainv.shape, a.shape)
        self.assertTrue(ht.allclose(ainv, ares))

        a = ht.array([[5.0, -3, 2], [-3, 2, -1], [-3, 2, -2]], split=1)
        ainv = ht.linalg.inv(a)
        self.assertEqual(ainv.split, a.split)
        self.assertEqual(ainv.device, a.device)
        self.assertTupleEqual(ainv.shape, a.shape)
        self.assertTrue(ht.allclose(ainv, ares))

        # array Size=(2,2,2,2)
        ares = ht.array(
            [[[2, -0.5], [-3, 1]], [[-3, 2], [2, -1]], [[-3, 2], [2, -1]], [[2, -0.5], [-3, 1]]],
            dtype=ht.float,
        )

        a = ht.array(
            [[[2, 1], [6, 4]], [[1, 2], [2, 3]], [[1, 2], [2, 3]], [[2, 1], [6, 4]]],
            dtype=ht.float,
            split=1,
        )
        ainv = ht.linalg.inv(a)
        self.assertEqual(ainv.split, a.split)
        self.assertEqual(ainv.device, a.device)
        self.assertTupleEqual(ainv.shape, a.shape)
        self.assertTrue(ht.allclose(ainv, ares))

        a = ht.array(
            [[[2, 1], [6, 4]], [[1, 2], [2, 3]], [[1, 2], [2, 3]], [[2, 1], [6, 4]]],
            dtype=ht.float,
            split=2,
        )
        ainv = ht.linalg.inv(a)
        self.assertEqual(ainv.split, a.split)
        self.assertEqual(ainv.device, a.device)
        self.assertTupleEqual(ainv.shape, a.shape)
        self.assertTrue(ht.allclose(ainv, ares))

        # pivoting row change
        ares = ht.array([[-1, 0, 2], [2, 0, -1], [-6, 3, 0]], dtype=ht.double) / 3.0
        a = ht.array([[1, 2, 0], [2, 4, 1], [2, 1, 0]], dtype=ht.double, split=0)
        ainv = ht.linalg.inv(a)
        self.assertEqual(ainv.split, a.split)
        self.assertEqual(ainv.device, a.device)
        self.assertTupleEqual(ainv.shape, a.shape)
        self.assertTrue(ht.allclose(ainv, ares))

        a = ht.array([[1, 2, 0], [2, 4, 1], [2, 1, 0]], dtype=ht.double, split=1)
        ainv = ht.linalg.inv(a)
        self.assertEqual(ainv.split, a.split)
        self.assertEqual(ainv.device, a.device)
        self.assertTupleEqual(ainv.shape, a.shape)
        self.assertTrue(ht.allclose(ainv, ares))

        ht.random.seed(42)
        a = ht.random.random((20, 20), dtype=ht.float64, split=1)
        ainv = ht.linalg.inv(a)
        i = ht.eye(a.shape, split=1, dtype=a.dtype)
        self.assertTrue(ht.allclose(a @ ainv, i))

        # ht.random.seed(42)
        # a = ht.random.random((20, 20), dtype=ht.float64, split=0)
        # ainv = ht.linalg.inv(a)
        # i = ht.eye(a.shape, split=0, dtype=a.dtype)
        # self.assertTrue(ht.allclose(a @ ainv, i))

        with self.assertRaises(RuntimeError):
            ht.linalg.inv(ht.array([1, 2, 3], split=0))
        with self.assertRaises(RuntimeError):
            ht.linalg.inv(ht.zeros((1, 2, 3), split=1))
        with self.assertRaises(RuntimeError):
            ht.linalg.inv(ht.zeros((2, 2), dtype=ht.int, split=1))
        with self.assertRaises(RuntimeError):
            ht.linalg.inv(ht.zeros((3, 3), split=0))
        with self.assertRaises(RuntimeError):
            ht.linalg.inv(ht.ones((3, 3), split=1))

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

            a = ht.ones((n, m), split=None, dtype=ht.int64)
            b = ht.ones((j), split=None, dtype=ht.int64)
            a[0] = ht.arange(1, m + 1, dtype=ht.int64)
            a[:, -1] = ht.arange(1, n + 1, dtype=ht.int64)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array((a_torch @ b_torch), split=None)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n,))
            self.assertEqual(ret00.dtype, ht.int64)
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

            a = ht.ones((n, m), split=0, dtype=ht.int64)
            b = ht.ones((j), split=None, dtype=ht.int64)
            a[0] = ht.arange(1, m + 1, dtype=ht.int64)
            a[:, -1] = ht.arange(1, n + 1, dtype=ht.int64)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array((a_torch @ b_torch), split=None)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n,))
            self.assertEqual(ret00.dtype, ht.int64)
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

            a = ht.ones((n, m), split=1, dtype=ht.int64)
            b = ht.ones((j), split=None, dtype=ht.int64)
            a[0] = ht.arange(1, m + 1, dtype=ht.int64)
            a[:, -1] = ht.arange(1, n + 1, dtype=ht.int64)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array((a_torch @ b_torch), split=None)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n,))
            self.assertEqual(ret00.dtype, ht.int64)
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

            a = ht.ones((n, m), split=None, dtype=ht.int64)
            b = ht.ones((j), split=0, dtype=ht.int64)
            a[0] = ht.arange(1, m + 1, dtype=ht.int64)
            a[:, -1] = ht.arange(1, n + 1, dtype=ht.int64)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array((a_torch @ b_torch), split=None)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n,))
            self.assertEqual(ret00.dtype, ht.int64)
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

            a = ht.ones((n, m), split=0, dtype=ht.int64)
            b = ht.ones((j), split=0, dtype=ht.int64)
            a[0] = ht.arange(1, m + 1, dtype=ht.int64)
            a[:, -1] = ht.arange(1, n + 1, dtype=ht.int64)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array((a_torch @ b_torch), split=None)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n,))
            self.assertEqual(ret00.dtype, ht.int64)
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

            a = ht.ones((n, m), split=1, dtype=ht.int64)
            b = ht.ones((j), split=0, dtype=ht.int64)
            a[0] = ht.arange(1, m + 1, dtype=ht.int64)
            a[:, -1] = ht.arange(1, n + 1, dtype=ht.int64)
            ret00 = ht.matmul(a, b)

            ret_comp = ht.array((a_torch @ b_torch), split=None)
            self.assertTrue(ht.equal(ret00, ret_comp))
            self.assertIsInstance(ret00, ht.DNDarray)
            self.assertEqual(ret00.shape, (n,))
            self.assertEqual(ret00.dtype, ht.int64)
            self.assertEqual(ret00.split, 0)

            with self.assertRaises(NotImplementedError):
                a = ht.zeros((3, 3, 3), split=2)
                b = a.copy()
                a @ b

    def test_matrix_norm(self):
        a = ht.arange(9, dtype=ht.float) - 4
        b = a.reshape((3, 3))
        b0 = a.reshape((3, 3), new_split=0)
        b1 = a.reshape((3, 3), new_split=1)

        # different ord
        mn = ht.linalg.matrix_norm(b, ord="fro")
        self.assertEqual(mn.split, b.split)
        self.assertEqual(mn.dtype, b.dtype)
        self.assertEqual(mn.device, b.device)
        self.assertTrue(ht.allclose(mn, ht.array(7.745966692414834)))

        mn = ht.linalg.matrix_norm(b0, ord=1)
        self.assertEqual(mn.split, b.split)
        self.assertEqual(mn.dtype, b.dtype)
        self.assertEqual(mn.device, b.device)
        self.assertEqual(mn.item(), 7.0)

        mn = ht.linalg.matrix_norm(b0, ord=-1)
        self.assertEqual(mn.split, b.split)
        self.assertEqual(mn.dtype, b.dtype)
        self.assertEqual(mn.device, b.device)
        self.assertEqual(mn.item(), 6.0)

        mn = ht.linalg.matrix_norm(b1)
        self.assertEqual(mn.split, b.split)
        self.assertEqual(mn.dtype, b.dtype)
        self.assertEqual(mn.device, b.device)
        self.assertTrue(ht.allclose(mn, ht.array(7.745966692414834)))

        # higher dimension + different dtype
        m = ht.arange(8).reshape(2, 2, 2)
        mn = ht.linalg.matrix_norm(m, axis=(2, 1), ord=ht.inf)
        self.assertEqual(mn.split, m.split)
        self.assertEqual(mn.dtype, ht.float)
        self.assertEqual(mn.device, m.device)
        self.assertTrue(ht.equal(mn, ht.array([4.0, 12.0])))

        mn = ht.linalg.matrix_norm(m, axis=(2, 1), ord=-ht.inf)
        self.assertEqual(mn.split, m.split)
        self.assertEqual(mn.dtype, ht.float)
        self.assertEqual(mn.device, m.device)
        self.assertTrue(ht.equal(mn, ht.array([2.0, 10.0])))

        # too many axis to infer
        with self.assertRaises(ValueError):
            ht.linalg.matrix_norm(ht.ones((2, 2, 2)))
        # bad axis
        with self.assertRaises(TypeError):
            ht.linalg.matrix_norm(ht.ones((2, 2)), axis=1)
        with self.assertRaises(TypeError):
            ht.linalg.matrix_norm(ht.ones(2, 2), axis=(1, 2, 3))
        # bad array
        with self.assertRaises(ValueError):
            ht.linalg.matrix_norm(ht.array([1, 2, 3]))
        # bad ord
        with self.assertRaises(ValueError):
            ht.linalg.matrix_norm(ht.ones((2, 2)), ord=3)
        # Not implemented yet; SVD needed
        with self.assertRaises(NotImplementedError):
            ht.linalg.matrix_norm(ht.ones((2, 2)), ord=2)
        with self.assertRaises(NotImplementedError):
            ht.linalg.matrix_norm(ht.ones((2, 2)), ord=-2)
        with self.assertRaises(NotImplementedError):
            ht.linalg.matrix_norm(ht.ones((2, 2)), ord="nuc")

    def test_norm(self):
        a = ht.arange(9, dtype=ht.float) - 4
        a0 = ht.array([1 + 1j, 2 - 2j, 0 + 1j, 2 + 1j], dtype=ht.complex64, split=0)
        b = a.reshape((3, 3))
        b0 = a.reshape((3, 3), new_split=0)
        b1 = a.reshape((3, 3), new_split=1)

        # vectors
        gn = ht.linalg.norm(a, axis=0, ord=1)
        self.assertEqual(gn.split, a.split)
        self.assertEqual(gn.dtype, a.dtype)
        self.assertEqual(gn.device, a.device)
        self.assertEqual(gn.item(), 20.0)

        # complex type
        gn = ht.linalg.norm(a0, keepdims=True)
        self.assertEqual(gn.split, None)
        self.assertEqual(gn.dtype, ht.float)
        self.assertEqual(gn.device, a0.device)
        self.assertEqual(gn.item(), 4.0)

        # matrices
        gn = ht.linalg.norm(b, ord="fro")
        self.assertEqual(gn.split, None)
        self.assertEqual(gn.dtype, b.dtype)
        self.assertEqual(gn.device, b.device)
        self.assertTrue(ht.allclose(gn, ht.array(7.745966692414834)))

        gn = ht.linalg.norm(b0, ord=ht.inf)
        self.assertEqual(gn.split, None)
        self.assertEqual(gn.dtype, b0.dtype)
        self.assertEqual(gn.device, b0.device)
        self.assertEqual(gn.item(), 9.0)

        gn = ht.linalg.norm(b1, axis=(0,), ord=-ht.inf, keepdims=True)
        self.assertEqual(gn.split, b1.split)
        self.assertEqual(gn.dtype, b1.dtype)
        self.assertEqual(gn.device, b1.device)
        self.assertTrue(ht.equal(gn, ht.array([[1.0, 0.0, 1.0]])))

        # higher dimension + different dtype
        gn = ht.linalg.norm(ht.ones((3, 3, 3), dtype=ht.int), axis=(-2, -1))
        self.assertEqual(gn.split, None)
        self.assertEqual(gn.dtype, ht.float)
        self.assertTrue(ht.equal(gn, ht.array([3.0, 3.0, 3.0])))

        # bad axis
        with self.assertRaises(ValueError):
            ht.linalg.norm(ht.ones(2), axis=(0, 1, 2))

    def test_outer(self):
        # test outer, a and b local, different dtypes
        a = ht.arange(3, dtype=ht.int32)
        b = ht.arange(8, dtype=ht.float32)
        ht_outer = ht.outer(a, b, split=None)
        np_outer = np.outer(a.numpy(), b.numpy())
        t_outer = torch.einsum("i,j->ij", a.larray, b.larray)
        self.assertTrue((ht_outer.numpy() == np_outer).all())
        self.assertTrue(ht_outer.larray.dtype is t_outer.dtype)

        # test outer, a and b distributed, no data on some ranks
        a_split = ht.arange(3, dtype=ht.float32, split=0)
        b_split = ht.arange(8, dtype=ht.float32, split=0)
        ht_outer_split = ht.outer(a_split, b_split, split=None)

        # a and b split 0, outer split 1
        ht_outer_split = ht.outer(a_split, b_split, split=1)
        self.assertTrue(ht_outer_split.split == 1)
        self.assertTrue((ht_outer_split.numpy() == np_outer).all())

        # a and b distributed, outer split unspecified
        ht_outer_split = ht.outer(a_split, b_split, split=None)
        self.assertTrue(ht_outer_split.split == 0)
        self.assertTrue((ht_outer_split.numpy() == np_outer).all())

        # a not distributed, outer.split = 1
        ht_outer_split = ht.outer(a, b_split, split=1)
        self.assertTrue(ht_outer_split.split == 1)
        self.assertTrue((ht_outer_split.numpy() == np_outer).all())

        # b not distributed, outer.split = 0
        ht_outer_split = ht.outer(a_split, b, split=0)
        self.assertTrue(ht_outer_split.split == 0)
        self.assertTrue((ht_outer_split.numpy() == np_outer).all())

        # a_split.ndim > 1 and a.split != 0
        a_split_3d = ht.random.randn(3, 3, 3, dtype=ht.float64, split=2)
        ht_outer_split = ht.outer(a_split_3d, b_split)
        np_outer_3d = np.outer(a_split_3d.numpy(), b_split.numpy())
        self.assertTrue(ht_outer_split.split == 0)
        self.assertTrue((ht_outer_split.numpy() == np_outer_3d).all())

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

    def test_trace(self):
        # ------------------------------------------------
        # UNDISTRIBUTED CASE
        # ------------------------------------------------
        # CASE 2-D
        # ------------------------------------------------
        x = ht.arange(24).reshape((6, 4))
        x_np = x.numpy()
        dtype = ht.float32

        result = ht.trace(x)
        result_np = np.trace(x_np)
        self.assertIsInstance(result, int)
        self.assertEqual(result, result_np)

        # direct call
        result = x.trace()
        self.assertIsInstance(result, int)
        self.assertEqual(result, result_np)

        # input = array_like (other than DNDarray)
        result = ht.trace(x.tolist())
        self.assertIsInstance(result, int)
        self.assertEqual(result, result_np)

        # dtype
        result = ht.trace(x, dtype=dtype)
        result_np = np.trace(x_np, dtype=np.float32)
        self.assertIsInstance(result, float)
        self.assertEqual(result, result_np)

        # offset != 0
        # negative offset
        o = -(x.gshape[0] - 1)
        result = ht.trace(x, offset=o)
        result_np = np.trace(x_np, offset=o)
        self.assertIsInstance(result, int)
        self.assertEqual(result, result_np)

        # positive offset
        o = x.gshape[1] - 1
        result = ht.trace(x, offset=o)
        result_np = np.trace(x_np, offset=o)
        self.assertIsInstance(result, int)
        self.assertEqual(result, result_np)

        # offset resulting into empty array
        # negative
        o = -x.gshape[0]
        result = ht.trace(x, offset=o)
        result_np = np.trace(x_np, offset=o)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 0)
        self.assertEqual(result, result_np)

        # positive
        o = x.gshape[1]
        result = ht.trace(x, offset=o)
        result_np = np.trace(x_np, offset=o)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 0)
        self.assertEqual(result, result_np)

        # Exceptions
        with self.assertRaises(TypeError):
            x = "[[1, 2], [3, 4]]"
            ht.trace(x)
        with self.assertRaises(ValueError):
            x = ht.arange(24)
            ht.trace(x)
        with self.assertRaises(TypeError):
            x = ht.arange(24).reshape((6, 4))
            ht.trace(x, axis1=0.2)
        with self.assertRaises(TypeError):
            ht.trace(x, axis2=1.4)
        with self.assertRaises(ValueError):
            ht.trace(x, axis1=2)
        with self.assertRaises(ValueError):
            ht.trace(x, axis2=2)
        with self.assertRaises(TypeError):
            ht.trace(x, offset=1.2)
        with self.assertRaises(ValueError):
            ht.trace(x, axis1=1, axis2=1)
        with self.assertRaises(ValueError):
            ht.trace(x, dtype="ht.int64")
        with self.assertRaises(TypeError):
            ht.trace(x, out=[])
        with self.assertRaises(ValueError):
            # As result is scalar
            out = ht.array([])
            ht.trace(x, out=out)
        with self.assertRaises(ValueError):
            ht.trace(x, dtype="ht.float32")

        # ------------------------------------------------
        # CASE > 2-D (4D)
        # ------------------------------------------------
        x = ht.arange(24).reshape((1, 2, 3, 4))
        x_np = x.numpy()
        out = ht.empty((3, 4))
        axis1 = 1
        axis2 = 3

        result = ht.trace(x)
        result_np = np.trace(x_np)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # input = array_like (other than DNDarray)
        result = ht.trace(x.tolist())
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # out
        result = ht.trace(x, out=out)
        result_np = np.trace(x_np)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)
        self.assert_array_equal(out, result_np)

        result = ht.trace(x, axis1=axis1, axis2=axis2)
        result_np = np.trace(x_np, axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # reversed axes order
        result = ht.trace(x, axis1=axis2, axis2=axis1)
        result_np = np.trace(x_np, axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # negative axes
        axis1 = 1
        axis2 = 2
        result = ht.trace(x, axis1=axis1, axis2=-axis2)
        result_np = np.trace(x_np, axis1=axis1, axis2=-axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        result = ht.trace(x, axis1=-axis1, axis2=axis2)
        result_np = np.trace(x_np, axis1=-axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        result = ht.trace(x, axis1=-axis1, axis2=-axis2)
        result_np = np.trace(x_np, axis1=-axis1, axis2=-axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # different axes
        axis1 = 1
        axis2 = 2
        o = 0
        result = ht.trace(x, offset=o, axis1=axis1, axis2=axis2, dtype=dtype)
        result_np = np.trace(x_np, offset=o, axis1=axis1, axis2=axis2, dtype=np.float32)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # offset != 0
        # negative offset
        o = -(x.gshape[0] - 1)
        result = ht.trace(x, offset=o)
        result_np = np.trace(x_np, offset=o)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # positive offset
        o = x.gshape[1] - 1
        result = ht.trace(x, offset=o)
        result_np = np.trace(x_np, offset=o)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # offset resulting into zero array
        axis1 = 1
        axis2 = 2
        # negative
        o = -x.gshape[axis1]
        result = ht.trace(x, offset=o, axis1=axis1, axis2=axis2)
        result_np = np.trace(x_np, offset=o, axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, np.zeros((1, 4)))
        self.assert_array_equal(result, result_np)

        # positive
        o = x.gshape[axis2]
        result = ht.trace(x, offset=o, axis1=axis1, axis2=axis2)
        result_np = np.trace(x_np, offset=o, axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, np.zeros((1, 4)))
        self.assert_array_equal(result, result_np)

        # Exceptions
        with self.assertRaises(ValueError):
            out = ht.array([])
            ht.trace(x, out=out)

        # ------------------------------------------------
        # DISTRIBUTED CASE
        # ------------------------------------------------
        # CASE 2-D
        # ------------------------------------------------
        x = ht.arange(24, split=0).reshape((6, 4))
        x_np = np.arange(24).reshape((6, 4))
        dtype = ht.float32

        result = ht.trace(x)
        result_np = np.trace(x_np)
        self.assertIsInstance(result, int)
        self.assertEqual(result, result_np)

        # different split axis
        x_2 = ht.array(torch.arange(24).reshape((6, 4)), split=1)
        result = ht.trace(x_2)
        result_np = np.trace(x_np)
        self.assertIsInstance(result, int)
        self.assertEqual(result, result_np)

        # input = array_like (other than DNDarray)
        result = ht.trace(x.tolist())
        self.assertIsInstance(result, int)
        self.assertEqual(result, result_np)

        # dtype
        result = ht.trace(x, dtype=dtype)
        result_np = np.trace(x_np, dtype=np.float32)
        self.assertIsInstance(result, float)
        self.assertEqual(result, result_np)

        # offset != 0
        # negative offset
        o = -(x.gshape[0] - 1)
        result = ht.trace(x, offset=o)
        result_np = np.trace(x_np, offset=o)
        self.assertIsInstance(result, int)
        self.assertEqual(result, result_np)

        # positive offset
        o = x.gshape[1] - 1
        result = ht.trace(x, offset=o)
        result_np = np.trace(x_np, offset=o)
        self.assertIsInstance(result, int)
        self.assertEqual(result, result_np)

        # offset resulting into empty array
        # negative
        o = -x.gshape[0]
        result = ht.trace(x, offset=o)
        result_np = np.trace(x_np, offset=o)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 0)
        self.assertEqual(result, result_np)

        # positive
        o = x.gshape[1]
        result = ht.trace(x, offset=o)
        result_np = np.trace(x_np, offset=o)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 0)
        self.assertEqual(result, result_np)

        # Exceptions
        with self.assertRaises(TypeError):
            x = "[[1, 2], [3, 4]]"
            ht.trace(x)
        with self.assertRaises(ValueError):
            x = ht.arange(24)
            ht.trace(x)
        with self.assertRaises(TypeError):
            x = ht.arange(24).reshape((6, 4))
            ht.trace(x, axis1=0.2)
        with self.assertRaises(TypeError):
            ht.trace(x, axis2=1.4)
        with self.assertRaises(ValueError):
            ht.trace(x, axis1=2)
        with self.assertRaises(ValueError):
            ht.trace(x, axis2=2)
        with self.assertRaises(TypeError):
            ht.trace(x, offset=1.2)
        with self.assertRaises(ValueError):
            ht.trace(x, axis1=1, axis2=1)
        with self.assertRaises(ValueError):
            ht.trace(x, dtype="ht.int64")
        with self.assertRaises(TypeError):
            ht.trace(x, out=[])
        with self.assertRaises(ValueError):
            # As result is scalar
            out = ht.array([])
            ht.trace(x, out=out)

        # ------------------------------------------------
        # CASE > 2-D (4D)
        # ------------------------------------------------
        x = ht.arange(24, split=0).reshape((1, 2, 3, 4))
        x_np = x.numpy()
        # ------------------------------------------------
        # CASE split axis NOT in (axis1, axis2)
        # ------------------------------------------------
        axis1 = 1
        axis2 = 2
        out = ht.empty((1, 4), split=0, dtype=x.dtype)

        result = ht.trace(x, axis1=axis1, axis2=axis2)
        result_np = np.trace(x_np, axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # input = array_like (other than DNDarray)
        result = ht.trace(x.tolist(), axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # out
        result = ht.trace(x, out=out, axis1=axis1, axis2=axis2)
        result_np = np.trace(x_np, axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)
        self.assert_array_equal(out, result_np)

        # reversed axes order
        result = ht.trace(x, axis1=axis2, axis2=axis1)
        result_np = np.trace(x_np, axis1=axis2, axis2=axis1)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # different axes (still not in x.split = 0)
        axis1 = 1
        axis2 = 3
        result = ht.trace(x, offset=0, axis1=axis1, axis2=axis2, dtype=dtype)
        result_np = np.trace(x_np, offset=0, axis1=axis1, axis2=axis2, dtype=np.float32)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # negative axes
        axis1 = 1
        axis2 = 2
        result = ht.trace(x, axis1=axis1, axis2=-axis2)
        result_np = np.trace(x_np, axis1=axis1, axis2=-axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        result = ht.trace(x, axis1=-axis1, axis2=axis2)
        result_np = np.trace(x_np, axis1=-axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        result = ht.trace(x, axis1=-axis1, axis2=-axis2)
        result_np = np.trace(x_np, axis1=-axis1, axis2=-axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # offset != 0
        # negative offset
        axis1 = 1
        axis2 = 2
        o = -(x.gshape[axis1] - 1)
        result = ht.trace(x, offset=o, axis1=axis1, axis2=axis2)
        result_np = np.trace(x_np, offset=o, axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # positive offset
        o = x.gshape[axis2] - 1
        result = ht.trace(x, offset=o, axis1=axis1, axis2=axis2)
        result_np = np.trace(x_np, offset=o, axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # offset resulting into zero array
        axis1 = 1
        axis2 = 2
        # negative
        o = -x.gshape[axis1]
        result = ht.trace(x, offset=o, axis1=axis1, axis2=axis2)
        result_np = np.trace(x_np, offset=o, axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, np.zeros((1, 4)))
        self.assert_array_equal(result, result_np)

        # positive
        o = x.gshape[axis2]
        result = ht.trace(x, offset=o, axis1=axis1, axis2=axis2)
        result_np = np.trace(x_np, offset=o, axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, np.zeros((1, 4)))
        self.assert_array_equal(result, result_np)

        # different split axis (that is still not in (axis1, axis2))
        x = ht.arange(24).reshape((1, 2, 3, 4, 1))
        x = ht.array(x, split=2, dtype=dtype)
        x_np = x.numpy()
        axis1 = 0
        axis2 = 1
        out = ht.empty((3, 4, 1), split=2, dtype=x.dtype)
        result = ht.trace(x, axis1=axis1, axis2=axis2, out=out)
        result_np = np.trace(x_np, axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)
        self.assert_array_equal(out, result_np)

        # different split axis (that is still not in (axis1, axis2))
        x = ht.arange(24).reshape((1, 2, 3, 4, 1))
        x = ht.array(x, split=3, dtype=dtype)
        x_np = x.numpy()
        axis1 = 2
        axis2 = 4
        out = ht.empty((1, 2, 4), split=1, dtype=x.dtype)
        result = ht.trace(x, axis1=axis1, axis2=axis2, out=out)
        result_np = np.trace(x_np, axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # Exceptions
        with self.assertRaises(ValueError):
            out = ht.array([])
            ht.trace(x, out=out, axis1=axis1, axis2=axis2)

        # ------------------------------------------------
        # CASE split axis IN (axis1, axis2)
        # ------------------------------------------------
        x = ht.arange(24).reshape((1, 2, 3, 4))
        split_axis = 1
        x = ht.array(x, split=split_axis, dtype=dtype)
        x_np = x.numpy()
        axis1 = 1
        axis2 = 2
        result_shape = list(x.gshape)
        del result_shape[axis1], result_shape[axis2 - 1]
        out = ht.empty(tuple(result_shape), split=split_axis, dtype=x.dtype)

        result = ht.trace(x, axis1=axis1, axis2=axis2)
        result_np = np.trace(x_np, axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # input = array_like (other than DNDarray)
        result = ht.trace(x.tolist(), axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # out
        result = ht.trace(x, out=out, axis1=axis1, axis2=axis2)
        result_np = np.trace(x_np, axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)
        self.assert_array_equal(out, result_np)

        # reversed axes order
        result = ht.trace(x, axis1=axis2, axis2=axis1)
        result_np = np.trace(x_np, axis1=axis2, axis2=axis1)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # axis2 = a.split
        axis1 = 0
        axis2 = 1
        result = ht.trace(x, axis1=axis1, axis2=axis2)
        result_np = np.trace(x_np, axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # offset != 0
        # negative offset
        o = -(x.gshape[0] - 1)
        result = ht.trace(x, offset=o, axis1=axis1, axis2=axis2)
        result_np = np.trace(x_np, offset=o, axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # positive offset
        o = x.gshape[1] - 1
        result = ht.trace(x, offset=o, axis1=axis1, axis2=axis2)
        result_np = np.trace(x_np, offset=o, axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # different axes
        axis1 = 1
        axis2 = 2
        result_shape = list(x.gshape)
        del result_shape[axis1], result_shape[axis2 - 1]
        o = 0
        result = ht.trace(x, offset=o, axis1=axis1, axis2=axis2, dtype=dtype)
        result_np = np.trace(x_np, offset=o, axis1=axis1, axis2=axis2, dtype=np.float32)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, result_np)

        # offset resulting into zero array
        # negative
        o = -x.gshape[axis1]
        result = ht.trace(x, offset=o, axis1=axis1, axis2=axis2)
        result_np = np.trace(x_np, offset=o, axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, np.zeros(result_shape, dtype=result_np.dtype))
        self.assert_array_equal(result, result_np)

        # positive
        o = x.gshape[axis2]
        result = ht.trace(x, offset=o, axis1=axis1, axis2=axis2)
        result_np = np.trace(x_np, offset=o, axis1=axis1, axis2=axis2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assert_array_equal(result, np.zeros(result_shape, dtype=result_np.dtype))
        self.assert_array_equal(result, result_np)

        # Exceptions
        with self.assertRaises(ValueError):
            out = ht.array([])
            ht.trace(x, out=out, axis1=axis1, axis2=axis2)

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
        self.assertEqual(simple_matrix_t.larray.shape, (4, 2))

        # 4D array, not distributed, with given axis
        array_4d = ht.zeros((2, 3, 4, 5))
        array_4d_t = ht.transpose(array_4d, axes=(-1, 0, 2, 1))
        self.assertIsInstance(array_4d_t, ht.DNDarray)
        self.assertEqual(array_4d_t.dtype, ht.float32)
        self.assertEqual(array_4d_t.split, None)
        self.assertEqual(array_4d_t.shape, (5, 2, 4, 3))
        self.assertEqual(array_4d_t.larray.shape, (5, 2, 4, 3))

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
        self.assertTrue((result.larray == comparison).all())

        # 1D case, positive offset, data is not split, module-level call
        result = ht.tril(local_ones, k=2)
        comparison = torch.ones((5, 5), device=self.device.torch_device).tril(diagonal=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.lshape, (5, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result.larray == comparison).all())

        # 1D case, negative offset, data is not split, module-level call
        result = ht.tril(local_ones, k=-2)
        comparison = torch.ones((5, 5), device=self.device.torch_device).tril(diagonal=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.lshape, (5, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result.larray == comparison).all())

        local_ones = ht.ones((4, 5))

        # 2D case, no offset, data is not split, method
        result = local_ones.tril()
        comparison = torch.ones((4, 5), device=self.device.torch_device).tril()
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result.larray == comparison).all())

        # 2D case, positive offset, data is not split, method
        result = local_ones.tril(k=2)
        comparison = torch.ones((4, 5), device=self.device.torch_device).tril(diagonal=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result.larray == comparison).all())

        # 2D case, negative offset, data is not split, method
        result = local_ones.tril(k=-2)
        comparison = torch.ones((4, 5), device=self.device.torch_device).tril(diagonal=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result.larray == comparison).all())

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
                self.assertTrue((result.larray[i, j] == comparison).all())

        # 2D+ case, positive offset, data is not split, module-level call
        result = local_ones.tril(k=2)
        comparison = torch.ones((5, 6), device=self.device.torch_device).tril(diagonal=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (3, 4, 5, 6))
        self.assertEqual(result.lshape, (3, 4, 5, 6))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result.larray[i, j] == comparison).all())

        # # 2D+ case, negative offset, data is not split, module-level call
        result = local_ones.tril(k=-2)
        comparison = torch.ones((5, 6), device=self.device.torch_device).tril(diagonal=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (3, 4, 5, 6))
        self.assertEqual(result.lshape, (3, 4, 5, 6))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result.larray[i, j] == comparison).all())

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
            self.assertTrue(result.larray[-1, 0] == 1)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result.larray[0, -1] == 0)

        # 1D case, positive offset, data is split, method
        result = distributed_ones.tril(k=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 22)
        if result.comm.rank == 0:
            self.assertTrue(result.larray[-1, 0] == 1)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result.larray[0, -1] == 0)

        # 1D case, negative offset, data is split, method
        result = distributed_ones.tril(k=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 6)
        if result.comm.rank == 0:
            self.assertTrue(result.larray[-1, 0] == 1)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result.larray[0, -1] == 0)

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
            self.assertTrue(result.larray[0, -1] == 0)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result.larray[-1, 0] == 1)

        # 2D case, positive offset, data is horizontally split, method
        result = distributed_ones.tril(k=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 17)
        if result.comm.rank == 0:
            self.assertTrue(result.larray[0, -1] == 0)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result.larray[-1, 0] == 1)

        # 2D case, negative offset, data is horizontally split, method
        result = distributed_ones.tril(k=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 3)
        if result.comm.rank == 0:
            self.assertTrue(result.larray[0, -1] == 0)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result.larray[-1, 0] == 1)

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
            self.assertTrue(result.larray[-1, 0] == 1)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result.larray[0, -1] == 0)

        # 2D case, positive offset, data is horizontally split, method
        result = distributed_ones.tril(k=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 17)
        if result.comm.rank == 0:
            self.assertTrue(result.larray[-1, 0] == 1)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result.larray[0, -1] == 0)

        # 2D case, negative offset, data is horizontally split, method
        result = distributed_ones.tril(k=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 3)
        if result.comm.rank == 0:
            self.assertTrue(result.larray[-1, 0] == 1)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result.larray[0, -1] == 0)

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
        self.assertTrue((result.larray == comparison).all())

        # 1D case, positive offset, data is not split, module-level call
        result = ht.triu(local_ones, k=2)
        comparison = torch.ones((5, 5), device=self.device.torch_device).triu(diagonal=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.lshape, (5, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result.larray == comparison).all())

        # 1D case, negative offset, data is not split, module-level call
        result = ht.triu(local_ones, k=-2)
        comparison = torch.ones((5, 5), device=self.device.torch_device).triu(diagonal=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.lshape, (5, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result.larray == comparison).all())

        local_ones = ht.ones((4, 5))

        # 2D case, no offset, data is not split, method
        result = local_ones.triu()
        comparison = torch.ones((4, 5), device=self.device.torch_device).triu()
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result.larray == comparison).all())

        # 2D case, positive offset, data is not split, method
        result = local_ones.triu(k=2)
        comparison = torch.ones((4, 5), device=self.device.torch_device).triu(diagonal=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result.larray == comparison).all())

        # 2D case, negative offset, data is not split, method
        result = local_ones.triu(k=-2)
        comparison = torch.ones((4, 5), device=self.device.torch_device).triu(diagonal=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result.larray == comparison).all())

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
                self.assertTrue((result.larray[i, j] == comparison).all())

        # 2D+ case, positive offset, data is not split, module-level call
        result = local_ones.triu(k=2)
        comparison = torch.ones((5, 6), device=self.device.torch_device).triu(diagonal=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (3, 4, 5, 6))
        self.assertEqual(result.lshape, (3, 4, 5, 6))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result.larray[i, j] == comparison).all())

        # # 2D+ case, negative offset, data is not split, module-level call
        result = local_ones.triu(k=-2)
        comparison = torch.ones((5, 6), device=self.device.torch_device).triu(diagonal=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (3, 4, 5, 6))
        self.assertEqual(result.lshape, (3, 4, 5, 6))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result.larray[i, j] == comparison).all())

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
            self.assertTrue(result.larray[-1, 0] == 0)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result.larray[0, -1] == 1)

        # 1D case, positive offset, data is split, method
        result = distributed_ones.triu(k=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 6)
        if result.comm.rank == 0:
            self.assertTrue(result.larray[-1, 0] == 0)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result.larray[0, -1] == 1)

        # 1D case, negative offset, data is split, method
        result = distributed_ones.triu(k=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 22)
        if result.comm.rank == 0:
            self.assertTrue(result.larray[-1, 0] == 0)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result.larray[0, -1] == 1)

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
            self.assertTrue(result.larray[0, -1] == 1)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result.larray[-1, 0] == 0)

        # # 2D case, positive offset, data is horizontally split, method
        result = distributed_ones.triu(k=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 6)
        if result.comm.rank == 0:
            self.assertTrue(result.larray[0, -1] == 1)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result.larray[-1, 0] == 0)

        # # 2D case, negative offset, data is horizontally split, method
        result = distributed_ones.triu(k=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 19)
        if result.comm.rank == 0:
            self.assertTrue(result.larray[0, -1] == 1)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result.larray[-1, 0] == 0)

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
            self.assertTrue(result.larray[-1, 0] == 0)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result.larray[0, -1] == 1)

        # 2D case, positive offset, data is horizontally split, method
        result = distributed_ones.triu(k=2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 6)
        if result.comm.rank == 0:
            self.assertTrue(result.larray[-1, 0] == 0)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result.larray[0, -1] == 1)

        # 2D case, negative offset, data is horizontally split, method
        result = distributed_ones.triu(k=-2)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 19)
        if result.comm.rank == 0:
            self.assertTrue(result.larray[-1, 0] == 0)
        if result.comm.rank == result.shape[0] - 1:
            self.assertTrue(result.larray[0, -1] == 1)

    def test_vdot(self):
        a = ht.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], split=0)
        b = ht.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], split=0)

        vdot = ht.vdot(a, b)
        self.assertEqual(vdot.dtype, a.dtype)
        self.assertEqual(vdot.split, None)
        self.assertTrue(ht.equal(vdot, ht.array([110 + 10j])))

        vdot = ht.vdot(b, a)
        self.assertTrue(ht.equal(vdot, ht.array([110 - 10j])))

        with self.assertRaises(ValueError):
            ht.vdot(ht.array([1, 2, 3]), ht.array([[1, 2], [3, 4]]))

    def test_vecdot(self):
        a = ht.array([1, 1, 1])
        b = ht.array([1, 2, 3])

        c = ht.linalg.vecdot(a, b)

        self.assertEqual(c.dtype, ht.int64)
        self.assertEqual(c.device, a.device)
        self.assertTrue(ht.equal(c, ht.array([6])))

        a = ht.full((4, 4), 2, split=0)
        b = ht.ones(4)

        c = ht.linalg.vecdot(a, b, axis=0, keepdim=True)
        self.assertEqual(c.dtype, ht.float32)
        self.assertEqual(c.device, a.device)
        self.assertTrue(ht.equal(c, ht.array([[8, 8, 8, 8]])))

    def test_vector_norm(self):
        a = ht.arange(9, dtype=ht.float) - 4
        a_split = ht.arange(9, dtype=ht.float, split=0) - 4
        b = a.reshape((3, 3))
        b0 = ht.reshape(a, (3, 3), new_split=0)
        b1 = ht.reshape(a, (3, 3), new_split=1)

        # vector infintity norm
        vn = ht.vector_norm(a, ord=ht.inf)
        self.assertEqual(vn.split, a.split)
        self.assertEqual(vn.dtype, a.dtype)
        self.assertEqual(vn.device, a.device)
        self.assertEqual(vn.item(), 4.0)

        # vector 0 norm
        vn = ht.vector_norm(a, ord=0)
        self.assertEqual(vn.split, a.split)
        self.assertEqual(vn.dtype, a.dtype)
        self.assertEqual(vn.device, a.device)
        self.assertEqual(vn.item(), 8.0)

        # split vector -infinity
        vn = ht.vector_norm(a_split, ord=-ht.inf)
        self.assertEqual(vn.split, a.split)
        self.assertEqual(vn.dtype, a.dtype)
        self.assertEqual(vn.device, a.device)
        self.assertEqual(vn.item(), 0.0)

        # matrix 1 norm no axis
        vn = ht.vector_norm(b, ord=1)
        self.assertEqual(vn.split, b.split)
        self.assertEqual(vn.dtype, b.dtype)
        self.assertEqual(vn.device, b.device)
        self.assertEqual(vn.item(), 20.0)

        # split matrix axis l2-norm
        vn = ht.vector_norm(b0, axis=1, ord=2)
        self.assertEqual(vn.split, 0)
        self.assertEqual(vn.dtype, b0.dtype)
        self.assertEqual(vn.device, b0.device)
        self.assertTrue(ht.allclose(vn, ht.array([5.38516481, 1.41421356, 5.38516481], split=0)))

        # split matrix axis keepdim norm 3
        vn = ht.vector_norm(b1, axis=1, keepdims=True, ord=3)
        self.assertEqual(vn.split, None)
        self.assertEqual(vn.dtype, b1.dtype)
        self.assertEqual(vn.device, b1.device)
        self.assertTrue(
            ht.allclose(vn, ht.array([[4.62606501], [1.25992105], [4.62606501]], split=None))
        )

        # different dtype
        vn = ht.linalg.vector_norm(ht.full((4, 4, 4), 1 + 1j, dtype=ht.int), axis=0, ord=4)
        self.assertEqual(vn.split, None)
        self.assertEqual(vn.dtype, ht.float)
        self.assertTrue(
            ht.equal(
                vn,
                ht.array(
                    [
                        [2.0, 2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0, 2.0],
                    ]
                ),
            )
        )

        # bad ord
        with self.assertRaises(ValueError):
            ht.vector_norm(ht.array([1, 2, 3]), ord="fro")
        # bad axis
        with self.assertRaises(TypeError):
            ht.vector_norm(ht.array([1, 2, 3]), axis=(1, 2))
        with self.assertRaises(TypeError):
            ht.vector_norm(ht.array([1, 2, 3]), axis="r")
