import numpy as np
import torch

from itertools import combinations
from scipy import stats as ss

import heat as ht
from .test_suites.basic_test import TestCase


class TestStatistics(TestCase):
    def test_argmax(self):
        torch.manual_seed(1)
        data = ht.random.randn(3, 4, 5)

        # 3D local tensor, major axis
        result = ht.argmax(data, axis=0)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result.larray.dtype, torch.int64)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result.larray == data.larray.argmax(0)).all())

        # 3D local tensor, minor axis
        result = ht.argmax(data, axis=-1, keepdims=True)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result.larray.dtype, torch.int64)
        self.assertEqual(result.shape, (3, 4, 1))
        self.assertEqual(result.lshape, (3, 4, 1))
        self.assertEqual(result.split, None)
        self.assertTrue((result.larray == data.larray.argmax(-1, keepdims=True)).all())

        # 1D split tensor, no axis
        data = ht.arange(-10, 10, split=0)
        result = ht.argmax(data)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result.larray.dtype, torch.int64)
        self.assertEqual(result.shape, (1,))
        self.assertEqual(result.lshape, (1,))
        self.assertEqual(result.split, None)
        self.assertTrue((result.larray == torch.tensor([19], device=self.device.torch_device)))

        # 2D split tensor, along the axis
        data = ht.array(ht.random.randn(4, 5), is_split=0)
        result = ht.argmax(data, axis=1)
        expected = torch.argmax(data.larray, dim=1)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result.larray.dtype, torch.int64)
        self.assertEqual(result.shape, (ht.MPI_WORLD.size * 4,))
        self.assertEqual(result.lshape, (4,))
        self.assertTrue((result.larray == expected).all())

        # 2D split tensor, across the axis
        size = ht.MPI_WORLD.size * 2
        data = ht.tril(ht.ones((size, size), split=0), k=-1)

        result = ht.argmax(data, axis=0)
        expected = torch.tensor(np.argmax(data.numpy(), axis=0))
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result.larray.dtype, torch.int64)
        self.assertEqual(result.shape, (size,))
        self.assertEqual(result.lshape, (size,))
        self.assertEqual(result.split, None)
        # skip test on gpu; argmax works different
        if not (torch.cuda.is_available() and result.device == ht.gpu):
            self.assertTrue((result.larray == expected).all())

        # 2D split tensor, across the axis, output tensor
        size = ht.MPI_WORLD.size * 2
        data = ht.tril(ht.ones((size, size), split=0), k=-1)

        output = ht.empty((size,), dtype=ht.int64)
        result = ht.argmax(data, axis=0, out=output)
        expected = torch.tensor(np.argmax(data.numpy(), axis=0))
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(output.dtype, ht.int64)
        self.assertEqual(output.larray.dtype, torch.int64)
        self.assertEqual(output.shape, (size,))
        self.assertEqual(output.lshape, (size,))
        self.assertEqual(output.split, None)
        # skip test on gpu; argmax works different
        if not (torch.cuda.is_available() and result.device == ht.gpu):
            self.assertTrue((output.larray == expected).all())

        # check exceptions
        with self.assertRaises(TypeError):
            data.argmax(axis=(0, 1))
        with self.assertRaises(TypeError):
            data.argmax(axis=1.1)
        with self.assertRaises(TypeError):
            data.argmax(axis="y")
        with self.assertRaises(ValueError):
            ht.argmax(data, axis=-4)
        output = ht.empty((size,), dtype=ht.float32)
        with self.assertRaises(TypeError):
            ht.argmax(data, axis=0, out=output)

    def test_argmin(self):
        torch.manual_seed(1)
        data = ht.random.randn(3, 4, 5)

        # 3D local tensor, no axis
        result = ht.argmin(data)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result.larray.dtype, torch.int64)
        self.assertEqual(result.shape, (1,))
        self.assertEqual(result.lshape, (1,))
        self.assertEqual(result.split, None)
        self.assertTrue((result.larray == data.larray.argmin()).all())

        # 3D local tensor, major axis
        result = ht.argmin(data, axis=0)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result.larray.dtype, torch.int64)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result.larray == data.larray.argmin(0)).all())

        # 3D local tensor, minor axis
        result = ht.argmin(data, axis=-1, keepdims=True)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result.larray.dtype, torch.int64)
        self.assertEqual(result.shape, (3, 4, 1))
        self.assertEqual(result.lshape, (3, 4, 1))
        self.assertEqual(result.split, None)
        self.assertTrue((result.larray == data.larray.argmin(-1, keepdim=True)).all())

        # 2D split tensor, along the axis
        data = ht.array(ht.random.randn(4, 5), is_split=0)
        result = ht.argmin(data, axis=1)
        expected = torch.argmin(data.larray, dim=1)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result.larray.dtype, torch.int64)
        self.assertEqual(result.shape, (ht.MPI_WORLD.size * 4,))
        self.assertEqual(result.lshape, (4,))
        self.assertTrue((result.larray == expected).all())

        # 2D split tensor, across the axis
        size = ht.MPI_WORLD.size * 2
        data = ht.triu(ht.ones((size, size), split=0), k=1)

        result = ht.argmin(data, axis=0)
        expected = torch.tensor(np.argmin(data.numpy(), axis=0))
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result.larray.dtype, torch.int64)
        self.assertEqual(result.shape, (size,))
        self.assertEqual(result.lshape, (size,))
        self.assertEqual(result.split, None)
        # skip test on gpu; argmin works different
        if not (torch.cuda.is_available() and result.device == ht.gpu):
            self.assertTrue((result.larray == expected).all())

        # 2D split tensor, across the axis, output tensor
        size = ht.MPI_WORLD.size * 2
        data = ht.triu(ht.ones((size, size), split=0), k=1)

        output = ht.empty((size,), dtype=ht.int64)
        result = ht.argmin(data, axis=0, out=output)
        expected = torch.tensor(np.argmin(data.numpy(), axis=0))
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(output.dtype, ht.int64)
        self.assertEqual(output.larray.dtype, torch.int64)
        self.assertEqual(output.shape, (size,))
        self.assertEqual(output.lshape, (size,))
        self.assertEqual(output.split, None)
        # skip test on gpu; argmin works different
        if not (torch.cuda.is_available() and result.device == ht.gpu):
            self.assertTrue((output.larray == expected).all())

        # check exceptions
        with self.assertRaises(TypeError):
            data.argmin(axis=(0, 1))
        with self.assertRaises(TypeError):
            data.argmin(axis=1.1)
        with self.assertRaises(TypeError):
            data.argmin(axis="y")
        with self.assertRaises(ValueError):
            ht.argmin(data, axis=-4)

    def test_average(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

        ht_array = ht.array(data, dtype=float)
        comparison = np.asanyarray(data)

        # check global average
        avg = ht.average(ht_array)

        self.assertIsInstance(avg, ht.DNDarray)
        self.assertEqual(avg.shape, ())
        self.assertEqual(avg.lshape, ())
        self.assertEqual(avg.split, None)
        self.assertEqual(avg.dtype, ht.float32)
        self.assertEqual(avg.larray.dtype, torch.float32)
        self.assertEqual(avg.numpy(), np.average(comparison))

        # average along first axis
        avg_vertical = ht.average(ht_array, axis=0)

        self.assertIsInstance(avg_vertical, ht.DNDarray)
        self.assertEqual(avg_vertical.shape, (3,))
        self.assertEqual(avg_vertical.lshape, (3,))
        self.assertEqual(avg_vertical.split, None)
        self.assertEqual(avg_vertical.dtype, ht.float32)
        self.assertEqual(avg_vertical.larray.dtype, torch.float32)
        self.assertTrue((avg_vertical.numpy() == np.average(comparison, axis=0)).all())

        # average along second axis
        avg_horizontal = ht.average(ht_array, axis=1)

        self.assertIsInstance(avg_horizontal, ht.DNDarray)
        self.assertEqual(avg_horizontal.shape, (4,))
        self.assertEqual(avg_horizontal.lshape, (4,))
        self.assertEqual(avg_horizontal.split, None)
        self.assertEqual(avg_horizontal.dtype, ht.float32)
        self.assertEqual(avg_horizontal.larray.dtype, torch.float32)
        self.assertTrue((avg_horizontal.numpy() == np.average(comparison, axis=1)).all())

        # check weighted average over all float elements of split 3d tensor, across split axis
        random_volume = ht.array(
            torch.randn((3, 3, 3), dtype=torch.float64, device=self.device.torch_device), is_split=1
        )
        size = random_volume.comm.size
        random_weights = ht.array(
            torch.randn((3 * size,), dtype=torch.float64, device=self.device.torch_device), split=0
        )
        avg_volume = ht.average(random_volume, weights=random_weights, axis=1)
        np_avg_volume = np.average(random_volume.numpy(), weights=random_weights.numpy(), axis=1)
        self.assertIsInstance(avg_volume, ht.DNDarray)
        self.assertEqual(avg_volume.shape, (3, 3))
        self.assertEqual(avg_volume.lshape, (3, 3))
        self.assertEqual(avg_volume.dtype, ht.float64)
        self.assertEqual(avg_volume.larray.dtype, torch.float64)
        self.assertEqual(avg_volume.split, None)
        self.assertAlmostEqual(avg_volume.numpy().all(), np_avg_volume.all())
        avg_volume_with_cumwgt = ht.average(
            random_volume, weights=random_weights, axis=1, returned=True
        )
        self.assertIsInstance(avg_volume_with_cumwgt, tuple)
        self.assertIsInstance(avg_volume_with_cumwgt[1], ht.DNDarray)
        self.assertEqual(avg_volume_with_cumwgt[1].gshape, avg_volume_with_cumwgt[0].gshape)
        self.assertEqual(avg_volume_with_cumwgt[1].split, avg_volume_with_cumwgt[0].split)

        # check weighted average over all float elements of split 3d tensor (3d weights)

        random_weights_3d = ht.array(
            torch.randn((3, 3, 3), dtype=torch.float64, device=self.device.torch_device), is_split=1
        )
        avg_volume = ht.average(random_volume, weights=random_weights_3d, axis=1)
        np_avg_volume = np.average(random_volume.numpy(), weights=random_weights.numpy(), axis=1)
        self.assertIsInstance(avg_volume, ht.DNDarray)
        self.assertEqual(avg_volume.shape, (3, 3))
        self.assertEqual(avg_volume.lshape, (3, 3))
        self.assertEqual(avg_volume.dtype, ht.float64)
        self.assertEqual(avg_volume.larray.dtype, torch.float64)
        self.assertEqual(avg_volume.split, None)
        self.assertAlmostEqual(avg_volume.numpy().all(), np_avg_volume.all())
        avg_volume_with_cumwgt = ht.average(
            random_volume, weights=random_weights, axis=1, returned=True
        )
        self.assertIsInstance(avg_volume_with_cumwgt, tuple)
        self.assertIsInstance(avg_volume_with_cumwgt[1], ht.DNDarray)
        self.assertEqual(avg_volume_with_cumwgt[1].gshape, avg_volume_with_cumwgt[0].gshape)
        self.assertEqual(avg_volume_with_cumwgt[1].split, avg_volume_with_cumwgt[0].split)

        # check average over all float elements of split 3d tensor, tuple axis
        random_volume = ht.random.randn(3, 3, 3, split=0)
        avg_volume = ht.average(random_volume, axis=(1, 2))

        self.assertIsInstance(avg_volume, ht.DNDarray)
        self.assertEqual(avg_volume.shape, (3,))
        self.assertEqual(avg_volume.lshape[0], random_volume.lshape[0])
        self.assertEqual(avg_volume.dtype, ht.float32)
        self.assertEqual(avg_volume.larray.dtype, torch.float32)
        self.assertEqual(avg_volume.split, 0)

        # check weighted average over all float elements of split 5d tensor, along split axis
        random_5d = ht.random.randn(random_volume.comm.size, 2, 3, 4, 5, split=0)
        axis = random_5d.split
        random_weights = ht.random.randn(random_5d.gshape[axis], split=0)
        avg_5d = random_5d.average(weights=random_weights, axis=axis)

        self.assertIsInstance(avg_5d, ht.DNDarray)
        self.assertEqual(avg_5d.gshape, (2, 3, 4, 5))
        self.assertLessEqual(avg_5d.lshape[1], 3)
        self.assertEqual(avg_5d.dtype, ht.float32)
        self.assertEqual(avg_5d.larray.dtype, torch.float32)
        self.assertEqual(avg_5d.split, None)

        # check exceptions
        with self.assertRaises(TypeError):
            ht.average(comparison)
        with self.assertRaises(TypeError):
            ht.average(random_5d, weights=random_weights.numpy(), axis=axis)
        with self.assertRaises(TypeError):
            ht.average(random_5d, weights=random_weights, axis=None)
        with self.assertRaises(NotImplementedError):
            ht.average(random_5d, weights=random_weights, axis=(1, 2))
        random_weights = ht.random.randn(random_5d.gshape[axis], random_5d.gshape[axis + 1])
        with self.assertRaises(TypeError):
            ht.average(random_5d, weights=random_weights, axis=axis)
        random_shape_weights = ht.random.randn(random_5d.gshape[axis] + 1)
        with self.assertRaises(ValueError):
            ht.average(random_5d, weights=random_shape_weights, axis=axis)
        zero_weights = ht.zeros((random_5d.gshape[axis]), split=0)
        with self.assertRaises(ZeroDivisionError):
            ht.average(random_5d, weights=zero_weights, axis=axis)
        weights_5d_split_mismatch = ht.ones(random_5d.gshape, split=-1)
        if ht.MPI_WORLD.size > 1:
            with self.assertRaises(NotImplementedError):
                ht.average(random_5d, weights=weights_5d_split_mismatch, axis=axis)

        with self.assertRaises(TypeError):
            ht_array.average(axis=1.1)
        with self.assertRaises(TypeError):
            ht_array.average(axis="y")
        with self.assertRaises(ValueError):
            ht.average(ht_array, axis=-4)

    def test_bincount(self):
        a = ht.array([], dtype=ht.int)
        res = ht.bincount(a)
        self.assertEqual(res.size, 0)
        self.assertEqual(res.dtype, ht.int64)

        a = ht.arange(5)
        res = ht.bincount(a)
        self.assertEqual(res.size, 5)
        self.assertEqual(res.dtype, ht.int64)
        self.assertTrue(ht.equal(res, ht.ones((5,), dtype=ht.int64)))

        w = ht.arange(5)
        res = ht.bincount(a, weights=w)
        self.assertEqual(res.size, 5)
        self.assertEqual(res.dtype, ht.float64)
        self.assertTrue(ht.equal(res, ht.arange(5, dtype=ht.float64)))

        res = ht.bincount(a, minlength=8)
        self.assertEqual(res.size, 8)
        self.assertEqual(res.dtype, ht.int64)
        self.assertTrue(ht.equal(res, ht.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=ht.int64)))

        a = ht.arange(4, split=0)
        w = ht.arange(4, split=0)
        res = ht.bincount(a, weights=w)
        self.assertEqual(res.size, 4)
        self.assertEqual(res.dtype, ht.float64)
        self.assertTrue(ht.equal(res, ht.arange(4, dtype=ht.float64)))

        with self.assertRaises(ValueError):
            ht.bincount(ht.array([0, 1, 2, 3], split=0), weights=ht.array([1, 2, 3, 4]))

    def test_bucketize(self):
        boundaries = ht.array([1, 3, 5, 7, 9])
        v = ht.array([[3, 6, 9], [3, 6, 9]])
        a = ht.bucketize(v, boundaries)

        self.assertTrue(ht.equal(a, ht.array([[1, 3, 4], [1, 3, 4]])))
        self.assertTrue(a.dtype, ht.int64)
        self.assertTrue(a.shape, v.shape)

        a = ht.bucketize(v, boundaries, right=True)
        self.assertTrue(ht.equal(a, ht.array([[2, 3, 5], [2, 3, 5]])))
        self.assertEqual(a.dtype, ht.int64)
        self.assertTrue(a.shape, v.shape)

        boundaries, _ = torch.sort(torch.rand(5, device=self.device.torch_device))
        v = torch.rand(6, device=self.device.torch_device)
        t = torch.bucketize(v, boundaries, out_int32=True)

        v = ht.array(v, split=0)
        a = ht.bucketize(v, boundaries, out_int32=True)
        self.assertTrue(ht.equal(ht.resplit(a, None), ht.asarray(t)))
        self.assertEqual(a.dtype, ht.int32)

        if ht.MPI_WORLD.size > 1:
            with self.assertRaises(RuntimeError):
                ht.bucketize(a, ht.array([0.0, 0.5, 1.0], split=0))

    def test_cov(self):
        x = ht.array([[0, 2], [1, 1], [2, 0]], dtype=ht.float, split=1).T
        if x.comm.size < 3:
            cov = ht.cov(x)
            actual = ht.array([[1, -1], [-1, 1]], split=0)
            self.assertTrue(ht.equal(cov, actual))

        data = np.loadtxt("heat/datasets/iris.csv", delimiter=";")
        np_cov = np.cov(data[:, 0], data[:, 1:3], rowvar=False)

        # split = None tests
        htdata = ht.load("heat/datasets/iris.csv", sep=";", split=None)
        ht_cov = ht.cov(htdata[:, 0], htdata[:, 1:3], rowvar=False)
        comp = ht.array(np_cov, dtype=ht.float)
        self.assertTrue(ht.allclose(comp - ht_cov, 0, atol=1e-4))

        np_cov = np.cov(data, rowvar=False)
        ht_cov = ht.cov(htdata, rowvar=False)
        self.assertTrue(ht.allclose(ht.array(np_cov, dtype=ht.float) - ht_cov, 0, atol=1e-4))

        np_cov = np.cov(data, rowvar=False, ddof=1)
        ht_cov = ht.cov(htdata, rowvar=False, ddof=1)
        self.assertTrue(ht.allclose(ht.array(np_cov, dtype=ht.float) - ht_cov, 0, atol=1e-4))

        np_cov = np.cov(data, rowvar=False, bias=True)
        ht_cov = ht.cov(htdata, rowvar=False, bias=True)
        self.assertTrue(ht.allclose(ht.array(np_cov, dtype=ht.float) - ht_cov, 0, atol=1e-4))

        # split = 0 tests
        data = np.loadtxt("heat/datasets/iris.csv", delimiter=";")
        np_cov = np.cov(data[:, 0], data[:, 1:3], rowvar=False)

        htdata = ht.load("heat/datasets/iris.csv", sep=";", split=0)
        ht_cov = ht.cov(htdata[:, 0], htdata[:, 1:3], rowvar=False)
        comp = ht.array(np_cov, dtype=ht.float)
        self.assertTrue(ht.allclose(comp - ht_cov, 0, atol=1e-4))

        np_cov = np.cov(data, rowvar=False)
        ht_cov = ht.cov(htdata, rowvar=False)
        self.assertTrue(ht.allclose(ht.array(np_cov, dtype=ht.float) - ht_cov, 0, atol=1e-4))

        np_cov = np.cov(data, rowvar=False, ddof=1)
        ht_cov = ht.cov(htdata, rowvar=False, ddof=1)
        self.assertTrue(ht.allclose(ht.array(np_cov, dtype=ht.float) - ht_cov, 0, atol=1e-4))

        np_cov = np.cov(data, rowvar=False, bias=True)
        ht_cov = ht.cov(htdata, rowvar=False, bias=True)
        self.assertTrue(ht.allclose(ht.array(np_cov, dtype=ht.float) - ht_cov, 0, atol=1e-4))

        if 1 < x.comm.size < 5:
            # split 1 tests
            htdata = ht.load("heat/datasets/iris.csv", sep=";", split=1)
            np_cov = np.cov(data, rowvar=False)
            ht_cov = ht.cov(htdata, rowvar=False)
            self.assertTrue(ht.allclose(ht.array(np_cov, dtype=ht.float), ht_cov, atol=1e-4))

            np_cov = np.cov(data, data, rowvar=True)

            htdata = ht.load("heat/datasets/iris.csv", sep=";", split=0)
            ht_cov = ht.cov(htdata, htdata, rowvar=True)
            self.assertTrue(ht.allclose(ht.array(np_cov, dtype=ht.float), ht_cov, atol=1e-4))

            htdata = ht.load("heat/datasets/iris.csv", sep=";", split=0)
            with self.assertRaises(RuntimeError):
                ht.cov(htdata[1:], rowvar=False)
            with self.assertRaises(RuntimeError):
                ht.cov(htdata, htdata[1:], rowvar=False)

        with self.assertRaises(TypeError):
            ht.cov(np_cov)
        with self.assertRaises(TypeError):
            ht.cov(htdata, np_cov)
        with self.assertRaises(TypeError):
            ht.cov(htdata, ddof="str")
        with self.assertRaises(ValueError):
            ht.cov(ht.zeros((1, 2, 3)))
        with self.assertRaises(ValueError):
            ht.cov(htdata, ht.zeros((1, 2, 3)))
        with self.assertRaises(ValueError):
            ht.cov(htdata, ddof=10000)

    def test_digitize(self):
        x = ht.array([1.2, 10.0, 12.4, 15.5, 20.0])
        bins = ht.array([0, 5, 10, 15, 20])
        a = ht.digitize(x, bins, right=True)
        t = np.digitize(x.numpy(), bins.numpy(), right=True)

        self.assertTrue((a.numpy() == t).all())
        self.assertTrue(a.dtype, ht.int64)
        self.assertTrue(a.shape, x.shape)

        a = ht.digitize(x, bins, right=False)
        t = np.digitize(x.numpy(), bins.numpy(), right=False)
        self.assertTrue((a.numpy() == t).all())
        self.assertEqual(a.dtype, ht.int64)
        self.assertTrue(a.shape, x.shape)

        bins = ht.flipud(bins)
        a = ht.digitize(x, bins, right=True)
        t = np.digitize(x.numpy(), bins.numpy(), right=True)
        self.assertTrue((a.numpy() == t).all())
        self.assertEqual(a.dtype, ht.int64)
        self.assertTrue(a.shape, x.shape)

        a = ht.digitize(x, bins, right=False)
        t = np.digitize(x.numpy(), bins.numpy(), right=False)
        self.assertTrue((a.numpy() == t).all())
        self.assertEqual(a.dtype, ht.int64)
        self.assertTrue(a.shape, x.shape)

        y = ht.array([[1.2, 7.3, 10.0], [12.4, 15.5, 20.0]], split=0)
        a = ht.digitize(y, bins, right=False)
        self.assertTrue(ht.equal(a, ht.array([[4, 3, 2], [2, 1, 0]], split=0)))
        self.assertEqual(a.dtype, ht.int64)
        self.assertTrue(a.shape, y.shape)

        y = ht.array([[1.2, 7.3, 10.0], [12.4, 15.5, 20.0]], split=1)
        a = ht.digitize(y, bins, right=False)
        self.assertTrue(ht.equal(a, ht.array([[4, 3, 2], [2, 1, 0]], split=1)))
        self.assertEqual(a.dtype, ht.int64)
        self.assertTrue(a.shape, x.shape)

        if ht.MPI_WORLD.size > 1:
            with self.assertRaises(RuntimeError):
                ht.digitize(a, ht.array([0.0, 0.5, 1.0], split=0))

    def test_histc(self):
        # few entries and float64
        c = torch.arange(4, dtype=torch.float64, device=self.device.torch_device)
        comp = torch.histc(c, 7)
        a = ht.array(c)
        res = ht.histc(a, 7)

        self.assertEqual(res.shape, (7,))
        self.assertEqual(res.dtype, ht.float64)
        self.assertEqual(res.device, self.device)
        self.assertEqual(res.split, None)
        self.assertTrue(torch.equal(res.larray, comp))

        # matrix and splits
        c = torch.rand([10, 10, 10], device=self.device.torch_device)
        comp = torch.histc(c)

        a = ht.array(c)
        res = ht.histc(a)
        self.assertEqual(res.shape, (100,))
        self.assertEqual(res.dtype, ht.float32)
        self.assertEqual(res.device, self.device)
        self.assertEqual(res.split, None)
        self.assertTrue(torch.equal(res.larray, comp))

        a = ht.array(c, split=0)
        res = ht.histc(a)
        self.assertEqual(res.shape, (100,))
        self.assertEqual(res.dtype, ht.float32)
        self.assertEqual(res.device, self.device)
        self.assertEqual(res.split, None)
        self.assertTrue(torch.equal(res.larray, comp))

        a = ht.array(c, split=1)
        res = ht.histc(a)
        self.assertEqual(res.shape, (100,))
        self.assertEqual(res.dtype, ht.float32)
        self.assertEqual(res.device, self.device)
        self.assertEqual(res.split, None)
        self.assertTrue(torch.equal(res.larray, comp))

        a = ht.array(c, split=2)
        res = ht.histc(a)
        self.assertEqual(res.shape, (100,))
        self.assertEqual(res.dtype, ht.float32)
        self.assertEqual(res.device, self.device)
        self.assertEqual(res.split, None)
        self.assertTrue(torch.equal(res.larray, comp))

        # out parameter, min max
        out = ht.empty(20, dtype=ht.float32, device=self.device)
        c = torch.randint(10, size=(8,), dtype=torch.float32, device=self.device.torch_device)
        comp = torch.histc(c, bins=20, min=0, max=20)

        a = ht.array(c)
        ht.histc(a, bins=20, min=0, max=20, out=out)
        self.assertEqual(out.shape, (20,))
        self.assertEqual(out.dtype, ht.float32)
        self.assertEqual(res.device, self.device)
        self.assertEqual(res.split, None)
        self.assertTrue(torch.equal(out.larray, comp))

        a = ht.array(c, split=0)
        ht.histc(a, bins=20, min=0, max=20, out=out)
        self.assertEqual(out.shape, (20,))
        self.assertEqual(out.dtype, ht.float32)
        self.assertEqual(res.device, self.device)
        self.assertEqual(res.split, None)
        self.assertTrue(torch.equal(out.larray, comp))

        # Alias
        a = ht.arange(10, dtype=ht.float)
        hist = ht.histc(a, 10)
        alias = ht.histogram(a)

        self.assertEqual(alias.gnumel, hist.gnumel)
        self.assertTrue(ht.equal(alias, hist))

        with self.assertRaises(NotImplementedError):
            ht.histogram(a, "str")
        with self.assertRaises(NotImplementedError):
            ht.histogram(a, [1, 2, 3])
        with self.assertRaises(NotImplementedError):
            ht.histogram(a, weights=[1, 2, 3])
        with self.assertRaises(NotImplementedError):
            ht.histogram(a, normed=True)
        with self.assertRaises(NotImplementedError):
            ht.histogram(a, density=True)

    def test_kurtosis(self):
        x = ht.zeros((2, 3, 4))
        with self.assertRaises(ValueError):
            x.kurtosis(axis=10)
        with self.assertRaises(TypeError):
            ht.kurtosis(x, axis="01")
        with self.assertRaises(TypeError):
            ht.kurtosis(x, axis=(0, "10"))

        def __split_calc(ht_split, axis):
            sp = ht_split if axis > ht_split else ht_split - 1
            if axis == ht_split:
                sp = None
            return sp

        # 1 dim
        ht_data = ht.random.rand(50)
        np_data = ht_data.copy().numpy()
        np_kurtosis32 = ht.array((ss.kurtosis(np_data, bias=False)), dtype=ht_data.dtype)
        self.assertAlmostEqual(ht.kurtosis(ht_data), np_kurtosis32.item(), places=5)
        ht_data = ht.resplit(ht_data, 0)
        self.assertAlmostEqual(ht.kurtosis(ht_data), np_kurtosis32.item(), places=5)

        # 2 dim
        ht_data = ht.random.rand(50, 30)
        np_data = ht_data.copy().numpy()
        np_kurtosis32 = ss.kurtosis(np_data, axis=None, bias=False)
        self.assertAlmostEqual(ht.kurtosis(ht_data) - np_kurtosis32, 0, places=5)
        ht_data = ht.resplit(ht_data, 0)
        for ax in range(2):
            np_kurtosis32 = ht.array(
                (ss.kurtosis(np_data, axis=ax, bias=True)), dtype=ht_data.dtype
            )
            ht_kurtosis = ht.kurtosis(ht_data, axis=ax, unbiased=False)
            self.assertTrue(ht.allclose(ht_kurtosis, np_kurtosis32, atol=1e-5))
            sp = __split_calc(ht_data.split, ax)
            self.assertEqual(ht_kurtosis.split, sp)
        ht_data = ht.resplit(ht_data, 1)
        for ax in range(2):
            np_kurtosis32 = ht.array(
                (ss.kurtosis(np_data, axis=ax, bias=True)), dtype=ht_data.dtype
            )
            ht_kurtosis = ht.kurtosis(ht_data, axis=ax, unbiased=False)
            self.assertTrue(ht.allclose(ht_kurtosis, np_kurtosis32, atol=1e-5))
            sp = __split_calc(ht_data.split, ax)
            self.assertEqual(ht_kurtosis.split, sp)

        # 2 dim float64
        ht_data = ht.random.rand(50, 30, dtype=ht.float64)
        np_data = ht_data.copy().numpy()
        np_kurtosis32 = ss.kurtosis(np_data, axis=None, bias=False)
        self.assertAlmostEqual(ht.kurtosis(ht_data) - np_kurtosis32, 0, places=5)
        ht_data = ht.resplit(ht_data, 0)
        for ax in range(2):
            np_kurtosis32 = ht.array(
                (ss.kurtosis(np_data, axis=ax, bias=False)), dtype=ht_data.dtype
            )
            ht_kurtosis = ht.kurtosis(ht_data, axis=ax)
            self.assertTrue(ht.allclose(ht_kurtosis, np_kurtosis32, atol=1e-5))
            sp = __split_calc(ht_data.split, ax)
            self.assertEqual(ht_kurtosis.split, sp)
            self.assertEqual(ht_kurtosis.dtype, ht.float64)
        ht_data = ht.resplit(ht_data, 1)
        for ax in range(2):
            np_kurtosis32 = ht.array(
                (ss.kurtosis(np_data, axis=ax, bias=False)), dtype=ht_data.dtype
            )
            ht_kurtosis = ht.kurtosis(ht_data, axis=ax)
            self.assertTrue(ht.allclose(ht_kurtosis, np_kurtosis32, atol=1e-5))
            sp = __split_calc(ht_data.split, ax)
            self.assertEqual(ht_kurtosis.split, sp)
            self.assertEqual(ht_kurtosis.dtype, ht.float64)

        # 3 dim
        ht_data = ht.random.rand(50, 30, 16)
        np_data = ht_data.copy().numpy()
        np_kurtosis32 = ss.kurtosis(np_data, axis=None, bias=False)
        self.assertAlmostEqual(ht.kurtosis(ht_data) - np_kurtosis32, 0, places=5)
        for split in range(3):
            ht_data = ht.resplit(ht_data, split)
            for ax in range(3):
                np_kurtosis32 = ht.array(
                    (ss.kurtosis(np_data, axis=ax, bias=False)), dtype=ht_data.dtype
                )
                ht_kurtosis = ht.kurtosis(ht_data, axis=ax)
                self.assertTrue(ht.allclose(ht_kurtosis, np_kurtosis32, atol=1e-5))
                sp = __split_calc(ht_data.split, ax)
                self.assertEqual(ht_kurtosis.split, sp)

    def test_max(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

        ht_array = ht.array(data)
        comparison = torch.tensor(data, device=self.device.torch_device)

        # check global max
        maximum = ht.max(ht_array)

        self.assertIsInstance(maximum, ht.DNDarray)
        self.assertEqual(maximum.shape, ())
        self.assertEqual(maximum.lshape, ())
        self.assertEqual(maximum.split, None)
        self.assertEqual(maximum.dtype, ht.int64)
        self.assertEqual(maximum.larray.dtype, torch.int64)
        self.assertEqual(maximum, 12)

        # maximum along first axis
        maximum_vertical = ht.max(ht_array, axis=0)

        self.assertIsInstance(maximum_vertical, ht.DNDarray)
        self.assertEqual(maximum_vertical.shape, (3,))
        self.assertEqual(maximum_vertical.lshape, (3,))
        self.assertEqual(maximum_vertical.split, None)
        self.assertEqual(maximum_vertical.dtype, ht.int64)
        self.assertEqual(maximum_vertical.larray.dtype, torch.int64)
        self.assertTrue((maximum_vertical.larray == comparison.max(dim=0, keepdim=True)[0]).all())

        # maximum along second axis
        maximum_horizontal = ht.max(ht_array, axis=1, keepdims=True)

        self.assertIsInstance(maximum_horizontal, ht.DNDarray)
        self.assertEqual(maximum_horizontal.shape, (4, 1))
        self.assertEqual(maximum_horizontal.lshape, (4, 1))
        self.assertEqual(maximum_horizontal.split, None)
        self.assertEqual(maximum_horizontal.dtype, ht.int64)
        self.assertEqual(maximum_horizontal.larray.dtype, torch.int64)
        self.assertTrue((maximum_horizontal.larray == comparison.max(dim=1, keepdim=True)[0]).all())

        # check max over all float elements of split 3d tensor, across split axis
        size = ht.MPI_WORLD.size
        random_volume = ht.random.randn(3, 3 * size, 3, split=1)
        maximum_volume = ht.max(random_volume, axis=1)

        self.assertIsInstance(maximum_volume, ht.DNDarray)
        self.assertEqual(maximum_volume.shape, (3, 3))
        self.assertEqual(maximum_volume.lshape, (3, 3))
        self.assertEqual(maximum_volume.dtype, ht.float32)
        self.assertEqual(maximum_volume.larray.dtype, torch.float32)
        self.assertEqual(maximum_volume.split, None)

        # check max over all float elements of split 3d tensor, tuple axis
        random_volume = ht.random.randn(3 * size, 3, 3, split=0)
        maximum_volume = ht.max(random_volume, axis=(1, 2))
        alt_maximum_volume = ht.max(random_volume, axis=(2, 1))

        self.assertIsInstance(maximum_volume, ht.DNDarray)
        self.assertEqual(maximum_volume.shape, (3 * size,))
        self.assertEqual(maximum_volume.dtype, ht.float32)
        self.assertEqual(maximum_volume.larray.dtype, torch.float32)
        self.assertEqual(maximum_volume.split, 0)
        self.assertTrue((maximum_volume == alt_maximum_volume).all())

        # check max over all float elements of split 5d tensor, along split axis
        random_5d = ht.random.randn(1 * size, 2, 3, 4, 5, split=0)
        maximum_5d = ht.max(random_5d, axis=1)

        self.assertIsInstance(maximum_5d, ht.DNDarray)
        self.assertEqual(maximum_5d.shape, (1 * size, 3, 4, 5))
        self.assertLessEqual(maximum_5d.lshape[1], 3)
        self.assertEqual(maximum_5d.dtype, ht.float32)
        self.assertEqual(maximum_5d.larray.dtype, torch.float32)
        self.assertEqual(maximum_5d.split, 0)

        # Calculating max with empty local vectors works
        if size > 1:
            a = ht.arange(size - 1, split=0)
            res = ht.max(a)
            expected = torch.tensor(
                size - 2, dtype=a.dtype.torch_type(), device=self.device.torch_device
            )
            self.assertTrue(torch.equal(res.larray, expected))

        # check exceptions
        with self.assertRaises(TypeError):
            ht_array.max(axis=1.1)
        with self.assertRaises(TypeError):
            ht_array.max(axis="y")
        with self.assertRaises(ValueError):
            ht.max(ht_array, axis=-4)

    def test_maximum(self):
        data1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        data2 = [[0, 3, 2], [5, 4, 7], [6, 9, 8], [9, 10, 11]]

        ht_array1 = ht.array(data1)
        ht_array2 = ht.array(data2)
        comparison1 = torch.tensor(data1, device=self.device.torch_device)
        comparison2 = torch.tensor(data2, device=self.device.torch_device)

        # check maximum
        maximum = ht.maximum(ht_array1, ht_array2)

        self.assertIsInstance(maximum, ht.DNDarray)
        self.assertEqual(maximum.shape, (4, 3))
        self.assertEqual(maximum.lshape, (4, 3))
        self.assertEqual(maximum.split, None)
        self.assertEqual(maximum.dtype, ht.int64)
        self.assertEqual(maximum.larray.dtype, torch.int64)
        self.assertTrue((maximum.larray == torch.max(comparison1, comparison2)).all())

        # check maximum over float elements of split 3d tensors
        torch.manual_seed(1)
        random_volume_1 = ht.random.randn(6, 3, 3, split=0)
        random_volume_2 = ht.random.randn(6, 1, 3, split=0)
        maximum_volume = ht.maximum(random_volume_1, random_volume_2)
        np_maximum = np.maximum(random_volume_1.numpy(), random_volume_2.numpy())

        self.assertIsInstance(maximum_volume, ht.DNDarray)
        self.assertEqual(maximum_volume.shape, (6, 3, 3))
        self.assertEqual(maximum_volume.dtype, ht.float32)
        self.assertEqual(maximum_volume.larray.dtype, torch.float32)
        self.assertEqual(maximum_volume.split, random_volume_1.split)
        self.assertTrue((maximum_volume.numpy() == np_maximum).all())

        # check maximum against size-1 array
        random_volume_1_split_none = ht.random.randn(1, split=None, dtype=ht.float64)
        random_volume_2_splitdiff = ht.random.randn(3, 3, 4, split=1)
        maximum_volume_splitdiff = ht.maximum(random_volume_1_split_none, random_volume_2_splitdiff)
        self.assertEqual(maximum_volume_splitdiff.split, 1)
        self.assertEqual(maximum_volume_splitdiff.dtype, ht.float64)

        random_volume_1_split_none = ht.random.randn(3, 3, 4, split=0)
        random_volume_2_splitdiff = ht.random.randn(1, split=None)
        maximum_volume_splitdiff = ht.maximum(random_volume_1_split_none, random_volume_2_splitdiff)
        self.assertEqual(maximum_volume_splitdiff.split, 0)

        # check maximum against scalar
        scalar = 5
        random_volume_2_splitdiff = ht.random.randn(3, 3, 4, split=1)
        maximum_volume_splitdiff = ht.maximum(scalar, random_volume_2_splitdiff)
        self.assertEqual(maximum_volume_splitdiff.split, 1)
        self.assertEqual(maximum_volume_splitdiff.dtype, ht.float32)

        scalar = 5.0
        maximum_volume_splitdiff = ht.maximum(random_volume_2_splitdiff, scalar)
        self.assertEqual(maximum_volume_splitdiff.split, 1)
        self.assertEqual(maximum_volume_splitdiff.dtype, ht.float32)

        # check output buffer
        out_shape = ht.stride_tricks.broadcast_shape(random_volume_1.gshape, random_volume_2.gshape)
        output = ht.empty(out_shape, split=0, dtype=ht.float32)
        ht.maximum(random_volume_1, random_volume_2, out=output)
        self.assertIsInstance(output, ht.DNDarray)
        self.assertEqual(output.shape, (6, 3, 3))
        self.assertEqual(output.dtype, ht.float32)
        self.assertEqual(output.larray.dtype, torch.float32)

        # check exceptions
        random_volume_3 = ht.array([])
        with self.assertRaises(ValueError):
            ht.maximum(random_volume_1, random_volume_3)
        random_volume_4 = ht.random.randn(4, 2, 3, split=0)
        with self.assertRaises(ValueError):
            ht.maximum(random_volume_1, random_volume_4)
        random_volume_5 = torch.ones(12, 3, 3, device=self.device.torch_device)
        with self.assertRaises(TypeError):
            ht.maximum(random_volume_1, random_volume_5)
        if ht.MPI_WORLD.size > 1:
            random_volume_6 = ht.random.randn(6, 3, 3, split=1)
            with self.assertRaises(NotImplementedError):
                ht.maximum(random_volume_1, random_volume_6)
        output1 = torch.ones(12, 3, 3, device=self.device.torch_device)
        with self.assertRaises(TypeError):
            ht.maximum(random_volume_1, random_volume_2, out=output1)
        output2 = ht.ones((12, 4, 3))
        with self.assertRaises(ValueError):
            ht.maximum(random_volume_1, random_volume_2, out=output2)
        output3 = ht.ones((6, 3, 3), split=1)
        with self.assertRaises(ValueError):
            ht.maximum(random_volume_1, random_volume_2, out=output3)

    def test_mean(self):
        array_0_len = 5
        array_1_len = 5
        array_2_len = 5

        x = ht.zeros((2, 3, 4))
        with self.assertRaises(ValueError):
            x.mean(axis=10)
        with self.assertRaises(ValueError):
            x.mean(axis=[4])
        with self.assertRaises(ValueError):
            x.mean(axis=[-4])
        with self.assertRaises(TypeError):
            ht.mean(x, axis="01")
        with self.assertRaises(ValueError):
            ht.mean(x, axis=(0, "10"))
        with self.assertRaises(ValueError):
            ht.mean(x, axis=(0, 0))
        with self.assertRaises(ValueError):
            ht.mean(x, axis=torch.Tensor([0, 0]))

        a = ht.arange(1, 5)
        self.assertEqual(a.mean(), 2.5)

        # ones
        dimensions = []

        for d in [array_0_len, array_1_len, array_2_len]:
            dimensions.extend([d])
            hold = list(range(len(dimensions)))
            hold.append(None)
            for split in hold:  # loop over the number of split dimension of the test array
                z = ht.ones(dimensions, split=split)
                res = z.mean()
                total_dims_list = list(z.shape)
                self.assertTrue((res == 1).all())
                for it in range(len(z.shape)):  # loop over the different single dimensions for mean
                    res = z.mean(axis=it)
                    self.assertTrue((res == 1).all())
                    target_dims = [
                        total_dims_list[q] for q in range(len(total_dims_list)) if q != it
                    ]
                    if not target_dims:
                        target_dims = ()
                    self.assertEqual(res.gshape, tuple(target_dims))
                    if z.split is None:
                        sp = None
                    else:
                        sp = z.split if it > z.split else z.split - 1
                        if it == split:
                            sp = None
                    self.assertEqual(res.split, sp)
                loop_list = [
                    ",".join(map(str, comb)) for comb in combinations(list(range(len(z.shape))), 2)
                ]

                for it in loop_list:  # loop over the different combinations of dimensions for mean
                    lp_split = [int(q) for q in it.split(",")]
                    res = z.mean(axis=lp_split)
                    self.assertTrue((res == 1).all())
                    target_dims = [
                        total_dims_list[q] for q in range(len(total_dims_list)) if q not in lp_split
                    ]
                    if not target_dims:
                        target_dims = (1,)
                    if res.gshape:
                        self.assertEqual(res.gshape, tuple(target_dims))
                    if res.split is not None:
                        if any([split >= x for x in lp_split]):
                            self.assertEqual(res.split, len(target_dims) - 1)
                        else:
                            self.assertEqual(res.split, z.split)

        # values for the iris dataset mean measured by libreoffice calc
        ax0 = ht.array([5.84333333333333, 3.054, 3.75866666666667, 1.19866666666667])
        for sp in [None, 0, 1]:
            iris = ht.load("heat/datasets/iris.csv", sep=";", split=sp)
            self.assertTrue(ht.allclose(ht.mean(iris), 3.46366666666667))
            self.assertTrue(ht.allclose(ht.mean(iris, axis=0), ax0))

    def test_min(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

        ht_array = ht.array(data)
        comparison = torch.tensor(data, device=self.device.torch_device)

        # check global min
        minimum = ht.min(ht_array)

        self.assertIsInstance(minimum, ht.DNDarray)
        self.assertEqual(minimum.shape, ())
        self.assertEqual(minimum.lshape, ())
        self.assertEqual(minimum.split, None)
        self.assertEqual(minimum.dtype, ht.int64)
        self.assertEqual(minimum.larray.dtype, torch.int64)
        self.assertEqual(minimum, 1)

        # min along first axis
        ht_array = ht.array(data, dtype=ht.int8)
        minimum_vertical = ht.min(ht_array, axis=0)

        self.assertIsInstance(minimum_vertical, ht.DNDarray)
        self.assertEqual(minimum_vertical.shape, (3,))
        self.assertEqual(minimum_vertical.lshape, (3,))
        self.assertEqual(minimum_vertical.split, None)
        self.assertEqual(minimum_vertical.dtype, ht.int8)
        self.assertEqual(minimum_vertical.larray.dtype, torch.int8)
        self.assertTrue((minimum_vertical.larray == comparison.min(dim=0, keepdim=True)[0]).all())

        # min along second axis
        ht_array = ht.array(data, dtype=ht.int16)
        minimum_horizontal = ht.min(ht_array, axis=1, keepdims=True)

        self.assertIsInstance(minimum_horizontal, ht.DNDarray)
        self.assertEqual(minimum_horizontal.shape, (4, 1))
        self.assertEqual(minimum_horizontal.lshape, (4, 1))
        self.assertEqual(minimum_horizontal.split, None)
        self.assertEqual(minimum_horizontal.dtype, ht.int16)
        self.assertEqual(minimum_horizontal.larray.dtype, torch.int16)
        self.assertTrue((minimum_horizontal.larray == comparison.min(dim=1, keepdim=True)[0]).all())

        # check min over all float elements of split 3d tensor, across split axis
        size = ht.MPI_WORLD.size
        random_volume = ht.random.randn(3, 3 * size, 3, split=1)
        minimum_volume = ht.min(random_volume, axis=1)

        self.assertIsInstance(minimum_volume, ht.DNDarray)
        self.assertEqual(minimum_volume.shape, (3, 3))
        self.assertEqual(minimum_volume.lshape, (3, 3))
        self.assertEqual(minimum_volume.dtype, ht.float32)
        self.assertEqual(minimum_volume.larray.dtype, torch.float32)
        self.assertEqual(minimum_volume.split, None)

        # check min over all float elements of split 3d tensor, tuple axis
        random_volume = ht.random.randn(3 * size, 3, 3, split=0)
        minimum_volume = ht.min(random_volume, axis=(1, 2))
        alt_minimum_volume = ht.min(random_volume, axis=(2, 1))

        self.assertIsInstance(minimum_volume, ht.DNDarray)
        self.assertEqual(minimum_volume.shape, (3 * size,))
        self.assertEqual(minimum_volume.dtype, ht.float32)
        self.assertEqual(minimum_volume.larray.dtype, torch.float32)
        self.assertEqual(minimum_volume.split, 0)
        self.assertTrue((minimum_volume == alt_minimum_volume).all())

        # check min over all float elements of split 5d tensor, along split axis
        random_5d = ht.random.randn(1 * size, 2, 3, 4, 5, split=0)
        minimum_5d = ht.min(random_5d, axis=1)

        self.assertIsInstance(minimum_5d, ht.DNDarray)
        self.assertEqual(minimum_5d.shape, (1 * size, 3, 4, 5))
        self.assertLessEqual(minimum_5d.lshape[1], 3)
        self.assertEqual(minimum_5d.dtype, ht.float32)
        self.assertEqual(minimum_5d.larray.dtype, torch.float32)
        self.assertEqual(minimum_5d.split, 0)

        # Calculating min with empty local vectors works
        size = ht.MPI_WORLD.size
        if size > 1:
            a = ht.arange(size - 1, split=0)
            res = ht.min(a)
            expected = torch.tensor(0, dtype=a.dtype.torch_type(), device=self.device.torch_device)
            self.assertTrue(torch.equal(res.larray, expected))

        # check exceptions
        with self.assertRaises(TypeError):
            ht_array.min(axis=1.1)
        with self.assertRaises(TypeError):
            ht_array.min(axis="y")
        with self.assertRaises(ValueError):
            ht.min(ht_array, axis=-4)

    def test_minimum(self):
        data1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        data2 = [[0, 3, 2], [5, 4, 7], [6, 9, 8], [9, 10, 11]]

        ht_array1 = ht.array(data1)
        ht_array2 = ht.array(data2)
        comparison1 = torch.tensor(data1, device=self.device.torch_device)
        comparison2 = torch.tensor(data2, device=self.device.torch_device)

        # check minimum
        minimum = ht.minimum(ht_array1, ht_array2)

        self.assertIsInstance(minimum, ht.DNDarray)
        self.assertEqual(minimum.shape, (4, 3))
        self.assertEqual(minimum.lshape, (4, 3))
        self.assertEqual(minimum.split, None)
        self.assertEqual(minimum.dtype, ht.int64)
        self.assertEqual(minimum.larray.dtype, torch.int64)
        self.assertTrue((minimum.larray == torch.min(comparison1, comparison2)).all())

        # check minimum over float elements of split 3d tensors
        torch.manual_seed(1)
        random_volume_1 = ht.random.randn(6, 3, 3, split=0)
        random_volume_2 = ht.random.randn(6, 1, 3, split=0)
        minimum_volume = ht.minimum(random_volume_1, random_volume_2)
        np_minimum = np.minimum(random_volume_1.numpy(), random_volume_2.numpy())

        self.assertIsInstance(minimum_volume, ht.DNDarray)
        self.assertEqual(minimum_volume.shape, (6, 3, 3))
        self.assertEqual(minimum_volume.dtype, ht.float32)
        self.assertEqual(minimum_volume.larray.dtype, torch.float32)
        self.assertEqual(minimum_volume.split, random_volume_1.split)
        self.assertTrue((minimum_volume.numpy() == np_minimum).all())

        # check minimum against size-1 array
        random_volume_1_split_none = ht.random.randn(1, split=None, dtype=ht.float64)
        random_volume_2_splitdiff = ht.random.randn(3, 3, 4, split=1)
        minimum_volume_splitdiff = ht.minimum(random_volume_1_split_none, random_volume_2_splitdiff)
        self.assertEqual(minimum_volume_splitdiff.split, 1)
        self.assertEqual(minimum_volume_splitdiff.dtype, ht.float64)

        random_volume_1_split_none = ht.random.randn(3, 3, 4, split=0)
        random_volume_2_splitdiff = ht.random.randn(1, split=None)
        minimum_volume_splitdiff = ht.minimum(random_volume_1_split_none, random_volume_2_splitdiff)
        self.assertEqual(minimum_volume_splitdiff.split, 0)

        # check minimum against scalar
        scalar = 5
        random_volume_2_splitdiff = ht.random.randn(3, 3, 4, split=1)
        minimum_volume_splitdiff = ht.minimum(scalar, random_volume_2_splitdiff)
        self.assertEqual(minimum_volume_splitdiff.split, 1)
        self.assertEqual(minimum_volume_splitdiff.dtype, ht.float32)

        scalar = 5.0
        minimum_volume_splitdiff = ht.minimum(random_volume_2_splitdiff, scalar)
        self.assertEqual(minimum_volume_splitdiff.split, 1)
        self.assertEqual(minimum_volume_splitdiff.dtype, ht.float32)

        # check output buffer
        out_shape = ht.stride_tricks.broadcast_shape(random_volume_1.gshape, random_volume_2.gshape)
        output = ht.empty(out_shape, split=0, dtype=ht.float32)
        ht.minimum(random_volume_1, random_volume_2, out=output)
        self.assertIsInstance(output, ht.DNDarray)
        self.assertEqual(output.shape, (6, 3, 3))
        self.assertEqual(output.dtype, ht.float32)
        self.assertEqual(output.larray.dtype, torch.float32)

        # check exceptions
        random_volume_3 = ht.array([])
        with self.assertRaises(ValueError):
            ht.minimum(random_volume_1, random_volume_3)
        random_volume_3 = ht.random.randn(4, 2, 3, split=0)
        with self.assertRaises(ValueError):
            ht.minimum(random_volume_1, random_volume_3)
        random_volume_3 = torch.ones(12, 3, 3, device=self.device.torch_device)
        with self.assertRaises(TypeError):
            ht.minimum(random_volume_1, random_volume_3)
        random_volume_3 = np.array(7.2)
        with self.assertRaises(TypeError):
            ht.minimum(random_volume_3, random_volume_1)
        if ht.MPI_WORLD.size > 1:
            random_volume_3 = ht.random.randn(6, 3, 3, split=1)
            with self.assertRaises(NotImplementedError):
                ht.minimum(random_volume_1, random_volume_3)
        output = torch.ones(12, 3, 3, device=self.device.torch_device)
        with self.assertRaises(TypeError):
            ht.minimum(random_volume_1, random_volume_2, out=output)
        output = ht.ones((12, 4, 3))
        with self.assertRaises(ValueError):
            ht.minimum(random_volume_1, random_volume_2, out=output)
        output = ht.ones((6, 3, 3), split=1)
        with self.assertRaises(ValueError):
            ht.minimum(random_volume_1, random_volume_2, out=output)
        output = ht.ones((6, 3, 3), split=None, comm=ht.MPI_SELF)
        with self.assertRaises(ValueError):
            ht.minimum(random_volume_1, random_volume_2, out=output)
        output = ht.ones((6, 3, 3), split=0, comm=ht.MPI_SELF)
        with self.assertRaises(NotImplementedError):
            ht.minimum(random_volume_1, random_volume_2, out=output)

    def test_percentile(self):
        # test local, distributed, split/axis combination, no data on process
        x_np = np.arange(3 * 10 * 10).reshape(3, 10, 10)
        x_ht = ht.array(x_np)
        x_ht_split0 = ht.array(x_np, split=0)
        x_ht_split1 = ht.array(x_np, split=1)
        x_ht_split2 = ht.array(x_np, split=2)
        q = 15.9
        for dim in range(x_ht.ndim):
            p_np = np.percentile(x_np, q, axis=dim)
            p_ht = ht.percentile(x_ht, q, axis=dim)
            p_ht_split0 = ht.percentile(x_ht_split0, q, axis=dim)
            p_ht_split1 = ht.percentile(x_ht_split1, q, axis=dim)
            p_ht_split2 = ht.percentile(x_ht_split2, q, axis=dim)
            self.assert_array_equal(p_ht, p_np)
            self.assert_array_equal(p_ht_split0, p_np)
            self.assert_array_equal(p_ht_split1, p_np)
            self.assert_array_equal(p_ht_split2, p_np)

        # test x, q dtypes combination plus edge-case 100th percentile
        q = 100
        p_np = np.percentile(x_np, q, axis=0)
        p_ht = ht.percentile(x_ht, q, axis=0)
        self.assertTrue((p_ht.numpy() == p_np).all())

        # test median (q = 50)
        q = 50
        p_np = np.percentile(x_np, q, axis=0)
        p_ht = ht.median(x_ht_split0, axis=0)
        self.assertTrue((p_ht.numpy() == p_np).all())

        # test list q and writing to output buffer
        q = [0.1, 2.3, 15.9, 50.0, 84.1, 97.7, 99.9]
        axis = 2
        try:
            p_np = np.percentile(x_np, q, axis=axis, method="lower", keepdims=True)
        except TypeError:
            p_np = np.percentile(x_np, q, axis=axis, interpolation="lower", keepdims=True)
        p_ht = ht.percentile(x_ht, q, axis=axis, interpolation="lower", keepdims=True)
        out = ht.empty(p_np.shape, dtype=ht.float32, split=None, device=x_ht.device)
        ht.percentile(x_ht, q, axis=axis, out=out, interpolation="lower", keepdims=True)
        self.assertEqual(p_ht.numpy()[5].all(), p_np[5].all())
        self.assertEqual(out.numpy()[2].all(), p_np[2].all())
        self.assertTrue(p_ht.shape == p_np.shape)
        axis = None
        try:
            p_np = np.percentile(x_np, q, axis=axis, method="higher")
        except TypeError:
            p_np = np.percentile(x_np, q, axis=axis, interpolation="higher")
        p_ht = ht.percentile(x_ht, q, axis=axis, interpolation="higher")
        self.assertEqual(p_ht.numpy()[6], p_np[6])
        self.assertTrue(p_ht.shape == p_np.shape)
        try:
            p_np = np.percentile(x_np, q, axis=axis, method="nearest")
        except TypeError:
            p_np = np.percentile(x_np, q, axis=axis, interpolation="nearest")
        p_ht = ht.percentile(x_ht, q, axis=axis, interpolation="nearest")
        self.assertEqual(p_ht.numpy()[2], p_np[2])

        # test split q
        q_ht = ht.array(q, split=0, comm=x_ht.comm)
        try:
            p_np = np.percentile(x_np, q, axis=axis, method="midpoint")
        except TypeError:
            p_np = np.percentile(x_np, q, axis=axis, interpolation="midpoint")
        p_ht = ht.percentile(x_ht, q_ht, axis=axis, interpolation="midpoint")
        self.assertEqual(p_ht.numpy()[4], p_np[4])

        # test scalar x
        x_sc = ht.array(4.5)
        p_ht = ht.percentile(x_sc, q=q)
        p_np = np.percentile(4.5, q=q)
        self.assertEqual(p_ht.numpy().all(), p_np.all())

        # test exceptions
        with self.assertRaises(TypeError):
            ht.percentile(x_np, q)
        with self.assertRaises(ValueError):
            ht.percentile(x_ht, q, interpolation="Homer!")
        with self.assertRaises(NotImplementedError):
            ht.percentile(x_ht, q, axis=(0, 1))
        q_np = np.array(q)
        with self.assertRaises(TypeError):
            ht.percentile(x_ht, q_np)
        t_out = torch.empty((len(q),), dtype=torch.float64)
        with self.assertRaises(TypeError):
            ht.percentile(x_ht, q, out=t_out)
        out_wrong_dtype = ht.empty((len(q),), dtype=ht.float64)
        with self.assertRaises(TypeError):
            ht.percentile(x_ht, q, out=out_wrong_dtype)
        out_wrong_shape = ht.empty((len(q) + 1,), dtype=ht.float32)
        with self.assertRaises(ValueError):
            ht.percentile(x_ht, q, out=out_wrong_shape)
        out_wrong_split = ht.empty((len(q),), dtype=ht.float32, split=0)
        with self.assertRaises(ValueError):
            ht.percentile(x_ht, q, out=out_wrong_split)

    def test_skew(self):
        x = ht.zeros((2, 3, 4))
        with self.assertRaises(ValueError):
            x.skew(axis=10)
        with self.assertRaises(TypeError):
            x.skew(axis=[1, 0])

        a = ht.arange(1, 5)
        self.assertEqual(a.skew(), 0.0)

        def __split_calc(ht_split, axis):
            sp = ht_split if axis > ht_split else ht_split - 1
            if axis == ht_split:
                sp = None
            return sp

        # 1 dim
        ht_data = ht.random.rand(50)
        np_data = ht_data.copy().numpy()
        np_skew32 = ht.array((ss.skew(np_data, bias=False)).astype(np.float32), dtype=ht_data.dtype)
        self.assertAlmostEqual(ht.skew(ht_data), np_skew32.item(), places=5)
        ht_data = ht.resplit(ht_data, 0)
        self.assertAlmostEqual(ht.skew(ht_data), np_skew32.item(), places=5)

        # 2 dim
        ht_data = ht.random.rand(50, 30)
        np_data = ht_data.copy().numpy()
        np_skew32 = ss.skew(np_data, axis=None, bias=True)
        self.assertAlmostEqual(ht.skew(ht_data, unbiased=False) - np_skew32, 0, places=5)
        ht_data = ht.resplit(ht_data, 0)
        for ax in range(2):
            np_skew32 = ht.array((ss.skew(np_data, axis=ax, bias=True)), dtype=ht_data.dtype)
            ht_skew = ht.skew(ht_data, axis=ax, unbiased=False)
            self.assertTrue(ht.allclose(ht_skew, np_skew32, atol=1e-5))
            sp = __split_calc(ht_data.split, ax)
            self.assertEqual(ht_skew.split, sp)
        ht_data = ht.resplit(ht_data, 1)
        for ax in range(2):
            np_skew32 = ht.array((ss.skew(np_data, axis=ax, bias=True)), dtype=ht_data.dtype)
            ht_skew = ht.skew(ht_data, axis=ax, unbiased=False)
            self.assertTrue(ht.allclose(ht_skew, np_skew32, atol=1e-5))
            sp = __split_calc(ht_data.split, ax)
            self.assertEqual(ht_skew.split, sp)

        # 2 dim float64
        ht_data = ht.random.rand(50, 30, dtype=ht.float64)
        np_data = ht_data.copy().numpy()
        np_skew32 = ss.skew(np_data, axis=None, bias=False)
        self.assertAlmostEqual(ht.skew(ht_data) - np_skew32, 0, places=5)
        ht_data = ht.resplit(ht_data, 0)
        for ax in range(2):
            np_skew32 = ht.array((ss.skew(np_data, axis=ax, bias=False)), dtype=ht_data.dtype)
            ht_skew = ht.skew(ht_data, axis=ax)
            self.assertTrue(ht.allclose(ht_skew, np_skew32, atol=1e-5))
            sp = __split_calc(ht_data.split, ax)
            self.assertEqual(ht_skew.split, sp)
            self.assertEqual(ht_skew.dtype, ht.float64)
        ht_data = ht.resplit(ht_data, 1)
        for ax in range(2):
            np_skew32 = ht.array((ss.skew(np_data, axis=ax, bias=False)), dtype=ht_data.dtype)
            ht_skew = ht.skew(ht_data, axis=ax)
            self.assertTrue(ht.allclose(ht_skew, np_skew32, atol=1e-5))
            sp = __split_calc(ht_data.split, ax)
            self.assertEqual(ht_skew.split, sp)
            self.assertEqual(ht_skew.dtype, ht.float64)

        # 3 dim
        ht_data = ht.random.rand(50, 30, 16)
        np_data = ht_data.copy().numpy()
        np_skew32 = ss.skew(np_data, axis=None, bias=False)
        self.assertAlmostEqual(ht.skew(ht_data) - np_skew32, 0, places=5)
        for split in range(3):
            ht_data = ht.resplit(ht_data, split)
            for ax in range(3):
                np_skew32 = ht.array((ss.skew(np_data, axis=ax, bias=False)), dtype=ht_data.dtype)
                ht_skew = ht.skew(ht_data, axis=ax)
                self.assertTrue(ht.allclose(ht_skew, np_skew32, atol=1e-5))
                sp = __split_calc(ht_data.split, ax)
                self.assertEqual(ht_skew.split, sp)

    def test_std(self):
        # test basics
        a = ht.arange(1, 5)
        self.assertAlmostEqual(a.std(), 1.118034)
        self.assertAlmostEqual(a.std(bessel=True), 1.2909944)

        # test raises
        x = ht.zeros((2, 3, 4))
        with self.assertRaises(TypeError):
            ht.std(x, axis=0, ddof=1.0)
        with self.assertRaises((ValueError, IndexError)):
            ht.std(x, axis=10)
        with self.assertRaises(TypeError):
            ht.std(x, axis="01")
        with self.assertRaises(ValueError):
            ht.std(x, ddof=-2)

        # the rest of the tests are covered by var

    def test_var(self):
        array_0_len = ht.MPI_WORLD.size * 2
        array_1_len = ht.MPI_WORLD.size * 2
        array_2_len = ht.MPI_WORLD.size * 2

        # test raises
        x = ht.zeros((2, 3, 4))
        with self.assertRaises(ValueError):
            x.var(axis=10)
        with self.assertRaises(ValueError):
            x.var(axis=[4])
        with self.assertRaises(ValueError):
            x.var(axis=[-4])
        with self.assertRaises(TypeError):
            ht.var(x, axis="01")
        with self.assertRaises(TypeError):
            ht.var(x, ddof="01")
        with self.assertRaises(ValueError):
            ht.var(x, axis=(0, "10"))
        with self.assertRaises(ValueError):
            ht.var(x, axis=(0, 0))
        with self.assertRaises(NotImplementedError):
            ht.var(x, ddof=2)
        with self.assertRaises(ValueError):
            ht.var(x, ddof=-2)
        with self.assertRaises(ValueError):
            ht.var(x, axis=torch.Tensor([0, 0]))

        a = ht.arange(1, 5)
        self.assertEqual(a.var(ddof=1), 1.666666666666666)

        # ones
        dimensions = []
        for d in [array_0_len, array_1_len, array_2_len]:
            dimensions.extend([d])
            hold = list(range(len(dimensions)))
            hold.append(None)
            for split in hold:  # loop over the number of dimensions of the test array
                z = ht.ones(dimensions, split=split)
                res = z.var(ddof=0)
                total_dims_list = list(z.shape)
                self.assertTrue((res == 0).all())
                # loop over the different single dimensions for var
                for it in range(len(z.shape)):
                    res = z.var(axis=it)
                    self.assertTrue(ht.allclose(res, 0))
                    target_dims = [
                        total_dims_list[q] for q in range(len(total_dims_list)) if q != it
                    ]
                    if not target_dims:
                        target_dims = ()
                    self.assertEqual(res.gshape, tuple(target_dims))
                    if z.split is None:
                        sp = None
                    else:
                        sp = z.split if it > z.split else z.split - 1
                        if it == split:
                            sp = None
                    self.assertEqual(res.split, sp)
                    if split == it:
                        res = z.var(axis=it)
                        self.assertTrue(ht.allclose(res, 0))
                loop_list = [
                    ",".join(map(str, comb)) for comb in combinations(list(range(len(z.shape))), 2)
                ]

                for it in loop_list:  # loop over the different combinations of dimensions for var
                    lp_split = [int(q) for q in it.split(",")]
                    res = z.var(axis=lp_split)
                    self.assertTrue((res == 0).all())
                    target_dims = [
                        total_dims_list[q] for q in range(len(total_dims_list)) if q not in lp_split
                    ]
                    if not target_dims:
                        target_dims = (1,)
                    if res.gshape:
                        self.assertEqual(res.gshape, tuple(target_dims))
                    if res.split is not None:
                        if any([split >= x for x in lp_split]):
                            self.assertEqual(res.split, len(target_dims) - 1)
                        else:
                            self.assertEqual(res.split, z.split)

        # values for the iris dataset var measured by libreoffice calc
        for sp in [None, 0, 1]:
            iris = ht.load("heat/datasets/iris.csv", sep=";", split=sp)
            self.assertTrue(ht.allclose(ht.var(iris, bessel=True), 3.90318519755147))
