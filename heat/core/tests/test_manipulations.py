import unittest
import torch

import heat as ht


class TestManipulations(unittest.TestCase):
    def test_expand_dims(self):
        # vector data
        a = ht.arange(10)
        b = ht.expand_dims(a, 0)

        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(len(b.shape), 2)

        self.assertEqual(b.shape[0], 1)
        self.assertEqual(b.shape[1], a.shape[0])

        self.assertEqual(b.lshape[0], 1)
        self.assertEqual(b.lshape[1], a.shape[0])

        self.assertIs(b.split, None)

        # vector data with out-of-bounds axis
        a = ht.arange(12)
        b = a.expand_dims(1)

        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(len(b.shape), 2)

        self.assertEqual(b.shape[0], a.shape[0])
        self.assertEqual(b.shape[1], 1)

        self.assertEqual(b.lshape[0], a.shape[0])
        self.assertEqual(b.lshape[1], 1)

        self.assertIs(b.split, None)

        # volume with intermediate axis
        a = ht.empty((3, 4, 5,))
        b = a.expand_dims(1)

        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(len(b.shape), 4)

        self.assertEqual(b.shape[0], a.shape[0])
        self.assertEqual(b.shape[1], 1)
        self.assertEqual(b.shape[2], a.shape[1])
        self.assertEqual(b.shape[3], a.shape[2])

        self.assertEqual(b.lshape[0], a.shape[0])
        self.assertEqual(b.lshape[1], 1)
        self.assertEqual(b.lshape[2], a.shape[1])
        self.assertEqual(b.lshape[3], a.shape[2])

        self.assertIs(b.split, None)

        # volume with negative axis
        a = ht.empty((3, 4, 5,))
        b = a.expand_dims(-4)

        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(len(b.shape), 4)

        self.assertEqual(b.shape[0], 1)
        self.assertEqual(b.shape[1], a.shape[0])
        self.assertEqual(b.shape[2], a.shape[1])
        self.assertEqual(b.shape[3], a.shape[2])

        self.assertEqual(b.lshape[0], 1)
        self.assertEqual(b.lshape[1], a.shape[0])
        self.assertEqual(b.lshape[2], a.shape[1])
        self.assertEqual(b.lshape[3], a.shape[2])

        self.assertIs(b.split, None)

        # split volume with negative axis expansion after the split
        a = ht.empty((3, 4, 5,), split=1)
        b = a.expand_dims(-2)

        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(len(b.shape), 4)

        self.assertEqual(b.shape[0], a.shape[0])
        self.assertEqual(b.shape[1], a.shape[1])
        self.assertEqual(b.shape[2], 1)
        self.assertEqual(b.shape[3], a.shape[2])

        self.assertEqual(b.lshape[0], a.shape[0])
        self.assertLessEqual(b.lshape[1], a.shape[1])
        self.assertEqual(b.lshape[2], 1)
        self.assertEqual(b.lshape[3], a.shape[2])

        self.assertIs(b.split, 1)

        # split volume with negative axis expansion before the split
        a = ht.empty((3, 4, 5,), split=2)
        b = a.expand_dims(-3)

        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(len(b.shape), 4)

        self.assertEqual(b.shape[0], a.shape[0])
        self.assertEqual(b.shape[1], 1)
        self.assertEqual(b.shape[2], a.shape[1])
        self.assertEqual(b.shape[3], a.shape[2])

        self.assertEqual(b.lshape[0], a.shape[0])
        self.assertEqual(b.lshape[1], 1)
        self.assertEqual(b.lshape[2], a.shape[1])
        self.assertLessEqual(b.lshape[3], a.shape[2])

        self.assertIs(b.split, 3)

        # exceptions
        with self.assertRaises(TypeError):
            ht.expand_dims('(3, 4, 5,)', 1)
        with self.assertRaises(TypeError):
            ht.empty((3, 4, 5,)).expand_dims('1')
        with self.assertRaises(ValueError):
            ht.empty((3, 4, 5,)).expand_dims(4)
        with self.assertRaises(ValueError):
            ht.empty((3, 4, 5,)).expand_dims(-5)

    def test_squeeze(self):
        torch.manual_seed(1)
        data = ht.random.randn(1, 4, 5, 1)

        # 4D local tensor, no axis
        result = ht.squeeze(data)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.float32)
        self.assertEqual(result._DNDarray__array.dtype, torch.float32)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == data._DNDarray__array.squeeze()).all())

        # 4D local tensor, major axis
        result = ht.squeeze(data, axis=0)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.float32)
        self.assertEqual(result._DNDarray__array.dtype, torch.float32)
        self.assertEqual(result.shape, (4, 5, 1))
        self.assertEqual(result.lshape, (4, 5, 1))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == data._DNDarray__array.squeeze(0)).all())

        # 4D local tensor, minor axis
        result = ht.squeeze(data, axis=-1)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.float32)
        self.assertEqual(result._DNDarray__array.dtype, torch.float32)
        self.assertEqual(result.shape, (1, 4, 5))
        self.assertEqual(result.lshape, (1, 4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == data._DNDarray__array.squeeze(-1)).all())

        # 4D local tensor, tuple axis
        result = data.squeeze(axis=(0, -1))
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.float32)
        self.assertEqual(result._DNDarray__array.dtype, torch.float32)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == data._DNDarray__array.squeeze()).all())

        # 4D split tensor, along the axis
        # TODO: reinstate this test of uneven dimensions distribution
        # after update to Allgatherv implementation (Issue  #273 depending on #233)
        # data = ht.array(ht.random.randn(1, 4, 5, 1), split=1)
        # result = ht.squeeze(data, axis=-1)
        # self.assertIsInstance(result, ht.DNDarray)
        # # TODO: the following works locally but not when distributed,
        # #self.assertEqual(result.dtype, ht.float32)
        # #self.assertEqual(result._DNDarray__array.dtype, torch.float32)
        # self.assertEqual(result.shape, (1, 12, 5))
        # self.assertEqual(result.lshape, (1, 12, 5))
        # self.assertEqual(result.split, 1)

        # 3D split tensor, across the axis
        size = ht.MPI_WORLD.size * 2
        data = ht.triu(ht.ones((1, size, size), split=1), k=1)

        result = ht.squeeze(data, axis=0)
        self.assertIsInstance(result, ht.DNDarray)
        # TODO: the following works locally but not when distributed,
        #self.assertEqual(result.dtype, ht.float32)
        #self.assertEqual(result._DNDarray__array.dtype, torch.float32)
        self.assertEqual(result.shape, (size, size))
        self.assertEqual(result.lshape, (size, size))
        #self.assertEqual(result.split, None)

        # check exceptions
        with self.assertRaises(ValueError):
            data.squeeze(axis=(0, 1))
        with self.assertRaises(TypeError):
            data.squeeze(axis=1.1)
        with self.assertRaises(TypeError):
            data.squeeze(axis='y')
        with self.assertRaises(ValueError):
            ht.argmin(data, axis=-4)

    def test_unique(self):
        size = ht.MPI_WORLD.size
        rank = ht.MPI_WORLD.rank
        torch_array = torch.arange(size, dtype=torch.int32).expand(size, size)
        split_zero = ht.array(torch_array, split=0)

        exp_axis_none = ht.arange(size, dtype=ht.int32)
        res = ht.unique(split_zero, sorted=True)
        self.assertTrue((res._DNDarray__array == exp_axis_none._DNDarray__array).all())

        exp_axis_zero = ht.arange(size, dtype=ht.int32).expand_dims(0)
        res = ht.unique(split_zero, sorted=True, axis=0)
        self.assertTrue((res._DNDarray__array == exp_axis_zero._DNDarray__array).all())

        exp_axis_one = ht.array([rank], dtype=ht.int32).expand_dims(0)
        split_zero_transposed = ht.array(torch_array.transpose(0, 1), split=0)
        res = ht.unique(split_zero_transposed, sorted=True, axis=1)
        self.assertTrue((res._DNDarray__array == exp_axis_one._DNDarray__array).all())

        split_one = ht.array(torch_array, dtype=ht.int32, split=1)

        exp_axis_none = ht.arange(size, dtype=ht.int32)
        res = ht.unique(split_one, sorted=True)
        self.assertTrue((res._DNDarray__array == exp_axis_none._DNDarray__array).all())

        exp_axis_zero = ht.array([rank], dtype=ht.int32).expand_dims(0)
        res = ht.unique(split_one, sorted=True, axis=0)
        self.assertTrue((res._DNDarray__array == exp_axis_zero._DNDarray__array).all())

        exp_axis_one = ht.array([rank] * size, dtype=ht.int32).expand_dims(1)
        res = ht.unique(split_one, sorted=True, axis=1)
        self.assertTrue((res._DNDarray__array == exp_axis_one._DNDarray__array).all())

        torch_array = torch.tensor([
            [1, 2],
            [2, 3],
            [1, 2],
            [2, 3],
            [1, 2]
        ], dtype=torch.int32)
        data = ht.array(torch_array, split=0)

        res, inv = ht.unique(data, return_inverse=True, axis=0)
        _, exp_inv = torch_array.unique(dim=0, return_inverse=True, sorted=True)
        self.assertTrue(torch.equal(inv, exp_inv.to(dtype=inv.dtype)))

        res, inv = ht.unique(data, return_inverse=True, axis=1)
        _, exp_inv = torch_array.unique(dim=1, return_inverse=True, sorted=True)
        self.assertTrue(torch.equal(inv, exp_inv.to(dtype=inv.dtype)))

        torch_array = torch.tensor([
            [1, 1, 2],
            [1, 2, 2],
            [2, 1, 2],
            [1, 3, 2],
            [0, 1, 2]
        ], dtype=torch.int32)
        exp_res, exp_inv = torch_array.unique(return_inverse=True, sorted=True)

        data_split_none = ht.array(torch_array)
        res, inv = ht.unique(data_split_none, return_inverse=True, sorted=True)
        self.assertTrue(torch.equal(inv, exp_inv.to(dtype=inv.dtype)))

        data_split_zero = ht.array(torch_array, split=0)
        res, inv = ht.unique(data_split_zero, return_inverse=True, sorted=True)
        self.assertTrue(torch.equal(inv, exp_inv.to(dtype=inv.dtype)))
