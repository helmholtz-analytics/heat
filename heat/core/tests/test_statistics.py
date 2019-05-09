import torch
import unittest

import heat as ht


class TestStatistics(unittest.TestCase):

    def test_argmax(self):
        torch.manual_seed(1)
        data = ht.random.randn(3, 4, 5)

        # 3D local tensor, major axis
        result = ht.argmax(data, axis=0)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._DNDarray__array.dtype, torch.int64)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.lshape, (4, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == data._DNDarray__array.argmax(0)).all())

        # 3D local tensor, minor axis
        result = ht.argmax(data, axis=-1, keepdim=True)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._DNDarray__array.dtype, torch.int64)
        self.assertEqual(result.shape, (3, 4, 1))
        self.assertEqual(result.lshape, (3, 4, 1))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == data._DNDarray__array.argmax(-1, keepdim=True)).all())

        # 1D split tensor, no axis
        data = ht.arange(-10, 10, split=0)
        result = ht.argmax(data)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._DNDarray__array.dtype, torch.int64)
        self.assertEqual(result.shape, (1,))
        self.assertEqual(result.lshape, (1,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == torch.tensor([19])))

        # 2D split tensor, along the axis
        torch.manual_seed(1)
        data = ht.array(ht.random.randn(4, 5), split=0)
        result = ht.argmax(data, axis=1)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._DNDarray__array.dtype, torch.int64)
        self.assertEqual(result.shape, (ht.MPI_WORLD.size * 4,))
        self.assertEqual(result.lshape, (4,))
        self.assertEqual(result.split, 0)
        self.assertTrue((result._DNDarray__array == torch.tensor([[4], [4], [2], [4]])).all())

        # 2D split tensor, across the axis
        size = ht.MPI_WORLD.size * 2
        data = ht.tril(ht.ones((size, size,), split=0), k=-1)

        result = ht.argmax(data, axis=0)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._DNDarray__array.dtype, torch.int64)
        self.assertEqual(result.shape, (size,))
        self.assertEqual(result.lshape, (size,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array != 0).all())

        # 2D split tensor, across the axis, output tensor
        size = ht.MPI_WORLD.size * 2
        data = ht.triu(ht.ones((size, size,), split=0), k=-1)

        output = ht.empty((size,))
        result = ht.argmax(data, axis=0, out=output)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(output.dtype, ht.int64)
        self.assertEqual(output._DNDarray__array.dtype, torch.int64)
        self.assertEqual(output.shape, (size,))
        self.assertEqual(output.lshape, (size,))
        self.assertEqual(output.split, None)
        self.assertTrue((output._DNDarray__array != 0).all())

        # check exceptions
        with self.assertRaises(TypeError):
            data.argmax(axis=(0, 1))
        with self.assertRaises(TypeError):
            data.argmax(axis=1.1)
        with self.assertRaises(TypeError):
            data.argmax(axis='y')
        with self.assertRaises(ValueError):
            ht.argmax(data, axis=-4)

    def test_argmin(self):
        torch.manual_seed(1)
        data = ht.random.randn(3, 4, 5)

        # 3D local tensor, no axis
        result = ht.argmin(data)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._DNDarray__array.dtype, torch.int64)
        self.assertEqual(result.shape, (1,))
        self.assertEqual(result.lshape, (1,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == data._DNDarray__array.argmin()).all())

        # 3D local tensor, major axis
        result = ht.argmin(data, axis=0)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._DNDarray__array.dtype, torch.int64)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.lshape, (4, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == data._DNDarray__array.argmin(0)).all())

        # 3D local tensor, minor axis
        result = ht.argmin(data, axis=-1, keepdim=True)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._DNDarray__array.dtype, torch.int64)
        self.assertEqual(result.shape, (3, 4, 1))
        self.assertEqual(result.lshape, (3, 4, 1))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == data._DNDarray__array.argmin(-1, keepdim=True)).all())

        # 2D split tensor, along the axis
        torch.manual_seed(1)
        data = ht.array(ht.random.randn(4, 5), split=0)
        result = ht.argmin(data, axis=1)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._DNDarray__array.dtype, torch.int64)
        self.assertEqual(result.shape, (ht.MPI_WORLD.size * 4,))
        self.assertEqual(result.lshape, (4,))
        self.assertEqual(result.split, 0)
        self.assertTrue((result._DNDarray__array == torch.tensor([[3], [1], [1], [3]])).all())

        # 2D split tensor, across the axis
        size = ht.MPI_WORLD.size * 2
        data = ht.triu(ht.ones((size, size,), split=0), k=1)

        result = ht.argmin(data, axis=0)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._DNDarray__array.dtype, torch.int64)
        self.assertEqual(result.shape, (size,))
        self.assertEqual(result.lshape, (size,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array != 0).all())

        # 2D split tensor, across the axis, output tensor
        size = ht.MPI_WORLD.size * 2
        data = ht.triu(ht.ones((size, size,), split=0), k=1)

        output = ht.empty((size,))
        result = ht.argmin(data, axis=0, out=output)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(output.dtype, ht.int64)
        self.assertEqual(output._DNDarray__array.dtype, torch.int64)
        self.assertEqual(output.shape, (size,))
        self.assertEqual(output.lshape, (size,))
        self.assertEqual(output.split, None)
        self.assertTrue((output._DNDarray__array != 0).all())

        # check exceptions
        with self.assertRaises(TypeError):
            data.argmin(axis=(0, 1))
        with self.assertRaises(TypeError):
            data.argmin(axis=1.1)
        with self.assertRaises(TypeError):
            data.argmin(axis='y')
        with self.assertRaises(ValueError):
            ht.argmin(data, axis=-4)

    def test_max(self):
        data = [
            [1,   2,  3],
            [4,   5,  6],
            [7,   8,  9],
            [10, 11, 12]
        ]

        ht_array = ht.array(data)
        comparison = torch.tensor(data)

        # check global max
        maximum = ht.max(ht_array)

        self.assertIsInstance(maximum, ht.DNDarray)
        self.assertEqual(maximum.shape, (1,))
        self.assertEqual(maximum.lshape, (1,))
        self.assertEqual(maximum.split, None)
        self.assertEqual(maximum.dtype, ht.int64)
        self.assertEqual(maximum._DNDarray__array.dtype, torch.int64)
        self.assertEqual(maximum, 12)

        # maximum along first axis
        maximum_vertical = ht.max(ht_array, axis=0)

        self.assertIsInstance(maximum_vertical, ht.DNDarray)
        self.assertEqual(maximum_vertical.shape, (3,))
        self.assertEqual(maximum_vertical.lshape, (3,))
        self.assertEqual(maximum_vertical.split, None)
        self.assertEqual(maximum_vertical.dtype, ht.int64)
        self.assertEqual(maximum_vertical._DNDarray__array.dtype, torch.int64)
        self.assertTrue((maximum_vertical._DNDarray__array ==
                         comparison.max(dim=0, keepdim=True)[0]).all())

        # maximum along second axis
        maximum_horizontal = ht.max(ht_array, axis=1, keepdim=True)

        self.assertIsInstance(maximum_horizontal, ht.DNDarray)
        self.assertEqual(maximum_horizontal.shape, (4, 1))
        self.assertEqual(maximum_horizontal.lshape, (4, 1))
        self.assertEqual(maximum_horizontal.split, None)
        self.assertEqual(maximum_horizontal.dtype, ht.int64)
        self.assertEqual(maximum_horizontal._DNDarray__array.dtype, torch.int64)
        self.assertTrue((maximum_horizontal._DNDarray__array == comparison.max(dim=1, keepdim=True)[0]).all())

        # check max over all float elements of split 3d tensor, across split axis
        random_volume = ht.random.randn(3, 3, 3, split=1)
        maximum_volume = ht.max(random_volume, axis=1)

        self.assertIsInstance(maximum_volume, ht.DNDarray)
        self.assertEqual(maximum_volume.shape, (3, 3))
        self.assertEqual(maximum_volume.lshape, (3, 3))
        self.assertEqual(maximum_volume.dtype, ht.float32)
        self.assertEqual(maximum_volume._DNDarray__array.dtype, torch.float32)
        self.assertEqual(maximum_volume.split, None)

        # check max over all float elements of split 3d tensor, tuple axis
        random_volume = ht.random.randn(3, 3, 3, split=0)
        maximum_volume = ht.max(random_volume, axis=(1, 2))
        alt_maximum_volume = ht.max(random_volume, axis=(2, 1))

        self.assertIsInstance(maximum_volume, ht.DNDarray)
        self.assertEqual(maximum_volume.shape, (3,))
        self.assertEqual(maximum_volume.lshape, (3,))
        self.assertEqual(maximum_volume.dtype, ht.float32)
        self.assertEqual(maximum_volume._DNDarray__array.dtype, torch.float32)
        self.assertEqual(maximum_volume.split, 0)
        self.assertEqual(maximum_volume, alt_maximum_volume)

        # check max over all float elements of split 5d tensor, along split axis
        random_5d = ht.random.randn(1, 2, 3, 4, 5, split=0)
        maximum_5d = ht.max(random_5d, axis=1)

        self.assertIsInstance(maximum_5d, ht.DNDarray)
        self.assertEqual(maximum_5d.shape, (1, 3, 4, 5))
        self.assertLessEqual(maximum_5d.lshape[1], 2)
        self.assertEqual(maximum_5d.dtype, ht.float32)
        self.assertEqual(maximum_5d._DNDarray__array.dtype, torch.float32)
        self.assertEqual(maximum_5d.split, 0)

        # check exceptions
        with self.assertRaises(TypeError):
            ht_array.max(axis=1.1)
        with self.assertRaises(TypeError):
            ht_array.max(axis='y')
        with self.assertRaises(ValueError):
            ht.max(ht_array, axis=-4)

    def test_min(self):
        data = [
            [1,   2,  3],
            [4,   5,  6],
            [7,   8,  9],
            [10, 11, 12]
        ]

        ht_array = ht.array(data)
        comparison = torch.tensor(data)

        # check global max
        minimum = ht.min(ht_array)

        self.assertIsInstance(minimum, ht.DNDarray)
        self.assertEqual(minimum.shape, (1,))
        self.assertEqual(minimum.lshape, (1,))
        self.assertEqual(minimum.split, None)
        self.assertEqual(minimum.dtype, ht.int64)
        self.assertEqual(minimum._DNDarray__array.dtype, torch.int64)
        self.assertEqual(minimum, 12)

        # maximum along first axis
        minimum_vertical = ht.min(ht_array, axis=0)

        self.assertIsInstance(minimum_vertical, ht.DNDarray)
        self.assertEqual(minimum_vertical.shape, (3,))
        self.assertEqual(minimum_vertical.lshape, (3,))
        self.assertEqual(minimum_vertical.split, None)
        self.assertEqual(minimum_vertical.dtype, ht.int64)
        self.assertEqual(minimum_vertical._DNDarray__array.dtype, torch.int64)
        self.assertTrue((minimum_vertical._DNDarray__array == comparison.min(dim=0, keepdim=True)[0]).all())

        # maximum along second axis
        minimum_horizontal = ht.min(ht_array, axis=1, keepdim=True)

        self.assertIsInstance(minimum_horizontal, ht.DNDarray)
        self.assertEqual(minimum_horizontal.shape, (4, 1))
        self.assertEqual(minimum_horizontal.lshape, (4, 1))
        self.assertEqual(minimum_horizontal.split, None)
        self.assertEqual(minimum_horizontal.dtype, ht.int64)
        self.assertEqual(minimum_horizontal._DNDarray__array.dtype, torch.int64)
        self.assertTrue((minimum_horizontal._DNDarray__array == comparison.min(dim=1, keepdim=True)[0]).all())

        # check max over all float elements of split 3d tensor, across split axis
        random_volume = ht.random.randn(3, 3, 3, split=1)
        minimum_volume = ht.min(random_volume, axis=1)

        self.assertIsInstance(minimum_volume, ht.DNDarray)
        self.assertEqual(minimum_volume.shape, (3, 3))
        self.assertEqual(minimum_volume.lshape, (3, 3))
        self.assertEqual(minimum_volume.dtype, ht.float32)
        self.assertEqual(minimum_volume._DNDarray__array.dtype, torch.float32)
        self.assertEqual(minimum_volume.split, None)

        # check min over all float elements of split 3d tensor, tuple axis
        random_volume = ht.random.randn(3, 3, 3, split=0)
        minimum_volume = ht.min(random_volume, axis=(1, 2))
        alt_minimum_volume = ht.min(random_volume, axis=(2, 1))

        self.assertIsInstance(minimum_volume, ht.DNDarray)
        self.assertEqual(minimum_volume.shape, (3,))
        self.assertEqual(minimum_volume.lshape, (3,))
        self.assertEqual(minimum_volume.dtype, ht.float32)
        self.assertEqual(minimum_volume._DNDarray__array.dtype, torch.float32)
        self.assertEqual(minimum_volume.split, 0)
        self.assertEqual(minimum_volume, alt_minimum_volume)

        # check max over all float elements of split 5d tensor, along split axis
        random_5d = ht.random.randn(1, 2, 3, 4, 5, split=0)
        minimum_5d = ht.min(random_5d, axis=1)

        self.assertIsInstance(minimum_5d, ht.DNDarray)
        self.assertEqual(minimum_5d.shape, (1, 3, 4, 5))
        self.assertLessEqual(minimum_5d.lshape[1], 2)
        self.assertEqual(minimum_5d.dtype, ht.float32)
        self.assertEqual(minimum_5d._DNDarray__array.dtype, torch.float32)
        self.assertEqual(minimum_5d.split, 0)

        # check exceptions
        with self.assertRaises(TypeError):
            ht_array.min(axis=1.1)
        with self.assertRaises(TypeError):
            ht_array.min(axis='y')
        with self.assertRaises(ValueError):
            ht.min(ht_array, axis=-4)
