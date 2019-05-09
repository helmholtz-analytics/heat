import torch
import unittest
from itertools import combinations

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
        self.assertEqual(result.lshape, (1, 4, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == data._DNDarray__array.argmax(0)).all())

        # 3D local tensor, minor axis
        result = ht.argmax(data, axis=-1)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._DNDarray__array.dtype, torch.int64)
        self.assertEqual(result.shape, (3, 4,))
        self.assertEqual(result.lshape, (3, 4, 1,))
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
        self.assertEqual(result.lshape, (4, 1,))
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
        self.assertEqual(result.lshape, (1, size,))
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
        self.assertEqual(output.lshape, (1, size,))
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
        self.assertEqual(result.lshape, (1, 4, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == data._DNDarray__array.argmin(0)).all())

        # 3D local tensor, minor axis
        result = ht.argmin(data, axis=-1)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._DNDarray__array.dtype, torch.int64)
        self.assertEqual(result.shape, (3, 4,))
        self.assertEqual(result.lshape, (3, 4, 1,))
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
        self.assertEqual(result.lshape, (4, 1,))
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
        self.assertEqual(result.lshape, (1, size,))
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
        self.assertEqual(output.lshape, (1, size,))
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
        self.assertEqual(maximum_vertical.lshape, (1, 3,))
        self.assertEqual(maximum_vertical.split, None)
        self.assertEqual(maximum_vertical.dtype, ht.int64)
        self.assertEqual(maximum_vertical._DNDarray__array.dtype, torch.int64)
        self.assertTrue((maximum_vertical._DNDarray__array ==
                         comparison.max(dim=0, keepdim=True)[0]).all())

        # maximum along second axis
        maximum_horizontal = ht.max(ht_array, axis=1)

        self.assertIsInstance(maximum_horizontal, ht.DNDarray)
        self.assertEqual(maximum_horizontal.shape, (4,))
        self.assertEqual(maximum_horizontal.lshape, (4, 1,))
        self.assertEqual(maximum_horizontal.split, None)
        self.assertEqual(maximum_horizontal.dtype, ht.int64)
        self.assertEqual(maximum_horizontal._DNDarray__array.dtype, torch.int64)
        self.assertTrue((maximum_horizontal._DNDarray__array == comparison.max(dim=1, keepdim=True)[0]).all())

        # check max over all float elements of split 3d tensor, across split axis
        random_volume = ht.random.randn(3, 3, 3, split=1)
        maximum_volume = ht.max(random_volume, axis=1)

        self.assertIsInstance(maximum_volume, ht.DNDarray)
        self.assertEqual(maximum_volume.shape, (3, 3))
        self.assertEqual(maximum_volume.lshape, (3, 1, 3))
        self.assertEqual(maximum_volume.dtype, ht.float32)
        self.assertEqual(maximum_volume._DNDarray__array.dtype, torch.float32)
        self.assertEqual(maximum_volume.split, None)

        # check max over all float elements of split 3d tensor, tuple axis
        random_volume = ht.random.randn(3, 3, 3, split=0)
        maximum_volume = ht.max(random_volume, axis=(1, 2))
        alt_maximum_volume = ht.max(random_volume, axis=(2, 1))

        self.assertIsInstance(maximum_volume, ht.DNDarray)
        self.assertEqual(maximum_volume.shape, (3,))
        self.assertEqual(maximum_volume.lshape, (3, 1, 1))
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

    def test_mean(self):
        array_0_len = 5
        array_1_len = 5
        array_2_len = 5
        # array_3_len = 7

        x = ht.zeros((2, 3, 4))
        with self.assertRaises(ValueError):
            ht.mean(x, axis=10)
        with self.assertRaises(TypeError):
            ht.mean(x, axis='01')
        with self.assertRaises(ValueError):
            ht.mean(x, axis=(0, '10'))

        # ones
        dimensions = []

        for d in [array_0_len, array_1_len, array_2_len]:
            dimensions.extend([d, ])
            hold = list(range(len(dimensions)))
            hold.append(None)
            for i in hold:  # loop over the number of split dimension of the test array
                z = ht.ones(dimensions, split=i)
                res = z.mean()
                total_dims_list = list(z.shape)
                self.assertEqual(res, 1)
                for it in range(len(z.shape)):  # loop over the different single dimensions for mean
                    res = z.mean(axis=it)
                    self.assertEqual(res, 1)
                    target_dims = [total_dims_list[q] for q in range(len(total_dims_list)) if q != it]
                    if not target_dims:
                        target_dims = (1, )

                    self.assertEqual(res.gshape, tuple(target_dims))
                    if res.split is not None:
                        if i >= it:
                            self.assertEqual(res.split, len(target_dims) - 1)
                        else:
                            self.assertEqual(res.split, z.split)

                loop_list = [",".join(map(str, comb)) for comb in combinations(list(range(len(z.shape))), 2)]

                for it in loop_list:  # loop over the different combinations of dimensions for mean
                    lp_split = [int(q) for q in it.split(',')]
                    res = z.mean(axis=lp_split)
                    self.assertEqual(res, 1)
                    target_dims = [total_dims_list[q] for q in range(len(total_dims_list)) if q not in lp_split]
                    if not target_dims:
                        target_dims = (1,)
                    if res.gshape:
                        self.assertEqual(res.gshape, tuple(target_dims))
                    if res.split is not None:
                        if any([i >= x for x in lp_split]):
                            self.assertEqual(res.split, len(target_dims) - 1)
                        else:
                            self.assertEqual(res.split, z.split)

        # values for the iris dataset mean measured by libreoffice calc
        ax0 = [5.84333333333333, 3.054, 3.75866666666667, 1.19866666666667]
        for sp in [None, 0, 1]:
            iris = ht.load('heat/datasets/data/iris.h5', 'data', split=sp)
            self.assertAlmostEqual(ht.mean(iris), 3.46366666666667)
            assert all([a == b for a, b in zip(ht.mean(iris, axis=0), ax0)])

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
        self.assertEqual(minimum_vertical.lshape, (1, 3,))
        self.assertEqual(minimum_vertical.split, None)
        self.assertEqual(minimum_vertical.dtype, ht.int64)
        self.assertEqual(minimum_vertical._DNDarray__array.dtype, torch.int64)
        self.assertTrue((minimum_vertical._DNDarray__array == comparison.min(dim=0, keepdim=True)[0]).all())

        # maximum along second axis
        minimum_horizontal = ht.min(ht_array, axis=1)

        self.assertIsInstance(minimum_horizontal, ht.DNDarray)
        self.assertEqual(minimum_horizontal.shape, (4,))
        self.assertEqual(minimum_horizontal.lshape, (4, 1,))
        self.assertEqual(minimum_horizontal.split, None)
        self.assertEqual(minimum_horizontal.dtype, ht.int64)
        self.assertEqual(minimum_horizontal._DNDarray__array.dtype, torch.int64)
        self.assertTrue((minimum_horizontal._DNDarray__array == comparison.min(dim=1, keepdim=True)[0]).all())

        # check max over all float elements of split 3d tensor, across split axis
        random_volume = ht.random.randn(3, 3, 3, split=1)
        minimum_volume = ht.min(random_volume, axis=1)

        self.assertIsInstance(minimum_volume, ht.DNDarray)
        self.assertEqual(minimum_volume.shape, (3, 3))
        self.assertEqual(minimum_volume.lshape, (3, 1, 3))
        self.assertEqual(minimum_volume.dtype, ht.float32)
        self.assertEqual(minimum_volume._DNDarray__array.dtype, torch.float32)
        self.assertEqual(minimum_volume.split, None)

        # check min over all float elements of split 3d tensor, tuple axis
        random_volume = ht.random.randn(3, 3, 3, split=0)
        minimum_volume = ht.min(random_volume, axis=(1, 2))
        alt_minimum_volume = ht.min(random_volume, axis=(2, 1))

        self.assertIsInstance(minimum_volume, ht.DNDarray)
        self.assertEqual(minimum_volume.shape, (3,))
        self.assertEqual(minimum_volume.lshape, (3, 1, 1))
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

    def test_var(self):
        array_0_len = 5
        array_1_len = 5
        array_2_len = 5

        # test raises
        x = ht.zeros((2, 3, 4))
        with self.assertRaises(TypeError):
            ht.var(x, axis=0, bessel=1)
        with self.assertRaises(ValueError):
            ht.var(x, axis=10)
        with self.assertRaises(TypeError):
            ht.var(x, axis='01')

        # ones
        dimensions = []
        for d in [array_0_len, array_1_len, array_2_len]:
            dimensions.extend([d, ])
            hold = list(range(len(dimensions)))
            hold.append(None)
            for i in hold:  # loop over the number of dimensions of the test array
                z = ht.ones(dimensions, split=i)
                res = z.var()
                total_dims_list = list(z.shape)
                self.assertEqual(res, 0)
                for it in range(len(z.shape)):  # loop over the different single dimensions for mean
                    res = z.var(axis=it)
                    self.assertEqual(res, 0)
                    target_dims = [total_dims_list[q] for q in range(len(total_dims_list)) if q != it]
                    if not target_dims:
                        target_dims = (1,)
                    self.assertEqual(res.gshape, tuple(target_dims))
                    if res.split is not None:
                        if i >= it:
                            self.assertEqual(res.split, len(target_dims) - 1)
                        else:
                            self.assertEqual(res.split, z.split)

                    if i == it:
                        res = z.var(axis=it)
                        self.assertEqual(res, 0)
                z = ht.ones(dimensions, split=i)
                res = z.var(bessel=False)
                self.assertEqual(res, 0)

        # values for the iris dataset var measured by libreoffice calc
        ax0 = [0.68569351230425, 0.188004026845638, 3.11317941834452, 0.582414317673378]
        for sp in [None, 0, 1]:
            iris = ht.load_hdf5('heat/datasets/data/iris.h5', 'data', split=sp)
            self.assertAlmostEqual(ht.var(iris, bessel=True), 3.90318519755147, 5)

    def test_std(self):
        # test raises
        x = ht.zeros((2, 3, 4))
        with self.assertRaises(TypeError):
            ht.std(x, axis=0, bessel=1)
        with self.assertRaises(ValueError):
            ht.std(x, axis=10)
        with self.assertRaises(TypeError):
            ht.std(x, axis='01')

        # the rest of the tests are covered by var
