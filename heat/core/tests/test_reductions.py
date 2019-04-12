import unittest
import torch
import numpy as np
import heat as ht

from itertools import combinations


FLOAT_EPSILON = 1e-4


class TestOperations(unittest.TestCase):
    def test_sum(self):
        array_len = 11

        # check sum over all float elements of 1d tensor locally
        shape_noaxis = ht.ones(array_len)
        no_axis_sum = shape_noaxis.sum()

        self.assertIsInstance(no_axis_sum, ht.tensor)
        self.assertEqual(no_axis_sum.shape, (1,))
        self.assertEqual(no_axis_sum.lshape, (1,))
        self.assertEqual(no_axis_sum.dtype, ht.float32)
        self.assertEqual(no_axis_sum._tensor__array.dtype, torch.float32)
        self.assertEqual(no_axis_sum.split, None)
        self.assertEqual(no_axis_sum._tensor__array, array_len)

        out_noaxis = ht.zeros((1,))
        ht.sum(shape_noaxis, out=out_noaxis)
        self.assertTrue(out_noaxis._tensor__array ==
                        shape_noaxis._tensor__array.sum())

        # check sum over all float elements of split 1d tensor
        shape_noaxis_split = ht.arange(array_len, split=0)
        shape_noaxis_split_sum = shape_noaxis_split.sum()

        self.assertIsInstance(shape_noaxis_split_sum, ht.tensor)
        self.assertEqual(shape_noaxis_split_sum.shape, (1,))
        self.assertEqual(shape_noaxis_split_sum.lshape, (1,))
        self.assertEqual(shape_noaxis_split_sum.dtype, ht.int64)
        self.assertEqual(
            shape_noaxis_split_sum._tensor__array.dtype, torch.int64)
        self.assertEqual(shape_noaxis_split_sum.split, None)
        self.assertEqual(shape_noaxis_split_sum, 55)

        out_noaxis = ht.zeros((1,))
        ht.sum(shape_noaxis_split, out=out_noaxis)
        self.assertEqual(out_noaxis._tensor__array, 55)

        # check sum over all float elements of 3d tensor locally
        shape_noaxis = ht.ones((3, 3, 3))
        no_axis_sum = shape_noaxis.sum()

        self.assertIsInstance(no_axis_sum, ht.tensor)
        self.assertEqual(no_axis_sum.shape, (1,))
        self.assertEqual(no_axis_sum.lshape, (1,))
        self.assertEqual(no_axis_sum.dtype, ht.float32)
        self.assertEqual(no_axis_sum._tensor__array.dtype, torch.float32)
        self.assertEqual(no_axis_sum.split, None)
        self.assertEqual(no_axis_sum._tensor__array, 27)

        out_noaxis = ht.zeros((1,))
        ht.sum(shape_noaxis, out=out_noaxis)
        self.assertEqual(out_noaxis._tensor__array, 27)

        # check sum over all float elements of split 3d tensor
        shape_noaxis_split_axis = ht.ones((3, 3, 3), split=0)
        split_axis_sum = shape_noaxis_split_axis.sum(axis=0)

        self.assertIsInstance(split_axis_sum, ht.tensor)
        self.assertEqual(split_axis_sum.shape, (3, 3))
        self.assertEqual(split_axis_sum.dtype, ht.float32)
        self.assertEqual(split_axis_sum._tensor__array.dtype, torch.float32)
        self.assertEqual(split_axis_sum.split, None)

        out_noaxis = ht.zeros((3, 3,))
        ht.sum(shape_noaxis, axis=0, out=out_noaxis)
        self.assertTrue((out_noaxis._tensor__array ==
                         torch.full((3, 3,), 3)).all())

        # check sum over all float elements of splitted 5d tensor with negative axis
        shape_noaxis_split_axis_neg = ht.ones((1, 2, 3, 4, 5), split=1)
        shape_noaxis_split_axis_neg_sum = shape_noaxis_split_axis_neg.sum(
            axis=-2)

        self.assertIsInstance(shape_noaxis_split_axis_neg_sum, ht.tensor)
        self.assertEqual(
            shape_noaxis_split_axis_neg_sum.shape, (1, 2, 3, 5))
        self.assertEqual(shape_noaxis_split_axis_neg_sum.dtype, ht.float32)
        self.assertEqual(
            shape_noaxis_split_axis_neg_sum._tensor__array.dtype, torch.float32)
        self.assertEqual(shape_noaxis_split_axis_neg_sum.split, 1)

        out_noaxis = ht.zeros((1, 2, 3, 5))
        ht.sum(shape_noaxis_split_axis_neg, axis=-2, out=out_noaxis)

        # check sum over all float elements of splitted 3d tensor with tuple axis
        shape_split_axis_tuple = ht.ones((3, 4, 5), split=1)
        shape_split_axis_tuple_sum = shape_split_axis_tuple.sum(axis=(-2, -3))
        expected_result = ht.ones((5,))*12.

        self.assertIsInstance(shape_split_axis_tuple_sum, ht.tensor)
        self.assertEqual(shape_split_axis_tuple_sum.shape, (5,))
        self.assertEqual(shape_split_axis_tuple_sum.dtype, ht.float32)
        self.assertEqual(shape_split_axis_tuple_sum._tensor__array.dtype, torch.float32)
        self.assertEqual(shape_split_axis_tuple_sum.split, None)
        self.assertEqual(shape_split_axis_tuple_sum, expected_result)

        # exceptions
        with self.assertRaises(ValueError):
            ht.ones(array_len).sum(axis=1)
        with self.assertRaises(ValueError):
            ht.ones(array_len).sum(axis=-2)
        with self.assertRaises(ValueError):
            ht.ones((4, 4)).sum(axis=0, out=out_noaxis)
        with self.assertRaises(TypeError):
            ht.ones(array_len).sum(axis='bad_axis_type')

    def test_mean(self):
        array_0_len = 5
        array_1_len = 5
        array_2_len = 5

        x = ht.zeros((2, 3, 4))
        with self.assertRaises(ValueError):
            ht.mean(x, axis=10)
        with self.assertRaises(TypeError):
            ht.mean(x, axis='01')
        with self.assertRaises(ValueError):
            ht.mean(x, axis=(0, '10'))

        # zeros
        dimensions = []
        for d in [array_0_len, array_1_len, array_2_len]:
            dimensions.extend([d, ])
            try:
                hold = list(range(len(dimensions)))
                hold.append(None)
            except TypeError:
                hold = [None, ]
            for i in hold:  # loop over the number of dimensions of the test array
                z = ht.zeros(dimensions, split=i)
                res = z.mean()
                # print(res, z.mean())
                total_dims_list = list(z.shape)
                # print(dimensions, i, res)
                if res != np.nan:
                    self.assertEqual(res, 0)
                for it in range(len(z.shape)):  # loop over the different single dimensions for mean
                    res = ht.mean(z, axis=it)
                    self.assertEqual(res, 0)
                    if not isinstance(res, float):
                        if res.split:
                            self.assertEqual(res.split, z.split)
                    target_dims = [total_dims_list[q] if q != it else 0 for q in range(len(total_dims_list))]
                    if all(target_dims) != 0:
                        self.assertEqual(res.lshape, tuple(target_dims))
                        self.assertEqual(res.split, z.split)
                    if i == it:
                        res = z.mean(axis=it)
                        self.assertEqual(res, 0)
                        target_dims = [total_dims_list[q] if q != it else 0 for q in range(len(total_dims_list))]
                        if all(target_dims) != 0:
                            self.assertEqual(res.lshape, tuple(target_dims))

                loop_list = [",".join(map(str, comb)) for comb in combinations(list(range(len(z.shape))), 2)]
                if len(z.shape) > 2:
                    for r in range(3, len(z.shape)):
                        loop_list.extend([",".join(map(str, comb)) for comb in combinations(list(range(len(z.shape))), r)])
                for it in loop_list:  # loop over the different combinations of dimensions for mean
                    res = z.mean(axis=tuple([int(q) for q in it.split(',')]))
                    self.assertEqual(res, 0)
                    if not isinstance(res, float):
                        if res.split:
                            self.assertEqual(res.split, z.split)
                    target_dims = [total_dims_list[int(q)] if q not in [int(q) for q in it.split(',')] else 0 for q in range(len(total_dims_list))]
                    if all(target_dims) != 0:
                        if i:
                            self.assertEqual(res.lshape, tuple(target_dims))
                            self.assertEqual(res.split, z.split)
                        else:
                            self.assertEqual(res.shape, tuple(target_dims))
                            self.assertEqual(res.split, z.split)

        # ones
        dimensions = []

        for d in [array_0_len, array_1_len, array_2_len]:
            dimensions.extend([d, ])
            # print("dimensions: ", dimensions)
            try:
                hold = list(range(len(dimensions)))
                hold.append(None)
            except TypeError:
                hold = [None, ]
            for i in hold:  # loop over the number of split dimension of the test array
                # print("Beginning of dimensions i=", i)
                z = ht.ones(dimensions, split=i)
                res = z.mean()
                total_dims_list = list(z.shape)
                self.assertEqual(res, 1)
                for it in range(len(z.shape)):  # loop over the different single dimensions for mean
                    # print('it=', it)
                    res = z.mean(axis=it)
                    self.assertEqual(res, 1)
                    if not isinstance(res, float) and res.split:
                        self.assertEqual(res.split, z.split)
                    target_dims = [total_dims_list[q] if q != it else 0 for q in range(len(total_dims_list))]
                    if all(target_dims) != 0:
                        self.assertEqual(res.split, z.split)
                        self.assertEqual(res.lshape, tuple(target_dims))
                    if i == it:
                        res = z.mean(axis=it)
                        self.assertEqual(res, 1)
                        target_dims = [total_dims_list[q] if q != it else 0 for q in range(len(total_dims_list))]
                        if all(target_dims) != 0:
                            self.assertEqual(res.lshape, tuple(target_dims))

                loop_list = [",".join(map(str, comb)) for comb in combinations(list(range(len(z.shape))), 2)]
                if len(z.shape) > 2:
                    for r in range(3, len(z.shape)):
                        loop_list.extend([",".join(map(str, comb)) for comb in combinations(list(range(len(z.shape))), r)])
                for it in loop_list:  # loop over the different combinations of dimensions for mean
                    # print("it combi:", it)
                    res = z.mean(axis=[int(q) for q in it.split(',')])
                    self.assertEqual(res, 1)
                    if not isinstance(res, float) and res.split:
                        self.assertEqual(res.split, z.split)
                    target_dims = [total_dims_list[int(q)] if q not in [int(q) for q in it.split(',')] else 0 for q in range(len(total_dims_list))]
                    if all(target_dims) != 0:
                        if i:
                            self.assertEqual(res.lshape, tuple(target_dims))
                            self.assertEqual(res.split, z.split)
                        else:
                            self.assertEqual(res.shape, tuple(target_dims))
                            self.assertEqual(res.split, z.split)

        # values for the iris dataset mean measured by libreoffice calc
        ax0 = [5.84333333333333, 3.054, 3.75866666666667, 1.19866666666667]
        for sp in [None, 0, 1]:
            iris = ht.load('heat/datasets/data/iris.h5', 'data', split=sp)
            self.assertAlmostEqual(ht.mean(iris), 3.46366666666667)
            assert all([a == b for a, b in zip(ht.mean(iris, axis=0), ax0)])

    def test_var(self):
        array_0_len = 10
        array_1_len = 9
        array_2_len = 8

        # test raises
        x = ht.zeros((2,3,4))
        with self.assertRaises(TypeError):
            ht.var(x, axis=0, bessel=1)
        with self.assertRaises(ValueError):
            ht.var(x, axis=10)
        with self.assertRaises(TypeError):
            ht.var(x, axis='01')

        # zeros
        dimensions = []
        for d in [array_0_len, array_1_len, array_2_len]:
            dimensions.extend([d, ])
            # print("dimensions: ", dimensions)
            try:
                hold = list(range(len(dimensions)))
                hold.append(None)
            except TypeError:
                hold = [None,]
            for i in hold:  # loop over the number of dimensions of the test array
                # print("Beginning of dimensions i=", i)
                z = ht.zeros(dimensions, split=i)
                res = z.var()
                total_dims_list = list(z.shape)
                self.assertEqual(res, 0)
                for it in range(len(z.shape)):  # loop over the different single dimensions for mean
                    res = z.var(axis=it)
                    self.assertEqual(res, 0)
                    if not isinstance(res, float) and res.split:
                        self.assertEqual(res.split, z.split)
                    target_dims = [total_dims_list[q] if q != it else 0 for q in range(len(total_dims_list))]
                    if all(target_dims) != 0:
                        self.assertEqual(res.lshape, tuple(target_dims))
                        self.assertEqual(res.split, z.split)
                    if i == it:
                        res = z.var(axis=it)
                        self.assertEqual(res, 0)
                        target_dims = [total_dims_list[q] if q != it else 0 for q in range(len(total_dims_list))]
                        if all(target_dims) != 0:
                            self.assertEqual(res.lshape, tuple(target_dims))
        #
        # ones
        dimensions = []
        for d in [array_0_len, array_1_len, array_2_len]:
            dimensions.extend([d, ])
            # print("dimensions: ", dimensions)
            try:
                hold = list(range(len(dimensions)))
                hold.append(None)
            except TypeError:
                hold = [None, ]
            for i in hold:  # loop over the number of dimensions of the test array
                # print("Beginning of dimensions i=", i)
                z = ht.ones(dimensions, split=i)
                res = z.var()
                total_dims_list = list(z.shape)
                self.assertEqual(res, 0)
                for it in range(len(z.shape)):  # loop over the different single dimensions for mean
                    res = z.var(axis=it)
                    self.assertEqual(res, 0)
                    if not isinstance(res, float) and res.split:
                        self.assertEqual(res.split, z.split)
                    target_dims = [total_dims_list[q] if q != it else 0 for q in range(len(total_dims_list))]
                    if all(target_dims) != 0:
                        self.assertEqual(res.lshape, tuple(target_dims))
                        self.assertEqual(res.split, z.split)
                    if i == it:
                        res = z.var(axis=it)
                        self.assertEqual(res, 0)
                        target_dims = [total_dims_list[q] if q != it else 0 for q in range(len(total_dims_list))]
                        if all(target_dims) != 0:
                            self.assertEqual(res.lshape, tuple(target_dims))

        # values for the iris dataset var measured by libreoffice calc
        ax0 = [0.68569351230425, 0.188004026845638, 3.11317941834452, 0.582414317673378]
        for sp in [None, 0, 1]:
            iris = ht.load_hdf5('heat/datasets/data/iris.h5', 'data', split=sp)
            self.assertAlmostEqual(ht.var(iris, bessel=True), 3.90318519755147, 5)
            assert all([a == b for a, b in zip(ht.var(iris, axis=0, bessel=True), ax0)])