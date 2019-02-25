from itertools import combinations
import numpy as np
import unittest
import torch

import heat as ht

FLOAT_EPSILON = 1e-4


class TestOperations(unittest.TestCase):
    def test_abs(self):
        float32_tensor = ht.arange(-10, 10, dtype=ht.float32, split=0)
        absolute_values = ht.abs(float32_tensor)

        # basic absolute test
        self.assertIsInstance(absolute_values, ht.tensor)
        self.assertEqual(absolute_values.dtype, ht.float32)
        self.assertEqual(absolute_values.sum(axis=0), 100)

        # check whether output works
        output_tensor = ht.zeros(20, split=0)
        self.assertEqual(output_tensor.sum(axis=0), 0)
        ht.absolute(float32_tensor, out=output_tensor)
        self.assertEqual(output_tensor.sum(axis=0), 100)

        # dtype parameter
        int64_tensor = ht.arange(-10, 10, dtype=ht.int64)
        absolute_values = ht.abs(int64_tensor, dtype=ht.float32)
        self.assertIsInstance(absolute_values, ht.tensor)
        self.assertEqual(absolute_values.sum(axis=0), 100)
        self.assertEqual(absolute_values.dtype, ht.float32)
        self.assertEqual(absolute_values._tensor__array.dtype, torch.float32)

        # exceptions
        with self.assertRaises(TypeError):
            ht.absolute('hello')
        with self.assertRaises(TypeError):
            float32_tensor.abs(out=1)
        with self.assertRaises(TypeError):
            float32_tensor.absolute(out=float32_tensor, dtype=3.2)

    def test_all(self):
        array_len = 9
        # check all over all float elements of 1d tensor locally
        ones_noaxis = ht.ones(array_len)
        x = (ones_noaxis == 1)
        self.assertEqual(x.all().shape, (1,))
        self.assertEqual(x.all().lshape, (1,))
        self.assertIsInstance(x.all(), ht.tensor)
        self.assertEqual(x.all().dtype, ht.int64)
        self.assertEqual(x.all()._tensor__array.dtype, ht.int64)
        self.assertEqual(x.all().split, None)
        out_noaxis = ht.zeros((1,))
        ht.all(x, out=out_noaxis)
        self.assertTrue(out_noaxis)

        # check all over all float elements of splitted 1d tensor
        ones_noaxis_split = ht.ones(array_len, split=0)
        self.assertEqual(ones_noaxis_split.all().shape, (1,))
        self.assertEqual(ones_noaxis_split.all().lshape, (1,))
        self.assertIsInstance(ones_noaxis_split.all(), ht.tensor)
        self.assertEqual(ones_noaxis_split.all().dtype, ht.int64)
        self.assertEqual(ones_noaxis_split._tensor__array.dtype, torch.float32)
        self.assertEqual(ones_noaxis_split.all().split, None)
        ht.all(ones_noaxis_split, out=out_noaxis)
        self.assertTrue(out_noaxis)

        # check all over all integer elements of 1d tensor locally
        ones_noaxis_int = ht.ones(array_len).astype(ht.int)
        self.assertEqual(ones_noaxis_int.all(axis=0).shape, (1,))
        self.assertEqual(ones_noaxis_int.all(axis=0).lshape, (1,))
        self.assertIsInstance(ones_noaxis_int.all(axis=0), ht.tensor)
        self.assertEqual(ones_noaxis_int.all(axis=0).dtype, ht.int64)
        self.assertEqual(ones_noaxis_int.all()._tensor__array.dtype, ht.int64)
        self.assertEqual(ones_noaxis_int.all().split, None)
        ht.sum(ones_noaxis_int, out=out_noaxis)
        self.assertTrue(out_noaxis)

        # check all over all integer elements of splitted 1d tensor
        ones_noaxis_split_int = ht.ones(array_len, split=0).astype(ht.int)
        self.assertEqual(ones_noaxis_split_int.all().shape, (1,))
        self.assertEqual(ones_noaxis_split_int.all().lshape, (1,))
        self.assertIsInstance(ones_noaxis_split_int.all(), ht.tensor)
        self.assertEqual(ones_noaxis_split_int.all().dtype, ht.int64)
        self.assertEqual(ones_noaxis_split_int.all()._tensor__array.dtype, ht.int64)
        self.assertEqual(ones_noaxis_split_int.all(), array_len)
        self.assertEqual(ones_noaxis_split_int.all().split, None)
        ht.sum(ones_noaxis_split_int, out=out_noaxis)
        self.assertTrue(out_noaxis)

        # check all over all float elements of 3d tensor locally
        ones_noaxis = ht.ones((3,3,3))
        self.assertEqual(ones_noaxis.all().shape, (1,))
        self.assertEqual(ones_noaxis.all().lshape, (1,))
        self.assertIsInstance(ones_noaxis.all(), ht.tensor)
        self.assertEqual(ones_noaxis.all().dtype, ht.int64)
        self.assertEqual(ones_noaxis.all()._tensor__array.dtype, ht.int64)
        self.assertEqual(ones_noaxis.all().split, None)
        ht.sum(ones_noaxis, out=out_noaxis)
        self.assertTrue(out_noaxis)

        # check all over all float elements of splitted 3d tensor
        ones_noaxis_split_axis = ht.ones((3,3,3), split=0)
        self.assertIsInstance(ones_noaxis_split_axis.all(axis=1), ht.tensor)
        self.assertEqual(ones_noaxis.all(axis=1).shape, (3,1,3))
        self.assertEqual(ones_noaxis_split_axis.all(axis=1).dtype, ht.int64)
        self.assertEqual(ones_noaxis_split_axis.all(axis=1)._tensor__array.dtype, ht.int64)
        self.assertEqual(ones_noaxis_split_axis.all().split, None)
        ht.sum(ones_noaxis, out=out_noaxis)
        self.assertTrue(out_noaxis)

        # check all over all float elements of splitted 5d tensor with negative axis
        ones_noaxis_split_axis_neg = ht.ones((1,2,3,4,5), split=1)
        self.assertIsInstance(ones_noaxis_split_axis_neg.all(axis=-2), ht.tensor)
        self.assertEqual(ones_noaxis_split_axis_neg.all(axis=-2).shape, (1,2,3,1,5))
        self.assertEqual(ones_noaxis_split_axis_neg.all(axis=-2).dtype, ht.int64)
        self.assertEqual(ones_noaxis_split_axis_neg.all(axis=-2)._tensor__array.dtype, ht.int64)
        self.assertEqual(ones_noaxis_split_axis_neg.all().split, None)

        # check all output to predefined distributed tensor
        torch.manual_seed(1)
        random_nosplit = ht.random.randn(3, 3, 3)
        torch.manual_seed(1)
        random_split = ht.random.randn(3, 3, 3, split=0)
        out_nosplit = ht.zeros((3, 1, 3))
        out_split = ht.zeros((3, 1, 3), split=2)
        ht.all(random_nosplit, axis=1, out=out_nosplit)
        ht.all(random_nosplit, axis=1, out=out_split)
        self.assertTrue(out_nosplit._tensor__array, out_split._tensor__array)
        self.assertEqual(out_nosplit.split, None)
        self.assertEqual(out_split.split, 2)
        ht.all(random_split, axis=1, out=out_nosplit)
        ht.all(random_split, axis=1, out=out_split)
        self.assertTrue(out_nosplit._tensor__array, out_split._tensor__array)
        self.assertEqual(out_nosplit.split, None)
        self.assertEqual(out_split.split, 2)

        # exceptions
        with self.assertRaises(ValueError):
            ht.ones(array_len).all(axis=1)
        with self.assertRaises(ValueError):
            ht.ones(array_len).all(axis=-2)
        with self.assertRaises(ValueError):
            ht.ones((4,4)).all(axis=0, out=out_noaxis)
        with self.assertRaises(TypeError):
            ht.ones(array_len).all(axis='bad_axis_type')

    def test_argmin(self):

        data = ht.float32([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])

        comparison = torch.Tensor([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])

        #check basics
        self.assertTrue((ht.argmin(data,axis=0)._tensor__array == comparison.argmin(0)).all())
        self.assertIsInstance(ht.argmin(data,axis=1),ht.tensor)
        self.assertIsInstance(data.argmin(),ht.tensor)

        #check combinations of split and axis
        torch.manual_seed(1)
        random_data = ht.random.randn(3,3,3)
        torch.manual_seed(1)
        random_data_split = ht.random.randn(3,3,3,split=0)

        self.assertTrue((ht.argmin(random_data,axis=0)._tensor__array == random_data_split.argmin(axis=0)._tensor__array).all())
        self.assertTrue((ht.argmin(random_data,axis=1)._tensor__array == random_data_split.argmin(axis=1)._tensor__array).all())
        self.assertIsInstance(ht.argmin(random_data_split,axis=1),ht.tensor)
        self.assertIsInstance(random_data_split.argmin(),ht.tensor)

        #check argmin over all float elements of 3d tensor locally
        self.assertEqual(random_data.argmin().shape, (1,))
        self.assertEqual(random_data.argmin().lshape, (1,))
        self.assertEqual(random_data.argmin().dtype, ht.int64)
        self.assertEqual(random_data.argmin().split, None)

        # check argmin over all float elements of splitted 3d tensor
        self.assertIsInstance(random_data_split.argmin(axis=1), ht.tensor)
        self.assertEqual(random_data_split.argmin(axis=1).shape, (3,1,3))
        self.assertEqual(random_data_split.argmin().split, None)

        # check argmin over all float elements of splitted 5d tensor with negative axis
        random_data_split_neg = ht.random.randn(1,2,3,4,5, split=1)
        self.assertIsInstance(random_data_split_neg.argmin(axis=-2), ht.tensor)
        self.assertEqual(random_data_split_neg.argmin(axis=-2).shape, (1,2,3,1,5))
        self.assertEqual(random_data_split_neg.argmin(axis=-2).dtype, ht.int64)
        self.assertEqual(random_data_split_neg.argmin().split, None)

        # check exceptions
        with self.assertRaises(NotImplementedError):
            data.argmin(axis=(0,1))
        with self.assertRaises(TypeError):
            data.argmin(axis=1.1)
        with self.assertRaises(TypeError):
            data.argmin(axis='y')
        with self.assertRaises(ValueError):
            ht.argmin(data, axis=-4)



    def test_clip(self):
        elements = 20

        # float tensor
        float32_tensor = ht.arange(elements, dtype=ht.float32, split=0)
        clipped = float32_tensor.clip(5, 15)
        self.assertIsInstance(clipped, ht.tensor)
        self.assertEqual(clipped.dtype, ht.float32)
        self.assertEqual(clipped.sum(axis=0), 195)

        # long tensor
        int64_tensor = ht.arange(elements, dtype=ht.int64, split=0)
        clipped = int64_tensor.clip(4, 16)
        self.assertIsInstance(clipped, ht.tensor)
        self.assertEqual(clipped.dtype, ht.int64)
        self.assertEqual(clipped.sum(axis=0), 195)

        # test the exceptions
        with self.assertRaises(TypeError):
            ht.clip(torch.arange(10), 2, 5)
        with self.assertRaises(ValueError):
            ht.arange(20).clip(None, None)
        with self.assertRaises(TypeError):
            ht.clip(ht.arange(20), 5, 15, out=torch.arange(20))

    def test_copy(self):
        tensor = ht.ones(5)
        copied = tensor.copy()

        # test identity inequality and value equality
        self.assertIsNot(tensor, copied)
        self.assertIsNot(tensor._tensor__array, copied._tensor__array)
        self.assertTrue((tensor == copied)._tensor__array.all())

        # test exceptions
        with self.assertRaises(TypeError):
            ht.copy('hello world')

    def test_exp(self):
        elements = 10
        comparison = torch.arange(elements, dtype=torch.float64).exp()

        # exponential of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_exp = ht.exp(float32_tensor)
        self.assertIsInstance(float32_exp, ht.tensor)
        self.assertEqual(float32_exp.dtype, ht.float32)
        self.assertEqual(float32_exp.dtype, ht.float32)
        in_range = (float32_exp._tensor__array - comparison.type(torch.float32)) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # exponential of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_exp = ht.exp(float64_tensor)
        self.assertIsInstance(float64_exp, ht.tensor)
        self.assertEqual(float64_exp.dtype, ht.float64)
        self.assertEqual(float64_exp.dtype, ht.float64)
        in_range = (float64_exp._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # exponential of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_exp = ht.exp(int32_tensor)
        self.assertIsInstance(int32_exp, ht.tensor)
        self.assertEqual(int32_exp.dtype, ht.float64)
        self.assertEqual(int32_exp.dtype, ht.float64)
        in_range = (int32_exp._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # exponential of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_exp = ht.exp(int64_tensor)
        self.assertIsInstance(int64_exp, ht.tensor)
        self.assertEqual(int64_exp.dtype, ht.float64)
        self.assertEqual(int64_exp.dtype, ht.float64)
        in_range = (int64_exp._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.exp([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.exp('hello world')

    def test_floor(self):
        start, end, step = -5.0, 5.0, 1.4
        comparison = torch.arange(start, end, step, dtype=torch.float64).floor()

        # exponential of float32
        float32_tensor = ht.arange(start, end, step, dtype=ht.float32)
        float32_floor = float32_tensor.floor()
        self.assertIsInstance(float32_floor, ht.tensor)
        self.assertEqual(float32_floor.dtype, ht.float32)
        self.assertEqual(float32_floor.dtype, ht.float32)
        self.assertTrue((float32_floor._tensor__array == comparison.type(torch.float32)).all())

        # exponential of float64
        float64_tensor = ht.arange(start, end, step, dtype=ht.float64)
        float64_floor = float64_tensor.floor()
        self.assertIsInstance(float64_floor, ht.tensor)
        self.assertEqual(float64_floor.dtype, ht.float64)
        self.assertEqual(float64_floor.dtype, ht.float64)
        self.assertTrue((float64_floor._tensor__array == comparison).all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.floor([0, 1, 2, 3])
        with self.assertRaises(TypeError):
            ht.floor(object())

    def test_log(self):
        elements = 15
        comparison = torch.arange(1, elements, dtype=torch.float64).log()

        # logarithm of float32
        float32_tensor = ht.arange(1, elements, dtype=ht.float32)
        float32_log = ht.log(float32_tensor)
        self.assertIsInstance(float32_log, ht.tensor)
        self.assertEqual(float32_log.dtype, ht.float32)
        self.assertEqual(float32_log.dtype, ht.float32)
        in_range = (float32_log._tensor__array - comparison.type(torch.float32)) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logarithm of float64
        float64_tensor = ht.arange(1, elements, dtype=ht.float64)
        float64_log = ht.log(float64_tensor)
        self.assertIsInstance(float64_log, ht.tensor)
        self.assertEqual(float64_log.dtype, ht.float64)
        self.assertEqual(float64_log.dtype, ht.float64)
        in_range = (float64_log._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logarithm of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(1, elements, dtype=ht.int32)
        int32_log = ht.log(int32_tensor)
        self.assertIsInstance(int32_log, ht.tensor)
        self.assertEqual(int32_log.dtype, ht.float64)
        self.assertEqual(int32_log.dtype, ht.float64)
        in_range = (int32_log._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logarithm of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(1, elements, dtype=ht.int64)
        int64_log = ht.log(int64_tensor)
        self.assertIsInstance(int64_log, ht.tensor)
        self.assertEqual(int64_log.dtype, ht.float64)
        self.assertEqual(int64_log.dtype, ht.float64)
        in_range = (int64_log._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.log([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.log('hello world')

    def test_max(self):
        data = ht.float32([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])

        comparison = torch.Tensor([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])

        #check basics
        self.assertTrue((ht.max(data,axis=0)._tensor__array[0] == comparison.max(0)[0]).all())
        self.assertIsInstance(ht.max(data,axis=1),ht.tensor)
        self.assertIsInstance(data.max(),ht.tensor)

        #check combinations of split and axis
        torch.manual_seed(1)
        random_data = ht.random.randn(3,3,3)
        torch.manual_seed(1)
        random_data_split = ht.random.randn(3,3,3,split=0)

        self.assertTrue((ht.max(random_data,axis=0)._tensor__array[0] == random_data_split.max(axis=0)._tensor__array[0]).all())
        self.assertTrue((ht.max(random_data,axis=1)._tensor__array[0] == random_data_split.max(axis=1)._tensor__array[0]).all())
        self.assertIsInstance(ht.max(random_data_split,axis=1),ht.tensor)
        self.assertIsInstance(random_data_split.max(),ht.tensor)
        output = ht.ones((1,))
        ht.max(random_data_split, out=output)
        self.assertTrue(output._tensor__array == random_data_split.max())

        #check max over all float elements of 3d tensor locally
        self.assertEqual(random_data.max().shape, (1,))
        self.assertEqual(random_data.max().lshape, (1,))
        self.assertEqual(random_data.max().dtype, ht.float32)
        self.assertEqual(random_data.max()._tensor__array[0].dtype, torch.float32)
        self.assertEqual(random_data.max().split, None)
        ht.max(random_data, out=output)
        self.assertTrue((output._tensor__array == random_data.max()))

        # check max over all float elements of splitted 3d tensor
        self.assertIsInstance(random_data_split.max(axis=1), ht.tensor)
        self.assertEqual(random_data_split.max(axis=1).shape, (3,1,3))
        self.assertEqual(random_data_split.max(axis=1).dtype, ht.float32)
        self.assertEqual(random_data_split.max(axis=1)._tensor__array[0].dtype, torch.float32)
        self.assertEqual(random_data_split.max().split, None)

        # check max over all float elements of splitted 5d tensor with negative axis
        random_data_split_neg = ht.random.randn(1,2,3,4,5, split=1)
        self.assertIsInstance(random_data_split_neg.max(axis=-2), ht.tensor)
        self.assertEqual(random_data_split_neg.max(axis=-2).shape, (1,2,3,1,5))
        self.assertEqual(random_data_split_neg.max(axis=-2).dtype, ht.float32)
        self.assertEqual(random_data_split_neg.max(axis=-2)._tensor__array[0].dtype, torch.float32)
        self.assertEqual(random_data_split_neg.max().split, None)

        # check max output to predefined distributed tensor
        torch.manual_seed(1)
        random_nosplit = ht.random.randn(3, 3, 3)
        torch.manual_seed(1)
        random_split = ht.random.randn(3, 3, 3, split=0)
        out_nosplit = ht.zeros((3, 1, 3))
        out_split = ht.zeros((3, 1, 3), split=2)
        ht.max(random_nosplit, axis=1, out=out_nosplit)
        ht.max(random_nosplit, axis=1, out=out_split)
        self.assertEqual(out_nosplit.split, None)
        self.assertEqual(out_split.split, 2)
        ht.max(random_split, axis=1, out=out_nosplit)
        ht.max(random_split, axis=1, out=out_split)
        self.assertEqual(out_nosplit.split, None)
        self.assertEqual(out_split.split, 2)

        # check exceptions
        with self.assertRaises(NotImplementedError):
            data.max(axis=(0,1))
        with self.assertRaises(TypeError):
            data.max(axis=1.1)
        with self.assertRaises(TypeError):
            data.max(axis='y')
        with self.assertRaises(ValueError):
            ht.max(data,axis=-4)
        with self.assertRaises(ValueError):
            ht.max(data, axis=0, out=output)

    def test_min(self):

        data = ht.float32([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])

        comparison = torch.Tensor([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])

        #check basics
        self.assertTrue((ht.min(data,axis=0)._tensor__array[0] == comparison.min(0)[0]).all())
        self.assertIsInstance(ht.min(data,axis=1),ht.tensor)
        self.assertIsInstance(data.min(),ht.tensor)

        #check combinations of split and axis
        torch.manual_seed(1)
        random_data = ht.random.randn(3,3,3)
        torch.manual_seed(1)
        random_data_split = ht.random.randn(3,3,3,split=0)

        self.assertTrue((ht.min(random_data,axis=0)._tensor__array[0] == random_data_split.min(axis=0)._tensor__array[0]).all())
        self.assertTrue((ht.min(random_data,axis=1)._tensor__array[0] == random_data_split.min(axis=1)._tensor__array[0]).all())
        self.assertIsInstance(ht.min(random_data_split,axis=1),ht.tensor)
        self.assertIsInstance(random_data_split.min(),ht.tensor)

        #check min over all float elements of 3d tensor locally
        self.assertEqual(random_data.min().shape, (1,))
        self.assertEqual(random_data.min().lshape, (1,))
        self.assertEqual(random_data.min().dtype, ht.float32)
        self.assertEqual(random_data.min()._tensor__array[0].dtype, torch.float32)
        self.assertEqual(random_data.min().split, None)
        output = ht.ones((1,))
        ht.min(random_data, out=output)
        self.assertTrue(output._tensor__array == random_data.min())

        # check min over all float elements of splitted 3d tensor
        self.assertIsInstance(random_data_split.min(axis=1), ht.tensor)
        self.assertEqual(random_data_split.min(axis=1).shape, (3,1,3))
        self.assertEqual(random_data_split.min(axis=1).dtype, ht.float32)
        self.assertEqual(random_data_split.min(axis=1)._tensor__array[0].dtype, torch.float32)
        self.assertEqual(random_data_split.min().split, None)
        ht.min(random_data_split, out=output)
        self.assertTrue((output._tensor__array == random_data_split.min()))

        # check min over all float elements of splitted 5d tensor with negative axis
        random_data_split_neg = ht.random.randn(1,2,3,4,5, split=1)
        self.assertIsInstance(random_data_split_neg.min(axis=-2), ht.tensor)
        self.assertEqual(random_data_split_neg.min(axis=-2).shape, (1,2,3,1,5))
        self.assertEqual(random_data_split_neg.min(axis=-2).dtype, ht.float32)
        self.assertEqual(random_data_split_neg.min(axis=-2)._tensor__array[0].dtype, torch.float32)
        self.assertEqual(random_data_split_neg.min().split, None)

        # check min output to predefined distributed tensor
        torch.manual_seed(1)
        random_nosplit = ht.random.randn(3, 3, 3)
        torch.manual_seed(1)
        random_split = ht.random.randn(3, 3, 3, split=0)
        out_nosplit = ht.zeros((3, 1, 3))
        out_split = ht.zeros((3, 1, 3), split=2)
        ht.min(random_nosplit, axis=1, out=out_nosplit)
        ht.min(random_nosplit, axis=1, out=out_split)
        self.assertEqual(out_nosplit.split, None)
        self.assertEqual(out_split.split, 2)
        ht.min(random_split, axis=1, out=out_nosplit)
        ht.min(random_split, axis=1, out=out_split)
        self.assertEqual(out_nosplit.split, None)
        self.assertEqual(out_split.split, 2)

        # check exceptions
        with self.assertRaises(NotImplementedError):
            data.min(axis=(0,1))
        with self.assertRaises(TypeError):
            data.min(axis=1.1)
        with self.assertRaises(TypeError):
            data.min(axis='y')
        with self.assertRaises(ValueError):
            ht.min(data, axis=-4)
        with self.assertRaises(ValueError):
            ht.min(data, axis=0, out=output)


    def test_sin(self):
        # base elements
        elements = 30
        comparison = torch.arange(elements, dtype=torch.float64).sin()

        # sine of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_sin = ht.sin(float32_tensor)
        self.assertIsInstance(float32_sin, ht.tensor)
        self.assertEqual(float32_sin.dtype, ht.float32)
        self.assertEqual(float32_sin.dtype, ht.float32)
        in_range = (float32_sin._tensor__array - comparison.type(torch.float32)) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # sine of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_sin = ht.sin(float64_tensor)
        self.assertIsInstance(float64_sin, ht.tensor)
        self.assertEqual(float64_sin.dtype, ht.float64)
        self.assertEqual(float64_sin.dtype, ht.float64)
        in_range = (float64_sin._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logarithm of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_sin = ht.sin(int32_tensor)
        self.assertIsInstance(int32_sin, ht.tensor)
        self.assertEqual(int32_sin.dtype, ht.float64)
        self.assertEqual(int32_sin.dtype, ht.float64)
        in_range = (int32_sin._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logathm of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_sin = ht.sin(int64_tensor)
        self.assertIsInstance(int64_sin, ht.tensor)
        self.assertEqual(int64_sin.dtype, ht.float64)
        self.assertEqual(int64_sin.dtype, ht.float64)
        in_range = (int64_sin._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.sin([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.sin('hello world')

    def test_sqrt(self):
        elements = 20
        comparison = torch.arange(elements, dtype=torch.float64).sqrt()

        # square roots of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_sqrt = ht.sqrt(float32_tensor)
        self.assertIsInstance(float32_sqrt, ht.tensor)
        self.assertEqual(float32_sqrt.dtype, ht.float32)
        self.assertEqual(float32_sqrt.dtype, ht.float32)
        self.assertTrue((float32_sqrt._tensor__array == comparison.type(torch.float32)).all())

        # square roots of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_sqrt = ht.sqrt(float64_tensor)
        self.assertIsInstance(float64_sqrt, ht.tensor)
        self.assertEqual(float64_sqrt.dtype, ht.float64)
        self.assertEqual(float64_sqrt.dtype, ht.float64)
        self.assertTrue((float64_sqrt._tensor__array == comparison).all())

        # square roots of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_sqrt = ht.sqrt(int32_tensor)
        self.assertIsInstance(int32_sqrt, ht.tensor)
        self.assertEqual(int32_sqrt.dtype, ht.float64)
        self.assertEqual(int32_sqrt.dtype, ht.float64)
        self.assertTrue((int32_sqrt._tensor__array == comparison).all())

        # square roots of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_sqrt = ht.sqrt(int64_tensor)
        self.assertIsInstance(int64_sqrt, ht.tensor)
        self.assertEqual(int64_sqrt.dtype, ht.float64)
        self.assertEqual(int64_sqrt.dtype, ht.float64)
        self.assertTrue((int64_sqrt._tensor__array == comparison).all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.sqrt([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.sqrt('hello world')

    def test_sqrt_method(self):
        elements = 25
        comparison = torch.arange(elements, dtype=torch.float64).sqrt()

        # square roots of float32
        float32_sqrt = ht.arange(elements, dtype=ht.float32).sqrt()
        self.assertIsInstance(float32_sqrt, ht.tensor)
        self.assertEqual(float32_sqrt.dtype, ht.float32)
        self.assertEqual(float32_sqrt.dtype, ht.float32)
        self.assertTrue((float32_sqrt._tensor__array == comparison.type(torch.float32)).all())

        # square roots of float64
        float64_sqrt = ht.arange(elements, dtype=ht.float64).sqrt()
        self.assertIsInstance(float64_sqrt, ht.tensor)
        self.assertEqual(float64_sqrt.dtype, ht.float64)
        self.assertEqual(float64_sqrt.dtype, ht.float64)
        self.assertTrue((float64_sqrt._tensor__array == comparison).all())

        # square roots of ints, automatic conversion to intermediate floats
        int32_sqrt = ht.arange(elements, dtype=ht.int32).sqrt()
        self.assertIsInstance(int32_sqrt, ht.tensor)
        self.assertEqual(int32_sqrt.dtype, ht.float64)
        self.assertEqual(int32_sqrt.dtype, ht.float64)
        self.assertTrue((int32_sqrt._tensor__array == comparison).all())

        # square roots of longs, automatic conversion to intermediate floats
        int64_sqrt = ht.arange(elements, dtype=ht.int64).sqrt()
        self.assertIsInstance(int64_sqrt, ht.tensor)
        self.assertEqual(int64_sqrt.dtype, ht.float64)
        self.assertEqual(int64_sqrt.dtype, ht.float64)
        self.assertTrue((int64_sqrt._tensor__array == comparison).all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.sqrt([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.sqrt('hello world')

    def test_sqrt_out_of_place(self):
        elements = 30
        output_shape = (3, elements)
        number_range = ht.arange(elements, dtype=ht.float32)
        output_buffer = ht.zeros(output_shape, dtype=ht.float32)

        # square roots
        float32_sqrt = ht.sqrt(number_range, out=output_buffer)
        comparison = torch.arange(elements, dtype=torch.float32).sqrt()

        # check whether the input range remain unchanged
        self.assertIsInstance(number_range, ht.tensor)
        self.assertEqual(number_range.sum(axis=0), 190)  # gaussian sum
        self.assertEqual(number_range.gshape, (elements,))

        # check whether the output buffer still has the correct shape
        self.assertIsInstance(float32_sqrt, ht.tensor)
        self.assertEqual(float32_sqrt.dtype, ht.float32)
        self.assertEqual(float32_sqrt._tensor__array.shape, output_shape)
        for row in range(output_shape[0]):
            self.assertTrue((float32_sqrt._tensor__array[row] == comparison).all())

        # exception
        with self.assertRaises(TypeError):
            ht.sqrt(number_range, 'hello world')

    def test_sum(self):
        array_len = 9
        # check sum over all float elements of 1d tensor locally
        shape_noaxis = ht.ones(array_len)
        self.assertEqual(shape_noaxis.sum().shape, (1,))
        self.assertEqual(shape_noaxis.sum().lshape, (1,))
        self.assertIsInstance(shape_noaxis.sum(), ht.tensor)
        self.assertEqual(shape_noaxis.sum().dtype, ht.float32)
        self.assertEqual(shape_noaxis._tensor__array.dtype, torch.float32)
        self.assertAlmostEqual(shape_noaxis.sum(), float(array_len), places=7, msg=None, delta=None)
        self.assertEqual(shape_noaxis.sum().split, None)
        out_noaxis = ht.ones((1,))
        ht.sum(shape_noaxis, out=out_noaxis)
        self.assertTrue(out_noaxis._tensor__array == shape_noaxis.sum())

        # check sum over all float elements of splitted 1d tensor
        shape_noaxis_split = ht.ones(array_len, split=0)
        self.assertEqual(shape_noaxis_split.sum().shape, (1,))
        self.assertEqual(shape_noaxis_split.sum().lshape, (1,))
        self.assertIsInstance(shape_noaxis_split.sum(), ht.tensor)
        self.assertEqual(shape_noaxis_split.sum().dtype, ht.float32)
        self.assertEqual(shape_noaxis_split.sum()._tensor__array.dtype, torch.float32)
        self.assertAlmostEqual(shape_noaxis_split.sum(), float(array_len), places=7, msg=None, delta=None)
        self.assertEqual(shape_noaxis_split.sum().split, None)
        ht.sum(shape_noaxis_split, out=out_noaxis)
        self.assertTrue(out_noaxis._tensor__array == shape_noaxis_split.sum())

        # check sum over all integer elements of 1d tensor locally
        shape_noaxis_int = ht.ones(array_len).astype(ht.int)
        self.assertEqual(shape_noaxis_int.sum(axis=0).shape, (1,))
        self.assertEqual(shape_noaxis_int.sum(axis=0).lshape, (1,))
        self.assertIsInstance(shape_noaxis_int.sum(axis=0), ht.tensor)
        self.assertEqual(shape_noaxis_int.sum(axis=0).dtype, ht.int64)
        self.assertEqual(shape_noaxis_int.sum()._tensor__array.dtype, torch.int64)
        self.assertEqual(shape_noaxis_int.sum(), array_len)
        self.assertEqual(shape_noaxis_int.sum().split, None)
        ht.sum(shape_noaxis_int, out=out_noaxis)
        self.assertTrue(out_noaxis._tensor__array == shape_noaxis_int.sum())

        # check sum over all integer elements of splitted 1d tensor
        shape_noaxis_split_int = ht.ones(array_len, split=0).astype(ht.int)
        self.assertEqual(shape_noaxis_split_int.sum().shape, (1,))
        self.assertEqual(shape_noaxis_split_int.sum().lshape, (1,))
        self.assertIsInstance(shape_noaxis_split_int.sum(), ht.tensor)
        self.assertEqual(shape_noaxis_split_int.sum().dtype, ht.int64)
        self.assertEqual(shape_noaxis_split_int.sum()._tensor__array.dtype, torch.int64)
        self.assertEqual(shape_noaxis_split_int.sum(), array_len)
        self.assertEqual(shape_noaxis_split_int.sum().split, None)
        ht.sum(shape_noaxis_split_int, out=out_noaxis)
        self.assertTrue(out_noaxis._tensor__array == shape_noaxis_split_int.sum())

        # check sum over all float elements of 3d tensor locally
        shape_noaxis = ht.ones((3,3,3))
        self.assertEqual(shape_noaxis.sum().shape, (1,))
        self.assertEqual(shape_noaxis.sum().lshape, (1,))
        self.assertIsInstance(shape_noaxis.sum(), ht.tensor)
        self.assertEqual(shape_noaxis.sum().dtype, ht.float32)
        self.assertEqual(shape_noaxis.sum()._tensor__array.dtype, torch.float32)
        self.assertAlmostEqual(shape_noaxis.sum(), 27., places=7, msg=None, delta=None)
        self.assertEqual(shape_noaxis.sum().split, None)
        ht.sum(shape_noaxis, out=out_noaxis)
        self.assertTrue(out_noaxis._tensor__array == shape_noaxis.sum())

        # check sum over all float elements of splitted 3d tensor
        shape_noaxis_split_axis = ht.ones((3,3,3), split=0)
        self.assertIsInstance(shape_noaxis_split_axis.sum(axis=1), ht.tensor)
        self.assertEqual(shape_noaxis.sum(axis=1).shape, (3,1,3))
        self.assertEqual(shape_noaxis_split_axis.sum(axis=1).dtype, ht.float32)
        self.assertEqual(shape_noaxis_split_axis.sum(axis=1)._tensor__array.dtype, torch.float32)
        self.assertEqual(shape_noaxis_split_axis.sum().split, None)
        ht.sum(shape_noaxis, out=out_noaxis)
        self.assertTrue(out_noaxis._tensor__array == shape_noaxis_split_axis.sum())

        # check sum over all float elements of splitted 5d tensor with negative axis
        shape_noaxis_split_axis_neg = ht.ones((1,2,3,4,5), split=1)
        self.assertIsInstance(shape_noaxis_split_axis_neg.sum(axis=-2), ht.tensor)
        self.assertEqual(shape_noaxis_split_axis_neg.sum(axis=-2).shape, (1,2,3,1,5))
        self.assertEqual(shape_noaxis_split_axis_neg.sum(axis=-2).dtype, ht.float32)
        self.assertEqual(shape_noaxis_split_axis_neg.sum(axis=-2)._tensor__array.dtype, torch.float32)
        self.assertEqual(shape_noaxis_split_axis_neg.sum().split, None)

        # check sum output to predefined distributed tensor
        torch.manual_seed(1)
        random_nosplit = ht.random.randn(3, 3, 3)
        torch.manual_seed(1)
        random_split = ht.random.randn(3, 3, 3, split=0)
        out_nosplit = ht.zeros((3, 1, 3))
        out_split = ht.zeros((3, 1, 3), split=2)
        ht.sum(random_nosplit, axis=1, out=out_nosplit)
        ht.sum(random_nosplit, axis=1, out=out_split)
        self.assertTrue(out_nosplit._tensor__array, out_split._tensor__array)
        self.assertEqual(out_nosplit.split, None)
        self.assertEqual(out_split.split, 2)
        ht.sum(random_split, axis=1, out=out_nosplit)
        ht.sum(random_split, axis=1, out=out_split)
        self.assertTrue(out_nosplit._tensor__array, out_split._tensor__array)
        self.assertEqual(out_nosplit.split, None)
        self.assertEqual(out_split.split, 2)

        # exceptions
        with self.assertRaises(ValueError):
            ht.ones(array_len).sum(axis=1)
        with self.assertRaises(ValueError):
            ht.ones(array_len).sum(axis=-2)
        with self.assertRaises(ValueError):
            ht.ones((4,4)).sum(axis=0, out=out_noaxis)
        with self.assertRaises(TypeError):
            ht.ones(array_len).sum(axis='bad_axis_type')

    def test_transpose(self):
        # vector transpose, not distributed
        vector = ht.arange(10)
        vector_t = vector.T
        self.assertIsInstance(vector_t, ht.tensor)
        self.assertEqual(vector_t.dtype, ht.int32)
        self.assertEqual(vector_t.split, None)
        self.assertEqual(vector_t.shape, (10,))

        # simple matrix transpose, not distributed
        simple_matrix = ht.zeros((2, 4))
        simple_matrix_t = simple_matrix.transpose()
        self.assertIsInstance(simple_matrix_t, ht.tensor)
        self.assertEqual(simple_matrix_t.dtype, ht.float32)
        self.assertEqual(simple_matrix_t.split, None)
        self.assertEqual(simple_matrix_t.shape, (4, 2,))
        self.assertEqual(simple_matrix_t._tensor__array.shape, (4, 2,))

        # 4D array, not distributed, with given axis
        array_4d = ht.zeros((2, 3, 4, 5))
        array_4d_t = ht.transpose(array_4d, axes=(-1, 0, 2, 1))
        self.assertIsInstance(array_4d_t, ht.tensor)
        self.assertEqual(array_4d_t.dtype, ht.float32)
        self.assertEqual(array_4d_t.split, None)
        self.assertEqual(array_4d_t.shape, (5, 2, 4, 3,))
        self.assertEqual(array_4d_t._tensor__array.shape, (5, 2, 4, 3,))

        # vector transpose, distributed
        vector_split = ht.arange(10, split=0)
        vector_split_t = vector_split.T
        self.assertIsInstance(vector_split_t, ht.tensor)
        self.assertEqual(vector_split_t.dtype, ht.int32)
        self.assertEqual(vector_split_t.split, 0)
        self.assertEqual(vector_split_t.shape, (10,))
        self.assertLessEqual(vector_split_t.lshape[0], 10)

        # matrix transpose, distributed
        matrix_split = ht.ones((10, 20,), split=1)
        matrix_split_t = matrix_split.transpose()
        self.assertIsInstance(matrix_split_t, ht.tensor)
        self.assertEqual(matrix_split_t.dtype, ht.float32)
        self.assertEqual(matrix_split_t.split, 0)
        self.assertEqual(matrix_split_t.shape, (20, 10,))
        self.assertLessEqual(matrix_split_t.lshape[0], 20)
        self.assertEqual(matrix_split_t.lshape[1], 10)

        # 4D array, distributed
        array_4d_split = ht.ones((3, 4, 5, 6,), split=3)
        array_4d_split_t = ht.transpose(array_4d_split, axes=(1, 0, 3, 2,))
        self.assertIsInstance(array_4d_t, ht.tensor)
        self.assertEqual(array_4d_split_t.dtype, ht.float32)
        self.assertEqual(array_4d_split_t.split, 2)
        self.assertEqual(array_4d_split_t.shape, (4, 3, 6, 5,))

        self.assertEqual(array_4d_split_t.lshape[0], 4)
        self.assertEqual(array_4d_split_t.lshape[1], 3)
        self.assertLessEqual(array_4d_split_t.lshape[2], 6)
        self.assertEqual(array_4d_split_t.lshape[3], 5)

        # exceptions
        with self.assertRaises(TypeError):
            ht.transpose(1)
        with self.assertRaises(ValueError):
            ht.transpose(ht.zeros((2, 3,)), axes=1.0)
        with self.assertRaises(ValueError):
            ht.transpose(ht.zeros((2, 3,)), axes=(-1,))
        with self.assertRaises(TypeError):
            ht.zeros((2, 3,)).transpose(axes='01')
        with self.assertRaises(TypeError):
            ht.zeros((2, 3,)).transpose(axes=(0, 1.0))
        with self.assertRaises(ValueError):
            ht.zeros((2, 3,)).transpose(axes=(0, 3))

    def test_tril(self):
        local_ones = ht.ones((5,))

        # 1D case, no offset, data is not split, module-level call
        result = ht.tril(local_ones)
        comparison = torch.ones((5, 5,)).tril()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.lshape, (5, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        # 1D case, positive offset, data is not split, module-level call
        result = ht.tril(local_ones, k=2)
        comparison = torch.ones((5, 5,)).tril(diagonal=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.lshape, (5, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        # 1D case, negative offset, data is not split, module-level call
        result = ht.tril(local_ones, k=-2)
        comparison = torch.ones((5, 5,)).tril(diagonal=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.lshape, (5, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        local_ones = ht.ones((4, 5,))

        # 2D case, no offset, data is not split, method
        result = local_ones.tril()
        comparison = torch.ones((4, 5,)).tril()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.lshape, (4, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        # 2D case, positive offset, data is not split, method
        result = local_ones.tril(k=2)
        comparison = torch.ones((4, 5,)).tril(diagonal=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.lshape, (4, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        # 2D case, negative offset, data is not split, method
        result = local_ones.tril(k=-2)
        comparison = torch.ones((4, 5,)).tril(diagonal=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.lshape, (4, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        local_ones = ht.ones((3, 4, 5, 6))

        # 2D+ case, no offset, data is not split, module-level call
        result = local_ones.tril()
        comparison = torch.ones((5, 6,)).tril()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (3, 4, 5, 6,))
        self.assertEqual(result.lshape, (3, 4, 5, 6,))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._tensor__array[i, j] == comparison).all())

        # 2D+ case, positive offset, data is not split, module-level call
        result = local_ones.tril(k=2)
        comparison = torch.ones((5, 6,)).tril(diagonal=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (3, 4, 5, 6,))
        self.assertEqual(result.lshape, (3, 4, 5, 6,))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._tensor__array[i, j] == comparison).all())

        # # 2D+ case, negative offset, data is not split, module-level call
        result = local_ones.tril(k=-2)
        comparison = torch.ones((5, 6,)).tril(diagonal=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (3, 4, 5, 6,))
        self.assertEqual(result.lshape, (3, 4, 5, 6,))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._tensor__array[i, j] == comparison).all())

        distributed_ones = ht.ones((5,), split=0)

        # 1D case, no offset, data is split, method
        result = distributed_ones.tril()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertTrue(result.sum(), 15)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 1)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 0)

        # 1D case, positive offset, data is split, method
        result = distributed_ones.tril(k=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 22)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 1)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 0)

        # 1D case, negative offset, data is split, method
        result = distributed_ones.tril(k=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 6)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 1)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 0)

        distributed_ones = ht.ones((4, 5,), split=0)

        # 2D case, no offset, data is horizontally split, method
        result = distributed_ones.tril()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 10)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[0, -1] == 0)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[-1, 0] == 1)

        # 2D case, positive offset, data is horizontally split, method
        result = distributed_ones.tril(k=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 17)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[0, -1] == 0)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[-1, 0] == 1)

        # 2D case, negative offset, data is horizontally split, method
        result = distributed_ones.tril(k=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 3)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[0, -1] == 0)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[-1, 0] == 1)

        distributed_ones = ht.ones((4, 5,), split=1)

        # 2D case, no offset, data is vertically split, method
        result = distributed_ones.tril()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 10)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 1)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 0)

        # 2D case, positive offset, data is horizontally split, method
        result = distributed_ones.tril(k=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 17)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 1)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 0)

        # 2D case, negative offset, data is horizontally split, method
        result = distributed_ones.tril(k=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 3)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 1)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 0)

    def test_triu(self):
        local_ones = ht.ones((5,))

        # 1D case, no offset, data is not split, module-level call
        result = ht.triu(local_ones)
        comparison = torch.ones((5, 5,)).triu()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.lshape, (5, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        # 1D case, positive offset, data is not split, module-level call
        result = ht.triu(local_ones, k=2)
        comparison = torch.ones((5, 5,)).triu(diagonal=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.lshape, (5, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        # 1D case, negative offset, data is not split, module-level call
        result = ht.triu(local_ones, k=-2)
        comparison = torch.ones((5, 5,)).triu(diagonal=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.lshape, (5, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        local_ones = ht.ones((4, 5,))

        # 2D case, no offset, data is not split, method
        result = local_ones.triu()
        comparison = torch.ones((4, 5,)).triu()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.lshape, (4, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        # 2D case, positive offset, data is not split, method
        result = local_ones.triu(k=2)
        comparison = torch.ones((4, 5,)).triu(diagonal=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.lshape, (4, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        # 2D case, negative offset, data is not split, method
        result = local_ones.triu(k=-2)
        comparison = torch.ones((4, 5,)).triu(diagonal=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.lshape, (4, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        local_ones = ht.ones((3, 4, 5, 6))

        # 2D+ case, no offset, data is not split, module-level call
        result = local_ones.triu()
        comparison = torch.ones((5, 6,)).triu()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (3, 4, 5, 6,))
        self.assertEqual(result.lshape, (3, 4, 5, 6,))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._tensor__array[i, j] == comparison).all())

        # 2D+ case, positive offset, data is not split, module-level call
        result = local_ones.triu(k=2)
        comparison = torch.ones((5, 6,)).triu(diagonal=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (3, 4, 5, 6,))
        self.assertEqual(result.lshape, (3, 4, 5, 6,))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._tensor__array[i, j] == comparison).all())

        # # 2D+ case, negative offset, data is not split, module-level call
        result = local_ones.triu(k=-2)
        comparison = torch.ones((5, 6,)).triu(diagonal=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (3, 4, 5, 6,))
        self.assertEqual(result.lshape, (3, 4, 5, 6,))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._tensor__array[i, j] == comparison).all())

        distributed_ones = ht.ones((5,), split=0)

        # 1D case, no offset, data is split, method
        result = distributed_ones.triu()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertTrue(result.sum(), 15)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 0)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 1)

        # 1D case, positive offset, data is split, method
        result = distributed_ones.triu(k=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 6)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 0)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 1)

        # 1D case, negative offset, data is split, method
        result = distributed_ones.triu(k=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 22)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 0)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 1)

        distributed_ones = ht.ones((4, 5,), split=0)

        # 2D case, no offset, data is horizontally split, method
        result = distributed_ones.triu()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 14)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[0, -1] == 1)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[-1, 0] == 0)

        # # 2D case, positive offset, data is horizontally split, method
        result = distributed_ones.triu(k=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 6)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[0, -1] == 1)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[-1, 0] == 0)

        # # 2D case, negative offset, data is horizontally split, method
        result = distributed_ones.triu(k=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 19)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[0, -1] == 1)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[-1, 0] == 0)

        distributed_ones = ht.ones((4, 5,), split=1)

        # 2D case, no offset, data is vertically split, method
        result = distributed_ones.triu()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 14)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 0)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 1)

        # 2D case, positive offset, data is horizontally split, method
        result = distributed_ones.triu(k=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 6)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 0)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 1)

        # 2D case, negative offset, data is horizontally split, method
        result = distributed_ones.triu(k=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 19)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 0)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 1)

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
            iris = ht.load('/home/d.coquelin/.git/heat/heat/datasets/data/iris.h5', 'data', split=sp)
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
            iris = ht.load('/home/d.coquelin/.git/heat/heat/datasets/data/iris.h5', 'data', split=sp)
            self.assertAlmostEqual(ht.var(iris, bessel=True), 3.90318519755147, 5)
            assert all([a == b for a, b in zip(ht.var(iris, axis=0, bessel=True), ax0)])
