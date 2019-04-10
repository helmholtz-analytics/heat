import torch
import unittest

import heat as ht

T = ht.float32([
    [1, 2],
    [3, 4]
])
s = 2.0
s_int = 2
T1 = ht.float32([
    [2, 2],
    [2, 2]
])
v = ht.float32([2, 2])
v2 = ht.float32([2, 2, 2])
T_s = ht.Tensor(T1._Tensor__array, T1.shape, T1.dtype, 0, T1.device, T1.comm)
otherType = (2, 2)


class TestArithmetics(unittest.TestCase):
    def test_add(self):
        T_r = ht.float32([
            [3, 4],
            [5, 6]
        ])
        
        self.assertTrue(ht.equal(ht.add(s, s), ht.float32([4.0])))
        self.assertTrue(ht.equal(ht.add(T, s), T_r))
        self.assertTrue(ht.equal(ht.add(s, T), T_r))
        self.assertTrue(ht.equal(ht.add(T, T1), T_r))
        self.assertTrue(ht.equal(ht.add(T, v), T_r))
        self.assertTrue(ht.equal(ht.add(T, s_int), T_r))
        self.assertTrue(ht.equal(ht.add(T_s, T), T_r))

        with self.assertRaises(ValueError):
            ht.add(T, v2)
        with self.assertRaises(NotImplementedError):
            ht.add(T, T_s)
        with self.assertRaises(TypeError):
            ht.add(T, otherType)
        with self.assertRaises(TypeError):
            ht.add('T', 's')

    def test_div(self):
        T_r = ht.float32([
            [0.5, 1],
            [1.5, 2]
        ])

        T_inv = ht.float32([
            [2, 1],
            [2/3, 0.5]
        ])

        self.assertTrue(ht.equal(ht.div(s, s), ht.float32([1.0])))
        self.assertTrue(ht.equal(ht.div(T, s),T_r))
        self.assertTrue(ht.equal(ht.div(s, T), T_inv))
        self.assertTrue(ht.equal(ht.div(T, T1), T_r))
        self.assertTrue(ht.equal(ht.div(T, v), T_r))
        self.assertTrue(ht.equal(ht.div(T, s_int), T_r))
        self.assertTrue(ht.equal(ht.div(T_s, T), T_inv))

        with self.assertRaises(ValueError):
            ht.div(T, v2)
        with self.assertRaises(NotImplementedError):
            ht.sub(T, T_s)
        with self.assertRaises(TypeError):
            ht.div(T, otherType)
        with self.assertRaises(TypeError):
            ht.div('T', 's')

    def test_fmod(self):
        T_r = ht.float32([
            [1., 0.],
            [1., 0.]
        ])
        T_int = ht.int32([
            [5, 3],
            [4, 1]
        ])
        T_r_int = ht.int32([
            [1, 1],
            [0, 1]
        ])
        T_inv = ht.float32([
            [0.0, 0.0],
            [2.0, 2.0]
        ])
        T_zero = ht.float32([
            [0.0, 0.0],
            [0.0, 0.0]
        ])
        float1 = ht.float32([5.3])
        float2 = ht.float32([1.9])
        float_res = ht.float32([1.5])

        self.assertTrue(ht.equal(ht.fmod(s, s), ht.float32([0.0])))
        self.assertTrue(ht.equal(ht.fmod(T, T), T_zero))
        self.assertTrue(ht.equal(ht.fmod(T, s_int), T_r))
        self.assertTrue(ht.equal(ht.fmod(T, T1), T_r))
        self.assertTrue(ht.equal(ht.fmod(T, v), T_r))
        self.assertTrue(ht.equal(ht.fmod(T, s_int), T_r))
        self.assertTrue(ht.equal(ht.fmod(T_int, s_int), T_r_int))
        self.assertTrue(ht.equal(ht.fmod(s, T), T_inv))
        self.assertTrue(ht.equal(ht.fmod(T_s, T), T_inv))
        self.assertTrue(ht.allclose(ht.fmod(float1, float2), float_res))

        with self.assertRaises(ValueError):
            ht.fmod(T, v2)
        with self.assertRaises(NotImplementedError):
            ht.fmod(T, T_s)
        with self.assertRaises(TypeError):
            ht.fmod(T, otherType)
        with self.assertRaises(TypeError):
            ht.fmod('T', 's')

    def test_mod(self):
        T_int_1 = ht.int32([[1, 4], [2, 2]])
        T_int_2 = ht.int32([[1, 2], [3, 4]])
        T_int_res_1 = ht.int32([[0, 0], [2, 2]])
        T_int_res_2 = ht.int32([[1, 0], [0, 0]])

        self.assertTrue(ht.equal(ht.mod(T_int_1, T_int_2), T_int_res_1))
        self.assertTrue(ht.equal(ht.mod(T_int_1, s_int), T_int_res_2))
        self.assertTrue(ht.equal(ht.mod(s_int, T_int_2), T_int_res_1))

    def test_mul(self):
        T_r = ht.float32([
            [2, 4],
            [6, 8]
        ])

        self.assertTrue(ht.equal(ht.mul(s, s), ht.float32([4.0])))
        self.assertTrue(ht.equal(ht.mul(T, s), T_r))
        self.assertTrue(ht.equal(ht.mul(s, T), T_r))
        self.assertTrue(ht.equal(ht.mul(T, T1), T_r))
        self.assertTrue(ht.equal(ht.mul(T, v), T_r))
        self.assertTrue(ht.equal(ht.mul(T, s_int), T_r))
        self.assertTrue(ht.equal(ht.mul(T_s, T), T_r))

        with self.assertRaises(ValueError):
            ht.mul(T, v2)
        with self.assertRaises(NotImplementedError):
            ht.mul(T, T_s)
        with self.assertRaises(TypeError):
            ht.mul(T, otherType)
        with self.assertRaises(TypeError):
            ht.mul('T', 's')

    def test_pow(self):
        T_r = ht.float32([
            [1, 4],
            [9, 16]
        ])

        T_inv = ht.float32([
            [2, 4],
            [8, 16]
        ])

        self.assertTrue(ht.equal(ht.pow(s, s), ht.float32([4.0])))
        self.assertTrue(ht.equal(ht.pow(T, s), T_r))
        self.assertTrue(ht.equal(ht.pow(s, T), T_inv))
        self.assertTrue(ht.equal(ht.pow(T, T1), T_r))
        self.assertTrue(ht.equal(ht.pow(T, v), T_r))
        self.assertTrue(ht.equal(ht.pow(T, s_int), T_r))
        self.assertTrue(ht.equal(ht.pow(T_s, T), T_inv))

        with self.assertRaises(ValueError):
            ht.pow(T, v2)
        with self.assertRaises(NotImplementedError):
            ht.pow(T, T_s)
        with self.assertRaises(TypeError):
            ht.pow(T, otherType)
        with self.assertRaises(TypeError):
            ht.pow('T', 's')

    def test_sub(self):
        T_r = ht.float32([
            [-1, 0],
            [1, 2]
        ])

        T_r_minus = ht.float32([
            [1, 0],
            [-1, -2]
        ])

        self.assertTrue(ht.equal(ht.sub(s, s), ht.float32([0.0])))
        self.assertTrue(ht.equal(ht.sub(T, s), T_r))
        self.assertTrue(ht.equal(ht.sub(s, T), T_r_minus))
        self.assertTrue(ht.equal(ht.sub(T, T1), T_r))
        self.assertTrue(ht.equal(ht.sub(T, v), T_r))
        self.assertTrue(ht.equal(ht.sub(T, s_int), T_r))
        self.assertTrue(ht.equal(ht.sub(T_s, T), T_r_minus))

        with self.assertRaises(ValueError):
            ht.sub(T, v2)
        with self.assertRaises(NotImplementedError):
            ht.sub(T, T_s)
        with self.assertRaises(TypeError):
            ht.sub(T, otherType)
        with self.assertRaises(TypeError):
            ht.sub('T', 's')

    def test_sum(self):
        array_len = 11

        # check sum over all float elements of 1d tensor locally
        shape_noaxis = ht.ones(array_len)
        no_axis_sum = shape_noaxis.sum()

        self.assertIsInstance(no_axis_sum, ht.Tensor)
        self.assertEqual(no_axis_sum.shape, (1,))
        self.assertEqual(no_axis_sum.lshape, (1,))
        self.assertEqual(no_axis_sum.dtype, ht.float32)
        self.assertEqual(no_axis_sum._Tensor__array.dtype, torch.float32)
        self.assertEqual(no_axis_sum.split, None)
        self.assertEqual(no_axis_sum._Tensor__array, array_len)

        out_noaxis = ht.zeros((1,))
        ht.sum(shape_noaxis, out=out_noaxis)
        self.assertTrue(out_noaxis._Tensor__array ==
                        shape_noaxis._Tensor__array.sum())

        # check sum over all float elements of split 1d tensor
        shape_noaxis_split = ht.arange(array_len, split=0)
        shape_noaxis_split_sum = shape_noaxis_split.sum()

        self.assertIsInstance(shape_noaxis_split_sum, ht.Tensor)
        self.assertEqual(shape_noaxis_split_sum.shape, (1,))
        self.assertEqual(shape_noaxis_split_sum.lshape, (1,))
        self.assertEqual(shape_noaxis_split_sum.dtype, ht.int64)
        self.assertEqual(
            shape_noaxis_split_sum._Tensor__array.dtype, torch.int64)
        self.assertEqual(shape_noaxis_split_sum.split, None)
        self.assertEqual(shape_noaxis_split_sum, 55)

        out_noaxis = ht.zeros((1,))
        ht.sum(shape_noaxis_split, out=out_noaxis)
        self.assertEqual(out_noaxis._Tensor__array, 55)

        # check sum over all float elements of 3d tensor locally
        shape_noaxis = ht.ones((3, 3, 3))
        no_axis_sum = shape_noaxis.sum()

        self.assertIsInstance(no_axis_sum, ht.Tensor)
        self.assertEqual(no_axis_sum.shape, (1,))
        self.assertEqual(no_axis_sum.lshape, (1,))
        self.assertEqual(no_axis_sum.dtype, ht.float32)
        self.assertEqual(no_axis_sum._Tensor__array.dtype, torch.float32)
        self.assertEqual(no_axis_sum.split, None)
        self.assertEqual(no_axis_sum._Tensor__array, 27)

        out_noaxis = ht.zeros((1,))
        ht.sum(shape_noaxis, out=out_noaxis)
        self.assertEqual(out_noaxis._Tensor__array, 27)

        # check sum over all float elements of split 3d tensor
        shape_noaxis_split_axis = ht.ones((3, 3, 3), split=0)
        split_axis_sum = shape_noaxis_split_axis.sum(axis=0)

        self.assertIsInstance(split_axis_sum, ht.Tensor)
        self.assertEqual(split_axis_sum.shape, (3, 3))
        self.assertEqual(split_axis_sum.dtype, ht.float32)
        self.assertEqual(split_axis_sum._Tensor__array.dtype, torch.float32)
        self.assertEqual(split_axis_sum.split, None)

        out_noaxis = ht.zeros((3, 3,))
        ht.sum(shape_noaxis, axis=0, out=out_noaxis)
        self.assertTrue((out_noaxis._Tensor__array ==
                         torch.full((3, 3,), 3)).all())

        # check sum over all float elements of splitted 5d tensor with negative axis
        shape_noaxis_split_axis_neg = ht.ones((1, 2, 3, 4, 5), split=1)
        shape_noaxis_split_axis_neg_sum = shape_noaxis_split_axis_neg.sum(
            axis=-2)

        self.assertIsInstance(shape_noaxis_split_axis_neg_sum, ht.Tensor)
        self.assertEqual(
            shape_noaxis_split_axis_neg_sum.shape, (1, 2, 3, 5))
        self.assertEqual(shape_noaxis_split_axis_neg_sum.dtype, ht.float32)
        self.assertEqual(
            shape_noaxis_split_axis_neg_sum._Tensor__array.dtype, torch.float32)
        self.assertEqual(shape_noaxis_split_axis_neg_sum.split, 1)

        out_noaxis = ht.zeros((1, 2, 3, 5))
        ht.sum(shape_noaxis_split_axis_neg, axis=-2, out=out_noaxis)

        # exceptions
        with self.assertRaises(ValueError):
            ht.ones(array_len).sum(axis=1)
        with self.assertRaises(ValueError):
            ht.ones(array_len).sum(axis=-2)
        with self.assertRaises(ValueError):
            ht.ones((4, 4)).sum(axis=0, out=out_noaxis)
        with self.assertRaises(TypeError):
            ht.ones(array_len).sum(axis='bad_axis_type')
