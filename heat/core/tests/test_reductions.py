import unittest
import torch

import heat as ht

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
        self.assertTrue(out_noaxis._tensor__array == shape_noaxis._tensor__array.sum())

        # check sum over all float elements of split 1d tensor
        shape_noaxis_split = ht.arange(array_len, split=0)
        shape_noaxis_split_sum = shape_noaxis_split.sum()

        self.assertIsInstance(shape_noaxis_split_sum, ht.tensor)
        self.assertEqual(shape_noaxis_split_sum.shape, (1,))
        self.assertEqual(shape_noaxis_split_sum.lshape, (1,))
        self.assertEqual(shape_noaxis_split_sum.dtype, ht.int64)
        self.assertEqual(shape_noaxis_split_sum._tensor__array.dtype, torch.int64)
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
        self.assertEqual(split_axis_sum.shape, (1, 3, 3))
        self.assertEqual(split_axis_sum.dtype, ht.float32)
        self.assertEqual(split_axis_sum._tensor__array.dtype, torch.float32)
        self.assertEqual(split_axis_sum.split, None)

        out_noaxis = ht.zeros((1, 3, 3,))
        ht.sum(shape_noaxis, axis=0, out=out_noaxis)
        self.assertTrue((out_noaxis._tensor__array == torch.full((1, 3, 3,), 3)).all())

        # check sum over all float elements of splitted 5d tensor with negative axis
        shape_noaxis_split_axis_neg = ht.ones((1, 2, 3, 4, 5), split=1)
        shape_noaxis_split_axis_neg_sum = shape_noaxis_split_axis_neg.sum(axis=-2)

        self.assertIsInstance(shape_noaxis_split_axis_neg_sum, ht.tensor)
        self.assertEqual(shape_noaxis_split_axis_neg_sum.shape, (1, 2, 3, 1, 5))
        self.assertEqual(shape_noaxis_split_axis_neg_sum.dtype, ht.float32)
        self.assertEqual(shape_noaxis_split_axis_neg_sum._tensor__array.dtype, torch.float32)
        self.assertEqual(shape_noaxis_split_axis_neg_sum.split, 1)

        out_noaxis = ht.zeros((1, 2, 3, 1, 5))
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
