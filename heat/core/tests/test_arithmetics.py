import operator

import torch
import unittest

import heat as ht


class TestArithmetics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a_scalar = 2.0
        cls.an_int_scalar = 2

        cls.a_vector = ht.float32([2, 2])
        cls.another_vector = ht.float32([2, 2, 2])

        cls.a_tensor = ht.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])
        cls.another_tensor = ht.array([
            [2.0, 2.0],
            [2.0, 2.0]
        ])
        cls.a_split_tensor = cls.another_tensor.copy().resplit(0)

        cls.errorneous_type = (2, 2)

    def test_add(self):
        result = ht.array([
            [3.0, 4.0],
            [5.0, 6.0]
        ])

        self.assertTrue(ht.equal(ht.add(self.a_scalar, self.a_scalar), ht.float32([4.0])))
        self.assertTrue(ht.equal(ht.add(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.add(self.a_scalar, self.a_tensor), result))
        self.assertTrue(ht.equal(ht.add(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.add(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.add(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.add(self.a_split_tensor, self.a_tensor), result))

        with self.assertRaises(ValueError):
            ht.add(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.add(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.add('T', 's')

    def test_div(self):
        result = ht.array([
            [0.5, 1.0],
            [1.5, 2.0]
        ])
        commutated_result = ht.array([
            [2.0,     1.0],
            [2.0/3.0, 0.5]
        ])

        self.assertTrue(ht.equal(ht.div(self.a_scalar, self.a_scalar), ht.float32([1.0])))
        self.assertTrue(ht.equal(ht.div(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.div(self.a_scalar, self.a_tensor), commutated_result))
        self.assertTrue(ht.equal(ht.div(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.div(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.div(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.div(self.a_split_tensor, self.a_tensor), commutated_result))

        with self.assertRaises(ValueError):
            ht.div(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.div(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.div('T', 's')

    def test_fmod(self):
        result = ht.array([
            [1., 0.],
            [1., 0.]
        ])
        an_int_tensor = ht.array([
            [5, 3],
            [4, 1]
        ])
        integer_result = ht.array([
            [1, 1],
            [0, 1]
        ])
        commutated_result = ht.array([
            [0.0, 0.0],
            [2.0, 2.0]
        ])
        zero_tensor = ht.zeros((2, 2,))

        a_float = ht.array([5.3])
        another_float = ht.array([1.9])
        result_float = ht.array([1.5])

        self.assertTrue(ht.equal(ht.fmod(self.a_scalar, self.a_scalar), ht.float32([0.0])))
        self.assertTrue(ht.equal(ht.fmod(self.a_tensor, self.a_tensor), zero_tensor))
        self.assertTrue(ht.equal(ht.fmod(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.fmod(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.fmod(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.fmod(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.fmod(an_int_tensor, self.an_int_scalar), integer_result))
        self.assertTrue(ht.equal(ht.fmod(self.a_scalar, self.a_tensor), commutated_result))
        self.assertTrue(ht.equal(ht.fmod(self.a_split_tensor, self.a_tensor), commutated_result))
        self.assertTrue(ht.allclose(ht.fmod(a_float, another_float), result_float))

        with self.assertRaises(ValueError):
            ht.fmod(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.fmod(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.fmod('T', 's')

    def test_mod(self):
        a_tensor = ht.array([
            [1, 4],
            [2, 2]
        ])
        another_tensor = ht.array([
            [1, 2],
            [3, 4]
        ])
        a_result = ht.array([
            [0, 0],
            [2, 2]
        ])
        another_result = ht.array([
            [1, 0],
            [0, 0]
        ])

        self.assertTrue(ht.equal(ht.mod(a_tensor, another_tensor), a_result))
        self.assertTrue(ht.equal(ht.mod(a_tensor, self.an_int_scalar), another_result))
        self.assertTrue(ht.equal(ht.mod(self.an_int_scalar, another_tensor), a_result))

    def test_mul(self):
        result = ht.array([
            [2.0, 4.0],
            [6.0, 8.0]
        ])

        self.assertTrue(ht.equal(ht.mul(self.a_scalar, self.a_scalar), ht.array([4.0])))
        self.assertTrue(ht.equal(ht.mul(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.mul(self.a_scalar, self.a_tensor), result))
        self.assertTrue(ht.equal(ht.mul(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.mul(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.mul(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.mul(self.a_split_tensor, self.a_tensor), result))

        with self.assertRaises(ValueError):
            ht.mul(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.mul(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.mul('T', 's')

    def test_pow(self):
        result = ht.array([
            [1.0,  4.0],
            [9.0, 16.0]
        ])
        commutated_result = ht.array([
            [2.0,  4.0],
            [8.0, 16.0]
        ])

        self.assertTrue(ht.equal(ht.pow(self.a_scalar, self.a_scalar), ht.array([4.0])))
        self.assertTrue(ht.equal(ht.pow(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.pow(self.a_scalar, self.a_tensor), commutated_result))
        self.assertTrue(ht.equal(ht.pow(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.pow(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.pow(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.pow(self.a_split_tensor, self.a_tensor), commutated_result))

        with self.assertRaises(ValueError):
            ht.pow(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.pow(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.pow('T', 's')

    def test_sub(self):
        result = ht.array([
            [-1.0, 0.0],
            [1.0, 2.0]
        ])
        minus_result = ht.array([
            [1.0,  0.0],
            [-1.0, -2.0]
        ])

        self.assertTrue(ht.equal(ht.sub(self.a_scalar, self.a_scalar), ht.array([0.0])))
        self.assertTrue(ht.equal(ht.sub(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.sub(self.a_scalar, self.a_tensor), minus_result))
        self.assertTrue(ht.equal(ht.sub(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.sub(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.sub(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.sub(self.a_split_tensor, self.a_tensor), minus_result))

        with self.assertRaises(ValueError):
            ht.sub(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.sub(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.sub('T', 's')

    def test_sum(self):
        array_len = 11

        # check sum over all float elements of 1d tensor locally
        shape_noaxis = ht.ones(array_len)
        no_axis_sum = shape_noaxis.sum()

        self.assertIsInstance(no_axis_sum, ht.DNDarray)
        self.assertEqual(no_axis_sum.shape, (1,))
        self.assertEqual(no_axis_sum.lshape, (1,))
        self.assertEqual(no_axis_sum.dtype, ht.float32)
        self.assertEqual(no_axis_sum._DNDarray__array.dtype, torch.float32)
        self.assertEqual(no_axis_sum.split, None)
        self.assertEqual(no_axis_sum._DNDarray__array, array_len)

        out_noaxis = ht.zeros((1,))
        ht.sum(shape_noaxis, out=out_noaxis)
        self.assertTrue(out_noaxis._DNDarray__array == shape_noaxis._DNDarray__array.sum())

        # check sum over all float elements of split 1d tensor
        shape_noaxis_split = ht.arange(array_len, split=0)
        shape_noaxis_split_sum = shape_noaxis_split.sum()

        self.assertIsInstance(shape_noaxis_split_sum, ht.DNDarray)
        self.assertEqual(shape_noaxis_split_sum.shape, (1,))
        self.assertEqual(shape_noaxis_split_sum.lshape, (1,))
        self.assertEqual(shape_noaxis_split_sum.dtype, ht.int64)
        self.assertEqual(shape_noaxis_split_sum._DNDarray__array.dtype, torch.int64)
        self.assertEqual(shape_noaxis_split_sum.split, None)
        self.assertEqual(shape_noaxis_split_sum, 55)

        out_noaxis = ht.zeros((1,))
        ht.sum(shape_noaxis_split, out=out_noaxis)
        self.assertEqual(out_noaxis._DNDarray__array, 55)

        # check sum over all float elements of 3d tensor locally
        shape_noaxis = ht.ones((3, 3, 3))
        no_axis_sum = shape_noaxis.sum()

        self.assertIsInstance(no_axis_sum, ht.DNDarray)
        self.assertEqual(no_axis_sum.shape, (1,))
        self.assertEqual(no_axis_sum.lshape, (1,))
        self.assertEqual(no_axis_sum.dtype, ht.float32)
        self.assertEqual(no_axis_sum._DNDarray__array.dtype, torch.float32)
        self.assertEqual(no_axis_sum.split, None)
        self.assertEqual(no_axis_sum._DNDarray__array, 27)

        out_noaxis = ht.zeros((1,))
        ht.sum(shape_noaxis, out=out_noaxis)
        self.assertEqual(out_noaxis._DNDarray__array, 27)

        # check sum over all float elements of split 3d tensor
        shape_noaxis_split_axis = ht.ones((3, 3, 3), split=0)
        split_axis_sum = shape_noaxis_split_axis.sum(axis=0)

        self.assertIsInstance(split_axis_sum, ht.DNDarray)
        self.assertEqual(split_axis_sum.shape, (3, 3))
        self.assertEqual(split_axis_sum.dtype, ht.float32)
        self.assertEqual(split_axis_sum._DNDarray__array.dtype, torch.float32)
        self.assertEqual(split_axis_sum.split, None)

        out_noaxis = ht.zeros((3, 3,))
        ht.sum(shape_noaxis, axis=0, out=out_noaxis)
        self.assertTrue((out_noaxis._DNDarray__array == torch.full((3, 3,), 3)).all())

        # check sum over all float elements of splitted 5d tensor with negative axis
        shape_noaxis_split_axis_neg = ht.ones((1, 2, 3, 4, 5), split=1)
        shape_noaxis_split_axis_neg_sum = shape_noaxis_split_axis_neg.sum(axis=-2)

        self.assertIsInstance(shape_noaxis_split_axis_neg_sum, ht.DNDarray)
        self.assertEqual(shape_noaxis_split_axis_neg_sum.shape, (1, 2, 3, 5))
        self.assertEqual(shape_noaxis_split_axis_neg_sum.dtype, ht.float32)
        self.assertEqual(shape_noaxis_split_axis_neg_sum._DNDarray__array.dtype, torch.float32)
        self.assertEqual(shape_noaxis_split_axis_neg_sum.split, 1)

        out_noaxis = ht.zeros((1, 2, 3, 5))
        ht.sum(shape_noaxis_split_axis_neg, axis=-2, out=out_noaxis)

        # check sum over all float elements of splitted 3d tensor with tuple axis
        shape_split_axis_tuple = ht.ones((3, 4, 5), split=1)
        shape_split_axis_tuple_sum = shape_split_axis_tuple.sum(axis=(-2, -3))
        expected_result = ht.ones((5,)) * 12.

        self.assertIsInstance(shape_split_axis_tuple_sum, ht.DNDarray)
        self.assertEqual(shape_split_axis_tuple_sum.shape, (5,))
        self.assertEqual(shape_split_axis_tuple_sum.dtype, ht.float32)
        self.assertEqual(shape_split_axis_tuple_sum._DNDarray__array.dtype, torch.float32)
        self.assertEqual(shape_split_axis_tuple_sum.split, None)
        self.assertTrue((shape_split_axis_tuple_sum == expected_result).all())

        # exceptions
        with self.assertRaises(ValueError):
            ht.ones(array_len).sum(axis=1)
        with self.assertRaises(ValueError):
            ht.ones(array_len).sum(axis=-2)
        with self.assertRaises(ValueError):
            ht.ones((4, 4)).sum(axis=0, out=out_noaxis)
        with self.assertRaises(TypeError):
            ht.ones(array_len).sum(axis='bad_axis_type')

    def test_right_hand_side_operations(self):
        """
        This test ensures that for each arithmetic operation (e.g. +, -, *, ...) that is implemented in the tensor
        class, it works both ways.

        Examples
        --------
        >>> import heat as ht
        >>> T = ht.float32([[1., 2.], [3., 4.]])
        >>> assert T * 3 == 3 * T

        """
        operators = (
            ('__add__', operator.add, True),
            ('__sub__', operator.sub, False),
            ('__mul__', operator.mul, True),
            ('__truediv__', operator.truediv, False),
            ('__floordiv__', operator.floordiv, False),
            ('__mod__', operator.mod, False),
            ('__pow__', operator.pow, False)
        )
        tensor = ht.float32([[1, 4], [2, 3]])
        num = 3
        for (attr, op, commutative) in operators:
            try:
                func = tensor.__getattribute__(attr)
            except AttributeError:
                continue
            self.assertTrue(callable(func))
            res_1 = op(tensor, num)
            res_2 = op(num, tensor)
            if commutative:
                self.assertTrue(ht.equal(res_1, res_2))
        # TODO: Test with split tensors when binary operations are working properly for split tensors
