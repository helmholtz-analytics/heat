import torch
import unittest

import heat as ht

FLOAT_EPSILON = 1e-4


class TestRelational(unittest.TestCase):
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
        cls.split_ones_tensor = ht.ones((2, 2), split=1)

        cls.errorneous_type = (2, 2)

    def test_eq(self):
        result = ht.uint8([
            [0, 1],
            [0, 0]
        ])

        self.assertTrue(ht.equal(ht.eq(self.a_scalar, self.a_scalar), ht.uint8([1])))
        self.assertTrue(ht.equal(ht.eq(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.eq(self.a_scalar, self.a_tensor), result))
        self.assertTrue(ht.equal(ht.eq(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.eq(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.eq(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.eq(self.a_split_tensor, self.a_tensor), result))

        with self.assertRaises(ValueError):
            ht.eq(self.a_tensor, self.another_vector)
        with self.assertRaises(NotImplementedError):
            ht.eq(self.a_tensor, self.split_ones_tensor)
        with self.assertRaises(TypeError):
            ht.eq(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.eq('self.a_tensor', 's')

    def test_equal(self):
        self.assertTrue(ht.equal(self.a_tensor, self.a_tensor))
        self.assertFalse(ht.equal(self.a_tensor, self.another_tensor))
        self.assertFalse(ht.equal(self.a_tensor, self.a_scalar))
        self.assertFalse(ht.equal(self.another_tensor, self.a_scalar))

    def test_ge(self):
        result = ht.uint8([
            [0, 1],
            [1, 1]
        ])
        commutated_result = ht.uint8([
            [1, 1],
            [0, 0]
        ])

        self.assertTrue(ht.equal(ht.ge(self.a_scalar, self.a_scalar), ht.uint8([1])))
        self.assertTrue(ht.equal(ht.ge(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.ge(self.a_scalar, self.a_tensor), commutated_result))
        self.assertTrue(ht.equal(ht.ge(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.ge(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.ge(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.ge(self.a_split_tensor, self.a_tensor), commutated_result))

        with self.assertRaises(ValueError):
            ht.ge(self.a_tensor, self.another_vector)
        with self.assertRaises(NotImplementedError):
            ht.ge(self.a_tensor, self.split_ones_tensor)
        with self.assertRaises(TypeError):
            ht.ge(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.ge('self.a_tensor', 's')

    def test_gt(self):
        result = ht.uint8([
            [0, 0],
            [1, 1]
        ])
        commutated_result = ht.uint8([
            [1, 0],
            [0, 0]
        ])

        self.assertTrue(ht.equal(ht.gt(self.a_scalar, self.a_scalar), ht.uint8([0])))
        self.assertTrue(ht.equal(ht.gt(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.gt(self.a_scalar, self.a_tensor), commutated_result))
        self.assertTrue(ht.equal(ht.gt(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.gt(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.gt(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.gt(self.a_split_tensor, self.a_tensor), commutated_result))

        with self.assertRaises(ValueError):
            ht.gt(self.a_tensor, self.another_vector)
        with self.assertRaises(NotImplementedError):
            ht.gt(self.a_tensor, self.split_ones_tensor)
        with self.assertRaises(TypeError):
            ht.gt(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.gt('self.a_tensor', 's')

    def test_le(self):
        result = ht.uint8([
            [1, 1],
            [0, 0]
        ])
        commutated_result = ht.uint8([
            [0, 1],
            [1, 1]
        ])

        self.assertTrue(ht.equal(ht.le(self.a_scalar, self.a_scalar), ht.uint8([1])))
        self.assertTrue(ht.equal(ht.le(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.le(self.a_scalar, self.a_tensor), commutated_result))
        self.assertTrue(ht.equal(ht.le(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.le(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.le(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.le(self.a_split_tensor, self.a_tensor), commutated_result))

        with self.assertRaises(ValueError):
            ht.le(self.a_tensor, self.another_vector)
        with self.assertRaises(NotImplementedError):
            ht.le(self.a_tensor, self.split_ones_tensor)
        with self.assertRaises(TypeError):
            ht.le(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.le('self.a_tensor', 's')

    def test_lt(self):
        result = ht.uint8([
            [1, 0],
            [0, 0]
        ])
        commutated_result = ht.uint8([
            [0, 0],
            [1, 1]
        ])

        self.assertTrue(ht.equal(ht.lt(self.a_scalar, self.a_scalar), ht.uint8([0])))
        self.assertTrue(ht.equal(ht.lt(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.lt(self.a_scalar, self.a_tensor), commutated_result))
        self.assertTrue(ht.equal(ht.lt(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.lt(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.lt(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.lt(self.a_split_tensor, self.a_tensor), commutated_result))

        with self.assertRaises(ValueError):
            ht.lt(self.a_tensor, self.another_vector)
        with self.assertRaises(NotImplementedError):
            ht.lt(self.a_tensor, self.split_ones_tensor)
        with self.assertRaises(TypeError):
            ht.lt(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.lt('self.a_tensor', 's')

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

        self.assertIsInstance(maximum, ht.Tensor)
        self.assertEqual(maximum.shape, (1,))
        self.assertEqual(maximum.lshape, (1,))
        self.assertEqual(maximum.split, None)
        self.assertEqual(maximum.dtype, ht.int64)
        self.assertEqual(maximum._Tensor__array.dtype, torch.int64)
        self.assertEqual(maximum, 12)

        # maximum along first axis
        maximum_vertical = ht.max(ht_array, axis=0)

        self.assertIsInstance(maximum_vertical, ht.Tensor)
        self.assertEqual(maximum_vertical.shape, (3,))
        self.assertEqual(maximum_vertical.lshape, (1, 3,))
        self.assertEqual(maximum_vertical.split, None)
        self.assertEqual(maximum_vertical.dtype, ht.int64)
        self.assertEqual(maximum_vertical._Tensor__array.dtype, torch.int64)
        self.assertTrue((maximum_vertical._Tensor__array ==
                         comparison.max(dim=0, keepdim=True)[0]).all())

        # maximum along second axis
        maximum_horizontal = ht.max(ht_array, axis=1)

        self.assertIsInstance(maximum_horizontal, ht.Tensor)
        self.assertEqual(maximum_horizontal.shape, (4,))
        self.assertEqual(maximum_horizontal.lshape, (4, 1,))
        self.assertEqual(maximum_horizontal.split, None)
        self.assertEqual(maximum_horizontal.dtype, ht.int64)
        self.assertEqual(maximum_horizontal._Tensor__array.dtype, torch.int64)
        self.assertTrue((maximum_horizontal._Tensor__array == comparison.max(dim=1, keepdim=True)[0]).all())

        # check max over all float elements of split 3d tensor, across split axis
        random_volume = ht.random.randn(3, 3, 3, split=1)
        maximum_volume = ht.max(random_volume, axis=1)

        self.assertIsInstance(maximum_volume, ht.Tensor)
        self.assertEqual(maximum_volume.shape, (3, 3))
        self.assertEqual(maximum_volume.lshape, (3, 1, 3))
        self.assertEqual(maximum_volume.dtype, ht.float32)
        self.assertEqual(maximum_volume._Tensor__array.dtype, torch.float32)
        self.assertEqual(maximum_volume.split, None)

        # check max over all float elements of split 5d tensor, along split axis
        random_5d = ht.random.randn(1, 2, 3, 4, 5, split=0)
        maximum_5d = ht.max(random_5d, axis=1)

        self.assertIsInstance(maximum_5d, ht.Tensor)
        self.assertEqual(maximum_5d.shape, (1, 3, 4, 5))
        self.assertLessEqual(maximum_5d.lshape[1], 2)
        self.assertEqual(maximum_5d.dtype, ht.float32)
        self.assertEqual(maximum_5d._Tensor__array.dtype, torch.float32)
        self.assertEqual(maximum_5d.split, 0)

        # check exceptions
        with self.assertRaises(NotImplementedError):
            ht_array.max(axis=(0, 1))
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

        self.assertIsInstance(minimum, ht.Tensor)
        self.assertEqual(minimum.shape, (1,))
        self.assertEqual(minimum.lshape, (1,))
        self.assertEqual(minimum.split, None)
        self.assertEqual(minimum.dtype, ht.int64)
        self.assertEqual(minimum._Tensor__array.dtype, torch.int64)
        self.assertEqual(minimum, 12)

        # maximum along first axis
        minimum_vertical = ht.min(ht_array, axis=0)

        self.assertIsInstance(minimum_vertical, ht.Tensor)
        self.assertEqual(minimum_vertical.shape, (3,))
        self.assertEqual(minimum_vertical.lshape, (1, 3,))
        self.assertEqual(minimum_vertical.split, None)
        self.assertEqual(minimum_vertical.dtype, ht.int64)
        self.assertEqual(minimum_vertical._Tensor__array.dtype, torch.int64)
        self.assertTrue((minimum_vertical._Tensor__array == comparison.min(dim=0, keepdim=True)[0]).all())

        # maximum along second axis
        minimum_horizontal = ht.min(ht_array, axis=1)

        self.assertIsInstance(minimum_horizontal, ht.Tensor)
        self.assertEqual(minimum_horizontal.shape, (4,))
        self.assertEqual(minimum_horizontal.lshape, (4, 1,))
        self.assertEqual(minimum_horizontal.split, None)
        self.assertEqual(minimum_horizontal.dtype, ht.int64)
        self.assertEqual(minimum_horizontal._Tensor__array.dtype, torch.int64)
        self.assertTrue((minimum_horizontal._Tensor__array == comparison.min(dim=1, keepdim=True)[0]).all())

        # check max over all float elements of split 3d tensor, across split axis
        random_volume = ht.random.randn(3, 3, 3, split=1)
        minimum_volume = ht.min(random_volume, axis=1)

        self.assertIsInstance(minimum_volume, ht.Tensor)
        self.assertEqual(minimum_volume.shape, (3, 3))
        self.assertEqual(minimum_volume.lshape, (3, 1, 3))
        self.assertEqual(minimum_volume.dtype, ht.float32)
        self.assertEqual(minimum_volume._Tensor__array.dtype, torch.float32)
        self.assertEqual(minimum_volume.split, None)

        # check max over all float elements of split 5d tensor, along split axis
        random_5d = ht.random.randn(1, 2, 3, 4, 5, split=0)
        minimum_5d = ht.min(random_5d, axis=1)

        self.assertIsInstance(minimum_5d, ht.Tensor)
        self.assertEqual(minimum_5d.shape, (1, 3, 4, 5))
        self.assertLessEqual(minimum_5d.lshape[1], 2)
        self.assertEqual(minimum_5d.dtype, ht.float32)
        self.assertEqual(minimum_5d._Tensor__array.dtype, torch.float32)
        self.assertEqual(minimum_5d.split, 0)

        # check exceptions
        with self.assertRaises(NotImplementedError):
            ht_array.min(axis=(0, 1))
        with self.assertRaises(TypeError):
            ht_array.min(axis=1.1)
        with self.assertRaises(TypeError):
            ht_array.min(axis='y')
        with self.assertRaises(ValueError):
            ht.min(ht_array, axis=-4)

    def test_ne(self):
        result = ht.uint8([
            [1, 0],
            [1, 1]
        ])

        self.assertTrue(ht.equal(ht.ne(self.a_scalar, self.a_scalar), ht.uint8([0])))
        self.assertTrue(ht.equal(ht.ne(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.ne(self.a_scalar, self.a_tensor), result))
        self.assertTrue(ht.equal(ht.ne(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.ne(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.ne(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.ne(self.a_split_tensor, self.a_tensor), result))

        with self.assertRaises(ValueError):
            ht.ne(self.a_tensor, self.another_vector)
        with self.assertRaises(NotImplementedError):
            ht.ne(self.a_tensor, self.split_ones_tensor)
        with self.assertRaises(TypeError):
            ht.ne(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.ne('self.a_tensor', 's')
