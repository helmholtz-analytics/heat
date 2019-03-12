import unittest
import torch

import heat as ht

FLOAT_EPSILON = 1e-4

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
T_s = ht.tensor(T1._tensor__array, T1.shape, T1.dtype, 0, None, None)
Ts = ht.ones((2,2), split=1)
otherType = (2,2)

class TestOperations(unittest.TestCase):

    def test_equal(self):
        self.assertTrue(ht.equal(T, T))
        self.assertFalse(ht.equal(T, T1))
        self.assertFalse(ht.equal(T, s))
        self.assertFalse(ht.equal(T1, s))


    def test_eq(self):
        T_r = ht.uint8([
            [0, 1],
            [0, 0]
        ])

        self.assertTrue(ht.equal(ht.eq(s, s), ht.uint8([1])))
        self.assertTrue(ht.equal(ht.eq(T, s), T_r))
        self.assertTrue(ht.equal(ht.eq(s, T), T_r))
        self.assertTrue(ht.equal(ht.eq(T, T1), T_r))
        self.assertTrue(ht.equal(ht.eq(T, v), T_r))
        self.assertTrue(ht.equal(ht.eq(T, s_int), T_r))
        self.assertTrue(ht.equal(ht.eq(T_s, T), T_r))

        with self.assertRaises(ValueError):
            ht.eq(T, v2)
        with self.assertRaises(NotImplementedError):
            ht.eq(T, Ts)
        with self.assertRaises(TypeError):
            ht.eq(T, otherType)
        with self.assertRaises(TypeError):
            ht.eq('T', 's')


    def test_ge(self):
        T_r = ht.uint8([
            [0, 1],
            [1, 1]
        ])

        T_inv = ht.uint8([
            [1, 1],
            [0, 0]
        ])

        self.assertTrue(ht.equal(ht.ge(s, s), ht.uint8([1])))
        self.assertTrue(ht.equal(ht.ge(T, s), T_r))
        self.assertTrue(ht.equal(ht.ge(s, T), T_inv))
        self.assertTrue(ht.equal(ht.ge(T, T1), T_r))
        self.assertTrue(ht.equal(ht.ge(T, v), T_r))
        self.assertTrue(ht.equal(ht.ge(T, s_int), T_r))
        self.assertTrue(ht.equal(ht.ge(T_s, T), T_inv))

        with self.assertRaises(ValueError):
            ht.ge(T, v2)
        with self.assertRaises(NotImplementedError):
            ht.ge(T, Ts)
        with self.assertRaises(TypeError):
            ht.ge(T, otherType)
        with self.assertRaises(TypeError):
            ht.ge('T', 's')


    def test_gt(self):
        T_r = ht.uint8([
            [0, 0],
            [1, 1]
        ])

        T_inv = ht.uint8([
            [1, 0],
            [0, 0]
        ])

        self.assertTrue(ht.equal(ht.gt(s, s), ht.uint8([0])))
        self.assertTrue(ht.equal(ht.gt(T, s), T_r))
        self.assertTrue(ht.equal(ht.gt(s, T), T_inv))
        self.assertTrue(ht.equal(ht.gt(T, T1), T_r))
        self.assertTrue(ht.equal(ht.gt(T, v), T_r))
        self.assertTrue(ht.equal(ht.gt(T, s_int), T_r))
        self.assertTrue(ht.equal(ht.gt(T_s, T), T_inv))

        with self.assertRaises(ValueError):
            ht.gt(T, v2)
        with self.assertRaises(NotImplementedError):
            ht.gt(T, Ts)
        with self.assertRaises(TypeError):
            ht.gt(T, otherType)
        with self.assertRaises(TypeError):
            ht.gt('T', 's')



    def test_le(self):
        T_r = ht.uint8([
            [1, 1],
            [0, 0]
        ])

        T_inv = ht.uint8([
            [0, 1],
            [1, 1]
        ])

        self.assertTrue(ht.equal(ht.le(s, s), ht.uint8([1])))
        self.assertTrue(ht.equal(ht.le(T, s), T_r))
        self.assertTrue(ht.equal(ht.le(s, T), T_inv))
        self.assertTrue(ht.equal(ht.le(T, T1), T_r))
        self.assertTrue(ht.equal(ht.le(T, v), T_r))
        self.assertTrue(ht.equal(ht.le(T, s_int), T_r))
        self.assertTrue(ht.equal(ht.le(T_s, T), T_inv))

        with self.assertRaises(ValueError):
            ht.le(T, v2)
        with self.assertRaises(NotImplementedError):
            ht.le(T, Ts)
        with self.assertRaises(TypeError):
            ht.le(T, otherType)
        with self.assertRaises(TypeError):
            ht.le('T', 's')

    def test_lt(self):
        T_r = ht.uint8([
            [1, 0],
            [0, 0]
        ])

        T_inv = ht.uint8([
            [0, 0],
            [1, 1]
        ])

        self.assertTrue(ht.equal(ht.lt(s, s), ht.uint8([0])))
        self.assertTrue(ht.equal(ht.lt(T, s), T_r))
        self.assertTrue(ht.equal(ht.lt(s, T), T_inv))
        self.assertTrue(ht.equal(ht.lt(T, T1), T_r))
        self.assertTrue(ht.equal(ht.lt(T, v), T_r))
        self.assertTrue(ht.equal(ht.lt(T, s_int), T_r))
        self.assertTrue(ht.equal(ht.lt(T_s, T), T_inv))

        with self.assertRaises(ValueError):
            ht.lt(T, v2)
        with self.assertRaises(NotImplementedError):
            ht.lt(T, Ts)
        with self.assertRaises(TypeError):
            ht.lt(T, otherType)
        with self.assertRaises(TypeError):
            ht.lt('T', 's')


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

        self.assertIsInstance(maximum, ht.tensor)
        self.assertEqual(maximum.shape, (1,))
        self.assertEqual(maximum.lshape, (1,))
        self.assertEqual(maximum.split, None)
        self.assertEqual(maximum.dtype, ht.int64)
        self.assertEqual(maximum._tensor__array.dtype, torch.int64)
        self.assertEqual(maximum, 12)

        # maximum along first axis
        maximum_vertical = ht.max(ht_array, axis=0)

        self.assertIsInstance(maximum_vertical, ht.tensor)
        self.assertEqual(maximum_vertical.shape, (1, 3,))
        self.assertEqual(maximum_vertical.lshape, (1, 3,))
        self.assertEqual(maximum_vertical.split, None)
        self.assertEqual(maximum_vertical.dtype, ht.int64)
        self.assertEqual(maximum_vertical._tensor__array.dtype, torch.int64)
        self.assertTrue((maximum_vertical._tensor__array == comparison.max(dim=0, keepdim=True)[0]).all())

        # maximum along second axis
        maximum_horizontal = ht.max(ht_array, axis=1)

        self.assertIsInstance(maximum_horizontal, ht.tensor)
        self.assertEqual(maximum_horizontal.shape, (4, 1,))
        self.assertEqual(maximum_horizontal.lshape, (4, 1,))
        self.assertEqual(maximum_horizontal.split, None)
        self.assertEqual(maximum_horizontal.dtype, ht.int64)
        self.assertEqual(maximum_horizontal._tensor__array.dtype, torch.int64)
        self.assertTrue((maximum_horizontal._tensor__array == comparison.max(dim=1, keepdim=True)[0]).all())

        # check max over all float elements of split 3d tensor, across split axis
        random_volume = ht.random.randn(3, 3, 3, split=1)
        maximum_volume = ht.max(random_volume, axis=1)

        self.assertIsInstance(maximum_volume, ht.tensor)
        self.assertEqual(maximum_volume.shape, (3, 1, 3))
        self.assertEqual(maximum_volume.lshape, (3, 1, 3))
        self.assertEqual(maximum_volume.dtype, ht.float32)
        self.assertEqual(maximum_volume._tensor__array.dtype, torch.float32)
        self.assertEqual(maximum_volume.split, None)

        # check max over all float elements of split 5d tensor, along split axis
        random_5d = ht.random.randn(1, 2, 3, 4, 5, split=0)
        maximum_5d = ht.max(random_5d, axis=1)

        self.assertIsInstance(maximum_5d, ht.tensor)
        self.assertEqual(maximum_5d.shape, (1, 1, 3, 4, 5))
        self.assertLessEqual(maximum_5d.lshape[1], 2)
        self.assertEqual(maximum_5d.dtype, ht.float32)
        self.assertEqual(maximum_5d._tensor__array.dtype, torch.float32)
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

        self.assertIsInstance(minimum, ht.tensor)
        self.assertEqual(minimum.shape, (1,))
        self.assertEqual(minimum.lshape, (1,))
        self.assertEqual(minimum.split, None)
        self.assertEqual(minimum.dtype, ht.int64)
        self.assertEqual(minimum._tensor__array.dtype, torch.int64)
        self.assertEqual(minimum, 12)

        # maximum along first axis
        minimum_vertical = ht.min(ht_array, axis=0)

        self.assertIsInstance(minimum_vertical, ht.tensor)
        self.assertEqual(minimum_vertical.shape, (1, 3,))
        self.assertEqual(minimum_vertical.lshape, (1, 3,))
        self.assertEqual(minimum_vertical.split, None)
        self.assertEqual(minimum_vertical.dtype, ht.int64)
        self.assertEqual(minimum_vertical._tensor__array.dtype, torch.int64)
        self.assertTrue((minimum_vertical._tensor__array == comparison.min(dim=0, keepdim=True)[0]).all())

        # maximum along second axis
        minimum_horizontal = ht.min(ht_array, axis=1)

        self.assertIsInstance(minimum_horizontal, ht.tensor)
        self.assertEqual(minimum_horizontal.shape, (4, 1,))
        self.assertEqual(minimum_horizontal.lshape, (4, 1,))
        self.assertEqual(minimum_horizontal.split, None)
        self.assertEqual(minimum_horizontal.dtype, ht.int64)
        self.assertEqual(minimum_horizontal._tensor__array.dtype, torch.int64)
        self.assertTrue((minimum_horizontal._tensor__array == comparison.min(dim=1, keepdim=True)[0]).all())

        # check max over all float elements of split 3d tensor, across split axis
        random_volume = ht.random.randn(3, 3, 3, split=1)
        minimum_volume = ht.min(random_volume, axis=1)

        self.assertIsInstance(minimum_volume, ht.tensor)
        self.assertEqual(minimum_volume.shape, (3, 1, 3))
        self.assertEqual(minimum_volume.lshape, (3, 1, 3))
        self.assertEqual(minimum_volume.dtype, ht.float32)
        self.assertEqual(minimum_volume._tensor__array.dtype, torch.float32)
        self.assertEqual(minimum_volume.split, None)

        # check max over all float elements of split 5d tensor, along split axis
        random_5d = ht.random.randn(1, 2, 3, 4, 5, split=0)
        minimum_5d = ht.min(random_5d, axis=1)

        self.assertIsInstance(minimum_5d, ht.tensor)
        self.assertEqual(minimum_5d.shape, (1, 1, 3, 4, 5))
        self.assertLessEqual(minimum_5d.lshape[1], 2)
        self.assertEqual(minimum_5d.dtype, ht.float32)
        self.assertEqual(minimum_5d._tensor__array.dtype, torch.float32)
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
        T_r = ht.uint8([
            [1, 0],
            [1, 1]
        ])

        self.assertTrue(ht.equal(ht.ne(s, s), ht.uint8([0])))
        self.assertTrue(ht.equal(ht.ne(T, s), T_r))
        self.assertTrue(ht.equal(ht.ne(s, T), T_r))
        self.assertTrue(ht.equal(ht.ne(T, T1), T_r))
        self.assertTrue(ht.equal(ht.ne(T, v), T_r))
        self.assertTrue(ht.equal(ht.ne(T, s_int), T_r))
        self.assertTrue(ht.equal(ht.ne(T_s, T), T_r))

        with self.assertRaises(ValueError):
            ht.ne(T, v2)
        with self.assertRaises(NotImplementedError):
            ht.ne(T, Ts)
        with self.assertRaises(TypeError):
            ht.ne(T, otherType)
        with self.assertRaises(TypeError):
            ht.ne('T', 's')

