import torch
import unittest

import heat as ht


class TestTensor(unittest.TestCase):
    def test_astype(self):
        data = ht.float32([
            [1, 2, 3],
            [4, 5, 6]
        ])

        # check starting invariant
        self.assertEqual(data.dtype, ht.float32)

        # check the copy case for uint8
        as_uint8 = data.astype(ht.uint8)
        self.assertIsInstance(as_uint8, ht.tensor)
        self.assertEqual(as_uint8.dtype, ht.uint8)
        self.assertEqual(as_uint8._tensor__array.dtype, torch.uint8)
        self.assertIsNot(as_uint8, data)

        # check the copy case for uint8
        as_float64 = data.astype(ht.float64, copy=False)
        self.assertIsInstance(as_float64, ht.tensor)
        self.assertEqual(as_float64.dtype, ht.float64)
        self.assertEqual(as_float64._tensor__array.dtype, torch.float64)
        self.assertIs(as_float64, data)


class TestTensorFactories(unittest.TestCase):
    def test_linspace(self):
        # simple linear space
        ascending = ht.linspace(-3, 5)
        self.assertIsInstance(ascending, ht.tensor)
        self.assertEqual(ascending.shape, (50,))
        self.assertLessEqual(ascending.lshape[0], 50)
        self.assertEqual(ascending.dtype, ht.float32)
        self.assertEqual(ascending._tensor__array.dtype, torch.float32)
        self.assertEqual(ascending.split, None)

        # simple inverse linear space
        descending = ht.linspace(-5, 3, num=100)
        self.assertIsInstance(descending, ht.tensor)
        self.assertEqual(descending.shape, (100,))
        self.assertLessEqual(descending.lshape[0], 100)
        self.assertEqual(descending.dtype, ht.float32)
        self.assertEqual(descending._tensor__array.dtype, torch.float32)
        self.assertEqual(descending.split, None)

        # split linear space
        split = ht.linspace(-5, 3, num=70, split=0)
        self.assertIsInstance(split, ht.tensor)
        self.assertEqual(split.shape, (70,))
        self.assertLessEqual(split.lshape[0], 70)
        self.assertEqual(split.dtype, ht.float32)
        self.assertEqual(split._tensor__array.dtype, torch.float32)
        self.assertEqual(split.split, 0)

        # with casted type
        casted = ht.linspace(-5, 3, num=70, dtype=ht.uint8, split=0)
        self.assertIsInstance(casted, ht.tensor)
        self.assertEqual(casted.shape, (70,))
        self.assertLessEqual(casted.lshape[0], 70)
        self.assertEqual(casted.dtype, ht.uint8)
        self.assertEqual(casted._tensor__array.dtype, torch.uint8)
        self.assertEqual(casted.split, 0)

        # retstep test
        result = ht.linspace(-5, 3, num=70, retstep=True, dtype=ht.uint8, split=0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        self.assertIsInstance(result[0], ht.tensor)
        self.assertEqual(result[0].shape, (70,))
        self.assertLessEqual(result[0].lshape[0], 70)
        self.assertEqual(result[0].dtype, ht.uint8)
        self.assertEqual(result[0]._tensor__array.dtype, torch.uint8)
        self.assertEqual(result[0].split, 0)

        self.assertIsInstance(result[1], float)
        self.assertEqual(result[1], 0.11594202898550725)

        # exceptions
        with self.assertRaises(ValueError):
            ht.linspace(-5, 3, split=1)
        with self.assertRaises(ValueError):
            ht.linspace(-5, 3, num=-1)
        with self.assertRaises(ValueError):
            ht.linspace(-5, 3, num=0)

    def test_ones(self):
        # scalar input
        simple_ones_float = ht.ones(3)
        self.assertIsInstance(simple_ones_float, ht.tensor)
        self.assertEqual(simple_ones_float.shape,  (3,))
        self.assertEqual(simple_ones_float.lshape, (3,))
        self.assertEqual(simple_ones_float.split,  None)
        self.assertEqual(simple_ones_float.dtype,  ht.float32)
        self.assertEqual((simple_ones_float._tensor__array == 1).all().item(), 1)

        # different data type
        simple_ones_uint = ht.ones(5, dtype=ht.bool)
        self.assertIsInstance(simple_ones_uint, ht.tensor)
        self.assertEqual(simple_ones_uint.shape,  (5,))
        self.assertEqual(simple_ones_uint.lshape, (5,))
        self.assertEqual(simple_ones_uint.split,  None)
        self.assertEqual(simple_ones_uint.dtype,  ht.bool)
        self.assertEqual((simple_ones_uint._tensor__array == 1).all().item(), 1)

        # multi-dimensional
        elaborate_ones_int = ht.ones((2, 3,), dtype=ht.int32)
        self.assertIsInstance(elaborate_ones_int, ht.tensor)
        self.assertEqual(elaborate_ones_int.shape,  (2, 3,))
        self.assertEqual(elaborate_ones_int.lshape, (2, 3,))
        self.assertEqual(elaborate_ones_int.split,  None)
        self.assertEqual(elaborate_ones_int.dtype,  ht.int32)
        self.assertEqual((elaborate_ones_int._tensor__array == 1).all().item(), 1)

        # split axis
        elaborate_ones_split = ht.ones((6, 4,), dtype=ht.int32, split=0)
        self.assertIsInstance(elaborate_ones_split, ht.tensor)
        self.assertEqual(elaborate_ones_split.shape,         (6, 4,))
        self.assertLessEqual(elaborate_ones_split.lshape[0], 6)
        self.assertEqual(elaborate_ones_split.lshape[1],     4)
        self.assertEqual(elaborate_ones_split.split,         0)
        self.assertEqual(elaborate_ones_split.dtype,         ht.int32)
        self.assertEqual((elaborate_ones_split._tensor__array == 1).all().item(), 1)

        # exceptions
        with self.assertRaises(TypeError):
            ht.ones('(2, 3,)', dtype=ht.float64)
        with self.assertRaises(ValueError):
            ht.ones((-1, 3,), dtype=ht.float64)
        with self.assertRaises(TypeError):
            ht.ones((2, 3,), dtype=ht.float64, split='axis')

    def test_zeros(self):
        # scalar input
        simple_zeros_float = ht.zeros(3)
        self.assertIsInstance(simple_zeros_float, ht.tensor)
        self.assertEqual(simple_zeros_float.shape,  (3,))
        self.assertEqual(simple_zeros_float.lshape, (3,))
        self.assertEqual(simple_zeros_float.split,  None)
        self.assertEqual(simple_zeros_float.dtype,  ht.float32)
        self.assertEqual((simple_zeros_float._tensor__array == 0).all().item(), 1)

        # different data type
        simple_zeros_uint = ht.zeros(5, dtype=ht.bool)
        self.assertIsInstance(simple_zeros_uint, ht.tensor)
        self.assertEqual(simple_zeros_uint.shape,  (5,))
        self.assertEqual(simple_zeros_uint.lshape, (5,))
        self.assertEqual(simple_zeros_uint.split,  None)
        self.assertEqual(simple_zeros_uint.dtype,  ht.bool)
        self.assertEqual((simple_zeros_uint._tensor__array == 0).all().item(), 1)

        # multi-dimensional
        elaborate_zeros_int = ht.zeros((2, 3,), dtype=ht.int32)
        self.assertIsInstance(elaborate_zeros_int, ht.tensor)
        self.assertEqual(elaborate_zeros_int.shape,  (2, 3,))
        self.assertEqual(elaborate_zeros_int.lshape, (2, 3,))
        self.assertEqual(elaborate_zeros_int.split,  None)
        self.assertEqual(elaborate_zeros_int.dtype,  ht.int32)
        self.assertEqual((elaborate_zeros_int._tensor__array == 0).all().item(), 1)

        # split axis
        elaborate_zeros_split = ht.zeros((6, 4,), dtype=ht.int32, split=0)
        self.assertIsInstance(elaborate_zeros_split, ht.tensor)
        self.assertEqual(elaborate_zeros_split.shape,         (6, 4,))
        self.assertLessEqual(elaborate_zeros_split.lshape[0], 6)
        self.assertEqual(elaborate_zeros_split.lshape[1],     4)
        self.assertEqual(elaborate_zeros_split.split,         0)
        self.assertEqual(elaborate_zeros_split.dtype,         ht.int32)
        self.assertEqual((elaborate_zeros_split._tensor__array == 0).all().item(), 1)

        # exceptions
        with self.assertRaises(TypeError):
            ht.zeros('(2, 3,)', dtype=ht.float64)
        with self.assertRaises(ValueError):
            ht.zeros((-1, 3,), dtype=ht.float64)
        with self.assertRaises(TypeError):
            ht.zeros((2, 3,), dtype=ht.float64, split='axis')

    def test_ones_like(self):
        # scalar
        like_int = ht.ones_like(3)
        self.assertIsInstance(like_int, ht.tensor)
        self.assertEqual(like_int.shape,  (1,))
        self.assertEqual(like_int.lshape, (1,))
        self.assertEqual(like_int.split,  None)
        self.assertEqual(like_int.dtype,  ht.int32)
        self.assertEqual((like_int._tensor__array == 1).all().item(), 1)

        # sequence
        like_str = ht.ones_like('abc')
        self.assertIsInstance(like_str, ht.tensor)
        self.assertEqual(like_str.shape,  (3,))
        self.assertEqual(like_str.lshape, (3,))
        self.assertEqual(like_str.split,  None)
        self.assertEqual(like_str.dtype,  ht.float32)
        self.assertEqual((like_str._tensor__array == 1).all().item(), 1)

        # elaborate tensor
        zeros = ht.zeros((2, 3,), dtype=ht.uint8)
        like_zeros = ht.ones_like(zeros)
        self.assertIsInstance(like_zeros, ht.tensor)
        self.assertEqual(like_zeros.shape,  (2, 3,))
        self.assertEqual(like_zeros.lshape, (2, 3,))
        self.assertEqual(like_zeros.split,  None)
        self.assertEqual(like_zeros.dtype,  ht.uint8)
        self.assertEqual((like_zeros._tensor__array == 1).all().item(), 1)

        # elaborate tensor with split
        zeros_split = ht.zeros((2, 3,), dtype=ht.uint8, split=0)
        like_zeros_split = ht.ones_like(zeros_split)
        self.assertIsInstance(like_zeros_split,          ht.tensor)
        self.assertEqual(like_zeros_split.shape,         (2, 3,))
        self.assertLessEqual(like_zeros_split.lshape[0], 2)
        self.assertEqual(like_zeros_split.lshape[1],     3)
        self.assertEqual(like_zeros_split.split,         0)
        self.assertEqual(like_zeros_split.dtype,         ht.uint8)
        self.assertEqual((like_zeros_split._tensor__array == 1).all().item(), 1)

        # exceptions
        with self.assertRaises(TypeError):
            ht.ones_like(zeros, dtype='abc')
        with self.assertRaises(TypeError):
            ht.ones_like(zeros, split='axis')

    def test_zeros_like(self):
        # scalar
        like_int = ht.zeros_like(3)
        self.assertIsInstance(like_int, ht.tensor)
        self.assertEqual(like_int.shape,  (1,))
        self.assertEqual(like_int.lshape, (1,))
        self.assertEqual(like_int.split,  None)
        self.assertEqual(like_int.dtype,  ht.int32)
        self.assertEqual((like_int._tensor__array == 0).all().item(), 1)

        # sequence
        like_str = ht.zeros_like('abc')
        self.assertIsInstance(like_str, ht.tensor)
        self.assertEqual(like_str.shape,  (3,))
        self.assertEqual(like_str.lshape, (3,))
        self.assertEqual(like_str.split,  None)
        self.assertEqual(like_str.dtype,  ht.float32)
        self.assertEqual((like_str._tensor__array == 0).all().item(), 1)

        # elaborate tensor
        ones = ht.ones((2, 3,), dtype=ht.uint8)
        like_ones = ht.zeros_like(ones)
        self.assertIsInstance(like_ones, ht.tensor)
        self.assertEqual(like_ones.shape,  (2, 3,))
        self.assertEqual(like_ones.lshape, (2, 3,))
        self.assertEqual(like_ones.split,  None)
        self.assertEqual(like_ones.dtype,  ht.uint8)
        self.assertEqual((like_ones._tensor__array == 0).all().item(), 1)

        # elaborate tensor with split
        ones_split = ht.ones((2, 3,), dtype=ht.uint8, split=0)
        like_ones_split = ht.zeros_like(ones_split)
        self.assertIsInstance(like_ones_split,          ht.tensor)
        self.assertEqual(like_ones_split.shape,         (2, 3,))
        self.assertLessEqual(like_ones_split.lshape[0], 2)
        self.assertEqual(like_ones_split.lshape[1],     3)
        self.assertEqual(like_ones_split.split,         0)
        self.assertEqual(like_ones_split.dtype,         ht.uint8)
        self.assertEqual((like_ones_split._tensor__array == 0).all().item(), 1)

        # exceptions
        with self.assertRaises(TypeError):
            ht.zeros_like(ones, dtype='abc')
        with self.assertRaises(TypeError):
            ht.zeros_like(ones, split='axis')
