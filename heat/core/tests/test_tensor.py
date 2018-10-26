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

        # check sum over all float elements of splitted 1d tensor 
        shape_noaxis_split = ht.ones(array_len, split=0)
        self.assertEqual(shape_noaxis_split.sum().shape, (1,))
        self.assertEqual(shape_noaxis_split.sum().lshape, (1,))
        self.assertIsInstance(shape_noaxis_split.sum(), ht.tensor)
        self.assertEqual(shape_noaxis_split.sum().dtype, ht.float32)
        self.assertEqual(shape_noaxis_split._tensor__array.dtype, torch.float32)
        self.assertAlmostEqual(shape_noaxis_split.sum(), float(array_len), places=7, msg=None, delta=None)
        self.assertEqual(shape_noaxis_split.sum().split, None)
       
        # check sum over all integer elements of 1d tensor locally 
        shape_noaxis_int = ht.ones(array_len).astype(ht.int)
        self.assertEqual(shape_noaxis_int.sum(axis=0).shape, (1,))
        self.assertEqual(shape_noaxis_int.sum(axis=0).lshape, (1,))
        self.assertIsInstance(shape_noaxis_int.sum(axis=0), ht.tensor)
        self.assertEqual(shape_noaxis_int.sum(axis=0).dtype, ht.int64)
        self.assertEqual(shape_noaxis_int.sum()._tensor__array.dtype, torch.int64)
        self.assertEqual(shape_noaxis_int.sum(), array_len)
        self.assertEqual(shape_noaxis_int.sum().split, None)

        # check sum over all integer elements of splitted 1d tensor
        shape_noaxis_split_int = ht.ones(array_len, split=0).astype(ht.int)
        self.assertEqual(shape_noaxis_split_int.sum().shape, (1,))
        self.assertEqual(shape_noaxis_split_int.sum().lshape, (1,))
        self.assertIsInstance(shape_noaxis_split_int.sum(), ht.tensor)
        self.assertEqual(shape_noaxis_split_int.sum().dtype, ht.int64)
        self.assertEqual(shape_noaxis_split_int.sum()._tensor__array.dtype, torch.int64)
        self.assertEqual(shape_noaxis_split_int.sum(), array_len)
        self.assertEqual(shape_noaxis_split_int.sum().split, None)

        # check sum over all float elements of 3d tensor locally 
        shape_noaxis = ht.ones((3,3,3))
        self.assertEqual(shape_noaxis.sum().shape, (1,))
        self.assertEqual(shape_noaxis.sum().lshape, (1,))
        self.assertIsInstance(shape_noaxis.sum(), ht.tensor)
        self.assertEqual(shape_noaxis.sum().dtype, ht.float32)
        self.assertEqual(shape_noaxis.sum()._tensor__array.dtype, torch.float32)
        self.assertAlmostEqual(shape_noaxis.sum(), 27., places=7, msg=None, delta=None)
        self.assertEqual(shape_noaxis.sum().split, None)
    
        # check sum over all float elements of splitted 3d tensor 
        shape_noaxis_split_axis = ht.ones((3,3,3), split=0)
        self.assertIsInstance(shape_noaxis_split_axis.sum(axis=1), ht.tensor)
        self.assertEqual(shape_noaxis.sum(axis=1).shape, (3,1,3))
        self.assertEqual(shape_noaxis_split_axis.sum(axis=1).dtype, ht.float32)
        self.assertEqual(shape_noaxis_split_axis.sum(axis=1)._tensor__array.dtype, torch.float32)
        self.assertEqual(shape_noaxis_split_axis.sum().split, None)
        
        # check sum over all float elements of splitted 5d tensor with negative axis 
        shape_noaxis_split_axis_neg = ht.ones((1,2,3,4,5), split=0)
        self.assertIsInstance(shape_noaxis_split_axis_neg.sum(axis=-2), ht.tensor)
        self.assertEqual(shape_noaxis_split_axis_neg.sum(axis=-2).shape, (1,2,3,1,5))
        self.assertEqual(shape_noaxis_split_axis_neg.sum(axis=-2).dtype, ht.float32)
        self.assertEqual(shape_noaxis_split_axis_neg.sum(axis=-2)._tensor__array.dtype, torch.float32)
        self.assertEqual(shape_noaxis_split_axis_neg.sum().split, None)
               
                
        # exceptions
        with self.assertRaises(ValueError):
            ht.ones(9).sum(axis=1)
        with self.assertRaises(ValueError):
            ht.ones(9).sum(axis=2)
        with self.assertRaises(ValueError):
            ht.ones(9).sum(axis='bad_axis_value')
 

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

    def test_arange(self):
        # testing one positional integer argument 
        one_arg_arange_int = ht.arange(10)
        self.assertIsInstance(one_arg_arange_int, ht.tensor)
        self.assertEqual(one_arg_arange_int.shape, (10,))
        self.assertLessEqual(one_arg_arange_int.lshape[0], 10)
        self.assertEqual(one_arg_arange_int.dtype, ht.int32)
        self.assertEqual(one_arg_arange_int._tensor__array.dtype, torch.int32)
        self.assertEqual(one_arg_arange_int.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(one_arg_arange_int.sum(), 45)
        
        # testing one positional float argument
        one_arg_arange_float = ht.arange(10.)
        self.assertIsInstance(one_arg_arange_float, ht.tensor)
        self.assertEqual(one_arg_arange_float.shape, (10,))
        self.assertLessEqual(one_arg_arange_float.lshape[0], 10)
        self.assertEqual(one_arg_arange_float.dtype, ht.int32)
        self.assertEqual(one_arg_arange_float._tensor__array.dtype, torch.int32)
        self.assertEqual(one_arg_arange_float.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(one_arg_arange_float.sum(), 45.0)

        # testing two positional integer arguments
        two_arg_arange_int = ht.arange(0, 10)
        self.assertIsInstance(two_arg_arange_int, ht.tensor)
        self.assertEqual(two_arg_arange_int.shape, (10,))
        self.assertLessEqual(two_arg_arange_int.lshape[0], 10)
        self.assertEqual(two_arg_arange_int.dtype, ht.int32)
        self.assertEqual(two_arg_arange_int._tensor__array.dtype, torch.int32)
        self.assertEqual(two_arg_arange_int.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(two_arg_arange_int.sum(), 45)

        # testing two positional arguments, one being float
        two_arg_arange_float = ht.arange(0., 10)
        self.assertIsInstance(two_arg_arange_float, ht.tensor)
        self.assertEqual(two_arg_arange_float.shape, (10,))
        self.assertLessEqual(two_arg_arange_float.lshape[0], 10)
        self.assertEqual(two_arg_arange_float.dtype, ht.float32)
        self.assertEqual(two_arg_arange_float._tensor__array.dtype, torch.float32)
        self.assertEqual(two_arg_arange_float.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(two_arg_arange_float.sum(), 45.0)

        # testing three positional integer arguments
        three_arg_arange_int = ht.arange(0, 10, 2)
        self.assertIsInstance(three_arg_arange_int, ht.tensor)
        self.assertEqual(three_arg_arange_int.shape, (5,))
        self.assertLessEqual(three_arg_arange_int.lshape[0], 5)
        self.assertEqual(three_arg_arange_int.dtype, ht.int32)
        self.assertEqual(three_arg_arange_int._tensor__array.dtype, torch.int32)
        self.assertEqual(three_arg_arange_int.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(three_arg_arange_int.sum(), 20)

        # testing three positional arguments, one being float
        three_arg_arange_float = ht.arange(0, 10, 2.)
        self.assertIsInstance(three_arg_arange_float, ht.tensor)
        self.assertEqual(three_arg_arange_float.shape, (5,))
        self.assertLessEqual(three_arg_arange_float.lshape[0], 5)
        self.assertEqual(three_arg_arange_float.dtype, ht.float32)
        self.assertEqual(three_arg_arange_float._tensor__array.dtype, torch.float32)
        self.assertEqual(three_arg_arange_float.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(three_arg_arange_float.sum(), 20.0)

        # testing splitting
        three_arg_arange_dtype_float32 = ht.arange(0, 10, 2., split=0)
        self.assertIsInstance(three_arg_arange_dtype_float32, ht.tensor)
        self.assertEqual(three_arg_arange_dtype_float32.shape, (5,))
        self.assertLessEqual(three_arg_arange_dtype_float32.lshape[0], 5)
        self.assertEqual(three_arg_arange_dtype_float32.dtype, ht.float32)
        self.assertEqual(three_arg_arange_dtype_float32._tensor__array.dtype, torch.float32)
        self.assertEqual(three_arg_arange_dtype_float32.split, 0)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(three_arg_arange_dtype_float32.sum(axis=0), 20.0)

        # testing setting dtype to int16
        three_arg_arange_dtype_short = ht.arange(0, 10, 2., dtype=torch.int16)
        self.assertIsInstance(three_arg_arange_dtype_short, ht.tensor)
        self.assertEqual(three_arg_arange_dtype_short.shape, (5,))
        self.assertLessEqual(three_arg_arange_dtype_short.lshape[0], 5)
        self.assertEqual(three_arg_arange_dtype_short.dtype, ht.int16)
        self.assertEqual(three_arg_arange_dtype_short._tensor__array.dtype, torch.int16)
        self.assertEqual(three_arg_arange_dtype_short.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(three_arg_arange_dtype_short.sum(axis=0), 20)

        # testing setting dtype to float64
        three_arg_arange_dtype_float64 = ht.arange(0, 10, 2, dtype=torch.float64)
        self.assertIsInstance(three_arg_arange_dtype_float64, ht.tensor)
        self.assertEqual(three_arg_arange_dtype_float64.shape, (5,))
        self.assertLessEqual(three_arg_arange_dtype_float64.lshape[0], 5)
        self.assertEqual(three_arg_arange_dtype_float64.dtype, ht.float64)
        self.assertEqual(three_arg_arange_dtype_float64._tensor__array.dtype, torch.float64)
        self.assertEqual(three_arg_arange_dtype_float64.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(three_arg_arange_dtype_float64.sum(axis=0), 20.0)

        # exceptions
        with self.assertRaises(ValueError):
            ht.arange(-5, 3, split=1)
        with self.assertRaises(TypeError):
            ht.arange()
        with self.assertRaises(TypeError):
            ht.arange(1, 2, 3, 4)

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

    def test_randn(self):
        # scalar input
        simple_randn_float = ht.randn(3)
        self.assertIsInstance(simple_randn_float, ht.tensor)
        self.assertEqual(simple_randn_float.shape,  (3,))
        self.assertEqual(simple_randn_float.lshape, (3,))
        self.assertEqual(simple_randn_float.split,  None)
        self.assertEqual(simple_randn_float.dtype,  ht.float32)

        # multi-dimensional
        elaborate_randn_float = ht.randn(2, 3)
        self.assertIsInstance(elaborate_randn_float, ht.tensor)
        self.assertEqual(elaborate_randn_float.shape,  (2, 3))
        self.assertEqual(elaborate_randn_float.lshape, (2, 3))
        self.assertEqual(elaborate_randn_float.split,  None)
        self.assertEqual(elaborate_randn_float.dtype,  ht.float32)

        #TODO: double-check this
        # split axis
        elaborate_randn_split = ht.randn(6, 4, dtype=ht.float32, split=0)
        self.assertIsInstance(elaborate_randn_split, ht.tensor)
        self.assertEqual(elaborate_randn_split.shape,         (6, 4,))
        self.assertLessEqual(elaborate_randn_split.lshape[0], 6)
        self.assertEqual(elaborate_randn_split.lshape[1],     4)
        self.assertEqual(elaborate_randn_split.split,         0)
        self.assertEqual(elaborate_randn_split.dtype,         ht.float32)

        # exceptions
        with self.assertRaises(TypeError):
            ht.randn('(2, 3,)', dtype=ht.float64)
        with self.assertRaises(ValueError):
            ht.randn(-1, 3, dtype=ht.float64)
        with self.assertRaises(TypeError):
            ht.randn(2, 3, dtype=ht.float64, split='axis')

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
