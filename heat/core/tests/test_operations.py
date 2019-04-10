import unittest
import torch

import heat as ht

FLOAT_EPSILON = 1e-4


class TestOperations(unittest.TestCase):
    def test___binary_op_broadcast(self):
        left_tensor = ht.ones((4, 1), split=0) 
        right_tensor = ht.ones((1, 2), split=0)
        result = left_tensor + right_tensor
        self.assertEqual(result.shape, (4, 2))
        result = right_tensor + left_tensor
        self.assertEqual(result.shape, (4, 2))

        left_tensor = ht.ones((4, 1), split=1) 
        right_tensor = ht.ones((1, 2), split=1)
        result = left_tensor + right_tensor
        self.assertEqual(result.shape, (4, 2))
        result = right_tensor + left_tensor
        self.assertEqual(result.shape, (4, 2))

        left_tensor = ht.ones((4, 1, 3, 1, 2), split=0, dtype=torch.uint8) 
        right_tensor = ht.ones((1, 2, 1, 3, 1), split=0, dtype=torch.uint8)
        result = left_tensor + right_tensor
        self.assertEqual(result.shape, (4, 2, 3, 3, 2))
        result = right_tensor + left_tensor
        self.assertEqual(result.shape, (4, 2, 3, 3, 2))

    def test_all(self):
        array_len = 9

        # check all over all float elements of 1d tensor locally
        ones_noaxis = ht.ones(array_len)
        x = (ones_noaxis == 1).all()

        self.assertIsInstance(x, ht.Tensor)
        self.assertEqual(x.shape, (1,))
        self.assertEqual(x.lshape, (1,))
        self.assertEqual(x.dtype, ht.bool)
        self.assertEqual(x._Tensor__array.dtype, torch.uint8)
        self.assertEqual(x.split, None)
        self.assertEqual(x._Tensor__array, 1)

        out_noaxis = ht.zeros((1,))
        ht.all(ones_noaxis, out=out_noaxis)
        self.assertEqual(out_noaxis._Tensor__array, 1)

        # check all over all float elements of split 1d tensor
        ones_noaxis_split = ht.ones(array_len, split=0)
        floats_is_one = ones_noaxis_split.all()

        self.assertIsInstance(floats_is_one, ht.Tensor)
        self.assertEqual(floats_is_one.shape, (1,))
        self.assertEqual(floats_is_one.lshape, (1,))
        self.assertEqual(floats_is_one.dtype, ht.bool)
        self.assertEqual(floats_is_one._Tensor__array.dtype, torch.uint8)
        self.assertEqual(floats_is_one.split, None)
        self.assertEqual(floats_is_one._Tensor__array, 1)

        out_noaxis = ht.zeros((1,))
        ht.all(ones_noaxis_split, out=out_noaxis)
        self.assertEqual(out_noaxis._Tensor__array, 1)

        # check all over all integer elements of 1d tensor locally
        ones_noaxis_int = ht.ones(array_len).astype(ht.int)
        int_is_one = ones_noaxis_int.all()

        self.assertIsInstance(int_is_one, ht.Tensor)
        self.assertEqual(int_is_one.shape, (1,))
        self.assertEqual(int_is_one.lshape, (1,))
        self.assertEqual(int_is_one.dtype, ht.bool)
        self.assertEqual(int_is_one._Tensor__array.dtype, torch.uint8)
        self.assertEqual(int_is_one.split, None)
        self.assertEqual(int_is_one._Tensor__array, 1)

        out_noaxis = ht.zeros((1,))
        ht.all(ones_noaxis_int, out=out_noaxis)
        self.assertEqual(out_noaxis._Tensor__array, 1)

        # check all over all integer elements of split 1d tensor
        ones_noaxis_split_int = ht.ones(array_len, split=0).astype(ht.int)
        split_int_is_one = ones_noaxis_split_int.all()

        self.assertIsInstance(split_int_is_one, ht.Tensor)
        self.assertEqual(split_int_is_one.shape, (1,))
        self.assertEqual(split_int_is_one.lshape, (1,))
        self.assertEqual(split_int_is_one.dtype, ht.bool)
        self.assertEqual(split_int_is_one._Tensor__array.dtype, torch.uint8)
        self.assertEqual(split_int_is_one.split, None)
        self.assertEqual(split_int_is_one._Tensor__array, 1)

        out_noaxis = ht.zeros((1,))
        ht.all(ones_noaxis_split_int, out=out_noaxis)
        self.assertEqual(out_noaxis._Tensor__array, 1)

        # check all over all float elements of 3d tensor locally
        ones_noaxis_volume = ht.ones((3, 3, 3))
        volume_is_one = ones_noaxis_volume.all()

        self.assertIsInstance(volume_is_one, ht.Tensor)
        self.assertEqual(volume_is_one.shape, (1,))
        self.assertEqual(volume_is_one.lshape, (1,))
        self.assertEqual(volume_is_one.dtype, ht.bool)
        self.assertEqual(volume_is_one._Tensor__array.dtype, torch.uint8)
        self.assertEqual(volume_is_one.split, None)
        self.assertEqual(volume_is_one._Tensor__array, 1)

        out_noaxis = ht.zeros((1,))
        ht.all(ones_noaxis_volume, out=out_noaxis)
        self.assertEqual(out_noaxis._Tensor__array, 1)

        # check sequence is not all one
        sequence = ht.arange(array_len)
        sequence_is_one = sequence.all()

        self.assertIsInstance(sequence_is_one, ht.Tensor)
        self.assertEqual(sequence_is_one.shape, (1,))
        self.assertEqual(sequence_is_one.lshape, (1,))
        self.assertEqual(sequence_is_one.dtype, ht.bool)
        self.assertEqual(sequence_is_one._Tensor__array.dtype, torch.uint8)
        self.assertEqual(sequence_is_one.split, None)
        self.assertEqual(sequence_is_one._Tensor__array, 0)

        out_noaxis = ht.zeros((1,))
        ht.all(sequence, out=out_noaxis)
        self.assertEqual(out_noaxis._Tensor__array, 0)

        # check all over all float elements of split 3d tensor
        ones_noaxis_split_axis = ht.ones((3, 3, 3), split=0)
        float_volume_is_one = ones_noaxis_split_axis.all(axis=0)

        self.assertIsInstance(float_volume_is_one, ht.Tensor)
        self.assertEqual(float_volume_is_one.shape, (3, 3))
        self.assertEqual(float_volume_is_one.all(axis=1).dtype, ht.bool)
        self.assertEqual(float_volume_is_one._Tensor__array.dtype, torch.uint8)
        self.assertEqual(float_volume_is_one.split, None)

        out_noaxis = ht.zeros((3, 3,))
        ht.all(ones_noaxis_split_axis, axis=0, out=out_noaxis)

        # check all over all float elements of split 5d tensor with negative axis
        ones_noaxis_split_axis_neg = ht.zeros((1, 2, 3, 4, 5), split=1)
        float_5d_is_one = ones_noaxis_split_axis_neg.all(axis=-2)

        self.assertIsInstance(float_5d_is_one, ht.Tensor)
        self.assertEqual(float_5d_is_one.shape, (1, 2, 3, 5))
        self.assertEqual(float_5d_is_one.dtype, ht.bool)
        self.assertEqual(float_5d_is_one._Tensor__array.dtype, torch.uint8)
        self.assertEqual(float_5d_is_one.split, 1)

        out_noaxis = ht.zeros((1, 2, 3, 5))
        ht.all(ones_noaxis_split_axis_neg, axis=-2, out=out_noaxis)

        # exceptions
        with self.assertRaises(ValueError):
            ht.ones(array_len).all(axis=1)
        with self.assertRaises(ValueError):
            ht.ones(array_len).all(axis=-2)
        with self.assertRaises(ValueError):
            ht.ones((4, 4)).all(axis=0, out=out_noaxis)
        with self.assertRaises(TypeError):
            ht.ones(array_len).all(axis='bad_axis_type')

    def test_allclose(self):
        a = ht.float32([[2, 2], [2, 2]])
        b = ht.float32([[2.00005, 2.00005], [2.00005, 2.00005]])

        self.assertFalse(ht.allclose(a, b))
        self.assertTrue(ht.allclose(a, b, atol=1e-04))
        self.assertTrue(ht.allclose(a, b, rtol=1e-04))

        with self.assertRaises(TypeError):
            ht.allclose(a, (2, 2, 2, 2))

    def test_any(self):
        x = ht.float32([[2.7, 0, 0],
                        [0, 0, 0],
                        [0, 0.3, 0]])
        any_tensor = ht.any(x, axis=1)
        res = ht.uint8([[1], [0], [1]])
        self.assertIsInstance(any_tensor, ht.Tensor)
        self.assertEqual(any_tensor.shape, (3,))
        self.assertEqual(any_tensor.dtype, ht.bool)
        self.assertTrue(ht.equal(any_tensor, res))

        any_tensor = ht.zeros((2,))
        x = ht.int32([[0, 0],
                      [0, 0],
                      [0, 1]])
        ht.any(x, axis=0, out=any_tensor)
        res = ht.uint8([[0, 1]])
        self.assertIsInstance(any_tensor, ht.Tensor)
        self.assertEqual(any_tensor.shape, (2,))
        self.assertEqual(any_tensor.dtype, ht.bool)
        self.assertTrue(ht.equal(any_tensor, res))

        any_tensor = ht.zeros(1)
        x = ht.float64([[0, 0, 0],
                        [0, 0, 0]])
        res = ht.zeros(1, dtype=ht.uint8)
        any_tensor = ht.any(x)
        self.assertIsInstance(any_tensor, ht.Tensor)
        self.assertEqual(any_tensor.shape, (1,))
        self.assertEqual(any_tensor.dtype, ht.bool)
        self.assertTrue(ht.equal(any_tensor, res))

    def test_argmax(self):
        torch.manual_seed(1)
        data = ht.random.randn(3, 4, 5)

        # 3D local tensor, major axis
        result = ht.argmax(data, axis=0)
        self.assertIsInstance(result, ht.Tensor)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._Tensor__array.dtype, torch.int64)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.lshape, (1, 4, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._Tensor__array == data._Tensor__array.argmax(0)).all())

        # 3D local tensor, minor axis
        result = ht.argmax(data, axis=-1)
        self.assertIsInstance(result, ht.Tensor)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._Tensor__array.dtype, torch.int64)
        self.assertEqual(result.shape, (3, 4,))
        self.assertEqual(result.lshape, (3, 4, 1,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._Tensor__array == data._Tensor__array.argmax(-1, keepdim=True)).all())

        # 1D split tensor, no axis
        data = ht.arange(-10, 10, split=0)
        result = ht.argmax(data)
        self.assertIsInstance(result, ht.Tensor)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._Tensor__array.dtype, torch.int64)
        self.assertEqual(result.shape, (1,))
        self.assertEqual(result.lshape, (1,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._Tensor__array == torch.tensor([19])))

        # 2D split tensor, along the axis
        torch.manual_seed(1)
        data = ht.array(ht.random.randn(4, 5), split=0)
        result = ht.argmax(data, axis=1)
        self.assertIsInstance(result, ht.Tensor)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._Tensor__array.dtype, torch.int64)
        self.assertEqual(result.shape, (ht.MPI_WORLD.size * 4,))
        self.assertEqual(result.lshape, (4, 1,))
        self.assertEqual(result.split, 0)
        self.assertTrue((result._Tensor__array == torch.tensor([[4], [4], [2], [4]])).all())

        # 2D split tensor, across the axis
        size = ht.MPI_WORLD.size * 2
        data = ht.tril(ht.ones((size, size,), split=0), k=-1)

        result = ht.argmax(data, axis=0)
        self.assertIsInstance(result, ht.Tensor)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._Tensor__array.dtype, torch.int64)
        self.assertEqual(result.shape, (size,))
        self.assertEqual(result.lshape, (1, size,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._Tensor__array != 0).all())

        # 2D split tensor, across the axis, output tensor
        size = ht.MPI_WORLD.size * 2
        data = ht.triu(ht.ones((size, size,), split=0), k=-1)

        output = ht.empty((size,))
        result = ht.argmax(data, axis=0, out=output)

        self.assertIsInstance(result, ht.Tensor)
        self.assertEqual(output.dtype, ht.int64)
        self.assertEqual(output._Tensor__array.dtype, torch.int64)
        self.assertEqual(output.shape, (size,))
        self.assertEqual(output.lshape, (1, size,))
        self.assertEqual(output.split, None)
        self.assertTrue((output._Tensor__array != 0).all())

        # check exceptions
        with self.assertRaises(NotImplementedError):
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
        self.assertIsInstance(result, ht.Tensor)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._Tensor__array.dtype, torch.int64)
        self.assertEqual(result.shape, (1,))
        self.assertEqual(result.lshape, (1,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._Tensor__array == data._Tensor__array.argmin()).all())

        # 3D local tensor, major axis
        result = ht.argmin(data, axis=0)
        self.assertIsInstance(result, ht.Tensor)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._Tensor__array.dtype, torch.int64)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.lshape, (1, 4, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._Tensor__array == data._Tensor__array.argmin(0)).all())

        # 3D local tensor, minor axis
        result = ht.argmin(data, axis=-1)
        self.assertIsInstance(result, ht.Tensor)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._Tensor__array.dtype, torch.int64)
        self.assertEqual(result.shape, (3, 4,))
        self.assertEqual(result.lshape, (3, 4, 1,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._Tensor__array == data._Tensor__array.argmin(-1, keepdim=True)).all())

        # 2D split tensor, along the axis
        torch.manual_seed(1)
        data = ht.array(ht.random.randn(4, 5), split=0)
        result = ht.argmin(data, axis=1)
        self.assertIsInstance(result, ht.Tensor)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._Tensor__array.dtype, torch.int64)
        self.assertEqual(result.shape, (ht.MPI_WORLD.size * 4,))
        self.assertEqual(result.lshape, (4, 1,))
        self.assertEqual(result.split, 0)
        self.assertTrue((result._Tensor__array == torch.tensor([[3], [1], [1], [3]])).all())

        # 2D split tensor, across the axis
        size = ht.MPI_WORLD.size * 2
        data = ht.triu(ht.ones((size, size,), split=0), k=1)

        result = ht.argmin(data, axis=0)
        self.assertIsInstance(result, ht.Tensor)
        self.assertEqual(result.dtype, ht.int64)
        self.assertEqual(result._Tensor__array.dtype, torch.int64)
        self.assertEqual(result.shape, (size,))
        self.assertEqual(result.lshape, (1, size,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._Tensor__array != 0).all())

        # 2D split tensor, across the axis, output tensor
        size = ht.MPI_WORLD.size * 2
        data = ht.triu(ht.ones((size, size,), split=0), k=1)

        output = ht.empty((size,))
        result = ht.argmin(data, axis=0, out=output)

        self.assertIsInstance(result, ht.Tensor)
        self.assertEqual(output.dtype, ht.int64)
        self.assertEqual(output._Tensor__array.dtype, torch.int64)
        self.assertEqual(output.shape, (size,))
        self.assertEqual(output.lshape, (1, size,))
        self.assertEqual(output.split, None)
        self.assertTrue((output._Tensor__array != 0).all())

        # check exceptions
        with self.assertRaises(NotImplementedError):
            data.argmin(axis=(0, 1))
        with self.assertRaises(TypeError):
            data.argmin(axis=1.1)
        with self.assertRaises(TypeError):
            data.argmin(axis='y')
        with self.assertRaises(ValueError):
            ht.argmin(data, axis=-4)
