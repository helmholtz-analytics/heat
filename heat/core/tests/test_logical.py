import torch
import unittest
import os
import heat as ht

if os.environ.get("DEVICE") == "gpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ht.use_device("gpu" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
    ht.use_device("cpu")


class TestLogical(unittest.TestCase):
    def test_all(self):
        array_len = 9

        # check all over all float elements of 1d tensor locally
        ones_noaxis = ht.ones(array_len)
        x = (ones_noaxis == 1).all()

        self.assertIsInstance(x, ht.DNDarray)
        self.assertEqual(x.shape, (1,))
        self.assertEqual(x.lshape, (1,))
        self.assertEqual(x.dtype, ht.bool)
        self.assertEqual(x._DNDarray__array.dtype, torch.bool)
        self.assertEqual(x.split, None)
        self.assertEqual(x._DNDarray__array, 1)

        out_noaxis = ht.zeros((1,))
        ht.all(ones_noaxis, out=out_noaxis)
        self.assertEqual(out_noaxis._DNDarray__array, 1)

        # check all over all float elements of split 1d tensor
        ones_noaxis_split = ht.ones(array_len, split=0)
        floats_is_one = ones_noaxis_split.all()

        self.assertIsInstance(floats_is_one, ht.DNDarray)
        self.assertEqual(floats_is_one.shape, (1,))
        self.assertEqual(floats_is_one.lshape, (1,))
        self.assertEqual(floats_is_one.dtype, ht.bool)
        self.assertEqual(floats_is_one._DNDarray__array.dtype, torch.bool)
        self.assertEqual(floats_is_one.split, None)
        self.assertEqual(floats_is_one._DNDarray__array, 1)

        out_noaxis = ht.zeros((1,))
        ht.all(ones_noaxis_split, out=out_noaxis)
        self.assertEqual(out_noaxis._DNDarray__array, 1)

        # check all over all integer elements of 1d tensor locally
        ones_noaxis_int = ht.ones(array_len).astype(ht.int)
        int_is_one = ones_noaxis_int.all()

        self.assertIsInstance(int_is_one, ht.DNDarray)
        self.assertEqual(int_is_one.shape, (1,))
        self.assertEqual(int_is_one.lshape, (1,))
        self.assertEqual(int_is_one.dtype, ht.bool)
        self.assertEqual(int_is_one._DNDarray__array.dtype, torch.bool)
        self.assertEqual(int_is_one.split, None)
        self.assertEqual(int_is_one._DNDarray__array, 1)

        out_noaxis = ht.zeros((1,))
        ht.all(ones_noaxis_int, out=out_noaxis)
        self.assertEqual(out_noaxis._DNDarray__array, 1)

        # check all over all integer elements of split 1d tensor
        ones_noaxis_split_int = ht.ones(array_len, split=0).astype(ht.int)
        split_int_is_one = ones_noaxis_split_int.all()

        self.assertIsInstance(split_int_is_one, ht.DNDarray)
        self.assertEqual(split_int_is_one.shape, (1,))
        self.assertEqual(split_int_is_one.lshape, (1,))
        self.assertEqual(split_int_is_one.dtype, ht.bool)
        self.assertEqual(split_int_is_one._DNDarray__array.dtype, torch.bool)
        self.assertEqual(split_int_is_one.split, None)
        self.assertEqual(split_int_is_one._DNDarray__array, 1)

        out_noaxis = ht.zeros((1,))
        ht.all(ones_noaxis_split_int, out=out_noaxis)
        self.assertEqual(out_noaxis._DNDarray__array, 1)

        # check all over all float elements of 3d tensor locally
        ones_noaxis_volume = ht.ones((3, 3, 3))
        volume_is_one = ones_noaxis_volume.all()

        self.assertIsInstance(volume_is_one, ht.DNDarray)
        self.assertEqual(volume_is_one.shape, (1,))
        self.assertEqual(volume_is_one.lshape, (1,))
        self.assertEqual(volume_is_one.dtype, ht.bool)
        self.assertEqual(volume_is_one._DNDarray__array.dtype, torch.bool)
        self.assertEqual(volume_is_one.split, None)
        self.assertEqual(volume_is_one._DNDarray__array, 1)

        out_noaxis = ht.zeros((1,))
        ht.all(ones_noaxis_volume, out=out_noaxis)
        self.assertEqual(out_noaxis._DNDarray__array, 1)

        # check sequence is not all one
        sequence = ht.arange(array_len)
        sequence_is_one = sequence.all()

        self.assertIsInstance(sequence_is_one, ht.DNDarray)
        self.assertEqual(sequence_is_one.shape, (1,))
        self.assertEqual(sequence_is_one.lshape, (1,))
        self.assertEqual(sequence_is_one.dtype, ht.bool)
        self.assertEqual(sequence_is_one._DNDarray__array.dtype, torch.bool)
        self.assertEqual(sequence_is_one.split, None)
        self.assertEqual(sequence_is_one._DNDarray__array, 0)

        out_noaxis = ht.zeros((1,))
        ht.all(sequence, out=out_noaxis)
        self.assertEqual(out_noaxis._DNDarray__array, 0)

        # check all over all float elements of split 3d tensor
        ones_noaxis_split_axis = ht.ones((3, 3, 3), split=0)
        float_volume_is_one = ones_noaxis_split_axis.all(axis=0)

        self.assertIsInstance(float_volume_is_one, ht.DNDarray)
        self.assertEqual(float_volume_is_one.shape, (3, 3))
        self.assertEqual(float_volume_is_one.all(axis=1).dtype, ht.bool)
        self.assertEqual(float_volume_is_one._DNDarray__array.dtype, torch.bool)
        self.assertEqual(float_volume_is_one.split, None)

        out_noaxis = ht.zeros((3, 3))
        ht.all(ones_noaxis_split_axis, axis=0, out=out_noaxis)

        # check all over all float elements of split 3d tensor with tuple axis
        ones_noaxis_split_axis = ht.ones((3, 3, 3), split=0)
        float_volume_is_one = ones_noaxis_split_axis.all(axis=(0, 1))

        self.assertIsInstance(float_volume_is_one, ht.DNDarray)
        self.assertEqual(float_volume_is_one.shape, (3,))
        self.assertEqual(float_volume_is_one.all(axis=0).dtype, ht.bool)
        self.assertEqual(float_volume_is_one._DNDarray__array.dtype, torch.bool)
        self.assertEqual(float_volume_is_one.split, None)

        # check all over all float elements of split 5d tensor with negative axis
        ones_noaxis_split_axis_neg = ht.zeros((1, 2, 3, 4, 5), split=1)
        float_5d_is_one = ones_noaxis_split_axis_neg.all(axis=-2)

        self.assertIsInstance(float_5d_is_one, ht.DNDarray)
        self.assertEqual(float_5d_is_one.shape, (1, 2, 3, 5))
        self.assertEqual(float_5d_is_one.dtype, ht.bool)
        self.assertEqual(float_5d_is_one._DNDarray__array.dtype, torch.bool)
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
            ht.ones(array_len).all(axis="bad_axis_type")

    def test_allclose(self):
        a = ht.float32([[2, 2], [2, 2]])
        b = ht.float32([[2.00005, 2.00005], [2.00005, 2.00005]])
        c = ht.zeros((4, 6), split=0)
        d = ht.zeros((4, 6), split=1)
        e = ht.zeros((4, 6))

        self.assertFalse(ht.allclose(a, b))
        self.assertTrue(ht.allclose(a, b, atol=1e-04))
        self.assertTrue(ht.allclose(a, b, rtol=1e-04))
        self.assertTrue(ht.allclose(a, 2))
        self.assertTrue(ht.allclose(a, 2.0))
        self.assertTrue(ht.allclose(2, a))
        self.assertTrue(ht.allclose(c, d))
        self.assertTrue(ht.allclose(c, e))
        self.assertTrue(e.allclose(c))

        with self.assertRaises(TypeError):
            ht.allclose(a, (2, 2, 2, 2))
        with self.assertRaises(TypeError):
            ht.allclose(a, "?")
        with self.assertRaises(TypeError):
            ht.allclose("?", a)

    def test_any(self):
        # float values, minor axis
        x = ht.float32([[2.7, 0, 0], [0, 0, 0], [0, 0.3, 0]])
        any_tensor = x.any(axis=1)
        res = ht.uint8([1, 0, 1])
        self.assertIsInstance(any_tensor, ht.DNDarray)
        self.assertEqual(any_tensor.shape, (3,))
        self.assertEqual(any_tensor.dtype, ht.bool)
        self.assertTrue(ht.equal(any_tensor, res))

        # integer values, major axis, output tensor
        any_tensor = ht.zeros((2,))
        x = ht.int32([[0, 0], [0, 0], [0, 1]])
        ht.any(x, axis=0, out=any_tensor)
        res = ht.uint8([0, 1])
        self.assertIsInstance(any_tensor, ht.DNDarray)
        self.assertEqual(any_tensor.shape, (2,))
        self.assertEqual(any_tensor.dtype, ht.bool)
        self.assertTrue(ht.equal(any_tensor, res))

        # float values, no axis
        x = ht.float64([[0, 0, 0], [0, 0, 0]])
        res = ht.zeros(1, dtype=ht.uint8)
        any_tensor = ht.any(x)
        self.assertIsInstance(any_tensor, ht.DNDarray)
        self.assertEqual(any_tensor.shape, (1,))
        self.assertEqual(any_tensor.dtype, ht.bool)
        self.assertTrue(ht.equal(any_tensor, res))

        # split tensor, along axis
        x = ht.arange(10, split=0)
        any_tensor = ht.any(x, axis=0)
        res = ht.uint8([1])
        self.assertIsInstance(any_tensor, ht.DNDarray)
        self.assertEqual(any_tensor.shape, (1,))
        self.assertEqual(any_tensor.dtype, ht.bool)
        self.assertTrue(ht.equal(any_tensor, res))
