import torch
import unittest
import os
import heat as ht

if os.environ.get("DEVICE") == "gpu" and torch.cuda.is_available():
    ht.use_device("gpu")
    torch.cuda.set_device(torch.device(ht.get_device().torch_device))
else:
    ht.use_device("cpu")
device = ht.get_device().torch_device
ht_device = None
if os.environ.get("DEVICE") == "lgpu" and torch.cuda.is_available():
    device = ht.gpu.torch_device
    ht_device = ht.gpu
    torch.cuda.set_device(device)


class TestLogical(unittest.TestCase):
    def test_all(self):
        array_len = 9

        # check all over all float elements of 1d tensor locally
        ones_noaxis = ht.ones(array_len, device=ht_device)
        x = (ones_noaxis == 1).all()

        self.assertIsInstance(x, ht.DNDarray)
        self.assertEqual(x.shape, (1,))
        self.assertEqual(x.lshape, (1,))
        self.assertEqual(x.dtype, ht.bool)
        self.assertEqual(x._DNDarray__array.dtype, torch.bool)
        self.assertEqual(x.split, None)
        self.assertEqual(x._DNDarray__array, 1)

        out_noaxis = ht.zeros((1,), device=ht_device)
        ht.all(ones_noaxis, out=out_noaxis)
        self.assertEqual(out_noaxis._DNDarray__array, 1)

        # check all over all float elements of split 1d tensor
        ones_noaxis_split = ht.ones(array_len, split=0, device=ht_device)
        floats_is_one = ones_noaxis_split.all()

        self.assertIsInstance(floats_is_one, ht.DNDarray)
        self.assertEqual(floats_is_one.shape, (1,))
        self.assertEqual(floats_is_one.lshape, (1,))
        self.assertEqual(floats_is_one.dtype, ht.bool)
        self.assertEqual(floats_is_one._DNDarray__array.dtype, torch.bool)
        self.assertEqual(floats_is_one.split, None)
        self.assertEqual(floats_is_one._DNDarray__array, 1)

        out_noaxis = ht.zeros((1,), device=ht_device)
        ht.all(ones_noaxis_split, out=out_noaxis)
        self.assertEqual(out_noaxis._DNDarray__array, 1)

        # check all over all integer elements of 1d tensor locally
        ones_noaxis_int = ht.ones(array_len, device=ht_device).astype(ht.int)
        int_is_one = ones_noaxis_int.all()

        self.assertIsInstance(int_is_one, ht.DNDarray)
        self.assertEqual(int_is_one.shape, (1,))
        self.assertEqual(int_is_one.lshape, (1,))
        self.assertEqual(int_is_one.dtype, ht.bool)
        self.assertEqual(int_is_one._DNDarray__array.dtype, torch.bool)
        self.assertEqual(int_is_one.split, None)
        self.assertEqual(int_is_one._DNDarray__array, 1)

        out_noaxis = ht.zeros((1,), device=ht_device)
        ht.all(ones_noaxis_int, out=out_noaxis)
        self.assertEqual(out_noaxis._DNDarray__array, 1)

        # check all over all integer elements of split 1d tensor
        ones_noaxis_split_int = ht.ones(array_len, split=0, device=ht_device).astype(ht.int)
        split_int_is_one = ones_noaxis_split_int.all()

        self.assertIsInstance(split_int_is_one, ht.DNDarray)
        self.assertEqual(split_int_is_one.shape, (1,))
        self.assertEqual(split_int_is_one.lshape, (1,))
        self.assertEqual(split_int_is_one.dtype, ht.bool)
        self.assertEqual(split_int_is_one._DNDarray__array.dtype, torch.bool)
        self.assertEqual(split_int_is_one.split, None)
        self.assertEqual(split_int_is_one._DNDarray__array, 1)

        out_noaxis = ht.zeros((1,), device=ht_device)
        ht.all(ones_noaxis_split_int, out=out_noaxis)
        self.assertEqual(out_noaxis._DNDarray__array, 1)

        # check all over all float elements of 3d tensor locally
        ones_noaxis_volume = ht.ones((3, 3, 3), device=ht_device)
        volume_is_one = ones_noaxis_volume.all()

        self.assertIsInstance(volume_is_one, ht.DNDarray)
        self.assertEqual(volume_is_one.shape, (1,))
        self.assertEqual(volume_is_one.lshape, (1,))
        self.assertEqual(volume_is_one.dtype, ht.bool)
        self.assertEqual(volume_is_one._DNDarray__array.dtype, torch.bool)
        self.assertEqual(volume_is_one.split, None)
        self.assertEqual(volume_is_one._DNDarray__array, 1)

        out_noaxis = ht.zeros((1,), device=ht_device)
        ht.all(ones_noaxis_volume, out=out_noaxis)
        self.assertEqual(out_noaxis._DNDarray__array, 1)

        # check sequence is not all one
        sequence = ht.arange(array_len, device=ht_device)
        sequence_is_one = sequence.all()

        self.assertIsInstance(sequence_is_one, ht.DNDarray)
        self.assertEqual(sequence_is_one.shape, (1,))
        self.assertEqual(sequence_is_one.lshape, (1,))
        self.assertEqual(sequence_is_one.dtype, ht.bool)
        self.assertEqual(sequence_is_one._DNDarray__array.dtype, torch.bool)
        self.assertEqual(sequence_is_one.split, None)
        self.assertEqual(sequence_is_one._DNDarray__array, 0)

        out_noaxis = ht.zeros((1,), device=ht_device)
        ht.all(sequence, out=out_noaxis)
        self.assertEqual(out_noaxis._DNDarray__array, 0)

        # check all over all float elements of split 3d tensor
        ones_noaxis_split_axis = ht.ones((3, 3, 3), split=0, device=ht_device)
        float_volume_is_one = ones_noaxis_split_axis.all(axis=0)

        self.assertIsInstance(float_volume_is_one, ht.DNDarray)
        self.assertEqual(float_volume_is_one.shape, (3, 3))
        self.assertEqual(float_volume_is_one.all(axis=1).dtype, ht.bool)
        self.assertEqual(float_volume_is_one._DNDarray__array.dtype, torch.bool)
        self.assertEqual(float_volume_is_one.split, None)

        out_noaxis = ht.zeros((3, 3), device=ht_device)
        ht.all(ones_noaxis_split_axis, axis=0, out=out_noaxis)

        # check all over all float elements of split 3d tensor with tuple axis
        ones_noaxis_split_axis = ht.ones((3, 3, 3), split=0, device=ht_device)
        float_volume_is_one = ones_noaxis_split_axis.all(axis=(0, 1))

        self.assertIsInstance(float_volume_is_one, ht.DNDarray)
        self.assertEqual(float_volume_is_one.shape, (3,))
        self.assertEqual(float_volume_is_one.all(axis=0).dtype, ht.bool)
        self.assertEqual(float_volume_is_one._DNDarray__array.dtype, torch.bool)
        self.assertEqual(float_volume_is_one.split, None)

        # check all over all float elements of split 5d tensor with negative axis
        ones_noaxis_split_axis_neg = ht.zeros((1, 2, 3, 4, 5), split=1, device=ht_device)
        float_5d_is_one = ones_noaxis_split_axis_neg.all(axis=-2)

        self.assertIsInstance(float_5d_is_one, ht.DNDarray)
        self.assertEqual(float_5d_is_one.shape, (1, 2, 3, 5))
        self.assertEqual(float_5d_is_one.dtype, ht.bool)
        self.assertEqual(float_5d_is_one._DNDarray__array.dtype, torch.bool)
        self.assertEqual(float_5d_is_one.split, 1)

        out_noaxis = ht.zeros((1, 2, 3, 5), device=ht_device)
        ht.all(ones_noaxis_split_axis_neg, axis=-2, out=out_noaxis)

        # exceptions
        with self.assertRaises(ValueError):
            ht.ones(array_len, device=ht_device).all(axis=1)
        with self.assertRaises(ValueError):
            ht.ones(array_len, device=ht_device).all(axis=-2)
        with self.assertRaises(ValueError):
            ht.ones((4, 4), device=ht_device).all(axis=0, out=out_noaxis)
        with self.assertRaises(TypeError):
            ht.ones(array_len, device=ht_device).all(axis="bad_axis_type")

    def test_allclose(self):
        a = ht.float32([[2, 2], [2, 2]], device=ht_device)
        b = ht.float32([[2.00005, 2.00005], [2.00005, 2.00005]], device=ht_device)
        c = ht.zeros((4, 6), split=0, device=ht_device)
        d = ht.zeros((4, 6), split=1, device=ht_device)
        e = ht.zeros((4, 6), device=ht_device)

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
        x = ht.float32([[2.7, 0, 0], [0, 0, 0], [0, 0.3, 0]], device=ht_device)
        any_tensor = x.any(axis=1)
        res = ht.uint8([1, 0, 1], device=ht_device)
        self.assertIsInstance(any_tensor, ht.DNDarray)
        self.assertEqual(any_tensor.shape, (3,))
        self.assertEqual(any_tensor.dtype, ht.bool)
        self.assertTrue(ht.equal(any_tensor, res))

        # integer values, major axis, output tensor
        any_tensor = ht.zeros((2,), device=ht_device)
        x = ht.int32([[0, 0], [0, 0], [0, 1]], device=ht_device)
        ht.any(x, axis=0, out=any_tensor)
        res = ht.uint8([0, 1], device=ht_device)
        self.assertIsInstance(any_tensor, ht.DNDarray)
        self.assertEqual(any_tensor.shape, (2,))
        self.assertEqual(any_tensor.dtype, ht.bool)
        self.assertTrue(ht.equal(any_tensor, res))

        # float values, no axis
        x = ht.float64([[0, 0, 0], [0, 0, 0]], device=ht_device)
        res = ht.zeros(1, dtype=ht.uint8, device=ht_device)
        any_tensor = ht.any(x)
        self.assertIsInstance(any_tensor, ht.DNDarray)
        self.assertEqual(any_tensor.shape, (1,))
        self.assertEqual(any_tensor.dtype, ht.bool)
        self.assertTrue(ht.equal(any_tensor, res))

        # split tensor, along axis
        x = ht.arange(10, split=0, device=ht_device)
        any_tensor = ht.any(x, axis=0)
        res = ht.uint8([1], device=ht_device)
        self.assertIsInstance(any_tensor, ht.DNDarray)
        self.assertEqual(any_tensor.shape, (1,))
        self.assertEqual(any_tensor.dtype, ht.bool)
        self.assertTrue(ht.equal(any_tensor, res))

    def test_logical_and(self):
        first_tensor = ht.array([[True, True], [False, False]])
        second_tensor = ht.array([[True, False], [True, False]])
        result_tensor = ht.array([[True, False], [False, False]])
        int_tensor = ht.array([[-1, 0], [2, 1]])
        float_tensor = ht.array([[-1.4, 0.2], [2.5, 1.3]])

        self.assertTrue(ht.equal(ht.logical_and(first_tensor, second_tensor), result_tensor))
        self.assertTrue(
            ht.equal(
                ht.logical_and(int_tensor, int_tensor), ht.array([[True, False], [True, True]])
            )
        )
        self.assertTrue(
            ht.equal(
                ht.logical_and(float_tensor.copy().resplit_(0), float_tensor),
                ht.array([[True, True], [True, True]]),
            )
        )

    def test_logical_not(self):
        first_tensor = ht.array([[True, True], [False, False]])
        second_tensor = ht.array([[True, False], [True, False]])
        int_tensor = ht.array([[-1, 0], [2, 1]])
        float_tensor = ht.array([[-1.4, 0.2], [2.5, 1.3]])

        self.assertTrue(
            ht.equal(ht.logical_not(first_tensor), ht.array([[False, False], [True, True]]))
        )
        self.assertTrue(
            ht.equal(ht.logical_not(second_tensor), ht.array([[False, True], [False, True]]))
        )
        self.assertTrue(
            ht.equal(ht.logical_not(int_tensor), ht.array([[False, True], [False, False]]))
        )
        self.assertTrue(
            ht.equal(
                ht.logical_not(float_tensor.copy().resplit_(0)),
                ht.array([[False, False], [False, False]]),
            )
        )

    def test_logical_or(self):
        first_tensor = ht.array([[True, True], [False, False]])
        second_tensor = ht.array([[True, False], [True, False]])
        result_tensor = ht.array([[True, True], [True, False]])
        int_tensor = ht.array([[-1, 0], [2, 1]])
        float_tensor = ht.array([[-1.4, 0.2], [2.5, 1.3]])

        self.assertTrue(ht.equal(ht.logical_or(first_tensor, second_tensor), result_tensor))
        self.assertTrue(
            ht.equal(ht.logical_or(int_tensor, int_tensor), ht.array([[True, False], [True, True]]))
        )
        self.assertTrue(
            ht.equal(
                ht.logical_or(float_tensor.copy().resplit_(0), float_tensor),
                ht.array([[True, True], [True, True]]),
            )
        )

    def test_logical_xor(self):
        first_tensor = ht.array([[True, True], [False, False]])
        second_tensor = ht.array([[True, False], [True, False]])
        result_tensor = ht.array([[False, True], [True, False]])

        self.assertTrue(ht.equal(ht.logical_xor(first_tensor, second_tensor), result_tensor))
        self.assertTrue(
            ht.equal(
                ht.logical_xor(first_tensor.copy().resplit_(0), first_tensor),
                ht.array([[False, False], [False, False]]),
            )
        )
