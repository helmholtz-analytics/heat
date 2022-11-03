import torch

import heat as ht
from .test_suites.basic_test import TestCase


class TestLogical(TestCase):
    def test_all(self):
        array_len = 9

        # check all over all float elements of 1d tensor locally
        ones_noaxis = ht.ones(array_len)
        x = (ones_noaxis == 1).all()

        self.assertIsInstance(x, ht.DNDarray)
        self.assertEqual(x.shape, (1,))
        self.assertEqual(x.lshape, (1,))
        self.assertEqual(x.dtype, ht.bool)
        self.assertEqual(x.larray.dtype, torch.bool)
        self.assertEqual(x.split, None)
        self.assertEqual(x.larray, 1)

        out_noaxis = ht.zeros((1,))
        ht.all(ones_noaxis, out=out_noaxis)
        self.assertEqual(out_noaxis.larray, 1)

        # check all over all float elements of split 1d tensor
        ones_noaxis_split = ht.ones(array_len, split=0)
        floats_is_one = ones_noaxis_split.all()

        self.assertIsInstance(floats_is_one, ht.DNDarray)
        self.assertEqual(floats_is_one.shape, (1,))
        self.assertEqual(floats_is_one.lshape, (1,))
        self.assertEqual(floats_is_one.dtype, ht.bool)
        self.assertEqual(floats_is_one.larray.dtype, torch.bool)
        self.assertEqual(floats_is_one.split, None)
        self.assertEqual(floats_is_one.larray, 1)

        out_noaxis = ht.zeros((1,))
        ht.all(ones_noaxis_split, out=out_noaxis)
        self.assertEqual(out_noaxis.larray, 1)

        # check all over all integer elements of 1d tensor locally
        ones_noaxis_int = ht.ones(array_len).astype(ht.int)
        int_is_one = ones_noaxis_int.all()

        self.assertIsInstance(int_is_one, ht.DNDarray)
        self.assertEqual(int_is_one.shape, (1,))
        self.assertEqual(int_is_one.lshape, (1,))
        self.assertEqual(int_is_one.dtype, ht.bool)
        self.assertEqual(int_is_one.larray.dtype, torch.bool)
        self.assertEqual(int_is_one.split, None)
        self.assertEqual(int_is_one.larray, 1)

        out_noaxis = ht.zeros((1,))
        ht.all(ones_noaxis_int, out=out_noaxis)
        self.assertEqual(out_noaxis.larray, 1)

        # check all over all integer elements of split 1d tensor
        ones_noaxis_split_int = ht.ones(array_len, split=0).astype(ht.int)
        split_int_is_one = ones_noaxis_split_int.all()

        self.assertIsInstance(split_int_is_one, ht.DNDarray)
        self.assertEqual(split_int_is_one.shape, (1,))
        self.assertEqual(split_int_is_one.lshape, (1,))
        self.assertEqual(split_int_is_one.dtype, ht.bool)
        self.assertEqual(split_int_is_one.larray.dtype, torch.bool)
        self.assertEqual(split_int_is_one.split, None)
        self.assertEqual(split_int_is_one.larray, 1)

        out_noaxis = ht.zeros((1,))
        ht.all(ones_noaxis_split_int, out=out_noaxis)
        self.assertEqual(out_noaxis.larray, 1)

        # check all over all float elements of 3d tensor locally
        ones_noaxis_volume = ht.ones((3, 3, 3))
        volume_is_one = ones_noaxis_volume.all()

        self.assertIsInstance(volume_is_one, ht.DNDarray)
        self.assertEqual(volume_is_one.shape, (1,))
        self.assertEqual(volume_is_one.lshape, (1,))
        self.assertEqual(volume_is_one.dtype, ht.bool)
        self.assertEqual(volume_is_one.larray.dtype, torch.bool)
        self.assertEqual(volume_is_one.split, None)
        self.assertEqual(volume_is_one.larray, 1)

        out_noaxis = ht.zeros((1,))
        ht.all(ones_noaxis_volume, out=out_noaxis)
        self.assertEqual(out_noaxis.larray, 1)

        # check sequence is not all one
        sequence = ht.arange(array_len)
        sequence_is_one = sequence.all()

        self.assertIsInstance(sequence_is_one, ht.DNDarray)
        self.assertEqual(sequence_is_one.shape, (1,))
        self.assertEqual(sequence_is_one.lshape, (1,))
        self.assertEqual(sequence_is_one.dtype, ht.bool)
        self.assertEqual(sequence_is_one.larray.dtype, torch.bool)
        self.assertEqual(sequence_is_one.split, None)
        self.assertEqual(sequence_is_one.larray, 0)

        out_noaxis = ht.zeros((1,))
        ht.all(sequence, out=out_noaxis)
        self.assertEqual(out_noaxis.larray, 0)

        # check all over all float elements of split 3d tensor
        ones_noaxis_split_axis = ht.ones((3, 3, 3), split=0)
        float_volume_is_one = ones_noaxis_split_axis.all(axis=0)

        self.assertIsInstance(float_volume_is_one, ht.DNDarray)
        self.assertEqual(float_volume_is_one.shape, (3, 3))
        self.assertEqual(float_volume_is_one.all(axis=1).dtype, ht.bool)
        self.assertEqual(float_volume_is_one.larray.dtype, torch.bool)
        self.assertEqual(float_volume_is_one.split, None)

        out_noaxis = ht.zeros((3, 3))
        ht.all(ones_noaxis_split_axis, axis=0, out=out_noaxis)

        # check all over all float elements of split 3d tensor with tuple axis
        ones_noaxis_split_axis = ht.ones((3, 3, 3), split=0)
        float_volume_is_one = ones_noaxis_split_axis.all(axis=(0, 1))

        self.assertIsInstance(float_volume_is_one, ht.DNDarray)
        self.assertEqual(float_volume_is_one.shape, (3,))
        self.assertEqual(float_volume_is_one.all(axis=0).dtype, ht.bool)
        self.assertEqual(float_volume_is_one.larray.dtype, torch.bool)
        self.assertEqual(float_volume_is_one.split, None)

        # check all over all float elements of split 5d tensor with negative axis
        ones_noaxis_split_axis_neg = ht.zeros((1, 2, 3, 4, 5), split=1)
        float_5d_is_one = ones_noaxis_split_axis_neg.all(axis=-2)

        self.assertIsInstance(float_5d_is_one, ht.DNDarray)
        self.assertEqual(float_5d_is_one.shape, (1, 2, 3, 5))
        self.assertEqual(float_5d_is_one.dtype, ht.bool)
        self.assertEqual(float_5d_is_one.larray.dtype, torch.bool)
        self.assertEqual(float_5d_is_one.split, 1)

        out_noaxis = ht.zeros((1, 2, 3, 5), split=1)
        ht.all(ones_noaxis_split_axis_neg, axis=-2, out=out_noaxis)

        # test keepdim
        ones_2d = ht.ones((1, 1))
        self.assertEqual(ones_2d.all(keepdim=True).shape, ones_2d.shape)

        ones_2d_split = ht.ones((2, 2), split=0)
        keepdim_is_one = ones_2d_split.all(keepdim=True)
        self.assertEqual(keepdim_is_one.shape, (1, 1))
        self.assertEqual(keepdim_is_one.split, None)
        keepdim_is_one = ones_2d_split.all(axis=0, keepdim=True)
        self.assertEqual(keepdim_is_one.shape, (1, 2))
        self.assertEqual(keepdim_is_one.split, None)
        keepdim_is_one = ones_2d_split.all(axis=1, keepdim=True)
        self.assertEqual(keepdim_is_one.shape, (2, 1))
        self.assertEqual(keepdim_is_one.split, 0)

        ones_2d_split = ht.ones((2, 2), split=1)
        keepdim_is_one = ones_2d_split.all(keepdim=True)
        self.assertEqual(keepdim_is_one.shape, (1, 1))
        self.assertEqual(keepdim_is_one.split, None)
        keepdim_is_one = ones_2d_split.all(axis=0, keepdim=True)
        self.assertEqual(keepdim_is_one.shape, (1, 2))
        self.assertEqual(keepdim_is_one.split, 1)
        keepdim_is_one = ones_2d_split.all(axis=1, keepdim=True)
        self.assertEqual(keepdim_is_one.shape, (2, 1))
        self.assertEqual(keepdim_is_one.split, None)

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
        any_tensor = ht.zeros((2,), dtype=ht.bool)
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

        # test keepdim
        ones_2d = ht.ones((1, 1))
        self.assertEqual(ones_2d.any(keepdim=True).shape, ones_2d.shape)

        ones_2d_split = ht.ones((2, 2), split=0)
        keepdim_any = ones_2d_split.any(keepdim=True)
        self.assertEqual(keepdim_any.shape, (1, 1))
        self.assertEqual(keepdim_any.split, None)
        keepdim_any = ones_2d_split.any(axis=0, keepdim=True)
        self.assertEqual(keepdim_any.shape, (1, 2))
        self.assertEqual(keepdim_any.split, None)
        keepdim_any = ones_2d_split.any(axis=1, keepdim=True)
        self.assertEqual(keepdim_any.shape, (2, 1))
        self.assertEqual(keepdim_any.split, 0)

        ones_2d_split = ht.ones((2, 2), split=1)
        keepdim_any = ones_2d_split.any(keepdim=True)
        self.assertEqual(keepdim_any.shape, (1, 1))
        self.assertEqual(keepdim_any.split, None)
        keepdim_any = ones_2d_split.any(axis=0, keepdim=True)
        self.assertEqual(keepdim_any.shape, (1, 2))
        self.assertEqual(keepdim_any.split, 1)
        keepdim_any = ones_2d_split.any(axis=1, keepdim=True)
        self.assertEqual(keepdim_any.shape, (2, 1))
        self.assertEqual(keepdim_any.split, None)

    def test_isclose(self):
        size = ht.communication.MPI_WORLD.size
        a = ht.float32([[2, 2], [2, 2]])
        b = ht.float32([[2.00005, 2.00005], [2.00005, 2.00005]])
        c = ht.zeros((4 * size, 6), split=0)
        d = ht.zeros((4 * size, 6), split=1)
        e = ht.zeros((4 * size, 6))

        self.assertIsInstance(ht.isclose(a, b), ht.DNDarray)
        self.assertTrue(ht.isclose(a, b).shape == (2, 2))
        self.assertFalse(ht.isclose(a, b)[0][0].item())
        self.assertTrue(ht.isclose(a, b, atol=1e-04)[0][1].item())
        self.assertTrue(ht.isclose(a, b, rtol=1e-04)[1][0].item())
        self.assertTrue(ht.isclose(a, 2)[0][1].item())
        self.assertTrue(ht.isclose(a, 2.0)[0][0].item())
        self.assertTrue(ht.isclose(2, a)[1][1].item())
        self.assertTrue(ht.isclose(c, d).shape == (4 * size, 6))
        self.assertTrue(ht.isclose(c, e)[0][0].item())
        self.assertTrue(e.isclose(c)[-1][-1].item())

        # test scalar input
        self.assertIsInstance(ht.isclose(2.0, 2.00005), bool)

        with self.assertRaises(TypeError):
            ht.isclose(a, (2, 2, 2, 2))
        with self.assertRaises(TypeError):
            ht.isclose(a, "?")
        with self.assertRaises(TypeError):
            ht.isclose("?", a)

    def test_isfinite(self):
        a = ht.array([1, ht.inf, -ht.inf, ht.nan])
        s = ht.array([True, False, False, False])
        r = ht.isfinite(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.array([1, ht.inf, -ht.inf, ht.nan], split=0)
        s = ht.array([True, False, False, False], split=0)
        r = ht.isfinite(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.ones((6, 6), dtype=ht.bool, split=0)
        s = ht.ones((6, 6), dtype=ht.bool, split=0)
        r = ht.isfinite(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.ones((5, 5), dtype=ht.int, split=1)
        s = ht.ones((5, 5), dtype=ht.bool, split=1)
        r = ht.isfinite(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

    def test_isinf(self):
        a = ht.array([1, ht.inf, -ht.inf, ht.nan])
        s = ht.array([False, True, True, False])
        r = ht.isinf(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.array([1, ht.inf, -ht.inf, ht.nan], split=0)
        s = ht.array([False, True, True, False], split=0)
        r = ht.isinf(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.ones((6, 6), dtype=ht.bool, split=0)
        s = ht.zeros((6, 6), dtype=ht.bool, split=0)
        r = ht.isinf(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.ones((5, 5), dtype=ht.int, split=1)
        s = ht.zeros((5, 5), dtype=ht.bool, split=1)
        r = ht.isinf(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

    def test_isnan(self):
        a = ht.array([1, ht.inf, -ht.inf, ht.nan])
        s = ht.array([False, False, False, True])
        r = ht.isnan(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.array([1, ht.inf, -ht.inf, ht.nan], split=0)
        s = ht.array([False, False, False, True], split=0)
        r = ht.isnan(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.ones((6, 6), dtype=ht.bool, split=0)
        s = ht.zeros((6, 6), dtype=ht.bool, split=0)
        r = ht.isnan(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.ones((5, 5), dtype=ht.int, split=1)
        s = ht.zeros((5, 5), dtype=ht.bool, split=1)
        r = ht.isnan(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

    def test_isneginf(self):
        a = ht.array([1, ht.inf, -ht.inf, ht.nan])
        s = ht.array([False, False, True, False])
        r = ht.isneginf(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.array([1, ht.inf, -ht.inf, ht.nan], split=0)
        out = ht.empty(4, dtype=ht.bool, split=0)
        s = ht.array([False, False, True, False], split=0)
        ht.isneginf(a, out)
        self.assertEqual(out.shape, s.shape)
        self.assertEqual(out.dtype, s.dtype)
        self.assertEqual(out.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.ones((6, 6), dtype=ht.bool, split=0)
        s = ht.zeros((6, 6), dtype=ht.bool, split=0)
        r = ht.isneginf(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.ones((5, 5), dtype=ht.int, split=1)
        s = ht.zeros((5, 5), dtype=ht.bool, split=1)
        r = ht.isneginf(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

    def test_isposinf(self):
        a = ht.array([1, ht.inf, -ht.inf, ht.nan])
        s = ht.array([False, True, False, False])
        r = ht.isposinf(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.array([1, ht.inf, -ht.inf, ht.nan], split=0)
        out = ht.empty(4, dtype=ht.bool, split=0)
        s = ht.array([False, True, False, False], split=0)
        ht.isposinf(a, out)
        self.assertEqual(out.shape, s.shape)
        self.assertEqual(out.dtype, s.dtype)
        self.assertEqual(out.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.ones((6, 6), dtype=ht.bool, split=0)
        s = ht.zeros((6, 6), dtype=ht.bool, split=0)
        r = ht.isposinf(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.ones((5, 5), dtype=ht.int, split=1)
        s = ht.zeros((5, 5), dtype=ht.bool, split=1)
        r = ht.isposinf(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

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

    def test_signbit(self):
        a = ht.array([2, -1.3, 0, -5], split=0)

        sb = ht.signbit(a)
        cmp = ht.array([False, True, False, True])

        self.assertEqual(sb.dtype, ht.bool)
        self.assertEqual(sb.split, 0)
        self.assertEqual(sb.device, a.device)
        self.assertTrue(ht.equal(sb, cmp))
