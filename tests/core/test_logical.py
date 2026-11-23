import numpy as np
import torch

import heat as ht
from heat.testing.basic_test import TestCase


class TestLogical(TestCase):
    def test_all(self):
        array_len = 9

        # check all over all float elements of 1d tensor locally
        ones_noaxis = ht.ones(array_len)
        x = (ones_noaxis == 1).all()

        self.assertIsInstance(x, ht.DNDarray)
        self.assertEqual(x.shape, ())
        self.assertEqual(x.lshape, ())
        self.assertEqual(x.dtype, ht.bool)
        self.assertEqual(x.larray.dtype, torch.bool)
        self.assertEqual(x.split, None)
        self.assertEqual(x.larray, 1)

        out_noaxis = ht.zeros(())
        ht.all(ones_noaxis, out=out_noaxis)
        self.assertEqual(out_noaxis.larray, 1)

        # check all over all float elements of split 1d tensor
        ones_noaxis_split = ht.ones(array_len, split=0)
        floats_is_one = ones_noaxis_split.all()

        self.assertIsInstance(floats_is_one, ht.DNDarray)
        self.assertEqual(floats_is_one.shape, ())
        self.assertEqual(floats_is_one.lshape, ())
        self.assertEqual(floats_is_one.dtype, ht.bool)
        self.assertEqual(floats_is_one.larray.dtype, torch.bool)
        self.assertEqual(floats_is_one.split, None)
        self.assertEqual(floats_is_one.larray, 1)

        out_noaxis = ht.zeros(())
        ht.all(ones_noaxis_split, out=out_noaxis)
        self.assertEqual(out_noaxis.larray, 1)

        # check all over all integer elements of 1d tensor locally
        ones_noaxis_int = ht.ones(array_len).astype(ht.int)
        int_is_one = ones_noaxis_int.all()

        self.assertIsInstance(int_is_one, ht.DNDarray)
        self.assertEqual(int_is_one.shape, ())
        self.assertEqual(int_is_one.lshape, ())
        self.assertEqual(int_is_one.dtype, ht.bool)
        self.assertEqual(int_is_one.larray.dtype, torch.bool)
        self.assertEqual(int_is_one.split, None)
        self.assertEqual(int_is_one.larray, 1)

        out_noaxis = ht.zeros(())
        ht.all(ones_noaxis_int, out=out_noaxis)
        self.assertEqual(out_noaxis.larray, 1)

        # check all over all integer elements of split 1d tensor
        ones_noaxis_split_int = ht.ones(array_len, split=0).astype(ht.int)
        split_int_is_one = ones_noaxis_split_int.all()

        self.assertIsInstance(split_int_is_one, ht.DNDarray)
        self.assertEqual(split_int_is_one.shape, ())
        self.assertEqual(split_int_is_one.lshape, ())
        self.assertEqual(split_int_is_one.dtype, ht.bool)
        self.assertEqual(split_int_is_one.larray.dtype, torch.bool)
        self.assertEqual(split_int_is_one.split, None)
        self.assertEqual(split_int_is_one.larray, 1)

        out_noaxis = ht.zeros(())
        ht.all(ones_noaxis_split_int, out=out_noaxis)
        self.assertEqual(out_noaxis.larray, 1)

        # check all over all float elements of 3d tensor locally
        ones_noaxis_volume = ht.ones((3, 3, 3))
        volume_is_one = ones_noaxis_volume.all()

        self.assertIsInstance(volume_is_one, ht.DNDarray)
        self.assertEqual(volume_is_one.shape, ())
        self.assertEqual(volume_is_one.lshape, ())
        self.assertEqual(volume_is_one.dtype, ht.bool)
        self.assertEqual(volume_is_one.larray.dtype, torch.bool)
        self.assertEqual(volume_is_one.split, None)
        self.assertEqual(volume_is_one.larray, 1)

        out_noaxis = ht.zeros(())
        ht.all(ones_noaxis_volume, out=out_noaxis)
        self.assertEqual(out_noaxis.larray, 1)

        # check sequence is not all one
        sequence = ht.arange(array_len)
        sequence_is_one = sequence.all()

        self.assertIsInstance(sequence_is_one, ht.DNDarray)
        self.assertEqual(sequence_is_one.shape, ())
        self.assertEqual(sequence_is_one.lshape, ())
        self.assertEqual(sequence_is_one.dtype, ht.bool)
        self.assertEqual(sequence_is_one.larray.dtype, torch.bool)
        self.assertEqual(sequence_is_one.split, None)
        self.assertEqual(sequence_is_one.larray, 0)

        out_noaxis = ht.zeros(())
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

        # test keepdims
        ones_2d = ht.ones((1, 1))
        self.assertEqual(ones_2d.all(keepdims=True).shape, ones_2d.shape)

        ones_2d_split = ht.ones((2, 2), split=0)
        keepdims_is_one = ones_2d_split.all(keepdims=True)
        self.assertEqual(keepdims_is_one.shape, (1, 1))
        self.assertEqual(keepdims_is_one.split, None)
        keepdims_is_one = ones_2d_split.all(axis=0, keepdims=True)
        self.assertEqual(keepdims_is_one.shape, (1, 2))
        self.assertEqual(keepdims_is_one.split, None)
        keepdims_is_one = ones_2d_split.all(axis=1, keepdims=True)
        self.assertEqual(keepdims_is_one.shape, (2, 1))
        self.assertEqual(keepdims_is_one.split, 0)

        ones_2d_split = ht.ones((2, 2), split=1)
        keepdims_is_one = ones_2d_split.all(keepdims=True)
        self.assertEqual(keepdims_is_one.shape, (1, 1))
        self.assertEqual(keepdims_is_one.split, None)
        keepdims_is_one = ones_2d_split.all(axis=0, keepdims=True)
        self.assertEqual(keepdims_is_one.shape, (1, 2))
        self.assertEqual(keepdims_is_one.split, 1)
        keepdims_is_one = ones_2d_split.all(axis=1, keepdims=True)
        self.assertEqual(keepdims_is_one.shape, (2, 1))
        self.assertEqual(keepdims_is_one.split, None)

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

        if not self.is_mps:
            f = ht.float64([[2.000005, 2.000005], [2.000005, 2.000005]])

        self.assertFalse(ht.allclose(a, b))
        self.assertTrue(ht.allclose(a, b, atol=1e-04))
        self.assertTrue(ht.allclose(a, b, rtol=1e-04))
        self.assertTrue(ht.allclose(a, 2))
        self.assertTrue(ht.allclose(a, 2.0))
        self.assertTrue(ht.allclose(2, a))
        if not self.is_mps:
            self.assertTrue(ht.allclose(f, a))
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
        if not self.is_mps:
            x = ht.float64([[0, 0, 0], [0, 0, 0]])
            res = ht.zeros(1, dtype=ht.uint8)
            any_tensor = ht.any(x)
            self.assertIsInstance(any_tensor, ht.DNDarray)
            self.assertEqual(any_tensor.shape, ())
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

        # test keepdims
        ones_2d = ht.ones((1, 1))
        self.assertEqual(ones_2d.any(keepdims=True).shape, ones_2d.shape)

        ones_2d_split = ht.ones((2, 2), split=0)
        keepdims_any = ones_2d_split.any(keepdims=True)
        self.assertEqual(keepdims_any.shape, (1, 1))
        self.assertEqual(keepdims_any.split, None)
        keepdims_any = ones_2d_split.any(axis=0, keepdims=True)
        self.assertEqual(keepdims_any.shape, (1, 2))
        self.assertEqual(keepdims_any.split, None)
        keepdims_any = ones_2d_split.any(axis=1, keepdims=True)
        self.assertEqual(keepdims_any.shape, (2, 1))
        self.assertEqual(keepdims_any.split, 0)

        ones_2d_split = ht.ones((2, 2), split=1)
        keepdims_any = ones_2d_split.any(keepdims=True)
        self.assertEqual(keepdims_any.shape, (1, 1))
        self.assertEqual(keepdims_any.split, None)
        keepdims_any = ones_2d_split.any(axis=0, keepdims=True)
        self.assertEqual(keepdims_any.shape, (1, 2))
        self.assertEqual(keepdims_any.split, 1)
        keepdims_any = ones_2d_split.any(axis=1, keepdims=True)
        self.assertEqual(keepdims_any.shape, (2, 1))
        self.assertEqual(keepdims_any.split, None)

    def test_in1d(self):
        size = ht.communication.MPI_WORLD.size
        ref_ar1 = list(range(2 * size + 3))
        ref_ar2 = [0, 3, 6, 9]
        expected = np.array([v in ref_ar2 for v in ref_ar1])

        ar1 = ht.array(ref_ar1, split=0)
        ar2_split = ht.array(ref_ar2, split=0)

        result = ht.in1d(ar1, ar2_split)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (len(ref_ar1),))
        self.assertEqual(result.dtype, ht.bool)
        self.assertEqual(result.split, 0)
        self.assert_array_equal(result, expected)

        # plain (replicated) and list inputs
        self.assert_array_equal(ht.in1d(ar1, ht.array(ref_ar2)), expected)
        self.assert_array_equal(ht.in1d(ar1, ref_ar2), expected)

        # a 2-D input is flattened, in contrast to isin
        ar1_2d = ht.reshape(ht.arange(4 * size, split=0), (2, 2 * size))
        self.assertEqual(ht.in1d(ar1_2d, [0, 1]).shape, (4 * size,))

        with self.assertRaises(TypeError):
            ht.in1d("?", ar2_split)

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

    def test_isin(self):
        size = ht.communication.MPI_WORLD.size
        ref_elements = list(range(2 * size + 3))
        ref_test = [0, 3, 6, 9]
        expected = np.array([v in ref_test for v in ref_elements])
        expected_inverse = np.array([v not in ref_test for v in ref_elements])

        elements = ht.array(ref_elements, split=0)
        test_split = ht.array(ref_test, split=0)
        test_plain = ht.array(ref_test)

        # distributed, replicated, and list test elements must yield identical results,
        # also for unsplit elements
        for test in (test_split, test_plain, ref_test):
            result = ht.isin(elements, test)
            self.assertIsInstance(result, ht.DNDarray)
            self.assertEqual(result.shape, elements.shape)
            self.assertEqual(result.dtype, ht.bool)
            self.assertEqual(result.split, 0)
            self.assert_array_equal(result, expected)
            self.assert_array_equal(ht.isin(elements, test, invert=True), expected_inverse)
        self.assert_array_equal(ht.isin(ht.array(ref_elements), test_split), expected)

        # scalar elements, as allowed by numpy and the Python array API standard,
        # yield zero-dimensional results
        self.assert_array_equal(ht.isin(ref_elements[3], test_plain), np.array(True))
        self.assert_array_equal(ht.isin(ref_elements[4], test_split), np.array(False))
        self.assert_array_equal(ht.isin(4, test_plain, invert=True), np.array(True))

        # 2-D elements keep shape and split
        elements_2d = ht.reshape(ht.arange(4 * size, split=0), (2, 2 * size))
        result_2d = ht.isin(elements_2d, [0, 4 * size - 1])
        self.assertEqual(result_2d.shape, (2, 2 * size))
        self.assertEqual(result_2d.split, 0)
        expected_2d = np.arange(4 * size).reshape(2, 2 * size)
        expected_2d = np.isin(expected_2d, [0, 4 * size - 1])
        self.assert_array_equal(result_2d, expected_2d)

        # boolean and complex dtypes, which torch.isin does not support natively
        bools = ht.array([True, False, True, False], split=0)
        self.assert_array_equal(
            ht.isin(bools, ht.array([True], split=0)),
            np.array([True, False, True, False]),
        )

        mixed = ht.isin(ht.array([1, 2, 3], split=0), ht.array([1.0, 3.0], split=0))
        self.assert_array_equal(mixed, np.array([True, False, True]))

        for dtype in (ht.complex64, ht.complex128):
            complex_elements = ht.array([1 + 2j, 3 + 4j, 1 + 3j], dtype=dtype, split=0)
            self.assert_array_equal(
                ht.isin(complex_elements, ht.array([1 + 2j], dtype=dtype, split=0)),
                np.array([True, False, False]),
            )

        # empty test_elements match nothing, replicated and distributed
        self.assert_array_equal(
            ht.isin(elements, ht.array([], dtype=ht.int64)),
            np.zeros(len(ref_elements), dtype=bool),
        )
        self.assert_array_equal(
            ht.isin(elements, ht.array([], dtype=ht.int64, split=0)),
            np.zeros(len(ref_elements), dtype=bool),
        )

        with self.assertRaises(TypeError):
            ht.isin("?", test_plain)

    def test_isin_numpy_suite(self):
        # adapted from the isin tests in numpy's numpy/lib/tests/test_arraysetops.py
        # (numpy is BSD-3-Clause licensed); cases relying on dtypes that heat does not
        # support (uint16/uint64, timedelta, object, structured, strings) are skipped
        ref_a = [5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5]

        # duplicated test elements (as in numpy's test_isin_additional); note that we
        # deliberately do not combine this with assume_unique=True: numpy tolerates that
        # misuse, but torch.isin's algorithm requires unique test values in that mode and
        # would silently return wrong results
        for mult in (1, 10):
            a = ht.array(ref_a, split=0)
            b = ht.array([2, 3, 4] * mult, split=0)
            expected = np.array([v in [2, 3, 4] for v in ref_a])
            self.assert_array_equal(ht.isin(a, b), expected)

            # assume_unique with genuinely unique inputs
            unique_b = ht.array([2, 3, 4], split=0)
            self.assert_array_equal(ht.isin(a, unique_b, assume_unique=True), expected)

            # invert is the logical negation, for integers and floats
            self.assert_array_equal(
                ht.isin(a, b, invert=True),
                np.invert(expected),
            )
            a = ht.array(ref_a, dtype=ht.float32, split=0)
            b = ht.array([2, 3, 4] * mult, dtype=ht.float32, split=0)
            self.assert_array_equal(ht.isin(a, b, invert=True), np.invert(expected))

        # values beyond the small-value fast path (numpy's test_isin_hit_alternate_algorithm)
        a = ht.array([5, 4, 5, 3, 4, 4, int(1e9)], split=0)
        b = ht.array([2, 3, 4, int(1e9)], split=0)
        self.assert_array_equal(ht.isin(a, b), np.array([False, True, False, True, True, True, True]))

        # boolean input, with and without invert
        a = ht.array([True, False], split=0)
        b = ht.array([False, False, False], split=0)
        self.assert_array_equal(ht.isin(a, b), np.array([False, True]))
        self.assert_array_equal(ht.isin(a, b, invert=True), np.array([True, False]))

        # mixed integer dtypes (numpy's test_isin_mixed_dtype)
        for dtype1, dtype2 in [
            (ht.int8, ht.int16),
            (ht.int16, ht.int8),
            (ht.uint8, ht.int16),
            (ht.int16, ht.uint8),
            (ht.uint8, ht.int32),
        ]:
            vals1 = [0, 0, 1, 1]
            vals2 = [-128, 0, 127] if dtype2 in (ht.int8, ht.int16, ht.int32, ht.int64) else [127, 0, 255]
            a = ht.array(vals1, dtype=dtype1, split=0)
            b = ht.array(vals2, dtype=dtype2, split=0)
            self.assert_array_equal(ht.isin(a, b), np.isin(vals1, vals2))

        # boolean against all other dtypes, both directions (numpy's test_isin_mixed_boolean)
        for dtype in (ht.int8, ht.int32, ht.uint8, ht.float32):
            a = ht.array([True, False, False], split=0)
            b = ht.array([0, 0, 0, 0], dtype=dtype, split=0)
            self.assert_array_equal(ht.isin(a, b), np.array([False, True, True]))
            self.assert_array_equal(ht.isin(b, a), np.array([True, True, True, True]))

        # multi-dimensional inputs on both sides, and zero-dimensional test elements
        a = ht.arange(24, split=0).reshape(2, 3, 4)
        b = ht.array([[10, 20, 30], [0, 1, 3], [11, 22, 33]], split=0)
        ref_3d = np.arange(24).reshape(2, 3, 4)
        self.assert_array_equal(ht.isin(a, b), np.isin(ref_3d, [[10, 20, 30], [0, 1, 3], [11, 22, 33]]))
        self.assert_array_equal(ht.isin(a, ht.array(3)), ref_3d == 3)
        self.assert_array_equal(ht.isin(ht.array(3), b), np.isin(3, [[10, 20, 30], [0, 1, 3], [11, 22, 33]]))

        # empty arrays of various types, as element, as test elements, and both
        for dtype, ref in [(ht.bool, [True, False, False]), (ht.int64, [10, 20, 30]), (ht.float64, [10.0, 20.0, 30.0])]:
            empty = ht.array([], dtype=dtype, split=0)
            full = ht.array(ref, dtype=dtype, split=0)
            self.assert_array_equal(ht.isin(empty, full), np.array([], dtype=bool))
            self.assert_array_equal(ht.isin(full, empty), np.array([False, False, False]))
            self.assert_array_equal(ht.isin(empty, empty), np.array([], dtype=bool))

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
