import operator

import torch
import unittest
import os

import heat as ht
import numpy as np

from heat.core.tests.test_suites.basic_test import TestCase


class TestArithmetics(TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestArithmetics, cls).setUpClass()
        cls.a_scalar = 2.0
        cls.an_int_scalar = 2

        cls.a_vector = ht.float32([2, 2], device=cls.ht_device)
        cls.another_vector = ht.float32([2, 2, 2], device=cls.ht_device)

        cls.a_tensor = ht.array([[1.0, 2.0], [3.0, 4.0]], device=cls.ht_device)
        cls.another_tensor = ht.array([[2.0, 2.0], [2.0, 2.0]], device=cls.ht_device)
        cls.a_split_tensor = cls.another_tensor.copy().resplit_(0)

        cls.errorneous_type = (2, 2)

    def test_add(self):
        result = ht.array([[3.0, 4.0], [5.0, 6.0]], device=self.ht_device)

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
            ht.add("T", "s")

    def test_bitwise_and(self):
        an_int_tensor = ht.array([[1, 2], [3, 4]], device=self.ht_device)
        an_int_vector = ht.array([2, 2], device=self.ht_device)
        another_int_vector = ht.array([2, 2, 2, 2], device=self.ht_device)
        int_result = ht.array([[0, 2], [2, 0]], device=self.ht_device)

        a_boolean_vector = ht.array([False, True, False, True], device=self.ht_device)
        another_boolean_vector = ht.array([False, False, True, True], device=self.ht_device)
        boolean_result = ht.array([False, False, False, True], device=self.ht_device)

        self.assertTrue(ht.equal(ht.bitwise_and(an_int_tensor, self.an_int_scalar), int_result))
        self.assertTrue(ht.equal(ht.bitwise_and(an_int_tensor, an_int_vector), int_result))
        self.assertTrue(
            ht.equal(ht.bitwise_and(a_boolean_vector, another_boolean_vector), boolean_result)
        )
        self.assertTrue(
            ht.equal(ht.bitwise_and(an_int_tensor.copy().resplit_(0), an_int_vector), int_result)
        )

        with self.assertRaises(TypeError):
            ht.bitwise_and(self.a_tensor, self.another_tensor)
        with self.assertRaises(ValueError):
            ht.bitwise_and(an_int_vector, another_int_vector)
        with self.assertRaises(TypeError):
            ht.bitwise_and(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.bitwise_and("T", "s")
        with self.assertRaises(TypeError):
            ht.bitwise_and(an_int_tensor, "s")
        with self.assertRaises(TypeError):
            ht.bitwise_and(self.an_int_scalar, "s")
        with self.assertRaises(TypeError):
            ht.bitwise_and("s", self.an_int_scalar)
        with self.assertRaises(TypeError):
            ht.bitwise_and(self.an_int_scalar, self.a_scalar)

    def test_bitwise_or(self):
        an_int_tensor = ht.array([[1, 2], [3, 4]], device=self.ht_device)
        an_int_vector = ht.array([2, 2], device=self.ht_device)
        another_int_vector = ht.array([2, 2, 2, 2], device=self.ht_device)
        int_result = ht.array([[3, 2], [3, 6]], device=self.ht_device)

        a_boolean_vector = ht.array([False, True, False, True], device=self.ht_device)
        another_boolean_vector = ht.array([False, False, True, True], device=self.ht_device)
        boolean_result = ht.array([False, True, True, True], device=self.ht_device)

        self.assertTrue(ht.equal(ht.bitwise_or(an_int_tensor, self.an_int_scalar), int_result))
        self.assertTrue(ht.equal(ht.bitwise_or(an_int_tensor, an_int_vector), int_result))
        self.assertTrue(
            ht.equal(ht.bitwise_or(a_boolean_vector, another_boolean_vector), boolean_result)
        )
        self.assertTrue(
            ht.equal(ht.bitwise_or(an_int_tensor.copy().resplit_(0), an_int_vector), int_result)
        )

        with self.assertRaises(TypeError):
            ht.bitwise_or(self.a_tensor, self.another_tensor)
        with self.assertRaises(ValueError):
            ht.bitwise_or(an_int_vector, another_int_vector)
        with self.assertRaises(TypeError):
            ht.bitwise_or(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.bitwise_or("T", "s")
        with self.assertRaises(TypeError):
            ht.bitwise_or(an_int_tensor, "s")
        with self.assertRaises(TypeError):
            ht.bitwise_or(self.an_int_scalar, "s")
        with self.assertRaises(TypeError):
            ht.bitwise_or("s", self.an_int_scalar)
        with self.assertRaises(TypeError):
            ht.bitwise_or(self.an_int_scalar, self.a_scalar)

    def test_bitwise_xor(self):
        an_int_tensor = ht.array([[1, 2], [3, 4]], device=self.ht_device)
        an_int_vector = ht.array([2, 2], device=self.ht_device)
        another_int_vector = ht.array([2, 2, 2, 2], device=self.ht_device)
        int_result = ht.array([[3, 0], [1, 6]], device=self.ht_device)

        a_boolean_vector = ht.array([False, True, False, True], device=self.ht_device)
        another_boolean_vector = ht.array([False, False, True, True], device=self.ht_device)
        boolean_result = ht.array([False, True, True, False], device=self.ht_device)

        self.assertTrue(ht.equal(ht.bitwise_xor(an_int_tensor, self.an_int_scalar), int_result))
        self.assertTrue(ht.equal(ht.bitwise_xor(an_int_tensor, an_int_vector), int_result))
        self.assertTrue(
            ht.equal(ht.bitwise_xor(a_boolean_vector, another_boolean_vector), boolean_result)
        )
        self.assertTrue(
            ht.equal(ht.bitwise_xor(an_int_tensor.copy().resplit_(0), an_int_vector), int_result)
        )

        with self.assertRaises(TypeError):
            ht.bitwise_xor(self.a_tensor, self.another_tensor)
        with self.assertRaises(ValueError):
            ht.bitwise_xor(an_int_vector, another_int_vector)
        with self.assertRaises(TypeError):
            ht.bitwise_xor(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.bitwise_xor("T", "s")
        with self.assertRaises(TypeError):
            ht.bitwise_xor(an_int_tensor, "s")
        with self.assertRaises(TypeError):
            ht.bitwise_xor(self.an_int_scalar, "s")
        with self.assertRaises(TypeError):
            ht.bitwise_xor("s", self.an_int_scalar)
        with self.assertRaises(TypeError):
            ht.bitwise_xor(self.an_int_scalar, self.a_scalar)

    def test_cumprod(self):
        a = ht.full((2, 4), 2, dtype=ht.int32, device=self.ht_device)
        result = ht.array([[2, 4, 8, 16], [2, 4, 8, 16]], dtype=ht.int32, device=self.ht_device)

        # split = None
        cumprod = ht.cumprod(a, 1)
        self.assertTrue(ht.equal(cumprod, result))

        # Alias
        cumprod = ht.cumproduct(a, 1)
        self.assertTrue(ht.equal(cumprod, result))

        a = ht.full((4, 2), 2, dtype=ht.int64, split=0, device=self.ht_device)
        result = ht.array(
            [[2, 2], [4, 4], [8, 8], [16, 16]], dtype=ht.int64, split=0, device=self.ht_device
        )

        cumprod = ht.cumprod(a, 0)
        self.assertTrue(ht.equal(cumprod, result))

        # 3D
        out = ht.empty((2, 2, 2), dtype=ht.float32, split=0, device=self.ht_device)

        a = ht.full((2, 2, 2), 2, split=0, device=self.ht_device)
        result = ht.array(
            [[[2, 2], [2, 2]], [[4, 4], [4, 4]]], dtype=ht.float32, split=0, device=self.ht_device
        )

        cumprod = ht.cumprod(a, 0, out=out)
        self.assertTrue(ht.equal(cumprod, out))
        self.assertTrue(ht.equal(cumprod, result))

        a = ht.full((2, 2, 2), 2, dtype=ht.int32, split=1, device=self.ht_device)
        result = ht.array(
            [[[2, 2], [4, 4]], [[2, 2], [4, 4]]], dtype=ht.float32, split=1, device=self.ht_device
        )

        cumprod = ht.cumprod(a, 1, dtype=ht.float64)
        self.assertTrue(ht.equal(cumprod, result))

        a = ht.full((2, 2, 2), 2, dtype=ht.float32, split=2, device=self.ht_device)
        result = ht.array(
            [[[2, 4], [2, 4]], [[2, 4], [2, 4]]], dtype=ht.float32, split=2, device=self.ht_device
        )

        cumprod = ht.cumprod(a, 2)
        self.assertTrue(ht.equal(cumprod, result))

        with self.assertRaises(NotImplementedError):
            ht.cumprod(ht.ones((2, 2), device=self.ht_device), axis=None)
        with self.assertRaises(TypeError):
            ht.cumprod(ht.ones((2, 2), device=self.ht_device), axis="1")
        with self.assertRaises(RuntimeError):
            ht.cumprod(a, 2, out=out)
        with self.assertRaises(ValueError):
            ht.cumprod(ht.ones((2, 2), device=self.ht_device), 2)

        if self.ht_device is not None:
            with self.assertRaises(RuntimeError):
                ht.cumprod(ht.ones((2, 2, 2)), 2, out=out)

    def test_cumsum(self):
        a = ht.ones((2, 4), dtype=ht.int32, device=self.ht_device)
        result = ht.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=ht.int32, device=self.ht_device)

        # split = None
        cumsum = ht.cumsum(a, 1)
        self.assertTrue(ht.equal(cumsum, result))

        a = ht.ones((4, 2), dtype=ht.int64, split=0, device=self.ht_device)
        result = ht.array(
            [[1, 1], [2, 2], [3, 3], [4, 4]], dtype=ht.int64, split=0, device=self.ht_device
        )

        cumsum = ht.cumsum(a, 0)
        self.assertTrue(ht.equal(cumsum, result))

        # 3D
        out = ht.empty((2, 2, 2), dtype=ht.float32, split=0)

        a = ht.ones((2, 2, 2), split=0, device=self.ht_device)
        result = ht.array(
            [[[1, 1], [1, 1]], [[2, 2], [2, 2]]], dtype=ht.float32, split=0, device=self.ht_device
        )

        cumsum = ht.cumsum(a, 0, out=out)
        self.assertTrue(ht.equal(cumsum, out))
        self.assertTrue(ht.equal(cumsum, result))

        a = ht.ones((2, 2, 2), dtype=ht.int32, split=1, device=self.ht_device)
        result = ht.array(
            [[[1, 1], [2, 2]], [[1, 1], [2, 2]]], dtype=ht.float32, split=1, device=self.ht_device
        )

        cumsum = ht.cumsum(a, 1, dtype=ht.float64)
        self.assertTrue(ht.equal(cumsum, result))

        a = ht.ones((2, 2, 2), dtype=ht.float32, split=2, device=self.ht_device)
        result = ht.array(
            [[[1, 2], [1, 2]], [[1, 2], [1, 2]]], dtype=ht.float32, split=2, device=self.ht_device
        )

        cumsum = ht.cumsum(a, 2)
        self.assertTrue(ht.equal(cumsum, result))

        with self.assertRaises(NotImplementedError):
            ht.cumsum(ht.ones((2, 2)), axis=None)
        with self.assertRaises(TypeError):
            ht.cumsum(ht.ones((2, 2)), axis="1")
        with self.assertRaises(RuntimeError):
            ht.cumsum(a, 2, out=out)
        with self.assertRaises(ValueError):
            ht.cumsum(ht.ones((2, 2)), 2)

        if self.ht_device is not None:
            with self.assertRaises(RuntimeError):
                ht.cumsum(ht.ones((2, 2, 2)), 2, out=out)

    def test_diff(self):
        ht_array = ht.random.rand(20, 20, 20, split=None, device=self.ht_device)
        arb_slice = [0] * 3
        for dim in range(3):  # loop over 3 dimensions
            arb_slice[dim] = slice(None)
            for ax in range(dim + 1):  # loop over the possible axis values
                for sp in range(dim + 1):  # loop over the possible split values
                    # loop to 3 for the number of times to do the diff
                    for nl in range(1, 4):
                        # only generating the number once and then
                        lp_array = ht.manipulations.resplit(ht_array[arb_slice], sp)
                        np_array = ht_array[arb_slice].numpy()

                        ht_diff = ht.diff(lp_array, n=nl, axis=ax)
                        np_diff = ht.array(np.diff(np_array, n=nl, axis=ax), device=self.ht_device)

                        self.assertTrue(ht.equal(ht_diff, np_diff))
                        self.assertEqual(ht_diff.split, sp)
                        self.assertEqual(ht_diff.dtype, lp_array.dtype)

        np_array = ht_array.numpy()
        ht_diff = ht.diff(ht_array, n=2)
        np_diff = ht.array(np.diff(np_array, n=2), device=self.ht_device)
        self.assertTrue(ht.equal(ht_diff, np_diff))
        self.assertEqual(ht_diff.split, None)
        self.assertEqual(ht_diff.dtype, ht_array.dtype)

        ht_array = ht.random.rand(20, 20, 20, split=1, dtype=ht.float64, device=self.ht_device)
        np_array = ht_array.copy().numpy()
        ht_diff = ht.diff(ht_array, n=2)
        np_diff = ht.array(np.diff(np_array, n=2), device=self.ht_device)
        self.assertTrue(ht.equal(ht_diff, np_diff))
        self.assertEqual(ht_diff.split, 1)
        self.assertEqual(ht_diff.dtype, ht_array.dtype)

        # raises
        with self.assertRaises(ValueError):
            ht.diff(ht_array, n=-2)
        with self.assertRaises(TypeError):
            ht.diff(ht_array, axis="string")
        with self.assertRaises(TypeError):
            ht.diff("string", axis=2)

    def test_div(self):
        result = ht.array([[0.5, 1.0], [1.5, 2.0]], device=self.ht_device)
        commutated_result = ht.array([[2.0, 1.0], [2.0 / 3.0, 0.5]], device=self.ht_device)

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
            ht.div("T", "s")

    def test_fmod(self):
        result = ht.array([[1.0, 0.0], [1.0, 0.0]], device=self.ht_device)
        an_int_tensor = ht.array([[5, 3], [4, 1]], device=self.ht_device)
        integer_result = ht.array([[1, 1], [0, 1]], device=self.ht_device)
        commutated_result = ht.array([[0.0, 0.0], [2.0, 2.0]], device=self.ht_device)
        zero_tensor = ht.zeros((2, 2), device=self.ht_device)

        a_float = ht.array([5.3], device=self.ht_device)
        another_float = ht.array([1.9], device=self.ht_device)
        result_float = ht.array([1.5], device=self.ht_device)

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
            ht.fmod("T", "s")

    def test_invert(self):
        int8_tensor = ht.array([[0, 1], [2, -2]], dtype=ht.int8)
        uint8_tensor = ht.array([[23, 2], [45, 234]], dtype=ht.uint8)
        bool_tensor = ht.array([[False, True], [True, False]])
        float_tensor = ht.array([[0.4, 1.3], [1.3, -2.1]])
        int8_result = ht.array([[-1, -2], [-3, 1]])
        uint8_result = ht.array([[232, 253], [210, 21]])
        bool_result = ht.array([[True, False], [False, True]])

        self.assertTrue(ht.equal(ht.invert(int8_tensor), int8_result))
        self.assertTrue(ht.equal(ht.invert(int8_tensor.copy().resplit_(0)), int8_result))
        self.assertTrue(ht.equal(ht.invert(uint8_tensor), uint8_result))
        self.assertTrue(ht.equal(ht.invert(bool_tensor), bool_result))

        with self.assertRaises(TypeError):
            ht.invert(float_tensor)

    def test_left_shift(self):
        int_tensor = ht.array([[0, 1], [2, 3]])
        int_result = ht.array([[0, 2], [4, 6]])

        self.assertTrue(ht.equal(ht.left_shift(int_tensor, 1), int_result))
        self.assertTrue(ht.equal(ht.left_shift(int_tensor.copy().resplit_(0), 1), int_result))

        with self.assertRaises(TypeError):
            ht.left_shift(int_tensor, 2.4)
        with self.assertRaises(TypeError):
            ht.left_shift(ht.array([True]), 2)

    def test_mod(self):
        a_tensor = ht.array([[1, 4], [2, 2]], device=self.ht_device)
        another_tensor = ht.array([[1, 2], [3, 4]], device=self.ht_device)
        a_result = ht.array([[0, 0], [2, 2]], device=self.ht_device)
        another_result = ht.array([[1, 0], [0, 0]], device=self.ht_device)

        self.assertTrue(ht.equal(ht.mod(a_tensor, another_tensor), a_result))
        self.assertTrue(ht.equal(ht.mod(a_tensor, self.an_int_scalar), another_result))
        self.assertTrue(ht.equal(ht.mod(self.an_int_scalar, another_tensor), a_result))

    def test_mul(self):
        result = ht.array([[2.0, 4.0], [6.0, 8.0]], device=self.ht_device)

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
            ht.mul("T", "s")

    def test_pow(self):
        result = ht.array([[1.0, 4.0], [9.0, 16.0]], device=self.ht_device)
        commutated_result = ht.array([[2.0, 4.0], [8.0, 16.0]], device=self.ht_device)

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
            ht.pow("T", "s")

    def test_prod(self):
        array_len = 11

        # check sum over all float elements of 1d tensor locally
        shape_noaxis = ht.ones(array_len, device=self.ht_device)
        no_axis_prod = shape_noaxis.prod()

        self.assertIsInstance(no_axis_prod, ht.DNDarray)
        self.assertEqual(no_axis_prod.shape, (1,))
        self.assertEqual(no_axis_prod.lshape, (1,))
        self.assertEqual(no_axis_prod.dtype, ht.float32)
        self.assertEqual(no_axis_prod._DNDarray__array.dtype, torch.float32)
        self.assertEqual(no_axis_prod.split, None)
        self.assertEqual(no_axis_prod._DNDarray__array, 1)

        out_noaxis = ht.zeros((1,), device=self.ht_device)
        ht.prod(shape_noaxis, out=out_noaxis)
        self.assertEqual(out_noaxis._DNDarray__array, 1)

        # check sum over all float elements of split 1d tensor
        shape_noaxis_split = ht.arange(1, array_len, split=0, device=self.ht_device)
        shape_noaxis_split_prod = shape_noaxis_split.prod()

        self.assertIsInstance(shape_noaxis_split_prod, ht.DNDarray)
        self.assertEqual(shape_noaxis_split_prod.shape, (1,))
        self.assertEqual(shape_noaxis_split_prod.lshape, (1,))
        self.assertEqual(shape_noaxis_split_prod.dtype, ht.int64)
        self.assertEqual(shape_noaxis_split_prod._DNDarray__array.dtype, torch.int64)
        self.assertEqual(shape_noaxis_split_prod.split, None)
        self.assertEqual(shape_noaxis_split_prod, 3628800)

        out_noaxis = ht.zeros((1,), device=self.ht_device)
        ht.prod(shape_noaxis_split, out=out_noaxis)
        self.assertEqual(out_noaxis._DNDarray__array, 3628800)

        # check sum over all float elements of 3d tensor locally
        shape_noaxis = ht.full((3, 3, 3), 2, device=self.ht_device)
        no_axis_prod = shape_noaxis.prod()

        self.assertIsInstance(no_axis_prod, ht.DNDarray)
        self.assertEqual(no_axis_prod.shape, (1,))
        self.assertEqual(no_axis_prod.lshape, (1,))
        self.assertEqual(no_axis_prod.dtype, ht.float32)
        self.assertEqual(no_axis_prod._DNDarray__array.dtype, torch.float32)
        self.assertEqual(no_axis_prod.split, None)
        self.assertEqual(no_axis_prod._DNDarray__array, 134217728)

        out_noaxis = ht.zeros((1,), device=self.ht_device)
        ht.prod(shape_noaxis, out=out_noaxis)
        self.assertEqual(out_noaxis._DNDarray__array, 134217728)

        # check sum over all float elements of split 3d tensor
        shape_noaxis_split_axis = ht.full((3, 3, 3), 2, split=0, device=self.ht_device)
        split_axis_prod = shape_noaxis_split_axis.prod(axis=0)

        self.assertIsInstance(split_axis_prod, ht.DNDarray)
        self.assertEqual(split_axis_prod.shape, (3, 3))
        self.assertEqual(split_axis_prod.dtype, ht.float32)
        self.assertEqual(split_axis_prod._DNDarray__array.dtype, torch.float32)
        self.assertEqual(split_axis_prod.split, None)

        out_axis = ht.ones((3, 3), device=self.ht_device)
        ht.prod(shape_noaxis, axis=0, out=out_axis)
        self.assertTrue(
            (
                out_axis._DNDarray__array
                == torch.full((3,), 8, dtype=torch.float, device=self.torch_device)
            ).all()
        )

        # check sum over all float elements of splitted 5d tensor with negative axis
        shape_noaxis_split_axis_neg = ht.full((1, 2, 3, 4, 5), 2, split=1, device=self.ht_device)
        shape_noaxis_split_axis_neg_prod = shape_noaxis_split_axis_neg.prod(axis=-2)

        self.assertIsInstance(shape_noaxis_split_axis_neg_prod, ht.DNDarray)
        self.assertEqual(shape_noaxis_split_axis_neg_prod.shape, (1, 2, 3, 5))
        self.assertEqual(shape_noaxis_split_axis_neg_prod.dtype, ht.float32)
        self.assertEqual(shape_noaxis_split_axis_neg_prod._DNDarray__array.dtype, torch.float32)
        self.assertEqual(shape_noaxis_split_axis_neg_prod.split, 1)

        out_noaxis = ht.zeros((1, 2, 3, 5), device=self.ht_device)
        ht.prod(shape_noaxis_split_axis_neg, axis=-2, out=out_noaxis)

        # check sum over all float elements of splitted 3d tensor with tuple axis
        shape_split_axis_tuple = ht.ones((3, 4, 5), split=1, device=self.ht_device)
        shape_split_axis_tuple_prod = shape_split_axis_tuple.prod(axis=(-2, -3))
        expected_result = ht.ones((5,), device=self.ht_device)

        self.assertIsInstance(shape_split_axis_tuple_prod, ht.DNDarray)
        self.assertEqual(shape_split_axis_tuple_prod.shape, (5,))
        self.assertEqual(shape_split_axis_tuple_prod.dtype, ht.float32)
        self.assertEqual(shape_split_axis_tuple_prod._DNDarray__array.dtype, torch.float32)
        self.assertEqual(shape_split_axis_tuple_prod.split, None)
        self.assertTrue((shape_split_axis_tuple_prod == expected_result).all())

        # exceptions
        with self.assertRaises(ValueError):
            ht.ones(array_len, device=self.ht_device).prod(axis=1)
        with self.assertRaises(ValueError):
            ht.ones(array_len, device=self.ht_device).prod(axis=-2)
        with self.assertRaises(ValueError):
            ht.ones((4, 4), device=self.ht_device).prod(axis=0, out=out_noaxis)
        with self.assertRaises(TypeError):
            ht.ones(array_len, device=self.ht_device).prod(axis="bad_axis_type")

    def test_right_shift(self):
        int_tensor = ht.array([[0, 1], [2, 3]])
        int_result = ht.array([[0, 0], [1, 1]])

        self.assertTrue(ht.equal(ht.right_shift(int_tensor, 1), int_result))
        self.assertTrue(ht.equal(ht.right_shift(int_tensor.copy().resplit_(0), 1), int_result))

        with self.assertRaises(TypeError):
            ht.left_shift(int_tensor, 2.4)
        with self.assertRaises(TypeError):
            ht.left_shift(ht.array([True]), 2)

    def test_sub(self):
        result = ht.array([[-1.0, 0.0], [1.0, 2.0]], device=self.ht_device)
        minus_result = ht.array([[1.0, 0.0], [-1.0, -2.0]], device=self.ht_device)

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
            ht.sub("T", "s")

    def test_sum(self):
        array_len = 11

        # check sum over all float elements of 1d tensor locally
        shape_noaxis = ht.ones(array_len, device=self.ht_device)
        no_axis_sum = shape_noaxis.sum()

        self.assertIsInstance(no_axis_sum, ht.DNDarray)
        self.assertEqual(no_axis_sum.shape, (1,))
        self.assertEqual(no_axis_sum.lshape, (1,))
        self.assertEqual(no_axis_sum.dtype, ht.float32)
        self.assertEqual(no_axis_sum._DNDarray__array.dtype, torch.float32)
        self.assertEqual(no_axis_sum.split, None)
        self.assertEqual(no_axis_sum._DNDarray__array, array_len)

        out_noaxis = ht.zeros((1,), device=self.ht_device)
        ht.sum(shape_noaxis, out=out_noaxis)
        self.assertTrue(out_noaxis._DNDarray__array == shape_noaxis._DNDarray__array.sum())

        # check sum over all float elements of split 1d tensor
        shape_noaxis_split = ht.arange(array_len, split=0, device=self.ht_device)
        shape_noaxis_split_sum = shape_noaxis_split.sum()

        self.assertIsInstance(shape_noaxis_split_sum, ht.DNDarray)
        self.assertEqual(shape_noaxis_split_sum.shape, (1,))
        self.assertEqual(shape_noaxis_split_sum.lshape, (1,))
        self.assertEqual(shape_noaxis_split_sum.dtype, ht.int64)
        self.assertEqual(shape_noaxis_split_sum._DNDarray__array.dtype, torch.int64)
        self.assertEqual(shape_noaxis_split_sum.split, None)
        self.assertEqual(shape_noaxis_split_sum, 55)

        out_noaxis = ht.zeros((1,), device=self.ht_device)
        ht.sum(shape_noaxis_split, out=out_noaxis)
        self.assertEqual(out_noaxis._DNDarray__array, 55)

        # check sum over all float elements of 3d tensor locally
        shape_noaxis = ht.ones((3, 3, 3), device=self.ht_device)
        no_axis_sum = shape_noaxis.sum()

        self.assertIsInstance(no_axis_sum, ht.DNDarray)
        self.assertEqual(no_axis_sum.shape, (1,))
        self.assertEqual(no_axis_sum.lshape, (1,))
        self.assertEqual(no_axis_sum.dtype, ht.float32)
        self.assertEqual(no_axis_sum._DNDarray__array.dtype, torch.float32)
        self.assertEqual(no_axis_sum.split, None)
        self.assertEqual(no_axis_sum._DNDarray__array, 27)

        out_noaxis = ht.zeros((1,), device=self.ht_device)
        ht.sum(shape_noaxis, out=out_noaxis)
        self.assertEqual(out_noaxis._DNDarray__array, 27)

        # check sum over all float elements of split 3d tensor
        shape_noaxis_split_axis = ht.ones((3, 3, 3), split=0, device=self.ht_device)
        split_axis_sum = shape_noaxis_split_axis.sum(axis=0)

        self.assertIsInstance(split_axis_sum, ht.DNDarray)
        self.assertEqual(split_axis_sum.shape, (3, 3))
        self.assertEqual(split_axis_sum.dtype, ht.float32)
        self.assertEqual(split_axis_sum._DNDarray__array.dtype, torch.float32)
        self.assertEqual(split_axis_sum.split, None)

        out_noaxis = ht.zeros((3, 3), device=self.ht_device)
        ht.sum(shape_noaxis, axis=0, out=out_noaxis)
        self.assertTrue(
            (
                out_noaxis._DNDarray__array
                == torch.full((3, 3), 3, dtype=torch.float, device=self.torch_device)
            ).all()
        )

        # check sum over all float elements of splitted 5d tensor with negative axis
        shape_noaxis_split_axis_neg = ht.ones((1, 2, 3, 4, 5), split=1, device=self.ht_device)
        shape_noaxis_split_axis_neg_sum = shape_noaxis_split_axis_neg.sum(axis=-2)

        self.assertIsInstance(shape_noaxis_split_axis_neg_sum, ht.DNDarray)
        self.assertEqual(shape_noaxis_split_axis_neg_sum.shape, (1, 2, 3, 5))
        self.assertEqual(shape_noaxis_split_axis_neg_sum.dtype, ht.float32)
        self.assertEqual(shape_noaxis_split_axis_neg_sum._DNDarray__array.dtype, torch.float32)
        self.assertEqual(shape_noaxis_split_axis_neg_sum.split, 1)

        out_noaxis = ht.zeros((1, 2, 3, 5), device=self.ht_device)
        ht.sum(shape_noaxis_split_axis_neg, axis=-2, out=out_noaxis)

        # check sum over all float elements of splitted 3d tensor with tuple axis
        shape_split_axis_tuple = ht.ones((3, 4, 5), split=1, device=self.ht_device)
        shape_split_axis_tuple_sum = shape_split_axis_tuple.sum(axis=(-2, -3))
        expected_result = ht.ones((5,), device=self.ht_device) * 12.0

        self.assertIsInstance(shape_split_axis_tuple_sum, ht.DNDarray)
        self.assertEqual(shape_split_axis_tuple_sum.shape, (5,))
        self.assertEqual(shape_split_axis_tuple_sum.dtype, ht.float32)
        self.assertEqual(shape_split_axis_tuple_sum._DNDarray__array.dtype, torch.float32)
        self.assertEqual(shape_split_axis_tuple_sum.split, None)
        self.assertTrue((shape_split_axis_tuple_sum == expected_result).all())

        # exceptions
        with self.assertRaises(ValueError):
            ht.ones(array_len, device=self.ht_device).sum(axis=1)
        with self.assertRaises(ValueError):
            ht.ones(array_len, device=self.ht_device).sum(axis=-2)
        with self.assertRaises(ValueError):
            ht.ones((4, 4), device=self.ht_device).sum(axis=0, out=out_noaxis)
        with self.assertRaises(TypeError):
            ht.ones(array_len, device=self.ht_device).sum(axis="bad_axis_type")

    def test_right_hand_side_operations(self):
        """
        This test ensures that for each arithmetic operation (e.g. +, -, *, ...) that is implemented in the tensor
        class, it works both ways.

        Examples
        --------
        >>> import heat as ht
        >>> T = ht.float32([[1., 2.], [3., 4.]], device=self.ht_device)
        >>> assert T * 3 == 3 * T
        """
        operators = (
            ("__add__", operator.add, True),
            ("__sub__", operator.sub, False),
            ("__mul__", operator.mul, True),
            ("__truediv__", operator.truediv, False),
            ("__floordiv__", operator.floordiv, False),
            ("__mod__", operator.mod, False),
            ("__pow__", operator.pow, False),
        )
        tensor = ht.float32([[1, 4], [2, 3]], device=self.ht_device)
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
