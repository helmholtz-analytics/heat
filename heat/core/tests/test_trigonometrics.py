import unittest
import torch

import heat as ht

FLOAT_EPSILON = 1e-4


class TestOperations(unittest.TestCase):
    def test_cos(self):
        # base elements
        elements = 30
        comparison = torch.arange(elements, dtype=torch.float64).cos()

        # cose of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_cos = ht.cos(float32_tensor)
        self.assertIsInstance(float32_cos, ht.tensor)
        self.assertEqual(float32_cos.dtype, ht.float32)
        self.assertEqual(float32_cos.dtype, ht.float32)
        in_range = (float32_cos._tensor__array - comparison.type(torch.float32)) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # cose of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_cos = ht.cos(float64_tensor)
        self.assertIsInstance(float64_cos, ht.tensor)
        self.assertEqual(float64_cos.dtype, ht.float64)
        self.assertEqual(float64_cos.dtype, ht.float64)
        in_range = (float64_cos._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logarithm of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_cos = ht.cos(int32_tensor)
        self.assertIsInstance(int32_cos, ht.tensor)
        self.assertEqual(int32_cos.dtype, ht.float64)
        self.assertEqual(int32_cos.dtype, ht.float64)
        in_range = (int32_cos._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logathm of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_cos = ht.cos(int64_tensor)
        self.assertIsInstance(int64_cos, ht.tensor)
        self.assertEqual(int64_cos.dtype, ht.float64)
        self.assertEqual(int64_cos.dtype, ht.float64)
        in_range = (int64_cos._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.cos([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.cos('hello world')

    def test_sin(self):
        # base elements
        elements = 30
        comparison = torch.arange(elements, dtype=torch.float64).sin()

        # sine of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_sin = ht.sin(float32_tensor)
        self.assertIsInstance(float32_sin, ht.tensor)
        self.assertEqual(float32_sin.dtype, ht.float32)
        self.assertEqual(float32_sin.dtype, ht.float32)
        in_range = (float32_sin._tensor__array - comparison.type(torch.float32)) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # sine of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_sin = ht.sin(float64_tensor)
        self.assertIsInstance(float64_sin, ht.tensor)
        self.assertEqual(float64_sin.dtype, ht.float64)
        self.assertEqual(float64_sin.dtype, ht.float64)
        in_range = (float64_sin._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logarithm of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_sin = ht.sin(int32_tensor)
        self.assertIsInstance(int32_sin, ht.tensor)
        self.assertEqual(int32_sin.dtype, ht.float64)
        self.assertEqual(int32_sin.dtype, ht.float64)
        in_range = (int32_sin._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logathm of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_sin = ht.sin(int64_tensor)
        self.assertIsInstance(int64_sin, ht.tensor)
        self.assertEqual(int64_sin.dtype, ht.float64)
        self.assertEqual(int64_sin.dtype, ht.float64)
        in_range = (int64_sin._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.sin([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.sin('hello world')
