import unittest
import torch

import heat as ht


class TestOperations(unittest.TestCase):
    def test_cos(self):
        # base elements
        elements = 30
        comparison = torch.arange(elements, dtype=torch.float64).cos()

        # cosine of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_cos = ht.cos(float32_tensor)
        self.assertIsInstance(float32_cos, ht.tensor)
        self.assertEqual(float32_cos.dtype, ht.float32)
        self.assertEqual(float32_cos.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_cos._tensor__array.type(torch.double), comparison))

        # cosine of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_cos = ht.cos(float64_tensor)
        self.assertIsInstance(float64_cos, ht.tensor)
        self.assertEqual(float64_cos.dtype, ht.float64)
        self.assertEqual(float64_cos.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_cos._tensor__array.type(torch.double), comparison))

        # cosine of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_cos = ht.cos(int32_tensor)
        self.assertIsInstance(int32_cos, ht.tensor)
        self.assertEqual(int32_cos.dtype, ht.float64)
        self.assertEqual(int32_cos.dtype, ht.float64)
        self.assertTrue(torch.allclose(float32_cos._tensor__array.type(torch.double), comparison))

        # cosine of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_cos = ht.cos(int64_tensor)
        self.assertIsInstance(int64_cos, ht.tensor)
        self.assertEqual(int64_cos.dtype, ht.float64)
        self.assertEqual(int64_cos.dtype, ht.float64)
        self.assertTrue(torch.allclose(int64_cos._tensor__array.type(torch.double), comparison))

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
        self.assertTrue(torch.allclose(float32_sin._tensor__array.type(torch.double), comparison))

        # sine of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_sin = ht.sin(float64_tensor)
        self.assertIsInstance(float64_sin, ht.tensor)
        self.assertEqual(float64_sin.dtype, ht.float64)
        self.assertEqual(float64_sin.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_sin._tensor__array.type(torch.double), comparison))

        # sine of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_sin = ht.sin(int32_tensor)
        self.assertIsInstance(int32_sin, ht.tensor)
        self.assertEqual(int32_sin.dtype, ht.float64)
        self.assertEqual(int32_sin.dtype, ht.float64)
        self.assertTrue(torch.allclose(int32_sin._tensor__array.type(torch.double), comparison))

        # sine of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_sin = ht.sin(int64_tensor)
        self.assertIsInstance(int64_sin, ht.tensor)
        self.assertEqual(int64_sin.dtype, ht.float64)
        self.assertEqual(int64_sin.dtype, ht.float64)
        self.assertTrue(torch.allclose(int64_sin._tensor__array.type(torch.double), comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.sin([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.sin('hello world')

    def test_tan(self):
        # base elements
        elements = 30
        comparison = torch.arange(elements, dtype=torch.float64).tan()

        # tangent of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_tan = ht.tan(float32_tensor)
        self.assertIsInstance(float32_tan, ht.tensor)
        self.assertEqual(float32_tan.dtype, ht.float32)
        self.assertEqual(float32_tan.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_tan._tensor__array.type(torch.double), comparison))

        # tangent of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_tan = ht.tan(float64_tensor)
        self.assertIsInstance(float64_tan, ht.tensor)
        self.assertEqual(float64_tan.dtype, ht.float64)
        self.assertEqual(float64_tan.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_tan._tensor__array.type(torch.double), comparison))

        # tangent of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_tan = ht.tan(int32_tensor)
        self.assertIsInstance(int32_tan, ht.tensor)
        self.assertEqual(int32_tan.dtype, ht.float64)
        self.assertEqual(int32_tan.dtype, ht.float64)
        self.assertTrue(torch.allclose(int32_tan._tensor__array.type(torch.double), comparison))

        # tangent of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_tan = ht.tan(int64_tensor)
        self.assertIsInstance(int64_tan, ht.tensor)
        self.assertEqual(int64_tan.dtype, ht.float64)
        self.assertEqual(int64_tan.dtype, ht.float64)
        self.assertTrue(torch.allclose(int64_tan._tensor__array.type(torch.double), comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.tan([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.tan('hello world')

