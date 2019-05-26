import torch
import unittest
import math
import heat as ht


class TestTrigonometrics(unittest.TestCase):
    def test_arcsin(self):
        # base elements
        elements = [-1.,-0.83,-0.12,0.,0.24,0.67,1.]
        comparison = torch.tensor(elements, dtype=torch.float64).asin()

        # arcsin of float32
        float32_tensor = ht.array(elements, dtype=ht.float32)
        float32_arcsin = ht.arcsin(float32_tensor)
        self.assertIsInstance(float32_arcsin, ht.DNDarray)
        self.assertEqual(float32_arcsin.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_arcsin._DNDarray__array.type(torch.double), comparison))
        
        # arcsin of float64
        float64_tensor = ht.array(elements, dtype=ht.float64)
        float64_arcsin = ht.arcsin(float64_tensor)
        self.assertIsInstance(float64_arcsin, ht.DNDarray)
        self.assertEqual(float64_arcsin.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_arcsin._DNDarray__array.type(torch.double), comparison))
       
        # arcsin of value out of domain 
        nan_tensor = ht.array([1.2])
        nan_arcsin = ht.arcsin(nan_tensor)
        self.assertIsInstance(float64_arcsin, ht.DNDarray)
        self.assertEqual(nan_arcsin.dtype, ht.float32)
        self.assertTrue(math.isnan(nan_arcsin._DNDarray__array.item()))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.arcsin([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.arcsin('hello world')


    def test_cos(self):
        # base elements
        elements = 30
        comparison = torch.arange(elements, dtype=torch.float64).cos()

        # cosine of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_cos = ht.cos(float32_tensor)
        self.assertIsInstance(float32_cos, ht.DNDarray)
        self.assertEqual(float32_cos.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_cos._DNDarray__array.type(torch.double), comparison))

        # cosine of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_cos = ht.cos(float64_tensor)
        self.assertIsInstance(float64_cos, ht.DNDarray)
        self.assertEqual(float64_cos.dtype, ht.float64)
        self.assertEqual(float64_cos.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_cos._DNDarray__array.type(torch.double), comparison))

        # cosine of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_cos = ht.cos(int32_tensor)
        self.assertIsInstance(int32_cos, ht.DNDarray)
        self.assertEqual(int32_cos.dtype, ht.float64)
        self.assertEqual(int32_cos.dtype, ht.float64)
        self.assertTrue(torch.allclose(float32_cos._DNDarray__array.type(torch.double), comparison))

        # cosine of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_cos = ht.cos(int64_tensor)
        self.assertIsInstance(int64_cos, ht.DNDarray)
        self.assertEqual(int64_cos.dtype, ht.float64)
        self.assertEqual(int64_cos.dtype, ht.float64)
        self.assertTrue(torch.allclose(int64_cos._DNDarray__array.type(torch.double), comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.cos([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.cos('hello world')

    def test_cosh(self):
        # base elements
        elements = 30
        comparison = torch.arange(elements, dtype=torch.float64).cosh()

        # hyperbolic cosine of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_cosh = ht.cosh(float32_tensor)
        self.assertIsInstance(float32_cosh, ht.DNDarray)
        self.assertEqual(float32_cosh.dtype, ht.float32)
        self.assertEqual(float32_cosh.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_cosh._DNDarray__array.type(torch.double), comparison))

        # hyperbolic cosine of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_cosh = ht.cosh(float64_tensor)
        self.assertIsInstance(float64_cosh, ht.DNDarray)
        self.assertEqual(float64_cosh.dtype, ht.float64)
        self.assertEqual(float64_cosh.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_cosh._DNDarray__array.type(torch.double), comparison))

        # hyperbolic cosine of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_cosh = ht.cosh(int32_tensor)
        self.assertIsInstance(int32_cosh, ht.DNDarray)
        self.assertEqual(int32_cosh.dtype, ht.float64)
        self.assertEqual(int32_cosh.dtype, ht.float64)
        self.assertTrue(torch.allclose(float32_cosh._DNDarray__array.type(torch.double), comparison))

        # hyperbolic cosine of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_cosh = ht.cosh(int64_tensor)
        self.assertIsInstance(int64_cosh, ht.DNDarray)
        self.assertEqual(int64_cosh.dtype, ht.float64)
        self.assertEqual(int64_cosh.dtype, ht.float64)
        self.assertTrue(torch.allclose(int64_cosh._DNDarray__array.type(torch.double), comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.cosh([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.cosh('hello world')

    def test_sin(self):
        # base elements
        elements = 30
        comparison = torch.arange(elements, dtype=torch.float64).sin()

        # sine of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_sin = ht.sin(float32_tensor)
        self.assertIsInstance(float32_sin, ht.DNDarray)
        self.assertEqual(float32_sin.dtype, ht.float32)
        self.assertEqual(float32_sin.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_sin._DNDarray__array.type(torch.double), comparison))

        # sine of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_sin = ht.sin(float64_tensor)
        self.assertIsInstance(float64_sin, ht.DNDarray)
        self.assertEqual(float64_sin.dtype, ht.float64)
        self.assertEqual(float64_sin.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_sin._DNDarray__array.type(torch.double), comparison))

        # sine of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_sin = ht.sin(int32_tensor)
        self.assertIsInstance(int32_sin, ht.DNDarray)
        self.assertEqual(int32_sin.dtype, ht.float64)
        self.assertEqual(int32_sin.dtype, ht.float64)
        self.assertTrue(torch.allclose(int32_sin._DNDarray__array.type(torch.double), comparison))

        # sine of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_sin = ht.sin(int64_tensor)
        self.assertIsInstance(int64_sin, ht.DNDarray)
        self.assertEqual(int64_sin.dtype, ht.float64)
        self.assertEqual(int64_sin.dtype, ht.float64)
        self.assertTrue(torch.allclose(int64_sin._DNDarray__array.type(torch.double), comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.sin([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.sin('hello world')

    def test_sinh(self):
        # base elements
        elements = 30
        comparison = torch.arange(elements, dtype=torch.float64).sinh()

        # hyperbolic sine of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_sinh = ht.sinh(float32_tensor)
        self.assertIsInstance(float32_sinh, ht.DNDarray)
        self.assertEqual(float32_sinh.dtype, ht.float32)
        self.assertEqual(float32_sinh.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_sinh._DNDarray__array.type(torch.double), comparison))

        # hyperbolic sine of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_sinh = ht.sinh(float64_tensor)
        self.assertIsInstance(float64_sinh, ht.DNDarray)
        self.assertEqual(float64_sinh.dtype, ht.float64)
        self.assertEqual(float64_sinh.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_sinh._DNDarray__array.type(torch.double), comparison))

        # hyperbolic sine of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_sinh = ht.sinh(int32_tensor)
        self.assertIsInstance(int32_sinh, ht.DNDarray)
        self.assertEqual(int32_sinh.dtype, ht.float64)
        self.assertEqual(int32_sinh.dtype, ht.float64)
        self.assertTrue(torch.allclose(int32_sinh._DNDarray__array.type(torch.double), comparison))

        # hyperbolic sine of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_sinh = ht.sinh(int64_tensor)
        self.assertIsInstance(int64_sinh, ht.DNDarray)
        self.assertEqual(int64_sinh.dtype, ht.float64)
        self.assertEqual(int64_sinh.dtype, ht.float64)
        self.assertTrue(torch.allclose(int64_sinh._DNDarray__array.type(torch.double), comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.sinh([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.sinh('hello world')

    def test_tan(self):
        # base elements
        elements = 30
        comparison = torch.arange(elements, dtype=torch.float64).tan()

        # tangent of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_tan = ht.tan(float32_tensor)
        self.assertIsInstance(float32_tan, ht.DNDarray)
        self.assertEqual(float32_tan.dtype, ht.float32)
        self.assertEqual(float32_tan.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_tan._DNDarray__array.type(torch.double), comparison))

        # tangent of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_tan = ht.tan(float64_tensor)
        self.assertIsInstance(float64_tan, ht.DNDarray)
        self.assertEqual(float64_tan.dtype, ht.float64)
        self.assertEqual(float64_tan.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_tan._DNDarray__array.type(torch.double), comparison))

        # tangent of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_tan = ht.tan(int32_tensor)
        self.assertIsInstance(int32_tan, ht.DNDarray)
        self.assertEqual(int32_tan.dtype, ht.float64)
        self.assertEqual(int32_tan.dtype, ht.float64)
        self.assertTrue(torch.allclose(int32_tan._DNDarray__array.type(torch.double), comparison))

        # tangent of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_tan = ht.tan(int64_tensor)
        self.assertIsInstance(int64_tan, ht.DNDarray)
        self.assertEqual(int64_tan.dtype, ht.float64)
        self.assertEqual(int64_tan.dtype, ht.float64)
        self.assertTrue(torch.allclose(int64_tan._DNDarray__array.type(torch.double), comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.tan([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.tan('hello world')

    def test_tanh(self):
        # base elements
        elements = 30
        comparison = torch.arange(elements, dtype=torch.float64).tanh()

        # hyperbolic tangent of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_tanh = ht.tanh(float32_tensor)
        self.assertIsInstance(float32_tanh, ht.DNDarray)
        self.assertEqual(float32_tanh.dtype, ht.float32)
        self.assertEqual(float32_tanh.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_tanh._DNDarray__array.type(torch.double), comparison))

        # hyperbolic tangent of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_tanh = ht.tanh(float64_tensor)
        self.assertIsInstance(float64_tanh, ht.DNDarray)
        self.assertEqual(float64_tanh.dtype, ht.float64)
        self.assertEqual(float64_tanh.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_tanh._DNDarray__array.type(torch.double), comparison))

        # hyperbolic tangent of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_tanh = ht.tanh(int32_tensor)
        self.assertIsInstance(int32_tanh, ht.DNDarray)
        self.assertEqual(int32_tanh.dtype, ht.float64)
        self.assertEqual(int32_tanh.dtype, ht.float64)
        self.assertTrue(torch.allclose(int32_tanh._DNDarray__array.type(torch.double), comparison))

        # hyperbolic tangent of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_tanh = ht.tanh(int64_tensor)
        self.assertIsInstance(int64_tanh, ht.DNDarray)
        self.assertEqual(int64_tanh.dtype, ht.float64)
        self.assertEqual(int64_tanh.dtype, ht.float64)
        self.assertTrue(torch.allclose(int64_tanh._DNDarray__array.type(torch.double), comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.tanh([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.tanh('hello world')
