import unittest
import torch

import numpy as np
import heat as ht

FLOAT_EPSILON = 1e-4


class TestOperations(unittest.TestCase):
    def test_exp(self):
        elements = 10
        comparison = torch.arange(elements, dtype=torch.float64).exp()

        # exponential of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_exp = ht.exp(float32_tensor)
        self.assertIsInstance(float32_exp, ht.tensor)
        self.assertEqual(float32_exp.dtype, ht.float32)
        self.assertEqual(float32_exp.dtype, ht.float32)
        in_range = (float32_exp._tensor__array - comparison.type(torch.float32)) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # exponential of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_exp = ht.exp(float64_tensor)
        self.assertIsInstance(float64_exp, ht.tensor)
        self.assertEqual(float64_exp.dtype, ht.float64)
        self.assertEqual(float64_exp.dtype, ht.float64)
        in_range = (float64_exp._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # exponential of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_exp = ht.exp(int32_tensor)
        self.assertIsInstance(int32_exp, ht.tensor)
        self.assertEqual(int32_exp.dtype, ht.float64)
        self.assertEqual(int32_exp.dtype, ht.float64)
        in_range = (int32_exp._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # exponential of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_exp = ht.exp(int64_tensor)
        self.assertIsInstance(int64_exp, ht.tensor)
        self.assertEqual(int64_exp.dtype, ht.float64)
        self.assertEqual(int64_exp.dtype, ht.float64)
        in_range = (int64_exp._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.exp([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.exp('hello world')

    def test_exp2(self):
        elements = 10
        comparison = np.exp2(torch.arange(elements, dtype=torch.float64))

        # exponential of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_exp2 = ht.exp2(float32_tensor)
        self.assertIsInstance(float32_exp2, ht.tensor)
        self.assertEqual(float32_exp2.dtype, ht.float32)
        self.assertEqual(float32_exp2.dtype, ht.float32)
        in_range = (float32_exp2._tensor__array - comparison.type(torch.float32)) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # exponential of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_exp2 = ht.exp2(float64_tensor)
        self.assertIsInstance(float64_exp2, ht.tensor)
        self.assertEqual(float64_exp2.dtype, ht.float64)
        self.assertEqual(float64_exp2.dtype, ht.float64)
        in_range = (float64_exp2._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # exponential of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_exp2 = ht.exp2(int32_tensor)
        self.assertIsInstance(int32_exp2, ht.tensor)
        self.assertEqual(int32_exp2.dtype, ht.float64)
        self.assertEqual(int32_exp2.dtype, ht.float64)
        in_range = (int32_exp2._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # exponential of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_exp2 = ht.exp2(int64_tensor)
        self.assertIsInstance(int64_exp2, ht.tensor)
        self.assertEqual(int64_exp2.dtype, ht.float64)
        self.assertEqual(int64_exp2.dtype, ht.float64)
        in_range = (int64_exp2._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.exp2([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.exp2('hello world')

    def test_log(self):
        elements = 15
        comparison = torch.arange(1, elements, dtype=torch.float64).log2()

        # logarithm of float32
        float32_tensor = ht.arange(1, elements, dtype=ht.float32)
        float32_log2 = ht.log2(float32_tensor)
        self.assertIsInstance(float32_log2, ht.tensor)
        self.assertEqual(float32_log2.dtype, ht.float32)
        self.assertEqual(float32_log2.dtype, ht.float32)
        in_range = (float32_log2._tensor__array - comparison.type(torch.float32)) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logarithm of float64
        float64_tensor = ht.arange(1, elements, dtype=ht.float64)
        float64_log2 = ht.log2(float64_tensor)
        self.assertIsInstance(float64_log2, ht.tensor)
        self.assertEqual(float64_log2.dtype, ht.float64)
        self.assertEqual(float64_log2.dtype, ht.float64)
        in_range = (float64_log2._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logarithm of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(1, elements, dtype=ht.int32)
        int32_log2 = ht.log2(int32_tensor)
        self.assertIsInstance(int32_log2, ht.tensor)
        self.assertEqual(int32_log2.dtype, ht.float64)
        self.assertEqual(int32_log2.dtype, ht.float64)
        in_range = (int32_log2._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # log2arithm of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(1, elements, dtype=ht.int64)
        int64_log2 = ht.log2(int64_tensor)
        self.assertIsInstance(int64_log2, ht.tensor)
        self.assertEqual(int64_log2.dtype, ht.float64)
        self.assertEqual(int64_log2.dtype, ht.float64)
        in_range = (int64_log2._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.log2([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.log2('hello world')

    def test_log2(self):
        elements = 15
        comparison = torch.arange(1, elements, dtype=torch.float64).log()

        # logarithm of float32
        float32_tensor = ht.arange(1, elements, dtype=ht.float32)
        float32_log = ht.log(float32_tensor)
        self.assertIsInstance(float32_log, ht.tensor)
        self.assertEqual(float32_log.dtype, ht.float32)
        self.assertEqual(float32_log.dtype, ht.float32)
        in_range = (float32_log._tensor__array - comparison.type(torch.float32)) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logarithm of float64
        float64_tensor = ht.arange(1, elements, dtype=ht.float64)
        float64_log = ht.log(float64_tensor)
        self.assertIsInstance(float64_log, ht.tensor)
        self.assertEqual(float64_log.dtype, ht.float64)
        self.assertEqual(float64_log.dtype, ht.float64)
        in_range = (float64_log._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logarithm of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(1, elements, dtype=ht.int32)
        int32_log = ht.log(int32_tensor)
        self.assertIsInstance(int32_log, ht.tensor)
        self.assertEqual(int32_log.dtype, ht.float64)
        self.assertEqual(int32_log.dtype, ht.float64)
        in_range = (int32_log._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logarithm of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(1, elements, dtype=ht.int64)
        int64_log = ht.log(int64_tensor)
        self.assertIsInstance(int64_log, ht.tensor)
        self.assertEqual(int64_log.dtype, ht.float64)
        self.assertEqual(int64_log.dtype, ht.float64)
        in_range = (int64_log._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.log([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.log('hello world')

    def test_log10(self):
        elements = 15
        comparison = torch.arange(1, elements, dtype=torch.float64).log10()

        # logarithm of float32
        float32_tensor = ht.arange(1, elements, dtype=ht.float32)
        float32_log10 = ht.log10(float32_tensor)
        self.assertIsInstance(float32_log10, ht.tensor)
        self.assertEqual(float32_log10.dtype, ht.float32)
        self.assertEqual(float32_log10.dtype, ht.float32)
        in_range = (float32_log10._tensor__array - comparison.type(torch.float32)) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logarithm of float64
        float64_tensor = ht.arange(1, elements, dtype=ht.float64)
        float64_log10 = ht.log10(float64_tensor)
        self.assertIsInstance(float64_log10, ht.tensor)
        self.assertEqual(float64_log10.dtype, ht.float64)
        self.assertEqual(float64_log10.dtype, ht.float64)
        in_range = (float64_log10._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logarithm of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(1, elements, dtype=ht.int32)
        int32_log10 = ht.log10(int32_tensor)
        self.assertIsInstance(int32_log10, ht.tensor)
        self.assertEqual(int32_log10.dtype, ht.float64)
        self.assertEqual(int32_log10.dtype, ht.float64)
        in_range = (int32_log10._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logarithm of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(1, elements, dtype=ht.int64)
        int64_log10 = ht.log10(int64_tensor)
        self.assertIsInstance(int64_log10, ht.tensor)
        self.assertEqual(int64_log10.dtype, ht.float64)
        self.assertEqual(int64_log10.dtype, ht.float64)
        in_range = (int64_log10._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.log10([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.log10('hello world')

    def test_sqrt(self):
        elements = 25
        comparison = ht.arange(elements, dtype=ht.float64).sqrt()

        # square roots of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_sqrt = ht.sqrt(float32_tensor)
        self.assertIsInstance(float32_sqrt, ht.tensor)
        self.assertEqual(float32_sqrt.dtype, ht.float32)
        self.assertEqual(float32_sqrt.dtype, ht.float32)
        self.assertTrue(ht.allclose(float32_sqrt, comparison.astype(ht.float32), 1e-06))

        # square roots of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_sqrt = ht.sqrt(float64_tensor)
        self.assertIsInstance(float64_sqrt, ht.tensor)
        self.assertEqual(float64_sqrt.dtype, ht.float64)
        self.assertEqual(float64_sqrt.dtype, ht.float64)
        self.assertTrue(ht.allclose(float64_sqrt, comparison, 1e-06))

        # square roots of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_sqrt = ht.sqrt(int32_tensor)
        self.assertIsInstance(int32_sqrt, ht.tensor)
        self.assertEqual(int32_sqrt.dtype, ht.float64)
        self.assertEqual(int32_sqrt.dtype, ht.float64)
        self.assertTrue(ht.allclose(int32_sqrt, comparison, 1e-06))

        # square roots of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_sqrt = ht.sqrt(int64_tensor)
        self.assertIsInstance(int64_sqrt, ht.tensor)
        self.assertEqual(int64_sqrt.dtype, ht.float64)
        self.assertEqual(int64_sqrt.dtype, ht.float64)
        self.assertTrue(ht.allclose(int64_sqrt, comparison, 1e-06))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.sqrt([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.sqrt('hello world')

    def test_sqrt_method(self):
        elements = 25
        comparison = ht.arange(elements, dtype=ht.float64).sqrt()

        # square roots of float32
        float32_sqrt = ht.arange(elements, dtype=ht.float32).sqrt()
        self.assertIsInstance(float32_sqrt, ht.tensor)
        self.assertEqual(float32_sqrt.dtype, ht.float32)
        self.assertEqual(float32_sqrt.dtype, ht.float32)
        self.assertTrue(ht.allclose(float32_sqrt, comparison.astype(ht.float32), 1e-05))

        # square roots of float64
        float64_sqrt = ht.arange(elements, dtype=ht.float64).sqrt()
        self.assertIsInstance(float64_sqrt, ht.tensor)
        self.assertEqual(float64_sqrt.dtype, ht.float64)
        self.assertEqual(float64_sqrt.dtype, ht.float64)
        self.assertTrue(ht.allclose(float64_sqrt, comparison, 1e-05))

        # square roots of ints, automatic conversion to intermediate floats
        int32_sqrt = ht.arange(elements, dtype=ht.int32).sqrt()
        self.assertIsInstance(int32_sqrt, ht.tensor)
        self.assertEqual(int32_sqrt.dtype, ht.float64)
        self.assertEqual(int32_sqrt.dtype, ht.float64)
        self.assertTrue(ht.allclose(int32_sqrt, comparison, 1e-05))

        # square roots of longs, automatic conversion to intermediate floats
        int64_sqrt = ht.arange(elements, dtype=ht.int64).sqrt()
        self.assertIsInstance(int64_sqrt, ht.tensor)
        self.assertEqual(int64_sqrt.dtype, ht.float64)
        self.assertEqual(int64_sqrt.dtype, ht.float64)
        self.assertTrue(ht.allclose(int64_sqrt, comparison, 1e-05))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.sqrt([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.sqrt('hello world')

    def test_sqrt_out_of_place(self):
        elements = 30
        output_shape = (3, elements)
        number_range = ht.arange(elements, dtype=ht.float32)
        output_buffer = ht.zeros(output_shape, dtype=ht.float32)

        # square roots
        float32_sqrt = ht.sqrt(number_range, out=output_buffer)
        comparison = torch.arange(elements, dtype=torch.float32).sqrt()

        # check whether the input range remain unchanged
        self.assertIsInstance(number_range, ht.tensor)
        self.assertEqual(number_range.sum(axis=0), 190)  # gaussian sum
        self.assertEqual(number_range.gshape, (elements,))

        # check whether the output buffer still has the correct shape
        self.assertIsInstance(float32_sqrt, ht.tensor)
        self.assertEqual(float32_sqrt.dtype, ht.float32)
        self.assertEqual(float32_sqrt._tensor__array.shape, output_shape)
        for row in range(output_shape[0]):
            self.assertTrue((float32_sqrt._tensor__array[row] == comparison).all())

        # exception
        with self.assertRaises(TypeError):
            ht.sqrt(number_range, 'hello world')
