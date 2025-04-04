import numpy as np
import torch

import heat as ht
from .test_suites.basic_test import TestCase


class TestExponential(TestCase):
    def test_exp(self):
        elements = 10
        tmp = torch.arange(elements, dtype=torch.float64, device=self.device.torch_device).exp()
        comparison = ht.array(tmp)

        # exponential of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_exp = ht.exp(float32_tensor)
        self.assertIsInstance(float32_exp, ht.DNDarray)
        self.assertEqual(float32_exp.dtype, ht.float32)
        self.assertTrue(ht.allclose(float32_exp, comparison.astype(ht.float32)))

        # exponential of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_exp = ht.exp(float64_tensor)
        self.assertIsInstance(float64_exp, ht.DNDarray)
        self.assertEqual(float64_exp.dtype, ht.float64)
        self.assertTrue(ht.allclose(float64_exp, comparison))

        # exponential of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_exp = ht.exp(int32_tensor)
        self.assertIsInstance(int32_exp, ht.DNDarray)
        self.assertEqual(int32_exp.dtype, ht.float32)
        self.assertTrue(ht.allclose(int32_exp, ht.float32(comparison)))

        # exponential of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_exp = int64_tensor.exp()
        self.assertIsInstance(int64_exp, ht.DNDarray)
        self.assertEqual(int64_exp.dtype, ht.float64)
        self.assertTrue(ht.allclose(int64_exp, comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.exp([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.exp("hello world")

        # Tests with split
        expected = torch.arange(10, dtype=torch.float32, device=self.device.torch_device).exp()
        actual = ht.arange(10, split=0, dtype=ht.float32).exp()
        self.assertEqual(actual.gshape, tuple(expected.shape))
        self.assertEqual(actual.split, 0)
        actual = actual.resplit_(None)
        self.assertEqual(actual.lshape, expected.shape)
        self.assertTrue(torch.equal(expected, actual.larray))
        self.assertEqual(actual.dtype, ht.float32)

    def test_expm1(self):
        elements = 10
        tmp = torch.arange(elements, dtype=torch.float64, device=self.device.torch_device).expm1()
        comparison = ht.array(tmp)

        # expm1onential of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_expm1 = ht.expm1(float32_tensor)
        self.assertIsInstance(float32_expm1, ht.DNDarray)
        self.assertEqual(float32_expm1.dtype, ht.float32)
        self.assertTrue(ht.allclose(float32_expm1, comparison.astype(ht.float32)))

        # expm1onential of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_expm1 = ht.expm1(float64_tensor)
        self.assertIsInstance(float64_expm1, ht.DNDarray)
        self.assertEqual(float64_expm1.dtype, ht.float64)
        self.assertTrue(ht.allclose(float64_expm1, comparison))

        # expm1onential of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_expm1 = ht.expm1(int32_tensor)
        self.assertIsInstance(int32_expm1, ht.DNDarray)
        self.assertEqual(int32_expm1.dtype, ht.float32)
        self.assertTrue(ht.allclose(int32_expm1, ht.float32(comparison)))

        # expm1onential of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_expm1 = int64_tensor.expm1()
        self.assertIsInstance(int64_expm1, ht.DNDarray)
        self.assertEqual(int64_expm1.dtype, ht.float64)
        self.assertTrue(ht.allclose(int64_expm1, comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.expm1([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.expm1("hello world")

    def test_exp2(self):
        elements = 10
        tmp = np.exp2(torch.arange(elements, dtype=torch.float64))
        tmp = tmp.to(self.device.torch_device)
        comparison = ht.array(tmp, device=self.device)

        # exponential of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_exp2 = ht.exp2(float32_tensor)
        self.assertIsInstance(float32_exp2, ht.DNDarray)
        self.assertEqual(float32_exp2.dtype, ht.float32)
        self.assertTrue(ht.allclose(float32_exp2, comparison.astype(ht.float32)))

        # exponential of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_exp2 = ht.exp2(float64_tensor)
        self.assertIsInstance(float64_exp2, ht.DNDarray)
        self.assertEqual(float64_exp2.dtype, ht.float64)
        self.assertTrue(ht.allclose(float64_exp2, comparison))

        # exponential of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_exp2 = ht.exp2(int32_tensor)
        self.assertIsInstance(int32_exp2, ht.DNDarray)
        self.assertEqual(int32_exp2.dtype, ht.float32)
        self.assertTrue(ht.allclose(int32_exp2, ht.float32(comparison)))

        # exponential of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_exp2 = int64_tensor.exp2()
        self.assertIsInstance(int64_exp2, ht.DNDarray)
        self.assertEqual(int64_exp2.dtype, ht.float64)
        self.assertTrue(ht.allclose(int64_exp2, comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.exp2([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.exp2("hello world")

    def test_log(self):
        elements = 15
        tmp = torch.arange(1, elements, dtype=torch.float64, device=self.device.torch_device).log()
        comparison = ht.array(tmp)

        # logarithm of float32
        float32_tensor = ht.arange(1, elements, dtype=ht.float32)
        float32_log = ht.log(float32_tensor)
        self.assertIsInstance(float32_log, ht.DNDarray)
        self.assertEqual(float32_log.dtype, ht.float32)
        self.assertTrue(ht.allclose(float32_log, comparison.astype(ht.float32)))

        # logarithm of float64
        float64_tensor = ht.arange(1, elements, dtype=ht.float64)
        float64_log = ht.log(float64_tensor)
        self.assertIsInstance(float64_log, ht.DNDarray)
        self.assertEqual(float64_log.dtype, ht.float64)
        self.assertTrue(ht.allclose(float64_log, comparison))

        # logarithm of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(1, elements, dtype=ht.int32)
        int32_log = ht.log(int32_tensor)
        self.assertIsInstance(int32_log, ht.DNDarray)
        self.assertEqual(int32_log.dtype, ht.float32)
        self.assertTrue(ht.allclose(int32_log, ht.float32(comparison)))

        # logarithm of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(1, elements, dtype=ht.int64)
        int64_log = int64_tensor.log()
        self.assertIsInstance(int64_log, ht.DNDarray)
        self.assertEqual(int64_log.dtype, ht.float64)
        self.assertTrue(ht.allclose(int64_log, comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.log([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.log("hello world")

    def test_log2(self):
        elements = 15
        tmp = torch.arange(1, elements, dtype=torch.float64, device=self.device.torch_device).log2()
        comparison = ht.array(tmp)

        # logarithm of float32
        float32_tensor = ht.arange(1, elements, dtype=ht.float32)
        float32_log2 = ht.log2(float32_tensor)
        self.assertIsInstance(float32_log2, ht.DNDarray)
        self.assertEqual(float32_log2.dtype, ht.float32)
        self.assertTrue(ht.allclose(float32_log2, comparison.astype(ht.float32)))

        # logarithm of float64
        float64_tensor = ht.arange(1, elements, dtype=ht.float64)
        float64_log2 = ht.log2(float64_tensor)
        self.assertIsInstance(float64_log2, ht.DNDarray)
        self.assertEqual(float64_log2.dtype, ht.float64)
        self.assertTrue(ht.allclose(float64_log2, comparison))

        # logarithm of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(1, elements, dtype=ht.int32)
        int32_log2 = ht.log2(int32_tensor)
        self.assertIsInstance(int32_log2, ht.DNDarray)
        self.assertEqual(int32_log2.dtype, ht.float32)
        self.assertTrue(ht.allclose(int32_log2, ht.float32(comparison)))

        # logarithm of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(1, elements, dtype=ht.int64)
        int64_log2 = int64_tensor.log2()
        self.assertIsInstance(int64_log2, ht.DNDarray)
        self.assertEqual(int64_log2.dtype, ht.float64)
        self.assertTrue(ht.allclose(int64_log2, comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.log2([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.log2("hello world")

    def test_log10(self):
        elements = 15
        tmp = torch.arange(
            1, elements, dtype=torch.float64, device=self.device.torch_device
        ).log10()
        comparison = ht.array(tmp)

        # logarithm of float32
        float32_tensor = ht.arange(1, elements, dtype=ht.float32)
        float32_log10 = ht.log10(float32_tensor)
        self.assertIsInstance(float32_log10, ht.DNDarray)
        self.assertEqual(float32_log10.dtype, ht.float32)
        self.assertTrue(ht.allclose(float32_log10, comparison.astype(ht.float32)))

        # logarithm of float64
        float64_tensor = ht.arange(1, elements, dtype=ht.float64)
        float64_log10 = ht.log10(float64_tensor)
        self.assertIsInstance(float64_log10, ht.DNDarray)
        self.assertEqual(float64_log10.dtype, ht.float64)
        self.assertTrue(ht.allclose(float64_log10, comparison))

        # logarithm of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(1, elements, dtype=ht.int32)
        int32_log10 = ht.log10(int32_tensor)
        self.assertIsInstance(int32_log10, ht.DNDarray)
        self.assertEqual(int32_log10.dtype, ht.float32)
        self.assertTrue(ht.allclose(int32_log10, ht.float32(comparison)))

        # logarithm of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(1, elements, dtype=ht.int64)
        int64_log10 = int64_tensor.log10()
        self.assertIsInstance(int64_log10, ht.DNDarray)
        self.assertEqual(int64_log10.dtype, ht.float64)
        self.assertTrue(ht.allclose(int64_log10, comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.log10([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.log10("hello world")

    def test_log1p(self):
        elements = 15
        tmp = torch.arange(
            1, elements, dtype=torch.float64, device=self.device.torch_device
        ).log1p()
        comparison = ht.array(tmp)

        # logarithm of float32
        float32_tensor = ht.arange(1, elements, dtype=ht.float32)
        float32_log1p = ht.log1p(float32_tensor)
        self.assertIsInstance(float32_log1p, ht.DNDarray)
        self.assertEqual(float32_log1p.dtype, ht.float32)
        self.assertTrue(ht.allclose(float32_log1p, comparison.astype(ht.float32)))

        # logarithm of float64
        float64_tensor = ht.arange(1, elements, dtype=ht.float64)
        float64_log1p = ht.log1p(float64_tensor)
        self.assertIsInstance(float64_log1p, ht.DNDarray)
        self.assertEqual(float64_log1p.dtype, ht.float64)
        self.assertTrue(ht.allclose(float64_log1p, comparison))

        # logarithm of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(1, elements, dtype=ht.int32)
        int32_log1p = ht.log1p(int32_tensor)
        self.assertIsInstance(int32_log1p, ht.DNDarray)
        self.assertEqual(int32_log1p.dtype, ht.float32)
        self.assertTrue(ht.allclose(int32_log1p, ht.float32(comparison)))

        # logarithm of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(1, elements, dtype=ht.int64)
        int64_log1p = int64_tensor.log1p()
        self.assertIsInstance(int64_log1p, ht.DNDarray)
        self.assertEqual(int64_log1p.dtype, ht.float64)
        self.assertTrue(ht.allclose(int64_log1p, comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.log1p([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.log1p("hello world")

    def test_logaddexp(self):
        elements = 15
        tmp = torch.arange(1, elements, dtype=torch.float64, device=self.device.torch_device)
        tmp = tmp.logaddexp(tmp)
        comparison = ht.array(tmp)

        # logaddexp of float32
        float32_tensor = ht.arange(1, elements, dtype=ht.float32)
        float32_logaddexp = ht.logaddexp(float32_tensor, float32_tensor)
        self.assertIsInstance(float32_logaddexp, ht.DNDarray)
        self.assertEqual(float32_logaddexp.dtype, ht.float32)
        self.assertTrue(ht.allclose(float32_logaddexp, comparison.astype(ht.float32)))

        # logaddexp of float64
        float64_tensor = ht.arange(1, elements, dtype=ht.float64)
        float64_logaddexp = ht.logaddexp(float64_tensor, float64_tensor)
        self.assertIsInstance(float64_logaddexp, ht.DNDarray)
        self.assertEqual(float64_logaddexp.dtype, ht.float64)
        self.assertTrue(ht.allclose(float64_logaddexp, comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.logaddexp([1, 2, 3], [1, 2, 3])
        with self.assertRaises(TypeError):
            ht.logaddexp("hello world", "hello world")

    def test_logaddexp2(self):
        elements = 15
        tmp = torch.arange(1, elements, dtype=torch.float64, device=self.device.torch_device)
        tmp = tmp.logaddexp2(tmp)
        comparison = ht.array(tmp)

        # logaddexp2 of float32
        float32_tensor = ht.arange(1, elements, dtype=ht.float32)
        float32_logaddexp2 = ht.logaddexp2(float32_tensor, float32_tensor)
        self.assertIsInstance(float32_logaddexp2, ht.DNDarray)
        self.assertEqual(float32_logaddexp2.dtype, ht.float32)
        self.assertTrue(ht.allclose(float32_logaddexp2, comparison.astype(ht.float32)))

        # logaddexp2 of float64
        float64_tensor = ht.arange(1, elements, dtype=ht.float64)
        float64_logaddexp2 = ht.logaddexp2(float64_tensor, float64_tensor)
        self.assertIsInstance(float64_logaddexp2, ht.DNDarray)
        self.assertEqual(float64_logaddexp2.dtype, ht.float64)
        self.assertTrue(ht.allclose(float64_logaddexp2, comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.logaddexp2([1, 2, 3], [1, 2, 3])
        with self.assertRaises(TypeError):
            ht.logaddexp2("hello world", "hello world")

    def test_sqrt(self):
        elements = 25
        tmp = torch.arange(elements, dtype=torch.float64, device=self.device.torch_device).sqrt()
        comparison = ht.array(tmp)

        # square roots of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_sqrt = ht.sqrt(float32_tensor)
        self.assertIsInstance(float32_sqrt, ht.DNDarray)
        self.assertEqual(float32_sqrt.dtype, ht.float32)
        self.assertTrue(ht.allclose(float32_sqrt, comparison.astype(ht.float32), 1e-06))

        # square roots of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_sqrt = ht.sqrt(float64_tensor)
        self.assertIsInstance(float64_sqrt, ht.DNDarray)
        self.assertEqual(float64_sqrt.dtype, ht.float64)
        self.assertTrue(ht.allclose(float64_sqrt, comparison, 1e-06))

        # square roots of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_sqrt = ht.sqrt(int32_tensor)
        self.assertIsInstance(int32_sqrt, ht.DNDarray)
        self.assertEqual(int32_sqrt.dtype, ht.float32)
        self.assertTrue(ht.allclose(int32_sqrt, ht.float32(comparison), 1e-06))

        # square roots of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_sqrt = int64_tensor.sqrt()
        self.assertIsInstance(int64_sqrt, ht.DNDarray)
        self.assertEqual(int64_sqrt.dtype, ht.float64)
        self.assertTrue(ht.allclose(int64_sqrt, comparison, 1e-06))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.sqrt([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.sqrt("hello world")

    def test_sqrt_method(self):
        elements = 25
        tmp = torch.arange(elements, dtype=torch.float64, device=self.device.torch_device).sqrt()
        comparison = ht.array(tmp)

        # square roots of float32
        float32_sqrt = ht.arange(elements, dtype=ht.float32).sqrt()
        self.assertIsInstance(float32_sqrt, ht.DNDarray)
        self.assertEqual(float32_sqrt.dtype, ht.float32)
        self.assertTrue(ht.allclose(float32_sqrt, comparison.astype(ht.float32), 1e-05))

        # square roots of float64
        float64_sqrt = ht.arange(elements, dtype=ht.float64).sqrt()
        self.assertIsInstance(float64_sqrt, ht.DNDarray)
        self.assertEqual(float64_sqrt.dtype, ht.float64)
        self.assertTrue(ht.allclose(float64_sqrt, comparison, 1e-05))

        # square roots of ints, automatic conversion to intermediate floats
        int32_sqrt = ht.arange(elements, dtype=ht.int32).sqrt()
        self.assertIsInstance(int32_sqrt, ht.DNDarray)
        self.assertEqual(int32_sqrt.dtype, ht.float32)
        self.assertTrue(ht.allclose(int32_sqrt, ht.float32(comparison), 1e-05))

        # square roots of longs, automatic conversion to intermediate floats
        int64_sqrt = ht.arange(elements, dtype=ht.int64).sqrt()
        self.assertIsInstance(int64_sqrt, ht.DNDarray)
        self.assertEqual(int64_sqrt.dtype, ht.float64)
        self.assertTrue(ht.allclose(int64_sqrt, comparison, 1e-05))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.sqrt([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.sqrt("hello world")

    def test_sqrt_out_of_place(self):
        elements = 30
        output_shape = (3, elements)
        number_range = ht.arange(elements, dtype=ht.float32)
        output_buffer = ht.zeros(output_shape, dtype=ht.float32)

        # square roots
        float32_sqrt = ht.sqrt(number_range, out=output_buffer)
        comparison = torch.arange(
            elements, dtype=torch.float32, device=self.device.torch_device
        ).sqrt()

        # check whether the input range remain unchanged
        self.assertIsInstance(number_range, ht.DNDarray)
        self.assertEqual(number_range.sum(axis=0), 435)  # gaussian sum
        self.assertEqual(number_range.gshape, (elements,))

        # check whether the output buffer still has the correct shape
        self.assertIsInstance(float32_sqrt, ht.DNDarray)
        self.assertEqual(float32_sqrt.dtype, ht.float32)
        self.assertEqual(float32_sqrt.larray.shape, output_shape)
        for row in range(output_shape[0]):
            self.assertTrue((float32_sqrt.larray[row] == comparison).all())

        # exception
        with self.assertRaises(TypeError):
            ht.sqrt(number_range, "hello world")

    def test_square(self):
        elements = 25
        tmp = torch.square(
            torch.arange(elements, dtype=torch.float64, device=self.device.torch_device)
        )
        comparison = ht.array(tmp)

        # squares of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_square = ht.square(float32_tensor)
        self.assertIsInstance(float32_square, ht.DNDarray)
        self.assertEqual(float32_square.dtype, ht.float32)
        self.assertTrue(ht.allclose(float32_square, comparison.astype(ht.float32), 1e-09))

        # squares of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_square = ht.square(float64_tensor)
        self.assertIsInstance(float64_square, ht.DNDarray)
        self.assertEqual(float64_square.dtype, ht.float64)
        self.assertTrue(ht.allclose(float64_square, comparison, 1e-09))

        # squares of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_square = ht.square(int32_tensor)
        self.assertIsInstance(int32_square, ht.DNDarray)
        self.assertEqual(int32_square.dtype, ht.float32)
        self.assertTrue(ht.allclose(int32_square, ht.float32(comparison), 1e-09))

        # squares of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_square = int64_tensor.square()
        self.assertIsInstance(int64_square, ht.DNDarray)
        self.assertEqual(int64_square.dtype, ht.float64)
        self.assertTrue(ht.allclose(int64_square, comparison, 1e-09))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.square([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.square("hello world")
