import unittest
import torch

import heat as ht

FLOAT_EPSILON = 1e-4


class TestOperations(unittest.TestCase):
    def test_abs(self):
        # for abs==absolute
        float32_tensor = ht.arange(-10, 10, dtype=ht.float32, split=0)
        absolute_values = ht.abs(float32_tensor)
        # for fabs
        int8_tensor_fabs = ht.arange(-10.5, 10.5, dtype=ht.int8, split=0)
        int8_absolute_values_fabs = ht.fabs(int8_tensor_fabs)
        int16_tensor_fabs = ht.arange(-10.5, 10.5, dtype=ht.int16, split=0)
        int16_absolute_values_fabs = ht.fabs(int16_tensor_fabs)
        int32_tensor_fabs = ht.arange(-10.5, 10.5, dtype=ht.int32, split=0)
        int32_absolute_values_fabs = ht.fabs(int32_tensor_fabs)
        int64_tensor_fabs = ht.arange(-10.5, 10.5, dtype=ht.int64, split=0)
        int64_absolute_values_fabs = ht.fabs(int64_tensor_fabs)
        float32_tensor_fabs = ht.arange(-10.5, 10.5, dtype=ht.float32, split=0)
        float32_absolute_values_fabs = ht.fabs(float32_tensor_fabs)
        float64_tensor_fabs = ht.arange(-10.5, 10.5, dtype=ht.float64, split=0)
        float64_absolute_values_fabs = ht.fabs(float64_tensor_fabs)

        # basic absolute test
        # for abs==absolute
        self.assertIsInstance(absolute_values, ht.tensor)
        self.assertEqual(absolute_values.dtype, ht.float32)
        self.assertEqual(absolute_values.sum(axis=0), 100)
        # for fabs
        self.assertEqual(int8_absolute_values_fabs.sum(axis=0), 110.5)
        self.assertEqual(int16_absolute_values_fabs.sum(axis=0), 110.5)
        self.assertEqual(int32_absolute_values_fabs.sum(axis=0), 110.5)
        self.assertEqual(int64_absolute_values_fabs.sum(axis=0), 110.5)
        self.assertEqual(float32_absolute_values_fabs.sum(axis=0), 110.5)
        self.assertEqual(float64_absolute_values_fabs.sum(axis=0), 110.5)

        # check whether output works
        # for abs==absolute
        output_tensor = ht.zeros(20, split=0)
        self.assertEqual(output_tensor.sum(axis=0), 0)
        ht.absolute(float32_tensor, out=output_tensor)
        self.assertEqual(output_tensor.sum(axis=0), 100)
        # for fabs
        output_tensor_fabs = ht.zeros(21, split=0)
        self.assertEqual(output_tensor_fabs.sum(axis=0), 0)
        ht.fabs(float32_tensor_fabs, out=output_tensor_fabs)
        self.assertEqual(output_tensor_fabs.sum(axis=0), 100)

        # dtype parameter
        # for abs==absolute
        int64_tensor = ht.arange(-10, 10, dtype=ht.int64)
        absolute_values = ht.abs(int64_tensor, dtype=ht.float32)
        self.assertIsInstance(absolute_values, ht.tensor)
        self.assertEqual(absolute_values.sum(axis=0), 100)
        self.assertEqual(absolute_values.dtype, ht.float32)
        self.assertEqual(absolute_values._tensor__array.dtype, torch.float32)
        # for fabs
        self.assertEqual(int8_absolute_values_fabs.dtype, ht.float32)
        self.assertEqual(int16_absolute_values_fabs.dtype, ht.float32)
        self.assertEqual(int32_absolute_values_fabs.dtype, ht.float64)
        self.assertEqual(int64_absolute_values_fabs.dtype, ht.float64)
        self.assertEqual(float32_absolute_values_fabs.dtype, ht.float32)
        self.assertEqual(float64_absolute_values_fabs.dtype, ht.float64)

        # exceptions
        # for abs==absolute
        with self.assertRaises(TypeError):
            ht.absolute('hello')
        with self.assertRaises(TypeError):
            float32_tensor.abs(out=1)
        with self.assertRaises(TypeError):
            float32_tensor.absolute(out=float32_tensor, dtype=3.2)
        # for fabs
        with self.assertRaises(TypeError):
            ht.fabs('hello')
        with self.assertRaises(TypeError):
            float32_tensor_fabs.fabs(out=1)

        # test with unsplit tensor
        # for fabs
        float32_unsplit_tensor_fabs = ht.arange(-10.5, 10.5, dtype=ht.float32)
        float32_unsplit_absolute_values_fabs = ht.fabs(float32_unsplit_tensor_fabs)
        self.assertEqual(float32_unsplit_absolute_values_fabs.sum(), 110.5)
        self.assertEqual(float32_unsplit_absolute_values_fabs.dtype, ht.float32)

    def test_ceil(self):
        start, end, step = -5.0, 5.0, 1.4
        comparison = torch.arange(start, end, step, dtype=torch.float64).ceil()

        # exponential of float32
        float32_tensor = ht.arange(start, end, step, dtype=ht.float32)
        float32_floor = float32_tensor.ceil()
        self.assertIsInstance(float32_floor, ht.tensor)
        self.assertEqual(float32_floor.dtype, ht.float32)
        self.assertEqual(float32_floor.dtype, ht.float32)
        self.assertTrue((float32_floor._tensor__array == comparison.type(torch.float32)).all())

        # exponential of float64
        float64_tensor = ht.arange(start, end, step, dtype=ht.float64)
        float64_floor = float64_tensor.ceil()
        self.assertIsInstance(float64_floor, ht.tensor)
        self.assertEqual(float64_floor.dtype, ht.float64)
        self.assertEqual(float64_floor.dtype, ht.float64)
        self.assertTrue((float64_floor._tensor__array == comparison).all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.floor([0, 1, 2, 3])
        with self.assertRaises(TypeError):
            ht.floor(object())

    def test_floor(self):
        start, end, step = -5.0, 5.0, 1.4
        comparison = torch.arange(start, end, step, dtype=torch.float64).floor()

        # exponential of float32
        float32_tensor = ht.arange(start, end, step, dtype=ht.float32)
        float32_floor = float32_tensor.floor()
        self.assertIsInstance(float32_floor, ht.tensor)
        self.assertEqual(float32_floor.dtype, ht.float32)
        self.assertEqual(float32_floor.dtype, ht.float32)
        self.assertTrue((float32_floor._tensor__array == comparison.type(torch.float32)).all())

        # exponential of float64
        float64_tensor = ht.arange(start, end, step, dtype=ht.float64)
        float64_floor = float64_tensor.floor()
        self.assertIsInstance(float64_floor, ht.tensor)
        self.assertEqual(float64_floor.dtype, ht.float64)
        self.assertEqual(float64_floor.dtype, ht.float64)
        self.assertTrue((float64_floor._tensor__array == comparison).all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.floor([0, 1, 2, 3])
        with self.assertRaises(TypeError):
            ht.floor(object())
