import unittest
import torch

import heat as ht

FLOAT_EPSILON = 1e-4


class TestOperations(unittest.TestCase):
    def test_abs(self):
        float32_tensor = ht.arange(-10, 10, dtype=ht.float32, split=0)
        absolute_values = ht.abs(float32_tensor)

        # basic absolute test
        self.assertIsInstance(absolute_values, ht.tensor)
        self.assertEqual(absolute_values.dtype, ht.float32)
        self.assertEqual(absolute_values.sum(axis=0), 100)

        # check whether output works
        output_tensor = ht.zeros(20, split=0)
        self.assertEqual(output_tensor.sum(axis=0), 0)
        ht.absolute(float32_tensor, out=output_tensor)
        self.assertEqual(output_tensor.sum(axis=0), 100)

        # dtype parameter
        int64_tensor = ht.arange(-10, 10, dtype=ht.int64)
        absolute_values = ht.abs(int64_tensor, dtype=ht.float32)
        self.assertIsInstance(absolute_values, ht.tensor)
        self.assertEqual(absolute_values.sum(axis=0), 100)
        self.assertEqual(absolute_values.dtype, ht.float32)
        self.assertEqual(absolute_values._tensor__array.dtype, torch.float32)

        # exceptions
        with self.assertRaises(TypeError):
            ht.absolute('hello')
        with self.assertRaises(TypeError):
            float32_tensor.abs(out=1)
        with self.assertRaises(TypeError):
            float32_tensor.absolute(out=float32_tensor, dtype=3.2)

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
