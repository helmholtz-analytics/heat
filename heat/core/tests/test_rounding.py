import torch
import unittest

import heat as ht


class TestRounding(unittest.TestCase):
    def test_abs(self):
        float32_tensor = ht.arange(-10, 10, dtype=ht.float32, split=0)
        absolute_values = ht.abs(float32_tensor)

        # basic absolute test
        self.assertIsInstance(absolute_values, ht.DNDarray)
        self.assertEqual(absolute_values.dtype, ht.float32)
        self.assertEqual(absolute_values.sum(axis=0, keepdim=True), 100)

        # check whether output works
        output_tensor = ht.zeros(20, split=0)
        self.assertEqual(output_tensor.sum(axis=0, keepdim=True), 0)
        ht.absolute(float32_tensor, out=output_tensor)
        self.assertEqual(output_tensor.sum(axis=0, keepdim=True), 100)

        # dtype parameter
        int64_tensor = ht.arange(-10, 10, dtype=ht.int64)
        absolute_values = ht.abs(int64_tensor, dtype=ht.float32)
        self.assertIsInstance(absolute_values, ht.DNDarray)
        self.assertEqual(absolute_values.sum(axis=0), 100)
        self.assertEqual(absolute_values.dtype, ht.float32)
        self.assertEqual(absolute_values._DNDarray__array.dtype, torch.float32)

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
        self.assertIsInstance(float32_floor, ht.DNDarray)
        self.assertEqual(float32_floor.dtype, ht.float32)
        self.assertEqual(float32_floor.dtype, ht.float32)
        self.assertTrue((float32_floor._DNDarray__array == comparison.type(torch.float32)).all())

        # exponential of float64
        float64_tensor = ht.arange(start, end, step, dtype=ht.float64)
        float64_floor = float64_tensor.ceil()
        self.assertIsInstance(float64_floor, ht.DNDarray)
        self.assertEqual(float64_floor.dtype, ht.float64)
        self.assertEqual(float64_floor.dtype, ht.float64)
        self.assertTrue((float64_floor._DNDarray__array == comparison).all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.floor([0, 1, 2, 3])
        with self.assertRaises(TypeError):
            ht.floor(object())

    def test_clip(self):
        elements = 20

        # float tensor
        float32_tensor = ht.arange(elements, dtype=ht.float32, split=0)
        clipped = float32_tensor.clip(5, 15)
        self.assertIsInstance(clipped, ht.DNDarray)
        self.assertEqual(clipped.dtype, ht.float32)
        self.assertEqual(clipped.sum(axis=0), 195)

        # long tensor
        int64_tensor = ht.arange(elements, dtype=ht.int64, split=0)
        clipped = int64_tensor.clip(4, 16)
        self.assertIsInstance(clipped, ht.DNDarray)
        self.assertEqual(clipped.dtype, ht.int64)
        self.assertEqual(clipped.sum(axis=0), 195)

        # test the exceptions
        with self.assertRaises(TypeError):
            ht.clip(torch.arange(10), 2, 5)
        with self.assertRaises(ValueError):
            ht.arange(20).clip(None, None)
        with self.assertRaises(TypeError):
            ht.clip(ht.arange(20), 5, 15, out=torch.arange(20))

    def test_floor(self):
        start, end, step = -5.0, 5.0, 1.4
        comparison = torch.arange(start, end, step, dtype=torch.float64).floor()

        # exponential of float32
        float32_tensor = ht.arange(start, end, step, dtype=ht.float32)
        float32_floor = float32_tensor.floor()
        self.assertIsInstance(float32_floor, ht.DNDarray)
        self.assertEqual(float32_floor.dtype, ht.float32)
        self.assertEqual(float32_floor.dtype, ht.float32)
        self.assertTrue((float32_floor._DNDarray__array == comparison.type(torch.float32)).all())

        # exponential of float64
        float64_tensor = ht.arange(start, end, step, dtype=ht.float64)
        float64_floor = float64_tensor.floor()
        self.assertIsInstance(float64_floor, ht.DNDarray)
        self.assertEqual(float64_floor.dtype, ht.float64)
        self.assertEqual(float64_floor.dtype, ht.float64)
        self.assertTrue((float64_floor._DNDarray__array == comparison).all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.floor([0, 1, 2, 3])
        with self.assertRaises(TypeError):
            ht.floor(object())
