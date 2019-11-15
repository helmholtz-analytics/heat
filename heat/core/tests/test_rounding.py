import torch
import unittest
import numpy as np
import heat as ht
import os

if os.environ.get("DEVICE") == "gpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ht.use_device("gpu" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
    ht.use_device("cpu")


class TestRounding(unittest.TestCase):
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
        self.assertIsInstance(absolute_values, ht.DNDarray)
        self.assertEqual(absolute_values.dtype, ht.float32)
        self.assertEqual(absolute_values.sum(axis=0), 100)
        # for fabs
        self.assertEqual(int8_absolute_values_fabs.sum(axis=0), 100.0)
        self.assertEqual(int16_absolute_values_fabs.sum(axis=0), 100.0)
        self.assertEqual(int32_absolute_values_fabs.sum(axis=0), 100.0)
        self.assertEqual(int64_absolute_values_fabs.sum(axis=0), 100.0)
        self.assertEqual(float32_absolute_values_fabs.sum(axis=0), 110.5)
        self.assertEqual(float64_absolute_values_fabs.sum(axis=0), 110.5)

        # check whether output works
        # for abs==absolute
        output_tensor = ht.zeros(20, split=0)
        self.assertEqual(output_tensor.sum(axis=0, keepdim=True), 0)
        ht.absolute(float32_tensor, out=output_tensor)

        self.assertEqual(output_tensor.sum(axis=0), 100)
        # for fabs
        output_tensor_fabs = ht.zeros(21, split=0)
        self.assertEqual(output_tensor_fabs.sum(axis=0), 0)
        ht.fabs(float32_tensor_fabs, out=output_tensor_fabs)
        self.assertEqual(output_tensor_fabs.sum(axis=0), 110.5)

        # dtype parameter
        # for abs==absolute
        int64_tensor = ht.arange(-10, 10, dtype=ht.int64)
        absolute_values = ht.abs(int64_tensor, dtype=ht.float32)
        self.assertIsInstance(absolute_values, ht.DNDarray)
        self.assertEqual(absolute_values.sum(axis=0), 100)
        self.assertEqual(absolute_values.dtype, ht.float32)
        self.assertEqual(absolute_values._DNDarray__array.dtype, torch.float32)
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
            ht.absolute("hello")
        with self.assertRaises(TypeError):
            float32_tensor.abs(out=1)
        with self.assertRaises(TypeError):
            float32_tensor.absolute(out=float32_tensor, dtype=3.2)
        # for fabs
        with self.assertRaises(TypeError):
            ht.fabs("hello")
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
        comparison = torch.arange(start, end, step, dtype=torch.float64, device=device).ceil()

        # exponential of float32
        float32_tensor = ht.arange(start, end, step, dtype=ht.float32)
        float32_floor = float32_tensor.ceil()
        self.assertIsInstance(float32_floor, ht.DNDarray)
        self.assertEqual(float32_floor.dtype, ht.float32)
        self.assertEqual(float32_floor.dtype, ht.float32)
        self.assertTrue((float32_floor._DNDarray__array == comparison.float()).all())

        # exponential of float64
        float64_tensor = ht.arange(start, end, step, dtype=ht.float64)
        float64_floor = float64_tensor.ceil()
        self.assertIsInstance(float64_floor, ht.DNDarray)
        self.assertEqual(float64_floor.dtype, ht.float64)
        self.assertEqual(float64_floor.dtype, ht.float64)
        self.assertTrue((float64_floor._DNDarray__array == comparison).all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.ceil([0, 1, 2, 3])
        with self.assertRaises(TypeError):
            ht.ceil(object())

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
        self.assertEqual(clipped.sum(axis=0), 194)

        # test the exceptions
        with self.assertRaises(TypeError):
            ht.clip(torch.arange(10, device=device), 2, 5)
        with self.assertRaises(ValueError):
            ht.arange(20).clip(None, None)
        with self.assertRaises(TypeError):
            ht.clip(ht.arange(20), 5, 15, out=torch.arange(20, device=device))

    def test_floor(self):
        start, end, step = -5.0, 5.0, 1.4
        comparison = torch.arange(start, end, step, dtype=torch.float64, device=device).floor()

        # exponential of float32
        float32_tensor = ht.arange(start, end, step, dtype=ht.float32)
        float32_floor = float32_tensor.floor()
        self.assertIsInstance(float32_floor, ht.DNDarray)
        self.assertEqual(float32_floor.dtype, ht.float32)
        self.assertEqual(float32_floor.dtype, ht.float32)
        self.assertTrue((float32_floor._DNDarray__array == comparison.float()).all())

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

    def test_trunc(self):
        base_array = np.random.randn(20)

        comparison = torch.tensor(base_array, dtype=torch.float64, device=device).trunc()

        # trunc of float32
        float32_tensor = ht.array(base_array, dtype=ht.float32)
        float32_floor = float32_tensor.trunc()
        self.assertIsInstance(float32_floor, ht.DNDarray)
        self.assertEqual(float32_floor.dtype, ht.float32)
        self.assertTrue((float32_floor._DNDarray__array == comparison.float()).all())

        # trunc of float64
        float64_tensor = ht.array(base_array, dtype=ht.float64)
        float64_floor = float64_tensor.trunc()
        self.assertIsInstance(float64_floor, ht.DNDarray)
        self.assertEqual(float64_floor.dtype, ht.float64)
        self.assertTrue((float64_floor._DNDarray__array == comparison).all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.trunc([0, 1, 2, 3])
        with self.assertRaises(TypeError):
            ht.trunc(object())
