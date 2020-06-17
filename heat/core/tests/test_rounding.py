import torch
import unittest
import numpy as np
import heat as ht
import os
from .test_suites.basic_test import TestCase


class TestRounding(TestCase):
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
        self.assertEqual(int32_absolute_values_fabs.dtype, ht.float32)
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
        comparison = torch.arange(
            start, end, step, dtype=torch.float64, device=self.device.torch_device
        ).ceil()

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
            ht.clip(torch.arange(10, device=self.device.torch_device), 2, 5)
        with self.assertRaises(ValueError):
            ht.arange(20).clip(None, None)
        with self.assertRaises(TypeError):
            ht.clip(ht.arange(20), 5, 15, out=torch.arange(20, device=self.device.torch_device))

    def test_floor(self):
        start, end, step = -5.0, 5.0, 1.4
        comparison = torch.arange(
            start, end, step, dtype=torch.float64, device=self.device.torch_device
        ).floor()

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

    def test_modf(self):
        size = ht.communication.MPI_WORLD.size
        start, end = -5.0, 5.0
        step = (end - start) / (2 * size)
        npArray = np.arange(start, end, step, dtype=np.float32)
        comparison = np.modf(npArray)

        # exponential of float32
        float32_tensor = ht.array(npArray, dtype=ht.float32)
        float32_modf = float32_tensor.modf()
        self.assertIsInstance(float32_modf[0], ht.DNDarray)
        self.assertIsInstance(float32_modf[1], ht.DNDarray)
        self.assertEqual(float32_modf[0].dtype, ht.float32)
        self.assertEqual(float32_modf[1].dtype, ht.float32)

        self.assert_array_equal(float32_modf[0], comparison[0])
        self.assert_array_equal(float32_modf[1], comparison[1])

        # exponential of float64
        npArray = np.arange(start, end, step, np.float64)
        comparison = np.modf(npArray)

        float64_tensor = ht.array(npArray, dtype=ht.float64)
        float64_modf = float64_tensor.modf()
        self.assertIsInstance(float64_modf[0], ht.DNDarray)
        self.assertIsInstance(float64_modf[1], ht.DNDarray)
        self.assertEqual(float64_modf[0].dtype, ht.float64)
        self.assertEqual(float64_modf[1].dtype, ht.float64)

        self.assert_array_equal(float64_modf[0], comparison[0])
        self.assert_array_equal(float64_modf[1], comparison[1])

        # check exceptions
        with self.assertRaises(TypeError):
            ht.modf([0, 1, 2, 3])
        with self.assertRaises(TypeError):
            ht.modf(object())
        with self.assertRaises(TypeError):
            ht.modf(float32_tensor, 1)
        with self.assertRaises(ValueError):
            ht.modf(float32_tensor, (float32_tensor, float32_tensor, float64_tensor))
        with self.assertRaises(TypeError):
            ht.modf(float32_tensor, (float32_tensor, 2))

        # with split tensors

        # exponential of float32
        npArray = np.arange(start, end, step, dtype=np.float32)
        comparison = np.modf(npArray)
        float32_tensor_distrbd = ht.array(npArray, split=0)
        float32_modf_distrbd = float32_tensor_distrbd.modf()

        self.assertIsInstance(float32_modf_distrbd[0], ht.DNDarray)
        self.assertIsInstance(float32_modf_distrbd[1], ht.DNDarray)
        self.assertEqual(float32_modf_distrbd[0].dtype, ht.float32)
        self.assertEqual(float32_modf_distrbd[1].dtype, ht.float32)

        self.assert_array_equal(float32_modf_distrbd[0], comparison[0])
        self.assert_array_equal(float32_modf_distrbd[1], comparison[1])

        # exponential of float64
        npArray = npArray = np.arange(start, end, step, np.float64)
        comparison = np.modf(npArray)

        float64_tensor_distrbd = ht.array(npArray, split=0)
        float64_modf_distrbd = (
            ht.zeros_like(float64_tensor_distrbd, dtype=float64_tensor_distrbd.dtype),
            ht.zeros_like(float64_tensor_distrbd, dtype=float64_tensor_distrbd.dtype),
        )
        # float64_modf_distrbd = float64_tensor_distrbd.modf()
        float64_tensor_distrbd.modf(out=float64_modf_distrbd)
        self.assertIsInstance(float64_modf_distrbd[0], ht.DNDarray)
        self.assertIsInstance(float64_modf_distrbd[1], ht.DNDarray)
        self.assertEqual(float64_modf_distrbd[0].dtype, ht.float64)
        self.assertEqual(float64_modf_distrbd[1].dtype, ht.float64)

        self.assert_array_equal(float64_modf_distrbd[0], comparison[0])
        self.assert_array_equal(float64_modf_distrbd[1], comparison[1])

    def test_round(self):
        size = ht.communication.MPI_WORLD.size
        start, end = -5.7, 5.1
        step = (end - start) / (2 * size)
        comparison = torch.arange(start, end, step, dtype=torch.float32).round()

        # exponential of float32
        float32_tensor = ht.array(comparison, dtype=ht.float32)
        float32_round = float32_tensor.round()
        self.assertIsInstance(float32_round, ht.DNDarray)
        self.assertEqual(float32_round.dtype, ht.float32)
        self.assertEqual(float32_round.dtype, ht.float32)
        self.assert_array_equal(float32_round, comparison)

        # exponential of float64
        comparison = torch.arange(start, end, step, dtype=torch.float64).round()
        float64_tensor = ht.array(comparison, dtype=ht.float64)
        float64_round = float64_tensor.round()
        self.assertIsInstance(float64_round, ht.DNDarray)
        self.assertEqual(float64_round.dtype, ht.float64)
        self.assertEqual(float64_round.dtype, ht.float64)
        self.assert_array_equal(float64_round, comparison)

        # check exceptions
        with self.assertRaises(TypeError):
            ht.round([0, 1, 2, 3])
        with self.assertRaises(TypeError):
            ht.round(object())
        with self.assertRaises(TypeError):
            ht.round(float32_tensor, 1, 1)
        with self.assertRaises(TypeError):
            ht.round(float32_tensor, dtype=np.int)

        # with split tensors

        # exponential of float32
        comparison = torch.arange(start, end, step, dtype=torch.float32)  # .round()
        float32_tensor_distrbd = ht.array(comparison, split=0, dtype=ht.double)
        comparison = comparison.round()
        float32_round_distrbd = float32_tensor_distrbd.round(dtype=ht.float)
        self.assertIsInstance(float32_round_distrbd, ht.DNDarray)
        self.assertEqual(float32_round_distrbd.dtype, ht.float32)
        self.assert_array_equal(float32_round_distrbd, comparison)

        # exponential of float64
        comparison = torch.arange(start, end, step, dtype=torch.float64)  # .round()
        float64_tensor_distrbd = ht.array(comparison, split=0)
        comparison = comparison.round()
        float64_round_distrbd = float64_tensor_distrbd.round()
        self.assertIsInstance(float64_round_distrbd, ht.DNDarray)
        self.assertEqual(float64_round_distrbd.dtype, ht.float64)
        self.assertEqual(float64_round_distrbd.dtype, ht.float64)
        self.assert_array_equal(float64_round_distrbd, comparison)

    def test_trunc(self):
        base_array = np.random.randn(20)

        comparison = torch.tensor(
            base_array, dtype=torch.float64, device=self.device.torch_device
        ).trunc()

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
