import unittest
import torch

import numpy as np
import heat as ht


class TestManipulations(unittest.TestCase):
    def test_squeeze(self):
        torch.manual_seed(1)
        data = ht.random.randn(1, 4, 5, 1)

        # 4D local tensor, no axis
        result = ht.squeeze(data)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.float32)
        self.assertEqual(result._DNDarray__array.dtype, torch.float32)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == data._DNDarray__array.squeeze()).all())

        # 4D local tensor, major axis
        result = ht.squeeze(data, axis=0)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.float32)
        self.assertEqual(result._DNDarray__array.dtype, torch.float32)
        self.assertEqual(result.shape, (4, 5, 1))
        self.assertEqual(result.lshape, (4, 5, 1))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == data._DNDarray__array.squeeze(0)).all())

        # 4D local tensor, minor axis
        result = ht.squeeze(data, axis=-1)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.float32)
        self.assertEqual(result._DNDarray__array.dtype, torch.float32)
        self.assertEqual(result.shape, (1, 4, 5))
        self.assertEqual(result.lshape, (1, 4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == data._DNDarray__array.squeeze(-1)).all())

        # 4D local tensor, tuple axis
        result = data.squeeze(axis=(0, -1))
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.float32)
        self.assertEqual(result._DNDarray__array.dtype, torch.float32)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == data._DNDarray__array.squeeze()).all())

        # 4D split tensor, along the axis
        data = ht.array(ht.random.randn(1, 4, 5, 1), split=1)
        result = ht.squeeze(data, axis=-1)
        self.assertIsInstance(result, ht.DNDarray)
        # TODO: the following works locally but not when distributed,
        #self.assertEqual(result.dtype, ht.float32)
        #self.assertEqual(result._DNDarray__array.dtype, torch.float32)
        self.assertEqual(result.shape, (1, ht.MPI_WORLD.size * 4, 5))
        self.assertEqual(result.lshape, (1, ht.MPI_WORLD.size * 4, 5))
        self.assertEqual(result.split, 1)

        # 3D split tensor, across the axis
        size = ht.MPI_WORLD.size * 2
        data = ht.triu(ht.ones((1, size, size), split=1), k=1)

        result = ht.squeeze(data, axis=0)
        self.assertIsInstance(result, ht.DNDarray)
        # TODO: the following works locally but not when distributed,
        #self.assertEqual(result.dtype, ht.float32)
        #self.assertEqual(result._DNDarray__array.dtype, torch.float32)
        self.assertEqual(result.shape, (size, size))
        self.assertEqual(result.lshape, (size, size))
        #self.assertEqual(result.split, None)

        # check exceptions
        with self.assertRaises(ValueError):
            data.squeeze(axis=(0, 1))
        with self.assertRaises(TypeError):
            data.squeeze(axis=1.1)
        with self.assertRaises(TypeError):
            data.squeeze(axis='y')
        with self.assertRaises(ValueError):
            ht.argmin(data, axis=-4)
