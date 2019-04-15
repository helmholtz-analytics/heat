import torch
import unittest

import heat as ht


class TestTensor(unittest.TestCase):
    def test_astype(self):
        data = ht.float32([
            [1, 2, 3],
            [4, 5, 6]
        ])

        # check starting invariant
        self.assertEqual(data.dtype, ht.float32)

        # check the copy case for uint8
        as_uint8 = data.astype(ht.uint8)
        self.assertIsInstance(as_uint8, ht.Tensor)
        self.assertEqual(as_uint8.dtype, ht.uint8)
        self.assertEqual(as_uint8._Tensor__array.dtype, torch.uint8)
        self.assertIsNot(as_uint8, data)

        # check the copy case for uint8
        as_float64 = data.astype(ht.float64, copy=False)
        self.assertIsInstance(as_float64, ht.Tensor)
        self.assertEqual(as_float64.dtype, ht.float64)
        self.assertEqual(as_float64._Tensor__array.dtype, torch.float64)
        self.assertIs(as_float64, data)

    def test_is_distributed(self):
        data = ht.zeros((5, 5,))
        self.assertFalse(data.is_distributed())

        data = ht.zeros((4, 4,), split=0)
        self.assertTrue(data.comm.size > 1 and data.is_distributed() or not data.is_distributed())

    def test_resplit(self):
        # resplitting with same axis, should leave everything unchanged
        shape = (ht.MPI_WORLD.size, ht.MPI_WORLD.size,)
        data = ht.zeros(shape, split=None)
        data.resplit(None)

        self.assertIsInstance(data, ht.Tensor)
        self.assertEqual(data.shape, shape)
        self.assertEqual(data.lshape, shape)
        self.assertEqual(data.split, None)

        # resplitting with same axis, should leave everything unchanged
        shape = (ht.MPI_WORLD.size, ht.MPI_WORLD.size,)
        data = ht.zeros(shape, split=1)
        data.resplit(1)

        self.assertIsInstance(data, ht.Tensor)
        self.assertEqual(data.shape, shape)
        self.assertEqual(data.lshape, (data.comm.size, 1,))
        self.assertEqual(data.split, 1)

        # splitting an unsplit tensor should result in slicing the tensor locally
        shape = (ht.MPI_WORLD.size, ht.MPI_WORLD.size,)
        data = ht.zeros(shape)
        data.resplit(-1)

        self.assertIsInstance(data, ht.Tensor)
        self.assertEqual(data.shape, shape)
        self.assertEqual(data.lshape, (data.comm.size, 1,))
        self.assertEqual(data.split, 1)

        # unsplitting, aka gathering a tensor
        shape = (ht.MPI_WORLD.size + 1, ht.MPI_WORLD.size,)
        data = ht.ones(shape, split=0)
        data.resplit(None)

        self.assertIsInstance(data, ht.Tensor)
        self.assertEqual(data.shape, shape)
        self.assertEqual(data.lshape, shape)
        self.assertEqual(data.split, None)

        # assign and entirely new split axis
        shape = (ht.MPI_WORLD.size + 2, ht.MPI_WORLD.size + 1,)
        data = ht.ones(shape, split=0)
        data.resplit(1)

        self.assertIsInstance(data, ht.Tensor)
        self.assertEqual(data.shape, shape)
        self.assertEqual(data.lshape[0], ht.MPI_WORLD.size + 2)
        self.assertTrue(data.lshape[1] == 1 or data.lshape[1] == 2)
        self.assertEqual(data.split, 1)
