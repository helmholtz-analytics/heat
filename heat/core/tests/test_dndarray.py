import numpy as np
import torch
import unittest

import heat as ht


class TestDNDarray(unittest.TestCase):
    def test_astype(self):
        data = ht.float32([
            [1, 2, 3],
            [4, 5, 6]
        ])

        # check starting invariant
        self.assertEqual(data.dtype, ht.float32)

        # check the copy case for uint8
        as_uint8 = data.astype(ht.uint8)
        self.assertIsInstance(as_uint8, ht.DNDarray)
        self.assertEqual(as_uint8.dtype, ht.uint8)
        self.assertEqual(as_uint8._DNDarray__array.dtype, torch.uint8)
        self.assertIsNot(as_uint8, data)

        # check the copy case for uint8
        as_float64 = data.astype(ht.float64, copy=False)
        self.assertIsInstance(as_float64, ht.DNDarray)
        self.assertEqual(as_float64.dtype, ht.float64)
        self.assertEqual(as_float64._DNDarray__array.dtype, torch.float64)
        self.assertIs(as_float64, data)

    def test_is_distributed(self):
        data = ht.zeros((5, 5,))
        self.assertFalse(data.is_distributed())

        data = ht.zeros((4, 4,), split=0)
        self.assertTrue(data.comm.size > 1 and data.is_distributed() or not data.is_distributed())

    def test_lloc(self):
        # single set
        a = ht.zeros((13, 5,), split=0)
        a.lloc[0, 0] = 1
        self.assertEqual(a._DNDarray__array[0, 0], 1)
        self.assertEqual(a.lloc[0, 0].dtype, torch.float32)

        # multiple set
        a = ht.zeros((13, 5,), split=0)
        a.lloc[1:3, 1] = 1
        self.assertTrue(all(a._DNDarray__array[1:3, 1] == 1))
        self.assertEqual(a.lloc[1:3, 1].dtype, torch.float32)

        # multiple set with specific indexing
        a = ht.zeros((13, 5,), split=0)
        a.lloc[3:7:2, 2:5:2] = 1
        self.assertTrue(torch.all(a._DNDarray__array[3:7:2, 2:5:2] == 1))
        self.assertEqual(a.lloc[3:7:2, 2:5:2].dtype, torch.float32)

    def test_resplit(self):
        # resplitting with same axis, should leave everything unchanged
        shape = (ht.MPI_WORLD.size, ht.MPI_WORLD.size,)
        data = ht.zeros(shape, split=None)
        data.resplit(None)

        self.assertIsInstance(data, ht.DNDarray)
        self.assertEqual(data.shape, shape)
        self.assertEqual(data.lshape, shape)
        self.assertEqual(data.split, None)

        # resplitting with same axis, should leave everything unchanged
        shape = (ht.MPI_WORLD.size, ht.MPI_WORLD.size,)
        data = ht.zeros(shape, split=1)
        data.resplit(1)

        self.assertIsInstance(data, ht.DNDarray)
        self.assertEqual(data.shape, shape)
        self.assertEqual(data.lshape, (data.comm.size, 1,))
        self.assertEqual(data.split, 1)

        # splitting an unsplit tensor should result in slicing the tensor locally
        shape = (ht.MPI_WORLD.size, ht.MPI_WORLD.size,)
        data = ht.zeros(shape)
        data.resplit(-1)

        self.assertIsInstance(data, ht.DNDarray)
        self.assertEqual(data.shape, shape)
        self.assertEqual(data.lshape, (data.comm.size, 1,))
        self.assertEqual(data.split, 1)

        # unsplitting, aka gathering a tensor
        shape = (ht.MPI_WORLD.size + 1, ht.MPI_WORLD.size,)
        data = ht.ones(shape, split=0)
        data.resplit(None)

        self.assertIsInstance(data, ht.DNDarray)
        self.assertEqual(data.shape, shape)
        self.assertEqual(data.lshape, shape)
        self.assertEqual(data.split, None)

        # assign and entirely new split axis
        shape = (ht.MPI_WORLD.size + 2, ht.MPI_WORLD.size + 1,)
        data = ht.ones(shape, split=0)
        data.resplit(1)

        self.assertIsInstance(data, ht.DNDarray)
        self.assertEqual(data.shape, shape)
        self.assertEqual(data.lshape[0], ht.MPI_WORLD.size + 2)
        self.assertTrue(data.lshape[1] == 1 or data.lshape[1] == 2)
        self.assertEqual(data.split, 1)

    def test_setitem_getitem(self):
        # set and get single value
        a = ht.zeros((13, 5,), split=0)
        # set value on one node
        a[10, 0] = 1
        self.assertEqual(a[10, 0], 1)
        self.assertEqual(a[10, 0].dtype, ht.float32)

        # slice in 1st dim only on 1 node
        a = ht.zeros((13, 5,), split=0)
        a[1:4] = 1
        self.assertEqual(a[1:4], 1)
        self.assertEqual(a[1:4].gshape, (3, 5))
        self.assertEqual(a[1:4].split, 0)
        self.assertEqual(a[1:4].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 0:
                self.assertEqual(a[1:4].lshape, (3, 5))
            else:
                self.assertEqual(a[1:4].lshape, (0,))

        # slice in 1st dim only on 1 node w/ singular second dim
        a = ht.zeros((13, 5,), split=0)
        a[1:4, 1] = 1
        b = a[1:4, 1]
        self.assertEqual(b, 1)
        self.assertEqual(b.gshape, (3,))
        self.assertEqual(b.split, 0)
        self.assertEqual(b.dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 0:
                self.assertEqual(b.lshape, (3,))
            else:
                self.assertEqual(b.lshape, (0,))

        # slice in 1st dim across both nodes (2 node case) w/ singular second dim
        a = ht.zeros((13, 5,), split=0)
        a[1:11, 1] = 1
        self.assertEqual(a[1:11, 1], 1)
        self.assertEqual(a[1:11, 1].gshape, (10,))
        self.assertEqual(a[1:11, 1].split, 0)
        self.assertEqual(a[1:11, 1].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 1:
                self.assertEqual(a[1:11, 1].lshape, (4,))
            if a.comm.rank == 0:
                self.assertEqual(a[1:11, 1].lshape, (6,))

        # slice in 1st dim across 1 node (2nd) w/ singular second dim
        c = ht.zeros((13, 5,), split=0)
        c[8:12, 1] = 1
        b = a[8:12, 1]
        self.assertEqual(b, 1)
        self.assertEqual(b.gshape, (4,))
        self.assertEqual(b.split, 0)
        self.assertEqual(b.dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 1:
                self.assertEqual(b.lshape, (4,))
            if a.comm.rank == 0:
                self.assertEqual(b.lshape, (0,))

        # slice in both directions
        a = ht.zeros((13, 5,), split=0)
        a[3:13, 2:5:2] = 1
        self.assertEqual(a[3:13, 2:5:2], 1)
        self.assertEqual(a[3:13, 2:5:2].gshape, (10, 2))
        self.assertEqual(a[3:13, 2:5:2].split, 0)
        self.assertEqual(a[3:13, 2:5:2].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 1:
                self.assertEqual(a[3:13, 2:5:2].lshape, (6, 2))
            if a.comm.rank == 0:
                self.assertEqual(a[3:13, 2:5:2].lshape, (4, 2))

        # setting with heat tensor
        a = ht.zeros((4, 5), split=0)
        a[1, 0:4] = ht.arange(4)
        # if a.comm.size == 2:
        for c, i in enumerate(range(4)):
            self.assertEqual(a[1, c], i)

        # setting with torch tensor
        a = ht.zeros((4, 5), split=0)
        a[1, 0:4] = torch.arange(4)
        # if a.comm.size == 2:
        for c, i in enumerate(range(4)):
            self.assertEqual(a[1, c], i)

        ####################################################
        a = ht.zeros((13, 5,), split=1)
        # # set value on one node
        a[10, :] = 1
        self.assertEqual(a[10, :].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 0:
                self.assertEqual(a[10, :].lshape, (3,))
            if a.comm.rank == 1:
                self.assertEqual(a[10, :].lshape, (2,))

        a = ht.zeros((13, 5,), split=1)
        # # set value on one node
        a[10, 0] = 1
        self.assertEqual(a[10, 0], 1)
        self.assertEqual(a[10, 0].dtype, ht.float32)

        # slice in 1st dim only on 1 node
        a = ht.zeros((13, 5), split=1)
        a[1:4] = 1
        self.assertEqual(a[1:4].gshape, (3, 5))
        self.assertEqual(a[1:4], 1)
        self.assertEqual(a[1:4].split, 1)
        self.assertEqual(a[1:4].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 0:
                self.assertEqual(a[1:4].lshape, (3, 3))
            if a.comm.rank == 1:
                self.assertEqual(a[1:4].lshape, (3, 2))

        # slice in 1st dim only on 1 node w/ singular second dim
        a = ht.zeros((13, 5,), split=1)
        a[1:4, 1] = 1
        self.assertEqual(a[1:4, 1], 1)
        self.assertEqual(a[1:4, 1].gshape, (3,))
        self.assertEqual(a[1:4, 1].split, 0)
        self.assertEqual(a[1:4, 1].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 0:
                self.assertEqual(a[1:4, 1].lshape, (3,))
            if a.comm.rank == 1:
                self.assertEqual(a[1:4, 1].lshape, (0,))

        # slice in 2st dim across both nodes (2 node case) w/ singular fist dim
        a = ht.zeros((13, 5,), split=1)
        a[11, 1:5] = 1
        self.assertEqual(a[11, 1:5], 1)
        self.assertEqual(a[11, 1:5].gshape, (4,))
        self.assertEqual(a[11, 1:5].split, 0)
        self.assertEqual(a[11, 1:5].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 1:
                self.assertEqual(a[11, 1:5].lshape, (2,))
            if a.comm.rank == 0:
                self.assertEqual(a[11, 1:5].lshape, (2,))

        # slice in 1st dim across 1 node (2nd) w/ singular second dim
        a = ht.zeros((13, 5,), split=1)
        a[8:12, 1] = 1
        self.assertEqual(a[8:12, 1], 1)
        self.assertEqual(a[8:12, 1].gshape, (4,))
        self.assertEqual(a[8:12, 1].split, 0)
        self.assertEqual(a[8:12, 1].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 0:
                self.assertEqual(a[8:12, 1].lshape, (4,))
            if a.comm.rank == 1:
                self.assertEqual(a[8:12, 1].lshape, (0,))

        # slice in both directions
        a = ht.zeros((13, 5,), split=1)
        a[3:13, 2:5:2] = 1
        self.assertEqual(a[3:13, 2:5:2], 1)
        self.assertEqual(a[3:13, 2:5:2].gshape, (10, 2))
        self.assertEqual(a[3:13, 2:5:2].split, 1)
        self.assertEqual(a[3:13, 2:5:2].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 1:
                self.assertEqual(a[3:13, 2:5:2].lshape, (10, 1))
            if a.comm.rank == 0:
                self.assertEqual(a[3:13, 2:5:2].lshape, (10, 1))

        # setting with heat tensor
        a = ht.zeros((4, 5), split=1)
        a[1, 0:4] = ht.arange(4)
        for c, i in enumerate(range(4)):
            self.assertEqual(a[1, c], i)

        # setting with torch tensor
        a = ht.zeros((4, 5), split=1)
        a[1, 0:4] = torch.arange(4)
        for c, i in enumerate(range(4)):
            self.assertEqual(a[1, c], i)

        ####################################################
        a = ht.zeros((13, 5, 7), split=2)
        # # set value on one node
        a[10, :, :] = 1
        self.assertEqual(a[10, :, :].dtype, ht.float32)
        self.assertEqual(a[10, :, :].gshape, (5, 7))
        if a.comm.size == 2:
            if a.comm.rank == 0:
                self.assertEqual(a[10, :, :].lshape, (5, 4))
            if a.comm.rank == 1:
                self.assertEqual(a[10, :, :].lshape, (5, 3))

        a = ht.zeros((13, 5, 8), split=2)
        # # set value on one node
        a[10, 0, 0] = 1
        self.assertEqual(a[10, 0, 0], 1)
        self.assertEqual(a[10, 0, 0].dtype, ht.float32)

        # # slice in 1st dim only on 1 node
        a = ht.zeros((13, 5, 7), split=2)
        a[1:4] = 1
        self.assertEqual(a[1:4].gshape, (3, 5, 7))
        self.assertEqual(a[1:4], 1)
        self.assertEqual(a[1:4].split, 2)
        self.assertEqual(a[1:4].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 0:
                self.assertEqual(a[1:4].lshape, (3, 5, 4))
            if a.comm.rank == 1:
                self.assertEqual(a[1:4].lshape, (3, 5, 3))

        # slice in 1st dim only on 1 node w/ singular second dim
        a = ht.zeros((13, 5, 7), split=2)
        a[1:4, 1, :] = 1
        self.assertEqual(a[1:4, 1, :].gshape, (3, 7))
        self.assertEqual(a[1:4, 1, :], 1)
        if a.comm.size == 2:
            self.assertEqual(a[1:4, 1, :].split, 1)
            self.assertEqual(a[1:4, 1, :].dtype, ht.float32)
            if a.comm.rank == 0:
                self.assertEqual(a[1:4, 1, :].lshape, (3, 4))
            if a.comm.rank == 1:
                self.assertEqual(a[1:4, 1, :].lshape, (3, 3))

        # slice in both directions
        a = ht.zeros((13, 5, 7), split=2)
        a[3:13, 2:5:2, 1:7:3] = 1
        self.assertEqual(a[3:13, 2:5:2, 1:7:3], 1)
        self.assertEqual(a[3:13, 2:5:2, 1:7:3].split, 2)
        self.assertEqual(a[3:13, 2:5:2, 1:7:3].dtype, ht.float32)
        self.assertEqual(a[3:13, 2:5:2, 1:7:3].gshape, (10, 2, 2))
        if a.comm.size == 2:
            if a.comm.rank == 1:
                self.assertEqual(a[3:13, 2:5:2, 1:7:3].lshape, (10, 2, 1))
            if a.comm.rank == 0:
                self.assertEqual(a[3:13, 2:5:2, 1:7:3].lshape, (10, 2, 1))

        a = ht.ones((4, 5,), split=0).tril()
        a[0] = [6, 6, 6, 6, 6]
        self.assertEqual(a[0], 6)

        a = ht.ones((4, 5,), split=0).tril()
        a[0] = (6, 6, 6, 6, 6)
        self.assertEqual(a[0], 6)

        a = ht.ones((4, 5,), split=0).tril()
        a[0] = np.array([6, 6, 6, 6, 6])
        self.assertEqual(a[0], 6)
