import unittest

import torch

import heat as ht


class TestManipulation(unittest.TestCase):
    def test_expand_dims(self):
        # vector data
        a = ht.arange(10)
        b = ht.expand_dims(a, 0)

        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(len(b.shape), 2)

        self.assertEqual(b.shape[0], 1)
        self.assertEqual(b.shape[1], a.shape[0])

        self.assertEqual(b.lshape[0], 1)
        self.assertEqual(b.lshape[1], a.shape[0])

        self.assertIs(b.split, None)

        # vector data with out-of-bounds axis
        a = ht.arange(12)
        b = a.expand_dims(1)

        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(len(b.shape), 2)

        self.assertEqual(b.shape[0], a.shape[0])
        self.assertEqual(b.shape[1], 1)

        self.assertEqual(b.lshape[0], a.shape[0])
        self.assertEqual(b.lshape[1], 1)

        self.assertIs(b.split, None)

        # volume with intermediate axis
        a = ht.empty((3, 4, 5,))
        b = a.expand_dims(1)

        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(len(b.shape), 4)

        self.assertEqual(b.shape[0], a.shape[0])
        self.assertEqual(b.shape[1], 1)
        self.assertEqual(b.shape[2], a.shape[1])
        self.assertEqual(b.shape[3], a.shape[2])

        self.assertEqual(b.lshape[0], a.shape[0])
        self.assertEqual(b.lshape[1], 1)
        self.assertEqual(b.lshape[2], a.shape[1])
        self.assertEqual(b.lshape[3], a.shape[2])

        self.assertIs(b.split, None)

        # volume with negative axis
        a = ht.empty((3, 4, 5,))
        b = a.expand_dims(-4)

        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(len(b.shape), 4)

        self.assertEqual(b.shape[0], 1)
        self.assertEqual(b.shape[1], a.shape[0])
        self.assertEqual(b.shape[2], a.shape[1])
        self.assertEqual(b.shape[3], a.shape[2])

        self.assertEqual(b.lshape[0], 1)
        self.assertEqual(b.lshape[1], a.shape[0])
        self.assertEqual(b.lshape[2], a.shape[1])
        self.assertEqual(b.lshape[3], a.shape[2])

        self.assertIs(b.split, None)

        # split volume with negative axis expansion after the split
        a = ht.empty((3, 4, 5,), split=1)
        b = a.expand_dims(-2)

        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(len(b.shape), 4)

        self.assertEqual(b.shape[0], a.shape[0])
        self.assertEqual(b.shape[1], a.shape[1])
        self.assertEqual(b.shape[2], 1)
        self.assertEqual(b.shape[3], a.shape[2])

        self.assertEqual(b.lshape[0], a.shape[0])
        self.assertLessEqual(b.lshape[1], a.shape[1])
        self.assertEqual(b.lshape[2], 1)
        self.assertEqual(b.lshape[3], a.shape[2])

        self.assertIs(b.split, 1)

        # split volume with negative axis expansion before the split
        a = ht.empty((3, 4, 5,), split=2)
        b = a.expand_dims(-3)

        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(len(b.shape), 4)

        self.assertEqual(b.shape[0], a.shape[0])
        self.assertEqual(b.shape[1], 1)
        self.assertEqual(b.shape[2], a.shape[1])
        self.assertEqual(b.shape[3], a.shape[2])

        self.assertEqual(b.lshape[0], a.shape[0])
        self.assertEqual(b.lshape[1], 1)
        self.assertEqual(b.lshape[2], a.shape[1])
        self.assertLessEqual(b.lshape[3], a.shape[2])

        self.assertIs(b.split, 3)

        # exceptions
        with self.assertRaises(TypeError):
            ht.expand_dims('(3, 4, 5,)', 1)
        with self.assertRaises(TypeError):
            ht.empty((3, 4, 5,)).expand_dims('1')
        with self.assertRaises(ValueError):
            ht.empty((3, 4, 5,)).expand_dims(4)
        with self.assertRaises(ValueError):
            ht.empty((3, 4, 5,)).expand_dims(-5)

    def test_unique(self):
        t_split_zero = ht.array([
            [1, 3, 1],
            [1, 3, 1],
            [1, 2, 3],
            [1, 3, 1]
        ]).resplit(axis=0)
        exp_axis_none = ht.array([1, 2, 3])
        exp_axis_zero = ht.array([[1, 2, 3], [1, 3, 1]])
        exp_axis_one = ht.array([[1, 3, 1], [1, 3, 1], [1, 2, 3], [1, 3, 1]])

        res = ht.unique(t_split_zero, sorted=True)
        print("res", res)

        self.assertTrue(ht.equal(res, exp_axis_none))

        res = ht.unique(t_split_zero, sorted=True, axis=0)
        print("res", res)

        self.assertTrue(ht.equal(res, exp_axis_zero))

        res = ht.unique(t_split_zero, sorted=False, axis=1)
        print("res", res)

        self.assertTrue(ht.equal(res, exp_axis_one))


        t_split_one = ht.array([
            [1, 3, 4, 3],
            [1, 1, 2, 1]
        ], dtype=ht.int32).resplit(axis=1)
        exp_axis_none = ht.array([1, 2, 3, 4], dtype=ht.int32)
        exp_axis_zero = ht.array([[1, 3, 4, 3], [1, 1, 2, 1]], dtype=ht.int32)
        exp_axis_one = ht.array([[1, 3, 4], [1, 1, 2]], dtype=ht.int32)

        res = ht.unique(t_split_one, sorted=True)
        print("res", res)

        self.assertTrue(ht.equal(res, exp_axis_none))

        # TODO: Allgatherv with matrix that is split along axis 1 weirdly transposes the result

        res = ht.unique(t_split_one, sorted=False, axis=0)
        print("res", res)

        self.assertTrue(ht.equal(res, exp_axis_zero))

        res = ht.unique(t_split_one, sorted=True, axis=1)
        print("res", res)
        self.assertTrue(ht.equal(res, exp_axis_one))

    def test_unique_multi(self):
        size = ht.MPI_WORLD.size
        rank = ht.MPI_WORLD.rank
        torch_array = torch.arange(size).expand(size, size)
        print("torch_array", torch_array)
        split_zero = ht.array(torch_array, split=0)
        print("split_zero", split_zero)

        exp_axis_none = ht.arange(size)
        res = ht.unique(split_zero, sorted=True)
        print("res", res)
        self.assertTrue(ht.equal(res, exp_axis_none))

        exp_axis_zero = ht.arange(size)
        res = ht.unique(split_zero, sorted=True, axis=0)
        print("res", res, "expected", exp_axis_zero)
        self.assertTrue(ht.equal(res, exp_axis_zero))
