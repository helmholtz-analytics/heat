import numpy as np
import torch

import heat as ht
from .test_suites.basic_test import TestCase


class TestFactories(TestCase):
    def test_arange(self):
        # testing one positional integer argument
        one_arg_arange_int = ht.arange(10)
        self.assertIsInstance(one_arg_arange_int, ht.DNDarray)
        self.assertEqual(one_arg_arange_int.shape, (10,))
        self.assertLessEqual(one_arg_arange_int.lshape[0], 10)
        self.assertEqual(one_arg_arange_int.dtype, ht.int32)
        self.assertEqual(one_arg_arange_int.larray.dtype, torch.int32)
        self.assertEqual(one_arg_arange_int.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(one_arg_arange_int.sum(), 45)

        # testing one positional float argument
        one_arg_arange_float = ht.arange(10.0)
        self.assertIsInstance(one_arg_arange_float, ht.DNDarray)
        self.assertEqual(one_arg_arange_float.shape, (10,))
        self.assertLessEqual(one_arg_arange_float.lshape[0], 10)
        self.assertEqual(one_arg_arange_float.dtype, ht.float32)
        self.assertEqual(one_arg_arange_float.larray.dtype, torch.float32)
        self.assertEqual(one_arg_arange_float.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(one_arg_arange_float.sum(), 45.0)

        # testing two positional integer arguments
        two_arg_arange_int = ht.arange(0, 10)
        self.assertIsInstance(two_arg_arange_int, ht.DNDarray)
        self.assertEqual(two_arg_arange_int.shape, (10,))
        self.assertLessEqual(two_arg_arange_int.lshape[0], 10)
        self.assertEqual(two_arg_arange_int.dtype, ht.int32)
        self.assertEqual(two_arg_arange_int.larray.dtype, torch.int32)
        self.assertEqual(two_arg_arange_int.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(two_arg_arange_int.sum(), 45)

        # testing two positional arguments, one being float
        two_arg_arange_float = ht.arange(0.0, 10)
        self.assertIsInstance(two_arg_arange_float, ht.DNDarray)
        self.assertEqual(two_arg_arange_float.shape, (10,))
        self.assertLessEqual(two_arg_arange_float.lshape[0], 10)
        self.assertEqual(two_arg_arange_float.dtype, ht.float32)
        self.assertEqual(two_arg_arange_float.larray.dtype, torch.float32)
        self.assertEqual(two_arg_arange_float.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(two_arg_arange_float.sum(), 45.0)

        # testing three positional integer arguments
        three_arg_arange_int = ht.arange(0, 10, 2)
        self.assertIsInstance(three_arg_arange_int, ht.DNDarray)
        self.assertEqual(three_arg_arange_int.shape, (5,))
        self.assertLessEqual(three_arg_arange_int.lshape[0], 5)
        self.assertEqual(three_arg_arange_int.dtype, ht.int32)
        self.assertEqual(three_arg_arange_int.larray.dtype, torch.int32)
        self.assertEqual(three_arg_arange_int.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(three_arg_arange_int.sum(), 20)

        # testing three positional arguments, one being float
        three_arg_arange_float = ht.arange(0, 10, 2.0)
        self.assertIsInstance(three_arg_arange_float, ht.DNDarray)
        self.assertEqual(three_arg_arange_float.shape, (5,))
        self.assertLessEqual(three_arg_arange_float.lshape[0], 5)
        self.assertEqual(three_arg_arange_float.dtype, ht.float32)
        self.assertEqual(three_arg_arange_float.larray.dtype, torch.float32)
        self.assertEqual(three_arg_arange_float.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(three_arg_arange_float.sum(), 20.0)

        # testing splitting
        three_arg_arange_dtype_float32 = ht.arange(0, 10, 2.0, split=0)
        self.assertIsInstance(three_arg_arange_dtype_float32, ht.DNDarray)
        self.assertEqual(three_arg_arange_dtype_float32.shape, (5,))
        self.assertLessEqual(three_arg_arange_dtype_float32.lshape[0], 5)
        self.assertEqual(three_arg_arange_dtype_float32.dtype, ht.float32)
        self.assertEqual(three_arg_arange_dtype_float32.larray.dtype, torch.float32)
        self.assertEqual(three_arg_arange_dtype_float32.split, 0)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(three_arg_arange_dtype_float32.sum(axis=0, keepdims=True), 20.0)

        # testing setting dtype to int16
        three_arg_arange_dtype_short = ht.arange(0, 10, 2.0, dtype=torch.int16)
        self.assertIsInstance(three_arg_arange_dtype_short, ht.DNDarray)
        self.assertEqual(three_arg_arange_dtype_short.shape, (5,))
        self.assertLessEqual(three_arg_arange_dtype_short.lshape[0], 5)
        self.assertEqual(three_arg_arange_dtype_short.dtype, ht.int16)
        self.assertEqual(three_arg_arange_dtype_short.larray.dtype, torch.int16)
        self.assertEqual(three_arg_arange_dtype_short.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(three_arg_arange_dtype_short.sum(axis=0, keepdims=True), 20)

        # testing setting dtype to float64
        three_arg_arange_dtype_float64 = ht.arange(0, 10, 2, dtype=torch.float64)
        self.assertIsInstance(three_arg_arange_dtype_float64, ht.DNDarray)
        self.assertEqual(three_arg_arange_dtype_float64.shape, (5,))
        self.assertLessEqual(three_arg_arange_dtype_float64.lshape[0], 5)
        self.assertEqual(three_arg_arange_dtype_float64.dtype, ht.float64)
        self.assertEqual(three_arg_arange_dtype_float64.larray.dtype, torch.float64)
        self.assertEqual(three_arg_arange_dtype_float64.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(three_arg_arange_dtype_float64.sum(axis=0, keepdims=True), 20.0)

        # exceptions
        with self.assertRaises(ValueError):
            ht.arange(-5, 3, split=1)
        with self.assertRaises(TypeError):
            ht.arange()
        with self.assertRaises(TypeError):
            ht.arange(1, 2, 3, 4)

    def test_array(self):
        # basic array function, unsplit data
        unsplit_data = [[1, 2, 3], [4, 5, 6]]
        a = ht.array(unsplit_data)
        self.assertIsInstance(a, ht.DNDarray)
        self.assertEqual(a.dtype, ht.int64)
        self.assertEqual(a.lshape, (2, 3))
        self.assertEqual(a.gshape, (2, 3))
        self.assertEqual(a.split, None)
        self.assertTrue(
            (a.larray == torch.tensor(unsplit_data, device=self.device.torch_device)).all()
        )

        # basic array function, unsplit data, different datatype
        tuple_data = ((0, 0), (1, 1))
        b = ht.array(tuple_data, dtype=ht.int8)
        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(b.dtype, ht.int8)
        self.assertEqual(b.larray.dtype, torch.int8)
        self.assertEqual(b.lshape, (2, 2))
        self.assertEqual(b.gshape, (2, 2))
        self.assertEqual(b.split, None)
        self.assertTrue(
            (
                b.larray
                == torch.tensor(tuple_data, dtype=torch.int8, device=self.device.torch_device)
            ).all()
        )

        # basic array function, unsplit data, no copy
        torch_tensor = torch.tensor([6, 5, 4, 3, 2, 1], device=self.device.torch_device)
        c = ht.array(torch_tensor, copy=False)
        self.assertIsInstance(c, ht.DNDarray)
        self.assertEqual(c.dtype, ht.int64)
        self.assertEqual(c.lshape, (6,))
        self.assertEqual(c.gshape, (6,))
        self.assertEqual(c.split, None)
        self.assertIs(c.larray, torch_tensor)
        self.assertTrue((c.larray == torch_tensor).all())

        # basic array function, unsplit data, additional dimensions
        vector_data = [4.0, 5.0, 6.0]
        d = ht.array(vector_data, ndmin=3)
        self.assertIsInstance(d, ht.DNDarray)
        self.assertEqual(d.dtype, ht.float32)
        self.assertEqual(d.lshape, (3, 1, 1))
        self.assertEqual(d.gshape, (3, 1, 1))
        self.assertEqual(d.split, None)
        self.assertTrue(
            (
                d.larray
                == torch.tensor(vector_data, device=self.device.torch_device).reshape(-1, 1, 1)
            ).all()
        )

        # basic array function, unsplit data, additional dimensions
        vector_data = [4.0, 5.0, 6.0]
        d = ht.array(vector_data, ndmin=-3)
        self.assertIsInstance(d, ht.DNDarray)
        self.assertEqual(d.dtype, ht.float32)
        self.assertEqual(d.lshape, (1, 1, 3))
        self.assertEqual(d.gshape, (1, 1, 3))
        self.assertEqual(d.split, None)
        self.assertTrue(
            (
                d.larray
                == torch.tensor(vector_data, device=self.device.torch_device).reshape(1, 1, -1)
            ).all()
        )

        # distributed array, chunk local data (split), copy True
        array_2d = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        dndarray_2d = ht.array(array_2d, split=0, copy=True)
        self.assertIsInstance(dndarray_2d, ht.DNDarray)
        self.assertEqual(dndarray_2d.dtype, ht.float64)
        self.assertEqual(dndarray_2d.gshape, (3, 3))
        self.assertEqual(len(dndarray_2d.lshape), 2)
        self.assertLessEqual(dndarray_2d.lshape[0], 3)
        self.assertEqual(dndarray_2d.lshape[1], 3)
        self.assertEqual(dndarray_2d.split, 0)
        self.assertTrue(
            (
                dndarray_2d.larray == torch.tensor([1.0, 2.0, 3.0], device=self.device.torch_device)
            ).all()
        )

        # distributed array, chunk local data (split), copy False, torch devices
        array_2d = torch.tensor(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            dtype=torch.double,
            device=self.device.torch_device,
        )
        dndarray_2d = ht.array(array_2d, split=0, copy=False, dtype=ht.double)
        self.assertIsInstance(dndarray_2d, ht.DNDarray)
        self.assertEqual(dndarray_2d.dtype, ht.float64)
        self.assertEqual(dndarray_2d.gshape, (3, 3))
        self.assertEqual(len(dndarray_2d.lshape), 2)
        self.assertLessEqual(dndarray_2d.lshape[0], 3)
        self.assertEqual(dndarray_2d.lshape[1], 3)
        self.assertEqual(dndarray_2d.split, 0)
        self.assertTrue(
            (
                dndarray_2d.larray == torch.tensor([1.0, 2.0, 3.0], device=self.device.torch_device)
            ).all()
        )
        # Check that the array is not a copy, (only really works when the array is not split)
        if ht.communication.MPI_WORLD.size == 1:
            self.assertIs(dndarray_2d.larray, array_2d)

        # The array should not change as all properties match
        dndarray_2d_new = ht.array(dndarray_2d, split=0, copy=False, dtype=ht.double)
        self.assertIsInstance(dndarray_2d_new, ht.DNDarray)
        self.assertEqual(dndarray_2d_new.dtype, ht.float64)
        self.assertEqual(dndarray_2d_new.gshape, (3, 3))
        self.assertEqual(len(dndarray_2d_new.lshape), 2)
        self.assertLessEqual(dndarray_2d_new.lshape[0], 3)
        self.assertEqual(dndarray_2d_new.lshape[1], 3)
        self.assertEqual(dndarray_2d_new.split, 0)
        self.assertTrue(
            (
                dndarray_2d.larray == torch.tensor([1.0, 2.0, 3.0], device=self.device.torch_device)
            ).all()
        )
        # Reuse the same array
        self.assertIs(dndarray_2d_new.larray, dndarray_2d.larray)

        # Should throw exeception because of resplit it causes a resplit
        with self.assertRaises(ValueError):
            dndarray_2d_new = ht.array(dndarray_2d, split=1, copy=False, dtype=ht.double)

        # The array should not change as all properties match
        dndarray_2d_new = ht.array(dndarray_2d, is_split=0, copy=False, dtype=ht.double)
        self.assertIsInstance(dndarray_2d_new, ht.DNDarray)
        self.assertEqual(dndarray_2d_new.dtype, ht.float64)
        self.assertEqual(dndarray_2d_new.gshape, (3, 3))
        self.assertEqual(len(dndarray_2d_new.lshape), 2)
        self.assertLessEqual(dndarray_2d_new.lshape[0], 3)
        self.assertEqual(dndarray_2d_new.lshape[1], 3)
        self.assertEqual(dndarray_2d_new.split, 0)
        self.assertTrue(
            (
                dndarray_2d.larray == torch.tensor([1.0, 2.0, 3.0], device=self.device.torch_device)
            ).all()
        )

        # Should throw exeception because of array is split along another dimension
        with self.assertRaises(ValueError):
            dndarray_2d_new = ht.array(dndarray_2d, is_split=1, copy=False, dtype=ht.double)

        # distributed array, partial data (is_split)
        if ht.communication.MPI_WORLD.rank == 0:
            split_data = [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]
        else:
            split_data = [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]
        e = ht.array(split_data, ndmin=3, is_split=0)

        self.assertIsInstance(e, ht.DNDarray)
        self.assertEqual(e.dtype, ht.float32)
        if ht.communication.MPI_WORLD.rank == 0:
            self.assertEqual(e.lshape, (3, 3, 1))
        else:
            self.assertEqual(e.lshape, (2, 3, 1))
        self.assertEqual(e.split, 0)
        for index, ele in enumerate(e.gshape):
            if index != e.split:
                self.assertEqual(ele, e.lshape[index])
            else:
                self.assertGreaterEqual(ele, e.lshape[index])

        # exception distributed shapes do not fit
        if ht.communication.MPI_WORLD.size > 1:
            if ht.communication.MPI_WORLD.rank == 0:
                split_data = [4.0, 5.0, 6.0]
            else:
                split_data = [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]

            # this will fail as the shapes do not match
            with self.assertRaises(ValueError):
                ht.array(split_data, is_split=0)

        # exception distributed shapes do not fit
        if ht.communication.MPI_WORLD.size > 1:
            if ht.communication.MPI_WORLD.rank == 0:
                split_data = [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]
            else:
                split_data = [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]

            # this will fail as the shapes do not match on a specific axis (here: 0)
            with self.assertRaises(ValueError):
                ht.array(split_data, is_split=1)

        # check exception on mutually exclusive split and is_split
        with self.assertRaises(ValueError):
            ht.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], split=0, is_split=0)

        e = ht.array(split_data, ndmin=-3, is_split=1)

        self.assertIsInstance(e, ht.DNDarray)
        self.assertEqual(e.dtype, ht.float32)
        if ht.communication.MPI_WORLD.rank == 0:
            self.assertEqual(e.lshape, (1, 3, 3))
        else:
            self.assertEqual(e.lshape, (1, 2, 3))
        self.assertEqual(e.split, 1)
        for index, ele in enumerate(e.gshape):
            if index != e.split:
                self.assertEqual(ele, e.lshape[index])
            else:
                self.assertGreaterEqual(ele, e.lshape[index])

        # exception distributed shapes do not fit
        if ht.communication.MPI_WORLD.size > 1:
            if ht.communication.MPI_WORLD.rank == 0:
                split_data = [4.0, 5.0, 6.0]
            else:
                split_data = [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]

            # this will fail as the shapes do not match
            with self.assertRaises(ValueError):
                ht.array(split_data, is_split=0)

        # exception distributed shapes do not fit
        if ht.communication.MPI_WORLD.size > 1:
            if ht.communication.MPI_WORLD.rank == 0:
                split_data = [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]
            else:
                split_data = [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]

            # this will fail as the shapes do not match on a specific axis (here: 0)
            with self.assertRaises(ValueError):
                ht.array(split_data, is_split=1)

        # check exception on mutually exclusive split and is_split
        with self.assertRaises(ValueError):
            ht.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], split=1, is_split=1)

        # non iterable type
        with self.assertRaises(TypeError):
            ht.array(map)
        # iterable, but unsuitable type
        with self.assertRaises(TypeError):
            ht.array("abc")
        # iterable, but unsuitable type, with copy=True
        with self.assertRaises(TypeError):
            ht.array("abc", copy=True)
        # unknown dtype
        with self.assertRaises(TypeError):
            ht.array((4,), dtype="a")
        # invalid ndmin
        with self.assertRaises(TypeError):
            ht.array((4,), ndmin=3.0)
        # invalid split axis type
        with self.assertRaises(TypeError):
            ht.array((4,), split="a")
        # invalid split axis value
        with self.assertRaises(ValueError):
            ht.array((4,), split=3)
        # invalid communicator
        with self.assertRaises(TypeError):
            ht.array((4,), comm={})
        # copy=False but copy is necessary
        data = np.arange(10)
        with self.assertRaises(ValueError):
            ht.array(data, dtype=ht.int32, copy=False)

        # data already distributed but don't match in shape
        if self.get_size() > 1:
            with self.assertRaises(ValueError):
                dim = self.get_rank() + 1
                ht.array([[0] * dim] * dim, is_split=0)

    def test_asarray(self):
        # same heat array
        arr = ht.array([1, 2])
        self.assertTrue(ht.asarray(arr) is arr)

        # from distributed python list
        arr = ht.array([1, 2, 3, 4, 5, 6], split=0)
        lst = arr.tolist(keepsplit=True)
        asarr = ht.asarray(lst, is_split=0)

        self.assertEqual(asarr.shape, arr.shape)
        self.assertEqual(asarr.split, 0)
        self.assertEqual(asarr.device, ht.get_device())
        self.assertTrue(ht.equal(asarr, arr))

        # from numpy array
        arr = np.array([1, 2, 3, 4])
        asarr = ht.asarray(arr)

        self.assertTrue(np.all(np.equal(asarr.numpy(), arr)))

        asarr[0] = 0
        if asarr.device == ht.cpu:
            self.assertEqual(asarr.numpy()[0], arr[0])

        # from torch tensor
        arr = torch.tensor([1, 2, 3, 4], device=self.device.torch_device)
        asarr = ht.asarray(arr)

        self.assertTrue(torch.equal(asarr.larray, arr))

        asarr[0] = 0
        self.assertEqual(asarr.larray[0].item(), arr[0].item())

    def test_empty(self):
        # scalar input
        simple_empty_float = ht.empty(3)
        self.assertIsInstance(simple_empty_float, ht.DNDarray)
        self.assertEqual(simple_empty_float.shape, (3,))
        self.assertEqual(simple_empty_float.lshape, (3,))
        self.assertEqual(simple_empty_float.split, None)
        self.assertEqual(simple_empty_float.dtype, ht.float32)

        # different data type
        simple_empty_uint = ht.empty(5, dtype=ht.bool)
        self.assertIsInstance(simple_empty_uint, ht.DNDarray)
        self.assertEqual(simple_empty_uint.shape, (5,))
        self.assertEqual(simple_empty_uint.lshape, (5,))
        self.assertEqual(simple_empty_uint.split, None)
        self.assertEqual(simple_empty_uint.dtype, ht.bool)

        # multi-dimensional
        elaborate_empty_int = ht.empty((2, 3), dtype=ht.int32)
        self.assertIsInstance(elaborate_empty_int, ht.DNDarray)
        self.assertEqual(elaborate_empty_int.shape, (2, 3))
        self.assertEqual(elaborate_empty_int.lshape, (2, 3))
        self.assertEqual(elaborate_empty_int.split, None)
        self.assertEqual(elaborate_empty_int.dtype, ht.int32)

        # split axis
        elaborate_empty_split = ht.empty((6, 4), dtype=ht.int32, split=0)
        self.assertIsInstance(elaborate_empty_split, ht.DNDarray)
        self.assertEqual(elaborate_empty_split.shape, (6, 4))
        self.assertLessEqual(elaborate_empty_split.lshape[0], 6)
        self.assertEqual(elaborate_empty_split.lshape[1], 4)
        self.assertEqual(elaborate_empty_split.split, 0)
        self.assertEqual(elaborate_empty_split.dtype, ht.int32)

        # exceptions
        with self.assertRaises(TypeError):
            ht.empty("(2, 3,)", dtype=ht.float64)
        with self.assertRaises(ValueError):
            ht.empty((-1, 3), dtype=ht.float64)
        with self.assertRaises(TypeError):
            ht.empty((2, 3), dtype=ht.float64, split="axis")

    def test_empty_like(self):
        # scalar
        like_int = ht.empty_like(3)
        self.assertIsInstance(like_int, ht.DNDarray)
        self.assertEqual(like_int.shape, (1,))
        self.assertEqual(like_int.lshape, (1,))
        self.assertEqual(like_int.split, None)
        self.assertEqual(like_int.dtype, ht.int32)

        # sequence
        like_str = ht.empty_like("abc")
        self.assertIsInstance(like_str, ht.DNDarray)
        self.assertEqual(like_str.shape, (3,))
        self.assertEqual(like_str.lshape, (3,))
        self.assertEqual(like_str.split, None)
        self.assertEqual(like_str.dtype, ht.float32)

        # elaborate tensor
        ones = ht.ones((2, 3), dtype=ht.uint8)
        like_ones = ht.empty_like(ones)
        self.assertIsInstance(like_ones, ht.DNDarray)
        self.assertEqual(like_ones.shape, (2, 3))
        self.assertEqual(like_ones.lshape, (2, 3))
        self.assertEqual(like_ones.split, None)
        self.assertEqual(like_ones.dtype, ht.uint8)

        # elaborate tensor with split
        ones_split = ht.ones((2, 3), dtype=ht.uint8, split=0)
        like_ones_split = ht.empty_like(ones_split)
        self.assertIsInstance(like_ones_split, ht.DNDarray)
        self.assertEqual(like_ones_split.shape, (2, 3))
        self.assertLessEqual(like_ones_split.lshape[0], 2)
        self.assertEqual(like_ones_split.lshape[1], 3)
        self.assertEqual(like_ones_split.split, 0)
        self.assertEqual(like_ones_split.dtype, ht.uint8)

        # exceptions
        with self.assertRaises(TypeError):
            ht.empty_like(ones, dtype="abc")
        with self.assertRaises(TypeError):
            ht.empty_like(ones, split="axis")

    def test_eye(self):
        def get_offset(tensor_array):
            x, y = tensor_array.shape
            for k in range(x):
                for li in range(y):
                    if tensor_array[k][li] == 1:
                        return k, li
            return x, y

        shape = 5
        eye = ht.eye(shape, dtype=ht.uint8, split=1)
        self.assertIsInstance(eye, ht.DNDarray)
        self.assertEqual(eye.dtype, ht.uint8)
        self.assertEqual(eye.shape, (shape, shape))
        self.assertEqual(eye.split, 1)

        offset_x, offset_y = get_offset(eye.larray)
        self.assertGreaterEqual(offset_x, 0)
        self.assertGreaterEqual(offset_y, 0)
        x, y = eye.larray.shape
        for i in range(x):
            for j in range(y):
                expected = 1 if i - offset_x is j - offset_y else 0
                self.assertEqual(eye.larray[i][j], expected)

        shape = (10, 20)
        eye = ht.eye(shape, dtype=ht.float32)
        self.assertIsInstance(eye, ht.DNDarray)
        self.assertEqual(eye.dtype, ht.float32)
        self.assertEqual(eye.shape, shape)
        self.assertEqual(eye.split, None)

        offset_x, offset_y = get_offset(eye.larray)
        self.assertGreaterEqual(offset_x, 0)
        self.assertGreaterEqual(offset_y, 0)
        x, y = eye.larray.shape
        for i in range(x):
            for j in range(y):
                expected = 1.0 if i - offset_x is j - offset_y else 0.0
                self.assertEqual(eye.larray[i][j], expected)

        shape = (10,)
        eye = ht.eye(shape, dtype=ht.int32, split=0)
        self.assertIsInstance(eye, ht.DNDarray)
        self.assertEqual(eye.dtype, ht.int32)
        self.assertEqual(eye.shape, shape * 2)
        self.assertEqual(eye.split, 0)

        offset_x, offset_y = get_offset(eye.larray)
        self.assertGreaterEqual(offset_x, 0)
        self.assertGreaterEqual(offset_y, 0)
        x, y = eye.larray.shape
        for i in range(x):
            for j in range(y):
                expected = 1 if i - offset_x is j - offset_y else 0
                self.assertEqual(eye.larray[i][j], expected)

        shape = (11, 30)
        eye = ht.eye(shape, split=1, dtype=ht.float32)
        self.assertIsInstance(eye, ht.DNDarray)
        self.assertEqual(eye.dtype, ht.float32)
        self.assertEqual(eye.shape, shape)
        self.assertEqual(eye.split, 1)

    def test_from_partitioned(self):
        a = ht.zeros((120, 120), split=0)
        b = ht.from_partitioned(a, comm=a.comm)
        a[2, :] = 128
        self.assertTrue(ht.equal(a, b))

        a.resplit_(None)
        b = ht.from_partitioned(a, comm=a.comm)
        self.assertTrue(ht.equal(a, b))

        a.resplit_(1)
        b = ht.from_partitioned(a, comm=a.comm)
        b[50] = 94
        self.assertTrue(ht.equal(a, b))

        del b.__partitioned__["shape"]
        with self.assertRaises(RuntimeError):
            _ = ht.from_partitioned(b)
        b.__partitions_dict__ = None
        _ = b.__partitioned__

        del b.__partitioned__["locals"]
        with self.assertRaises(RuntimeError):
            _ = ht.from_partitioned(b)
        b.__partitions_dict__ = None
        _ = b.__partitioned__

        del b.__partitioned__["locals"]
        with self.assertRaises(RuntimeError):
            _ = ht.from_partitioned(b)
        b.__partitions_dict__ = None
        _ = b.__partitioned__

    def test_from_partition_dict(self):
        a = ht.zeros((120, 120), split=0)
        b = ht.from_partition_dict(a.__partitioned__, comm=a.comm)
        a[0, 0] = 100
        self.assertTrue(ht.equal(a, b))

        a.resplit_(None)
        a[0, 0] = 50
        b = ht.from_partition_dict(a.__partitioned__, comm=a.comm)
        self.assertTrue(ht.equal(a, b))

        del b.__partitioned__["shape"]
        with self.assertRaises(RuntimeError):
            _ = ht.from_partition_dict(b.__partitioned__)
        b.__partitions_dict__ = None
        _ = b.__partitioned__

        del b.__partitioned__["locals"]
        with self.assertRaises(RuntimeError):
            _ = ht.from_partition_dict(b.__partitioned__)
        b.__partitions_dict__ = None
        _ = b.__partitioned__

        del b.__partitioned__["locals"]
        with self.assertRaises(RuntimeError):
            _ = ht.from_partition_dict(b.__partitioned__)
        b.__partitions_dict__ = None
        _ = b.__partitioned__

    def test_full(self):
        # simple tensor
        data = ht.full((10, 2), 4)
        self.assertIsInstance(data, ht.DNDarray)
        self.assertEqual(data.shape, (10, 2))
        self.assertEqual(data.lshape, (10, 2))
        self.assertEqual(data.dtype, ht.float32)
        self.assertEqual(data.larray.dtype, torch.float32)
        self.assertEqual(data.split, None)
        self.assertTrue(ht.allclose(data, ht.float32(4.0)))

        # non-standard dtype tensor
        data = ht.full((10, 2), 4, dtype=ht.int32)
        self.assertIsInstance(data, ht.DNDarray)
        self.assertEqual(data.shape, (10, 2))
        self.assertEqual(data.lshape, (10, 2))
        self.assertEqual(data.dtype, ht.int32)
        self.assertEqual(data.larray.dtype, torch.int32)
        self.assertEqual(data.split, None)
        self.assertTrue(ht.allclose(data, ht.int32(4)))

        # split tensor
        data = ht.full((10, 2), 4, split=0)
        self.assertIsInstance(data, ht.DNDarray)
        self.assertEqual(data.shape, (10, 2))
        self.assertLessEqual(data.lshape[0], 10)
        self.assertEqual(data.lshape[1], 2)
        self.assertEqual(data.dtype, ht.float32)
        self.assertEqual(data.larray.dtype, torch.float32)
        self.assertEqual(data.split, 0)
        self.assertTrue(ht.allclose(data, ht.float32(4.0)))

        # exceptions
        with self.assertRaises(TypeError):
            ht.full("(2, 3,)", 4, dtype=ht.float64)
        with self.assertRaises(ValueError):
            ht.full((-1, 3), 2, dtype=ht.float64)
        with self.assertRaises(TypeError):
            ht.full((2, 3), dtype=ht.float64, split="axis")

    def test_full_like(self):
        # scalar
        like_int = ht.full_like(3, 4)
        self.assertIsInstance(like_int, ht.DNDarray)
        self.assertEqual(like_int.shape, (1,))
        self.assertEqual(like_int.lshape, (1,))
        self.assertEqual(like_int.split, None)
        self.assertEqual(like_int.dtype, ht.float32)
        self.assertTrue(ht.allclose(like_int, ht.float32(4)))

        # sequence
        like_str = ht.full_like("abc", 2)
        self.assertIsInstance(like_str, ht.DNDarray)
        self.assertEqual(like_str.shape, (3,))
        self.assertEqual(like_str.lshape, (3,))
        self.assertEqual(like_str.split, None)
        self.assertEqual(like_str.dtype, ht.float32)
        self.assertTrue(ht.allclose(like_str, ht.float32(2)))

        # elaborate tensor
        zeros = ht.zeros((2, 3), dtype=ht.uint8)
        like_zeros = ht.full_like(zeros, 7)
        self.assertIsInstance(like_zeros, ht.DNDarray)
        self.assertEqual(like_zeros.shape, (2, 3))
        self.assertEqual(like_zeros.lshape, (2, 3))
        self.assertEqual(like_zeros.split, None)
        self.assertEqual(like_zeros.dtype, ht.float32)
        self.assertTrue(ht.allclose(like_zeros, ht.float32(7)))

        # elaborate tensor with split
        zeros_split = ht.zeros((2, 3), dtype=ht.uint8, split=0)
        like_zeros_split = ht.full_like(zeros_split, 6)
        self.assertIsInstance(like_zeros_split, ht.DNDarray)
        self.assertEqual(like_zeros_split.shape, (2, 3))
        self.assertLessEqual(like_zeros_split.lshape[0], 2)
        self.assertEqual(like_zeros_split.lshape[1], 3)
        self.assertEqual(like_zeros_split.split, 0)
        self.assertEqual(like_zeros_split.dtype, ht.float32)
        self.assertTrue(ht.allclose(like_zeros_split, ht.float32(6)))

        # exceptions
        with self.assertRaises(TypeError):
            ht.ones_like(zeros, dtype="abc")
        with self.assertRaises(TypeError):
            ht.ones_like(zeros, split="axis")

    def test_linspace(self):
        # simple linear space
        ascending = ht.linspace(-3, 5)
        self.assertIsInstance(ascending, ht.DNDarray)
        self.assertEqual(ascending.shape, (50,))
        self.assertLessEqual(ascending.lshape[0], 50)
        self.assertEqual(ascending.dtype, ht.float32)
        self.assertEqual(ascending.larray.dtype, torch.float32)
        self.assertEqual(ascending.split, None)

        zero_samples = ht.linspace(-3, 5, num=0)
        self.assertEqual(zero_samples.size, 0)

        # simple inverse linear space
        descending = ht.linspace(-5, 3, num=100)
        self.assertIsInstance(descending, ht.DNDarray)
        self.assertEqual(descending.shape, (100,))
        self.assertLessEqual(descending.lshape[0], 100)
        self.assertEqual(descending.dtype, ht.float32)
        self.assertEqual(descending.larray.dtype, torch.float32)
        self.assertEqual(descending.split, None)

        # split linear space
        split = ht.linspace(-5, 3, num=70, split=0)
        self.assertIsInstance(split, ht.DNDarray)
        self.assertEqual(split.shape, (70,))
        self.assertLessEqual(split.lshape[0], 70)
        self.assertEqual(split.dtype, ht.float32)
        self.assertEqual(split.larray.dtype, torch.float32)
        self.assertEqual(split.split, 0)

        # with casted type
        casted = ht.linspace(-5, 3, num=70, dtype=ht.uint8, split=0)
        self.assertIsInstance(casted, ht.DNDarray)
        self.assertEqual(casted.shape, (70,))
        self.assertLessEqual(casted.lshape[0], 70)
        self.assertEqual(casted.dtype, ht.uint8)
        self.assertEqual(casted.larray.dtype, torch.uint8)
        self.assertEqual(casted.split, 0)

        # retstep test
        result = ht.linspace(-5, 3, num=70, retstep=True, dtype=ht.uint8, split=0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        self.assertIsInstance(result[0], ht.DNDarray)
        self.assertEqual(result[0].shape, (70,))
        self.assertLessEqual(result[0].lshape[0], 70)
        self.assertEqual(result[0].dtype, ht.uint8)
        self.assertEqual(result[0].larray.dtype, torch.uint8)
        self.assertEqual(result[0].split, 0)

        self.assertIsInstance(result[1], float)
        self.assertEqual(result[1], 0.11594202898550725)

        # exceptions
        with self.assertRaises(ValueError):
            ht.linspace(-5, 3, split=1)
        with self.assertRaises(ValueError):
            ht.linspace(-5, 3, num=-1)

    def test_logspace(self):
        # simple log space
        ascending = ht.logspace(-3, 5)
        self.assertIsInstance(ascending, ht.DNDarray)
        self.assertEqual(ascending.shape, (50,))
        self.assertLessEqual(ascending.lshape[0], 50)
        self.assertEqual(ascending.dtype, ht.float32)
        self.assertEqual(ascending.larray.dtype, torch.float32)
        self.assertEqual(ascending.split, None)

        zero_samples = ht.logspace(-3, 5, num=0)
        self.assertEqual(zero_samples.size, 0)

        # simple inverse log space
        descending = ht.logspace(-5, 3, num=100)
        self.assertIsInstance(descending, ht.DNDarray)
        self.assertEqual(descending.shape, (100,))
        self.assertLessEqual(descending.lshape[0], 100)
        self.assertEqual(descending.dtype, ht.float32)
        self.assertEqual(descending.larray.dtype, torch.float32)
        self.assertEqual(descending.split, None)

        # split log space
        split = ht.logspace(-5, 3, num=70, split=0)
        self.assertIsInstance(split, ht.DNDarray)
        self.assertEqual(split.shape, (70,))
        self.assertLessEqual(split.lshape[0], 70)
        self.assertEqual(split.dtype, ht.float32)
        self.assertEqual(split.larray.dtype, torch.float32)
        self.assertEqual(split.split, 0)

        # with casted type
        casted = ht.logspace(-5, 3, num=70, dtype=ht.uint8, split=0)
        self.assertIsInstance(casted, ht.DNDarray)
        self.assertEqual(casted.shape, (70,))
        self.assertLessEqual(casted.lshape[0], 70)
        self.assertEqual(casted.dtype, ht.uint8)
        self.assertEqual(casted.larray.dtype, torch.uint8)
        self.assertEqual(casted.split, 0)

        # base test
        base = ht.logspace(-5, 3, num=70, base=2.0)
        self.assertIsInstance(base, ht.DNDarray)
        self.assertEqual(base.shape, (70,))
        self.assertLessEqual(base.lshape[0], 70)
        self.assertEqual(base.dtype, ht.float32)
        self.assertEqual(base.larray.dtype, torch.float32)
        self.assertEqual(base.split, None)

        # exceptions
        with self.assertRaises(ValueError):
            ht.logspace(-5, 3, split=1)
        with self.assertRaises(ValueError):
            ht.logspace(-5, 3, num=-1)

    def test_meshgrid(self):
        # arrays < 2
        self.assertEqual(ht.meshgrid(), [])
        self.assertEqual(ht.meshgrid(ht.array(1)), [ht.array(1)])

        # 2 arrays
        x = ht.arange(4)
        y = ht.arange(3)
        xx, yy = ht.meshgrid(x, y)
        res_xx = ht.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], dtype=ht.int32, split=None)
        res_yy = ht.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]], dtype=ht.int32, split=None)

        self.assertEqual(xx.shape, res_xx.shape)
        self.assertEqual(yy.shape, res_yy.shape)
        self.assertEqual(xx.device, res_xx.device)
        self.assertEqual(yy.device, res_yy.device)
        self.assertEqual(xx.dtype, x.dtype)
        self.assertEqual(yy.dtype, y.dtype)
        self.assertTrue(ht.equal(xx, res_xx))
        self.assertTrue(ht.equal(yy, res_yy))

        # different dtype + indexing parameter
        x = ht.linspace(0, 4, 3)
        y = ht.linspace(0, 4, 5)
        xx, yy = ht.meshgrid(x, y, indexing="ij")

        res_xx = ht.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0, 2.0], [4.0, 4.0, 4.0, 4.0, 4.0]],
            dtype=ht.float32,
            split=None,
        )
        res_yy = ht.array(
            [[0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0]],
            dtype=ht.float32,
            split=None,
        )

        self.assertEqual(xx.shape, res_xx.shape)
        self.assertEqual(yy.shape, res_yy.shape)
        self.assertEqual(xx.device, res_xx.device)
        self.assertEqual(yy.device, res_yy.device)
        self.assertEqual(xx.dtype, x.dtype)
        self.assertEqual(yy.dtype, y.dtype)
        self.assertTrue(ht.equal(xx, res_xx))
        self.assertTrue(ht.equal(yy, res_yy))

        # 3 arrays
        z = ht.linspace(0, 1, 3, split=0)

        # split 0
        zz, xx, yy = ht.meshgrid(z, x, y)

        res_zz, res_xx, res_yy = np.meshgrid(z.numpy(), x.numpy(), y.numpy())

        self.assertEqual(xx.split, 0)
        self.assertEqual(yy.split, 0)
        self.assertEqual(zz.split, 0)
        self.assertTrue(ht.equal(xx, ht.array(res_xx)))
        self.assertTrue(ht.equal(yy, ht.array(res_yy)))
        self.assertTrue(ht.equal(zz, ht.array(res_zz)))

        # split 1
        xx, zz, yy = ht.meshgrid(x, z, y)

        res_xx, res_zz, res_yy = np.meshgrid(x.numpy(), z.numpy(), y.numpy())

        self.assertEqual(xx.split, 1)
        self.assertEqual(yy.split, 1)
        self.assertEqual(zz.split, 1)
        self.assertTrue(ht.equal(xx, ht.array(res_xx)))
        self.assertTrue(ht.equal(yy, ht.array(res_yy)))
        self.assertTrue(ht.equal(zz, ht.array(res_zz)))

        # split 2
        xx, yy, zz = ht.meshgrid(x, y, z)

        res_xx, res_yy, res_zz = np.meshgrid(x.numpy(), y.numpy(), z.numpy())

        self.assertEqual(xx.split, 2)
        self.assertEqual(yy.split, 2)
        self.assertEqual(zz.split, 2)
        self.assertTrue(ht.equal(xx, ht.array(res_xx)))
        self.assertTrue(ht.equal(yy, ht.array(res_yy)))
        self.assertTrue(ht.equal(zz, ht.array(res_zz)))

        # exceptions
        with self.assertRaises(ValueError):
            ht.meshgrid(ht.array(1), indexing="abc")

        with self.assertRaises(ValueError):
            ht.meshgrid(ht.array([0, 1], split=0), ht.array([1, 2], split=0))

    def test_ones(self):
        # scalar input
        simple_ones_float = ht.ones(3)
        self.assertIsInstance(simple_ones_float, ht.DNDarray)
        self.assertEqual(simple_ones_float.shape, (3,))
        self.assertEqual(simple_ones_float.lshape, (3,))
        self.assertEqual(simple_ones_float.split, None)
        self.assertEqual(simple_ones_float.dtype, ht.float32)
        self.assertEqual((simple_ones_float.larray == 1).all().item(), 1)

        # different data type
        simple_ones_uint = ht.ones(5, dtype=ht.bool)
        self.assertIsInstance(simple_ones_uint, ht.DNDarray)
        self.assertEqual(simple_ones_uint.shape, (5,))
        self.assertEqual(simple_ones_uint.lshape, (5,))
        self.assertEqual(simple_ones_uint.split, None)
        self.assertEqual(simple_ones_uint.dtype, ht.bool)
        self.assertEqual((simple_ones_uint.larray == 1).all().item(), 1)

        # multi-dimensional
        elaborate_ones_int = ht.ones((2, 3), dtype=ht.int32)
        self.assertIsInstance(elaborate_ones_int, ht.DNDarray)
        self.assertEqual(elaborate_ones_int.shape, (2, 3))
        self.assertEqual(elaborate_ones_int.lshape, (2, 3))
        self.assertEqual(elaborate_ones_int.split, None)
        self.assertEqual(elaborate_ones_int.dtype, ht.int32)
        self.assertEqual((elaborate_ones_int.larray == 1).all().item(), 1)

        # split axis
        elaborate_ones_split = ht.ones((6, 4), dtype=ht.int32, split=0)
        self.assertIsInstance(elaborate_ones_split, ht.DNDarray)
        self.assertEqual(elaborate_ones_split.shape, (6, 4))
        self.assertLessEqual(elaborate_ones_split.lshape[0], 6)
        self.assertEqual(elaborate_ones_split.lshape[1], 4)
        self.assertEqual(elaborate_ones_split.split, 0)
        self.assertEqual(elaborate_ones_split.dtype, ht.int32)
        self.assertEqual((elaborate_ones_split.larray == 1).all().item(), 1)

        # exceptions
        with self.assertRaises(TypeError):
            ht.ones("(2, 3,)", dtype=ht.float64)
        with self.assertRaises(ValueError):
            ht.ones((-1, 3), dtype=ht.float64)
        with self.assertRaises(TypeError):
            ht.ones((2, 3), dtype=ht.float64, split="axis")

    def test_ones_like(self):
        # scalar
        like_int = ht.ones_like(3)
        self.assertIsInstance(like_int, ht.DNDarray)
        self.assertEqual(like_int.shape, (1,))
        self.assertEqual(like_int.lshape, (1,))
        self.assertEqual(like_int.split, None)
        self.assertEqual(like_int.dtype, ht.int32)
        self.assertEqual((like_int.larray == 1).all().item(), 1)

        # sequence
        like_str = ht.ones_like("abc")
        self.assertIsInstance(like_str, ht.DNDarray)
        self.assertEqual(like_str.shape, (3,))
        self.assertEqual(like_str.lshape, (3,))
        self.assertEqual(like_str.split, None)
        self.assertEqual(like_str.dtype, ht.float32)
        self.assertEqual((like_str.larray == 1).all().item(), 1)

        # elaborate tensor
        zeros = ht.zeros((2, 3), dtype=ht.uint8)
        like_zeros = ht.ones_like(zeros)
        self.assertIsInstance(like_zeros, ht.DNDarray)
        self.assertEqual(like_zeros.shape, (2, 3))
        self.assertEqual(like_zeros.lshape, (2, 3))
        self.assertEqual(like_zeros.split, None)
        self.assertEqual(like_zeros.dtype, ht.uint8)
        self.assertEqual((like_zeros.larray == 1).all().item(), 1)

        # elaborate tensor with split
        zeros_split = ht.zeros((2, 3), dtype=ht.uint8, split=0)
        like_zeros_split = ht.ones_like(zeros_split)
        self.assertIsInstance(like_zeros_split, ht.DNDarray)
        self.assertEqual(like_zeros_split.shape, (2, 3))
        self.assertLessEqual(like_zeros_split.lshape[0], 2)
        self.assertEqual(like_zeros_split.lshape[1], 3)
        self.assertEqual(like_zeros_split.split, 0)
        self.assertEqual(like_zeros_split.dtype, ht.uint8)
        self.assertEqual((like_zeros_split.larray == 1).all().item(), 1)

        # exceptions
        with self.assertRaises(TypeError):
            ht.ones_like(zeros, dtype="abc")
        with self.assertRaises(TypeError):
            ht.ones_like(zeros, split="axis")

    def test_zeros(self):
        # scalar input
        simple_zeros_float = ht.zeros(3)
        self.assertIsInstance(simple_zeros_float, ht.DNDarray)
        self.assertEqual(simple_zeros_float.shape, (3,))
        self.assertEqual(simple_zeros_float.lshape, (3,))
        self.assertEqual(simple_zeros_float.split, None)
        self.assertEqual(simple_zeros_float.dtype, ht.float32)
        self.assertEqual((simple_zeros_float.larray == 0).all().item(), 1)

        # different data type
        simple_zeros_uint = ht.zeros(5, dtype=ht.bool)
        self.assertIsInstance(simple_zeros_uint, ht.DNDarray)
        self.assertEqual(simple_zeros_uint.shape, (5,))
        self.assertEqual(simple_zeros_uint.lshape, (5,))
        self.assertEqual(simple_zeros_uint.split, None)
        self.assertEqual(simple_zeros_uint.dtype, ht.bool)
        self.assertEqual((simple_zeros_uint.larray == 0).all().item(), 1)

        # multi-dimensional
        elaborate_zeros_int = ht.zeros((2, 3), dtype=ht.int32)
        self.assertIsInstance(elaborate_zeros_int, ht.DNDarray)
        self.assertEqual(elaborate_zeros_int.shape, (2, 3))
        self.assertEqual(elaborate_zeros_int.lshape, (2, 3))
        self.assertEqual(elaborate_zeros_int.split, None)
        self.assertEqual(elaborate_zeros_int.dtype, ht.int32)
        self.assertEqual((elaborate_zeros_int.larray == 0).all().item(), 1)

        # split axis
        elaborate_zeros_split = ht.zeros((6, 4), dtype=ht.int32, split=0)
        self.assertIsInstance(elaborate_zeros_split, ht.DNDarray)
        self.assertEqual(elaborate_zeros_split.shape, (6, 4))
        self.assertLessEqual(elaborate_zeros_split.lshape[0], 6)
        self.assertEqual(elaborate_zeros_split.lshape[1], 4)
        self.assertEqual(elaborate_zeros_split.split, 0)
        self.assertEqual(elaborate_zeros_split.dtype, ht.int32)
        self.assertEqual((elaborate_zeros_split.larray == 0).all().item(), 1)

        # exceptions
        with self.assertRaises(TypeError):
            ht.zeros("(2, 3,)", dtype=ht.float64)
        with self.assertRaises(ValueError):
            ht.zeros((-1, 3), dtype=ht.float64)
        with self.assertRaises(TypeError):
            ht.zeros((2, 3), dtype=ht.float64, split="axis")

    def test_zeros_like(self):
        # scalar
        like_int = ht.zeros_like(3)
        self.assertIsInstance(like_int, ht.DNDarray)
        self.assertEqual(like_int.shape, (1,))
        self.assertEqual(like_int.lshape, (1,))
        self.assertEqual(like_int.split, None)
        self.assertEqual(like_int.dtype, ht.int32)
        self.assertEqual((like_int.larray == 0).all().item(), 1)

        # sequence
        like_str = ht.zeros_like("abc")
        self.assertIsInstance(like_str, ht.DNDarray)
        self.assertEqual(like_str.shape, (3,))
        self.assertEqual(like_str.lshape, (3,))
        self.assertEqual(like_str.split, None)
        self.assertEqual(like_str.dtype, ht.float32)
        self.assertEqual((like_str.larray == 0).all().item(), 1)

        # elaborate tensor
        ones = ht.ones((2, 3), dtype=ht.uint8)
        like_ones = ht.zeros_like(ones)
        self.assertIsInstance(like_ones, ht.DNDarray)
        self.assertEqual(like_ones.shape, (2, 3))
        self.assertEqual(like_ones.lshape, (2, 3))
        self.assertEqual(like_ones.split, None)
        self.assertEqual(like_ones.dtype, ht.uint8)
        self.assertEqual((like_ones.larray == 0).all().item(), 1)

        # elaborate tensor with split
        ones_split = ht.ones((2, 3), dtype=ht.uint8, split=0)
        like_ones_split = ht.zeros_like(ones_split)
        self.assertIsInstance(like_ones_split, ht.DNDarray)
        self.assertEqual(like_ones_split.shape, (2, 3))
        self.assertLessEqual(like_ones_split.lshape[0], 2)
        self.assertEqual(like_ones_split.lshape[1], 3)
        self.assertEqual(like_ones_split.split, 0)
        self.assertEqual(like_ones_split.dtype, ht.uint8)
        self.assertEqual((like_ones_split.larray == 0).all().item(), 1)

        # exceptions
        with self.assertRaises(TypeError):
            ht.zeros_like(ones, dtype="abc")
        with self.assertRaises(TypeError):
            ht.zeros_like(ones, split="axis")
