import unittest
import torch
import os
import heat as ht

if os.environ.get("DEVICE") == "gpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ht.use_device("gpu" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
    ht.use_device("cpu")


class TestFactories(unittest.TestCase):
    def test_arange(self):
        # testing one positional integer argument
        one_arg_arange_int = ht.arange(10)
        self.assertIsInstance(one_arg_arange_int, ht.DNDarray)
        self.assertEqual(one_arg_arange_int.shape, (10,))
        self.assertLessEqual(one_arg_arange_int.lshape[0], 10)
        self.assertEqual(one_arg_arange_int.dtype, ht.int32)
        self.assertEqual(one_arg_arange_int._DNDarray__array.dtype, torch.int32)
        self.assertEqual(one_arg_arange_int.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(one_arg_arange_int.sum(), 45)

        # testing one positional float argument
        one_arg_arange_float = ht.arange(10.0)
        self.assertIsInstance(one_arg_arange_float, ht.DNDarray)
        self.assertEqual(one_arg_arange_float.shape, (10,))
        self.assertLessEqual(one_arg_arange_float.lshape[0], 10)
        self.assertEqual(one_arg_arange_float.dtype, ht.int32)
        self.assertEqual(one_arg_arange_float._DNDarray__array.dtype, torch.int32)
        self.assertEqual(one_arg_arange_float.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(one_arg_arange_float.sum(), 45.0)

        # testing two positional integer arguments
        two_arg_arange_int = ht.arange(0, 10)
        self.assertIsInstance(two_arg_arange_int, ht.DNDarray)
        self.assertEqual(two_arg_arange_int.shape, (10,))
        self.assertLessEqual(two_arg_arange_int.lshape[0], 10)
        self.assertEqual(two_arg_arange_int.dtype, ht.int32)
        self.assertEqual(two_arg_arange_int._DNDarray__array.dtype, torch.int32)
        self.assertEqual(two_arg_arange_int.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(two_arg_arange_int.sum(), 45)

        # testing two positional arguments, one being float
        two_arg_arange_float = ht.arange(0.0, 10)
        self.assertIsInstance(two_arg_arange_float, ht.DNDarray)
        self.assertEqual(two_arg_arange_float.shape, (10,))
        self.assertLessEqual(two_arg_arange_float.lshape[0], 10)
        self.assertEqual(two_arg_arange_float.dtype, ht.float32)
        self.assertEqual(two_arg_arange_float._DNDarray__array.dtype, torch.float32)
        self.assertEqual(two_arg_arange_float.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(two_arg_arange_float.sum(), 45.0)

        # testing three positional integer arguments
        three_arg_arange_int = ht.arange(0, 10, 2)
        self.assertIsInstance(three_arg_arange_int, ht.DNDarray)
        self.assertEqual(three_arg_arange_int.shape, (5,))
        self.assertLessEqual(three_arg_arange_int.lshape[0], 5)
        self.assertEqual(three_arg_arange_int.dtype, ht.int32)
        self.assertEqual(three_arg_arange_int._DNDarray__array.dtype, torch.int32)
        self.assertEqual(three_arg_arange_int.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(three_arg_arange_int.sum(), 20)

        # testing three positional arguments, one being float
        three_arg_arange_float = ht.arange(0, 10, 2.0)
        self.assertIsInstance(three_arg_arange_float, ht.DNDarray)
        self.assertEqual(three_arg_arange_float.shape, (5,))
        self.assertLessEqual(three_arg_arange_float.lshape[0], 5)
        self.assertEqual(three_arg_arange_float.dtype, ht.float32)
        self.assertEqual(three_arg_arange_float._DNDarray__array.dtype, torch.float32)
        self.assertEqual(three_arg_arange_float.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(three_arg_arange_float.sum(), 20.0)

        # testing splitting
        three_arg_arange_dtype_float32 = ht.arange(0, 10, 2.0, split=0)
        self.assertIsInstance(three_arg_arange_dtype_float32, ht.DNDarray)
        self.assertEqual(three_arg_arange_dtype_float32.shape, (5,))
        self.assertLessEqual(three_arg_arange_dtype_float32.lshape[0], 5)
        self.assertEqual(three_arg_arange_dtype_float32.dtype, ht.float32)
        self.assertEqual(three_arg_arange_dtype_float32._DNDarray__array.dtype, torch.float32)
        self.assertEqual(three_arg_arange_dtype_float32.split, 0)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(three_arg_arange_dtype_float32.sum(axis=0, keepdim=True), 20.0)

        # testing setting dtype to int16
        three_arg_arange_dtype_short = ht.arange(0, 10, 2.0, dtype=torch.int16)
        self.assertIsInstance(three_arg_arange_dtype_short, ht.DNDarray)
        self.assertEqual(three_arg_arange_dtype_short.shape, (5,))
        self.assertLessEqual(three_arg_arange_dtype_short.lshape[0], 5)
        self.assertEqual(three_arg_arange_dtype_short.dtype, ht.int16)
        self.assertEqual(three_arg_arange_dtype_short._DNDarray__array.dtype, torch.int16)
        self.assertEqual(three_arg_arange_dtype_short.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(three_arg_arange_dtype_short.sum(axis=0, keepdim=True), 20)

        # testing setting dtype to float64
        three_arg_arange_dtype_float64 = ht.arange(0, 10, 2, dtype=torch.float64)
        self.assertIsInstance(three_arg_arange_dtype_float64, ht.DNDarray)
        self.assertEqual(three_arg_arange_dtype_float64.shape, (5,))
        self.assertLessEqual(three_arg_arange_dtype_float64.lshape[0], 5)
        self.assertEqual(three_arg_arange_dtype_float64.dtype, ht.float64)
        self.assertEqual(three_arg_arange_dtype_float64._DNDarray__array.dtype, torch.float64)
        self.assertEqual(three_arg_arange_dtype_float64.split, None)
        # make an in direct check for the sequence, compare against the gaussian sum
        self.assertEqual(three_arg_arange_dtype_float64.sum(axis=0, keepdim=True), 20.0)

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
        self.assertTrue((a._DNDarray__array == torch.tensor(unsplit_data, device=device)).all())

        # basic array function, unsplit data, different datatype
        tuple_data = ((0, 0), (1, 1))
        b = ht.array(tuple_data, dtype=ht.int8)
        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(b.dtype, ht.int8)
        self.assertEqual(b._DNDarray__array.dtype, torch.int8)
        self.assertEqual(b.lshape, (2, 2))
        self.assertEqual(b.gshape, (2, 2))
        self.assertEqual(b.split, None)
        self.assertTrue(
            (b._DNDarray__array == torch.tensor(tuple_data, dtype=torch.int8, device=device)).all()
        )

        # basic array function, unsplit data, no copy
        torch_tensor = torch.tensor([6, 5, 4, 3, 2, 1], device=device)
        c = ht.array(torch_tensor, copy=False)
        self.assertIsInstance(c, ht.DNDarray)
        self.assertEqual(c.dtype, ht.int64)
        self.assertEqual(c.lshape, (6,))
        self.assertEqual(c.gshape, (6,))
        self.assertEqual(c.split, None)
        self.assertIs(c._DNDarray__array, torch_tensor)
        self.assertTrue((c._DNDarray__array == torch_tensor).all())

        # basic array function, unsplit data, additional dimensions
        vector_data = [4.0, 5.0, 6.0]
        d = ht.array(vector_data, ndmin=3)
        self.assertIsInstance(d, ht.DNDarray)
        self.assertEqual(d.dtype, ht.float32)
        self.assertEqual(d.lshape, (3, 1, 1))
        self.assertEqual(d.gshape, (3, 1, 1))
        self.assertEqual(d.split, None)
        self.assertTrue(
            (d._DNDarray__array == torch.tensor(vector_data, device=device).reshape(-1, 1, 1)).all()
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
            (d._DNDarray__array == torch.tensor(vector_data, device=device).reshape(1, 1, -1)).all()
        )

        # distributed array, chunk local data (split)
        tensor_2d = ht.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], split=0)
        self.assertIsInstance(tensor_2d, ht.DNDarray)
        self.assertEqual(tensor_2d.dtype, ht.float32)
        self.assertEqual(tensor_2d.gshape, (3, 3))
        self.assertEqual(len(tensor_2d.lshape), 2)
        self.assertLessEqual(tensor_2d.lshape[0], 3)
        self.assertEqual(tensor_2d.lshape[1], 3)
        self.assertEqual(tensor_2d.split, 0)
        self.assertTrue(
            (tensor_2d._DNDarray__array == torch.tensor([1.0, 2.0, 3.0], device=device)).all()
        )

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
                for l in range(y):
                    if tensor_array[k][l] == 1:
                        return k, l
            return x, y

        shape = 5
        eye = ht.eye(shape, dtype=ht.uint8, split=1)
        self.assertIsInstance(eye, ht.DNDarray)
        self.assertEqual(eye.dtype, ht.uint8)
        self.assertEqual(eye.shape, (shape, shape))
        self.assertEqual(eye.split, 1)

        offset_x, offset_y = get_offset(eye._DNDarray__array)
        self.assertGreaterEqual(offset_x, 0)
        self.assertGreaterEqual(offset_y, 0)
        x, y = eye._DNDarray__array.shape
        for i in range(x):
            for j in range(y):
                expected = 1 if i - offset_x is j - offset_y else 0
                self.assertEqual(eye._DNDarray__array[i][j], expected)

        shape = (10, 20)
        eye = ht.eye(shape, dtype=ht.float32)
        self.assertIsInstance(eye, ht.DNDarray)
        self.assertEqual(eye.dtype, ht.float32)
        self.assertEqual(eye.shape, shape)
        self.assertEqual(eye.split, None)

        offset_x, offset_y = get_offset(eye._DNDarray__array)
        self.assertGreaterEqual(offset_x, 0)
        self.assertGreaterEqual(offset_y, 0)
        x, y = eye._DNDarray__array.shape
        for i in range(x):
            for j in range(y):
                expected = 1.0 if i - offset_x is j - offset_y else 0.0
                self.assertEqual(eye._DNDarray__array[i][j], expected)

        shape = (10,)
        eye = ht.eye(shape, dtype=ht.int32, split=0)
        self.assertIsInstance(eye, ht.DNDarray)
        self.assertEqual(eye.dtype, ht.int32)
        self.assertEqual(eye.shape, shape * 2)
        self.assertEqual(eye.split, 0)

        offset_x, offset_y = get_offset(eye._DNDarray__array)
        self.assertGreaterEqual(offset_x, 0)
        self.assertGreaterEqual(offset_y, 0)
        x, y = eye._DNDarray__array.shape
        for i in range(x):
            for j in range(y):
                expected = 1 if i - offset_x is j - offset_y else 0
                self.assertEqual(eye._DNDarray__array[i][j], expected)

    def test_full(self):
        # simple tensor
        data = ht.full((10, 2), 4)
        self.assertIsInstance(data, ht.DNDarray)
        self.assertEqual(data.shape, (10, 2))
        self.assertEqual(data.lshape, (10, 2))
        self.assertEqual(data.dtype, ht.float32)
        self.assertEqual(data._DNDarray__array.dtype, torch.float32)
        self.assertEqual(data.split, None)
        self.assertTrue(ht.allclose(data, ht.float32(4.0)))

        # non-standard dtype tensor
        data = ht.full((10, 2), 4, dtype=ht.int32)
        self.assertIsInstance(data, ht.DNDarray)
        self.assertEqual(data.shape, (10, 2))
        self.assertEqual(data.lshape, (10, 2))
        self.assertEqual(data.dtype, ht.int32)
        self.assertEqual(data._DNDarray__array.dtype, torch.int32)
        self.assertEqual(data.split, None)
        self.assertTrue(ht.allclose(data, ht.int32(4)))

        # split tensor
        data = ht.full((10, 2), 4, split=0)
        self.assertIsInstance(data, ht.DNDarray)
        self.assertEqual(data.shape, (10, 2))
        self.assertLessEqual(data.lshape[0], 10)
        self.assertEqual(data.lshape[1], 2)
        self.assertEqual(data.dtype, ht.float32)
        self.assertEqual(data._DNDarray__array.dtype, torch.float32)
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
        self.assertEqual(ascending._DNDarray__array.dtype, torch.float32)
        self.assertEqual(ascending.split, None)

        # simple inverse linear space
        descending = ht.linspace(-5, 3, num=100)
        self.assertIsInstance(descending, ht.DNDarray)
        self.assertEqual(descending.shape, (100,))
        self.assertLessEqual(descending.lshape[0], 100)
        self.assertEqual(descending.dtype, ht.float32)
        self.assertEqual(descending._DNDarray__array.dtype, torch.float32)
        self.assertEqual(descending.split, None)

        # split linear space
        split = ht.linspace(-5, 3, num=70, split=0)
        self.assertIsInstance(split, ht.DNDarray)
        self.assertEqual(split.shape, (70,))
        self.assertLessEqual(split.lshape[0], 70)
        self.assertEqual(split.dtype, ht.float32)
        self.assertEqual(split._DNDarray__array.dtype, torch.float32)
        self.assertEqual(split.split, 0)

        # with casted type
        casted = ht.linspace(-5, 3, num=70, dtype=ht.uint8, split=0)
        self.assertIsInstance(casted, ht.DNDarray)
        self.assertEqual(casted.shape, (70,))
        self.assertLessEqual(casted.lshape[0], 70)
        self.assertEqual(casted.dtype, ht.uint8)
        self.assertEqual(casted._DNDarray__array.dtype, torch.uint8)
        self.assertEqual(casted.split, 0)

        # retstep test
        result = ht.linspace(-5, 3, num=70, retstep=True, dtype=ht.uint8, split=0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        self.assertIsInstance(result[0], ht.DNDarray)
        self.assertEqual(result[0].shape, (70,))
        self.assertLessEqual(result[0].lshape[0], 70)
        self.assertEqual(result[0].dtype, ht.uint8)
        self.assertEqual(result[0]._DNDarray__array.dtype, torch.uint8)
        self.assertEqual(result[0].split, 0)

        self.assertIsInstance(result[1], float)
        self.assertEqual(result[1], 0.11594202898550725)

        # exceptions
        with self.assertRaises(ValueError):
            ht.linspace(-5, 3, split=1)
        with self.assertRaises(ValueError):
            ht.linspace(-5, 3, num=-1)
        with self.assertRaises(ValueError):
            ht.linspace(-5, 3, num=0)

    def test_logspace(self):
        # simple log space
        ascending = ht.logspace(-3, 5)
        self.assertIsInstance(ascending, ht.DNDarray)
        self.assertEqual(ascending.shape, (50,))
        self.assertLessEqual(ascending.lshape[0], 50)
        self.assertEqual(ascending.dtype, ht.float32)
        self.assertEqual(ascending._DNDarray__array.dtype, torch.float32)
        self.assertEqual(ascending.split, None)

        # simple inverse log space
        descending = ht.logspace(-5, 3, num=100)
        self.assertIsInstance(descending, ht.DNDarray)
        self.assertEqual(descending.shape, (100,))
        self.assertLessEqual(descending.lshape[0], 100)
        self.assertEqual(descending.dtype, ht.float32)
        self.assertEqual(descending._DNDarray__array.dtype, torch.float32)
        self.assertEqual(descending.split, None)

        # split log space
        split = ht.logspace(-5, 3, num=70, split=0)
        self.assertIsInstance(split, ht.DNDarray)
        self.assertEqual(split.shape, (70,))
        self.assertLessEqual(split.lshape[0], 70)
        self.assertEqual(split.dtype, ht.float32)
        self.assertEqual(split._DNDarray__array.dtype, torch.float32)
        self.assertEqual(split.split, 0)

        # with casted type
        casted = ht.logspace(-5, 3, num=70, dtype=ht.uint8, split=0)
        self.assertIsInstance(casted, ht.DNDarray)
        self.assertEqual(casted.shape, (70,))
        self.assertLessEqual(casted.lshape[0], 70)
        self.assertEqual(casted.dtype, ht.uint8)
        self.assertEqual(casted._DNDarray__array.dtype, torch.uint8)
        self.assertEqual(casted.split, 0)

        # base test
        base = ht.logspace(-5, 3, num=70, base=2.0)
        self.assertIsInstance(base, ht.DNDarray)
        self.assertEqual(base.shape, (70,))
        self.assertLessEqual(base.lshape[0], 70)
        self.assertEqual(base.dtype, ht.float32)
        self.assertEqual(base._DNDarray__array.dtype, torch.float32)
        self.assertEqual(base.split, None)

        # exceptions
        with self.assertRaises(ValueError):
            ht.logspace(-5, 3, split=1)
        with self.assertRaises(ValueError):
            ht.logspace(-5, 3, num=-1)
        with self.assertRaises(ValueError):
            ht.logspace(-5, 3, num=0)

    def test_ones(self):
        # scalar input
        simple_ones_float = ht.ones(3)
        self.assertIsInstance(simple_ones_float, ht.DNDarray)
        self.assertEqual(simple_ones_float.shape, (3,))
        self.assertEqual(simple_ones_float.lshape, (3,))
        self.assertEqual(simple_ones_float.split, None)
        self.assertEqual(simple_ones_float.dtype, ht.float32)
        self.assertEqual((simple_ones_float._DNDarray__array == 1).all().item(), 1)

        # different data type
        simple_ones_uint = ht.ones(5, dtype=ht.bool)
        self.assertIsInstance(simple_ones_uint, ht.DNDarray)
        self.assertEqual(simple_ones_uint.shape, (5,))
        self.assertEqual(simple_ones_uint.lshape, (5,))
        self.assertEqual(simple_ones_uint.split, None)
        self.assertEqual(simple_ones_uint.dtype, ht.bool)
        self.assertEqual((simple_ones_uint._DNDarray__array == 1).all().item(), 1)

        # multi-dimensional
        elaborate_ones_int = ht.ones((2, 3), dtype=ht.int32)
        self.assertIsInstance(elaborate_ones_int, ht.DNDarray)
        self.assertEqual(elaborate_ones_int.shape, (2, 3))
        self.assertEqual(elaborate_ones_int.lshape, (2, 3))
        self.assertEqual(elaborate_ones_int.split, None)
        self.assertEqual(elaborate_ones_int.dtype, ht.int32)
        self.assertEqual((elaborate_ones_int._DNDarray__array == 1).all().item(), 1)

        # split axis
        elaborate_ones_split = ht.ones((6, 4), dtype=ht.int32, split=0)
        self.assertIsInstance(elaborate_ones_split, ht.DNDarray)
        self.assertEqual(elaborate_ones_split.shape, (6, 4))
        self.assertLessEqual(elaborate_ones_split.lshape[0], 6)
        self.assertEqual(elaborate_ones_split.lshape[1], 4)
        self.assertEqual(elaborate_ones_split.split, 0)
        self.assertEqual(elaborate_ones_split.dtype, ht.int32)
        self.assertEqual((elaborate_ones_split._DNDarray__array == 1).all().item(), 1)

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
        self.assertEqual((like_int._DNDarray__array == 1).all().item(), 1)

        # sequence
        like_str = ht.ones_like("abc")
        self.assertIsInstance(like_str, ht.DNDarray)
        self.assertEqual(like_str.shape, (3,))
        self.assertEqual(like_str.lshape, (3,))
        self.assertEqual(like_str.split, None)
        self.assertEqual(like_str.dtype, ht.float32)
        self.assertEqual((like_str._DNDarray__array == 1).all().item(), 1)

        # elaborate tensor
        zeros = ht.zeros((2, 3), dtype=ht.uint8)
        like_zeros = ht.ones_like(zeros)
        self.assertIsInstance(like_zeros, ht.DNDarray)
        self.assertEqual(like_zeros.shape, (2, 3))
        self.assertEqual(like_zeros.lshape, (2, 3))
        self.assertEqual(like_zeros.split, None)
        self.assertEqual(like_zeros.dtype, ht.uint8)
        self.assertEqual((like_zeros._DNDarray__array == 1).all().item(), 1)

        # elaborate tensor with split
        zeros_split = ht.zeros((2, 3), dtype=ht.uint8, split=0)
        like_zeros_split = ht.ones_like(zeros_split)
        self.assertIsInstance(like_zeros_split, ht.DNDarray)
        self.assertEqual(like_zeros_split.shape, (2, 3))
        self.assertLessEqual(like_zeros_split.lshape[0], 2)
        self.assertEqual(like_zeros_split.lshape[1], 3)
        self.assertEqual(like_zeros_split.split, 0)
        self.assertEqual(like_zeros_split.dtype, ht.uint8)
        self.assertEqual((like_zeros_split._DNDarray__array == 1).all().item(), 1)

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
        self.assertEqual((simple_zeros_float._DNDarray__array == 0).all().item(), 1)

        # different data type
        simple_zeros_uint = ht.zeros(5, dtype=ht.bool)
        self.assertIsInstance(simple_zeros_uint, ht.DNDarray)
        self.assertEqual(simple_zeros_uint.shape, (5,))
        self.assertEqual(simple_zeros_uint.lshape, (5,))
        self.assertEqual(simple_zeros_uint.split, None)
        self.assertEqual(simple_zeros_uint.dtype, ht.bool)
        self.assertEqual((simple_zeros_uint._DNDarray__array == 0).all().item(), 1)

        # multi-dimensional
        elaborate_zeros_int = ht.zeros((2, 3), dtype=ht.int32)
        self.assertIsInstance(elaborate_zeros_int, ht.DNDarray)
        self.assertEqual(elaborate_zeros_int.shape, (2, 3))
        self.assertEqual(elaborate_zeros_int.lshape, (2, 3))
        self.assertEqual(elaborate_zeros_int.split, None)
        self.assertEqual(elaborate_zeros_int.dtype, ht.int32)
        self.assertEqual((elaborate_zeros_int._DNDarray__array == 0).all().item(), 1)

        # split axis
        elaborate_zeros_split = ht.zeros((6, 4), dtype=ht.int32, split=0)
        self.assertIsInstance(elaborate_zeros_split, ht.DNDarray)
        self.assertEqual(elaborate_zeros_split.shape, (6, 4))
        self.assertLessEqual(elaborate_zeros_split.lshape[0], 6)
        self.assertEqual(elaborate_zeros_split.lshape[1], 4)
        self.assertEqual(elaborate_zeros_split.split, 0)
        self.assertEqual(elaborate_zeros_split.dtype, ht.int32)
        self.assertEqual((elaborate_zeros_split._DNDarray__array == 0).all().item(), 1)

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
        self.assertEqual((like_int._DNDarray__array == 0).all().item(), 1)

        # sequence
        like_str = ht.zeros_like("abc")
        self.assertIsInstance(like_str, ht.DNDarray)
        self.assertEqual(like_str.shape, (3,))
        self.assertEqual(like_str.lshape, (3,))
        self.assertEqual(like_str.split, None)
        self.assertEqual(like_str.dtype, ht.float32)
        self.assertEqual((like_str._DNDarray__array == 0).all().item(), 1)

        # elaborate tensor
        ones = ht.ones((2, 3), dtype=ht.uint8)
        like_ones = ht.zeros_like(ones)
        self.assertIsInstance(like_ones, ht.DNDarray)
        self.assertEqual(like_ones.shape, (2, 3))
        self.assertEqual(like_ones.lshape, (2, 3))
        self.assertEqual(like_ones.split, None)
        self.assertEqual(like_ones.dtype, ht.uint8)
        self.assertEqual((like_ones._DNDarray__array == 0).all().item(), 1)

        # elaborate tensor with split
        ones_split = ht.ones((2, 3), dtype=ht.uint8, split=0)
        like_ones_split = ht.zeros_like(ones_split)
        self.assertIsInstance(like_ones_split, ht.DNDarray)
        self.assertEqual(like_ones_split.shape, (2, 3))
        self.assertLessEqual(like_ones_split.lshape[0], 2)
        self.assertEqual(like_ones_split.lshape[1], 3)
        self.assertEqual(like_ones_split.split, 0)
        self.assertEqual(like_ones_split.dtype, ht.uint8)
        self.assertEqual((like_ones_split._DNDarray__array == 0).all().item(), 1)

        # exceptions
        with self.assertRaises(TypeError):
            ht.zeros_like(ones, dtype="abc")
        with self.assertRaises(TypeError):
            ht.zeros_like(ones, split="axis")
