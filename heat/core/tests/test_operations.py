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

    def test_clip(self):
        elements = 20

        # float tensor
        float32_tensor = ht.arange(elements, dtype=ht.float32, split=0)
        clipped = float32_tensor.clip(5, 15)
        self.assertIsInstance(clipped, ht.tensor)
        self.assertEqual(clipped.dtype, ht.float32)
        self.assertEqual(clipped.sum(axis=0), 195)

        # long tensor
        int64_tensor = ht.arange(elements, dtype=ht.int64, split=0)
        clipped = int64_tensor.clip(4, 16)
        self.assertIsInstance(clipped, ht.tensor)
        self.assertEqual(clipped.dtype, ht.int64)
        self.assertEqual(clipped.sum(axis=0), 195)

        # test the exceptions
        with self.assertRaises(TypeError):
            ht.clip(torch.arange(10), 2, 5)
        with self.assertRaises(ValueError):
            ht.arange(20).clip(None, None)
        with self.assertRaises(TypeError):
            ht.clip(ht.arange(20), 5, 15, out=torch.arange(20))

    def test_copy(self):
        tensor = ht.ones(5)
        copied = tensor.copy()

        # test identity inequality and value equality
        self.assertIsNot(tensor, copied)
        self.assertIsNot(tensor._tensor__array, copied._tensor__array)
        self.assertTrue((tensor == copied)._tensor__array.all())

        # test exceptions
        with self.assertRaises(TypeError):
            ht.copy('hello world')

    def test_exp(self):
        elements = 10
        comparison = torch.arange(elements, dtype=torch.float64).exp()

        # exponential of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_exp = ht.exp(float32_tensor)
        self.assertIsInstance(float32_exp, ht.tensor)
        self.assertEqual(float32_exp.dtype, ht.float32)
        self.assertEqual(float32_exp.dtype, ht.float32)
        in_range = (float32_exp._tensor__array - comparison.type(torch.float32)) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # exponential of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_exp = ht.exp(float64_tensor)
        self.assertIsInstance(float64_exp, ht.tensor)
        self.assertEqual(float64_exp.dtype, ht.float64)
        self.assertEqual(float64_exp.dtype, ht.float64)
        in_range = (float64_exp._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # exponential of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_exp = ht.exp(int32_tensor)
        self.assertIsInstance(int32_exp, ht.tensor)
        self.assertEqual(int32_exp.dtype, ht.float64)
        self.assertEqual(int32_exp.dtype, ht.float64)
        in_range = (int32_exp._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # exponential of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_exp = ht.exp(int64_tensor)
        self.assertIsInstance(int64_exp, ht.tensor)
        self.assertEqual(int64_exp.dtype, ht.float64)
        self.assertEqual(int64_exp.dtype, ht.float64)
        in_range = (int64_exp._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.exp([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.exp('hello world')

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

    def test_log(self):
        elements = 15
        comparison = torch.arange(1, elements, dtype=torch.float64).log()

        # logarithm of float32
        float32_tensor = ht.arange(1, elements, dtype=ht.float32)
        float32_log = ht.log(float32_tensor)
        self.assertIsInstance(float32_log, ht.tensor)
        self.assertEqual(float32_log.dtype, ht.float32)
        self.assertEqual(float32_log.dtype, ht.float32)
        in_range = (float32_log._tensor__array - comparison.type(torch.float32)) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logarithm of float64
        float64_tensor = ht.arange(1, elements, dtype=ht.float64)
        float64_log = ht.log(float64_tensor)
        self.assertIsInstance(float64_log, ht.tensor)
        self.assertEqual(float64_log.dtype, ht.float64)
        self.assertEqual(float64_log.dtype, ht.float64)
        in_range = (float64_log._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logarithm of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(1, elements, dtype=ht.int32)
        int32_log = ht.log(int32_tensor)
        self.assertIsInstance(int32_log, ht.tensor)
        self.assertEqual(int32_log.dtype, ht.float64)
        self.assertEqual(int32_log.dtype, ht.float64)
        in_range = (int32_log._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logarithm of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(1, elements, dtype=ht.int64)
        int64_log = ht.log(int64_tensor)
        self.assertIsInstance(int64_log, ht.tensor)
        self.assertEqual(int64_log.dtype, ht.float64)
        self.assertEqual(int64_log.dtype, ht.float64)
        in_range = (int64_log._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.log([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.log('hello world')

    def test_max(self):
        data = ht.float32([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])

        comparison = torch.Tensor([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])       
        # check basic equivalence to torch.max()
        self.assertEqual(data.max(),comparison.max())
        self.assertTrue((ht.max(data,axis=0)._tensor__array[0] == comparison.max(0)[0]).all())
        self.assertIsInstance(ht.max(data,axis=1),ht.tensor)
        
        #TODO: check combinations of split and axis

        # check exceptions
        with self.assertRaises(NotImplementedError):
            data.max(axis=(0,1))
        with self.assertRaises(TypeError):
            data.max(axis=1.1)
        with self.assertRaises(ValueError):
            ht.max(data,axis=-4)

    def test_min(self):
        data = ht.float32([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])

        comparison = torch.Tensor([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])       
        #check basic equivalence to torch.max()
        self.assertEqual(data.min(),comparison.min())
        self.assertTrue((ht.min(data,axis=0)._tensor__array[0] == comparison.min(0)[0]).all())
        self.assertIsInstance(ht.min(data,axis=1),ht.tensor)
        
        #TODO: check combinations of split and axis

        # check exceptions
        with self.assertRaises(NotImplementedError):
            data.min(axis=(0,1))
        with self.assertRaises(TypeError):
            data.min(axis=1.1)
        with self.assertRaises(ValueError):
            ht.min(data,axis=-4)

    def test_sin(self):
        # base elements
        elements = 30
        comparison = torch.arange(elements, dtype=torch.float64).sin()

        # sine of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_sin = ht.sin(float32_tensor)
        self.assertIsInstance(float32_sin, ht.tensor)
        self.assertEqual(float32_sin.dtype, ht.float32)
        self.assertEqual(float32_sin.dtype, ht.float32)
        in_range = (float32_sin._tensor__array - comparison.type(torch.float32)) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # sine of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_sin = ht.sin(float64_tensor)
        self.assertIsInstance(float64_sin, ht.tensor)
        self.assertEqual(float64_sin.dtype, ht.float64)
        self.assertEqual(float64_sin.dtype, ht.float64)
        in_range = (float64_sin._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logarithm of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_sin = ht.sin(int32_tensor)
        self.assertIsInstance(int32_sin, ht.tensor)
        self.assertEqual(int32_sin.dtype, ht.float64)
        self.assertEqual(int32_sin.dtype, ht.float64)
        in_range = (int32_sin._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # logathm of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_sin = ht.sin(int64_tensor)
        self.assertIsInstance(int64_sin, ht.tensor)
        self.assertEqual(int64_sin.dtype, ht.float64)
        self.assertEqual(int64_sin.dtype, ht.float64)
        in_range = (int64_sin._tensor__array - comparison) < FLOAT_EPSILON
        self.assertTrue(in_range.all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.sin([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.sin('hello world')

    def test_sqrt(self):
        elements = 20
        comparison = torch.arange(elements, dtype=torch.float64).sqrt()

        # square roots of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_sqrt = ht.sqrt(float32_tensor)
        self.assertIsInstance(float32_sqrt, ht.tensor)
        self.assertEqual(float32_sqrt.dtype, ht.float32)
        self.assertEqual(float32_sqrt.dtype, ht.float32)
        self.assertTrue((float32_sqrt._tensor__array == comparison.type(torch.float32)).all())

        # square roots of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_sqrt = ht.sqrt(float64_tensor)
        self.assertIsInstance(float64_sqrt, ht.tensor)
        self.assertEqual(float64_sqrt.dtype, ht.float64)
        self.assertEqual(float64_sqrt.dtype, ht.float64)
        self.assertTrue((float64_sqrt._tensor__array == comparison).all())

        # square roots of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_sqrt = ht.sqrt(int32_tensor)
        self.assertIsInstance(int32_sqrt, ht.tensor)
        self.assertEqual(int32_sqrt.dtype, ht.float64)
        self.assertEqual(int32_sqrt.dtype, ht.float64)
        self.assertTrue((int32_sqrt._tensor__array == comparison).all())

        # square roots of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_sqrt = ht.sqrt(int64_tensor)
        self.assertIsInstance(int64_sqrt, ht.tensor)
        self.assertEqual(int64_sqrt.dtype, ht.float64)
        self.assertEqual(int64_sqrt.dtype, ht.float64)
        self.assertTrue((int64_sqrt._tensor__array == comparison).all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.sqrt([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.sqrt('hello world')

    def test_sqrt_method(self):
        elements = 25
        comparison = torch.arange(elements, dtype=torch.float64).sqrt()

        # square roots of float32
        float32_sqrt = ht.arange(elements, dtype=ht.float32).sqrt()
        self.assertIsInstance(float32_sqrt, ht.tensor)
        self.assertEqual(float32_sqrt.dtype, ht.float32)
        self.assertEqual(float32_sqrt.dtype, ht.float32)
        self.assertTrue((float32_sqrt._tensor__array == comparison.type(torch.float32)).all())

        # square roots of float64
        float64_sqrt = ht.arange(elements, dtype=ht.float64).sqrt()
        self.assertIsInstance(float64_sqrt, ht.tensor)
        self.assertEqual(float64_sqrt.dtype, ht.float64)
        self.assertEqual(float64_sqrt.dtype, ht.float64)
        self.assertTrue((float64_sqrt._tensor__array == comparison).all())

        # square roots of ints, automatic conversion to intermediate floats
        int32_sqrt = ht.arange(elements, dtype=ht.int32).sqrt()
        self.assertIsInstance(int32_sqrt, ht.tensor)
        self.assertEqual(int32_sqrt.dtype, ht.float64)
        self.assertEqual(int32_sqrt.dtype, ht.float64)
        self.assertTrue((int32_sqrt._tensor__array == comparison).all())

        # square roots of longs, automatic conversion to intermediate floats
        int64_sqrt = ht.arange(elements, dtype=ht.int64).sqrt()
        self.assertIsInstance(int64_sqrt, ht.tensor)
        self.assertEqual(int64_sqrt.dtype, ht.float64)
        self.assertEqual(int64_sqrt.dtype, ht.float64)
        self.assertTrue((int64_sqrt._tensor__array == comparison).all())

        # check exceptions
        with self.assertRaises(TypeError):
            ht.sqrt([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.sqrt('hello world')

    def test_sqrt_out_of_place(self):
        elements = 30
        output_shape = (3, elements)
        number_range = ht.arange(elements, dtype=ht.float32)
        output_buffer = ht.zeros(output_shape, dtype=ht.float32)

        # square roots
        float32_sqrt = ht.sqrt(number_range, out=output_buffer)
        comparison = torch.arange(elements, dtype=torch.float32).sqrt()

        # check whether the input range remain unchanged
        self.assertIsInstance(number_range, ht.tensor)
        self.assertEqual(number_range.sum(axis=0), 190)  # gaussian sum
        self.assertEqual(number_range.gshape, (elements,))

        # check whether the output buffer still has the correct shape
        self.assertIsInstance(float32_sqrt, ht.tensor)
        self.assertEqual(float32_sqrt.dtype, ht.float32)
        self.assertEqual(float32_sqrt._tensor__array.shape, output_shape)
        for row in range(output_shape[0]):
            self.assertTrue((float32_sqrt._tensor__array[row] == comparison).all())

        # exception
        with self.assertRaises(TypeError):
            ht.sqrt(number_range, 'hello world')

    def test_tril(self):
        local_ones = ht.ones((5,))

        # 1D case, no offset, data is not split, module-level call
        result = ht.tril(local_ones)
        comparison = torch.ones((5, 5,)).tril()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.lshape, (5, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        # 1D case, positive offset, data is not split, module-level call
        result = ht.tril(local_ones, k=2)
        comparison = torch.ones((5, 5,)).tril(diagonal=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.lshape, (5, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        # 1D case, negative offset, data is not split, module-level call
        result = ht.tril(local_ones, k=-2)
        comparison = torch.ones((5, 5,)).tril(diagonal=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.lshape, (5, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        local_ones = ht.ones((4, 5,))

        # 2D case, no offset, data is not split, method
        result = local_ones.tril()
        comparison = torch.ones((4, 5,)).tril()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.lshape, (4, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        # 2D case, positive offset, data is not split, method
        result = local_ones.tril(k=2)
        comparison = torch.ones((4, 5,)).tril(diagonal=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.lshape, (4, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        # 2D case, negative offset, data is not split, method
        result = local_ones.tril(k=-2)
        comparison = torch.ones((4, 5,)).tril(diagonal=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.lshape, (4, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        local_ones = ht.ones((3, 4, 5, 6))

        # 2D+ case, no offset, data is not split, module-level call
        result = local_ones.tril()
        comparison = torch.ones((5, 6,)).tril()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (3, 4, 5, 6,))
        self.assertEqual(result.lshape, (3, 4, 5, 6,))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._tensor__array[i, j] == comparison).all())

        # 2D+ case, positive offset, data is not split, module-level call
        result = local_ones.tril(k=2)
        comparison = torch.ones((5, 6,)).tril(diagonal=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (3, 4, 5, 6,))
        self.assertEqual(result.lshape, (3, 4, 5, 6,))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._tensor__array[i, j] == comparison).all())

        # # 2D+ case, negative offset, data is not split, module-level call
        result = local_ones.tril(k=-2)
        comparison = torch.ones((5, 6,)).tril(diagonal=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (3, 4, 5, 6,))
        self.assertEqual(result.lshape, (3, 4, 5, 6,))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._tensor__array[i, j] == comparison).all())

        distributed_ones = ht.ones((5,), split=0)

        # 1D case, no offset, data is split, method
        result = distributed_ones.tril()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertTrue(result.sum(), 15)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 1)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 0)

        # 1D case, positive offset, data is split, method
        result = distributed_ones.tril(k=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 22)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 1)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 0)

        # 1D case, negative offset, data is split, method
        result = distributed_ones.tril(k=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 6)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 1)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 0)

        distributed_ones = ht.ones((4, 5,), split=0)

        # 2D case, no offset, data is horizontally split, method
        result = distributed_ones.tril()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 10)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[0, -1] == 0)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[-1, 0] == 1)

        # 2D case, positive offset, data is horizontally split, method
        result = distributed_ones.tril(k=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 17)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[0, -1] == 0)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[-1, 0] == 1)

        # 2D case, negative offset, data is horizontally split, method
        result = distributed_ones.tril(k=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 3)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[0, -1] == 0)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[-1, 0] == 1)

        distributed_ones = ht.ones((4, 5,), split=1)

        # 2D case, no offset, data is vertically split, method
        result = distributed_ones.tril()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 10)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 1)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 0)

        # 2D case, positive offset, data is horizontally split, method
        result = distributed_ones.tril(k=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 17)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 1)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 0)

        # 2D case, negative offset, data is horizontally split, method
        result = distributed_ones.tril(k=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 3)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 1)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 0)

    def test_triu(self):
        local_ones = ht.ones((5,))

        # 1D case, no offset, data is not split, module-level call
        result = ht.triu(local_ones)
        comparison = torch.ones((5, 5,)).triu()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.lshape, (5, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        # 1D case, positive offset, data is not split, module-level call
        result = ht.triu(local_ones, k=2)
        comparison = torch.ones((5, 5,)).triu(diagonal=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.lshape, (5, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        # 1D case, negative offset, data is not split, module-level call
        result = ht.triu(local_ones, k=-2)
        comparison = torch.ones((5, 5,)).triu(diagonal=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.lshape, (5, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        local_ones = ht.ones((4, 5,))

        # 2D case, no offset, data is not split, method
        result = local_ones.triu()
        comparison = torch.ones((4, 5,)).triu()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.lshape, (4, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        # 2D case, positive offset, data is not split, method
        result = local_ones.triu(k=2)
        comparison = torch.ones((4, 5,)).triu(diagonal=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.lshape, (4, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        # 2D case, negative offset, data is not split, method
        result = local_ones.triu(k=-2)
        comparison = torch.ones((4, 5,)).triu(diagonal=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.lshape, (4, 5,))
        self.assertEqual(result.split, None)
        self.assertTrue((result._tensor__array == comparison).all())

        local_ones = ht.ones((3, 4, 5, 6))

        # 2D+ case, no offset, data is not split, module-level call
        result = local_ones.triu()
        comparison = torch.ones((5, 6,)).triu()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (3, 4, 5, 6,))
        self.assertEqual(result.lshape, (3, 4, 5, 6,))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._tensor__array[i, j] == comparison).all())

        # 2D+ case, positive offset, data is not split, module-level call
        result = local_ones.triu(k=2)
        comparison = torch.ones((5, 6,)).triu(diagonal=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (3, 4, 5, 6,))
        self.assertEqual(result.lshape, (3, 4, 5, 6,))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._tensor__array[i, j] == comparison).all())

        # # 2D+ case, negative offset, data is not split, module-level call
        result = local_ones.triu(k=-2)
        comparison = torch.ones((5, 6,)).triu(diagonal=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (3, 4, 5, 6,))
        self.assertEqual(result.lshape, (3, 4, 5, 6,))
        self.assertEqual(result.split, None)
        for i in range(3):
            for j in range(4):
                self.assertTrue((result._tensor__array[i, j] == comparison).all())

        distributed_ones = ht.ones((5,), split=0)

        # 1D case, no offset, data is split, method
        result = distributed_ones.triu()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertTrue(result.sum(), 15)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 0)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 1)

        # 1D case, positive offset, data is split, method
        result = distributed_ones.triu(k=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 6)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 0)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 1)

        # 1D case, negative offset, data is split, method
        result = distributed_ones.triu(k=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (5, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 5)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 22)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 0)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 1)

        distributed_ones = ht.ones((4, 5,), split=0)

        # 2D case, no offset, data is horizontally split, method
        result = distributed_ones.triu()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 14)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[0, -1] == 1)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[-1, 0] == 0)

        # # 2D case, positive offset, data is horizontally split, method
        result = distributed_ones.triu(k=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 6)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[0, -1] == 1)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[-1, 0] == 0)

        # # 2D case, negative offset, data is horizontally split, method
        result = distributed_ones.triu(k=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 0)
        self.assertLessEqual(result.lshape[0], 4)
        self.assertEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 19)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[0, -1] == 1)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[-1, 0] == 0)

        distributed_ones = ht.ones((4, 5,), split=1)

        # 2D case, no offset, data is vertically split, method
        result = distributed_ones.triu()
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 14)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 0)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 1)

        # 2D case, positive offset, data is horizontally split, method
        result = distributed_ones.triu(k=2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 6)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 0)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 1)

        # 2D case, negative offset, data is horizontally split, method
        result = distributed_ones.triu(k=-2)
        self.assertIsInstance(result, ht.tensor)
        self.assertEqual(result.shape, (4, 5,))
        self.assertEqual(result.split, 1)
        self.assertEqual(result.lshape[0], 4)
        self.assertLessEqual(result.lshape[1], 5)
        self.assertEqual(result.sum(), 19)
        if result.comm.rank == 0:
            self.assertTrue(result._tensor__array[-1, 0] == 0)
        if result.comm.rank == result.comm.size - 1:
            self.assertTrue(result._tensor__array[0, -1] == 1)
