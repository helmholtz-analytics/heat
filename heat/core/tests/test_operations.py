import unittest
import torch

import heat as ht

FLOAT_EPSILON = 1e-4



class TestOperations(unittest.TestCase):
    def test_all(self):
        array_len = 9

        # check all over all float elements of 1d tensor locally 
        ones_noaxis = ht.ones(array_len)
        x = (ones_noaxis == 1).all()

        self.assertIsInstance(x, ht.tensor)
        self.assertEqual(x.shape, (1,))
        self.assertEqual(x.lshape, (1,))
        self.assertEqual(x.dtype, ht.bool)
        self.assertEqual(x._tensor__array.dtype, torch.uint8)
        self.assertEqual(x.split, None)
        self.assertEqual(x._tensor__array, 1)

        out_noaxis = ht.zeros((1,))
        ht.all(ones_noaxis, out=out_noaxis)
        self.assertEqual(out_noaxis._tensor__array, 1)

        # check all over all float elements of split 1d tensor
        ones_noaxis_split = ht.ones(array_len, split=0)
        floats_is_one = ones_noaxis_split.all()

        self.assertIsInstance(floats_is_one, ht.tensor)
        self.assertEqual(floats_is_one.shape, (1,))
        self.assertEqual(floats_is_one.lshape, (1,))
        self.assertEqual(floats_is_one.dtype, ht.bool)
        self.assertEqual(floats_is_one._tensor__array.dtype, torch.uint8)
        self.assertEqual(floats_is_one.split, None)
        self.assertEqual(floats_is_one._tensor__array, 1)

        out_noaxis = ht.zeros((1,))
        ht.all(ones_noaxis_split, out=out_noaxis)
        self.assertEqual(out_noaxis._tensor__array, 1)

        # check all over all integer elements of 1d tensor locally
        ones_noaxis_int = ht.ones(array_len).astype(ht.int)
        int_is_one = ones_noaxis_int.all()

        self.assertIsInstance(int_is_one, ht.tensor)
        self.assertEqual(int_is_one.shape, (1,))
        self.assertEqual(int_is_one.lshape, (1,))
        self.assertEqual(int_is_one.dtype, ht.bool)
        self.assertEqual(int_is_one._tensor__array.dtype, torch.uint8)
        self.assertEqual(int_is_one.split, None)
        self.assertEqual(int_is_one._tensor__array, 1)

        out_noaxis = ht.zeros((1,))
        ht.all(ones_noaxis_int, out=out_noaxis)
        self.assertEqual(out_noaxis._tensor__array, 1)

        # check all over all integer elements of split 1d tensor
        ones_noaxis_split_int = ht.ones(array_len, split=0).astype(ht.int)
        split_int_is_one = ones_noaxis_split_int.all()

        self.assertIsInstance(split_int_is_one, ht.tensor)
        self.assertEqual(split_int_is_one.shape, (1,))
        self.assertEqual(split_int_is_one.lshape, (1,))
        self.assertEqual(split_int_is_one.dtype, ht.bool)
        self.assertEqual(split_int_is_one._tensor__array.dtype, torch.uint8)
        self.assertEqual(split_int_is_one.split, None)
        self.assertEqual(split_int_is_one._tensor__array, 1)

        out_noaxis = ht.zeros((1,))
        ht.all(ones_noaxis_split_int, out=out_noaxis)
        self.assertEqual(out_noaxis._tensor__array, 1)

        # check all over all float elements of 3d tensor locally
        ones_noaxis_volume = ht.ones((3, 3, 3))
        volume_is_one = ones_noaxis_volume.all()

        self.assertIsInstance(volume_is_one, ht.tensor)
        self.assertEqual(volume_is_one.shape, (1,))
        self.assertEqual(volume_is_one.lshape, (1,))
        self.assertEqual(volume_is_one.dtype, ht.bool)
        self.assertEqual(volume_is_one._tensor__array.dtype, torch.uint8)
        self.assertEqual(volume_is_one.split, None)
        self.assertEqual(volume_is_one._tensor__array, 1)

        out_noaxis = ht.zeros((1,))
        ht.all(ones_noaxis_volume, out=out_noaxis)
        self.assertEqual(out_noaxis._tensor__array, 1)

        # check sequence is not all one
        sequence = ht.arange(array_len)
        sequence_is_one = sequence.all()

        self.assertIsInstance(sequence_is_one, ht.tensor)
        self.assertEqual(sequence_is_one.shape, (1,))
        self.assertEqual(sequence_is_one.lshape, (1,))
        self.assertEqual(sequence_is_one.dtype, ht.bool)
        self.assertEqual(sequence_is_one._tensor__array.dtype, torch.uint8)
        self.assertEqual(sequence_is_one.split, None)
        self.assertEqual(sequence_is_one._tensor__array, 0)

        out_noaxis = ht.zeros((1,))
        ht.all(sequence, out=out_noaxis)
        self.assertEqual(out_noaxis._tensor__array, 0)

        # check all over all float elements of split 3d tensor
        ones_noaxis_split_axis = ht.ones((3, 3, 3), split=0)
        float_volume_is_one = ones_noaxis_split_axis.all(axis=0)

        self.assertIsInstance(float_volume_is_one, ht.tensor)
        self.assertEqual(float_volume_is_one.shape, (1, 3, 3))
        self.assertEqual(float_volume_is_one.all(axis=1).dtype, ht.bool)
        self.assertEqual(float_volume_is_one._tensor__array.dtype, torch.uint8)
        self.assertEqual(float_volume_is_one.split, None)

        out_noaxis = ht.zeros((1, 3, 3,))
        ht.all(ones_noaxis_split_axis, axis=0, out=out_noaxis)

        # check all over all float elements of split 5d tensor with negative axis
        ones_noaxis_split_axis_neg = ht.zeros((1, 2, 3, 4, 5), split=1)
        float_5d_is_one = ones_noaxis_split_axis_neg.all(axis=-2)

        self.assertIsInstance(float_5d_is_one, ht.tensor)
        self.assertEqual(float_5d_is_one.shape, (1, 2, 3, 1, 5))
        self.assertEqual(float_5d_is_one.dtype, ht.bool)
        self.assertEqual(float_5d_is_one._tensor__array.dtype, torch.uint8)
        self.assertEqual(float_5d_is_one.split, 1)

        out_noaxis = ht.zeros((1, 2, 3, 1, 5))
        ht.all(ones_noaxis_split_axis_neg, axis=-2, out=out_noaxis)

        # exceptions
        with self.assertRaises(ValueError):
            ht.ones(array_len).all(axis=1)
        with self.assertRaises(ValueError):
            ht.ones(array_len).all(axis=-2)
        with self.assertRaises(ValueError):
            ht.ones((4, 4)).all(axis=0, out=out_noaxis)
        with self.assertRaises(TypeError):
            ht.ones(array_len).all(axis='bad_axis_type')

    def test_argmin(self):
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

        # # check basics
        # #self.assertTrue((ht.argmin(data, axis=0)._tensor__array == comparison.argmin(0)).all())
        # self.assertIsInstance(ht.argmin(data, axis=1), ht.tensor)
        # self.assertIsInstance(data.argmin(), ht.tensor)

        # # check combinations of split and axis
        # torch.manual_seed(1)
        # random_data = ht.random.randn(3, 3, 3)
        # torch.manual_seed(1)
        # random_data_split = ht.random.randn(3, 3, 3, split=0)
        
        # self.assertTrue((ht.argmin(random_data, axis=0)._tensor__array == random_data_split.argmin(axis=0)._tensor__array).all())
        # self.assertTrue((ht.argmin(random_data, axis=1)._tensor__array == random_data_split.argmin(axis=1)._tensor__array).all())
        # self.assertIsInstance(ht.argmin(random_data_split, axis=1), ht.tensor)
        # self.assertIsInstance(random_data_split.argmin(), ht.tensor)

        # # check argmin over all float elements of 3d tensor locally
        # self.assertEqual(random_data.argmin().shape, (1,))
        # self.assertEqual(random_data.argmin().lshape, (1,))
        # self.assertEqual(random_data.argmin().dtype, ht.int64)
        # self.assertEqual(random_data.argmin().split, None)

        # # check argmin over all float elements of splitted 3d tensor 
        # self.assertIsInstance(random_data_split.argmin(axis=1), ht.tensor)
        # self.assertEqual(random_data_split.argmin(axis=1).shape, (3, 1, 3))
        # self.assertEqual(random_data_split.argmin().split, None)

        # # check argmin over all float elements of splitted 5d tensor with negative axis 
        # random_data_split_neg = ht.random.randn(1, 2, 3, 4, 5, split=1)
        # self.assertIsInstance(random_data_split_neg.argmin(axis=-2), ht.tensor)
        # self.assertEqual(random_data_split_neg.argmin(axis=-2).shape, (1, 2, 3, 1, 5))
        # self.assertEqual(random_data_split_neg.argmin(axis=-2).dtype, ht.int64) 
        # self.assertEqual(random_data_split_neg.argmin().split, None)    
           
        # # check exceptions
        # with self.assertRaises(NotImplementedError):
        #     data.argmin(axis=(0,1))
        # with self.assertRaises(TypeError):
        #     data.argmin(axis=1.1)
        # with self.assertRaises(TypeError):
        #     data.argmin(axis='y')
        # with self.assertRaises(ValueError):
        #     ht.argmin(data, axis=-4)

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


    def test_transpose(self):
        # vector transpose, not distributed
        vector = ht.arange(10)
        vector_t = vector.T
        self.assertIsInstance(vector_t, ht.tensor)
        self.assertEqual(vector_t.dtype, ht.int32)
        self.assertEqual(vector_t.split, None)
        self.assertEqual(vector_t.shape, (10,))

        # simple matrix transpose, not distributed
        simple_matrix = ht.zeros((2, 4))
        simple_matrix_t = simple_matrix.transpose()
        self.assertIsInstance(simple_matrix_t, ht.tensor)
        self.assertEqual(simple_matrix_t.dtype, ht.float32)
        self.assertEqual(simple_matrix_t.split, None)
        self.assertEqual(simple_matrix_t.shape, (4, 2,))
        self.assertEqual(simple_matrix_t._tensor__array.shape, (4, 2,))

        # 4D array, not distributed, with given axis
        array_4d = ht.zeros((2, 3, 4, 5))
        array_4d_t = ht.transpose(array_4d, axes=(-1, 0, 2, 1))
        self.assertIsInstance(array_4d_t, ht.tensor)
        self.assertEqual(array_4d_t.dtype, ht.float32)
        self.assertEqual(array_4d_t.split, None)
        self.assertEqual(array_4d_t.shape, (5, 2, 4, 3,))
        self.assertEqual(array_4d_t._tensor__array.shape, (5, 2, 4, 3,))

        # vector transpose, distributed
        vector_split = ht.arange(10, split=0)
        vector_split_t = vector_split.T
        self.assertIsInstance(vector_split_t, ht.tensor)
        self.assertEqual(vector_split_t.dtype, ht.int32)
        self.assertEqual(vector_split_t.split, 0)
        self.assertEqual(vector_split_t.shape, (10,))
        self.assertLessEqual(vector_split_t.lshape[0], 10)

        # matrix transpose, distributed
        matrix_split = ht.ones((10, 20,), split=1)
        matrix_split_t = matrix_split.transpose()
        self.assertIsInstance(matrix_split_t, ht.tensor)
        self.assertEqual(matrix_split_t.dtype, ht.float32)
        self.assertEqual(matrix_split_t.split, 0)
        self.assertEqual(matrix_split_t.shape, (20, 10,))
        self.assertLessEqual(matrix_split_t.lshape[0], 20)
        self.assertEqual(matrix_split_t.lshape[1], 10)

        # 4D array, distributed
        array_4d_split = ht.ones((3, 4, 5, 6,), split=3)
        array_4d_split_t = ht.transpose(array_4d_split, axes=(1, 0, 3, 2,))
        self.assertIsInstance(array_4d_t, ht.tensor)
        self.assertEqual(array_4d_split_t.dtype, ht.float32)
        self.assertEqual(array_4d_split_t.split, 2)
        self.assertEqual(array_4d_split_t.shape, (4, 3, 6, 5,))

        self.assertEqual(array_4d_split_t.lshape[0], 4)
        self.assertEqual(array_4d_split_t.lshape[1], 3)
        self.assertLessEqual(array_4d_split_t.lshape[2], 6)
        self.assertEqual(array_4d_split_t.lshape[3], 5)

        # exceptions
        with self.assertRaises(TypeError):
            ht.transpose(1)
        with self.assertRaises(ValueError):
            ht.transpose(ht.zeros((2, 3,)), axes=1.0)
        with self.assertRaises(ValueError):
            ht.transpose(ht.zeros((2, 3,)), axes=(-1,))
        with self.assertRaises(TypeError):
            ht.zeros((2, 3,)).transpose(axes='01')
        with self.assertRaises(TypeError):
            ht.zeros((2, 3,)).transpose(axes=(0, 1.0))
        with self.assertRaises(ValueError):
            ht.zeros((2, 3,)).transpose(axes=(0, 3))

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



