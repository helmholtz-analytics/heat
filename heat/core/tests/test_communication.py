import unittest
import torch

import heat as ht


class TestCommunication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = torch.tensor([
            [3, 2, 1],
            [4, 5, 6]
        ], dtype=torch.float32)

    def test_self_communicator(self):
        comm = ht.core.communication.MPI_SELF

        with self.assertRaises(ValueError):
            comm.chunk(self.data.shape, split=2)
        with self.assertRaises(ValueError):
            comm.chunk(self.data.shape, split=-3)

        offset, lshape, chunks = comm.chunk(self.data.shape, split=0)

        self.assertIsInstance(offset, int)
        self.assertEqual(offset, 0)

        self.assertIsInstance(lshape, tuple)
        self.assertEqual(len(lshape), len(self.data.shape))
        self.assertEqual(lshape, self.data.shape)

        self.assertIsInstance(chunks, tuple)
        self.assertEqual(len(chunks), len(self.data.shape))
        self.assertEqual(1, (self.data == self.data[chunks]).all().item())

    def test_lala(self):
        # if ht.core.MPI_WORLD.rank == 0:
        #     input = ht.arange(3 * 4 * 7)
        #     input._tensor__array = input._tensor__array.reshape(4, 3, 7).permute(1, 0, 2)
        #
        # else:
        #     input = ht.ones((3 * 4 * 7), dtype=ht.int32)
        #     input._tensor__array = input._tensor__array.reshape(4, 3, 7).permute(1, 0, 2)
        #
        # output = ht.zeros((4, 3, 7), dtype=ht.int32)
        # output._tensor__array = output._tensor__array.permute(1, 0, 2)
        # input.comm.Reduce(input, output, root=0)
        # print(output)

        # if ht.core.MPI_WORLD.rank == 0:
        #     input = ht.arange(3 * 4 * 7)
        #     input._tensor__array = input._tensor__array.reshape(3, 4, 7).permute(1, 0, 2)
        # else:
        #     input = ht.ones((3 * 4 * 7), dtype=ht.int32)
        #     input._tensor__array = input._tensor__array.reshape(3, 4, 7).permute(1, 0, 2)
        #
        # output = ht.zeros((3, 4, 7), dtype=ht.int32)
        # output._tensor__array = output._tensor__array.permute(1, 0, 2)
        # input.comm.Scan(input, output)
        # print(output)
        #
        # if ht.core.MPI_WORLD.rank == 0:
        #     input = ht.arange(3 * 4 * 7)
        #     input._tensor__array = input._tensor__array.reshape(3, 4, 7).permute(1, 0, 2)
        # else:
        #     input = ht.ones((3 * 4 * 7), dtype=ht.int32)
        #     input._tensor__array = input._tensor__array.reshape(3, 4, 7).permute(1, 0, 2)
        #
        # print(input)
        # input.comm.Bcast(input)
        # print(input)
        #
        # if ht.core.MPI_WORLD.rank == 0:
        #     input = ht.arange(3 * 4 * 7)
        #     input._tensor__array = input._tensor__array.reshape(3, 4, 7).permute(1, 0, 2)
        # else:
        #     input = ht.ones((3 * 4 * 7), dtype=ht.int32)
        #     input._tensor__array = input._tensor__array.reshape(3, 4, 7).permute(1, 0, 2)
        #
        # output = ht.zeros((3, 4, 7), dtype=ht.int32)
        # output._tensor__array = output._tensor__array.permute(1, 0, 2)
        # input.comm.Allreduce(input, output)
        # print(output)

        first_dim = ht.core.MPI_WORLD.size * 2
        input = ht.arange(2 * first_dim * 5)
        input._tensor__array = input._tensor__array.reshape(first_dim, 2, 5).permute(1, 0, 2)
        # input._tensor__array = input._tensor__array.reshape(first_dim, 4, 7)

        if ht.core.MPI_WORLD.rank == 0:
            print(input)

        # output = ht.zeros((3, 7, 4), dtype=ht.int32)
        # output._tensor__array = output._tensor__array.permute(0, 2, 1)
        output = ht.zeros((int(first_dim / ht.core.MPI_WORLD.size), 2, 5), dtype=ht.int32)
        # output._tensor__array = output._tensor__array#.permute(0, 2, 1)
        input.comm.Scatter(input, output, root=0)
        print(output)

    def test_mpi_communicator(self):
        comm = ht.core.communication.MPI_WORLD
        self.assertLess(comm.rank, comm.size)

        with self.assertRaises(ValueError):
            comm.chunk(self.data.shape, split=2)
        with self.assertRaises(ValueError):
            comm.chunk(self.data.shape, split=-3)

        offset, lshape, chunks = comm.chunk(self.data.shape, split=0)
        self.assertIsInstance(offset, int)
        self.assertGreaterEqual(offset, 0)
        self.assertLessEqual(offset, self.data.shape[0])

        self.assertIsInstance(lshape, tuple)
        self.assertEqual(len(lshape), len(self.data.shape))
        self.assertGreaterEqual(lshape[0], 0)
        self.assertLessEqual(lshape[0], self.data.shape[0])

        self.assertIsInstance(chunks, tuple)
        self.assertEqual(len(chunks), len(self.data.shape))

    def test_contiguous_memory_buffer(self):
        # vector heat tensor
        vector_data = ht.arange(1, 10)
        vector_out = ht.zeros_like(vector_data)

        # test that target and destination are not equal
        self.assertTrue((vector_data._tensor__array != vector_out._tensor__array).all())
        self.assertTrue(vector_data._tensor__array.is_contiguous())
        self.assertTrue(vector_out._tensor__array.is_contiguous())

        # send message to self that is received into a separate buffer afterwards
        vector_data.comm.Isend(vector_data, dest=vector_data.comm.rank)
        vector_out.comm.Recv(vector_out, source=vector_out.comm.rank)

        # check that after sending the data everything is equal
        self.assertTrue((vector_data._tensor__array == vector_out._tensor__array).all())
        self.assertTrue(vector_out._tensor__array.is_contiguous())

        # multi-dimensional torch tensor
        tensor_data = torch.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6) + 1
        tensor_out = torch.zeros_like(tensor_data)

        # test that target and destination are not equal
        self.assertTrue((tensor_data != tensor_out).all())
        self.assertTrue(tensor_data.is_contiguous())
        self.assertTrue(tensor_out.is_contiguous())

        # send message to self that is received into a separate buffer afterwards
        comm = ht.core.communication.MPI_WORLD
        comm.Isend(tensor_data, dest=comm.rank)
        comm.Recv(tensor_out, source=comm.rank)

        # check that after sending the data everything is equal
        self.assertTrue((tensor_data == tensor_out).all())
        self.assertTrue(tensor_out.is_contiguous())

    def test_non_contiguous_memory_buffer(self):
        # vector heat tensor
        non_contiguous_data = ht.ones((3, 2,)).T
        contiguous_out = ht.zeros_like(non_contiguous_data)

        # test that target and destination are not equal
        self.assertTrue((non_contiguous_data._tensor__array != contiguous_out._tensor__array).all())
        self.assertFalse(non_contiguous_data._tensor__array.is_contiguous())
        self.assertTrue(contiguous_out._tensor__array.is_contiguous())
        #
        # # send message to self that is received into a separate buffer afterwards
        # vector_data.comm.Isend(vector_data, dest=vector_data.comm.rank)
        # vector_out.comm.Recv(vector_out, source=vector_out.comm.rank)
        #
        # # check that after sending the data everything is equal
        # self.assertTrue((vector_data._tensor__array == vector_out._tensor__array).all())
        # self.assertTrue(vector_out._tensor__array.is_contiguous())
        #
        # # multi-dimensional torch tensor
        # tensor_data = torch.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6) + 1
        # tensor_out = torch.zeros_like(tensor_data)
        #
        # # test that target and destination are not equal
        # self.assertTrue((tensor_data != tensor_out).all())
        # self.assertTrue(tensor_data.is_contiguous())
        # self.assertTrue(tensor_out.is_contiguous())
        #
        # # send message to self that is received into a separate buffer afterwards
        # comm = ht.core.communication.MPI_WORLD
        # comm.Isend(tensor_data, dest=comm.rank)
        # comm.Recv(tensor_out, source=comm.rank)
        #
        # # check that after sending the data everything is equal
        # self.assertTrue((tensor_data == tensor_out).all())
        # self.assertTrue(tensor_out.is_contiguous())

    def test_cuda_aware_mpi(self):
        self.assertTrue(hasattr(ht.communication, 'CUDA_AWARE_MPI'))
