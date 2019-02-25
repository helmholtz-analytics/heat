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
        # non-contiguous source
        non_contiguous_data = ht.ones((3, 2,)).T
        contiguous_out = ht.zeros_like(non_contiguous_data)

        # test that target and destination are not equal
        self.assertTrue((non_contiguous_data._tensor__array != contiguous_out._tensor__array).all())
        self.assertFalse(non_contiguous_data._tensor__array.is_contiguous())
        self.assertTrue(contiguous_out._tensor__array.is_contiguous())

        # send message to self that is received into a separate buffer afterwards
        non_contiguous_data.comm.Isend(non_contiguous_data, dest=non_contiguous_data.comm.rank)
        contiguous_out.comm.Recv(contiguous_out, source=contiguous_out.comm.rank)

        # check that after sending the data everything is equal
        self.assertTrue((non_contiguous_data._tensor__array == contiguous_out._tensor__array).all())
        self.assertTrue(contiguous_out._tensor__array.is_contiguous())

        # non-contiguous destination
        contiguous_data = ht.ones((3, 2,))
        non_contiguous_out = ht.zeros((2, 3,)).T

        # test that target and destination are not equal
        self.assertTrue((contiguous_data._tensor__array != non_contiguous_out._tensor__array).all())
        self.assertTrue(contiguous_data._tensor__array.is_contiguous())
        self.assertFalse(non_contiguous_out._tensor__array.is_contiguous())

        # send message to self that is received into a separate buffer afterwards
        contiguous_data.comm.Isend(contiguous_data, dest=contiguous_data.comm.rank)
        non_contiguous_out.comm.Recv(non_contiguous_out, source=non_contiguous_out.comm.rank)

        # check that after sending the data everything is equal
        self.assertTrue((contiguous_data._tensor__array == non_contiguous_out._tensor__array).all())
        self.assertFalse(non_contiguous_out._tensor__array.is_contiguous())

        # non-contiguous destination
        both_non_contiguous_data = ht.ones((3, 2,)).T
        both_non_contiguous_out = ht.zeros((3, 2,)).T

        # test that target and destination are not equal
        self.assertTrue((both_non_contiguous_data._tensor__array != both_non_contiguous_out._tensor__array).all())
        self.assertFalse(both_non_contiguous_data._tensor__array.is_contiguous())
        self.assertFalse(both_non_contiguous_out._tensor__array.is_contiguous())

        # send message to self that is received into a separate buffer afterwards
        both_non_contiguous_data.comm.Isend(both_non_contiguous_data, dest=both_non_contiguous_data.comm.rank)
        both_non_contiguous_out.comm.Recv(both_non_contiguous_out, source=both_non_contiguous_out.comm.rank)

        # check that after sending the data everything is equal
        self.assertTrue((both_non_contiguous_data._tensor__array == both_non_contiguous_out._tensor__array).all())
        self.assertFalse(both_non_contiguous_out._tensor__array.is_contiguous())

    def test_bcast(self):
        # contiguous data
        data = ht.arange(10, dtype=ht.int64)
        if ht.MPI_WORLD.rank != 0:
            data = ht.zeros_like(data, dtype=ht.int64)

        # broadcast data to all nodes
        self.assertTrue(data._tensor__array.is_contiguous())
        data.comm.Bcast(data, root=0)

        # assert output is equal
        self.assertTrue(data._tensor__array.is_contiguous())
        self.assertTrue((data._tensor__array == torch.arange(10)).all())

        # non-contiguous data
        data = ht.ones((2, 5,), dtype=ht.float32).T
        if ht.MPI_WORLD.rank != 0:
            data = ht.zeros((2, 5,), dtype=ht.float32).T

        # broadcast data to all nodes
        self.assertFalse(data._tensor__array.is_contiguous())
        data.comm.Bcast(data, root=0)

        # assert output is equal
        self.assertFalse(data._tensor__array.is_contiguous())
        self.assertTrue((data._tensor__array == torch.ones((5, 2,), dtype=torch.float32)).all())

    def test_reduce(self):
        # contiguous data
        data = ht.ones((10, 2,), dtype=ht.int32)
        out = ht.zeros_like(data)

        # reduce across all nodes
        self.assertTrue(data._tensor__array.is_contiguous())
        self.assertTrue(out._tensor__array.is_contiguous())
        data.comm.Reduce(data, out, op=ht.MPI.SUM, root=0)

        # check the reduction result
        self.assertTrue(data._tensor__array.is_contiguous())
        self.assertTrue(out._tensor__array.is_contiguous())
        if data.comm.rank == 0:
            self.assertTrue((out._tensor__array == data.comm.size).all())

        # non-contiguous data
        data = ht.ones((10, 2,), dtype=ht.int32).T
        out = ht.zeros_like(data)

        # reduce across all nodes
        self.assertFalse(data._tensor__array.is_contiguous())
        self.assertTrue(out._tensor__array.is_contiguous())
        data.comm.Reduce(data, out, op=ht.MPI.SUM, root=0)

        # check the reduction result
        # the data tensor will be contiguous after the reduction
        # MPI enforces the same data type for send and receive buffer
        # the reduction implementation takes care of making the internal Torch storage consistent
        self.assertTrue(data._tensor__array.is_contiguous())
        self.assertTrue(out._tensor__array.is_contiguous())
        if data.comm.rank == 0:
            self.assertTrue((out._tensor__array == data.comm.size).all())

        # non-contiguous output
        data = ht.ones((10, 2,), dtype=ht.int32)
        out = ht.zeros((2, 10), dtype=ht.int32).T

        # reduce across all nodes
        self.assertTrue(data._tensor__array.is_contiguous())
        self.assertFalse(out._tensor__array.is_contiguous())
        data.comm.Reduce(data, out, op=ht.MPI.SUM, root=0)

        # check the reduction result
        # the data tensor will be contiguous after the reduction
        # MPI enforces the same data type for send and receive buffer
        # the reduction implementation takes care of making the internal Torch storage consistent
        self.assertTrue(data._tensor__array.is_contiguous())
        self.assertTrue(out._tensor__array.is_contiguous())
        if data.comm.rank == 0:
            self.assertTrue((out._tensor__array == data.comm.size).all())

    def test_scan(self):
        # contiguous data
        data = ht.ones((5, 3,), dtype=ht.float64)
        out = ht.zeros_like(data)

        # reduce across all nodes
        self.assertTrue(data._tensor__array.is_contiguous())
        self.assertTrue(out._tensor__array.is_contiguous())
        data.comm.Scan(data, out)

        # check the reduction result
        self.assertTrue(data._tensor__array.is_contiguous())
        self.assertTrue(out._tensor__array.is_contiguous())
        self.assertTrue((out._tensor__array == data.comm.rank + 1).all())

        # non-contiguous data
        data = ht.ones((5, 3,), dtype=ht.float64).T
        out = ht.zeros_like(data)

        # reduce across all nodes
        self.assertFalse(data._tensor__array.is_contiguous())
        self.assertTrue(out._tensor__array.is_contiguous())
        data.comm.Scan(data, out)

        # check the reduction result
        # the data tensor will be contiguous after the reduction
        # MPI enforces the same data type for send and receive buffer
        # the reduction implementation takes care of making the internal Torch storage consistent
        self.assertTrue(data._tensor__array.is_contiguous())
        self.assertTrue(out._tensor__array.is_contiguous())
        self.assertTrue((out._tensor__array == data.comm.rank + 1).all())

        # non-contiguous output
        data = ht.ones((5, 3,), dtype=ht.float64)
        out = ht.zeros((3, 5), dtype=ht.float64).T

        # reduce across all nodes
        self.assertTrue(data._tensor__array.is_contiguous())
        self.assertFalse(out._tensor__array.is_contiguous())
        data.comm.Scan(data, out)

        # check the reduction result
        # the data tensor will be contiguous after the reduction
        # MPI enforces the same data type for send and receive buffer
        # the reduction implementation takes care of making the internal Torch storage consistent
        self.assertTrue(data._tensor__array.is_contiguous())
        self.assertTrue(out._tensor__array.is_contiguous())
        self.assertTrue((out._tensor__array == data.comm.rank + 1).all())

    def test_exscan(self):
        # contiguous data
        data = ht.ones((5, 3,), dtype=ht.int64)
        out = ht.zeros_like(data)

        # reduce across all nodes
        self.assertTrue(data._tensor__array.is_contiguous())
        self.assertTrue(out._tensor__array.is_contiguous())
        data.comm.Exscan(data, out)

        # check the reduction result
        self.assertTrue(data._tensor__array.is_contiguous())
        self.assertTrue(out._tensor__array.is_contiguous())
        self.assertTrue((out._tensor__array == data.comm.rank).all())

        # non-contiguous data
        data = ht.ones((5, 3,), dtype=ht.int64).T
        out = ht.zeros_like(data)

        # reduce across all nodes
        self.assertFalse(data._tensor__array.is_contiguous())
        self.assertTrue(out._tensor__array.is_contiguous())
        data.comm.Exscan(data, out)

        # check the reduction result
        # the data tensor will be contiguous after the reduction
        # MPI enforces the same data type for send and receive buffer
        # the reduction implementation takes care of making the internal Torch storage consistent
        self.assertTrue(data._tensor__array.is_contiguous())
        self.assertTrue(out._tensor__array.is_contiguous())
        self.assertTrue((out._tensor__array == data.comm.rank).all())

        # non-contiguous output
        data = ht.ones((5, 3,), dtype=ht.int64)
        out = ht.zeros((3, 5), dtype=ht.int64).T

        # reduce across all nodes
        self.assertTrue(data._tensor__array.is_contiguous())
        self.assertFalse(out._tensor__array.is_contiguous())
        data.comm.Exscan(data, out)

        # check the reduction result
        # the data tensor will be contiguous after the reduction
        # MPI enforces the same data type for send and receive buffer
        # the reduction implementation takes care of making the internal Torch storage consistent
        self.assertTrue(data._tensor__array.is_contiguous())
        self.assertTrue(out._tensor__array.is_contiguous())
        self.assertTrue((out._tensor__array == data.comm.rank).all())

    def test_cuda_aware_mpi(self):
        self.assertTrue(hasattr(ht.communication, 'CUDA_AWARE_MPI'))
