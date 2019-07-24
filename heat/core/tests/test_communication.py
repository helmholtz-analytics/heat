import numpy as np
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

    def test_cuda_aware_mpi(self):
        self.assertTrue(hasattr(ht.communication, 'CUDA_AWARE_MPI'))
        self.assertIsInstance(ht.communication.CUDA_AWARE_MPI, bool)

    def test_contiguous_memory_buffer(self):
        # vector heat tensor
        vector_data = ht.arange(1, 10)
        vector_out = ht.zeros_like(vector_data)

        # test that target and destination are not equal
        self.assertTrue((vector_data._DNDarray__array != vector_out._DNDarray__array).all())
        self.assertTrue(vector_data._DNDarray__array.is_contiguous())
        self.assertTrue(vector_out._DNDarray__array.is_contiguous())

        # send message to self that is received into a separate buffer afterwards
        vector_data.comm.Isend(vector_data, dest=vector_data.comm.rank)
        vector_out.comm.Recv(vector_out, source=vector_out.comm.rank)

        # check that after sending the data everything is equal
        self.assertTrue((vector_data._DNDarray__array == vector_out._DNDarray__array).all())
        self.assertTrue(vector_out._DNDarray__array.is_contiguous())

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
        self.assertTrue((non_contiguous_data._DNDarray__array != contiguous_out._DNDarray__array).all())
        self.assertFalse(non_contiguous_data._DNDarray__array.is_contiguous())
        self.assertTrue(contiguous_out._DNDarray__array.is_contiguous())

        # send message to self that is received into a separate buffer afterwards
        non_contiguous_data.comm.Isend(non_contiguous_data, dest=non_contiguous_data.comm.rank)
        contiguous_out.comm.Recv(contiguous_out, source=contiguous_out.comm.rank)

        # check that after sending the data everything is equal
        self.assertTrue((non_contiguous_data._DNDarray__array == contiguous_out._DNDarray__array).all())
        self.assertTrue(contiguous_out._DNDarray__array.is_contiguous())

        # non-contiguous destination
        contiguous_data = ht.ones((3, 2,))
        non_contiguous_out = ht.zeros((2, 3,)).T

        # test that target and destination are not equal
        self.assertTrue((contiguous_data._DNDarray__array != non_contiguous_out._DNDarray__array).all())
        self.assertTrue(contiguous_data._DNDarray__array.is_contiguous())
        self.assertFalse(non_contiguous_out._DNDarray__array.is_contiguous())

        # send message to self that is received into a separate buffer afterwards
        contiguous_data.comm.Isend(contiguous_data, dest=contiguous_data.comm.rank)
        non_contiguous_out.comm.Recv(non_contiguous_out, source=non_contiguous_out.comm.rank)

        # check that after sending the data everything is equal
        self.assertTrue((contiguous_data._DNDarray__array == non_contiguous_out._DNDarray__array).all())
        self.assertFalse(non_contiguous_out._DNDarray__array.is_contiguous())

        # non-contiguous destination
        both_non_contiguous_data = ht.ones((3, 2,)).T
        both_non_contiguous_out = ht.zeros((3, 2,)).T

        # test that target and destination are not equal
        self.assertTrue((both_non_contiguous_data._DNDarray__array != both_non_contiguous_out._DNDarray__array).all())
        self.assertFalse(both_non_contiguous_data._DNDarray__array.is_contiguous())
        self.assertFalse(both_non_contiguous_out._DNDarray__array.is_contiguous())

        # send message to self that is received into a separate buffer afterwards
        both_non_contiguous_data.comm.Isend(both_non_contiguous_data, dest=both_non_contiguous_data.comm.rank)
        both_non_contiguous_out.comm.Recv(both_non_contiguous_out, source=both_non_contiguous_out.comm.rank)

        # check that after sending the data everything is equal
        self.assertTrue((both_non_contiguous_data._DNDarray__array == both_non_contiguous_out._DNDarray__array).all())
        self.assertFalse(both_non_contiguous_out._DNDarray__array.is_contiguous())

    def test_default_comm(self):
        # default comm is world
        a = ht.zeros((4, 5,))
        self.assertIs(ht.get_comm(), ht.MPI_WORLD)
        self.assertIs(a.comm, ht.MPI_WORLD)

        # we can set a new comm that is being used for new allocation, old are not affected
        ht.use_comm(ht.MPI_SELF)
        b = ht.zeros((4, 5,))
        self.assertIs(ht.get_comm(), ht.MPI_SELF)
        self.assertIs(b.comm, ht.MPI_SELF)
        self.assertIsNot(a.comm, ht.MPI_SELF)

        # reset the comm
        ht.use_comm(ht.MPI_WORLD)

        # test for proper sanitation
        with self.assertRaises(TypeError):
            ht.use_comm('1')

    def test_allgather(self):
        # contiguous data
        data = ht.ones((1, 7,))
        output = ht.zeros((ht.MPI_WORLD.size, 7,))

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        data.comm.Allgather(data, output)

        # check result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        self.assertTrue((output._DNDarray__array == torch.ones(ht.MPI_WORLD.size, 7,)).all())

        # contiguous data, different gather axis
        data = ht.ones((7, 2,))
        output = ht.random.randn(7, 2 * ht.MPI_WORLD.size)

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        data.comm.Allgather(data, output, send_axis=1)

        # check result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        self.assertTrue((output._DNDarray__array == torch.ones(7, 2 * ht.MPI_WORLD.size)).all())

        # non-contiguous data
        data = ht.ones((4, 5,)).T
        output = ht.zeros((5, 4 * ht.MPI_WORLD.size))

        # ensure prior invariants
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        data.comm.Allgather(data, output)

        # check result
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        self.assertTrue((output._DNDarray__array == torch.ones(5, 4 * ht.MPI_WORLD.size,)).all())

        # non-contiguous output, different gather axis
        data = ht.ones((5, 7,))
        output = ht.zeros((7 * ht.MPI_WORLD.size, 5,)).T

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())
        data.comm.Allgather(data, output, send_axis=1)

        # check result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())
        self.assertTrue((output._DNDarray__array == torch.ones(5, 7 * ht.MPI_WORLD.size,)).all())

        # contiguous data
        data = ht.array([[ht.MPI_WORLD.rank] * 10])
        output = ht.array([[0]*10]*ht.MPI_WORLD.size )

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())

        # perform the allgather operation
        data.comm.Allgather(data, output, send_axis=0)

        # check  result
        result = ht.array([np.arange(0, ht.MPI_WORLD.size)] * 10).T
        self.assertTrue(ht.equal(output, result))

        # contiguous data
        data = ht.array([[ht.MPI_WORLD.rank]] * 10)
        output = ht.array([[0] * ht.MPI_WORLD.size] * 10)

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())

        # perform the allgather operation
        data.comm.Allgather(data, output, send_axis=1)

        # check  result
        result = ht.array([np.arange(0, ht.MPI_WORLD.size)] * 10)
        self.assertTrue(ht.equal(output, result))

    def test_allgatherv(self):
        # contiguous data buffer, contiguous output buffer
        data = ht.ones((ht.MPI_WORLD.rank + 1, 10,))
        output_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1) // 2
        output = ht.zeros((output_count, 10,))

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())

        # perform the allgather operation
        counts = tuple(range(1, ht.MPI_WORLD.size + 1))
        displs = tuple(np.cumsum(range(ht.MPI_WORLD.size)))
        data.comm.Allgatherv(data, (output, counts, displs,))

        # check  result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        self.assertTrue((output._DNDarray__array == torch.ones(output_count, 10,)).all())

        # non-contiguous data buffer, contiguous output buffer
        data = ht.ones((10, 2 * (ht.MPI_WORLD.rank + 1))).T
        output_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1)
        output = ht.zeros((output_count, 10,))

        # ensure prior invariants
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())

        # perform the allgather operation
        counts = tuple(range(2, 2 * (ht.MPI_WORLD.size + 1), 2))
        displs = tuple(np.cumsum(range(0, 2 * ht.MPI_WORLD.size, 2)))
        data.comm.Allgatherv(data, (output, counts, displs,))

        # check  result
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        self.assertTrue((output._DNDarray__array == torch.ones(output_count, 10,)).all())

        # contiguous data buffer, non-contiguous output buffer
        data = ht.ones((2 * (ht.MPI_WORLD.rank + 1), 10,))
        output_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1)
        output = ht.zeros((10, output_count,)).T

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())

        # perform the allgather operation
        counts = tuple(range(2, 2 * (ht.MPI_WORLD.size + 1), 2))
        displs = tuple(np.cumsum(range(0, 2 * ht.MPI_WORLD.size, 2)))
        data.comm.Allgatherv(data, (output, counts, displs,))

        # check result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())
        self.assertTrue((output._DNDarray__array == torch.ones(output_count, 10,)).all())

        # non-contiguous data buffer, non-contiguous output buffer
        data = ht.ones((10, 2 * (ht.MPI_WORLD.rank + 1))).T
        output_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1)
        output = ht.zeros((10, output_count,)).T

        # ensure prior invariants
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())

        # perform the allgather operation
        counts = tuple(range(2, 2 * (ht.MPI_WORLD.size + 1), 2))
        displs = tuple(np.cumsum(range(0, 2 * ht.MPI_WORLD.size, 2)))
        data.comm.Allgatherv(data, (output, counts, displs,))

        # check result
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())
        self.assertTrue((output._DNDarray__array == torch.ones(output_count, 10,)).all())

        # contiguous data buffer
        data = ht.array([[ht.MPI_WORLD.rank] * 10] * (ht.MPI_WORLD.size + 1))

        # contiguous output buffer
        output_shape = data.lshape
        output = ht.zeros(output_shape, dtype=ht.int64)

        # Results for comparison
        first_line = ht.array([[0] * 10])
        last_line = ht.array([[ht.MPI_WORLD.size - 1] * 10])

        # perform allgather operation
        send_counts, send_displs, _ = data.comm.counts_displs_shape(data.lshape, 0)
        recv_counts, recv_displs, _ = data.comm.counts_displs_shape(output.lshape, 0)
        data.comm.Allgatherv((data, send_counts, send_displs,), (output, recv_counts, recv_displs,))

        # check result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        self.assertTrue((output[0] == first_line).all())
        self.assertTrue((output[output.lshape[0]-1] == last_line).all())


    def test_allreduce(self):
        # contiguous data
        data = ht.ones((10, 2,), dtype=ht.int8)
        out = ht.zeros_like(data)

        # reduce across all nodes
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(out._DNDarray__array.is_contiguous())
        data.comm.Allreduce(data, out, op=ht.MPI.SUM)

        # check the reduction result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(out._DNDarray__array.is_contiguous())
        self.assertTrue((out._DNDarray__array == data.comm.size).all())

        # non-contiguous data
        data = ht.ones((10, 2,), dtype=ht.int8).T
        out = ht.zeros_like(data)

        # reduce across all nodes
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertTrue(out._DNDarray__array.is_contiguous())
        data.comm.Allreduce(data, out, op=ht.MPI.SUM)

        # check the reduction result
        # the data tensor will be contiguous after the reduction
        # MPI enforces the same data type for send and receive buffer
        # the reduction implementation takes care of making the internal Torch storage consistent
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(out._DNDarray__array.is_contiguous())
        self.assertTrue((out._DNDarray__array == data.comm.size).all())

        # non-contiguous output
        data = ht.ones((10, 2,), dtype=ht.int8)
        out = ht.zeros((2, 10), dtype=ht.int8).T

        # reduce across all nodes
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertFalse(out._DNDarray__array.is_contiguous())
        data.comm.Allreduce(data, out, op=ht.MPI.SUM)

        # check the reduction result
        # the data tensor will be contiguous after the reduction
        # MPI enforces the same data type for send and receive buffer
        # the reduction implementation takes care of making the internal Torch storage consistent
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(out._DNDarray__array.is_contiguous())
        self.assertTrue((out._DNDarray__array == data.comm.size).all())

    def test_alltoall(self):
        # contiguous data
        data = ht.array([[ht.MPI_WORLD.rank] * 10] * ht.MPI_WORLD.size)
        output = ht.zeros((ht.MPI_WORLD.size, 10,), dtype=ht.int64)

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        data.comm.Alltoall(data, output)

        # check scatter result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        comparison = torch.arange(ht.MPI_WORLD.size).reshape(-1, 1).expand(ht.MPI_WORLD.size, 10)
        self.assertTrue((output._DNDarray__array == comparison).all())

        # contiguous data, different gather axis
        data = ht.array([[ht.MPI_WORLD.rank] * ht.MPI_WORLD.size] * 10)
        output = ht.zeros((10, ht.MPI_WORLD.size,), dtype=ht.int64)

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        data.comm.Alltoall(data, output, axis=1)

        # check scatter result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        comparison = torch.arange(ht.MPI_WORLD.size).repeat(10).reshape(10, ht.MPI_WORLD.size)
        self.assertTrue((output._DNDarray__array == comparison).all())

        # non-contiguous data
        data = ht.ones((10, 2 * ht.MPI_WORLD.size), dtype=ht.int64).T
        output = ht.zeros((2 * ht.MPI_WORLD.size, 10,), dtype=ht.int64)

        # ensure prior invariants
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        data.comm.Alltoall(data, output)

        # check scatter result
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        comparison = torch.ones((2 * ht.MPI_WORLD.size, 10,), dtype=torch.int64)
        self.assertTrue((output._DNDarray__array == comparison).all())

        # non-contiguous output, different gather axis
        data = ht.ones((10, 2 * ht.MPI_WORLD.size), dtype=ht.int64)
        output = ht.zeros((2 * ht.MPI_WORLD.size, 10,), dtype=ht.int64).T

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())
        data.comm.Alltoall(data, output, axis=1)

        # check scatter result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())
        comparison = torch.ones((10, 2 * ht.MPI_WORLD.size,), dtype=torch.int64)
        self.assertTrue((output._DNDarray__array == comparison).all())

    def test_alltoallv(self):
        # contiguous data buffer
        data = ht.array([[ht.MPI_WORLD.rank] * 10] * (ht.MPI_WORLD.size + 1))
        send_counts, send_displs, output_shape = data.comm.counts_displs_shape(data.lshape, 0)

        # contiguous output buffer
        output = ht.zeros(output_shape, dtype=ht.int64)
        recv_counts, recv_displs, _ = data.comm.counts_displs_shape(output.lshape, 0)

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        if ht.MPI_WORLD.size != 1:
            self.assertNotEqual(data.shape[0] % ht.MPI_WORLD.size, 0)
        else:
            self.assertEqual(data.shape[0] % ht.MPI_WORLD.size, 0)

        data.comm.Alltoallv((data, send_counts, send_displs,), (output, recv_counts, recv_displs,))
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        stack_count = output_shape[0] // ht.MPI_WORLD.size * 10
        comparison = torch.arange(ht.MPI_WORLD.size).reshape(-1, 1).expand(-1, stack_count).reshape(-1, 10)
        self.assertTrue((output._DNDarray__array == comparison).all())

        # non-contiguous data buffer
        data = ht.array([[ht.MPI_WORLD.rank] * (ht.MPI_WORLD.size + 1)] * 10).T
        send_counts, send_displs, output_shape = data.comm.counts_displs_shape(data.lshape, 0)

        # contiguous output buffer
        output = ht.zeros(output_shape, dtype=ht.int64)
        recv_counts, recv_displs, _ = data.comm.counts_displs_shape(output.lshape, 0)

        # ensure prior invariants
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        if ht.MPI_WORLD.size != 1:
            self.assertNotEqual(data.shape[0] % ht.MPI_WORLD.size, 0)
        else:
            self.assertEqual(data.shape[0] % ht.MPI_WORLD.size, 0)

        data.comm.Alltoallv((data, send_counts, send_displs,), (output, recv_counts, recv_displs,))
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        stack_count = output_shape[0] // ht.MPI_WORLD.size * 10
        comparison = torch.arange(ht.MPI_WORLD.size).reshape(-1, 1).expand(-1, stack_count).reshape(-1, 10)
        self.assertTrue((output._DNDarray__array == comparison).all())

        # contiguous data buffer
        data = ht.array([[ht.MPI_WORLD.rank] * 10] * (ht.MPI_WORLD.size + 1))
        send_counts, send_displs, output_shape = data.comm.counts_displs_shape(data.lshape, 0)

        # non-contiguous output buffer
        output_shape = tuple(reversed(output_shape))
        output = ht.zeros(output_shape, dtype=ht.int64).T
        recv_counts, recv_displs, _ = data.comm.counts_displs_shape(output.lshape, 0)

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())
        if ht.MPI_WORLD.size != 1:
            self.assertNotEqual(data.shape[0] % ht.MPI_WORLD.size, 0)
        else:
            self.assertEqual(data.shape[0] % ht.MPI_WORLD.size, 0)

        data.comm.Alltoallv((data, send_counts, send_displs,), (output, recv_counts, recv_displs,))
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())
        stack_count = output_shape[1] // ht.MPI_WORLD.size * 10
        comparison = torch.arange(ht.MPI_WORLD.size).reshape(-1, 1).expand(-1, stack_count).reshape(-1, 10)
        self.assertTrue((output._DNDarray__array == comparison).all())

        # non-contiguous data buffer
        data = ht.array([[ht.MPI_WORLD.rank] * (ht.MPI_WORLD.size + 1)] * 10).T
        send_counts, send_displs, output_shape = data.comm.counts_displs_shape(data.lshape, 0)

        # non-contiguous output buffer
        output_shape = tuple(reversed(output_shape))
        output = ht.zeros(output_shape, dtype=ht.int64).T
        recv_counts, recv_displs, _ = data.comm.counts_displs_shape(output.lshape, 0)

        # ensure prior invariants
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())
        if ht.MPI_WORLD.size != 1:
            self.assertNotEqual(data.shape[0] % ht.MPI_WORLD.size, 0)
        else:
            self.assertEqual(data.shape[0] % ht.MPI_WORLD.size, 0)

        data.comm.Alltoallv((data, send_counts, send_displs,), (output, recv_counts, recv_displs,))
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())
        stack_count = output_shape[1] // ht.MPI_WORLD.size * 10
        comparison = torch.arange(ht.MPI_WORLD.size).reshape(-1, 1).expand(-1, stack_count).reshape(-1, 10)
        self.assertTrue((output._DNDarray__array == comparison).all())

    def test_bcast(self):
        # contiguous data
        data = ht.arange(10, dtype=ht.int64)
        if ht.MPI_WORLD.rank != 0:
            data = ht.zeros_like(data, dtype=ht.int64)

        # broadcast data to all nodes
        self.assertTrue(data._DNDarray__array.is_contiguous())
        data.comm.Bcast(data, root=0)

        # assert output is equal
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue((data._DNDarray__array == torch.arange(10)).all())

        # non-contiguous data
        data = ht.ones((2, 5,), dtype=ht.float32).T
        if ht.MPI_WORLD.rank != 0:
            data = ht.zeros((2, 5,), dtype=ht.float32).T

        # broadcast data to all nodes
        self.assertFalse(data._DNDarray__array.is_contiguous())
        data.comm.Bcast(data, root=0)

        # assert output is equal
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertTrue((data._DNDarray__array == torch.ones((5, 2,), dtype=torch.float32)).all())

    def test_exscan(self):
        # contiguous data
        data = ht.ones((5, 3,), dtype=ht.int64)
        out = ht.zeros_like(data)

        # reduce across all nodes
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(out._DNDarray__array.is_contiguous())
        data.comm.Exscan(data, out)

        # check the reduction result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(out._DNDarray__array.is_contiguous())
        self.assertTrue((out._DNDarray__array == data.comm.rank).all())

        # non-contiguous data
        data = ht.ones((5, 3,), dtype=ht.int64).T
        out = ht.zeros_like(data)

        # reduce across all nodes
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertTrue(out._DNDarray__array.is_contiguous())
        data.comm.Exscan(data, out)

        # check the reduction result
        # the data tensor will be contiguous after the reduction
        # MPI enforces the same data type for send and receive buffer
        # the reduction implementation takes care of making the internal Torch storage consistent
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(out._DNDarray__array.is_contiguous())
        self.assertTrue((out._DNDarray__array == data.comm.rank).all())

        # non-contiguous output
        data = ht.ones((5, 3,), dtype=ht.int64)
        out = ht.zeros((3, 5), dtype=ht.int64).T

        # reduce across all nodes
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertFalse(out._DNDarray__array.is_contiguous())
        data.comm.Exscan(data, out)

        # check the reduction result
        # the data tensor will be contiguous after the reduction
        # MPI enforces the same data type for send and receive buffer
        # the reduction implementation takes care of making the internal Torch storage consistent
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(out._DNDarray__array.is_contiguous())
        self.assertTrue((out._DNDarray__array == data.comm.rank).all())

    def test_gather(self):
        # contiguous data
        data = ht.ones((1, 5,))
        output = ht.zeros((ht.MPI_WORLD.size, 5))

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        data.comm.Gather(data, output, root=0)

        # check scatter result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        if data.comm.rank == 0:
            self.assertTrue((output._DNDarray__array == torch.ones(ht.MPI_WORLD.size, 5,)).all())

        # contiguous data, different gather axis
        data = ht.ones((5, 2,))
        output = ht.zeros((5, 2 * ht.MPI_WORLD.size,))

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        data.comm.Gather(data, output, root=0, axis=1)

        # check scatter result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        if data.comm.rank == 0:
            self.assertTrue((output._DNDarray__array == torch.ones(5, 2 * ht.MPI_WORLD.size)).all())

        # non-contiguous data
        data = ht.ones((3, 5,)).T
        output = ht.zeros((5, 3 * ht.MPI_WORLD.size))

        # ensure prior invariants
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        data.comm.Gather(data, output, root=0)

        # check scatter result
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        if data.comm.rank == 0:
            self.assertTrue((output._DNDarray__array == torch.ones(5, 3 * ht.MPI_WORLD.size,)).all())

        # non-contiguous output, different gather axis
        data = ht.ones((5, 3,))
        output = ht.zeros((3 * ht.MPI_WORLD.size, 5,)).T

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())
        data.comm.Gather(data, output, root=0, axis=1)

        # check scatter result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())
        if data.comm.rank == 0:
            self.assertTrue((output._DNDarray__array == torch.ones(5, 3 * ht.MPI_WORLD.size,)).all())

    def test_gatherv(self):
        # contiguous data buffer, contiguous output buffer
        data = ht.ones((ht.MPI_WORLD.rank + 1, 10,))
        output_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1) // 2
        output = ht.zeros((output_count, 10,))

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())

        # perform the scatter operation
        counts = tuple(range(1, ht.MPI_WORLD.size + 1))
        displs = tuple(np.cumsum(range(ht.MPI_WORLD.size)))
        data.comm.Gatherv(data, (output, counts, displs,), root=0)

        # check scatter result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        if data.comm.rank == 0:
            self.assertTrue((output._DNDarray__array == torch.ones(output_count, 10,)).all())

        # non-contiguous data buffer, contiguous output buffer
        data = ht.ones((10, 2 * (ht.MPI_WORLD.rank + 1))).T
        output_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1)
        output = ht.zeros((output_count, 10,))

        # ensure prior invariants
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())

        # perform the scatter operation
        counts = tuple(range(2, 2 * (ht.MPI_WORLD.size + 1), 2))
        displs = tuple(np.cumsum(range(0, 2 * ht.MPI_WORLD.size, 2)))
        data.comm.Gatherv(data, (output, counts, displs,), root=0)

        # check scatter result
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        if data.comm.rank == 0:
            self.assertTrue((output._DNDarray__array == torch.ones(output_count, 10,)).all())

        # contiguous data buffer, non-contiguous output buffer
        data = ht.ones((2 * (ht.MPI_WORLD.rank + 1), 10,))
        output_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1)
        output = ht.zeros((10, output_count,)).T

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())

        # perform the scatter operation
        counts = tuple(range(2, 2 * (ht.MPI_WORLD.size + 1), 2))
        displs = tuple(np.cumsum(range(0, 2 * ht.MPI_WORLD.size, 2)))
        data.comm.Gatherv(data, (output, counts, displs,), root=0)

        # check scatter result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())
        if data.comm.rank == 0:
            self.assertTrue((output._DNDarray__array == torch.ones(output_count, 10,)).all())

        # non-contiguous data buffer, non-contiguous output buffer
        data = ht.ones((10, 2 * (ht.MPI_WORLD.rank + 1))).T
        output_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1)
        output = ht.zeros((10, output_count,)).T

        # ensure prior invariants
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())

        # perform the scatter operation
        counts = tuple(range(2, 2 * (ht.MPI_WORLD.size + 1), 2))
        displs = tuple(np.cumsum(range(0, 2 * ht.MPI_WORLD.size, 2)))
        data.comm.Gatherv(data, (output, counts, displs,), root=0)

        # check scatter result
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())
        if data.comm.rank == 0:
            self.assertTrue((output._DNDarray__array == torch.ones(output_count, 10,)).all())

    def test_iallgather(self):
        try:
            # contiguous data
            data = ht.ones((1, 7,))
            output = ht.zeros((ht.MPI_WORLD.size, 7,))

            # ensure prior invariants
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            req = data.comm.Iallgather(data, output)
            req.wait()

            # check scatter result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            self.assertTrue((output._DNDarray__array == torch.ones(ht.MPI_WORLD.size, 7,)).all())

            # contiguous data, different gather axis
            data = ht.ones((7, 2,))
            output = ht.random.randn(7, 2 * ht.MPI_WORLD.size)

            # ensure prior invariants
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            req = data.comm.Iallgather(data, output, send_axis=1)
            req.wait()

            # check scatter result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            self.assertTrue((output._DNDarray__array == torch.ones(7, 2 * ht.MPI_WORLD.size)).all())

            # non-contiguous data
            data = ht.ones((4, 5,)).T
            output = ht.zeros((5, 4 * ht.MPI_WORLD.size))

            # ensure prior invariants
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            req = data.comm.Iallgather(data, output)
            req.wait()

            # check scatter result
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            self.assertTrue((output._DNDarray__array == torch.ones(5, 4 * ht.MPI_WORLD.size,)).all())

            # non-contiguous output, different gather axis
            data = ht.ones((5, 7,))
            output = ht.zeros((7 * ht.MPI_WORLD.size, 5,)).T

            # ensure prior invariants
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())
            req = data.comm.Iallgather(data, output, send_axis=1)
            req.wait()

            # check scatter result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())
            self.assertTrue((output._DNDarray__array == torch.ones(5, 7 * ht.MPI_WORLD.size,)).all())

        # MPI implementation may not support asynchronous operations
        except NotImplementedError:
            pass

    def test_iallgatherv(self):
        try:
            # contiguous data buffer, contiguous output buffer
            data = ht.ones((ht.MPI_WORLD.rank + 1, 10,))
            output_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1) // 2
            output = ht.zeros((output_count, 10,))

            # ensure prior invariants
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())

            # perform the scatter operation
            counts = tuple(range(1, ht.MPI_WORLD.size + 1))
            displs = tuple(np.cumsum(range(ht.MPI_WORLD.size)))
            req = data.comm.Iallgatherv(data, (output, counts, displs,))
            req.wait()

            # check scatter result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            self.assertTrue((output._DNDarray__array == torch.ones(output_count, 10,)).all())

            # non-contiguous data buffer, contiguous output buffer
            data = ht.ones((10, 2 * (ht.MPI_WORLD.rank + 1))).T
            output_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1)
            output = ht.zeros((output_count, 10,))

            # ensure prior invariants
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())

            # perform the scatter operation
            counts = tuple(range(2, 2 * (ht.MPI_WORLD.size + 1), 2))
            displs = tuple(np.cumsum(range(0, 2 * ht.MPI_WORLD.size, 2)))
            req = data.comm.Iallgatherv(data, (output, counts, displs,))
            req.wait()

            # check scatter result
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            self.assertTrue((output._DNDarray__array == torch.ones(output_count, 10,)).all())

            # contiguous data buffer, non-contiguous output buffer
            data = ht.ones((2 * (ht.MPI_WORLD.rank + 1), 10,))
            output_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1)
            output = ht.zeros((10, output_count,)).T

            # ensure prior invariants
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())

            # perform the scatter operation
            counts = tuple(range(2, 2 * (ht.MPI_WORLD.size + 1), 2))
            displs = tuple(np.cumsum(range(0, 2 * ht.MPI_WORLD.size, 2)))
            req = data.comm.Iallgatherv(data, (output, counts, displs,))
            req.wait()

            # check scatter result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())
            self.assertTrue((output._DNDarray__array == torch.ones(output_count, 10,)).all())

            # non-contiguous data buffer, non-contiguous output buffer
            data = ht.ones((10, 2 * (ht.MPI_WORLD.rank + 1))).T
            output_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1)
            output = ht.zeros((10, output_count,)).T

            # ensure prior invariants
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())

            # perform the scatter operation
            counts = tuple(range(2, 2 * (ht.MPI_WORLD.size + 1), 2))
            displs = tuple(np.cumsum(range(0, 2 * ht.MPI_WORLD.size, 2)))
            req = data.comm.Iallgatherv(data, (output, counts, displs,))
            req.wait()

            # check scatter result
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())
            self.assertTrue((output._DNDarray__array == torch.ones(output_count, 10,)).all())

        # MPI implementation may not support asynchronous operations
        except NotImplementedError:
            pass

    def test_iallreduce(self):
        try:
            # contiguous data
            data = ht.ones((10, 2,), dtype=ht.int8)
            out = ht.zeros_like(data)

            # reduce across all nodes
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(out._DNDarray__array.is_contiguous())
            req = data.comm.Iallreduce(data, out, op=ht.MPI.SUM)
            req.wait()

            # check the reduction result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(out._DNDarray__array.is_contiguous())
            self.assertTrue((out._DNDarray__array == data.comm.size).all())

            # non-contiguous data
            data = ht.ones((10, 2,), dtype=ht.int8).T
            out = ht.zeros_like(data)

            # reduce across all nodes
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertTrue(out._DNDarray__array.is_contiguous())
            req = data.comm.Iallreduce(data, out, op=ht.MPI.SUM)
            req.wait()

            # check the reduction result
            # the data tensor will be contiguous after the reduction
            # MPI enforces the same data type for send and receive buffer
            # the reduction implementation takes care of making the internal Torch storage consistent
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(out._DNDarray__array.is_contiguous())
            self.assertTrue((out._DNDarray__array == data.comm.size).all())

            # non-contiguous output
            data = ht.ones((10, 2,), dtype=ht.int8)
            out = ht.zeros((2, 10), dtype=ht.int8).T

            # reduce across all nodes
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertFalse(out._DNDarray__array.is_contiguous())
            req = data.comm.Iallreduce(data, out, op=ht.MPI.SUM)
            req.wait()

            # check the reduction result
            # the data tensor will be contiguous after the reduction
            # MPI enforces the same data type for send and receive buffer
            # the reduction implementation takes care of making the internal Torch storage consistent
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(out._DNDarray__array.is_contiguous())
            self.assertTrue((out._DNDarray__array == data.comm.size).all())

        # MPI implementation may not support asynchronous operations
        except NotImplementedError:
            pass

    def test_ialltoall(self):
        try:
            # contiguous data
            data = ht.array([[ht.MPI_WORLD.rank] * 10] * ht.MPI_WORLD.size)
            output = ht.zeros((ht.MPI_WORLD.size, 10,), dtype=ht.int64)

            # ensure prior invariants
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            req = data.comm.Ialltoall(data, output)
            req.wait()

            # check scatter result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            comparison = torch.arange(ht.MPI_WORLD.size).reshape(-1, 1).expand(ht.MPI_WORLD.size, 10)
            self.assertTrue((output._DNDarray__array == comparison).all())

            # contiguous data, different gather axis
            data = ht.array([[ht.MPI_WORLD.rank] * ht.MPI_WORLD.size] * 10)
            output = ht.zeros((10, ht.MPI_WORLD.size,), dtype=ht.int64)

            # ensure prior invariants
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            req = data.comm.Ialltoall(data, output, axis=1)
            req.wait()

            # check scatter result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            comparison = torch.arange(ht.MPI_WORLD.size).repeat(10).reshape(10, ht.MPI_WORLD.size)
            self.assertTrue((output._DNDarray__array == comparison).all())

            # non-contiguous data
            data = ht.ones((10, 2 * ht.MPI_WORLD.size), dtype=ht.int64).T
            output = ht.zeros((2 * ht.MPI_WORLD.size, 10,), dtype=ht.int64)

            # ensure prior invariants
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            req = data.comm.Ialltoall(data, output)
            req.wait()

            # check scatter result
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            comparison = torch.ones((2 * ht.MPI_WORLD.size, 10,), dtype=torch.int64)
            self.assertTrue((output._DNDarray__array == comparison).all())

            # non-contiguous output, different gather axis
            data = ht.ones((10, 2 * ht.MPI_WORLD.size), dtype=ht.int64)
            output = ht.zeros((2 * ht.MPI_WORLD.size, 10,), dtype=ht.int64).T

            # ensure prior invariants
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())
            req = data.comm.Ialltoall(data, output, axis=1)
            req.wait()

            # check scatter result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())
            comparison = torch.ones((10, 2 * ht.MPI_WORLD.size,), dtype=torch.int64)
            self.assertTrue((output._DNDarray__array == comparison).all())

        # MPI implementation may not support asynchronous operations
        except NotImplementedError:
            pass

    def test_ialltoallv(self):
        try:
            # contiguous data buffer
            data = ht.array([[ht.MPI_WORLD.rank] * 10] * (ht.MPI_WORLD.size + 1))
            send_counts, send_displs, output_shape = data.comm.counts_displs_shape(data.lshape, 0)

            # contiguous output buffer
            output = ht.zeros(output_shape, dtype=ht.int64)
            recv_counts, recv_displs, _ = data.comm.counts_displs_shape(output.lshape, 0)

            # ensure prior invariants
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            if ht.MPI_WORLD.size != 1:
                self.assertNotEqual(data.shape[0] % ht.MPI_WORLD.size, 0)
            else:
                self.assertEqual(data.shape[0] % ht.MPI_WORLD.size, 0)

            req = data.comm.Ialltoallv((data, send_counts, send_displs,), (output, recv_counts, recv_displs,))
            req.wait()

            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            stack_count = output_shape[0] // ht.MPI_WORLD.size * 10
            comparison = torch.arange(ht.MPI_WORLD.size).reshape(-1, 1).expand(-1, stack_count).reshape(-1, 10)
            self.assertTrue((output._DNDarray__array == comparison).all())

            # non-contiguous data buffer
            data = ht.array([[ht.MPI_WORLD.rank] * (ht.MPI_WORLD.size + 1)] * 10).T
            send_counts, send_displs, output_shape = data.comm.counts_displs_shape(data.lshape, 0)

            # contiguous output buffer
            output = ht.zeros(output_shape, dtype=ht.int64)
            recv_counts, recv_displs, _ = data.comm.counts_displs_shape(output.lshape, 0)

            # ensure prior invariants
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            if ht.MPI_WORLD.size != 1:
                self.assertNotEqual(data.shape[0] % ht.MPI_WORLD.size, 0)
            else:
                self.assertEqual(data.shape[0] % ht.MPI_WORLD.size, 0)

            req = data.comm.Ialltoallv((data, send_counts, send_displs,), (output, recv_counts, recv_displs,))
            req.wait()

            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            stack_count = output_shape[0] // ht.MPI_WORLD.size * 10
            comparison = torch.arange(ht.MPI_WORLD.size).reshape(-1, 1).expand(-1, stack_count).reshape(-1, 10)
            self.assertTrue((output._DNDarray__array == comparison).all())

            # contiguous data buffer
            data = ht.array([[ht.MPI_WORLD.rank] * 10] * (ht.MPI_WORLD.size + 1))
            send_counts, send_displs, output_shape = data.comm.counts_displs_shape(data.lshape, 0)

            # non-contiguous output buffer
            output_shape = tuple(reversed(output_shape))
            output = ht.zeros(output_shape, dtype=ht.int64).T
            recv_counts, recv_displs, _ = data.comm.counts_displs_shape(output.lshape, 0)

            # ensure prior invariants
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())
            if ht.MPI_WORLD.size != 1:
                self.assertNotEqual(data.shape[0] % ht.MPI_WORLD.size, 0)
            else:
                self.assertEqual(data.shape[0] % ht.MPI_WORLD.size, 0)

            req = data.comm.Ialltoallv((data, send_counts, send_displs,), (output, recv_counts, recv_displs,))
            req.wait()

            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())
            stack_count = output_shape[1] // ht.MPI_WORLD.size * 10
            comparison = torch.arange(ht.MPI_WORLD.size).reshape(-1, 1).expand(-1, stack_count).reshape(-1, 10)
            self.assertTrue((output._DNDarray__array == comparison).all())

            # non-contiguous data buffer
            data = ht.array([[ht.MPI_WORLD.rank] * (ht.MPI_WORLD.size + 1)] * 10).T
            send_counts, send_displs, output_shape = data.comm.counts_displs_shape(data.lshape, 0)

            # non-contiguous output buffer
            output_shape = tuple(reversed(output_shape))
            output = ht.zeros(output_shape, dtype=ht.int64).T
            recv_counts, recv_displs, _ = data.comm.counts_displs_shape(output.lshape, 0)

            # ensure prior invariants
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())
            if ht.MPI_WORLD.size != 1:
                self.assertNotEqual(data.shape[0] % ht.MPI_WORLD.size, 0)
            else:
                self.assertEqual(data.shape[0] % ht.MPI_WORLD.size, 0)

            req = data.comm.Ialltoallv((data, send_counts, send_displs,), (output, recv_counts, recv_displs,))
            req.wait()

            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())
            stack_count = output_shape[1] // ht.MPI_WORLD.size * 10
            comparison = torch.arange(ht.MPI_WORLD.size).reshape(-1, 1).expand(-1, stack_count).reshape(-1, 10)
            self.assertTrue((output._DNDarray__array == comparison).all())

        # MPI implementation may not support asynchronous operations
        except NotImplementedError:
            pass

    def test_ibcast(self):
        try:
            # contiguous data
            data = ht.arange(10, dtype=ht.int64)
            if ht.MPI_WORLD.rank != 0:
                data = ht.zeros_like(data, dtype=ht.int64)

            # broadcast data to all nodes
            self.assertTrue(data._DNDarray__array.is_contiguous())
            req = data.comm.Ibcast(data, root=0)
            req.wait()

            # assert output is equal
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue((data._DNDarray__array == torch.arange(10)).all())

            # non-contiguous data
            data = ht.ones((2, 5,), dtype=ht.float32).T
            if ht.MPI_WORLD.rank != 0:
                data = ht.zeros((2, 5,), dtype=ht.float32).T

            # broadcast data to all nodes
            self.assertFalse(data._DNDarray__array.is_contiguous())
            req = data.comm.Ibcast(data, root=0)
            req.wait()

            # assert output is equal
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertTrue((data._DNDarray__array == torch.ones((5, 2,), dtype=torch.float32)).all())

        # MPI implementation may not support asynchronous operations
        except NotImplementedError:
            pass

    def test_iexscan(self):
        try:
            # contiguous data
            data = ht.ones((5, 3,), dtype=ht.int64)
            out = ht.zeros_like(data)

            # reduce across all nodes
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(out._DNDarray__array.is_contiguous())
            req = data.comm.Iexscan(data, out)
            req.wait()

            # check the reduction result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(out._DNDarray__array.is_contiguous())
            self.assertTrue((out._DNDarray__array == data.comm.rank).all())

            # non-contiguous data
            data = ht.ones((5, 3,), dtype=ht.int64).T
            out = ht.zeros_like(data)

            # reduce across all nodes
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertTrue(out._DNDarray__array.is_contiguous())
            req = data.comm.Iexscan(data, out)
            req.wait()

            # check the reduction result
            # the data tensor will be contiguous after the reduction
            # MPI enforces the same data type for send and receive buffer
            # the reduction implementation takes care of making the internal Torch storage consistent
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(out._DNDarray__array.is_contiguous())
            self.assertTrue((out._DNDarray__array == data.comm.rank).all())

            # non-contiguous output
            data = ht.ones((5, 3,), dtype=ht.int64)
            out = ht.zeros((3, 5), dtype=ht.int64).T

            # reduce across all nodes
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertFalse(out._DNDarray__array.is_contiguous())
            req = data.comm.Iexscan(data, out)
            req.wait()

            # check the reduction result
            # the data tensor will be contiguous after the reduction
            # MPI enforces the same data type for send and receive buffer
            # the reduction implementation takes care of making the internal Torch storage consistent
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(out._DNDarray__array.is_contiguous())
            self.assertTrue((out._DNDarray__array == data.comm.rank).all())

        # MPI implementation may not support asynchronous operations
        except NotImplementedError:
            pass

    def test_igather(self):
        try:
            # contiguous data
            data = ht.ones((1, 5,), dtype=ht.float32)
            output = ht.random.randn(ht.MPI_WORLD.size, 5)

            # ensure prior invariants
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            req = data.comm.Igather(data, output, root=0)
            req.wait()

            # check scatter result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            if data.comm.rank == 0:
                self.assertTrue((output._DNDarray__array == torch.ones((ht.MPI_WORLD.size, 5,), dtype=torch.float32)).all())

            # contiguous data, different gather axis
            data = ht.ones((5, 2,), dtype=ht.float32)
            output = ht.random.randn(5, 2 * ht.MPI_WORLD.size)

            # ensure prior invariants
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            req = data.comm.Igather(data, output, root=0, axis=1)
            req.wait()

            # check scatter result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            if data.comm.rank == 0:
                self.assertTrue(
                    (output._DNDarray__array == torch.ones((5, 2 * ht.MPI_WORLD.size,), dtype=torch.float32)).all()
                )

            # non-contiguous data
            data = ht.ones((3, 5,), dtype=ht.float32).T
            output = ht.random.randn(5, 3 * ht.MPI_WORLD.size)

            # ensure prior invariants
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            req = data.comm.Igather(data, output, root=0)
            req.wait()

            # check scatter result
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            if data.comm.rank == 0:
                self.assertTrue(
                    (output._DNDarray__array == torch.ones((5, 3 * ht.MPI_WORLD.size,), dtype=torch.float32)).all()
                )

            # non-contiguous output, different gather axis
            data = ht.ones((5, 3,), dtype=ht.float32)
            output = ht.random.randn(3 * ht.MPI_WORLD.size, 5).T

            # ensure prior invariants
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())
            req = data.comm.Igather(data, output, root=0, axis=1)
            req.wait()

            # check scatter result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())
            if data.comm.rank == 0:
                self.assertTrue(
                    (output._DNDarray__array == torch.ones((5, 3 * ht.MPI_WORLD.size,), dtype=torch.float32)).all()
                )

        # MPI implementation may not support asynchronous operations
        except NotImplementedError:
            pass

    def test_igatherv(self):
        try:
            # contiguous data buffer, contiguous output buffer
            data = ht.ones((ht.MPI_WORLD.rank + 1, 10,))
            output_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1) // 2
            output = ht.zeros((output_count, 10,))

            # ensure prior invariants
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())

            # perform the scatter operation
            counts = tuple(range(1, ht.MPI_WORLD.size + 1))
            displs = tuple(np.cumsum(range(ht.MPI_WORLD.size)))
            req = data.comm.Igatherv(data, (output, counts, displs,), root=0)
            req.wait()

            # check scatter result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            if data.comm.rank == 0:
                self.assertTrue((output._DNDarray__array == torch.ones(output_count, 10,)).all())

            # non-contiguous data buffer, contiguous output buffer
            data = ht.ones((10, 2 * (ht.MPI_WORLD.rank + 1))).T
            output_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1)
            output = ht.zeros((output_count, 10,))

            # ensure prior invariants
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())

            # perform the scatter operation
            counts = tuple(range(2, 2 * (ht.MPI_WORLD.size + 1), 2))
            displs = tuple(np.cumsum(range(0, 2 * ht.MPI_WORLD.size, 2)))
            req = data.comm.Igatherv(data, (output, counts, displs,), root=0)
            req.wait()

            # check scatter result
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            if data.comm.rank == 0:
                self.assertTrue((output._DNDarray__array == torch.ones(output_count, 10,)).all())

            # contiguous data buffer, non-contiguous output buffer
            data = ht.ones((2 * (ht.MPI_WORLD.rank + 1), 10,))
            output_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1)
            output = ht.zeros((10, output_count,)).T

            # ensure prior invariants
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())

            # perform the scatter operation
            counts = tuple(range(2, 2 * (ht.MPI_WORLD.size + 1), 2))
            displs = tuple(np.cumsum(range(0, 2 * ht.MPI_WORLD.size, 2)))
            req = data.comm.Igatherv(data, (output, counts, displs,), root=0)
            req.wait()

            # check scatter result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())
            if data.comm.rank == 0:
                self.assertTrue((output._DNDarray__array == torch.ones(output_count, 10,)).all())

            # non-contiguous data buffer, non-contiguous output buffer
            data = ht.ones((10, 2 * (ht.MPI_WORLD.rank + 1))).T
            output_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1)
            output = ht.zeros((10, output_count,)).T

            # ensure prior invariants
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())

            # perform the scatter operation
            counts = tuple(range(2, 2 * (ht.MPI_WORLD.size + 1), 2))
            displs = tuple(np.cumsum(range(0, 2 * ht.MPI_WORLD.size, 2)))
            req = data.comm.Igatherv(data, (output, counts, displs,), root=0)
            req.wait()

            # check scatter result
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())
            if data.comm.rank == 0:
                self.assertTrue((output._DNDarray__array == torch.ones(output_count, 10,)).all())

        # MPI implementation may not support asynchronous operations
        except NotImplementedError:
            pass

    def test_ireduce(self):
        try:
            # contiguous data
            data = ht.ones((10, 2,), dtype=ht.int32)
            out = ht.zeros_like(data)

            # reduce across all nodes
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(out._DNDarray__array.is_contiguous())
            req = data.comm.Ireduce(data, out, op=ht.MPI.SUM, root=0)
            req.wait()

            # check the reduction result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(out._DNDarray__array.is_contiguous())
            if data.comm.rank == 0:
                self.assertTrue((out._DNDarray__array == data.comm.size).all())

            # non-contiguous data
            data = ht.ones((10, 2,), dtype=ht.int32).T
            out = ht.zeros_like(data)

            # reduce across all nodes
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertTrue(out._DNDarray__array.is_contiguous())
            req = data.comm.Ireduce(data, out, op=ht.MPI.SUM, root=0)
            req.wait()

            # check the reduction result
            # the data tensor will be contiguous after the reduction
            # MPI enforces the same data type for send and receive buffer
            # the reduction implementation takes care of making the internal Torch storage consistent
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(out._DNDarray__array.is_contiguous())
            if data.comm.rank == 0:
                self.assertTrue((out._DNDarray__array == data.comm.size).all())

            # non-contiguous output
            data = ht.ones((10, 2,), dtype=ht.int32)
            out = ht.zeros((2, 10), dtype=ht.int32).T

            # reduce across all nodes
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertFalse(out._DNDarray__array.is_contiguous())
            req = data.comm.Ireduce(data, out, op=ht.MPI.SUM, root=0)
            req.wait()

            # check the reduction result
            # the data tensor will be contiguous after the reduction
            # MPI enforces the same data type for send and receive buffer
            # the reduction implementation takes care of making the internal Torch storage consistent
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(out._DNDarray__array.is_contiguous())
            if data.comm.rank == 0:
                self.assertTrue((out._DNDarray__array == data.comm.size).all())

        # MPI implementation may not support asynchronous operations
        except NotImplementedError:
            pass

    def test_iscan(self):
        try:
            # contiguous data
            data = ht.ones((5, 3,), dtype=ht.float64)
            out = ht.zeros_like(data)

            # reduce across all nodes
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(out._DNDarray__array.is_contiguous())
            req = data.comm.Iscan(data, out)
            req.wait()

            # check the reduction result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(out._DNDarray__array.is_contiguous())
            self.assertTrue((out._DNDarray__array == data.comm.rank + 1).all())

            # non-contiguous data
            data = ht.ones((5, 3,), dtype=ht.float64).T
            out = ht.zeros_like(data)

            # reduce across all nodes
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertTrue(out._DNDarray__array.is_contiguous())
            req = data.comm.Iscan(data, out)
            req.wait()

            # check the reduction result
            # the data tensor will be contiguous after the reduction
            # MPI enforces the same data type for send and receive buffer
            # the reduction implementation takes care of making the internal Torch storage consistent
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(out._DNDarray__array.is_contiguous())
            self.assertTrue((out._DNDarray__array == data.comm.rank + 1).all())

            # non-contiguous output
            data = ht.ones((5, 3,), dtype=ht.float64)
            out = ht.zeros((3, 5), dtype=ht.float64).T

            # reduce across all nodes
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertFalse(out._DNDarray__array.is_contiguous())
            req = data.comm.Iscan(data, out)
            req.wait()

            # check the reduction result
            # the data tensor will be contiguous after the reduction
            # MPI enforces the same data type for send and receive buffer
            # the reduction implementation takes care of making the internal Torch storage consistent
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(out._DNDarray__array.is_contiguous())
            self.assertTrue((out._DNDarray__array == data.comm.rank + 1).all())

        # MPI implementation may not support asynchronous operations
        except NotImplementedError:
            pass

    def test_iscatter(self):
        try:
            # contiguous data
            if ht.MPI_WORLD.rank == 0:
                data = ht.ones((ht.MPI_WORLD.size, 5))
            else:
                data = ht.zeros((1,))
            output = ht.zeros((1, 5,))

            # ensure prior invariants
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            req = data.comm.Iscatter(data, output, root=0)
            req.wait()

            # check scatter result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            self.assertTrue((output._DNDarray__array == torch.ones(1, 5,)).all())

            # contiguous data, different scatter axis
            if ht.MPI_WORLD.rank == 0:
                data = ht.ones((5, ht.MPI_WORLD.size,))
            else:
                data = ht.zeros((1,))
            output = ht.zeros((5, 1,))

            # ensure prior invariants
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            req = data.comm.Iscatter(data, output, root=0, axis=1)
            req.wait()

            # check scatter result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            self.assertTrue((output._DNDarray__array == torch.ones(5, 1,)).all())

            # non-contiguous data
            if ht.MPI_WORLD.rank == 0:
                data = ht.ones((5, ht.MPI_WORLD.size * 2,)).T
                self.assertFalse(data._DNDarray__array.is_contiguous())
            else:
                data = ht.zeros((1,))
                self.assertTrue(data._DNDarray__array.is_contiguous())
            output = ht.zeros((2, 5))

            # ensure prior invariants
            self.assertTrue(output._DNDarray__array.is_contiguous())
            req = data.comm.Iscatter(data, output, root=0)
            req.wait()

            # check scatter result
            if ht.MPI_WORLD.rank == 0:
                self.assertFalse(data._DNDarray__array.is_contiguous())
            else:
                self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            self.assertTrue((output._DNDarray__array == torch.ones(2, 5,)).all())

            # non-contiguous destination, different split axis
            if ht.MPI_WORLD.rank == 0:
                data = ht.ones((5, ht.MPI_WORLD.size * 2,))
            else:
                data = ht.zeros((1,))
            output = ht.zeros((2, 5)).T

            # ensure prior invariants
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())
            req = data.comm.Iscatter(data, output, root=0, axis=1)
            req.wait()

            # check scatter result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())
            self.assertTrue((output._DNDarray__array == torch.ones(5, 2,)).all())

        # MPI implementation may not support asynchronous operations
        except NotImplementedError:
            pass

    def test_iscatterv(self):
        try:
            # contiguous data buffer, contiguous output buffer
            input_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1)
            data = ht.ones((input_count, 12,))
            output_count = 2 * (ht.MPI_WORLD.rank + 1)
            output = ht.zeros((output_count, 12,))

            # ensure prior invariants
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())

            # perform the scatter operation
            counts = tuple(range(2, 2 * (ht.MPI_WORLD.size + 1), 2))
            displs = tuple(np.cumsum(range(0, 2 * ht.MPI_WORLD.size, 2)))
            req = data.comm.Iscatterv((data, counts, displs,), output, root=0)
            req.wait()

            # check scatter result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            self.assertTrue((output._DNDarray__array == torch.ones(output_count, 12,)).all())

            # non-contiguous data buffer, contiguous output buffer
            input_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1)
            data = ht.ones((12, input_count,)).T
            output_count = 2 * (ht.MPI_WORLD.rank + 1)
            output = ht.zeros((output_count, 12,))

            # ensure prior invariants
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())

            # perform the scatter operation
            counts = tuple(range(2, 2 * (ht.MPI_WORLD.size + 1), 2))
            displs = tuple(np.cumsum(range(0, 2 * ht.MPI_WORLD.size, 2)))
            req = data.comm.Iscatterv((data, counts, displs,), output, root=0)
            req.wait()

            # check scatter result
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertTrue(output._DNDarray__array.is_contiguous())
            self.assertTrue((output._DNDarray__array == torch.ones(output_count, 12,)).all())

            # contiguous data buffer, non-contiguous output buffer
            input_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1)
            data = ht.ones((input_count, 12,))
            output_count = 2 * (ht.MPI_WORLD.rank + 1)
            output = ht.zeros((12, output_count,)).T

            # ensure prior invariants
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())

            # perform the scatter operation
            counts = tuple(range(2, 2 * (ht.MPI_WORLD.size + 1), 2))
            displs = tuple(np.cumsum(range(0, 2 * ht.MPI_WORLD.size, 2)))
            req = data.comm.Iscatterv((data, counts, displs,), output, root=0)
            req.wait()

            # check scatter result
            self.assertTrue(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())
            self.assertTrue((output._DNDarray__array == torch.ones(output_count, 12,)).all())

            # non-contiguous data buffer, non-contiguous output buffer
            input_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1)
            data = ht.ones((12, input_count,)).T
            output_count = 2 * (ht.MPI_WORLD.rank + 1)
            output = ht.zeros((12, output_count,)).T

            # ensure prior invariants
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())

            # perform the scatter operation
            counts = tuple(range(2, 2 * (ht.MPI_WORLD.size + 1), 2))
            displs = tuple(np.cumsum(range(0, 2 * ht.MPI_WORLD.size, 2)))
            req = data.comm.Iscatterv((data, counts, displs,), output, root=0)
            req.wait()

            # check scatter result
            self.assertFalse(data._DNDarray__array.is_contiguous())
            self.assertFalse(output._DNDarray__array.is_contiguous())
            self.assertTrue((output._DNDarray__array == torch.ones(output_count, 12,)).all())

        # MPI implementation may not support asynchronous operations
        except NotImplementedError:
            pass

    def test_mpi_in_place(self):
        size = ht.MPI_WORLD.size
        data = ht.ones((size, size,), dtype=ht.int32)
        data.comm.Allreduce(ht.MPI.IN_PLACE, data, op=ht.MPI.SUM)

        self.assertTrue((data._DNDarray__array == size).all())

        tensor = torch.arange(size).repeat(size).reshape(size, size)
        data = ht.array(tensor, split=0)
        data.comm.Alltoall(ht.MPI.IN_PLACE, data)
        self.assertTrue((data._DNDarray__array == ht.MPI_WORLD.rank).all())


    def test_reduce(self):
        # contiguous data
        data = ht.ones((10, 2,), dtype=ht.int32)
        out = ht.zeros_like(data)

        # reduce across all nodes
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(out._DNDarray__array.is_contiguous())
        data.comm.Reduce(data, out, op=ht.MPI.SUM, root=0)

        # check the reduction result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(out._DNDarray__array.is_contiguous())
        if data.comm.rank == 0:
            self.assertTrue((out._DNDarray__array == data.comm.size).all())

        # non-contiguous data
        data = ht.ones((10, 2,), dtype=ht.int32).T
        out = ht.zeros_like(data)

        # reduce across all nodes
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertTrue(out._DNDarray__array.is_contiguous())
        data.comm.Reduce(data, out, op=ht.MPI.SUM, root=0)

        # check the reduction result
        # the data tensor will be contiguous after the reduction
        # MPI enforces the same data type for send and receive buffer
        # the reduction implementation takes care of making the internal Torch storage consistent
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(out._DNDarray__array.is_contiguous())
        if data.comm.rank == 0:
            self.assertTrue((out._DNDarray__array == data.comm.size).all())

        # non-contiguous output
        data = ht.ones((10, 2,), dtype=ht.int32)
        out = ht.zeros((2, 10), dtype=ht.int32).T

        # reduce across all nodes
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertFalse(out._DNDarray__array.is_contiguous())
        data.comm.Reduce(data, out, op=ht.MPI.SUM, root=0)

        # check the reduction result
        # the data tensor will be contiguous after the reduction
        # MPI enforces the same data type for send and receive buffer
        # the reduction implementation takes care of making the internal Torch storage consistent
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(out._DNDarray__array.is_contiguous())
        if data.comm.rank == 0:
            self.assertTrue((out._DNDarray__array == data.comm.size).all())

    def test_scan(self):
        # contiguous data
        data = ht.ones((5, 3,), dtype=ht.float64)
        out = ht.zeros_like(data)

        # reduce across all nodes
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(out._DNDarray__array.is_contiguous())
        data.comm.Scan(data, out)

        # check the reduction result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(out._DNDarray__array.is_contiguous())
        self.assertTrue((out._DNDarray__array == data.comm.rank + 1).all())

        # non-contiguous data
        data = ht.ones((5, 3,), dtype=ht.float64).T
        out = ht.zeros_like(data)

        # reduce across all nodes
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertTrue(out._DNDarray__array.is_contiguous())
        data.comm.Scan(data, out)

        # check the reduction result
        # the data tensor will be contiguous after the reduction
        # MPI enforces the same data type for send and receive buffer
        # the reduction implementation takes care of making the internal Torch storage consistent
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(out._DNDarray__array.is_contiguous())
        self.assertTrue((out._DNDarray__array == data.comm.rank + 1).all())

        # non-contiguous output
        data = ht.ones((5, 3,), dtype=ht.float64)
        out = ht.zeros((3, 5), dtype=ht.float64).T

        # reduce across all nodes
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertFalse(out._DNDarray__array.is_contiguous())
        data.comm.Scan(data, out)

        # check the reduction result
        # the data tensor will be contiguous after the reduction
        # MPI enforces the same data type for send and receive buffer
        # the reduction implementation takes care of making the internal Torch storage consistent
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(out._DNDarray__array.is_contiguous())
        self.assertTrue((out._DNDarray__array == data.comm.rank + 1).all())

    def test_scatter(self):
        # contiguous data
        if ht.MPI_WORLD.rank == 0:
            data = ht.ones((ht.MPI_WORLD.size, 5))
        else:
            data = ht.zeros((1,))
        output = ht.zeros((1, 5,))

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        data.comm.Scatter(data, output, root=0)

        # check scatter result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        self.assertTrue((output._DNDarray__array == torch.ones(1, 5,)).all())

        # contiguous data, different scatter axis
        if ht.MPI_WORLD.rank == 0:
            data = ht.ones((5, ht.MPI_WORLD.size,))
        else:
            data = ht.zeros((1,))
        output = ht.zeros((5, 1,))

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        data.comm.Scatter(data, output, root=0, axis=1)

        # check scatter result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        self.assertTrue((output._DNDarray__array == torch.ones(5, 1,)).all())

        # non-contiguous data
        if ht.MPI_WORLD.rank == 0:
            data = ht.ones((5, ht.MPI_WORLD.size * 2,)).T
            self.assertFalse(data._DNDarray__array.is_contiguous())
        else:
            data = ht.zeros((1,))
            self.assertTrue(data._DNDarray__array.is_contiguous())
        output = ht.zeros((2, 5))

        # ensure prior invariants
        self.assertTrue(output._DNDarray__array.is_contiguous())
        data.comm.Scatter(data, output, root=0)

        # check scatter result
        if ht.MPI_WORLD.rank == 0:
            self.assertFalse(data._DNDarray__array.is_contiguous())
        else:
            self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        self.assertTrue((output._DNDarray__array == torch.ones(2, 5,)).all())

        # non-contiguous destination, different split axis
        if ht.MPI_WORLD.rank == 0:
            data = ht.ones((5, ht.MPI_WORLD.size * 2,))
        else:
            data = ht.zeros((1,))
        output = ht.zeros((2, 5)).T

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())
        data.comm.Scatter(data, output, root=0, axis=1)

        # check scatter result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())
        self.assertTrue((output._DNDarray__array == torch.ones(5, 2,)).all())

    def test_scatter_like_axes(self):
        # input and output are not split
        data = ht.array([[ht.MPI_WORLD.rank] * ht.MPI_WORLD.size] * ht.MPI_WORLD.size)
        output = ht.zeros_like(data)

        # main axis send buffer, main axis receive buffer
        data.comm.Alltoall(data, output, axis=0)
        comparison = torch.arange(ht.MPI_WORLD.size).reshape(-1, 1).repeat(1, ht.MPI_WORLD.size)
        self.assertTrue((output._DNDarray__array == comparison).all())

        # minor axis send buffer, main axis receive buffer
        data.comm.Alltoall(data, output, axis=1)
        comparison = torch.arange(ht.MPI_WORLD.size).reshape(1, -1).repeat(ht.MPI_WORLD.size, 1)
        self.assertTrue((output._DNDarray__array == comparison).all())

        # main axis send buffer, minor axis receive buffer
        data = ht.array([[ht.MPI_WORLD.rank] * (2 * ht.MPI_WORLD.size)] * ht.MPI_WORLD.size)
        output = ht.zeros((2 * ht.MPI_WORLD.size, ht.MPI_WORLD.size), dtype=data.dtype)
        data.comm.Alltoall(data, output, axis=0, recv_axis=1)
        comparison = torch.arange(ht.MPI_WORLD.size).reshape(1, -1).repeat(2 * ht.MPI_WORLD.size, 1)
        self.assertTrue((output._DNDarray__array == comparison).all())

        # minor axis send buffer, minor axis receive buffer
        data = ht.array([range(ht.MPI_WORLD.size)] * ht.MPI_WORLD.size)
        output = ht.zeros((ht.MPI_WORLD.size, ht.MPI_WORLD.size), dtype=data.dtype)
        data.comm.Alltoall(data, output, axis=1, recv_axis=1)
        comparison = torch.arange(ht.MPI_WORLD.size).reshape(-1, 1).repeat(1, ht.MPI_WORLD.size)
        self.assertTrue((output._DNDarray__array == comparison).all())

    def test_scatterv(self):
        # contiguous data buffer, contiguous output buffer
        input_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1)
        data = ht.ones((input_count, 12,))
        output_count = 2 * (ht.MPI_WORLD.rank + 1)
        output = ht.zeros((output_count, 12,))

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())

        # perform the scatter operation
        counts = tuple(range(2, 2 * (ht.MPI_WORLD.size + 1), 2))
        displs = tuple(np.cumsum(range(0, 2 * ht.MPI_WORLD.size, 2)))
        data.comm.Scatterv((data, counts, displs,), output, root=0)

        # check scatter result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        self.assertTrue((output._DNDarray__array == torch.ones(output_count, 12,)).all())

        # non-contiguous data buffer, contiguous output buffer
        input_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1)
        data = ht.ones((12, input_count,)).T
        output_count = 2 * (ht.MPI_WORLD.rank + 1)
        output = ht.zeros((output_count, 12,))

        # ensure prior invariants
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())

        # perform the scatter operation
        counts = tuple(range(2, 2 * (ht.MPI_WORLD.size + 1), 2))
        displs = tuple(np.cumsum(range(0, 2 * ht.MPI_WORLD.size, 2)))
        data.comm.Scatterv((data, counts, displs,), output, root=0)

        # check scatter result
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertTrue(output._DNDarray__array.is_contiguous())
        self.assertTrue((output._DNDarray__array == torch.ones(output_count, 12,)).all())

        # contiguous data buffer, non-contiguous output buffer
        input_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1)
        data = ht.ones((input_count, 12,))
        output_count = 2 * (ht.MPI_WORLD.rank + 1)
        output = ht.zeros((12, output_count,)).T

        # ensure prior invariants
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())

        # perform the scatter operation
        counts = tuple(range(2, 2 * (ht.MPI_WORLD.size + 1), 2))
        displs = tuple(np.cumsum(range(0, 2 * ht.MPI_WORLD.size, 2)))
        data.comm.Scatterv((data, counts, displs,), output, root=0)

        # check scatter result
        self.assertTrue(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())
        self.assertTrue((output._DNDarray__array == torch.ones(output_count, 12,)).all())

        # non-contiguous data buffer, non-contiguous output buffer
        input_count = ht.MPI_WORLD.size * (ht.MPI_WORLD.size + 1)
        data = ht.ones((12, input_count,)).T
        output_count = 2 * (ht.MPI_WORLD.rank + 1)
        output = ht.zeros((12, output_count,)).T

        # ensure prior invariants
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())

        # perform the scatter operation
        counts = tuple(range(2, 2 * (ht.MPI_WORLD.size + 1), 2))
        displs = tuple(np.cumsum(range(0, 2 * ht.MPI_WORLD.size, 2)))
        data.comm.Scatterv((data, counts, displs,), output, root=0)

        # check scatter result
        self.assertFalse(data._DNDarray__array.is_contiguous())
        self.assertFalse(output._DNDarray__array.is_contiguous())
        self.assertTrue((output._DNDarray__array == torch.ones(output_count, 12,)).all())

    def test_allgathervSorting(self):
        data = ht.zeros((2, 3, 4))
        data[0, 0, 0] = 0
        data[0, 0, 1] = 1
        data[0, 0, 2] = 2
        data[0, 0, 3] = 3

        data[0, 1, 0] = 10
        data[0, 1, 1] = 11
        data[0, 1, 2] = 12
        data[0, 1, 3] = 13

        data[0, 2, 0] = 20
        data[0, 2, 1] = 21
        data[0, 2, 2] = 22
        data[0, 2, 3] = 23

        data[1, 0, 0] = 100
        data[1, 0, 1] = 101
        data[1, 0, 2] = 102
        data[1, 0, 3] = 103

        data[1, 1, 0] = 110
        data[1, 1, 1] = 111
        data[1, 1, 2] = 112
        data[1, 1, 3] = 113

        data[1, 2, 0] = 120
        data[1, 2, 1] = 121
        data[1, 2, 2] = 122
        data[1, 2, 3] = 123

        result = data


        data.resplit_(axis=0)

        gathered = torch.empty(data.shape)
        recv_counts, recv_displs, _ = data.comm.counts_displs_shape(data.shape, data.split)
        data.comm.Allgatherv(data._DNDarray__array, (gathered, recv_counts, recv_displs,), send_axis=data.split)

        self.assertTrue(ht.equal(data, result))

        data.resplit_(axis=1)

        gathered2 = torch.empty(data.shape)
        recv_counts2, recv_displs2, _ = data.comm.counts_displs_shape(data.shape, data.split)
        data.comm.Allgatherv(data._DNDarray__array, (gathered2, recv_counts2, recv_displs2,), send_axis=data.split)

        self.assertTrue(ht.equal(data, result))

        data.resplit_(axis=2)

        gathered2 = torch.empty(data.shape)
        recv_counts2, recv_displs2, _ = data.comm.counts_displs_shape(data.shape, data.split)
        data.comm.Allgatherv(data._DNDarray__array, (gathered2, recv_counts2, recv_displs2,), send_axis=data.split)

        self.assertTrue(ht.equal(data, result))
