import numpy as np
import torch

import heat as ht
from .test_suites.basic_test import TestCase

pytorch_major_version = int(torch.__version__.split(".")[0])


class TestDNDarray(TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     super(TestDNDarray, cls).setUpClass()
    #     N = ht.MPI_WORLD.size
    #     cls.reference_tensor = ht.zeros((N, N + 1, 2 * N))

    #     for n in range(N):
    #         for m in range(N + 1):
    #             cls.reference_tensor[n, m, :] = ht.arange(0, 2 * N) + m * 10 + n * 100

    def test_and(self):
        int16_tensor = ht.array([[1, 1], [2, 2]], dtype=ht.int16)
        int16_vector = ht.array([[3, 4]], dtype=ht.int16)

        self.assertTrue(
            ht.equal(int16_tensor & int16_vector, ht.bitwise_and(int16_tensor, int16_vector))
        )

    def test_gethalo(self):
        data_np = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
        data = ht.array(data_np, split=1)

        if data.comm.size == 2:
            halo_next = torch.tensor(np.array([[4, 5], [10, 11]]), device=data.device.torch_device)
            halo_prev = torch.tensor(np.array([[2, 3], [8, 9]]), device=data.device.torch_device)

            data.get_halo(2)

            data_with_halos = data.array_with_halos
            self.assertEqual(data_with_halos.shape, (2, 5))

            if data.comm.rank == 0:
                self.assertTrue(torch.equal(data.halo_next, halo_next))
                self.assertEqual(data.halo_prev, None)
            if data.comm.rank == 1:
                self.assertTrue(torch.equal(data.halo_prev, halo_prev))
                self.assertEqual(data.halo_next, None)

            self.assertEqual(data.array_with_halos.shape, (2, 5))
            # exception on wrong argument type in get_halo
            with self.assertRaises(TypeError):
                data.get_halo("wrong_type")
            # exception on wrong argument in get_halo
            with self.assertRaises(ValueError):
                data.get_halo(-99)
            # exception for too large halos
            with self.assertRaises(ValueError):
                data.get_halo(4)

            data_np = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])
            data = ht.array(data_np, split=1)

            halo_next = torch.tensor(
                np.array([[4.0, 5.0], [10.0, 11.0]]), device=data.device.torch_device
            )
            halo_prev = torch.tensor(
                np.array([[2.0, 3.0], [8.0, 9.0]]), device=data.device.torch_device
            )

            data.get_halo(2)

            if data.comm.rank == 0:
                self.assertTrue(np.isclose(((data.halo_next - halo_next) ** 2).mean().item(), 0.0))
                self.assertEqual(data.halo_prev, None)
            if data.comm.rank == 1:
                self.assertTrue(np.isclose(((data.halo_prev - halo_prev) ** 2).mean().item(), 0.0))
                self.assertEqual(data.halo_next, None)

            data = ht.ones((10, 2), split=0)

            halo_next = torch.tensor(
                np.array([[1.0, 1.0], [1.0, 1.0]]), device=data.device.torch_device
            )
            halo_prev = torch.tensor(
                np.array([[1.0, 1.0], [1.0, 1.0]]), device=data.device.torch_device
            )

            data.get_halo(2)

            if data.comm.rank == 0:
                self.assertTrue(np.isclose(((data.halo_next - halo_next) ** 2).mean().item(), 0.0))
                self.assertEqual(data.halo_prev, None)
            if data.comm.rank == 1:
                self.assertTrue(np.isclose(((data.halo_prev - halo_prev) ** 2).mean().item(), 0.0))
                self.assertEqual(data.halo_next, None)

        if data.comm.size == 3:
            halo_1 = torch.tensor(np.array([[2], [8]]), device=data.device.torch_device)
            halo_2 = torch.tensor(np.array([[3], [9]]), device=data.device.torch_device)
            halo_3 = torch.tensor(np.array([[4], [10]]), device=data.device.torch_device)
            halo_4 = torch.tensor(np.array([[5], [11]]), device=data.device.torch_device)
            data.get_halo(1)
            data_with_halos = data.array_with_halos

            if data.comm.rank == 0:
                self.assertTrue(torch.equal(data.halo_next, halo_2))
                self.assertEqual(data.halo_prev, None)
                self.assertEqual(data_with_halos.shape, (2, 3))
            if data.comm.rank == 1:
                self.assertTrue(torch.equal(data.halo_prev, halo_1))
                self.assertTrue(torch.equal(data.halo_next, halo_4))
                self.assertEqual(data_with_halos.shape, (2, 4))
            if data.comm.rank == 2:
                self.assertEqual(data.halo_next, None)
                self.assertTrue(torch.equal(data.halo_prev, halo_3))
                self.assertEqual(data_with_halos.shape, (2, 3))
            # exception on wrong argument type in get_halo
            with self.assertRaises(TypeError):
                data.get_halo("wrong_type")
            # exception on wrong argument in get_halo
            with self.assertRaises(ValueError):
                data.get_halo(-99)
            # exception for too large halos
            with self.assertRaises(ValueError):
                data.get_halo(4)
            # test no data on process
            data_np = np.arange(2 * 12).reshape(2, 12)
            data = ht.array(data_np, split=0)
            data.get_halo(1)

            data_with_halos = data.array_with_halos

            if data.comm.rank == 0:
                self.assertTrue(data.halo_prev is None)
                self.assertTrue(data.halo_next is not None)
                self.assertEqual(data_with_halos.shape, (2, 12))
            if data.comm.rank == 1:
                self.assertTrue(data.halo_prev is not None)
                self.assertTrue(data.halo_next is None)
                self.assertEqual(data_with_halos.shape, (2, 12))
            if data.comm.rank == 2:
                self.assertTrue(data.halo_prev is None)
                self.assertTrue(data.halo_next is None)
                self.assertEqual(data_with_halos.shape, (0, 12))

            data = data.reshape((12, 2), new_split=1)
            data.get_halo(1)

            data_with_halos = data.array_with_halos

            if data.comm.rank == 0:
                self.assertTrue(data.halo_prev is None)
                self.assertTrue(data.halo_next is not None)
                self.assertEqual(data_with_halos.shape, (12, 2))
            if data.comm.rank == 1:
                self.assertTrue(data.halo_prev is not None)
                self.assertTrue(data.halo_next is None)
                self.assertEqual(data_with_halos.shape, (12, 2))
            if data.comm.rank == 2:
                self.assertTrue(data.halo_prev is None)
                self.assertTrue(data.halo_next is None)
                self.assertEqual(data_with_halos.shape, (12, 0))

        # test halo of imbalanced dndarray
        if data.comm.size > 2:
            # test for split=0
            t_data = torch.arange(
                5 * data.comm.rank, dtype=torch.float64, device=data.larray.device
            ).reshape(data.comm.rank, 5)
            if data.comm.rank > 0:
                prev_data = torch.arange(
                    5 * (data.comm.rank - 1), dtype=torch.float64, device=data.larray.device
                ).reshape(data.comm.rank - 1, 5)
            if data.comm.rank < data.comm.size - 1:
                next_data = torch.arange(
                    5 * (data.comm.rank + 1), dtype=torch.float64, device=data.larray.device
                ).reshape(data.comm.rank + 1, 5)
            data = ht.array(t_data, is_split=0)
            data.get_halo(1)
            data_with_halos = data.array_with_halos
            if data.comm.rank == 0:
                prev_halo = None
                next_halo = None
                new_split_size = 0
            elif data.comm.rank == 1:
                prev_halo = None
                next_halo = next_data[0]
                new_split_size = data.larray.shape[0] + 1
            elif data.comm.rank == data.comm.size - 1:
                prev_halo = prev_data[-1]
                next_halo = None
                new_split_size = data.larray.shape[0] + 1
            else:
                prev_halo = prev_data[-1]
                next_halo = next_data[0]
                new_split_size = data.larray.shape[0] + 2
            self.assertEqual(data_with_halos.shape, (new_split_size, 5))
            self.assertTrue(data.halo_prev is prev_halo or (data.halo_prev == prev_halo).all())
            self.assertTrue(data.halo_next is next_halo or (data.halo_next == next_halo).all())

            # test for split=1
            t_data = torch.arange(
                5 * data.comm.rank, dtype=torch.float64, device=data.larray.device
            ).reshape(5, -1)
            if data.comm.rank > 0:
                prev_data = torch.arange(
                    5 * (data.comm.rank - 1), dtype=torch.float64, device=data.larray.device
                ).reshape(5, -1)
            if data.comm.rank < data.comm.size - 1:
                next_data = torch.arange(
                    5 * (data.comm.rank + 1), dtype=torch.float64, device=data.larray.device
                ).reshape(5, -1)
            data = ht.array(t_data, is_split=1)
            data.get_halo(1)
            data_with_halos = data.array_with_halos
            if data.comm.rank == 0:
                prev_halo = None
                next_halo = None
                new_split_size = 0
            elif data.comm.rank == 1:
                prev_halo = None
                next_halo = next_data[:, 0].unsqueeze_(1)
                new_split_size = data.larray.shape[1] + 1
            elif data.comm.rank == data.comm.size - 1:
                prev_halo = prev_data[:, -1].unsqueeze_(1)
                next_halo = None
                new_split_size = data.larray.shape[1] + 1
            else:
                prev_halo = prev_data[:, -1].unsqueeze_(1)
                next_halo = next_data[:, 0].unsqueeze_(1)
                new_split_size = data.larray.shape[1] + 2
            self.assertEqual(data_with_halos.shape, (5, new_split_size))
            self.assertTrue(data.halo_prev is prev_halo or (data.halo_prev == prev_halo).all())
            self.assertTrue(data.halo_next is next_halo or (data.halo_next == next_halo).all())

    def test_array(self):
        # undistributed case
        x = ht.arange(6 * 7 * 8).reshape((6, 7, 8))
        x_np = np.arange(6 * 7 * 8, dtype=np.int32).reshape((6, 7, 8))

        self.assertTrue((x.__array__() == x_np).all())
        self.assertIsInstance(x.__array__(), np.ndarray)
        self.assertEqual(x.__array__().dtype, x_np.dtype)
        self.assertEqual(x.__array__().shape, x.gshape)

        # distributed case
        x = ht.arange(6 * 7 * 8, dtype=ht.float64, split=0).reshape((6, 7, 8))
        x_np = np.arange(6 * 7 * 8, dtype=np.float64).reshape((6, 7, 8))

        self.assertTrue((x.__array__() == x.larray.cpu().numpy()).all())
        self.assertIsInstance(x.__array__(), np.ndarray)
        self.assertEqual(x.__array__().dtype, x_np.dtype)
        self.assertEqual(x.__array__().shape, x.lshape)

    def test_larray(self):
        # undistributed case
        x = ht.arange(6 * 7 * 8).reshape((6, 7, 8))

        self.assertTrue((x.larray == x.larray).all())
        self.assertIsInstance(x.larray, torch.Tensor)
        self.assertEqual(x.larray.shape, x.lshape)

        x.larray = torch.randn(6, 7, 8)

        self.assertTrue((x.larray == x.larray).all())
        self.assertTrue(x.gshape, (6, 7, 8))
        self.assertIsInstance(x.larray, torch.Tensor)
        self.assertEqual(x.larray.shape, x.lshape)

        # Exceptions
        with self.assertRaises(ValueError):
            x.larray = torch.arange(42)
        with self.assertRaises(TypeError):
            x.larray = ht.array([1, 2, 3])
        with self.assertRaises(TypeError):
            x.larray = np.array([1, 2, 3])
        with self.assertRaises(TypeError):
            x.larray = [1, 2, 3]
        with self.assertRaises(TypeError):
            x.larray = "[1, 2, 3]"

        # distributed case
        x = ht.arange(6 * 7 * 8, split=0).reshape((6, 7, 8))

        self.assertTrue((x.larray == x.larray).all())
        self.assertIsInstance(x.larray, torch.Tensor)
        self.assertEqual(x.larray.shape, x.lshape)

        if x.comm.rank == 0:
            x.larray = torch.randn(4, 7, 8)

        self.assertTrue((x.larray == x.larray).all())
        # self.assertTrue(x.gshape, (42,))
        self.assertIsInstance(x.larray, torch.Tensor)
        self.assertEqual(x.larray.shape, x.lshape)
        self.assertEqual(x.split, 0)

        # Exceptions
        with self.assertRaises(ValueError):
            x.larray = torch.arange(42)
        with self.assertRaises(TypeError):
            x.larray = ht.array([1, 2, 3])
        with self.assertRaises(TypeError):
            x.larray = np.array([1, 2, 3])
        with self.assertRaises(TypeError):
            x.larray = [1, 2, 3]
        with self.assertRaises(TypeError):
            x.larray = "[1, 2, 3]"

    def test_astype(self):
        data = ht.float32([[1, 2, 3], [4, 5, 6]])

        # check starting invariant
        self.assertEqual(data.dtype, ht.float32)

        # check the copy case for uint8
        as_uint8 = data.astype(ht.uint8)
        self.assertIsInstance(as_uint8, ht.DNDarray)
        self.assertEqual(as_uint8.dtype, ht.uint8)
        self.assertEqual(as_uint8.larray.dtype, torch.uint8)
        self.assertIsNot(as_uint8, data)

        # check the copy case for uint8
        as_float64 = data.astype(ht.float64, copy=False)
        self.assertIsInstance(as_float64, ht.DNDarray)
        self.assertEqual(as_float64.dtype, ht.float64)
        self.assertEqual(as_float64.larray.dtype, torch.float64)
        self.assertIs(as_float64, data)

    def test_balance_and_lshape_map(self):
        data = ht.zeros((70, 20), split=0)
        data = data[:50]
        data.lshape_map
        lshape_map = data.create_lshape_map(force_check=False)  # tests the property
        self.assertEqual(sum(lshape_map[..., 0]), 50)
        if sum(data.lshape) == 0:
            self.assertTrue(all(lshape_map[data.comm.rank] == 0))
        data.balance_()
        self.assertTrue(data.is_balanced())

        data = ht.zeros((4, 120), split=1)
        data = data[:, 40:70]
        lshape_map = data.create_lshape_map()
        self.assertEqual(sum(lshape_map[..., 1]), 30)
        if sum(data.lshape) == 0:
            self.assertTrue(all(lshape_map[data.comm.rank] == 0))
        data.balance_()
        self.assertTrue(data.is_balanced())

        data = ht.zeros((70, 20), split=0, dtype=ht.float64)
        data = ht.balance(data[:50], copy=True)
        self.assertTrue(data.is_balanced())

        data = ht.zeros((4, 120), split=1, dtype=ht.int64)
        data = data[:, 40:70].balance()
        self.assertTrue(data.is_balanced())

        data = np.loadtxt("heat/datasets/iris.csv", delimiter=";")
        htdata = ht.load("heat/datasets/iris.csv", sep=";", split=0)
        self.assertTrue(ht.equal(htdata, ht.array(data, split=0, dtype=ht.float)))

        if ht.MPI_WORLD.size > 4:
            rank = ht.MPI_WORLD.rank
            if rank == 2:
                arr = torch.tensor([0, 1], device=htdata.device.torch_device)
            elif rank == 3:
                arr = torch.tensor([2, 3, 4, 5], device=htdata.device.torch_device)
            elif rank == 4:
                arr = torch.tensor([6, 7, 8, 9], device=htdata.device.torch_device)
            else:
                arr = torch.empty([0], dtype=torch.int64, device=htdata.device.torch_device)
            a = ht.array(arr, is_split=0)
            a.balance_()
            comp = ht.arange(10, split=0)

            self.assertTrue(ht.equal(a, comp))

    def test_bool_cast(self):
        # simple scalar tensor
        a = ht.ones(1)
        casted_a = bool(a)
        self.assertEqual(casted_a, True)
        self.assertIsInstance(casted_a, bool)

        # multi-dimensional scalar tensor
        b = ht.zeros((1, 1, 1, 1))
        casted_b = bool(b)
        self.assertEqual(casted_b, False)
        self.assertIsInstance(casted_b, bool)

        # split scalar tensor
        c = ht.full((1,), 5, split=0)
        casted_c = bool(c)
        self.assertEqual(casted_c, True)
        self.assertIsInstance(casted_c, bool)

        # exception on non-scalar tensor
        with self.assertRaises(TypeError):
            bool(ht.empty(1, 2, 1, 1))
        # exception on empty tensor
        with self.assertRaises(TypeError):
            bool(ht.empty((0, 1, 2)))
        # exception on split tensor, where each chunk has size 1
        if ht.MPI_WORLD.size > 1:
            with self.assertRaises(TypeError):
                bool(ht.full((ht.MPI_WORLD.size,), 2, split=0))

    def test_collect(self):
        st = ht.zeros((50,), split=0)
        if st.comm.size >= 3:
            st.collect_()
            if st.comm.rank == 0:
                self.assertEqual(st.lshape, (50,))
            else:
                self.assertEqual(st.lshape, (0,))

            st = ht.zeros((50, 50), split=1)
            st.collect_(2)
            if st.comm.rank == 2:
                self.assertEqual(st.lshape, (50, 50))
            else:
                self.assertEqual(st.lshape, (50, 0))
            st.collect_(1)
            if st.comm.rank == 1:
                self.assertEqual(st.lshape, (50, 50))
            else:
                self.assertEqual(st.lshape, (50, 0))

            st = ht.zeros((50, 81, 67), split=2)
            st.collect_(1)
            if st.comm.rank == 1:
                self.assertEqual(st.lshape, (50, 81, 67))
            else:
                self.assertEqual(st.lshape, (50, 81, 0))

            st = ht.zeros((5, 8, 31), split=None)  # nothing should happen
            st.collect_()
            self.assertEqual(st.lshape, st.gshape)

        st = ht.zeros((50, 81, 67), split=0)
        with self.assertRaises(TypeError):
            st.collect_("st.comm.size + 1")
        with self.assertRaises(TypeError):
            st.collect_(1.0)
        with self.assertRaises(TypeError):
            st.collect_((1, 3))
        with self.assertRaises(ValueError):
            st.collect_(st.comm.size + 1)

    def test_complex_cast(self):
        # simple scalar tensor
        a = ht.ones(1)
        casted_a = complex(a)
        self.assertEqual(casted_a, 1 + 0j)
        self.assertIsInstance(casted_a, complex)

        # multi-dimensional scalar tensor
        b = ht.zeros((1, 1, 1, 1))
        casted_b = complex(b)
        self.assertEqual(casted_b, 0 + 0j)
        self.assertIsInstance(casted_b, complex)

        # split scalar tensor
        c = ht.full((1,), 5, split=0)
        casted_c = complex(c)
        self.assertEqual(casted_c, 5 + 0j)
        self.assertIsInstance(casted_c, complex)

        # exception on non-scalar tensor
        with self.assertRaises(TypeError):
            complex(ht.empty(1, 2, 1, 1))
        # exception on empty tensor
        with self.assertRaises(TypeError):
            complex(ht.empty((0, 1, 2)))
        # exception on split tensor, where each chunk has size 1
        if ht.MPI_WORLD.size > 1:
            with self.assertRaises(TypeError):
                complex(ht.full((ht.MPI_WORLD.size,), 2, split=0))

    def test_counts_displs(self):
        # balanced distributed DNDarray
        a = ht.arange(128, split=0).reshape((8, 8, 2))
        counts, displs = a.counts_displs()
        comm_counts, comm_displs, _ = a.comm.counts_displs_shape(a.gshape, a.split)
        self.assertTrue(counts == comm_counts)
        self.assertTrue(displs == comm_displs)

        # non-balanced distributed DNDarray
        rank = a.comm.rank
        size = a.comm.size
        t_a = torch.ones(8, rank * 2, 2)
        comp_counts = torch.arange(0, size) * 2
        comp_displs = torch.cumsum(comp_counts, dim=0)
        a = ht.array(t_a, is_split=1)
        counts, displs = a.counts_displs()
        self.assertTrue((torch.tensor(counts) == comp_counts).all())
        self.assertTrue((torch.tensor(displs[1:]) == comp_displs[:-1]).all())

        # exception
        a_nosplit = ht.arange(128).reshape((8, 8, 2))
        with self.assertRaises(ValueError):
            a_nosplit.counts_displs()

    def test_flatten(self):
        a = ht.ones((4, 4, 4), split=1)
        result = ht.ones((64,), split=0)
        flat = a.flatten()

        self.assertEqual(flat.shape, result.shape)
        self.assertTrue(ht.equal(flat, result))

    def test_fill_diagonal(self):
        ref = ht.zeros((ht.MPI_WORLD.size * 2, ht.MPI_WORLD.size * 2), dtype=ht.float32, split=0)
        a = ht.eye(ht.MPI_WORLD.size * 2, dtype=ht.float32, split=0)
        a.fill_diagonal(0)
        self.assertTrue(ht.equal(a, ref))

        ref = ht.zeros((ht.MPI_WORLD.size * 2, ht.MPI_WORLD.size * 2), dtype=ht.int32, split=0)
        a = ht.eye(ht.MPI_WORLD.size * 2, dtype=ht.int32, split=0)
        a.fill_diagonal(0)
        self.assertTrue(ht.equal(a, ref))

        ref = ht.zeros((ht.MPI_WORLD.size * 2, ht.MPI_WORLD.size * 2), dtype=ht.float32, split=1)
        a = ht.eye(ht.MPI_WORLD.size * 2, dtype=ht.float32, split=1)
        a.fill_diagonal(0)
        self.assertTrue(ht.equal(a, ref))

        ref = ht.zeros((ht.MPI_WORLD.size * 2, ht.MPI_WORLD.size * 3), dtype=ht.float32, split=0)
        a = ht.eye((ht.MPI_WORLD.size * 2, ht.MPI_WORLD.size * 3), dtype=ht.float32, split=0)
        a.fill_diagonal(0)
        self.assertTrue(ht.equal(a, ref))

        # ToDo: uneven tensor dimensions x and y when bug in factories.eye is fixed
        ref = ht.zeros((ht.MPI_WORLD.size * 3, ht.MPI_WORLD.size * 3), dtype=ht.float32, split=1)
        a = ht.eye((ht.MPI_WORLD.size * 3, ht.MPI_WORLD.size * 3), dtype=ht.float32, split=1)
        a.fill_diagonal(0)
        self.assertTrue(ht.equal(a, ref))

        # ToDo: uneven tensor dimensions x and y when bug in factories.eye is fixed
        ref = ht.zeros((ht.MPI_WORLD.size * 4, ht.MPI_WORLD.size * 4), dtype=ht.float32, split=0)
        a = ht.eye((ht.MPI_WORLD.size * 4, ht.MPI_WORLD.size * 4), dtype=ht.float32, split=0)
        a.fill_diagonal(0)
        self.assertTrue(ht.equal(a, ref))

        a = ht.ones((ht.MPI_WORLD.size * 2,), dtype=ht.float32, split=0)
        with self.assertRaises(ValueError):
            a.fill_diagonal(0)

    def test_float_cast(self):
        # simple scalar tensor
        a = ht.ones(1)
        casted_a = float(a)
        self.assertEqual(casted_a, 1.0)
        self.assertIsInstance(casted_a, float)

        # multi-dimensional scalar tensor
        b = ht.zeros((1, 1, 1, 1))
        casted_b = float(b)
        self.assertEqual(casted_b, 0.0)
        self.assertIsInstance(casted_b, float)

        # split scalar tensor
        c = ht.full((1,), 5, split=0)
        casted_c = float(c)
        self.assertEqual(casted_c, 5.0)
        self.assertIsInstance(casted_c, float)

        # exception on non-scalar tensor
        with self.assertRaises(TypeError):
            float(ht.empty(1, 2, 1, 1))
        # exception on empty tensor
        with self.assertRaises(TypeError):
            float(ht.empty((0, 1, 2)))
        # exception on split tensor, where each chunk has size 1
        if ht.MPI_WORLD.size > 1:
            with self.assertRaises(TypeError):
                float(ht.full((ht.MPI_WORLD.size,), 2, split=0))

    def test_getitem(self):
        # following https://numpy.org/doc/stable/user/basics.indexing.html

        # Single element indexing
        # 1D, local
        x = ht.arange(10)
        self.assertTrue(x[2].item() == 2)
        self.assertTrue(x[-2].item() == 8)
        self.assertTrue(x[2].dtype == ht.int32)
        # 1D, distributed
        x = ht.arange(10, split=0, dtype=ht.float64)
        self.assertTrue(x[2].item() == 2.0)
        self.assertTrue(x[-2].item() == 8.0)
        self.assertTrue(x[2].dtype == ht.float64)
        self.assertTrue(x[2].split is None)
        # 2D, local
        x = ht.arange(10).reshape(2, 5)
        self.assertTrue((x[0] == ht.arange(5)).all().item())
        self.assertTrue(x[0].dtype == ht.int32)
        # 2D, distributed
        x_split0 = ht.array(x, split=0)
        self.assertTrue((x_split0[0] == ht.arange(5, split=None)).all().item())
        x_split1 = ht.array(x, split=1)
        self.assertTrue((x_split1[-2] == ht.arange(5, split=0)).all().item())
        # 3D, local
        x = ht.arange(27).reshape(3, 3, 3)
        key = -2
        indexed = x[key]
        self.assertTrue((indexed.larray == x.larray[key]).all())
        self.assertTrue(indexed.dtype == ht.int32)
        self.assertTrue(indexed.split is None)
        # 3D, distributed, split = 0
        x_split0 = ht.array(x, dtype=ht.float32, split=0)
        indexed_split0 = x_split0[key]
        self.assertTrue((indexed_split0.larray == x.larray[key]).all())
        self.assertTrue(indexed_split0.dtype == ht.float32)
        self.assertTrue(indexed_split0.split is None)
        # 3D, distributed split, != 0
        x_split2 = ht.array(x, dtype=ht.int64, split=2)
        key = ht.array(2)
        indexed_split2 = x_split2[key]
        self.assertTrue((indexed_split2.numpy() == x.numpy()[key.item()]).all())
        self.assertTrue(indexed_split2.dtype == ht.int64)
        self.assertTrue(indexed_split2.split == 1)

        # Slicing and striding
        x = ht.arange(20, split=0)
        x_sliced = x[1:11:3]
        x_np = np.arange(20)
        x_sliced_np = x_np[1:11:3]
        self.assert_array_equal(x_sliced, x_sliced_np)
        self.assertTrue(x_sliced.split == 0)

        # 1-element slice along split axis
        x = ht.arange(20).reshape(4, 5)
        x.resplit_(axis=1)
        x_sliced = x[:, 2:3]
        x_np = np.arange(20).reshape(4, 5)
        x_sliced_np = x_np[:, 2:3]
        self.assert_array_equal(x_sliced, x_sliced_np)
        self.assertTrue(x_sliced.split == 1)

        # slicing with negative step along split axis 0
        shape = (20, 4, 3)
        x_3d = ht.arange(20 * 4 * 3, split=0).reshape(shape)
        x_3d_sliced = x_3d[17:2:-2, :2, ht.array(1)]
        x_3d_sliced_np = np.arange(20 * 4 * 3).reshape(shape)[17:2:-2, :2, 1]
        self.assert_array_equal(x_3d_sliced, x_3d_sliced_np)
        self.assertTrue(x_3d_sliced.split == 0)

        # slicing with negative step along split 1
        shape = (4, 20, 3)
        x_3d = ht.arange(20 * 4 * 3).reshape(shape)
        x_3d.resplit_(axis=1)
        key = (slice(None, 2), slice(17, 2, -2), 1)
        x_3d_sliced = x_3d[key]
        x_3d_sliced_np = np.arange(20 * 4 * 3).reshape(shape)[:2, 17:2:-2, 1]
        self.assert_array_equal(x_3d_sliced, x_3d_sliced_np)
        self.assertTrue(x_3d_sliced.split == 1)

        # slicing with negative step along split 2 and loss of axis < split
        shape = (4, 3, 20)
        x_3d = ht.arange(20 * 4 * 3).reshape(shape)
        x_3d.resplit_(axis=2)
        key = (slice(None, 2), 1, slice(17, 10, -2))
        x_3d_sliced = x_3d[key]
        x_3d_sliced_np = np.arange(20 * 4 * 3).reshape(shape)[:2, 1, 17:10:-2]
        self.assert_array_equal(x_3d_sliced, x_3d_sliced_np)
        self.assertTrue(x_3d_sliced.split == 1)

        # slicing with negative step along split 2 and loss of all axes but split
        shape = (4, 3, 20)
        x_3d = ht.arange(20 * 4 * 3).reshape(shape)
        x_3d.resplit_(axis=2)
        key = (0, 1, slice(17, 13, -1))
        x_3d_sliced = x_3d[key]
        x_3d_sliced_np = np.arange(20 * 4 * 3).reshape(shape)[0, 1, 17:13:-1]
        self.assert_array_equal(x_3d_sliced, x_3d_sliced_np)
        self.assertTrue(x_3d_sliced.split == 0)

        # DIMENSIONAL INDEXING
        # ellipsis
        x_np = np.array([[[1], [2], [3]], [[4], [5], [6]]])
        x_np_ellipsis = x_np[..., 0]
        x = ht.array([[[1], [2], [3]], [[4], [5], [6]]])

        # local
        x_ellipsis = x[..., 0]
        x_slice = x[:, :, 0]
        self.assert_array_equal(x_ellipsis, x_np_ellipsis)
        self.assert_array_equal(x_slice, x_np_ellipsis)

        # distributed
        x.resplit_(axis=1)
        x_ellipsis = x[..., 0]
        x_slice = x[:, :, 0]
        self.assert_array_equal(x_ellipsis, x_np_ellipsis)
        self.assert_array_equal(x_slice, x_np_ellipsis)
        self.assertTrue(x_ellipsis.split == 1)

        # newaxis: local
        x = ht.array([[[1], [2], [3]], [[4], [5], [6]]])
        x_np_newaxis = x_np[:, np.newaxis, :2, :]
        x_newaxis = x[:, np.newaxis, :2, :]
        x_none = x[:, None, :2, :]
        self.assert_array_equal(x_newaxis, x_np_newaxis)
        self.assert_array_equal(x_none, x_np_newaxis)

        # newaxis: distributed
        x.resplit_(axis=1)
        x_newaxis = x[:, np.newaxis, :2, :]
        x_none = x[:, None, :2, :]
        self.assert_array_equal(x_newaxis, x_np_newaxis)
        self.assert_array_equal(x_none, x_np_newaxis)
        self.assertTrue(x_newaxis.split == 2)
        self.assertTrue(x_none.split == 2)

        x = ht.arange(5, split=0)
        x_np = np.arange(5)
        y = x[:, np.newaxis] + x[np.newaxis, :]
        y_np = x_np[:, np.newaxis] + x_np[np.newaxis, :]
        self.assert_array_equal(y, y_np)
        self.assertTrue(y.split == 0)

        # ADVANCED INDEXING
        # "x[(1, 2, 3),] is fundamentally different from x[(1, 2, 3)]"

        x_np = np.arange(60).reshape(5, 3, 4)
        indexed_x_np = x_np[(1, 2, 3)]
        adv_indexed_x_np = x_np[(1, 2, 3),]
        x = ht.array(x_np, split=0)
        indexed_x = x[(1, 2, 3)]
        self.assertTrue(indexed_x.item() == np.array(indexed_x_np))
        adv_indexed_x = x[(1, 2, 3),]
        self.assert_array_equal(adv_indexed_x, adv_indexed_x_np)

        # 1d
        x = ht.arange(10, 1, -1, split=0)
        x_np = np.arange(10, 1, -1)
        x_adv_ind = x[np.array([3, 3, 1, 8])]
        x_np_adv_ind = x_np[np.array([3, 3, 1, 8])]
        self.assert_array_equal(x_adv_ind, x_np_adv_ind)

        # 3d, split 0, non-unique, non-ordered key along split axis
        x = ht.arange(60, split=0).reshape(5, 3, 4)
        x_np = np.arange(60).reshape(5, 3, 4)
        k1 = np.array([0, 4, 1, 0])
        k2 = np.array([0, 2, 1, 0])
        k3 = np.array([1, 2, 3, 1])
        self.assert_array_equal(
            x[ht.array(k1, split=0), ht.array(k2, split=0), ht.array(k3, split=0)], x_np[k1, k2, k3]
        )
        # advanced indexing on non-consecutive dimensions
        x = ht.arange(60, split=0).reshape(5, 3, 4, new_split=1)
        x_copy = x.copy()
        x_np = np.arange(60).reshape(5, 3, 4)
        k1 = np.array([0, 4, 1, 0])
        k2 = 0
        k3 = np.array([1, 2, 3, 1])
        key = (k1, k2, k3)
        self.assert_array_equal(x[key], x_np[key])
        # check that x is unchanged after internal manipulation
        self.assertTrue(x.shape == x_copy.shape)
        self.assertTrue(x.split == x_copy.split)
        self.assertTrue(x.lshape == x_copy.lshape)
        self.assertTrue((x == x_copy).all().item())

        # broadcasting shapes
        x.resplit_(axis=0)
        self.assert_array_equal(x[ht.array(k1, split=0), ht.array(1), 2], x_np[k1, 1, 2])
        # test exception: broadcasting mismatching shapes
        k2 = np.array([0, 2, 1])
        with self.assertRaises(IndexError):
            x[k1, k2, k3]

        # more broadcasting
        x_np = np.arange(12).reshape(4, 3)
        rows = np.array([0, 3])
        cols = np.array([0, 2])
        x = ht.arange(12).reshape(4, 3)
        x.resplit_(1)
        x_np_indexed = x_np[rows[:, np.newaxis], cols]
        x_indexed = x[ht.array(rows)[:, np.newaxis], cols]
        self.assert_array_equal(x_indexed, x_np_indexed)
        self.assertTrue(x_indexed.split == 1)

        # combining advanced and basic indexing
        y_np = np.arange(35).reshape(5, 7)
        y_np_indexed = y_np[np.array([0, 2, 4]), 1:3]
        y = ht.array(y_np, split=1)
        y_indexed = y[ht.array([0, 2, 4]), 1:3]
        self.assert_array_equal(y_indexed, y_np_indexed)
        self.assertTrue(y_indexed.split == 1)

        x_np = np.arange(10 * 20 * 30).reshape(10, 20, 30)
        x = ht.array(x_np, split=1)
        ind_array = ht.random.randint(0, 20, (2, 3, 4), dtype=ht.int64)
        ind_array_np = ind_array.numpy()
        x_np_indexed = x_np[..., ind_array_np, :]
        x_indexed = x[..., ind_array, :]
        self.assert_array_equal(x_indexed, x_np_indexed)
        self.assertTrue(x_indexed.split == 3)

        # boolean mask, local
        arr = ht.arange(3 * 4 * 5).reshape(3, 4, 5)
        np.random.seed(42)
        mask = np.random.randint(0, 2, arr.shape, dtype=bool)
        self.assertTrue((arr[mask].numpy() == arr.numpy()[mask]).all())

        # boolean mask, distributed
        arr_split0 = ht.array(arr, split=0)
        mask_split0 = ht.array(mask, split=0)
        self.assertTrue((arr_split0[mask_split0].numpy() == arr.numpy()[mask]).all())

        arr_split1 = ht.array(arr, split=1)
        mask_split1 = ht.array(mask, split=1)
        self.assert_array_equal(arr_split1[mask_split1], arr.numpy()[mask])

        arr_split2 = ht.array(arr, split=2)
        mask_split2 = ht.array(mask, split=2)
        self.assert_array_equal(arr_split2[mask_split2], arr.numpy()[mask])

    def test_int_cast(self):
        # simple scalar tensor
        a = ht.ones(1)
        casted_a = int(a)
        self.assertEqual(casted_a, 1)
        self.assertIsInstance(casted_a, int)

        # multi-dimensional scalar tensor
        b = ht.zeros((1, 1, 1, 1))
        casted_b = int(b)
        self.assertEqual(casted_b, 0)
        self.assertIsInstance(casted_b, int)

        # split scalar tensor
        c = ht.full((1,), 5, split=0)
        casted_c = int(c)
        self.assertEqual(casted_c, 5)
        self.assertIsInstance(casted_c, int)

        # exception on non-scalar tensor
        with self.assertRaises(TypeError):
            int(ht.empty(1, 2, 1, 1))
        # exception on empty tensor
        with self.assertRaises(TypeError):
            int(ht.empty((0, 1, 2)))
        # exception on split tensor, where each chunk has size 1
        if ht.MPI_WORLD.size > 1:
            with self.assertRaises(TypeError):
                int(ht.full((ht.MPI_WORLD.size,), 2, split=0))

    def test_invert(self):
        int_tensor = ht.array([[0, 1], [2, -2]])
        bool_tensor = ht.array([[False, True], [True, False]])
        float_tensor = ht.array([[0.4, 1.3], [1.3, -2.1]])
        int_result = ht.array([[-1, -2], [-3, 1]])
        bool_result = ht.array([[True, False], [False, True]])

        self.assertTrue(ht.equal(~int_tensor, int_result))
        self.assertTrue(ht.equal(~bool_tensor, bool_result))

        with self.assertRaises(TypeError):
            ~float_tensor

    def test_is_balanced(self):
        data = ht.zeros((70, 20), split=0)
        if data.comm.size != 1:
            data = data[:50]
            self.assertFalse(data.is_balanced())
            data.balance_()
            self.assertTrue(data.is_balanced())

    def test_is_distributed(self):
        data = ht.zeros((5, 5))
        self.assertFalse(data.is_distributed())

        data = ht.zeros((4, 4), split=0)
        self.assertTrue(data.comm.size > 1 and data.is_distributed() or not data.is_distributed())

    def test_item(self):
        x = ht.zeros((1,))
        self.assertEqual(x.item(), 0)
        self.assertEqual(type(x.item()), float)

        x = ht.zeros((1, 2))

        with self.assertRaises(ValueError):
            x.item()

    def test_len(self):
        # vector
        a = ht.zeros((10,))
        a_length = len(a)

        self.assertIsInstance(a_length, int)
        self.assertEqual(a_length, 10)

        # matrix
        b = ht.ones((50, 2))
        b_length = len(b)

        self.assertIsInstance(b_length, int)
        self.assertEqual(b_length, 50)

        # split 5D array
        c = ht.empty((3, 4, 5, 6, 7), split=-1)
        c_length = len(c)

        self.assertIsInstance(c_length, int)
        self.assertEqual(c_length, 3)

    def test_lloc(self):
        # single set
        a = ht.zeros((13, 5), split=0)
        a.lloc[0, 0] = 1
        self.assertEqual(a.larray[0, 0], 1)
        self.assertEqual(a.lloc[0, 0].dtype, torch.float32)

        # multiple set
        a = ht.zeros((13, 5), split=0)
        a.lloc[1:3, 1] = 1
        self.assertTrue(all(a.larray[1:3, 1] == 1))
        self.assertEqual(a.lloc[1:3, 1].dtype, torch.float32)

        # multiple set with specific indexing
        a = ht.zeros((13, 5), split=0)
        a.lloc[3:7:2, 2:5:2] = 1
        self.assertTrue(torch.all(a.larray[3:7:2, 2:5:2] == 1))
        self.assertEqual(a.lloc[3:7:2, 2:5:2].dtype, torch.float32)

    def test_lnbytes(self):
        # undistributed case

        # integer
        x_uint8 = ht.arange(6 * 7 * 8, dtype=ht.uint8).reshape((6, 7, 8))
        x_int8 = ht.arange(6 * 7 * 8, dtype=ht.int8).reshape((6, 7, 8))
        x_int16 = ht.arange(6 * 7 * 8, dtype=ht.int16).reshape((6, 7, 8))
        x_int32 = ht.arange(6 * 7 * 8, dtype=ht.int32).reshape((6, 7, 8))
        x_int64 = ht.arange(6 * 7 * 8, dtype=ht.int64).reshape((6, 7, 8))

        # float
        x_float32 = ht.arange(6 * 7 * 8, dtype=ht.float32).reshape((6, 7, 8))
        x_float64 = ht.arange(6 * 7 * 8, dtype=ht.float64).reshape((6, 7, 8))

        # bool
        x_bool = ht.arange(6 * 7 * 8, dtype=ht.bool).reshape((6, 7, 8))

        self.assertEqual(x_uint8.lnbytes, x_uint8.gnbytes)
        self.assertEqual(x_int8.lnbytes, x_int8.gnbytes)
        self.assertEqual(x_int16.lnbytes, x_int16.gnbytes)
        self.assertEqual(x_int32.lnbytes, x_int32.gnbytes)
        self.assertEqual(x_int64.lnbytes, x_int64.gnbytes)

        self.assertEqual(x_float32.lnbytes, x_float32.gnbytes)
        self.assertEqual(x_float64.lnbytes, x_float64.gnbytes)

        self.assertEqual(x_bool.lnbytes, x_bool.gnbytes)

        # distributed case

        # integer
        x_uint8_d = ht.arange(6 * 7 * 8, split=0, dtype=ht.uint8)
        x_int8_d = ht.arange(6 * 7 * 8, split=0, dtype=ht.int8)
        x_int16_d = ht.arange(6 * 7 * 8, split=0, dtype=ht.int16)
        x_int32_d = ht.arange(6 * 7 * 8, split=0, dtype=ht.int32)
        x_int64_d = ht.arange(6 * 7 * 8, split=0, dtype=ht.int64)

        # float
        x_float32_d = ht.arange(6 * 7 * 8, split=0, dtype=ht.float32)
        x_float64_d = ht.arange(6 * 7 * 8, split=0, dtype=ht.float64)

        # bool
        x_bool_d = ht.arange(6 * 7 * 8, split=0, dtype=ht.bool)

        self.assertEqual(x_uint8_d.lnbytes, x_uint8_d.lnumel * 1)
        self.assertEqual(x_int8_d.lnbytes, x_int8_d.lnumel * 1)
        self.assertEqual(x_int16_d.lnbytes, x_int16_d.lnumel * 2)
        self.assertEqual(x_int32_d.lnbytes, x_int32_d.lnumel * 4)
        self.assertEqual(x_int64_d.lnbytes, x_int64_d.lnumel * 8)

        self.assertEqual(x_float32_d.lnbytes, x_float32_d.lnumel * 4)
        self.assertEqual(x_float64_d.lnbytes, x_float64_d.lnumel * 8)

        self.assertEqual(x_bool_d.lnbytes, x_bool_d.lnumel * 1)

    def test_nbytes(self):
        # undistributed case

        # integer
        x_uint8 = ht.arange(6 * 7 * 8, dtype=ht.uint8).reshape((6, 7, 8))
        x_int8 = ht.arange(6 * 7 * 8, dtype=ht.int8).reshape((6, 7, 8))
        x_int16 = ht.arange(6 * 7 * 8, dtype=ht.int16).reshape((6, 7, 8))
        x_int32 = ht.arange(6 * 7 * 8, dtype=ht.int32).reshape((6, 7, 8))
        x_int64 = ht.arange(6 * 7 * 8, dtype=ht.int64).reshape((6, 7, 8))

        # float
        x_float32 = ht.arange(6 * 7 * 8, dtype=ht.float32).reshape((6, 7, 8))
        x_float64 = ht.arange(6 * 7 * 8, dtype=ht.float64).reshape((6, 7, 8))

        # bool
        x_bool = ht.arange(6 * 7 * 8, dtype=ht.bool).reshape((6, 7, 8))

        self.assertEqual(x_uint8.nbytes, 336 * 1)
        self.assertEqual(x_int8.nbytes, 336 * 1)
        self.assertEqual(x_int16.nbytes, 336 * 2)
        self.assertEqual(x_int32.nbytes, 336 * 4)
        self.assertEqual(x_int64.nbytes, 336 * 8)

        self.assertEqual(x_float32.nbytes, 336 * 4)
        self.assertEqual(x_float64.nbytes, 336 * 8)

        self.assertEqual(x_bool.nbytes, 336 * 1)

        # equivalent function gnbytes
        self.assertEqual(x_uint8.nbytes, x_uint8.gnbytes)
        self.assertEqual(x_int8.nbytes, x_int8.gnbytes)
        self.assertEqual(x_int16.nbytes, x_int16.gnbytes)
        self.assertEqual(x_int32.nbytes, x_int32.gnbytes)
        self.assertEqual(x_int64.nbytes, x_int64.gnbytes)

        self.assertEqual(x_float32.nbytes, x_float32.gnbytes)
        self.assertEqual(x_float64.nbytes, x_float64.gnbytes)

        self.assertEqual(x_bool.nbytes, x_bool.gnbytes)

        # distributed case

        # integer
        x_uint8_d = ht.arange(6 * 7 * 8, split=0, dtype=ht.uint8)
        x_int8_d = ht.arange(6 * 7 * 8, split=0, dtype=ht.int8)
        x_int16_d = ht.arange(6 * 7 * 8, split=0, dtype=ht.int16)
        x_int32_d = ht.arange(6 * 7 * 8, split=0, dtype=ht.int32)
        x_int64_d = ht.arange(6 * 7 * 8, split=0, dtype=ht.int64)

        # float
        x_float32_d = ht.arange(6 * 7 * 8, split=0, dtype=ht.float32)
        x_float64_d = ht.arange(6 * 7 * 8, split=0, dtype=ht.float64)

        # bool
        x_bool_d = ht.arange(6 * 7 * 8, split=0, dtype=ht.bool)

        self.assertEqual(x_uint8_d.nbytes, 336 * 1)
        self.assertEqual(x_int8_d.nbytes, 336 * 1)
        self.assertEqual(x_int16_d.nbytes, 336 * 2)
        self.assertEqual(x_int32_d.nbytes, 336 * 4)
        self.assertEqual(x_int64_d.nbytes, 336 * 8)

        self.assertEqual(x_float32_d.nbytes, 336 * 4)
        self.assertEqual(x_float64_d.nbytes, 336 * 8)

        self.assertEqual(x_bool_d.nbytes, 336 * 1)

        # equivalent function gnbytes
        self.assertEqual(x_uint8_d.nbytes, x_uint8_d.gnbytes)
        self.assertEqual(x_int8_d.nbytes, x_int8_d.gnbytes)
        self.assertEqual(x_int16_d.nbytes, x_int16_d.gnbytes)
        self.assertEqual(x_int32_d.nbytes, x_int32_d.gnbytes)
        self.assertEqual(x_int64_d.nbytes, x_int64_d.gnbytes)

        self.assertEqual(x_float32_d.nbytes, x_float32_d.gnbytes)
        self.assertEqual(x_float64_d.nbytes, x_float64_d.gnbytes)

        self.assertEqual(x_bool_d.nbytes, x_bool_d.gnbytes)

    def test_ndim(self):
        a = ht.empty([2, 3, 3, 2])
        self.assertEqual(a.ndim, 4)

    def test_numpy(self):
        # ToDo: numpy does not work for distributed tensors du to issue#
        # Add additional tests if the issue is solved
        a = np.random.randn(10, 8)
        b = ht.array(a)
        self.assertIsInstance(b.numpy(), np.ndarray)
        self.assertEqual(b.numpy().shape, a.shape)
        self.assertEqual(b.numpy().tolist(), b.larray.cpu().numpy().tolist())

        a = ht.ones((10, 8), dtype=ht.float32)
        b = np.ones((2, 2)).astype("float32")
        self.assertEqual(a.numpy().dtype, b.dtype)

        a = ht.ones((10, 8), dtype=ht.float64)
        b = np.ones((2, 2)).astype("float64")
        self.assertEqual(a.numpy().dtype, b.dtype)

        a = ht.ones((10, 8), dtype=ht.int32)
        b = np.ones((2, 2)).astype("int32")
        self.assertEqual(a.numpy().dtype, b.dtype)

        a = ht.ones((10, 8), dtype=ht.int64)
        b = np.ones((2, 2)).astype("int64")
        self.assertEqual(a.numpy().dtype, b.dtype)

    def test_or(self):
        int16_tensor = ht.array([[1, 1], [2, 2]], dtype=ht.int16)
        int16_vector = ht.array([[3, 4]], dtype=ht.int16)

        self.assertTrue(
            ht.equal(int16_tensor | int16_vector, ht.bitwise_or(int16_tensor, int16_vector))
        )

    def test_partitioned(self):
        a = ht.zeros((120, 120), split=0)
        parted = a.__partitioned__
        self.assertEqual(parted["shape"], (120, 120))
        self.assertEqual(parted["partition_tiling"], (a.comm.size, 1))
        self.assertEqual(parted["partitions"][(0, 0)]["start"], (0, 0))

        a.resplit_(None)
        self.assertIsNone(a.__partitions_dict__)
        parted = a.__partitioned__
        self.assertEqual(parted["shape"], (120, 120))
        self.assertEqual(parted["partition_tiling"], (1, 1))
        self.assertEqual(parted["partitions"][(0, 0)]["start"], (0, 0))

    def test_redistribute(self):
        # need to test with 1, 2, 3, and 4 dims
        st = ht.zeros((50,), split=0)
        if st.comm.size >= 3:
            target_map = torch.zeros(
                (st.comm.size, 1), dtype=torch.int, device=self.device.torch_device
            )
            target_map[1] = 30
            target_map[2] = 20
            st.redistribute_(target_map=target_map)
            if st.comm.rank == 1:
                self.assertEqual(st.lshape, (30,))
            elif st.comm.rank == 2:
                self.assertEqual(st.lshape, (20,))
            else:
                self.assertEqual(st.lshape, (0,))

            st = ht.zeros((50, 50), split=1)
            target_map = torch.zeros(
                (st.comm.size, 2), dtype=torch.int, device=self.device.torch_device
            )
            target_map[0, 1] = 13
            target_map[2, 1] = 50 - 13
            st.redistribute_(target_map=target_map)
            if st.comm.rank == 0:
                self.assertEqual(st.lshape, (50, 13))
            elif st.comm.rank == 2:
                self.assertEqual(st.lshape, (50, 50 - 13))
            else:
                self.assertEqual(st.lshape, (50, 0))

            st = ht.zeros((50, 81, 67), split=2)
            target_map = torch.zeros(
                (st.comm.size, 3), dtype=torch.int, device=self.device.torch_device
            )
            target_map[0, 2] = 67
            st.redistribute_(target_map=target_map)
            if st.comm.rank == 0:
                self.assertEqual(st.lshape, (50, 81, 67))
            else:
                self.assertEqual(st.lshape, (50, 81, 0))

            st = ht.zeros((8, 8, 8), split=None)
            target_map = torch.zeros(
                (st.comm.size, 3), dtype=torch.int, device=self.device.torch_device
            )
            # this will do nothing!
            st.redistribute_(target_map=target_map)
            self.assertTrue(st.lshape, st.gshape)

            st = ht.zeros((50, 81, 67), split=0)
            with self.assertRaises(ValueError):
                target_map *= 0
                st.redistribute_(target_map=target_map)

            with self.assertRaises(TypeError):
                st.redistribute_(target_map="sdfibn")
            with self.assertRaises(TypeError):
                st.redistribute_(lshape_map="sdfibn")
            with self.assertRaises(ValueError):
                st.redistribute_(lshape_map=torch.zeros(2))
            with self.assertRaises(ValueError):
                st.redistribute_(target_map=torch.zeros((2, 4)))

    def test_repr(self):
        a = ht.array([1, 2, 3, 4])
        self.assertEqual(a.__repr__(), a.__str__())

    def test_resplit(self):
        # resplitting with same axis, should leave everything unchanged
        shape = (ht.MPI_WORLD.size, ht.MPI_WORLD.size)
        data = ht.zeros(shape, split=None)
        data.resplit_(None)

        self.assertIsInstance(data, ht.DNDarray)
        self.assertEqual(data.shape, shape)
        self.assertEqual(data.lshape, shape)
        self.assertEqual(data.split, None)

        # resplitting with same axis, should leave everything unchanged
        shape = (ht.MPI_WORLD.size, ht.MPI_WORLD.size)
        data = ht.zeros(shape, split=1)
        data.resplit_(1)

        self.assertIsInstance(data, ht.DNDarray)
        self.assertEqual(data.shape, shape)
        self.assertEqual(data.lshape, (data.comm.size, 1))
        self.assertEqual(data.split, 1)

        # splitting an unsplit tensor should result in slicing the tensor locally
        shape = (ht.MPI_WORLD.size, ht.MPI_WORLD.size)
        data = ht.zeros(shape)
        data.resplit_(-1)

        self.assertIsInstance(data, ht.DNDarray)
        self.assertEqual(data.shape, shape)
        self.assertEqual(data.lshape, (data.comm.size, 1))
        self.assertEqual(data.split, 1)

        # unsplitting, aka gathering a tensor
        shape = (ht.MPI_WORLD.size + 1, ht.MPI_WORLD.size)
        data = ht.ones(shape, split=0)
        data.resplit_(None)

        self.assertIsInstance(data, ht.DNDarray)
        self.assertEqual(data.shape, shape)
        self.assertEqual(data.lshape, shape)
        self.assertEqual(data.split, None)

        # assign and entirely new split axis
        shape = (ht.MPI_WORLD.size + 2, ht.MPI_WORLD.size + 1)
        data = ht.ones(shape, split=0)
        data.resplit_(1)

        self.assertIsInstance(data, ht.DNDarray)
        self.assertEqual(data.shape, shape)
        self.assertEqual(data.lshape[0], ht.MPI_WORLD.size + 2)
        self.assertTrue(data.lshape[1] == 1 or data.lshape[1] == 2)
        self.assertEqual(data.split, 1)

        # test sorting order of resplit
        a_tensor = self.reference_tensor.copy()
        N = ht.MPI_WORLD.size

        # split along axis = 0
        a_tensor.resplit_(axis=0)
        local_shape = (1, N + 1, 2 * N)
        local_tensor = self.reference_tensor[ht.MPI_WORLD.rank, :, :]
        self.assertEqual(a_tensor.lshape, local_shape)
        self.assertTrue((a_tensor.larray == local_tensor.larray).all())

        # unsplit
        a_tensor.resplit_(axis=None)
        self.assertTrue((a_tensor.larray == self.reference_tensor.larray).all())

        # split along axis = 1
        a_tensor.resplit_(axis=1)
        if ht.MPI_WORLD.rank == 0:
            local_shape = (N, 2, 2 * N)
            local_tensor = self.reference_tensor[:, 0:2, :]
        else:
            local_shape = (N, 1, 2 * N)
            local_tensor = self.reference_tensor[
                :, ht.MPI_WORLD.rank + 1 : ht.MPI_WORLD.rank + 2, :
            ]

        self.assertEqual(a_tensor.lshape, local_shape)
        self.assertTrue((a_tensor.larray == local_tensor.larray).all())

        # unsplit
        a_tensor.resplit_(axis=None)
        self.assertTrue((a_tensor.larray == self.reference_tensor.larray).all())

        # split along axis = 2
        a_tensor.resplit_(axis=2)
        local_shape = (N, N + 1, 2)
        local_tensor = self.reference_tensor[
            :, :, 2 * ht.MPI_WORLD.rank : 2 * ht.MPI_WORLD.rank + 2
        ]

        self.assertEqual(a_tensor.lshape, local_shape)
        self.assertTrue((a_tensor.larray == local_tensor.larray).all())

        expected = torch.ones(
            (ht.MPI_WORLD.size, 100), dtype=torch.int64, device=self.device.torch_device
        )
        data = ht.array(expected, split=1)
        data.resplit_(None)

        self.assertTrue(torch.equal(data.larray, expected))
        self.assertFalse(data.is_distributed())
        self.assertIsNone(data.split)
        self.assertEqual(data.dtype, ht.int64)
        self.assertEqual(data.larray.dtype, expected.dtype)

        expected = torch.zeros(
            (100, ht.MPI_WORLD.size), dtype=torch.uint8, device=self.device.torch_device
        )
        data = ht.array(expected, split=0)
        data.resplit_(None)

        self.assertTrue(torch.equal(data.larray, expected))
        self.assertFalse(data.is_distributed())
        self.assertIsNone(data.split)
        self.assertEqual(data.dtype, ht.uint8)
        self.assertEqual(data.larray.dtype, expected.dtype)

        # "in place"
        length = torch.tensor([i + 20 for i in range(2)], device=self.device.torch_device)
        test = torch.arange(
            torch.prod(length), dtype=torch.float64, device=self.device.torch_device
        ).reshape([i + 20 for i in range(2)])
        a = ht.array(test, split=1)
        a.resplit_(axis=0)
        self.assertTrue(ht.equal(a, ht.array(test, split=0)))
        self.assertEqual(a.split, 0)
        self.assertEqual(a.dtype, ht.float64)
        del a

        test = torch.arange(torch.prod(length), device=self.device.torch_device)
        a = ht.array(test, split=0)
        a.resplit_(axis=None)
        self.assertTrue(ht.equal(a, ht.array(test, split=None)))
        self.assertEqual(a.split, None)
        self.assertEqual(a.dtype, ht.int64)
        del a

        a = ht.array(test, split=None)
        a.resplit_(axis=0)
        self.assertTrue(ht.equal(a, ht.array(test, split=0)))
        self.assertEqual(a.split, 0)
        self.assertEqual(a.dtype, ht.int64)
        del a

        a = ht.array(test, split=0)
        resplit_a = ht.manipulations.resplit(a, axis=None)
        self.assertTrue(ht.equal(resplit_a, ht.array(test, split=None)))
        self.assertEqual(resplit_a.split, None)
        self.assertEqual(resplit_a.dtype, ht.int64)
        del a

        a = ht.array(test, split=None)
        resplit_a = ht.manipulations.resplit(a, axis=0)
        self.assertTrue(ht.equal(resplit_a, ht.array(test, split=0)))
        self.assertEqual(resplit_a.split, 0)
        self.assertEqual(resplit_a.dtype, ht.int64)
        del a

        # 1D non-contiguous resplit testing
        t1 = ht.arange(10 * 10, split=0).reshape((10, 10))
        t1_sub = t1[:, 1]  # .expand_dims(0)
        res = ht.array([1, 11, 21, 31, 41, 51, 61, 71, 81, 91])
        t1_sub.resplit_(axis=None)
        self.assertTrue(ht.all(t1_sub == res))
        self.assertEqual(t1_sub.split, None)

    def test_setitem(self):
        # following https://numpy.org/doc/stable/user/basics.indexing.html

        # Single element indexing
        # 1D, local
        x = ht.zeros(10)
        x[2] = 2
        x[-2] = 8
        self.assertTrue(x[2].item() == 2)
        self.assertTrue(x[-2].item() == 8)
        self.assertTrue(x[2].dtype == ht.float32)
        # 1D, distributed
        x = ht.zeros(10, split=0, dtype=ht.float64)
        x[2] = 2
        x[-2] = 8
        self.assertTrue(x[2].item() == 2.0)
        self.assertTrue(x[-2].item() == 8.0)
        self.assertTrue(x[2].dtype == ht.float64)
        self.assertTrue(x.split == 0)
        # 2D, local
        x = ht.zeros(10).reshape(2, 5)
        x[0] = ht.arange(5)
        self.assertTrue((x[0] == ht.arange(5)).all().item())
        self.assertTrue(x[0].dtype == ht.float32)
        # 2D, distributed
        x_split0 = ht.zeros(10, split=0).reshape(2, 5)
        x_split0[0] = ht.arange(5)
        self.assertTrue((x_split0[0] == ht.arange(5, split=None)).all().item())
        x_split1 = ht.zeros(10, split=0).reshape(2, 5, new_split=1)
        x_split1[-2] = ht.arange(5)
        self.assertTrue((x_split1[-2] == ht.arange(5, split=0)).all().item())
        # 3D, distributed, split = 0
        x_split0 = ht.zeros(27, split=0).reshape(3, 3, 3)
        key = -2
        x_split0[key] = ht.arange(3)
        self.assertTrue((x_split0[key].larray == torch.arange(3)).all())
        self.assertTrue(x_split0[key].dtype == ht.float32)
        self.assertTrue(x_split0.split == 0)
        # 3D, distributed split, != 0
        x_split2 = ht.zeros(27, dtype=ht.int64, split=0).reshape(3, 3, 3, new_split=2)
        key = ht.array(2)
        x_split2[key] = [6, 7, 8]
        indexed_split2 = x_split2[key]
        self.assertTrue((indexed_split2.numpy()[0] == np.array([6, 7, 8])).all())
        self.assertTrue(indexed_split2.dtype == ht.int64)
        self.assertTrue(x_split2.split == 2)

        # Slicing and striding
        x = ht.arange(20, split=0)
        x[1:11:3] = ht.array([10, 40, 70, 100])
        x_np = np.arange(20)
        x_np[1:11:3] = np.array([10, 40, 70, 100])
        self.assert_array_equal(x, x_np)
        self.assertTrue(x.split == 0)

        # 1-element slice along split axis
        x = ht.arange(20).reshape(4, 5)
        x.resplit_(axis=1)
        x[:, 2:3] = ht.array([10, 40, 70, 100]).reshape(4, 1)
        x_np = np.arange(20).reshape(4, 5)
        x_np[:, 2:3] = np.array([10, 40, 70, 100]).reshape(4, 1)
        self.assert_array_equal(x, x_np)
        self.assertTrue(x.split == 1)
        with self.assertRaises(ValueError):
            x[:, 2:3] = ht.array([10, 40, 70, 100])

        # slicing with negative step along split axis 0
        # assign different dtype
        shape = (20, 4, 3)
        x_3d = ht.arange(20 * 4 * 3, split=0).reshape(shape)
        value = ht.random.randn(8, 2)
        x_3d[17:2:-2, :2, ht.array(1)] = value
        x_3d_sliced = x_3d[17:2:-2, :2, ht.array(1)]
        self.assertTrue(ht.allclose(x_3d_sliced, value.astype(x_3d.dtype)))
        self.assertTrue(x_3d_sliced.dtype == x_3d.dtype)

        # slicing with negative step along split 1
        shape = (4, 20, 3)
        x_3d = ht.arange(20 * 4 * 3, dtype=ht.float32).reshape(shape)
        x_3d.resplit_(axis=1)
        key = (slice(None, 2), slice(17, 2, -2), 1)
        value = ht.random.randn(2, 8)
        x_3d[key] = value
        x_3d_sliced = x_3d[key]
        self.assertTrue(ht.allclose(x_3d_sliced, value.astype(x_3d.dtype)))
        self.assertTrue(x_3d_sliced.dtype == x_3d.dtype)

        # slicing with negative step along split 2 and loss of axis < split
        shape = (4, 3, 20)
        x_3d = ht.arange(20 * 4 * 3, dtype=ht.float64).reshape(shape)
        x_3d.resplit_(axis=2)
        key = (slice(None, 2), 1, slice(17, 10, -2))
        value = ht.random.randn(2, 4)
        x_3d[key] = value
        x_3d_sliced = x_3d[key]
        self.assertTrue(ht.allclose(x_3d_sliced, value.astype(x_3d.dtype)))
        self.assertTrue(x_3d_sliced.dtype == x_3d.dtype)

        # slicing with negative step along split 2 and loss of all axes but split
        shape = (4, 3, 20)
        x_3d = ht.arange(20 * 4 * 3).reshape(shape)
        x_3d.resplit_(axis=2)
        key = (0, 1, slice(17, 13, -1))
        value = ht.random.randint(
            0,
            5,
            (
                1,
                4,
            ),
            split=1,
        )
        x_3d[key] = value
        x_3d_sliced = x_3d[key]
        self.assertTrue(ht.allclose(x_3d_sliced, value.squeeze(0).astype(x_3d.dtype)))
        self.assertTrue(x_3d_sliced.dtype == x_3d.dtype)

        # DIMENSIONAL INDEXING

        # ellipsis
        x = ht.array([[[1], [2], [3]], [[4], [5], [6]]])
        # local
        value = x.squeeze() + 7
        x[..., 0] = value
        self.assertTrue(ht.all(x[..., 0] == value).item())
        value -= 7
        x[:, :, 0] = value
        self.assertTrue(ht.all(x[:, :, 0] == value).item())

        # distributed
        x.resplit_(axis=1)
        value *= 2
        x[..., 0] = value
        x_ellipsis = x[..., 0]
        self.assertTrue(ht.all(x_ellipsis == value).item())
        value += 2
        x[:, :, 0] = value
        self.assertTrue(ht.all(x[:, :, 0] == value).item())
        self.assertTrue(x_ellipsis.split == 1)

        # newaxis: local, w. broadcasting and different dtype
        x = ht.array([[[1], [2], [3]], [[4], [5], [6]]])
        value = ht.array([10.0, 20.0]).reshape(2, 1)
        x[:, None, :2, :] = value
        x_newaxis = x[:, None, :2, :]
        self.assertTrue(ht.all(x_newaxis == value).item())
        value += 2
        x[:, None, :2, :] = value
        self.assertTrue(ht.all(x[:, None, :2, :] == value).item())
        self.assertTrue(x[:, None, :2, :].dtype == x.dtype)

        # newaxis: distributed w. broadcasting and different dtype
        x.resplit_(axis=1)
        value = ht.array([30.0, 40.0]).reshape(1, 2, 1)
        x[:, np.newaxis, :2, :] = value
        x_newaxis = x[:, np.newaxis, :2, :]
        self.assertTrue(ht.all(x_newaxis == value).item())
        value += 2
        x[:, None, :2, :] = value
        x_none = x[:, None, :2, :]
        self.assertTrue(ht.all(x_none == value).item())
        self.assertTrue(x_none.dtype == x.dtype)

        # distributed value
        x = ht.arange(6).reshape(1, 1, 2, 3)
        x.resplit_(axis=-1)
        value = ht.arange(3).reshape(1, 3)
        value.resplit_(axis=1)
        x[..., 0, :] = value
        self.assertTrue(ht.all(x[..., 0, :] == value).item())

        # ADVANCED INDEXING
        # "x[(1, 2, 3),] is fundamentally different from x[(1, 2, 3)]"

        x = ht.arange(60, split=0).reshape(5, 3, 4)
        value = 99.0
        x[(1, 2, 3)] = value
        indexed_x = x[(1, 2, 3)]
        self.assertTrue((indexed_x == value).item())
        self.assertTrue(indexed_x.dtype == x.dtype)
        x[(1, 2, 3),] = value
        adv_indexed_x = x[(1, 2, 3),]
        self.assertTrue(ht.all(adv_indexed_x == value).item())
        self.assertTrue(adv_indexed_x.dtype == x.dtype)

        # 1d
        x = ht.arange(10, 1, -1, split=0)
        value = ht.arange(4)
        x[ht.array([3, 2, 1, 8])] = value
        x_adv_ind = x[np.array([3, 2, 1, 8])]
        self.assertTrue(ht.all(x_adv_ind == value).item())
        self.assertTrue(x_adv_ind.dtype == x.dtype)
        # # 3d, split 0, non-unique, non-ordered key along split axis
        # x = ht.arange(60, split=0).reshape(5, 3, 4)
        # x_np = np.arange(60).reshape(5, 3, 4)
        # k1 = np.array([0, 4, 1, 0])
        # k2 = np.array([0, 2, 1, 0])
        # k3 = np.array([1, 2, 3, 1])
        # self.assert_array_equal(
        #     x[ht.array(k1, split=0), ht.array(k2, split=0), ht.array(k3, split=0)], x_np[k1, k2, k3]
        # )
        # # advanced indexing on non-consecutive dimensions
        # x = ht.arange(60, split=0).reshape(5, 3, 4, new_split=1)
        # x_copy = x.copy()
        # x_np = np.arange(60).reshape(5, 3, 4)
        # k1 = np.array([0, 4, 1, 0])
        # k2 = 0
        # k3 = np.array([1, 2, 3, 1])
        # key = (k1, k2, k3)
        # self.assert_array_equal(x[key], x_np[key])
        # # check that x is unchanged after internal manipulation
        # self.assertTrue(x.shape == x_copy.shape)
        # self.assertTrue(x.split == x_copy.split)
        # self.assertTrue(x.lshape == x_copy.lshape)
        # self.assertTrue((x == x_copy).all().item())

        # # broadcasting shapes
        # x.resplit_(axis=0)
        # self.assert_array_equal(x[ht.array(k1, split=0), ht.array(1), 2], x_np[k1, 1, 2])
        # # test exception: broadcasting mismatching shapes
        # k2 = np.array([0, 2, 1])
        # with self.assertRaises(IndexError):
        #     x[k1, k2, k3]

        # # more broadcasting
        # x_np = np.arange(12).reshape(4, 3)
        # rows = np.array([0, 3])
        # cols = np.array([0, 2])
        # x = ht.arange(12).reshape(4, 3)
        # x.resplit_(1)
        # x_np_indexed = x_np[rows[:, np.newaxis], cols]
        # x_indexed = x[ht.array(rows)[:, np.newaxis], cols]
        # self.assert_array_equal(x_indexed, x_np_indexed)
        # self.assertTrue(x_indexed.split == 1)

        # # combining advanced and basic indexing
        # y_np = np.arange(35).reshape(5, 7)
        # y_np_indexed = y_np[np.array([0, 2, 4]), 1:3]
        # y = ht.array(y_np, split=1)
        # y_indexed = y[ht.array([0, 2, 4]), 1:3]
        # self.assert_array_equal(y_indexed, y_np_indexed)
        # self.assertTrue(y_indexed.split == 1)

        # x_np = np.arange(10 * 20 * 30).reshape(10, 20, 30)
        # x = ht.array(x_np, split=1)
        # ind_array = ht.random.randint(0, 20, (2, 3, 4), dtype=ht.int64)
        # ind_array_np = ind_array.numpy()
        # x_np_indexed = x_np[..., ind_array_np, :]
        # x_indexed = x[..., ind_array, :]
        # self.assert_array_equal(x_indexed, x_np_indexed)
        # self.assertTrue(x_indexed.split == 3)

        # boolean mask, local
        arr = ht.arange(3 * 4 * 5).reshape(3, 4, 5)
        np.random.seed(42)
        mask = np.random.randint(0, 2, arr.shape, dtype=bool)
        value = 99.0
        arr[mask] = value
        self.assertTrue((arr[mask] == value).all().item())
        self.assertTrue(arr[mask].dtype == arr.dtype)
        value = ht.ones_like(arr)
        arr[mask] = value[mask]
        self.assertTrue((arr[mask] == value[mask]).all().item())

        # boolean mask, distributed, non-distributed `value`
        arr_split0 = ht.array(arr, split=0)
        mask_split0 = ht.array(mask, split=0)
        arr_split0[mask_split0] = value[mask]
        self.assertTrue((arr_split0[mask_split0] == value[mask]).all().item())
        arr_split1 = ht.array(arr, split=1)
        mask_split1 = ht.array(mask, split=1)
        arr_split1[mask_split1] = value[mask]
        self.assertTrue((arr_split1[mask_split1] == value[mask]).all().item())
        arr_split2 = ht.array(arr, split=2)
        mask_split2 = ht.array(mask, split=2)
        arr_split2[mask_split2] = value[mask]
        self.assertTrue((arr_split2[mask_split2] == value[mask]).all().item())

        # TODO boolean mask, distributed, distributed `value`

    # def test_setitem_getitem(self):
    #     # tests for bug #825
    #     a = ht.ones((102, 102), split=0)
    #     setting = ht.zeros((100, 100), split=0)
    #     a[1:-1, 1:-1] = setting
    #     self.assertTrue(ht.all(a[1:-1, 1:-1] == 0))

    #     a = ht.ones((102, 102), split=1)
    #     setting = ht.zeros((30, 100), split=1)
    #     a[-30:, 1:-1] = setting
    #     self.assertTrue(ht.all(a[-30:, 1:-1] == 0))

    #     a = ht.ones((102, 102), split=1)
    #     setting = ht.zeros((100, 100), split=1)
    #     a[1:-1, 1:-1] = setting
    #     self.assertTrue(ht.all(a[1:-1, 1:-1] == 0))

    #     a = ht.ones((102, 102), split=1)
    #     setting = ht.zeros((100, 20), split=1)
    #     a[1:-1, :20] = setting
    #     self.assertTrue(ht.all(a[1:-1, :20] == 0))

    #     # tests for bug 730:
    #     a = ht.ones((10, 25, 30), split=1)
    #     if a.comm.size > 1:
    #         self.assertEqual(a[0].split, 0)
    #         self.assertEqual(a[:, 0, :].split, None)
    #         self.assertEqual(a[:, :, 0].split, 1)

    #     # set and get single value
    #     a = ht.zeros((13, 5), split=0)
    #     # set value on one node
    #     a[10, np.array(0)] = 1
    #     self.assertEqual(a[10, 0], 1)
    #     self.assertEqual(a[10, 0].dtype, ht.float32)

    #     a = ht.zeros((13, 5), split=0)
    #     a[10] = 1
    #     b = a[torch.tensor(10)]
    #     self.assertTrue((b == 1).all())
    #     self.assertEqual(b.dtype, ht.float32)
    #     self.assertEqual(b.gshape, (5,))

    #     a = ht.zeros((13, 5), split=0)
    #     a[-1] = 1
    #     b = a[-1]
    #     self.assertTrue((b == 1).all())
    #     self.assertEqual(b.dtype, ht.float32)
    #     self.assertEqual(b.gshape, (5,))

    #     # slice in 1st dim only on 1 node
    #     a = ht.zeros((13, 5), split=0)
    #     a[1:4] = 1
    #     self.assertTrue((a[1:4] == 1).all())
    #     self.assertEqual(a[1:4].gshape, (3, 5))
    #     self.assertEqual(a[1:4].split, 0)
    #     self.assertEqual(a[1:4].dtype, ht.float32)
    #     if a.comm.size == 2:
    #         if a.comm.rank == 0:
    #             self.assertEqual(a[1:4].lshape, (3, 5))
    #         else:
    #             self.assertEqual(a[1:4].lshape, (0, 5))

    #     a = ht.zeros((13, 5), split=0)
    #     a[1:2] = 1
    #     self.assertTrue((a[1:2] == 1).all())
    #     self.assertEqual(a[1:2].gshape, (1, 5))
    #     self.assertEqual(a[1:2].split, 0)
    #     self.assertEqual(a[1:2].dtype, ht.float32)
    #     if a.comm.size == 2:
    #         if a.comm.rank == 0:
    #             self.assertEqual(a[1:2].lshape, (1, 5))
    #         else:
    #             self.assertEqual(a[1:2].lshape, (0, 5))

    #     # slice in 1st dim only on 1 node w/ singular second dim
    #     a = ht.zeros((13, 5), split=0)
    #     a[1:4, 1] = 1
    #     b = a[1:4, np.int64(1)]
    #     self.assertTrue((b == 1).all())
    #     self.assertEqual(b.gshape, (3,))
    #     self.assertEqual(b.split, 0)
    #     self.assertEqual(b.dtype, ht.float32)
    #     if a.comm.size == 2:
    #         if a.comm.rank == 0:
    #             self.assertEqual(b.lshape, (3,))
    #         else:
    #             self.assertEqual(b.lshape, (0,))

    #     # slice in 1st dim across both nodes (2 node case) w/ singular second dim
    #     a = ht.zeros((13, 5), split=0)
    #     a[1:11, 1] = 1
    #     self.assertTrue((a[1:11, 1] == 1).all())
    #     self.assertEqual(a[1:11, 1].gshape, (10,))
    #     self.assertEqual(a[1:11, torch.tensor(1)].split, 0)
    #     self.assertEqual(a[1:11, 1].dtype, ht.float32)
    #     if a.comm.size == 2:
    #         if a.comm.rank == 1:
    #             self.assertEqual(a[1:11, 1].lshape, (4,))
    #         if a.comm.rank == 0:
    #             self.assertEqual(a[1:11, 1].lshape, (6,))

    #     # slice in 1st dim across 1 node (2nd) w/ singular second dim
    #     c = ht.zeros((13, 5), split=0)
    #     c[8:12, ht.array(1)] = 1
    #     b = c[8:12, np.int64(1)]
    #     self.assertTrue((b == 1).all())
    #     self.assertEqual(b.gshape, (4,))
    #     self.assertEqual(b.split, 0)
    #     self.assertEqual(b.dtype, ht.float32)
    #     if a.comm.size == 2:
    #         if a.comm.rank == 1:
    #             self.assertEqual(b.lshape, (4,))
    #         if a.comm.rank == 0:
    #             self.assertEqual(b.lshape, (0,))

    #     # slice in both directions
    #     a = ht.zeros((13, 5), split=0)
    #     a[3:13, 2:5:2] = 1
    #     self.assertTrue((a[3:13, 2:5:2] == 1).all())
    #     self.assertEqual(a[3:13, 2:5:2].gshape, (10, 2))
    #     self.assertEqual(a[3:13, 2:5:2].split, 0)
    #     self.assertEqual(a[3:13, 2:5:2].dtype, ht.float32)
    #     if a.comm.size == 2:
    #         if a.comm.rank == 1:
    #             self.assertEqual(a[3:13, 2:5:2].lshape, (6, 2))
    #         if a.comm.rank == 0:
    #             self.assertEqual(a[3:13, 2:5:2].lshape, (4, 2))

    #     # setting with heat tensor
    #     a = ht.zeros((4, 5), split=0)
    #     a[1, 0:4] = ht.arange(4)
    #     # if a.comm.size == 2:
    #     for c, i in enumerate(range(4)):
    #         self.assertEqual(a[1, c], i)

    #     # setting with torch tensor
    #     a = ht.zeros((4, 5), split=0)
    #     a[1, 0:4] = torch.arange(4, device=self.device.torch_device)
    #     # if a.comm.size == 2:
    #     for c, i in enumerate(range(4)):
    #         self.assertEqual(a[1, c], i)

    #     ###################################################
    #     a = ht.zeros((13, 5), split=1)
    #     # # set value on one node
    #     a[10] = 1
    #     self.assertEqual(a[10].dtype, ht.float32)
    #     if a.comm.size == 2:
    #         if a.comm.rank == 0:
    #             self.assertEqual(a[10].lshape, (3,))
    #         if a.comm.rank == 1:
    #             self.assertEqual(a[10].lshape, (2,))

    #     a = ht.zeros((13, 5), split=1)
    #     # # set value on one node
    #     a[10, 0] = 1
    #     self.assertEqual(a[10, 0], 1)
    #     self.assertEqual(a[10, 0].dtype, ht.float32)

    #     # slice in 1st dim only on 1 node
    #     a = ht.zeros((13, 5), split=1)
    #     a[1:4] = 1
    #     self.assertTrue((a[1:4] == 1).all())
    #     self.assertEqual(a[1:4].gshape, (3, 5))
    #     self.assertEqual(a[1:4].split, 1)
    #     self.assertEqual(a[1:4].dtype, ht.float32)
    #     if a.comm.size == 2:
    #         if a.comm.rank == 0:
    #             self.assertEqual(a[1:4].lshape, (3, 3))
    #         if a.comm.rank == 1:
    #             self.assertEqual(a[1:4].lshape, (3, 2))

    #     # slice in 1st dim only on 1 node w/ singular second dim
    #     a = ht.zeros((13, 5), split=1)
    #     a[1:4, 1] = 1
    #     self.assertTrue((a[1:4, 1] == 1).all())
    #     self.assertEqual(a[1:4, 1].gshape, (3,))
    #     self.assertEqual(a[1:4, 1].split, None)
    #     self.assertEqual(a[1:4, 1].dtype, ht.float32)
    #     if a.comm.size == 2:
    #         if a.comm.rank == 0:
    #             self.assertEqual(a[1:4, 1].lshape, (3,))
    #         if a.comm.rank == 1:
    #             self.assertEqual(a[1:4, 1].lshape, (3,))

    #     # slice in 2st dim across both nodes (2 node case) w/ singular fist dim
    #     a = ht.zeros((13, 5), split=1)
    #     a[11, 1:5] = 1
    #     self.assertTrue((a[11, 1:5] == 1).all())
    #     self.assertEqual(a[11, 1:5].gshape, (4,))
    #     self.assertEqual(a[11, 1:5].split, 0)
    #     self.assertEqual(a[11, 1:5].dtype, ht.float32)
    #     if a.comm.size == 2:
    #         if a.comm.rank == 1:
    #             self.assertEqual(a[11, 1:5].lshape, (2,))
    #         if a.comm.rank == 0:
    #             self.assertEqual(a[11, 1:5].lshape, (2,))

    #     # slice in 1st dim across 1 node (2nd) w/ singular second dim
    #     a = ht.zeros((13, 5), split=1)
    #     a[8:12, 1] = 1
    #     self.assertTrue((a[8:12, 1] == 1).all())
    #     self.assertEqual(a[8:12, 1].gshape, (4,))
    #     self.assertEqual(a[8:12, 1].split, None)
    #     self.assertEqual(a[8:12, 1].dtype, ht.float32)
    #     if a.comm.size == 2:
    #         if a.comm.rank == 0:
    #             self.assertEqual(a[8:12, 1].lshape, (4,))
    #         if a.comm.rank == 1:
    #             self.assertEqual(a[8:12, 1].lshape, (4,))

    #     # slice in both directions
    #     a = ht.zeros((13, 5), split=1)
    #     a[3:13, 2::2] = 1
    #     self.assertTrue((a[3:13, 2:5:2] == 1).all())
    #     self.assertEqual(a[3:13, 2:5:2].gshape, (10, 2))
    #     self.assertEqual(a[3:13, 2:5:2].split, 1)
    #     self.assertEqual(a[3:13, 2:5:2].dtype, ht.float32)
    #     if a.comm.size == 2:
    #         if a.comm.rank == 1:
    #             self.assertEqual(a[3:13, 2:5:2].lshape, (10, 1))
    #         if a.comm.rank == 0:
    #             self.assertEqual(a[3:13, 2:5:2].lshape, (10, 1))

    #     a = ht.zeros((13, 5), split=1)
    #     a[..., 2::2] = 1
    #     self.assertTrue((a[:, 2:5:2] == 1).all())
    #     self.assertEqual(a[..., 2:5:2].gshape, (13, 2))
    #     self.assertEqual(a[..., 2:5:2].split, 1)
    #     self.assertEqual(a[..., 2:5:2].dtype, ht.float32)
    #     if a.comm.size == 2:
    #         if a.comm.rank == 1:
    #             self.assertEqual(a[..., 2:5:2].lshape, (13, 1))
    #         if a.comm.rank == 0:
    #             self.assertEqual(a[:, 2:5:2].lshape, (13, 1))

    #     # setting with heat tensor
    #     a = ht.zeros((4, 5), split=1)
    #     a[1, 0:4] = ht.arange(4)
    #     for c, i in enumerate(range(4)):
    #         b = a[1, c]
    #         if b.larray.numel() > 0:
    #             self.assertEqual(b.item(), i)

    #     # setting with torch tensor
    #     a = ht.zeros((4, 5), split=1)
    #     a[1, 0:4] = torch.arange(4, device=self.device.torch_device)
    #     for c, i in enumerate(range(4)):
    #         self.assertEqual(a[1, c], i)

    #     ####################################################
    #     a = ht.zeros((13, 5, 7), split=2)
    #     # # set value on one node
    #     a[10, :, :] = 1
    #     self.assertEqual(a[10, :, :].dtype, ht.float32)
    #     self.assertEqual(a[10, :, :].gshape, (5, 7))
    #     if a.comm.size == 2:
    #         if a.comm.rank == 0:
    #             self.assertEqual(a[10, :, :].lshape, (5, 4))
    #         if a.comm.rank == 1:
    #             self.assertEqual(a[10, :, :].lshape, (5, 3))

    #     a = ht.zeros((13, 5, 7), split=2)
    #     # # set value on one node
    #     a[10, ...] = 1
    #     self.assertEqual(a[10, ...].dtype, ht.float32)
    #     self.assertEqual(a[10, ...].gshape, (5, 7))
    #     if a.comm.size == 2:
    #         if a.comm.rank == 0:
    #             self.assertEqual(a[10, ...].lshape, (5, 4))
    #         if a.comm.rank == 1:
    #             self.assertEqual(a[10, ...].lshape, (5, 3))

    #     a = ht.zeros((13, 5, 8), split=2)
    #     # # set value on one node
    #     a[10, 0, 0] = 1
    #     self.assertEqual(a[10, 0, 0], 1)
    #     self.assertEqual(a[10, 0, 0].dtype, ht.float32)

    #     # # slice in 1st dim only on 1 node
    #     a = ht.zeros((13, 5, 7), split=2)
    #     a[1:4] = 1
    #     self.assertTrue((a[1:4] == 1).all())
    #     self.assertEqual(a[1:4].gshape, (3, 5, 7))
    #     self.assertEqual(a[1:4].split, 2)
    #     self.assertEqual(a[1:4].dtype, ht.float32)
    #     if a.comm.size == 2:
    #         if a.comm.rank == 0:
    #             self.assertEqual(a[1:4].lshape, (3, 5, 4))
    #         if a.comm.rank == 1:
    #             self.assertEqual(a[1:4].lshape, (3, 5, 3))

    #     # slice in 1st dim only on 1 node w/ singular second dim
    #     a = ht.zeros((13, 5, 7), split=2)
    #     a[1:4, 1, :] = 1
    #     self.assertTrue((a[1:4, 1, :] == 1).all())
    #     self.assertEqual(a[1:4, 1, :].gshape, (3, 7))
    #     if a.comm.size == 2:
    #         self.assertEqual(a[1:4, 1, :].split, 1)
    #         self.assertEqual(a[1:4, 1, :].dtype, ht.float32)
    #         if a.comm.rank == 0:
    #             self.assertEqual(a[1:4, 1, :].lshape, (3, 4))
    #         if a.comm.rank == 1:
    #             self.assertEqual(a[1:4, 1, :].lshape, (3, 3))

    #     # slice in both directions
    #     a = ht.zeros((13, 5, 7), split=2)
    #     a[3:13, 2:5:2, 1:7:3] = 1
    #     self.assertTrue((a[3:13, 2:5:2, 1:7:3] == 1).all())
    #     self.assertEqual(a[3:13, 2:5:2, 1:7:3].split, 2)
    #     self.assertEqual(a[3:13, 2:5:2, 1:7:3].dtype, ht.float32)
    #     self.assertEqual(a[3:13, 2:5:2, 1:7:3].gshape, (10, 2, 2))
    #     if a.comm.size == 2:
    #         out = ht.ones((4, 5, 5), split=1)
    #         self.assertEqual(out[0].gshape, (5, 5))
    #         if a.comm.rank == 1:
    #             self.assertEqual(a[3:13, 2:5:2, 1:7:3].lshape, (10, 2, 1))
    #             self.assertEqual(out[0].lshape, (2, 5))
    #         if a.comm.rank == 0:
    #             self.assertEqual(a[3:13, 2:5:2, 1:7:3].lshape, (10, 2, 1))
    #             self.assertEqual(out[0].lshape, (3, 5))

    #     a = ht.ones((4, 5), split=0).tril()
    #     a[0] = [6, 6, 6, 6, 6]
    #     self.assertTrue((a[0] == 6).all())

    #     a = ht.ones((4, 5), split=0).tril()
    #     a[0] = (6, 6, 6, 6, 6)
    #     self.assertTrue((a[0] == 6).all())

    #     a = ht.ones((4, 5), split=0).tril()
    #     a[0] = np.array([6, 6, 6, 6, 6])
    #     self.assertTrue((a[0] == 6).all())

    #     a = ht.ones((4, 5), split=0).tril()
    #     a[0] = ht.array([6, 6, 6, 6, 6])
    #     self.assertTrue((a[ht.array((0,))] == 6).all())

    #     a = ht.ones((4, 5), split=0).tril()
    #     a[0] = ht.array([6, 6, 6, 6, 6])
    #     self.assertTrue((a[ht.array((0,))] == 6).all())

    #     # ======================= indexing with bools =================================
    #     split = None
    #     arr = ht.random.random((20, 20)).resplit(split)
    #     np_arr = arr.numpy()
    #     np_key = np_arr < 0.5
    #     ht_key = ht.array(np_key, split=split)
    #     arr[ht_key] = 10.0
    #     np_arr[np_key] = 10.0
    #     self.assertTrue(np.all(arr.numpy() == np_arr))
    #     self.assertTrue(ht.all(arr[ht_key] == 10.0))

    #     split = 0
    #     arr = ht.random.random((20, 20)).resplit(split)
    #     np_arr = arr.numpy()
    #     np_key = (np_arr < 0.5)[0]
    #     ht_key = ht.array(np_key, split=split)
    #     arr[ht_key] = 10.0
    #     np_arr[np_key] = 10.0
    #     self.assertTrue(np.all(arr.numpy() == np_arr))
    #     self.assertTrue(ht.all(arr[ht_key] == 10.0))

    #     # key -> tuple(ht.bool, int)
    #     split = 0
    #     arr = ht.random.random((20, 20)).resplit(split)
    #     np_arr = arr.numpy()
    #     np_key = (np_arr < 0.5)[0]
    #     ht_key = ht.array(np_key, split=split)
    #     arr[ht_key, 4] = 10.0
    #     np_arr[np_key, 4] = 10.0
    #     self.assertTrue(np.all(arr.numpy() == np_arr))
    #     self.assertTrue(ht.all(arr[ht_key, 4] == 10.0))

    #     # key -> tuple(torch.bool, int)
    #     split = 0
    #     arr = ht.random.random((20, 20)).resplit(split)
    #     np_arr = arr.numpy()
    #     np_key = (np_arr < 0.5)[0]
    #     t_key = torch.tensor(np_key, device=arr.larray.device)
    #     arr[t_key, 4] = 10.0
    #     np_arr[np_key, 4] = 10.0
    #     self.assertTrue(np.all(arr.numpy() == np_arr))
    #     self.assertTrue(ht.all(arr[t_key, 4] == 10.0))

    #     # key -> torch.bool
    #     split = 0
    #     arr = ht.random.random((20, 20)).resplit(split)
    #     np_arr = arr.numpy()
    #     np_key = (np_arr < 0.5)[0]
    #     t_key = torch.tensor(np_key, device=arr.larray.device)
    #     arr[t_key] = 10.0
    #     np_arr[np_key] = 10.0
    #     self.assertTrue(np.all(arr.numpy() == np_arr))
    #     self.assertTrue(ht.all(arr[t_key] == 10.0))

    #     split = 1
    #     arr = ht.random.random((20, 20, 10)).resplit(split)
    #     np_arr = arr.numpy()
    #     np_key = np_arr < 0.5
    #     ht_key = ht.array(np_key, split=split)
    #     arr[ht_key] = 10.0
    #     np_arr[np_key] = 10.0
    #     self.assertTrue(np.all(arr.numpy() == np_arr))
    #     self.assertTrue(ht.all(arr[ht_key] == 10.0))

    #     split = 2
    #     arr = ht.random.random((15, 20, 20)).resplit(split)
    #     np_arr = arr.numpy()
    #     np_key = np_arr < 0.5
    #     ht_key = ht.array(np_key, split=split)
    #     arr[ht_key] = 10.0
    #     np_arr[np_key] = 10.0
    #     self.assertTrue(np.all(arr.numpy() == np_arr))
    #     self.assertTrue(ht.all(arr[ht_key] == 10.0))

    #     with self.assertRaises(ValueError):
    #         a[..., ...]
    #     with self.assertRaises(ValueError):
    #         a[..., ...] = 1
    #     if a.comm.size > 1:
    #         with self.assertRaises(ValueError):
    #             x = ht.ones((10, 10), split=0)
    #             setting = ht.zeros((8, 8), split=1)
    #             x[1:-1, 1:-1] = setting

    #     for split in [None, 0, 1, 2]:
    #         for new_dim in [0, 1, 2]:
    #             for add in [np.newaxis, None]:
    #                 arr = ht.ones((4, 3, 2), split=split, dtype=ht.int32)
    #                 check = torch.ones((4, 3, 2), dtype=torch.int32)
    #                 idx = [slice(None), slice(None), slice(None)]
    #                 idx[new_dim] = add
    #                 idx = tuple(idx)
    #                 arr = arr[idx]
    #                 check = check[idx]
    #                 self.assertTrue(arr.shape == check.shape)
    #                 self.assertTrue(arr.lshape[new_dim] == 1)

    def test_size_gnumel(self):
        a = ht.zeros((10, 10, 10), split=None)
        self.assertEqual(a.size, 10 * 10 * 10)
        self.assertEqual(a.gnumel, 10 * 10 * 10)

        a = ht.zeros((10, 10, 10), split=0)
        self.assertEqual(a.size, 10 * 10 * 10)
        self.assertEqual(a.gnumel, 10 * 10 * 10)

        a = ht.zeros((10, 10, 10), split=1)
        self.assertEqual(a.size, 10 * 10 * 10)
        self.assertEqual(a.gnumel, 10 * 10 * 10)

        a = ht.zeros((10, 10, 10), split=2)
        self.assertEqual(a.size, 10 * 10 * 10)
        self.assertEqual(a.gnumel, 10 * 10 * 10)

        self.assertEqual(ht.array(0).size, 1)

    def test_stride_and_strides(self):
        # Local, int16, row-major memory layout
        torch_int16 = torch.arange(
            6 * 5 * 3 * 4 * 5 * 7, dtype=torch.int16, device=self.device.torch_device
        ).reshape(6, 5, 3, 4, 5, 7)
        heat_int16 = ht.array(torch_int16)
        numpy_int16 = torch_int16.cpu().numpy()
        self.assertEqual(heat_int16.stride(), torch_int16.stride())
        if pytorch_major_version >= 2:
            self.assertTrue(
                (np.asarray(heat_int16.strides) * 2 == np.asarray(numpy_int16.strides)).all()
            )
        else:
            self.assertEqual(heat_int16.strides, numpy_int16.strides)

        # Local, float32, row-major memory layout
        torch_float32 = torch.arange(
            6 * 5 * 3 * 4 * 5 * 7, dtype=torch.float32, device=self.device.torch_device
        ).reshape(6, 5, 3, 4, 5, 7)
        heat_float32 = ht.array(torch_float32)
        numpy_float32 = torch_float32.cpu().numpy()
        self.assertEqual(heat_float32.stride(), torch_float32.stride())
        if pytorch_major_version >= 2:
            self.assertTrue(
                (np.asarray(heat_float32.strides) * 4 == np.asarray(numpy_float32.strides)).all()
            )
        else:
            self.assertEqual(heat_float32.strides, numpy_float32.strides)

        # Local, float64, column-major memory layout
        torch_float64 = torch.arange(
            6 * 5 * 3 * 4 * 5 * 7, dtype=torch.float64, device=self.device.torch_device
        ).reshape(6, 5, 3, 4, 5, 7)
        heat_float64_F = ht.array(torch_float64, order="F")
        numpy_float64_F = np.array(torch_float64.cpu().numpy(), order="F")
        self.assertNotEqual(heat_float64_F.stride(), torch_float64.stride())
        if pytorch_major_version >= 2:
            self.assertTrue(
                (
                    np.asarray(heat_float64_F.strides) * 8 == np.asarray(numpy_float64_F.strides)
                ).all()
            )
        else:
            self.assertEqual(heat_float64_F.strides, numpy_float64_F.strides)

        # Distributed, int16, row-major memory layout
        size = ht.communication.MPI_WORLD.size
        split = 2
        torch_int16 = torch.arange(
            6 * 5 * 3 * size * 4 * 5 * 7, dtype=torch.int16, device=self.device.torch_device
        ).reshape(6, 5, 3 * size, 4, 5, 7)
        heat_int16_split = ht.array(torch_int16, split=split)
        numpy_int16 = torch_int16.cpu().numpy()
        if size > 1:
            self.assertNotEqual(heat_int16_split.stride(), torch_int16.stride())
        numpy_int16_split_strides = (
            tuple(np.array(numpy_int16.strides[:split]) / size) + numpy_int16.strides[split:]
        )
        if pytorch_major_version >= 2:
            self.assertTrue(
                (
                    np.asarray(heat_int16_split.strides) * 2
                    == np.asarray(numpy_int16_split_strides)
                ).all()
            )
        else:
            self.assertEqual(heat_int16_split.strides, numpy_int16_split_strides)

        # Distributed, float32, row-major memory layout
        split = -1
        torch_float32 = torch.arange(
            6 * 5 * 3 * 4 * 5 * 7 * size, dtype=torch.float32, device=self.device.torch_device
        ).reshape(6, 5, 3, 4, 5, 7 * size)
        heat_float32_split = ht.array(torch_float32, split=split)
        numpy_float32 = torch_float32.cpu().numpy()
        numpy_float32_split_strides = (
            tuple(np.array(numpy_float32.strides[:split]) / size) + numpy_float32.strides[split:]
        )
        if pytorch_major_version >= 2:
            self.assertTrue(
                (
                    np.asarray(heat_float32_split.strides) * 4
                    == np.asarray(numpy_float32_split_strides)
                ).all()
            )
        else:
            self.assertEqual(heat_float32_split.strides, numpy_float32_split_strides)

        # Distributed, float64, column-major memory layout
        split = -2
        torch_float64 = torch.arange(
            6 * 5 * 3 * 4 * 5 * size * 7, dtype=torch.float64, device=self.device.torch_device
        ).reshape(6, 5, 3, 4, 5 * size, 7)
        heat_float64_F_split = ht.array(torch_float64, order="F", split=split)
        numpy_float64_F = np.array(torch_float64.cpu().numpy(), order="F")
        numpy_float64_F_split_strides = numpy_float64_F.strides[: split + 1] + tuple(
            np.array(numpy_float64_F.strides[split + 1 :]) / size
        )
        if pytorch_major_version >= 2:
            self.assertTrue(
                (
                    np.asarray(heat_float64_F_split.strides) * 8
                    == np.asarray(numpy_float64_F_split_strides)
                ).all()
            )
        else:
            self.assertEqual(heat_float64_F_split.strides, numpy_float64_F_split_strides)

    def test_tolist(self):
        a = ht.zeros([ht.MPI_WORLD.size, ht.MPI_WORLD.size, ht.MPI_WORLD.size], dtype=ht.int32)
        res = [
            [[0 for z in range(ht.MPI_WORLD.size)] for y in range(ht.MPI_WORLD.size)]
            for x in range(ht.MPI_WORLD.size)
        ]
        self.assertListEqual(a.tolist(), res)

        a = ht.zeros(
            [ht.MPI_WORLD.size, ht.MPI_WORLD.size, ht.MPI_WORLD.size], dtype=ht.int32, split=0
        )
        res = [
            [[0 for z in range(ht.MPI_WORLD.size)] for y in range(ht.MPI_WORLD.size)]
            for x in range(ht.MPI_WORLD.size)
        ]
        self.assertListEqual(a.tolist(), res)

        a = ht.zeros(
            [ht.MPI_WORLD.size, ht.MPI_WORLD.size, ht.MPI_WORLD.size], dtype=ht.float32, split=1
        )
        res = [
            [[0.0 for z in range(ht.MPI_WORLD.size)] for y in [ht.MPI_WORLD.rank]]
            for x in range(ht.MPI_WORLD.size)
        ]
        self.assertListEqual(a.tolist(keepsplit=True), res)

        a = ht.zeros(
            [ht.MPI_WORLD.size, ht.MPI_WORLD.size, ht.MPI_WORLD.size], dtype=ht.bool, split=2
        )
        res = [
            [[False for z in [ht.MPI_WORLD.rank]] for y in range(ht.MPI_WORLD.size)]
            for x in range(ht.MPI_WORLD.size)
        ]
        self.assertListEqual(a.tolist(keepsplit=True), res)

    def test_torch_proxy(self):
        scalar_array = ht.array(1)
        scalar_proxy = scalar_array.__torch_proxy__()
        self.assertTrue(scalar_proxy.ndim == 0)
        if pytorch_major_version >= 2:
            scalar_proxy_nbytes = (
                scalar_proxy.untyped_storage().size()
                * scalar_proxy.untyped_storage().element_size()
            )
        else:
            scalar_proxy_nbytes = (
                scalar_proxy.storage().size() * scalar_proxy.storage().element_size()
            )
        self.assertTrue(scalar_proxy_nbytes == 1)

        dndarray = ht.zeros((4, 7, 6), split=1)
        dndarray_proxy = dndarray.__torch_proxy__()
        self.assertTrue(dndarray_proxy.ndim == dndarray.ndim)
        self.assertTrue(tuple(dndarray_proxy.shape) == dndarray.gshape)
        if pytorch_major_version >= 2:
            dndarray_proxy_nbytes = (
                dndarray_proxy.untyped_storage().size()
                * dndarray_proxy.untyped_storage().element_size()
            )
        else:
            dndarray_proxy_nbytes = (
                dndarray_proxy.storage().size() * dndarray_proxy.storage().element_size()
            )
        self.assertTrue(dndarray_proxy_nbytes == 1)
        self.assertTrue(dndarray_proxy.names.index("split") == 1)

    def test_xor(self):
        int16_tensor = ht.array([[1, 1], [2, 2]], dtype=ht.int16)
        int16_vector = ht.array([[3, 4]], dtype=ht.int16)

        self.assertTrue(
            ht.equal(int16_tensor ^ int16_vector, ht.bitwise_xor(int16_tensor, int16_vector))
        )
