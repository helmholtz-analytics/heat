import numpy as np
import torch
import unittest
import os
import heat as ht

if os.environ.get("DEVICE") == "gpu" and torch.cuda.is_available():
    ht.use_device("gpu")
    torch.cuda.set_device(torch.device(ht.get_device().torch_device))
else:
    ht.use_device("cpu")
device = ht.get_device().torch_device
ht_device = None
if os.environ.get("DEVICE") == "lgpu" and torch.cuda.is_available():
    device = ht.gpu.torch_device
    ht_device = ht.gpu
    torch.cuda.set_device(device)


class TestDNDarray(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        N = ht.MPI_WORLD.size
        cls.reference_tensor = ht.zeros((N, N + 1, 2 * N), device=ht_device)

        for n in range(N):
            for m in range(N + 1):
                cls.reference_tensor[n, m, :] = ht.arange(0, 2 * N) + m * 10 + n * 100

    def test_and(self):
        int16_tensor = ht.array([[1, 1], [2, 2]], dtype=ht.int16, device=ht_device)
        int16_vector = ht.array([[3, 4]], dtype=ht.int16, device=ht_device)

        self.assertTrue(
            ht.equal(int16_tensor & int16_vector, ht.bitwise_and(int16_tensor, int16_vector))
        )

    def test_gethalo(self):
        data_np = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
        data = ht.array(data_np, split=1)

        if data.comm.size == 2:

            halo_next = torch.tensor(np.array([[4, 5], [10, 11]]))
            halo_prev = torch.tensor(np.array([[2, 3], [8, 9]]))

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

            halo_next = torch.tensor(np.array([[4.0, 5.0], [10.0, 11.0]]))
            halo_prev = torch.tensor(np.array([[2.0, 3.0], [8.0, 9.0]]))

            data.get_halo(2)

            if data.comm.rank == 0:
                self.assertTrue(np.isclose(((data.halo_next - halo_next) ** 2).mean().item(), 0.0))
                self.assertEqual(data.halo_prev, None)
            if data.comm.rank == 1:
                self.assertTrue(np.isclose(((data.halo_prev - halo_prev) ** 2).mean().item(), 0.0))
                self.assertEqual(data.halo_next, None)

            data = ht.ones((10, 2), split=0)

            halo_next = torch.tensor(np.array([[1.0, 1.0], [1.0, 1.0]]))
            halo_prev = torch.tensor(np.array([[1.0, 1.0], [1.0, 1.0]]))

            data.get_halo(2)

            if data.comm.rank == 0:
                self.assertTrue(np.isclose(((data.halo_next - halo_next) ** 2).mean().item(), 0.0))
                self.assertEqual(data.halo_prev, None)
            if data.comm.rank == 1:
                self.assertTrue(np.isclose(((data.halo_prev - halo_prev) ** 2).mean().item(), 0.0))
                self.assertEqual(data.halo_next, None)

        if data.comm.size == 3:

            halo_1 = torch.tensor(np.array([[2], [8]]))
            halo_2 = torch.tensor(np.array([[3], [9]]))
            halo_3 = torch.tensor(np.array([[4], [10]]))
            halo_4 = torch.tensor(np.array([[5], [11]]))

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

    def test_astype(self):
        data = ht.float32([[1, 2, 3], [4, 5, 6]], device=ht_device)

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

    def test_balance_and_lshape_map(self):
        data = ht.zeros((70, 20), split=0, device=ht_device)
        data = data[:50]
        lshape_map = data.create_lshape_map()
        self.assertEqual(sum(lshape_map[..., 0]), 50)
        if sum(data.lshape) == 0:
            self.assertTrue(all(lshape_map[data.comm.rank] == 0))
        data.balance_()
        self.assertTrue(data.is_balanced())

        data = ht.zeros((4, 120), split=1, device=ht_device)
        data = data[:, 40:70]
        lshape_map = data.create_lshape_map()
        self.assertEqual(sum(lshape_map[..., 1]), 30)
        if sum(data.lshape) == 0:
            self.assertTrue(all(lshape_map[data.comm.rank] == 0))
        data.balance_()
        self.assertTrue(data.is_balanced())

        data = ht.zeros((70, 20), split=0, dtype=ht.float64, device=ht_device)
        data = data[:50]
        data.balance_()
        self.assertTrue(data.is_balanced())

        data = ht.zeros((4, 120), split=1, dtype=ht.int64, device=ht_device)
        data = data[:, 40:70]
        data.balance_()
        self.assertTrue(data.is_balanced())

        data = np.loadtxt("heat/datasets/data/iris.csv", delimiter=";")
        htdata = ht.load("heat/datasets/data/iris.csv", sep=";", split=0, device=ht_device)
        self.assertTrue(ht.equal(htdata, ht.array(data, split=0, dtype=ht.float, device=ht_device)))

        if ht.MPI_WORLD.size > 4:
            rank = ht.MPI_WORLD.rank
            if rank == 2:
                arr = torch.tensor([0, 1])
            elif rank == 3:
                arr = torch.tensor([2, 3, 4, 5])
            elif rank == 4:
                arr = torch.tensor([6, 7, 8, 9])
            else:
                arr = torch.empty([0], dtype=torch.int64)
            a = ht.array(arr, is_split=0, device=ht_device)
            a.balance_()
            comp = ht.arange(10, split=0, device=ht_device)

            self.assertTrue(ht.equal(a, comp))

    def test_bool_cast(self):
        # simple scalar tensor
        a = ht.ones(1, device=ht_device)
        casted_a = bool(a)
        self.assertEqual(casted_a, True)
        self.assertIsInstance(casted_a, bool)

        # multi-dimensional scalar tensor
        b = ht.zeros((1, 1, 1, 1), device=ht_device)
        casted_b = bool(b)
        self.assertEqual(casted_b, False)
        self.assertIsInstance(casted_b, bool)

        # split scalar tensor
        c = ht.full((1,), 5, split=0, device=ht_device)
        casted_c = bool(c)
        self.assertEqual(casted_c, True)
        self.assertIsInstance(casted_c, bool)

        # exception on non-scalar tensor
        with self.assertRaises(TypeError):
            bool(ht.empty(1, 2, 1, 1, device=ht_device))
        # exception on empty tensor
        with self.assertRaises(TypeError):
            bool(ht.empty((0, 1, 2), device=ht_device))
        # exception on split tensor, where each chunk has size 1
        if ht.MPI_WORLD.size > 1:
            with self.assertRaises(TypeError):
                bool(ht.full((ht.MPI_WORLD.size,), 2, split=0, device=ht_device))

    def test_complex_cast(self):
        # simple scalar tensor
        a = ht.ones(1, device=ht_device)
        casted_a = complex(a)
        self.assertEqual(casted_a, 1 + 0j)
        self.assertIsInstance(casted_a, complex)

        # multi-dimensional scalar tensor
        b = ht.zeros((1, 1, 1, 1), device=ht_device)
        casted_b = complex(b)
        self.assertEqual(casted_b, 0 + 0j)
        self.assertIsInstance(casted_b, complex)

        # split scalar tensor
        c = ht.full((1,), 5, split=0, device=ht_device)
        casted_c = complex(c)
        self.assertEqual(casted_c, 5 + 0j)
        self.assertIsInstance(casted_c, complex)

        # exception on non-scalar tensor
        with self.assertRaises(TypeError):
            complex(ht.empty(1, 2, 1, 1, device=ht_device))
        # exception on empty tensor
        with self.assertRaises(TypeError):
            complex(ht.empty((0, 1, 2), device=ht_device))
        # exception on split tensor, where each chunk has size 1
        if ht.MPI_WORLD.size > 1:
            with self.assertRaises(TypeError):
                complex(ht.full((ht.MPI_WORLD.size,), 2, split=0, device=ht_device))

    def test_fill_diagonal(self):
        ref = ht.zeros(
            (ht.MPI_WORLD.size * 2, ht.MPI_WORLD.size * 2),
            dtype=ht.float32,
            split=0,
            device=ht_device,
        )
        a = ht.eye(ht.MPI_WORLD.size * 2, dtype=ht.float32, split=0, device=ht_device)
        a.fill_diagonal(0)
        self.assertTrue(ht.equal(a, ref))

        ref = ht.zeros(
            (ht.MPI_WORLD.size * 2, ht.MPI_WORLD.size * 2),
            dtype=ht.int32,
            split=0,
            device=ht_device,
        )
        a = ht.eye(ht.MPI_WORLD.size * 2, dtype=ht.int32, split=0, device=ht_device)
        a.fill_diagonal(0)
        self.assertTrue(ht.equal(a, ref))

        ref = ht.zeros(
            (ht.MPI_WORLD.size * 2, ht.MPI_WORLD.size * 2),
            dtype=ht.float32,
            split=1,
            device=ht_device,
        )
        a = ht.eye(ht.MPI_WORLD.size * 2, dtype=ht.float32, split=1, device=ht_device)
        a.fill_diagonal(0)
        self.assertTrue(ht.equal(a, ref))

        ref = ht.zeros(
            (ht.MPI_WORLD.size * 2, ht.MPI_WORLD.size * 3),
            dtype=ht.float32,
            split=0,
            device=ht_device,
        )
        a = ht.eye(
            (ht.MPI_WORLD.size * 2, ht.MPI_WORLD.size * 3),
            dtype=ht.float32,
            split=0,
            device=ht_device,
        )
        a.fill_diagonal(0)
        self.assertTrue(ht.equal(a, ref))

        # ToDo: uneven tensor dimensions x and y when bug in factories.eye is fixed
        ref = ht.zeros(
            (ht.MPI_WORLD.size * 3, ht.MPI_WORLD.size * 3),
            dtype=ht.float32,
            split=1,
            device=ht_device,
        )
        a = ht.eye(
            (ht.MPI_WORLD.size * 3, ht.MPI_WORLD.size * 3),
            dtype=ht.float32,
            split=1,
            device=ht_device,
        )
        a.fill_diagonal(0)
        self.assertTrue(ht.equal(a, ref))

        # ToDo: uneven tensor dimensions x and y when bug in factories.eye is fixed
        ref = ht.zeros(
            (ht.MPI_WORLD.size * 4, ht.MPI_WORLD.size * 4),
            dtype=ht.float32,
            split=0,
            device=ht_device,
        )
        a = ht.eye(
            (ht.MPI_WORLD.size * 4, ht.MPI_WORLD.size * 4),
            dtype=ht.float32,
            split=0,
            device=ht_device,
        )
        a.fill_diagonal(0)
        self.assertTrue(ht.equal(a, ref))

        a = ht.ones((ht.MPI_WORLD.size * 2,), dtype=ht.float32, split=0, device=ht_device)
        with self.assertRaises(ValueError):
            a.fill_diagonal(0)

    def test_float_cast(self):
        # simple scalar tensor
        a = ht.ones(1, device=ht_device)
        casted_a = float(a)
        self.assertEqual(casted_a, 1.0)
        self.assertIsInstance(casted_a, float)

        # multi-dimensional scalar tensor
        b = ht.zeros((1, 1, 1, 1), device=ht_device)
        casted_b = float(b)
        self.assertEqual(casted_b, 0.0)
        self.assertIsInstance(casted_b, float)

        # split scalar tensor
        c = ht.full((1,), 5, split=0, device=ht_device)
        casted_c = float(c)
        self.assertEqual(casted_c, 5.0)
        self.assertIsInstance(casted_c, float)

        # exception on non-scalar tensor
        with self.assertRaises(TypeError):
            float(ht.empty(1, 2, 1, 1, device=ht_device))
        # exception on empty tensor
        with self.assertRaises(TypeError):
            float(ht.empty((0, 1, 2), device=ht_device))
        # exception on split tensor, where each chunk has size 1
        if ht.MPI_WORLD.size > 1:
            with self.assertRaises(TypeError):
                float(ht.full((ht.MPI_WORLD.size,), 2, split=0), device=ht_device)

    def test_int_cast(self):
        # simple scalar tensor
        a = ht.ones(1, device=ht_device)
        casted_a = int(a)
        self.assertEqual(casted_a, 1)
        self.assertIsInstance(casted_a, int)

        # multi-dimensional scalar tensor
        b = ht.zeros((1, 1, 1, 1), device=ht_device)
        casted_b = int(b)
        self.assertEqual(casted_b, 0)
        self.assertIsInstance(casted_b, int)

        # split scalar tensor
        c = ht.full((1,), 5, split=0, device=ht_device)
        casted_c = int(c)
        self.assertEqual(casted_c, 5)
        self.assertIsInstance(casted_c, int)

        # exception on non-scalar tensor
        with self.assertRaises(TypeError):
            int(ht.empty(1, 2, 1, 1, device=ht_device))
        # exception on empty tensor
        with self.assertRaises(TypeError):
            int(ht.empty((0, 1, 2), device=ht_device))
        # exception on split tensor, where each chunk has size 1
        if ht.MPI_WORLD.size > 1:
            with self.assertRaises(TypeError):
                int(ht.full((ht.MPI_WORLD.size,), 2, split=0, device=ht_device))

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
        data = ht.zeros((70, 20), split=0, device=ht_device)
        if data.comm.size != 1:
            data = data[:50]
            self.assertFalse(data.is_balanced())
            data.balance_()
            self.assertTrue(data.is_balanced())

    def test_is_distributed(self):
        data = ht.zeros((5, 5), device=ht_device)
        self.assertFalse(data.is_distributed())

        data = ht.zeros((4, 4), split=0, device=ht_device)
        self.assertTrue(data.comm.size > 1 and data.is_distributed() or not data.is_distributed())

    def test_item(self):
        x = ht.zeros((1,), device=ht_device)
        self.assertEqual(x.item(), 0)
        self.assertEqual(type(x.item()), float)

        x = ht.zeros((1, 2), device=ht_device)
        with self.assertRaises(ValueError):
            x.item()

    def test_len(self):
        # vector
        a = ht.zeros((10,), device=ht_device)
        a_length = len(a)

        self.assertIsInstance(a_length, int)
        self.assertEqual(a_length, 10)

        # matrix
        b = ht.ones((50, 2), device=ht_device)
        b_length = len(b)

        self.assertIsInstance(b_length, int)
        self.assertEqual(b_length, 50)

        # split 5D array
        c = ht.empty((3, 4, 5, 6, 7), split=-1, device=ht_device)
        c_length = len(c)

        self.assertIsInstance(c_length, int)
        self.assertEqual(c_length, 3)

    def test_lloc(self):
        # single set
        a = ht.zeros((13, 5), split=0, device=ht_device)
        a.lloc[0, 0] = 1
        self.assertEqual(a._DNDarray__array[0, 0], 1)
        self.assertEqual(a.lloc[0, 0].dtype, torch.float32)

        # multiple set
        a = ht.zeros((13, 5), split=0, device=ht_device)
        a.lloc[1:3, 1] = 1
        self.assertTrue(all(a._DNDarray__array[1:3, 1] == 1))
        self.assertEqual(a.lloc[1:3, 1].dtype, torch.float32)

        # multiple set with specific indexing
        a = ht.zeros((13, 5), split=0, device=ht_device)
        a.lloc[3:7:2, 2:5:2] = 1
        self.assertTrue(torch.all(a._DNDarray__array[3:7:2, 2:5:2] == 1))
        self.assertEqual(a.lloc[3:7:2, 2:5:2].dtype, torch.float32)

    def test_lshift(self):
        int_tensor = ht.array([[0, 1], [2, 3]])
        int_result = ht.array([[0, 4], [8, 12]])

        self.assertTrue(ht.equal(int_tensor << 2, int_result))

        with self.assertRaises(TypeError):
            int_tensor << 2.4
        with self.assertRaises(TypeError):
            ht.array([True]) << 2

    def test_numpy(self):
        # ToDo: numpy does not work for distributed tensors du to issue#
        # Add additional tests if the issue is solved
        a = np.random.randn(10, 8)
        b = ht.array(a, device=ht_device)
        self.assertIsInstance(b.numpy(), np.ndarray)
        self.assertEqual(b.numpy().shape, a.shape)
        self.assertEqual(b.numpy().tolist(), b._DNDarray__array.cpu().numpy().tolist())

        a = ht.ones((10, 8), dtype=ht.float32, device=ht_device)
        b = np.ones((2, 2)).astype("float32")
        self.assertEqual(a.numpy().dtype, b.dtype)

        a = ht.ones((10, 8), dtype=ht.float64, device=ht_device)
        b = np.ones((2, 2)).astype("float64")
        self.assertEqual(a.numpy().dtype, b.dtype)

        a = ht.ones((10, 8), dtype=ht.int32, device=ht_device)
        b = np.ones((2, 2)).astype("int32")
        self.assertEqual(a.numpy().dtype, b.dtype)

        a = ht.ones((10, 8), dtype=ht.int64, device=ht_device)
        b = np.ones((2, 2)).astype("int64")
        self.assertEqual(a.numpy().dtype, b.dtype)

    def test_or(self):
        int16_tensor = ht.array([[1, 1], [2, 2]], dtype=ht.int16, device=ht_device)
        int16_vector = ht.array([[3, 4]], dtype=ht.int16, device=ht_device)

        self.assertTrue(
            ht.equal(int16_tensor | int16_vector, ht.bitwise_or(int16_tensor, int16_vector))
        )

    def test_redistribute(self):
        # need to test with 1, 2, 3, and 4 dims
        st = ht.zeros((50,), split=0, device=ht_device)
        if st.comm.size >= 3:
            target_map = torch.zeros((st.comm.size, 1), dtype=torch.int, device=device)
            target_map[1] = 30
            target_map[2] = 20
            st.redistribute_(target_map=target_map)
            if st.comm.rank == 1:
                self.assertEqual(st.lshape, (30,))
            elif st.comm.rank == 2:
                self.assertEqual(st.lshape, (20,))
            else:
                self.assertEqual(st.lshape, (0,))

            st = ht.zeros((50, 50), split=1, device=ht_device)
            target_map = torch.zeros((st.comm.size, 2), dtype=torch.int, device=device)
            target_map[0, 1] = 13
            target_map[2, 1] = 50 - 13
            st.redistribute_(target_map=target_map)
            if st.comm.rank == 0:
                self.assertEqual(st.lshape, (50, 13))
            elif st.comm.rank == 2:
                self.assertEqual(st.lshape, (50, 50 - 13))
            else:
                self.assertEqual(st.lshape, (50, 0))

            st = ht.zeros((50, 81, 67), split=2, device=ht_device)
            target_map = torch.zeros((st.comm.size, 3), dtype=torch.int, device=device)
            target_map[0, 2] = 67
            st.redistribute_(target_map=target_map)
            if st.comm.rank == 0:
                self.assertEqual(st.lshape, (50, 81, 67))
            else:
                self.assertEqual(st.lshape, (50, 81, 0))

            st = ht.zeros((8, 8, 8), split=None, device=ht_device)
            target_map = torch.zeros((st.comm.size, 3), dtype=torch.int, device=device)
            # this will do nothing!
            st.redistribute_(target_map=target_map)
            self.assertTrue(st.lshape, st.gshape)

            st = ht.zeros((50, 81, 67), split=0, device=ht_device)
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

    def test_resplit(self):
        # resplitting with same axis, should leave everything unchanged
        shape = (ht.MPI_WORLD.size, ht.MPI_WORLD.size)
        data = ht.zeros(shape, split=None, device=ht_device)
        data.resplit_(None)

        self.assertIsInstance(data, ht.DNDarray)
        self.assertEqual(data.shape, shape)
        self.assertEqual(data.lshape, shape)
        self.assertEqual(data.split, None)

        # resplitting with same axis, should leave everything unchanged
        shape = (ht.MPI_WORLD.size, ht.MPI_WORLD.size)
        data = ht.zeros(shape, split=1, device=ht_device)
        data.resplit_(1)

        self.assertIsInstance(data, ht.DNDarray)
        self.assertEqual(data.shape, shape)
        self.assertEqual(data.lshape, (data.comm.size, 1))
        self.assertEqual(data.split, 1)

        # splitting an unsplit tensor should result in slicing the tensor locally
        shape = (ht.MPI_WORLD.size, ht.MPI_WORLD.size)
        data = ht.zeros(shape, device=ht_device)
        data.resplit_(-1)

        self.assertIsInstance(data, ht.DNDarray)
        self.assertEqual(data.shape, shape)
        self.assertEqual(data.lshape, (data.comm.size, 1))
        self.assertEqual(data.split, 1)

        # unsplitting, aka gathering a tensor
        shape = (ht.MPI_WORLD.size + 1, ht.MPI_WORLD.size)
        data = ht.ones(shape, split=0, device=ht_device)
        data.resplit_(None)

        self.assertIsInstance(data, ht.DNDarray)
        self.assertEqual(data.shape, shape)
        self.assertEqual(data.lshape, shape)
        self.assertEqual(data.split, None)

        # assign and entirely new split axis
        shape = (ht.MPI_WORLD.size + 2, ht.MPI_WORLD.size + 1)
        data = ht.ones(shape, split=0, device=ht_device)
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
        self.assertTrue((a_tensor._DNDarray__array == local_tensor._DNDarray__array).all())

        # unsplit
        a_tensor.resplit_(axis=None)
        self.assertTrue((a_tensor._DNDarray__array == self.reference_tensor._DNDarray__array).all())

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
        self.assertTrue((a_tensor._DNDarray__array == local_tensor._DNDarray__array).all())

        # unsplit
        a_tensor.resplit_(axis=None)
        self.assertTrue((a_tensor._DNDarray__array == self.reference_tensor._DNDarray__array).all())

        # split along axis = 2
        a_tensor.resplit_(axis=2)
        local_shape = (N, N + 1, 2)
        local_tensor = self.reference_tensor[
            :, :, 2 * ht.MPI_WORLD.rank : 2 * ht.MPI_WORLD.rank + 2
        ]

        self.assertEqual(a_tensor.lshape, local_shape)
        self.assertTrue((a_tensor._DNDarray__array == local_tensor._DNDarray__array).all())

        expected = torch.ones((ht.MPI_WORLD.size, 100), dtype=torch.int64, device=device)
        data = ht.array(expected, split=1, device=ht_device)
        data.resplit_(None)

        self.assertTrue(torch.equal(data._DNDarray__array, expected))
        self.assertFalse(data.is_distributed())
        self.assertIsNone(data.split)
        self.assertEqual(data.dtype, ht.int64)
        self.assertEqual(data._DNDarray__array.dtype, expected.dtype)

        expected = torch.zeros((100, ht.MPI_WORLD.size), dtype=torch.uint8, device=device)
        data = ht.array(expected, split=0, device=ht_device)
        data.resplit_(None)

        self.assertTrue(torch.equal(data._DNDarray__array, expected))
        self.assertFalse(data.is_distributed())
        self.assertIsNone(data.split)
        self.assertEqual(data.dtype, ht.uint8)
        self.assertEqual(data._DNDarray__array.dtype, expected.dtype)

        # "in place"
        length = torch.tensor([i + 20 for i in range(2)], device=device)
        test = torch.arange(torch.prod(length), dtype=torch.float64, device=device).reshape(
            [i + 20 for i in range(2)]
        )
        a = ht.array(test, split=1)
        a.resplit_(axis=0)
        self.assertTrue(ht.equal(a, ht.array(test, split=0)))
        self.assertEqual(a.split, 0)
        self.assertEqual(a.dtype, ht.float64)
        del a

        test = torch.arange(torch.prod(length), device=device)
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

    def test_rshift(self):
        int_tensor = ht.array([[0, 2], [4, 8]])
        int_result = ht.array([[0, 0], [1, 2]])

        self.assertTrue(ht.equal(int_tensor >> 2, int_result))

        with self.assertRaises(TypeError):
            int_tensor >> 2.4
        with self.assertRaises(TypeError):
            ht.array([True]) >> 2

    def test_setitem_getitem(self):
        # set and get single value
        a = ht.zeros((13, 5), split=0, device=ht_device)
        # set value on one node
        a[10, 0] = 1
        self.assertEqual(a[10, 0], 1)
        self.assertEqual(a[10, 0].dtype, ht.float32)

        a = ht.zeros((13, 5), split=0, device=ht_device)
        a[10] = 1
        b = a[10]
        self.assertTrue((b == 1).all())
        self.assertEqual(b.dtype, ht.float32)
        self.assertEqual(b.gshape, (5,))

        a = ht.zeros((13, 5), split=0, device=ht_device)
        a[-1] = 1
        b = a[-1]
        self.assertTrue((b == 1).all())
        self.assertEqual(b.dtype, ht.float32)
        self.assertEqual(b.gshape, (5,))

        # slice in 1st dim only on 1 node
        a = ht.zeros((13, 5), split=0, device=ht_device)
        a[1:4] = 1
        self.assertTrue((a[1:4] == 1).all())
        self.assertEqual(a[1:4].gshape, (3, 5))
        self.assertEqual(a[1:4].split, 0)
        self.assertEqual(a[1:4].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 0:
                self.assertEqual(a[1:4].lshape, (3, 5))
            else:
                self.assertEqual(a[1:4].lshape, (0,))

        a = ht.zeros((13, 5), split=0, device=ht_device)
        a[1:2] = 1
        self.assertTrue((a[1:2] == 1).all())
        self.assertEqual(a[1:2].gshape, (1, 5))
        self.assertEqual(a[1:2].split, 0)
        self.assertEqual(a[1:2].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 0:
                self.assertEqual(a[1:2].lshape, (1, 5))
            else:
                self.assertEqual(a[1:2].lshape, (0,))

        # slice in 1st dim only on 1 node w/ singular second dim
        a = ht.zeros((13, 5), split=0, device=ht_device)
        a[1:4, 1] = 1
        b = a[1:4, 1]
        self.assertTrue((b == 1).all())
        self.assertEqual(b.gshape, (3,))
        self.assertEqual(b.split, 0)
        self.assertEqual(b.dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 0:
                self.assertEqual(b.lshape, (3,))
            else:
                self.assertEqual(b.lshape, (0,))

        # slice in 1st dim across both nodes (2 node case) w/ singular second dim
        a = ht.zeros((13, 5), split=0, device=ht_device)
        a[1:11, 1] = 1
        self.assertTrue((a[1:11, 1] == 1).all())
        self.assertEqual(a[1:11, 1].gshape, (10,))
        self.assertEqual(a[1:11, 1].split, 0)
        self.assertEqual(a[1:11, 1].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 1:
                self.assertEqual(a[1:11, 1].lshape, (4,))
            if a.comm.rank == 0:
                self.assertEqual(a[1:11, 1].lshape, (6,))

        # slice in 1st dim across 1 node (2nd) w/ singular second dim
        c = ht.zeros((13, 5), split=0, device=ht_device)
        c[8:12, 1] = 1
        b = c[8:12, 1]
        self.assertTrue((b == 1).all())
        self.assertEqual(b.gshape, (4,))
        self.assertEqual(b.split, 0)
        self.assertEqual(b.dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 1:
                self.assertEqual(b.lshape, (4,))
            if a.comm.rank == 0:
                self.assertEqual(b.lshape, (0,))

        # slice in both directions
        a = ht.zeros((13, 5), split=0, device=ht_device)
        a[3:13, 2:5:2] = 1
        self.assertTrue((a[3:13, 2:5:2] == 1).all())
        self.assertEqual(a[3:13, 2:5:2].gshape, (10, 2))
        self.assertEqual(a[3:13, 2:5:2].split, 0)
        self.assertEqual(a[3:13, 2:5:2].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 1:
                self.assertEqual(a[3:13, 2:5:2].lshape, (6, 2))
            if a.comm.rank == 0:
                self.assertEqual(a[3:13, 2:5:2].lshape, (4, 2))

        # setting with heat tensor
        a = ht.zeros((4, 5), split=0, device=ht_device)
        a[1, 0:4] = ht.arange(4, device=ht_device)
        # if a.comm.size == 2:
        for c, i in enumerate(range(4)):
            self.assertEqual(a[1, c], i)

        # setting with torch tensor
        a = ht.zeros((4, 5), split=0, device=ht_device)
        a[1, 0:4] = torch.arange(4, device=device)
        # if a.comm.size == 2:
        for c, i in enumerate(range(4)):
            self.assertEqual(a[1, c], i)

        ###################################################
        a = ht.zeros((13, 5), split=1, device=ht_device)
        # # set value on one node
        a[10] = 1
        self.assertEqual(a[10].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 0:
                self.assertEqual(a[10].lshape, (3,))
            if a.comm.rank == 1:
                self.assertEqual(a[10].lshape, (2,))

        a = ht.zeros((13, 5), split=1, device=ht_device)
        # # set value on one node
        a[10, 0] = 1
        self.assertEqual(a[10, 0], 1)
        self.assertEqual(a[10, 0].dtype, ht.float32)

        # slice in 1st dim only on 1 node
        a = ht.zeros((13, 5), split=1, device=ht_device)
        a[1:4] = 1
        self.assertTrue((a[1:4] == 1).all())
        self.assertEqual(a[1:4].gshape, (3, 5))
        self.assertEqual(a[1:4].split, 1)
        self.assertEqual(a[1:4].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 0:
                self.assertEqual(a[1:4].lshape, (3, 3))
            if a.comm.rank == 1:
                self.assertEqual(a[1:4].lshape, (3, 2))

        # slice in 1st dim only on 1 node w/ singular second dim
        a = ht.zeros((13, 5), split=1, device=ht_device)
        a[1:4, 1] = 1
        self.assertTrue((a[1:4, 1] == 1).all())
        self.assertEqual(a[1:4, 1].gshape, (3,))
        self.assertEqual(a[1:4, 1].split, 0)
        self.assertEqual(a[1:4, 1].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 0:
                self.assertEqual(a[1:4, 1].lshape, (3,))
            if a.comm.rank == 1:
                self.assertEqual(a[1:4, 1].lshape, (0,))

        # slice in 2st dim across both nodes (2 node case) w/ singular fist dim
        a = ht.zeros((13, 5), split=1, device=ht_device)
        a[11, 1:5] = 1
        self.assertTrue((a[11, 1:5] == 1).all())
        self.assertEqual(a[11, 1:5].gshape, (4,))
        self.assertEqual(a[11, 1:5].split, 0)
        self.assertEqual(a[11, 1:5].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 1:
                self.assertEqual(a[11, 1:5].lshape, (2,))
            if a.comm.rank == 0:
                self.assertEqual(a[11, 1:5].lshape, (2,))

        # slice in 1st dim across 1 node (2nd) w/ singular second dim
        a = ht.zeros((13, 5), split=1, device=ht_device)
        a[8:12, 1] = 1
        self.assertTrue((a[8:12, 1] == 1).all())
        self.assertEqual(a[8:12, 1].gshape, (4,))
        self.assertEqual(a[8:12, 1].split, 0)
        self.assertEqual(a[8:12, 1].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 0:
                self.assertEqual(a[8:12, 1].lshape, (4,))
            if a.comm.rank == 1:
                self.assertEqual(a[8:12, 1].lshape, (0,))

        # slice in both directions
        a = ht.zeros((13, 5), split=1, device=ht_device)
        a[3:13, 2::2] = 1
        self.assertTrue((a[3:13, 2:5:2] == 1).all())
        self.assertEqual(a[3:13, 2:5:2].gshape, (10, 2))
        self.assertEqual(a[3:13, 2:5:2].split, 1)
        self.assertEqual(a[3:13, 2:5:2].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 1:
                self.assertEqual(a[3:13, 2:5:2].lshape, (10, 1))
            if a.comm.rank == 0:
                self.assertEqual(a[3:13, 2:5:2].lshape, (10, 1))

        # setting with heat tensor
        a = ht.zeros((4, 5), split=1, device=ht_device)
        a[1, 0:4] = ht.arange(4, device=ht_device)
        for c, i in enumerate(range(4)):
            self.assertEqual(a[1, c], i)

        # setting with torch tensor
        a = ht.zeros((4, 5), split=1, device=ht_device)
        a[1, 0:4] = torch.arange(4, device=device)
        for c, i in enumerate(range(4)):
            self.assertEqual(a[1, c], i)

        ####################################################
        a = ht.zeros((13, 5, 7), split=2, device=ht_device)
        # # set value on one node
        a[10, :, :] = 1
        self.assertEqual(a[10, :, :].dtype, ht.float32)
        self.assertEqual(a[10, :, :].gshape, (5, 7))
        if a.comm.size == 2:
            if a.comm.rank == 0:
                self.assertEqual(a[10, :, :].lshape, (5, 4))
            if a.comm.rank == 1:
                self.assertEqual(a[10, :, :].lshape, (5, 3))

        a = ht.zeros((13, 5, 8), split=2, device=ht_device)
        # # set value on one node
        a[10, 0, 0] = 1
        self.assertEqual(a[10, 0, 0], 1)
        self.assertEqual(a[10, 0, 0].dtype, ht.float32)

        # # slice in 1st dim only on 1 node
        a = ht.zeros((13, 5, 7), split=2, device=ht_device)
        a[1:4] = 1
        self.assertTrue((a[1:4] == 1).all())
        self.assertEqual(a[1:4].gshape, (3, 5, 7))
        self.assertEqual(a[1:4].split, 2)
        self.assertEqual(a[1:4].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 0:
                self.assertEqual(a[1:4].lshape, (3, 5, 4))
            if a.comm.rank == 1:
                self.assertEqual(a[1:4].lshape, (3, 5, 3))

        # slice in 1st dim only on 1 node w/ singular second dim
        a = ht.zeros((13, 5, 7), split=2, device=ht_device)
        a[1:4, 1, :] = 1
        self.assertTrue((a[1:4, 1, :] == 1).all())
        self.assertEqual(a[1:4, 1, :].gshape, (3, 7))
        if a.comm.size == 2:
            self.assertEqual(a[1:4, 1, :].split, 1)
            self.assertEqual(a[1:4, 1, :].dtype, ht.float32)
            if a.comm.rank == 0:
                self.assertEqual(a[1:4, 1, :].lshape, (3, 4))
            if a.comm.rank == 1:
                self.assertEqual(a[1:4, 1, :].lshape, (3, 3))

        # slice in both directions
        a = ht.zeros((13, 5, 7), split=2, device=ht_device)
        a[3:13, 2:5:2, 1:7:3] = 1
        self.assertTrue((a[3:13, 2:5:2, 1:7:3] == 1).all())
        self.assertEqual(a[3:13, 2:5:2, 1:7:3].split, 2)
        self.assertEqual(a[3:13, 2:5:2, 1:7:3].dtype, ht.float32)
        self.assertEqual(a[3:13, 2:5:2, 1:7:3].gshape, (10, 2, 2))
        if a.comm.size == 2:
            out = ht.ones((4, 5, 5), split=1, device=ht_device)
            self.assertEqual(out[0].gshape, (5, 5))
            if a.comm.rank == 1:
                self.assertEqual(a[3:13, 2:5:2, 1:7:3].lshape, (10, 2, 1))
                self.assertEqual(out[0].lshape, (2, 5))
            if a.comm.rank == 0:
                self.assertEqual(a[3:13, 2:5:2, 1:7:3].lshape, (10, 2, 1))
                self.assertEqual(out[0].lshape, (3, 5))

        a = ht.ones((4, 5), split=0, device=ht_device).tril()
        a[0] = [6, 6, 6, 6, 6]
        self.assertTrue((a[0] == 6).all())

        a = ht.ones((4, 5), split=0, device=ht_device).tril()
        a[0] = (6, 6, 6, 6, 6)
        self.assertTrue((a[0] == 6).all())

        a = ht.ones((4, 5), split=0, device=ht_device).tril()
        a[0] = np.array([6, 6, 6, 6, 6])
        self.assertTrue((a[0] == 6).all())

        a = ht.ones((4, 5), split=0, device=ht_device).tril()
        a[0] = ht.array([6, 6, 6, 6, 6], device=ht_device)
        self.assertTrue((a[ht.array((0,), device=ht_device)] == 6).all())

        a = ht.ones((4, 5), split=0, device=ht_device).tril()
        a[0] = ht.array([6, 6, 6, 6, 6], device=ht_device)
        self.assertTrue((a[ht.array((0,), device=ht_device)] == 6).all())

    def test_size_gnumel(self):
        a = ht.zeros((10, 10, 10), split=None, device=ht_device)
        self.assertEqual(a.size, 10 * 10 * 10)
        self.assertEqual(a.gnumel, 10 * 10 * 10)

        a = ht.zeros((10, 10, 10), split=0, device=ht_device)
        self.assertEqual(a.size, 10 * 10 * 10)
        self.assertEqual(a.gnumel, 10 * 10 * 10)

        a = ht.zeros((10, 10, 10), split=1, device=ht_device)
        self.assertEqual(a.size, 10 * 10 * 10)
        self.assertEqual(a.gnumel, 10 * 10 * 10)

        a = ht.zeros((10, 10, 10), split=2, device=ht_device)
        self.assertEqual(a.size, 10 * 10 * 10)
        self.assertEqual(a.gnumel, 10 * 10 * 10)

        self.assertEqual(ht.array(0, device=ht_device).size, 1)

    def test_stride_and_strides(self):
        # Local, int16, row-major memory layout
        torch_int16 = torch.arange(6 * 5 * 3 * 4 * 5 * 7, dtype=torch.int16, device=device).reshape(
            6, 5, 3, 4, 5, 7
        )
        heat_int16 = ht.array(torch_int16, device=ht_device)
        numpy_int16 = torch_int16.cpu().numpy()
        self.assertEqual(heat_int16.stride(), torch_int16.stride())
        self.assertEqual(heat_int16.strides, numpy_int16.strides)

        # Local, float32, row-major memory layout
        torch_float32 = torch.arange(
            6 * 5 * 3 * 4 * 5 * 7, dtype=torch.float32, device=device
        ).reshape(6, 5, 3, 4, 5, 7)
        heat_float32 = ht.array(torch_float32, device=ht_device)
        numpy_float32 = torch_float32.cpu().numpy()
        self.assertEqual(heat_float32.stride(), torch_float32.stride())
        self.assertEqual(heat_float32.strides, numpy_float32.strides)

        # Local, float64, column-major memory layout
        torch_float64 = torch.arange(
            6 * 5 * 3 * 4 * 5 * 7, dtype=torch.float64, device=device
        ).reshape(6, 5, 3, 4, 5, 7)
        heat_float64_F = ht.array(torch_float64, order="F", device=ht_device)
        numpy_float64_F = np.array(torch_float64.cpu().numpy(), order="F")
        self.assertNotEqual(heat_float64_F.stride(), torch_float64.stride())
        self.assertEqual(heat_float64_F.strides, numpy_float64_F.strides)

        # Distributed, int16, row-major memory layout
        size = ht.communication.MPI_WORLD.size
        split = 2
        torch_int16 = torch.arange(
            6 * 5 * 3 * size * 4 * 5 * 7, dtype=torch.int16, device=device
        ).reshape(6, 5, 3 * size, 4, 5, 7)
        heat_int16_split = ht.array(torch_int16, split=split, device=ht_device)
        numpy_int16 = torch_int16.cpu().numpy()
        if size > 1:
            self.assertNotEqual(heat_int16_split.stride(), torch_int16.stride())
        numpy_int16_split_strides = (
            tuple(np.array(numpy_int16.strides[:split]) / size) + numpy_int16.strides[split:]
        )
        self.assertEqual(heat_int16_split.strides, numpy_int16_split_strides)

        # Distributed, float32, row-major memory layout
        split = -1
        torch_float32 = torch.arange(
            6 * 5 * 3 * 4 * 5 * 7 * size, dtype=torch.float32, device=device
        ).reshape(6, 5, 3, 4, 5, 7 * size)
        heat_float32_split = ht.array(torch_float32, split=split, device=ht_device)
        numpy_float32 = torch_float32.cpu().numpy()
        numpy_float32_split_strides = (
            tuple(np.array(numpy_float32.strides[:split]) / size) + numpy_float32.strides[split:]
        )
        self.assertEqual(heat_float32_split.strides, numpy_float32_split_strides)

        # Distributed, float64, column-major memory layout
        split = -2
        torch_float64 = torch.arange(
            6 * 5 * 3 * 4 * 5 * size * 7, dtype=torch.float64, device=device
        ).reshape(6, 5, 3, 4, 5 * size, 7)
        heat_float64_F_split = ht.array(torch_float64, order="F", split=split, device=ht_device)
        numpy_float64_F = np.array(torch_float64.cpu().numpy(), order="F")
        numpy_float64_F_split_strides = numpy_float64_F.strides[: split + 1] + tuple(
            np.array(numpy_float64_F.strides[split + 1 :]) / size
        )
        self.assertEqual(heat_float64_F_split.strides, numpy_float64_F_split_strides)

    def test_xor(self):
        int16_tensor = ht.array([[1, 1], [2, 2]], dtype=ht.int16, device=ht_device)
        int16_vector = ht.array([[3, 4]], dtype=ht.int16, device=ht_device)

        self.assertTrue(
            ht.equal(int16_tensor ^ int16_vector, ht.bitwise_xor(int16_tensor, int16_vector))
        )
