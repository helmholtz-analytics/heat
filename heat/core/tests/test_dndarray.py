import numpy as np
import torch
import unittest
import os
import heat as ht

if os.environ.get("DEVICE") == "gpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ht.use_device("gpu" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
    ht.use_device("cpu")


class TestDNDarray(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        N = ht.MPI_WORLD.size
        cls.reference_tensor = ht.zeros((N, N + 1, 2 * N))

        for n in range(N):
            for m in range(N + 1):
                cls.reference_tensor[n, m, :] = ht.arange(0, 2 * N) + m * 10 + n * 100

    def test_astype(self):
        data = ht.float32([[1, 2, 3], [4, 5, 6]])

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

    def test_balance_(self):
        data = ht.zeros((70, 20), split=0)
        data = data[:50]
        data.balance_()
        self.assertTrue(data.is_balanced())

        data = ht.zeros((4, 120), split=1)
        data = data[:, 40:70]
        data.balance_()
        self.assertTrue(data.is_balanced())

        data = ht.zeros((70, 20), split=0, dtype=ht.float64)
        data = data[:50]
        data.balance_()
        self.assertTrue(data.is_balanced())

        data = ht.zeros((4, 120), split=1, dtype=ht.int64)
        data = data[:, 40:70]
        data.balance_()
        self.assertTrue(data.is_balanced())

        data = np.loadtxt("heat/datasets/data/iris.csv", delimiter=";")
        htdata = ht.load("heat/datasets/data/iris.csv", sep=";", split=0)
        self.assertTrue(ht.equal(htdata, ht.array(data, split=0, dtype=ht.float)))

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
        self.assertEqual(a._DNDarray__array[0, 0], 1)
        self.assertEqual(a.lloc[0, 0].dtype, torch.float32)

        # multiple set
        a = ht.zeros((13, 5), split=0)
        a.lloc[1:3, 1] = 1
        self.assertTrue(all(a._DNDarray__array[1:3, 1] == 1))
        self.assertEqual(a.lloc[1:3, 1].dtype, torch.float32)

        # multiple set with specific indexing
        a = ht.zeros((13, 5), split=0)
        a.lloc[3:7:2, 2:5:2] = 1
        self.assertTrue(torch.all(a._DNDarray__array[3:7:2, 2:5:2] == 1))
        self.assertEqual(a.lloc[3:7:2, 2:5:2].dtype, torch.float32)

    def test_numpy(self):
        # ToDo: numpy does not work for distributed tensors du to issue#
        # Add additional tests if the issue is solved
        a = np.random.randn(10, 8)
        b = ht.array(a)
        self.assertIsInstance(b.numpy(), np.ndarray)
        self.assertEqual(b.numpy().shape, a.shape)
        self.assertEqual(b.numpy().tolist(), b._DNDarray__array.cpu().numpy().tolist())

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
        data = ht.array(expected, split=1)
        data.resplit_(None)

        self.assertTrue(torch.equal(data._DNDarray__array, expected))
        self.assertFalse(data.is_distributed())
        self.assertIsNone(data.split)
        self.assertEqual(data.dtype, ht.int64)
        self.assertEqual(data._DNDarray__array.dtype, expected.dtype)

        expected = torch.zeros((100, ht.MPI_WORLD.size), dtype=torch.uint8, device=device)
        data = ht.array(expected, split=0)
        data.resplit_(None)

        self.assertTrue(torch.equal(data._DNDarray__array, expected))
        self.assertFalse(data.is_distributed())
        self.assertIsNone(data.split)
        self.assertEqual(data.dtype, ht.uint8)
        self.assertEqual(data._DNDarray__array.dtype, expected.dtype)

    def test_setitem_getitem(self):
        # set and get single value
        a = ht.zeros((13, 5), split=0)
        # set value on one node
        a[10, 0] = 1
        self.assertEqual(a[10, 0], 1)
        self.assertEqual(a[10, 0].dtype, ht.float32)

        a = ht.zeros((13, 5), split=0)
        a[10] = 1
        b = a[10]
        self.assertTrue((b == 1).all())
        self.assertEqual(b.dtype, ht.float32)
        self.assertEqual(b.gshape, (5,))

        a = ht.zeros((13, 5), split=0)
        a[-1] = 1
        b = a[-1]
        self.assertTrue((b == 1).all())
        self.assertEqual(b.dtype, ht.float32)
        self.assertEqual(b.gshape, (5,))

        # slice in 1st dim only on 1 node
        a = ht.zeros((13, 5), split=0)
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

        a = ht.zeros((13, 5), split=0)
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
        a = ht.zeros((13, 5), split=0)
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
        a = ht.zeros((13, 5), split=0)
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
        c = ht.zeros((13, 5), split=0)
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
        a = ht.zeros((13, 5), split=0)
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
        a = ht.zeros((4, 5), split=0)
        a[1, 0:4] = ht.arange(4)
        # if a.comm.size == 2:
        for c, i in enumerate(range(4)):
            self.assertEqual(a[1, c], i)

        # setting with torch tensor
        a = ht.zeros((4, 5), split=0)
        a[1, 0:4] = torch.arange(4, device=device)
        # if a.comm.size == 2:
        for c, i in enumerate(range(4)):
            self.assertEqual(a[1, c], i)

        ###################################################
        a = ht.zeros((13, 5), split=1)
        # # set value on one node
        a[10] = 1
        self.assertEqual(a[10].dtype, ht.float32)
        if a.comm.size == 2:
            if a.comm.rank == 0:
                self.assertEqual(a[10].lshape, (3,))
            if a.comm.rank == 1:
                self.assertEqual(a[10].lshape, (2,))

        a = ht.zeros((13, 5), split=1)
        # # set value on one node
        a[10, 0] = 1
        self.assertEqual(a[10, 0], 1)
        self.assertEqual(a[10, 0].dtype, ht.float32)

        # slice in 1st dim only on 1 node
        a = ht.zeros((13, 5), split=1)
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
        a = ht.zeros((13, 5), split=1)
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
        a = ht.zeros((13, 5), split=1)
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
        a = ht.zeros((13, 5), split=1)
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
        a = ht.zeros((13, 5), split=1)
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
        a = ht.zeros((4, 5), split=1)
        a[1, 0:4] = ht.arange(4)
        for c, i in enumerate(range(4)):
            self.assertEqual(a[1, c], i)

        # setting with torch tensor
        a = ht.zeros((4, 5), split=1)
        a[1, 0:4] = torch.arange(4, device=device)
        for c, i in enumerate(range(4)):
            self.assertEqual(a[1, c], i)

        ####################################################
        a = ht.zeros((13, 5, 7), split=2)
        # # set value on one node
        a[10, :, :] = 1
        self.assertEqual(a[10, :, :].dtype, ht.float32)
        self.assertEqual(a[10, :, :].gshape, (5, 7))
        if a.comm.size == 2:
            if a.comm.rank == 0:
                self.assertEqual(a[10, :, :].lshape, (5, 4))
            if a.comm.rank == 1:
                self.assertEqual(a[10, :, :].lshape, (5, 3))

        a = ht.zeros((13, 5, 8), split=2)
        # # set value on one node
        a[10, 0, 0] = 1
        self.assertEqual(a[10, 0, 0], 1)
        self.assertEqual(a[10, 0, 0].dtype, ht.float32)

        # # slice in 1st dim only on 1 node
        a = ht.zeros((13, 5, 7), split=2)
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
        a = ht.zeros((13, 5, 7), split=2)
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
        a = ht.zeros((13, 5, 7), split=2)
        a[3:13, 2:5:2, 1:7:3] = 1
        self.assertTrue((a[3:13, 2:5:2, 1:7:3] == 1).all())
        self.assertEqual(a[3:13, 2:5:2, 1:7:3].split, 2)
        self.assertEqual(a[3:13, 2:5:2, 1:7:3].dtype, ht.float32)
        self.assertEqual(a[3:13, 2:5:2, 1:7:3].gshape, (10, 2, 2))
        if a.comm.size == 2:
            out = ht.ones((4, 5, 5), split=1)
            self.assertEqual(out[0].gshape, (5, 5))
            if a.comm.rank == 1:
                self.assertEqual(a[3:13, 2:5:2, 1:7:3].lshape, (10, 2, 1))
                self.assertEqual(out[0].lshape, (2, 5))
            if a.comm.rank == 0:
                self.assertEqual(a[3:13, 2:5:2, 1:7:3].lshape, (10, 2, 1))
                self.assertEqual(out[0].lshape, (3, 5))

        a = ht.ones((4, 5), split=0).tril()
        a[0] = [6, 6, 6, 6, 6]
        self.assertTrue((a[0] == 6).all())

        a = ht.ones((4, 5), split=0).tril()
        a[0] = (6, 6, 6, 6, 6)
        self.assertTrue((a[0] == 6).all())

        a = ht.ones((4, 5), split=0).tril()
        a[0] = np.array([6, 6, 6, 6, 6])
        self.assertTrue((a[0] == 6).all())

        a = ht.ones((4, 5), split=0).tril()
        a[0] = ht.array([6, 6, 6, 6, 6])
        self.assertTrue((a[ht.array((0,))] == 6).all())

        a = ht.ones((4, 5), split=0).tril()
        a[0] = ht.array([6, 6, 6, 6, 6])
        self.assertTrue((a[ht.array((0,))] == 6).all())

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
