import numpy as np
import torch

import heat as ht
from .test_suites.basic_test import TestCase


class TestManipulations(TestCase):
    def test_broadcast_arrays(self):
        a = ht.array([[1], [2]])
        b = ht.array([[0, 1]])
        a_broadcasted, b_broadcasted = ht.broadcast_arrays(a, b)
        self.assertTrue(ht.equal(a_broadcasted, ht.array([[1, 1], [2, 2]])))
        self.assertTrue(ht.equal(b_broadcasted, ht.array([[0, 1], [0, 1]])))

        # check dtype
        arrays = [
            ht.arange(9, dtype=ht.int32).reshape((3, 1, 3)),
            ht.arange(6, dtype=ht.float32).reshape((1, 2, 3)),
            ht.array([[[True], [False]]]),
        ]
        broadcasted = ht.broadcast_arrays(*arrays)
        self.assertTrue(all(x.shape == (3, 2, 3) for x in broadcasted))
        self.assertTrue(all(x.dtype == a.dtype for a, x in zip(arrays, broadcasted)))

        # check broadcast_arrays on one array only
        a = ht.array([[1], [2]])
        (a_broadcasted,) = ht.broadcast_arrays(a)
        self.assertTrue(ht.equal(a_broadcasted, a))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.broadcast_arrays(ht.ones((2, 3)), 4, False)
        with self.assertRaises(ValueError):
            ht.broadcast_arrays(ht.ones((10, 6), split=0), ht.ones((10), split=0))

        if a.comm.size > 1:
            a = ht.ones((5, 1, 5), split=0)
            b = ht.ones((5, 5, 5), split=1)
            with self.assertRaises(ValueError):
                ht.broadcast_arrays(a, b)

    def tests_broadcast_to(self):
        a = ht.array([1, 2, 3])
        broadcasted = ht.broadcast_to(a, (3, 3))
        self.assertTrue(ht.equal(broadcasted, ht.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])))

        # check dtype
        bool_array = ht.array([[False, True]])
        broadcasted = ht.broadcast_to(bool_array, (5, 2))
        self.assertEqual(broadcasted.shape, (5, 2))
        self.assertEqual(broadcasted.dtype, ht.bool)
        float_array = ht.arange(3, dtype=ht.float32).reshape((3, 1))
        broadcasted = ht.broadcast_to(float_array, (3, 4))
        self.assertEqual(broadcasted.shape, (3, 4))
        self.assertEqual(broadcasted.dtype, ht.float32)

        # check split
        a = ht.zeros((5, 5), split=0)
        broadcasted = ht.broadcast_to(a, (5, 5, 5))
        self.assertEqual(broadcasted.split, 1)

        # test view
        a = ht.arange(5)
        broadcasted = ht.broadcast_to(a, (10, 5))
        a[0] = 5
        self.assertTrue(ht.equal(a, broadcasted[0]))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.broadcast_to(a.larray, (10, 5))

    def test_column_stack(self):
        # test local column_stack, 2-D arrays
        a = np.arange(10, dtype=np.float32).reshape(5, 2)
        b = np.arange(15, dtype=np.float32).reshape(5, 3)
        np_cstack = np.column_stack((a, b))
        ht_a = ht.array(a)
        ht_b = ht.array(b)
        ht_cstack = ht.column_stack((ht_a, ht_b))
        self.assertTrue((np_cstack == ht_cstack.numpy()).all())

        # 2-D and 1-D arrays
        c = np.arange(5, dtype=np.float32)
        np_cstack = np.column_stack((a, b, c))
        ht_c = ht.array(c)
        ht_cstack = ht.column_stack((ht_a, ht_b, ht_c))
        self.assertTrue((np_cstack == ht_cstack.numpy()).all())

        # 2-D and 1-D arrays, distributed
        c = np.arange(5, dtype=np.float32)
        np_cstack = np.column_stack((a, b, c))
        ht_a = ht.array(a, split=1)
        ht_b = ht.array(b, split=1)
        ht_c = ht.array(c, split=0)
        ht_cstack = ht.column_stack((ht_a, ht_b, ht_c))
        self.assertTrue((ht_cstack.numpy() == np_cstack).all())
        self.assertTrue(ht_cstack.split == 1)

        # 1-D arrays, distributed, different dtypes
        d = np.arange(10).astype(np.float32)
        e = np.arange(10)
        np_cstack = np.column_stack((d, e))
        ht_d = ht.array(d, split=0)
        ht_e = ht.array(e, split=0)
        ht_cstack = ht.column_stack((ht_d, ht_e))
        self.assertTrue((ht_cstack.numpy() == np_cstack).all())
        self.assertTrue(ht_cstack.dtype == ht.float32)
        self.assertTrue(ht_cstack.split == 0)

        # test exceptions
        f = ht.random.randn(5, 4, 2, split=1)
        with self.assertRaises(ValueError):
            ht.column_stack((a, b, f))

    def test_collect(self):
        st = ht.zeros((50,), split=0)
        prev_lshape = st.lshape
        if st.comm.size >= 3:
            stc = ht.collect(st)
            if stc.comm.rank == 0:
                self.assertEqual(stc.lshape, (50,))
            else:
                self.assertEqual(stc.lshape, (0,))
            self.assertEqual(st.lshape, prev_lshape)

            st = ht.zeros((50, 50), split=1)
            prev_lshape = st.lshape
            stc = ht.collect(st, 2)
            if stc.comm.rank == 2:
                self.assertEqual(stc.lshape, (50, 50))
            else:
                self.assertEqual(stc.lshape, (50, 0))
            self.assertEqual(st.lshape, prev_lshape)

            prev_lshape = stc.lshape
            stc2 = ht.collect(stc, 1)
            if stc2.comm.rank == 1:
                self.assertEqual(stc2.lshape, (50, 50))
            else:
                self.assertEqual(stc2.lshape, (50, 0))
            self.assertEqual(stc.lshape, prev_lshape)

            st = ht.zeros((50, 81, 67), split=2)
            prev_lshape = st.lshape
            stc = ht.collect(st, 1)
            if stc.comm.rank == 1:
                self.assertEqual(stc.lshape, (50, 81, 67))
            else:
                self.assertEqual(stc.lshape, (50, 81, 0))
            self.assertEqual(st.lshape, prev_lshape)

            st = ht.zeros((5, 8, 31), split=None)  # nothing should happen
            prev_lshape = st.lshape
            stc = ht.collect(st)
            self.assertEqual(stc.lshape, stc.gshape)
            self.assertEqual(st.lshape, prev_lshape)

        st = ht.zeros((50, 81, 67), split=0)
        with self.assertRaises(TypeError):
            stc = ht.collect(st, "st.comm.size + 1")
        with self.assertRaises(TypeError):
            stc = ht.collect(st, 1.0)
        with self.assertRaises(TypeError):
            stc = ht.collect(st, (1, 3))
        with self.assertRaises(ValueError):
            stc = ht.collect(st, st.comm.size + 1)

    def test_concatenate(self):
        # cases to test:
        # Matrices / Vectors
        # s0    s1  axis
        # None None 0
        x = ht.zeros((16, 15), split=None)
        y = ht.ones((16, 15), split=None)
        res = ht.concatenate((x, y), axis=0)

        self.assertEqual(res.gshape, (32, 15))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((32, 15), res.split)
        lshape = [0, 0]
        for i in range(2):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))

        # None None 1
        res = ht.concatenate((x, y), axis=1)
        self.assertEqual(res.gshape, (16, 30))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((16, 30), res.split)
        lshape = [0, 0]
        for i in range(2):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))
        # =============================================
        # None 0 0
        x = ht.zeros((16, 15), split=None)
        y = ht.ones((16, 15), split=0)
        res = ht.concatenate((x, y), axis=0)

        self.assertEqual(res.gshape, (32, 15))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((32, 15), res.split)
        lshape = [0, 0]
        for i in range(2):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))

        # None 0 1
        res = ht.concatenate((x, y), axis=1)
        self.assertEqual(res.gshape, (16, 30))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((16, 30), res.split)
        lshape = [0, 0]
        for i in range(2):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))
        # =============================================
        # None 1 1
        x = ht.zeros((16, 15), split=None)
        y = ht.ones((16, 15), split=1)
        res = ht.concatenate((x, y), axis=1)
        self.assertEqual(res.gshape, (16, 30))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((16, 30), res.split)
        lshape = [0, 0]
        for i in range(2):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))
        #
        # None 1 0
        x = ht.zeros((16, 15), split=None)
        y = ht.ones((16, 15), split=1)
        res = ht.concatenate((x, y), axis=0)
        self.assertEqual(res.gshape, (32, 15))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((32, 15), res.split)
        lshape = [0, 0]
        for i in range(2):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))

        # # =============================================
        # # 0 None 0
        x = ht.zeros((16, 15), split=0)
        y = ht.ones((16, 15), split=None)
        res = ht.concatenate((x, y), axis=0)
        self.assertEqual(res.gshape, (32, 15))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((32, 15), res.split)
        lshape = [0, 0]
        for i in range(2):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))
        # # 0 None 1
        res = ht.concatenate((x, y), axis=1)
        self.assertEqual(res.gshape, (16, 30))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((16, 30), res.split)
        lshape = [0, 0]
        for i in range(2):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))

        # =============================================
        # 1 None 0
        x = ht.zeros((16, 15), split=1)
        y = ht.ones((16, 15), split=None)
        res = ht.concatenate((x, y), axis=0)
        self.assertEqual(res.gshape, (32, 15))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((32, 15), res.split)
        lshape = [0, 0]
        for i in range(2):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))
        # 1 None 1
        res = ht.concatenate((x, y), axis=1)
        self.assertEqual(res.gshape, (16, 30))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((16, 30), res.split)
        lshape = [0, 0]
        for i in range(2):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))

        # =============================================
        x = ht.zeros((16, 15), split=0)
        y = ht.ones((16, 15), split=0)
        # # 0 0 0
        res = ht.concatenate((x, y), axis=0)
        self.assertEqual(res.gshape, (32, 15))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((32, 15), res.split)
        lshape = [0, 0]
        for i in range(2):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))
        # 0 0 1
        res = ht.concatenate((x, y), axis=1)
        self.assertEqual(res.gshape, (16, 30))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((16, 30), res.split)
        lshape = [0, 0]
        for i in range(2):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))

        # =============================================
        x = ht.zeros((16, 15), split=1)
        y = ht.ones((16, 15), split=1)
        # 1 1 0
        res = ht.concatenate((x, y), axis=0)
        self.assertEqual(res.gshape, (32, 15))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((32, 15), res.split)
        lshape = [0, 0]
        for i in range(2):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))
        # # 1 1 1
        res = ht.concatenate((x, y), axis=1)
        self.assertEqual(res.gshape, (16, 30))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((16, 30), res.split)
        lshape = [0, 0]
        for i in range(2):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))

        # =============================================
        x = ht.zeros((16, 15, 14), split=2)
        y = ht.ones((16, 15, 14), split=2)
        # 2 2 0
        res = ht.concatenate((x, y), axis=0)
        self.assertEqual(res.gshape, (32, 15, 14))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((32, 15, 14), res.split)
        lshape = [0, 0, 0]
        for i in range(3):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))
        # 2 2 1
        res = ht.concatenate((x, y), axis=1)
        self.assertEqual(res.gshape, (16, 30, 14))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((16, 30, 14), res.split)
        lshape = [0, 0, 0]
        for i in range(3):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))
        # # 2 2 2
        res = ht.concatenate((x, y), axis=2)
        self.assertEqual(res.gshape, (16, 15, 28))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((16, 15, 28), res.split)
        lshape = [0, 0, 0]
        for i in range(3):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))
        #
        # =============================================
        y = ht.ones((16, 15, 14), split=None)
        # 2 None 1
        res = ht.concatenate((x, y), axis=1)
        self.assertEqual(res.gshape, (16, 30, 14))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((16, 30, 14), res.split)
        lshape = [0, 0, 0]
        for i in range(3):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))
        # 2 None 2
        res = ht.concatenate((x, y), axis=2)
        self.assertEqual(res.gshape, (16, 15, 28))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((16, 15, 28), res.split)
        lshape = [0, 0, 0]
        for i in range(3):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))

        res = ht.concatenate((x, y), axis=-1)
        self.assertEqual(res.gshape, (16, 15, 28))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((16, 15, 28), res.split)
        lshape = [0, 0, 0]
        for i in range(3):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))

        # =============================================
        x = ht.zeros((16, 15, 14), split=None)
        y = ht.ones((16, 15, 14), split=2)
        # None 2 0
        res = ht.concatenate((x, y), axis=0)
        self.assertEqual(res.gshape, (32, 15, 14))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((32, 15, 14), res.split)
        lshape = [0, 0, 0]
        for i in range(3):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))

        x = ht.zeros((16, 15, 14), split=None)
        y = ht.ones((16, 15, 14), split=2)
        # None 2 0
        res = ht.concatenate((x, y, y), axis=0)
        self.assertEqual(res.gshape, (32 + 16, 15, 14))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((32 + 16, 15, 14), res.split)
        lshape = [0, 0, 0]
        for i in range(3):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))

        # None 2 2
        res = ht.concatenate((x, y), axis=2)
        self.assertEqual(res.gshape, (16, 15, 28))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((16, 15, 28), res.split)
        lshape = [0, 0, 0]
        for i in range(3):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))

        # vectors
        # None None 0
        x = ht.zeros((16,), split=None)
        y = ht.ones((16,), split=None)
        res = ht.concatenate((x, y), axis=0)
        self.assertEqual(res.gshape, (32,))
        self.assertEqual(res.dtype, ht.float)
        # None 0 0
        y = ht.ones((16,), split=0)
        res = ht.concatenate((x, y), axis=0)
        self.assertEqual(res.gshape, (32,))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((32,), res.split)
        lshape = [0]
        lshape[0] = chk[0].stop - chk[0].start
        self.assertEqual(res.lshape, tuple(lshape))

        # 0 0 0
        x = ht.ones((16,), split=0, dtype=ht.float64)
        res = ht.concatenate((x, y), axis=0)
        self.assertEqual(res.gshape, (32,))
        self.assertEqual(res.dtype, ht.float64)
        _, _, chk = res.comm.chunk((32,), res.split)
        lshape = [0]
        lshape[0] = chk[0].stop - chk[0].start
        self.assertEqual(res.lshape, tuple(lshape))
        # 0 None 0
        x = ht.ones((16,), split=0)
        y = ht.ones((16,), split=None, dtype=ht.int64)
        res = ht.concatenate((x, y), axis=0)
        self.assertEqual(res.gshape, (32,))
        self.assertEqual(res.dtype, ht.float64)
        _, _, chk = res.comm.chunk((32,), res.split)
        lshape = [0]
        lshape[0] = chk[0].stop - chk[0].start
        self.assertEqual(res.lshape, tuple(lshape))

        # test raises
        with self.assertRaises(ValueError):
            ht.concatenate((ht.zeros((6, 3, 5)), ht.zeros((4, 5, 1))))
        with self.assertRaises(TypeError):
            ht.concatenate((x, "5"))
        with self.assertRaises(TypeError):
            ht.concatenate((x))
        with self.assertRaises(TypeError):
            ht.concatenate((x, x), axis=x)
        with self.assertRaises(ValueError):
            ht.concatenate((x, ht.zeros((2, 2))), axis=0)
        with self.assertRaises(RuntimeError):
            a = ht.zeros((10,), comm=ht.communication.MPI_WORLD)
            b = ht.zeros((10,), comm=ht.communication.MPI_SELF)
            ht.concatenate([a, b])
        with self.assertRaises(ValueError):
            ht.concatenate((ht.zeros((12, 12)), ht.zeros((2, 2))), axis=0)
        with self.assertRaises(RuntimeError):
            ht.concatenate((ht.zeros((2, 2), split=0), ht.zeros((2, 2), split=1)), axis=0)

    def test_diag(self):
        size = ht.MPI_WORLD.size
        rank = ht.MPI_WORLD.rank

        data = torch.arange(size * 2, device=self.device.torch_device)
        a = ht.array(data)
        res = ht.diag(a)
        self.assertTrue(torch.equal(res.larray, torch.diag(data)))

        res = ht.diag(a, offset=size)
        self.assertTrue(torch.equal(res.larray, torch.diag(data, diagonal=size)))

        res = ht.diag(a, offset=-size)
        self.assertTrue(torch.equal(res.larray, torch.diag(data, diagonal=-size)))

        a = ht.array(data, split=0)
        res = ht.diag(a)

        self.assertEqual(res.split, a.split)
        self.assertEqual(res.shape, (size * 2, size * 2))
        self.assertEqual(res.lshape[res.split], 2)
        exp = torch.diag(data)
        counts, displs = res.counts_displs()
        local_exp = exp[displs[rank] : displs[rank] + counts[rank]]
        self.assertTrue(torch.equal(res.larray, local_exp))

        res = ht.diag(a, offset=size)

        self.assertEqual(res.split, a.split)
        self.assertEqual(res.shape, (size * 3, size * 3))
        self.assertEqual(res.lshape[res.split], 3)
        exp = torch.diag(data, diagonal=size)

        torch.manual_seed(size)
        i = torch.randint(a.shape[0], ()).item()
        self.assertTrue(torch.equal(res[i, i + size].larray, exp[i, i + size]))

        res = ht.diag(a, offset=-size)
        self.assertEqual(res.split, a.split)
        self.assertEqual(res.shape, (size * 3, size * 3))
        self.assertEqual(res.lshape[res.split], 3)
        exp = torch.diag(data, diagonal=-size)
        counts, displs = res.counts_displs()
        local_exp = exp[displs[rank] : displs[rank] + counts[rank]]
        self.assertTrue(torch.equal(res.larray, local_exp))

        self.assertTrue(ht.equal(ht.diag(ht.diag(a)), a))

        a = ht.random.rand(15, 20, 5, split=1)
        res_1 = ht.diag(a)
        res_2 = ht.diagonal(a)
        self.assertTrue(ht.equal(res_1, res_2))

        with self.assertRaises(TypeError):
            ht.diag(data)

        with self.assertRaises(ValueError):
            ht.diag(a, offset=None)

        a = ht.arange(size)
        with self.assertRaises(ValueError):
            ht.diag(a, offset="3")

        a = ht.empty([])
        with self.assertRaises(ValueError):
            ht.diag(a)

        if rank == 0:
            data = torch.ones(size, dtype=torch.int32, device=self.device.torch_device)
        else:
            data = torch.empty(0, dtype=torch.int32, device=self.device.torch_device)
        a = ht.array(data, is_split=0)
        res = ht.diag(a)
        torch.manual_seed(size)
        i = torch.randint(size, ()).item()
        self.assertTrue(
            torch.equal(
                res[i, i].larray,
                torch.tensor(1, dtype=torch.int32, device=self.device.torch_device),
            )
        )

        self.assert_func_equal_for_tensor(
            np.arange(23),
            heat_func=ht.diag,
            numpy_func=np.diag,
            heat_args={"offset": 2},
            numpy_args={"k": 2},
        )

        self.assert_func_equal(
            (5,),
            heat_func=ht.diag,
            numpy_func=np.diag,
            heat_args={"offset": -3},
            numpy_args={"k": -3},
        )

    def test_diagonal(self):
        size = ht.MPI_WORLD.size
        rank = ht.MPI_WORLD.rank

        data = torch.arange(size, device=self.device.torch_device).repeat(size).reshape(size, size)
        a = ht.array(data)
        res = ht.diagonal(a)
        self.assertTrue(
            torch.equal(res.larray, torch.arange(size, device=self.device.torch_device))
        )
        self.assertEqual(res.split, None)

        a = ht.array(data, split=0)
        res = ht.diagonal(a)
        self.assertTrue(
            torch.equal(res.larray, torch.tensor([rank], device=self.device.torch_device))
        )
        self.assertEqual(res.split, 0)

        a = ht.array(data, split=1)
        res2 = ht.diagonal(a, dim1=1, dim2=0)
        self.assertTrue(ht.equal(res, res2))

        res = ht.diagonal(a)
        self.assertTrue(
            torch.equal(res.larray, torch.tensor([rank], device=self.device.torch_device))
        )
        self.assertEqual(res.split, 0)

        a = ht.array(data, split=0)
        res2 = ht.diagonal(a, dim1=1, dim2=0)
        self.assertTrue(ht.equal(res, res2))

        data = (
            torch.arange(size + 1, device=self.device.torch_device)
            .repeat(size + 1)
            .reshape(size + 1, size + 1)
        )
        a = ht.array(data)
        res = ht.diagonal(a, offset=0)
        self.assertTrue(
            torch.equal(res.larray, torch.arange(size + 1, device=self.device.torch_device))
        )
        res = ht.diagonal(a, offset=1)
        self.assertTrue(
            torch.equal(res.larray, torch.arange(1, size + 1, device=self.device.torch_device))
        )
        res = ht.diagonal(a, offset=-1)
        self.assertTrue(
            torch.equal(res.larray, torch.arange(0, size, device=self.device.torch_device))
        )

        a = ht.array(data, split=0)
        res = ht.diagonal(a, offset=1)
        res.balance_()
        self.assertTrue(
            torch.equal(res.larray, torch.tensor([rank + 1], device=self.device.torch_device))
        )
        res = ht.diagonal(a, offset=-1)
        res.balance_()
        self.assertTrue(
            torch.equal(res.larray, torch.tensor([rank], device=self.device.torch_device))
        )

        a = ht.array(data, split=1)
        res = ht.diagonal(a, offset=1)
        res.balance_()
        self.assertTrue(
            torch.equal(res.larray, torch.tensor([rank + 1], device=self.device.torch_device))
        )
        res = ht.diagonal(a, offset=-1)
        res.balance_()
        self.assertTrue(
            torch.equal(res.larray, torch.tensor([rank], device=self.device.torch_device))
        )

        data = (
            torch.arange(size * 2 + 10, device=self.device.torch_device)
            .repeat(size * 2 + 10)
            .reshape(size * 2 + 10, size * 2 + 10)
        )
        a = ht.array(data)
        res = ht.diagonal(a, offset=10)
        self.assertTrue(
            torch.equal(
                res.larray, torch.arange(10, 10 + size * 2, device=self.device.torch_device)
            )
        )
        res = ht.diagonal(a, offset=-10)
        self.assertTrue(
            torch.equal(res.larray, torch.arange(0, size * 2, device=self.device.torch_device))
        )

        a = ht.array(data, split=0)
        res = ht.diagonal(a, offset=10)
        res.balance_()
        self.assertTrue(
            torch.equal(
                res.larray,
                torch.tensor([10 + rank * 2, 11 + rank * 2], device=self.device.torch_device),
            )
        )
        res = ht.diagonal(a, offset=-10)
        res.balance_()
        self.assertTrue(
            torch.equal(
                res.larray, torch.tensor([rank * 2, 1 + rank * 2], device=self.device.torch_device)
            )
        )

        a = ht.array(data, split=1)
        res = ht.diagonal(a, offset=10)
        res.balance_()
        self.assertTrue(
            torch.equal(
                res.larray,
                torch.tensor([10 + rank * 2, 11 + rank * 2], device=self.device.torch_device),
            )
        )
        res = ht.diagonal(a, offset=-10)
        res.balance_()
        self.assertTrue(
            torch.equal(
                res.larray, torch.tensor([rank * 2, 1 + rank * 2], device=self.device.torch_device)
            )
        )

        data = (
            torch.arange(size + 1, device=self.device.torch_device)
            .repeat((size + 1) * (size + 1))
            .reshape(size + 1, size + 1, size + 1)
        )
        a = ht.array(data)
        res = ht.diagonal(a)
        self.assertTrue(
            torch.equal(
                res.larray,
                torch.arange(size + 1, device=self.device.torch_device)
                .repeat(size + 1)
                .reshape(size + 1, size + 1)
                .t(),
            )
        )
        res = ht.diagonal(a, offset=1)
        self.assertTrue(
            torch.equal(
                res.larray,
                torch.arange(size + 1, device=self.device.torch_device)
                .repeat(size)
                .reshape(size, size + 1)
                .t(),
            )
        )
        res = ht.diagonal(a, offset=-1)
        self.assertTrue(
            torch.equal(
                res.larray,
                torch.arange(size + 1, device=self.device.torch_device)
                .repeat(size)
                .reshape(size, size + 1)
                .t(),
            )
        )

        res = ht.diagonal(a, dim1=1, dim2=2)
        self.assertTrue(
            torch.equal(
                res.larray,
                torch.arange(size + 1, device=self.device.torch_device)
                .repeat(size + 1)
                .reshape(size + 1, size + 1),
            )
        )
        res = ht.diagonal(a, offset=1, dim1=1, dim2=2)
        self.assertTrue(
            torch.equal(
                res.larray,
                torch.arange(1, size + 1, device=self.device.torch_device)
                .repeat(size + 1)
                .reshape(size + 1, size),
            )
        )
        res = ht.diagonal(a, offset=-1, dim1=1, dim2=2)
        self.assertTrue(
            torch.equal(
                res.larray,
                torch.arange(size, device=self.device.torch_device)
                .repeat(size + 1)
                .reshape(size + 1, size),
            )
        )

        res = ht.diagonal(a, dim1=0, dim2=2)
        self.assertTrue(
            torch.equal(
                res.larray,
                torch.arange(size + 1, device=self.device.torch_device)
                .repeat(size + 1)
                .reshape(size + 1, size + 1),
            )
        )

        a = ht.array(data, split=0)
        res = ht.diagonal(a, offset=1, dim1=0, dim2=1)
        res.balance_()
        self.assertTrue(
            torch.equal(
                res.larray,
                torch.arange(size + 1, device=self.device.torch_device).reshape(size + 1, 1),
            )
        )
        self.assertEqual(res.split, 1)

        res = ht.diagonal(a, offset=-1, dim1=0, dim2=1)
        res.balance_()
        self.assertTrue(
            torch.equal(
                res.larray,
                torch.arange(size + 1, device=self.device.torch_device).reshape(size + 1, 1),
            )
        )
        self.assertEqual(res.split, 1)

        res = ht.diagonal(a, offset=size + 1, dim1=0, dim2=1)
        res.balance_()
        self.assertTrue(
            torch.equal(
                res.larray,
                torch.empty((size + 1, 0), dtype=torch.int64, device=self.device.torch_device),
            )
        )
        self.assertTrue(res.shape[res.split] == 0)

        with self.assertRaises(ValueError):
            ht.diagonal(a, offset=None)

        with self.assertRaises(ValueError):
            ht.diagonal(a, dim1=1, dim2=1)

        with self.assertRaises(ValueError):
            ht.diagonal(a, dim1=1, dim2=-2)

        with self.assertRaises(ValueError):
            ht.diagonal(data)

        self.assert_func_equal(
            (5, 5, 5),
            heat_func=ht.diagonal,
            numpy_func=np.diagonal,
            heat_args={"dim1": 0, "dim2": 2},
            numpy_args={"axis1": 0, "axis2": 2},
        )

        self.assert_func_equal(
            (5, 4, 3, 2),
            heat_func=ht.diagonal,
            numpy_func=np.diagonal,
            heat_args={"dim1": 1, "dim2": 2},
            numpy_args={"axis1": 1, "axis2": 2},
        )

        self.assert_func_equal(
            (4, 6, 3),
            heat_func=ht.diagonal,
            numpy_func=np.diagonal,
            heat_args={"dim1": 0, "dim2": 1},
            numpy_args={"axis1": 0, "axis2": 1},
        )

    def test_dsplit(self):
        # for further testing, see test_split
        data_ht = ht.arange(24).reshape((2, 3, 4))
        data_np = data_ht.numpy()

        # indices_or_sections = int
        result = ht.dsplit(data_ht, 2)
        comparison = np.dsplit(data_np, 2)

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = tuple
        result = ht.dsplit(data_ht, (0, 1))
        comparison = np.dsplit(data_np, (0, 1))

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = list
        result = ht.dsplit(data_ht, [0, 1])
        comparison = np.dsplit(data_np, [0, 1])

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = undistributed DNDarray
        result = ht.dsplit(data_ht, ht.array([0, 1]))
        comparison = np.dsplit(data_np, np.array([0, 1]))

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = distributed DNDarray
        with self.assertWarns(Warning):
            result = ht.dsplit(data_ht, ht.array([0, 1], split=0))
        comparison = np.dsplit(data_np, np.array([0, 1]))

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

    def test_expand_dims(self):
        # vector data
        a = ht.arange(10)
        b = ht.expand_dims(a, 0)

        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(len(b.shape), 2)

        self.assertEqual(b.shape[0], 1)
        self.assertEqual(b.shape[1], a.shape[0])

        self.assertEqual(b.lshape[0], 1)
        self.assertEqual(b.lshape[1], a.shape[0])

        self.assertIs(b.split, None)

        # vector data with out-of-bounds axis
        a = ht.arange(12)
        b = a.expand_dims(1)

        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(len(b.shape), 2)

        self.assertEqual(b.shape[0], a.shape[0])
        self.assertEqual(b.shape[1], 1)

        self.assertEqual(b.lshape[0], a.shape[0])
        self.assertEqual(b.lshape[1], 1)

        self.assertIs(b.split, None)

        # volume with intermediate axis
        a = ht.empty((3, 4, 5))
        b = a.expand_dims(1)

        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(len(b.shape), 4)

        self.assertEqual(b.shape[0], a.shape[0])
        self.assertEqual(b.shape[1], 1)
        self.assertEqual(b.shape[2], a.shape[1])
        self.assertEqual(b.shape[3], a.shape[2])

        self.assertEqual(b.lshape[0], a.shape[0])
        self.assertEqual(b.lshape[1], 1)
        self.assertEqual(b.lshape[2], a.shape[1])
        self.assertEqual(b.lshape[3], a.shape[2])

        self.assertIs(b.split, None)

        # volume with negative axis
        a = ht.empty((3, 4, 5))
        b = a.expand_dims(-4)

        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(len(b.shape), 4)

        self.assertEqual(b.shape[0], 1)
        self.assertEqual(b.shape[1], a.shape[0])
        self.assertEqual(b.shape[2], a.shape[1])
        self.assertEqual(b.shape[3], a.shape[2])

        self.assertEqual(b.lshape[0], 1)
        self.assertEqual(b.lshape[1], a.shape[0])
        self.assertEqual(b.lshape[2], a.shape[1])
        self.assertEqual(b.lshape[3], a.shape[2])

        self.assertIs(b.split, None)

        # split volume with negative axis expansion after the split
        a = ht.empty((3, 4, 5), split=1)
        b = a.expand_dims(-2)

        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(len(b.shape), 4)

        self.assertEqual(b.shape[0], a.shape[0])
        self.assertEqual(b.shape[1], a.shape[1])
        self.assertEqual(b.shape[2], 1)
        self.assertEqual(b.shape[3], a.shape[2])

        self.assertEqual(b.lshape[0], a.shape[0])
        self.assertLessEqual(b.lshape[1], a.shape[1])
        self.assertEqual(b.lshape[2], 1)
        self.assertEqual(b.lshape[3], a.shape[2])

        self.assertIs(b.split, 1)

        # split volume with negative axis expansion before the split
        a = ht.empty((3, 4, 5), split=2)
        b = a.expand_dims(-3)

        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(len(b.shape), 4)

        self.assertEqual(b.shape[0], a.shape[0])
        self.assertEqual(b.shape[1], 1)
        self.assertEqual(b.shape[2], a.shape[1])
        self.assertEqual(b.shape[3], a.shape[2])

        self.assertEqual(b.lshape[0], a.shape[0])
        self.assertEqual(b.lshape[1], 1)
        self.assertEqual(b.lshape[2], a.shape[1])
        self.assertLessEqual(b.lshape[3], a.shape[2])

        self.assertIs(b.split, 3)

        # exceptions
        with self.assertRaises(TypeError):
            ht.expand_dims("(3, 4, 5,)", 1)
        with self.assertRaises(TypeError):
            ht.empty((3, 4, 5)).expand_dims("1")
        with self.assertRaises(ValueError):
            ht.empty((3, 4, 5)).expand_dims(4)
        with self.assertRaises(ValueError):
            ht.empty((3, 4, 5)).expand_dims(-5)

    def test_flatten(self):
        a = ht.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], device=self.device)
        res = ht.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=a.dtype, device=self.device)
        flat = ht.flatten(a)
        self.assertTrue(ht.equal(flat, res))
        self.assertEqual(flat.dtype, res.dtype)
        self.assertEqual(flat.device, res.device)

        a = ht.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], split=0, dtype=ht.int8, device=self.device
        )
        res = ht.array([1, 2, 3, 4, 5, 6, 7, 8], split=0, dtype=ht.int8, device=self.device)
        flat = ht.flatten(a)
        self.assertTrue(ht.equal(flat, res))
        self.assertEqual(flat.dtype, res.dtype)
        self.assertEqual(flat.device, res.device)

        a = ht.array(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], split=1, device=self.device
        )
        res = ht.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], split=0, device=self.device)
        flat = ht.flatten(a)
        self.assertTrue(ht.equal(flat, res))
        self.assertEqual(flat.dtype, res.dtype)
        self.assertEqual(flat.device, res.device)

        a = ht.array(
            [[[False, False], [False, True]], [[True, False], [True, True]]],
            split=2,
            dtype=ht.bool,
            device=self.device,
        )
        res = ht.array(
            [False, False, False, True, True, False, True, True],
            split=0,
            dtype=a.dtype,
            device=self.device,
        )
        flat = ht.flatten(a)
        self.assertTrue(ht.equal(flat, res))
        self.assertEqual(flat.dtype, res.dtype)
        self.assertEqual(flat.device, res.device)

    def test_flip(self):
        a = ht.array([1, 2])
        r_a = ht.array([2, 1])
        self.assertTrue(ht.equal(ht.flip(a, 0), r_a))

        a = ht.array([[1, 2], [3, 4]])
        r_a = ht.array([[4, 3], [2, 1]])
        self.assertTrue(ht.equal(ht.flip(a), r_a))

        a = ht.array([[2, 3], [4, 5], [6, 7], [8, 9]], split=1, dtype=ht.float32)
        r_a = ht.array([[9, 8], [7, 6], [5, 4], [3, 2]], split=1, dtype=ht.float32)
        self.assertTrue(ht.equal(ht.flip(a, [0, 1]), r_a))

        a = ht.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], split=0, dtype=ht.uint8)
        r_a = ht.array([[[3, 2], [1, 0]], [[7, 6], [5, 4]]], split=0, dtype=ht.uint8)
        self.assertTrue(ht.equal(ht.flip(a, [1, 2]), r_a))

    def test_fliplr(self):
        b = ht.array([[1, 2], [3, 4]])
        r_b = ht.array([[2, 1], [4, 3]])
        self.assertTrue(ht.equal(ht.fliplr(b), r_b))

        # splitted
        c = ht.array(
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]], [[12, 13], [14, 15]]], split=0
        )
        r_c = ht.array(
            [[[2, 3], [0, 1]], [[6, 7], [4, 5]], [[10, 11], [8, 9]], [[14, 15], [12, 13]]], split=0
        )
        self.assertTrue(ht.equal(ht.fliplr(c), r_c))

        c = ht.array(
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]], [[12, 13], [14, 15]]],
            split=1,
            dtype=ht.float32,
        )
        self.assertTrue(ht.equal(ht.resplit(ht.fliplr(c), 0), r_c))

        c = ht.array(
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]], [[12, 13], [14, 15]]],
            split=2,
            dtype=ht.int8,
        )
        self.assertTrue(ht.equal(ht.resplit(ht.fliplr(c), 0), r_c))

        # test exception
        a = ht.arange(10)
        with self.assertRaises(IndexError):
            ht.fliplr(a)

    def test_flipud(self):
        a = ht.array([1, 2])
        r_a = ht.array([2, 1])
        self.assertTrue(ht.equal(ht.flipud(a), r_a))

        b = ht.array([[1, 2], [3, 4]])
        r_b = ht.array([[3, 4], [1, 2]])
        self.assertTrue(ht.equal(ht.flipud(b), r_b))

        # splitted
        c = ht.array(
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]], [[12, 13], [14, 15]]], split=0
        )
        r_c = ht.array(
            [[[12, 13], [14, 15]], [[8, 9], [10, 11]], [[4, 5], [6, 7]], [[0, 1], [2, 3]]], split=0
        )
        self.assertTrue(ht.equal(ht.flipud(c), r_c))

        c = ht.array(
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]], [[12, 13], [14, 15]]],
            split=1,
            dtype=ht.float32,
        )
        self.assertTrue(ht.equal(ht.resplit(ht.flipud(c), 0), r_c))

        c = ht.array(
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]], [[12, 13], [14, 15]]],
            split=2,
            dtype=ht.int8,
        )
        self.assertTrue(ht.equal(ht.resplit(ht.flipud(c), 0), r_c))

    def test_hsplit(self):
        # for further testing, see test_split
        # 1-dimensional array (as forbidden in split)
        data_ht = ht.arange(24)
        data_np = data_ht.numpy()

        # indices_or_sections = int
        result = ht.hsplit(data_ht, 2)
        comparison = np.hsplit(data_np, 2)

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = tuple
        result = ht.hsplit(data_ht, (0, 1))
        comparison = np.hsplit(data_np, (0, 1))

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = list
        result = ht.hsplit(data_ht, [0, 1])
        comparison = np.hsplit(data_np, [0, 1])

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = undistributed DNDarray
        result = ht.hsplit(data_ht, ht.array([0, 1]))
        comparison = np.hsplit(data_np, np.array([0, 1]))

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = distributed DNDarray
        with self.assertWarns(Warning):
            result = ht.hsplit(data_ht, ht.array([0, 1], split=0))
        comparison = np.hsplit(data_np, np.array([0, 1]))

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        data_ht = ht.arange(24).reshape((2, 4, 3))
        data_np = data_ht.numpy()

        # indices_or_sections = int
        result = ht.hsplit(data_ht, 2)
        comparison = np.hsplit(data_np, 2)

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = tuple
        result = ht.hsplit(data_ht, (0, 1))
        comparison = np.hsplit(data_np, (0, 1))

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = list
        result = ht.hsplit(data_ht, [0, 1])
        comparison = np.hsplit(data_np, [0, 1])

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = undistributed DNDarray
        result = ht.hsplit(data_ht, ht.array([0, 1]))
        comparison = np.hsplit(data_np, np.array([0, 1]))

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = distributed DNDarray
        with self.assertWarns(Warning):
            result = ht.hsplit(data_ht, ht.array([0, 1], split=0))
        comparison = np.hsplit(data_np, np.array([0, 1]))

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

    def test_hstack(self):
        # cases to test:
        # MM===================================
        # NN,
        a = ht.ones((10, 12), split=None)
        b = ht.ones((10, 12), split=None)
        res = ht.hstack((a, b))
        self.assertEqual(res.shape, (10, 24))
        # 11,
        a = ht.ones((10, 12), split=1)
        b = ht.ones((10, 12), split=1)
        res = ht.hstack((a, b))
        self.assertEqual(res.shape, (10, 24))

        # VM===================================
        # NN,
        a = ht.ones((12,), split=None)
        b = ht.ones((12, 10), split=None)
        res = ht.hstack((a, b))
        self.assertEqual(res.shape, (12, 11))
        # 00
        a = ht.ones((12,), split=0)
        b = ht.ones((12, 10), split=0)
        res = ht.hstack((a, b))
        self.assertEqual(res.shape, (12, 11))

        # MV===================================
        # NN,
        a = ht.ones((12, 10), split=None)
        b = ht.ones((12,), split=None)
        res = ht.hstack((a, b))
        self.assertEqual(res.shape, (12, 11))
        # 00
        a = ht.ones((12, 10), split=0)
        b = ht.ones((12,), split=0)
        res = ht.hstack((a, b))
        self.assertEqual(res.shape, (12, 11))

        # VV===================================
        # NN,
        a = ht.ones((12,), split=None)
        b = ht.ones((12,), split=None)
        res = ht.hstack((a, b))
        self.assertEqual(res.shape, (24,))
        # 00
        a = ht.ones((12,), split=0)
        b = ht.ones((12,), split=0)
        res = ht.hstack((a, b))
        self.assertEqual(res.shape, (24,))

    def test_moveaxis(self):
        a = ht.zeros((3, 4, 5))

        moved = ht.moveaxis(a, 0, -1)
        self.assertEqual(moved.shape, (4, 5, 3))

        moved = ht.moveaxis(a, [0, 1], [-1, -2])
        self.assertEqual(moved.shape, (5, 4, 3))

        with self.assertRaises(TypeError):
            ht.moveaxis(a, source="r", destination=3)
        with self.assertRaises(TypeError):
            ht.moveaxis(a, source=2, destination=3.6)
        with self.assertRaises(ValueError):
            ht.moveaxis(a, source=[0, 1, 2], destination=[0, 1])

    def test_pad(self):
        # ======================================
        # test padding of non-distributed tensor
        # ======================================

        data = torch.arange(2 * 3 * 4, device=self.device.torch_device).reshape(2, 3, 4)
        data_ht = ht.array(data, device=self.device)
        data_np = data_ht.numpy()

        # padding with default (0 for all dimensions)
        pad_torch = torch.nn.functional.pad(data, (1, 2, 1, 0, 2, 1))
        pad_ht = ht.pad(data_ht, pad_width=((2, 1), (1, 0), (1, 2)))

        self.assert_array_equal(pad_ht, pad_torch)
        self.assertIsInstance(pad_ht, ht.DNDarray)

        # padding with other values than default
        pad_numpy = np.pad(
            data_np,
            pad_width=((2, 1), (1, 0), (1, 2)),
            mode="constant",
            constant_values=((0, 3), (1, 4), (2, 5)),
        )
        pad_ht = ht.pad(
            data_ht,
            pad_width=((2, 1), (1, 0), (1, 2)),
            mode="constant",
            constant_values=((0, 3), (1, 4), (2, 5)),
        )
        self.assert_array_equal(pad_ht, pad_numpy)

        # shortcuts pad_width===================================
        pad_numpy = np.pad(
            data_np, pad_width=((2, 1),), mode="constant", constant_values=((0, 3), (1, 4), (2, 5))
        )
        pad_ht = ht.pad(
            data_ht, pad_width=((2, 1),), mode="constant", constant_values=((0, 3), (1, 4), (2, 5))
        )
        self.assert_array_equal(pad_ht, pad_numpy)

        pad_numpy = np.pad(
            data_np, pad_width=(2, 1), mode="constant", constant_values=((0, 3), (1, 4), (2, 5))
        )
        pad_ht = ht.pad(
            data_ht, pad_width=(2, 1), mode="constant", constant_values=((0, 3), (1, 4), (2, 5))
        )
        self.assert_array_equal(pad_ht, pad_numpy)

        pad_numpy = np.pad(
            data_np, pad_width=(2,), mode="constant", constant_values=((0, 3), (1, 4), (2, 5))
        )
        pad_ht = ht.pad(
            data_ht, pad_width=(2,), mode="constant", constant_values=((0, 3), (1, 4), (2, 5))
        )
        self.assert_array_equal(pad_ht, pad_numpy)

        pad_numpy = np.pad(
            data_np, pad_width=2, mode="constant", constant_values=((0, 3), (1, 4), (2, 5))
        )
        pad_ht = ht.pad(
            data_ht, pad_width=2, mode="constant", constant_values=((0, 3), (1, 4), (2, 5))
        )
        self.assert_array_equal(pad_ht, pad_numpy)

        # pad_width datatype list===================================
        # padding with default (0 for all dimensions)
        pad_torch = torch.nn.functional.pad(data, (1, 2, 1, 0, 2, 1))
        pad_ht = ht.pad(data_ht, pad_width=((2, 1), [1, 0], [1, 2]))

        self.assert_array_equal(pad_ht, pad_torch)

        # padding with other values than default
        pad_numpy = np.pad(
            data_np,
            pad_width=((2, 1), (1, 0), (1, 2)),
            mode="constant",
            constant_values=((0, 3), (1, 4), (2, 5)),
        )
        pad_ht = ht.pad(
            data_ht,
            pad_width=[(2, 1), (1, 0), (1, 2)],
            mode="constant",
            constant_values=((0, 3), (1, 4), (2, 5)),
        )
        self.assert_array_equal(pad_ht, pad_numpy)

        # shortcuts constant_values===================================

        pad_numpy = np.pad(
            data_np, pad_width=((2, 1), (1, 0), (1, 2)), mode="constant", constant_values=((0, 3),)
        )
        pad_ht = ht.pad(
            data_ht, pad_width=((2, 1), (1, 0), (1, 2)), mode="constant", constant_values=((0, 3),)
        )
        self.assert_array_equal(pad_ht, pad_numpy)

        pad_numpy = np.pad(
            data_np, pad_width=((2, 1), (1, 0), (1, 2)), mode="constant", constant_values=(0, 3)
        )
        pad_ht = ht.pad(
            data_ht, pad_width=((2, 1), (1, 0), (1, 2)), mode="constant", constant_values=(0, 3)
        )
        self.assert_array_equal(pad_ht, pad_numpy)

        pad_numpy = np.pad(
            data_np, pad_width=((2, 1), (1, 0), (1, 2)), mode="constant", constant_values=(3,)
        )
        pad_ht = ht.pad(
            data_ht, pad_width=((2, 1), (1, 0), (1, 2)), mode="constant", constant_values=(3,)
        )
        self.assert_array_equal(pad_ht, pad_numpy)

        pad_numpy = np.pad(
            data_np, pad_width=((2, 1), (1, 0), (1, 2)), mode="constant", constant_values=4
        )
        pad_ht = ht.pad(
            data_ht, pad_width=((2, 1), (1, 0), (1, 2)), mode="constant", constant_values=4
        )
        self.assert_array_equal(pad_ht, pad_numpy)

        # values datatype list/int/float===================================
        pad_numpy = np.pad(
            data_np, pad_width=((2, 1), (1, 0), (1, 2)), mode="constant", constant_values=2
        )
        pad_ht = ht.pad(
            data_ht, pad_width=[(2, 1), (1, 0), (1, 2)], mode="constant", constant_values=2
        )
        self.assert_array_equal(pad_ht, pad_numpy)

        pad_numpy = np.pad(
            data_np, pad_width=((2, 1), (1, 0), (1, 2)), mode="constant", constant_values=1.2
        )
        pad_ht = ht.pad(
            data_ht, pad_width=[(2, 1), (1, 0), (1, 2)], mode="constant", constant_values=1.2
        )
        self.assert_array_equal(pad_ht, pad_numpy)

        pad_numpy = np.pad(
            data_np, pad_width=((2, 1), (1, 0), (1, 2)), mode="constant", constant_values=(2,)
        )
        pad_ht = ht.pad(
            data_ht, pad_width=[(2, 1), (1, 0), (1, 2)], mode="constant", constant_values=(2,)
        )
        self.assert_array_equal(pad_ht, pad_numpy)

        pad_numpy = np.pad(
            data_np,
            pad_width=((2, 1), (1, 0), (1, 2)),
            mode="constant",
            constant_values=((0, 3), (1, 4), (2, 5)),
        )
        pad_ht = ht.pad(
            data_ht,
            pad_width=((2, 1), (1, 0), (1, 2)),
            mode="constant",
            constant_values=([0, 3], [1, 4], (2, 5)),
        )
        self.assert_array_equal(pad_ht, pad_numpy)

        pad_numpy = np.pad(
            data_np,
            pad_width=((2, 1), (1, 0), (1, 2)),
            mode="constant",
            constant_values=((0, 3), (1, 4), (2, 5)),
        )
        pad_ht = ht.pad(
            data_ht,
            pad_width=((2, 1), (1, 0), (1, 2)),
            mode="constant",
            constant_values=[(0, 3), (1, 4), (2, 5)],
        )
        self.assert_array_equal(pad_ht, pad_numpy)

        # ==================================
        # test padding of distributed tensor
        # ==================================

        # rank = ht.MPI_WORLD.rank
        data_ht_split = ht.array(data, split=0, device=self.device)

        # padding in split dimension
        pad_np_split = np.pad(
            data_np, pad_width=(2, 1), mode="constant", constant_values=((0, 3), (1, 4), (2, 5))
        )
        pad_ht_split = ht.pad(
            data_ht_split,
            pad_width=(2, 1),
            mode="constant",
            constant_values=((0, 3), (1, 4), (2, 5)),
        )

        self.assert_array_equal(pad_ht_split, pad_np_split)

        # padding in split dimension, constant_values = int
        pad_np_split = np.pad(data_np, pad_width=(2, 1), mode="constant", constant_values=2)
        pad_ht_split = ht.pad(data_ht_split, pad_width=(2, 1), mode="constant", constant_values=2)

        self.assert_array_equal(pad_ht_split, pad_np_split)

        # padding in split dimension, constant_values = [int,]
        pad_np_split = np.pad(data_np, pad_width=(2, 1), mode="constant", constant_values=[2])
        pad_ht_split = ht.pad(data_ht_split, pad_width=(2, 1), mode="constant", constant_values=[2])

        self.assert_array_equal(pad_ht_split, pad_np_split)

        # padding in non split dimension
        # weird syntax necessary due to np restrictions (tuples for every axis obligatory apart from shortcuts)
        pad_np_split = np.pad(
            data_np,
            pad_width=((0, 0), (2, 1), (1, 0)),
            mode="constant",
            constant_values=((-1, 1), (0, 3), (1, 4)),
        )
        pad_ht_split = ht.pad(
            data_ht_split,
            pad_width=((2, 1), (1, 0)),
            mode="constant",
            constant_values=((0, 3), (1, 4)),
        )

        self.assert_array_equal(pad_ht_split, pad_np_split)

        # shortcuts constant_values===================================

        pad_numpy = np.pad(
            data_np, pad_width=((2, 1), (1, 0), (1, 2)), mode="constant", constant_values=((0, 3),)
        )
        pad_ht = ht.pad(
            data_ht, pad_width=((2, 1), (1, 0), (1, 2)), mode="constant", constant_values=((0, 3),)
        )
        self.assert_array_equal(pad_ht, pad_numpy)

        pad_numpy = np.pad(
            data_np, pad_width=((2, 1), (1, 0), (1, 2)), mode="constant", constant_values=(0, 3)
        )
        pad_ht = ht.pad(
            data_ht, pad_width=((2, 1), (1, 0), (1, 2)), mode="constant", constant_values=(0, 3)
        )
        self.assert_array_equal(pad_ht, pad_numpy)

        pad_numpy = np.pad(
            data_np, pad_width=((2, 1), (1, 0), (1, 2)), mode="constant", constant_values=(3,)
        )
        pad_ht = ht.pad(
            data_ht, pad_width=((2, 1), (1, 0), (1, 2)), mode="constant", constant_values=(3,)
        )
        self.assert_array_equal(pad_ht, pad_numpy)

        pad_numpy = np.pad(
            data_np, pad_width=((2, 1), (1, 0), (1, 2)), mode="constant", constant_values=4
        )
        pad_ht = ht.pad(
            data_ht, pad_width=((2, 1), (1, 0), (1, 2)), mode="constant", constant_values=4
        )
        self.assert_array_equal(pad_ht, pad_numpy)

        # exceptions===================================

        with self.assertRaises(TypeError):
            ht.pad("[[3, 4, 5],[6,7,8]]", 3)
        with self.assertRaises(TypeError):
            ht.pad(data_ht, "(1,3)")
        with self.assertRaises(TypeError):
            ht.pad(data_ht, 3, mode=["constant"])
        with self.assertRaises(TypeError):
            ht.pad(data_ht, pad_width=("(1,2),",))
        with self.assertRaises(TypeError):
            ht.pad(data_ht, ((1, 2), "(3,4)", (5, 6)))
        with self.assertRaises(TypeError):
            ht.pad(
                data_ht,
                ((2, 1), (1, 0), (1, 2)),
                mode="constant",
                constant_values=((0, 3), "(1, 4)", (2, 5)),
            )
        with self.assertRaises(ValueError):
            ht.pad(data_ht, ((1, 2, 3),))
        with self.assertRaises(ValueError):
            ht.pad(data_ht, ((1, 2), (3, 4, 5), (6, 7)))
        with self.assertRaises(ValueError):
            ht.pad(data_ht, ((2, 1), (1, 0), (1, 2), (1, 2)))
        with self.assertRaises(ValueError):
            ht.pad(
                data_ht,
                ((1, 2), (3, 4), (0, 1)),
                mode="constant",
                constant_values=((0, 3), (1, 4), (2, 5, 1)),
            )

        # =========================================
        # test padding of large distributed tensor
        # =========================================

        data = torch.arange(8 * 3 * 4, device=self.device.torch_device).reshape(8, 3, 4)
        data_ht_split = ht.array(data, split=0)
        data_np = data_ht_split.numpy()

        # padding in split dimension
        pad_np_split = np.pad(
            data_np, pad_width=(2, 1), mode="constant", constant_values=((0, 3), (1, 4), (2, 5))
        )
        pad_ht_split = ht.pad(
            data_ht_split,
            pad_width=(2, 1),
            mode="constant",
            constant_values=((0, 3), (1, 4), (2, 5)),
        )

        self.assertTrue((ht.array(pad_np_split) == pad_ht_split).all())

        # padding in non split dimension
        # weird syntax necessary due to np restrictions (tuples for every axis obligatory apart from shortcuts)
        pad_np_split = np.pad(
            data_np,
            pad_width=((0, 0), (2, 1), (1, 0)),
            mode="constant",
            constant_values=((-1, 1), (0, 3), (1, 4)),
        )
        pad_ht_split = ht.pad(
            data_ht_split,
            pad_width=((2, 1), (1, 0)),
            mode="constant",
            constant_values=((0, 3), (1, 4)),
        )

        self.assert_array_equal(pad_ht_split, pad_np_split)

    def test_ravel(self):
        a = ht.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], device=self.device)
        res = ht.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=a.dtype, device=self.device)
        ravel = a.ravel()
        self.assertTrue(ht.equal(ravel, res))
        self.assertEqual(ravel.dtype, res.dtype)
        self.assertEqual(ravel.device, res.device)
        a[0, 0, 0] = 9
        self.assertEqual(ravel[0], 9)

        a = ht.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], split=0, dtype=ht.int8, device=self.device
        )
        res = ht.flatten(a)
        ravel = ht.ravel(a)

        if a.comm.size > 2:
            ravel.balance_()

        self.assertTrue(ht.equal(ravel, res))
        self.assertEqual(ravel.dtype, res.dtype)
        self.assertEqual(ravel.device, res.device)

        a[0, 0, 0] = 9
        self.assertEqual(ravel[0], 9)

        a = ht.array(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], split=1, device=self.device
        )
        res = ht.flatten(a)
        ravel = ht.ravel(a)
        self.assertTrue(ht.equal(ravel, res))
        self.assertEqual(ravel.dtype, res.dtype)
        self.assertEqual(ravel.device, res.device)

        a[0, 0, 0] = 9
        self.assertEqual(ravel[0], 1)

        a = ht.array(
            [[[False, False], [False, True]], [[True, False], [True, True]]],
            split=2,
            dtype=ht.bool,
            device=self.device,
        )
        res = ht.flatten(a)
        ravel = ht.ravel(a)
        self.assertTrue(ht.equal(ravel, res))
        self.assertEqual(ravel.dtype, res.dtype)
        self.assertEqual(ravel.device, res.device)

        a[0, 0, 0] = 9
        self.assertEqual(ravel[0], False)

    def test_repeat(self):
        # -------------------
        # a = int
        # -------------------
        a = 42

        # axis = None
        # repeats = scalar
        repeats = 2
        result = ht.repeat(a, repeats)
        comparison = np.repeat(a, repeats)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)

        # -------------------
        # a = float
        # -------------------
        a = 4.2

        # axis = None
        # repeats = scalar
        repeats = 2
        result = ht.repeat(a, repeats)
        comparison = np.repeat(a, repeats)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)

        # -------------------
        # a = tuple
        # -------------------
        a = (1, 2, 3, 4, 5)

        # axis = None
        # repeats = scalar
        repeats = 2
        result = ht.repeat(a, repeats)
        comparison = np.repeat(a, repeats)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)

        # -------------------
        # a = list
        # -------------------
        a = [1.2, 2.4, 3, 4, 5]

        # axis = None
        # repeats = scalar
        repeats = 2
        result = ht.repeat(a, repeats)
        comparison = np.repeat(a, repeats)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)

        # -------------------
        # a = np.ndarray
        # -------------------
        a = np.array([1.2, 2.4, 3, 4, 5])
        # axis is None
        # repeats = scalar
        repeats = 2
        result = ht.repeat(a, repeats)
        comparison = np.repeat(a, repeats)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)

        # -------------------
        # a = DNDarray
        # -------------------

        # -------------------
        # UNDISTRIBUTED case
        # -------------------
        # axis = None
        # -------------------

        # a is empty
        a = ht.array([])
        a_np = a.numpy()
        repeats = 2
        result = ht.repeat(a, repeats)
        comparison = np.repeat(a_np, repeats)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)

        a = ht.arange(12).reshape((2, 2, 3))
        a_np = a.numpy()

        # repeats = scalar
        repeats = 2
        result = ht.repeat(a, repeats)
        comparison = np.repeat(a_np, repeats)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (a.size * repeats,))
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)

        # repeats = list
        repeats = [1, 2, 0, 0, 1, 3, 2, 5, 1, 0, 2, 3]
        result = ht.repeat(a, repeats)
        comparison = np.repeat(a_np, repeats)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (sum(repeats),))
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)

        # repeats = tuple
        repeats = (1, 2, 0, 0, 1, 3, 2, 5, 1, 0, 2, 3)
        result = ht.repeat(a, repeats)
        comparison = np.repeat(a_np, repeats)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (sum(repeats),))
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)

        # repeats = np.ndarray
        repeats = np.array([1, 2, 0, 0, 1, 3, 2, 5, 1, 0, 2, 3])
        result = ht.repeat(a, repeats)
        comparison = np.repeat(a_np, repeats)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (sum(repeats),))
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)

        # repeats = undistributed ht.DNDarray
        repeats = ht.array([1, 2, 0, 0, 1, 3, 2, 5, 1, 0, 2, 3])
        repeats_np = repeats.numpy()
        result = ht.repeat(a, repeats)
        comparison = np.repeat(a_np, repeats_np)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)
        self.assertIsInstance(repeats, ht.DNDarray)
        self.assertEqual(repeats.split, None)

        # dtype = ht.int32
        repeats = ht.array([1, 2, 0, 0, 1, 3, 2, 5, 1, 0, 2, 3], dtype=ht.int32)
        repeats_np = repeats.numpy()
        result = ht.repeat(a, repeats)
        comparison = np.repeat(a_np, repeats_np)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)
        self.assertIsInstance(repeats, ht.DNDarray)
        self.assertEqual(repeats.split, None)

        # Broadcast
        repeats = ht.array([3])
        repeats_np = repeats.numpy()
        result = ht.repeat(a, repeats)
        comparison = np.repeat(a_np, repeats_np)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)
        self.assertIsInstance(repeats, ht.DNDarray)
        self.assertEqual(repeats.split, None)

        # repeats = distributed ht.DNDarray
        repeats = ht.array([1, 2, 0, 0, 1, 3, 2, 5, 1, 0, 2, 3], split=0)
        with self.assertWarns(Warning):
            result = ht.repeat(a, repeats)
        comparison = np.repeat(a_np, repeats.numpy())

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)
        self.assertIsInstance(repeats, ht.DNDarray)
        self.assertEqual(repeats.split, 0)

        # Broadcast
        repeats = ht.array([3], split=0)
        repeats_np = repeats.numpy()
        with self.assertWarns(Warning):
            result = ht.repeat(a, repeats)
        comparison = np.repeat(a_np, repeats_np)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)
        self.assertIsInstance(repeats, ht.DNDarray)
        self.assertEqual(repeats.split, 0)

        # exceptions
        with self.assertRaises(TypeError):
            ht.repeat(a, repeats, axis="0")
        with self.assertRaises(TypeError):
            ht.repeat("[1, 2, 3]", repeats)
        with self.assertRaises(ValueError):
            ht.repeat(a, repeats, axis=-1)
        with self.assertRaises(ValueError):
            ht.repeat(a, repeats, axis=len(a.shape))
        with self.assertRaises(TypeError):
            repeats = np.array([1, 2, 0, 0, 1, 3, 2, 5, 1, 0, 2, 3], dtype=np.float32)
            ht.repeat(a, repeats)
        with self.assertRaises(TypeError):
            repeats = [1, 2, 0, 0, 1, "3", 2, 5, 1, 0, 2, 3]
            ht.repeat(a, repeats)
        with self.assertRaises(TypeError):
            repeats = [1, 2.4, 0, 0, 1, 3, 2, 5, 1, 0, 2, 3]
            ht.repeat(a, repeats)
        with self.assertRaises(ValueError):
            repeats = [1, 2, 0, 0, 1, 3, 2, 5, 1, 0, 2]
            ht.repeat(a, repeats)
        with self.assertRaises(ValueError):
            repeats = [1, 2]
            ht.repeat(a, repeats, axis=2)
        with self.assertRaises(TypeError):
            repeats = "[1, 2, 3]"
            ht.repeat(a, repeats, axis=2)
        with self.assertRaises(TypeError):
            repeats = np.array([1, 2, 0, 0, 1, 3, 2, 5, 1, 0, 2, 3], dtype=ht.float64)
            ht.repeat(a, repeats)
        with self.assertRaises(ValueError):
            repeats = ht.array([], dtype=ht.int64)
            ht.repeat(a, repeats)
        with self.assertRaises(TypeError):
            repeats = ht.array([1, 2, 0, 0, 1, 3, 2, 5, 1, 0, 2, 3], split=0, dtype=ht.float32)
            ht.repeat(a, repeats)
        with self.assertRaises(ValueError):
            repeats = ht.array([[1, 2, 0, 0, 1, 3, 2, 5, 1, 0, 2, 3]], split=0)
            ht.repeat(a, repeats)

        # -------------------
        # axis != None
        # -------------------

        # repeats = scalar
        repeats = 2
        result = ht.repeat(a, repeats, 2)
        comparison = np.repeat(a_np, repeats, 2)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)

        # repeats = list
        repeats = [1, 2, 0]
        result = ht.repeat(a, repeats, 2)
        comparison = np.repeat(a_np, repeats, 2)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)

        # repeats = tuple
        repeats = (1, 2, 0)
        result = ht.repeat(a, repeats, 2)
        comparison = np.repeat(a_np, repeats, 2)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)

        # repeats = np.ndarray
        repeats = np.array([1, 2, 0])
        result = ht.repeat(a, repeats, 2)
        comparison = np.repeat(a_np, repeats, 2)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)

        # repeats = undistributed ht.DNDarray
        repeats = ht.array([1, 2, 0])
        repeats_np = repeats.numpy()
        result = ht.repeat(a, repeats, 2)
        comparison = np.repeat(a_np, repeats_np, 2)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)
        self.assertIsInstance(repeats, ht.DNDarray)
        self.assertEqual(repeats.split, None)

        # repeats = distributed ht.DNDarray
        repeats = ht.array([1, 2, 0], split=0)
        with self.assertWarns(Warning):
            result = ht.repeat(a, repeats, 2)
        comparison = np.repeat(a_np, repeats.numpy(), 2)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, None)
        self.assertIsInstance(repeats, ht.DNDarray)
        self.assertEqual(repeats.split, 0)

        # -------------------
        # DISTRIBUTED CASE
        # -------------------
        # axis = None
        # -------------------
        a = ht.arange(12, split=0).reshape((2, 2, 3), new_split=1)
        a_np = a.numpy()

        # repeats = scalar
        repeats = 2
        with self.assertWarns(Warning):
            result = ht.repeat(a, repeats)
        comparison = np.repeat(a_np, repeats)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (a.size * repeats,))
        self.assert_array_equal(result, comparison)

        # repeats = list
        repeats = [1, 2, 0, 0, 1, 3, 2, 5, 1, 0, 2, 3]
        with self.assertWarns(Warning):
            result = ht.repeat(a, repeats)
        comparison = np.repeat(a_np, repeats)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.gshape, (sum(repeats),))
        self.assertEqual(result.split, 0)
        self.assertTrue((ht.array(comparison) == result).all())

        # repeats = tuple
        repeats = (1, 2, 0, 0, 1, 3, 2, 5, 1, 0, 2, 3)
        with self.assertWarns(Warning):
            result = ht.repeat(a, repeats)
        comparison = np.repeat(a_np, repeats)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (sum(repeats),))
        self.assertEqual(result.split, 0)
        self.assertTrue((ht.array(comparison) == result).all())

        # repeats = np.ndarray
        repeats = np.array([1, 2, 0, 0, 1, 3, 2, 5, 1, 0, 2, 3])
        with self.assertWarns(Warning):
            result = ht.repeat(a, repeats)
        comparison = np.repeat(a_np, repeats)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, (sum(repeats),))
        self.assertEqual(result.split, 0)
        self.assertTrue((ht.array(comparison) == result).all())

        # repeats = undistributed ht.DNDarray
        repeats = ht.array([1, 2, 0, 0, 1, 3, 2, 5, 1, 0, 2, 3])
        repeats_np = repeats.numpy()
        with self.assertWarns(Warning):
            result = ht.repeat(a, repeats)
        comparison = np.repeat(a_np, repeats_np)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assertTrue((ht.array(comparison) == result).all())
        self.assertEqual(result.split, 0)
        self.assertIsInstance(repeats, ht.DNDarray)
        self.assertEqual(repeats.split, None)

        # repeats = distributed ht.DNDarray
        repeats = ht.array([1, 2, 0, 0, 1, 3, 2, 5, 1, 0, 2, 3], split=0)
        with self.assertWarns(Warning):
            result = ht.repeat(a, repeats)

        comparison = np.repeat(a_np, repeats.numpy())

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assertTrue((ht.array(comparison) == result).all())
        self.assertEqual(result.split, 0)
        self.assertIsInstance(repeats, ht.DNDarray)
        self.assertEqual(repeats.split, 0)

        # exceptions
        with self.assertRaises(ValueError):
            repeats = [1, 2, 0, 0, 1, 3, 2, 5, 1, 0, 2]
            ht.repeat(a, repeats)

        # -------------------
        # axis != None
        # -------------------

        # repeats = scalar
        repeats = 2
        result = ht.repeat(a, repeats, 2)
        comparison = np.repeat(a_np, repeats, 2)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, a.split)

        # repeats = list
        repeats = [1, 2, 0]
        result = ht.repeat(a, repeats, 2)
        comparison = np.repeat(a_np, repeats, 2)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, a.split)

        # repeats = tuple
        repeats = (1, 2, 0)
        result = ht.repeat(a, repeats, 2)
        comparison = np.repeat(a_np, repeats, 2)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, a.split)

        # repeats = np.ndarray
        repeats = np.array([1, 2, 0])
        result = ht.repeat(a, repeats, 2)
        comparison = np.repeat(a_np, repeats, 2)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, a.split)

        # repeats = undistributed ht.DNDarray (axis != a.split)
        repeats = ht.array([1, 2, 0])
        repeats_np = repeats.numpy()
        result = ht.repeat(a, repeats, 2)
        comparison = np.repeat(a_np, repeats_np, 2)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assert_array_equal(result, comparison)
        self.assertEqual(result.split, a.split)
        self.assertIsInstance(repeats, ht.DNDarray)
        self.assertEqual(repeats.split, None)

        # exceptions
        with self.assertRaises(ValueError):
            repeats = ht.array([1, 2])
            ht.repeat(a, repeats, 2)

        # repeats = undistributed ht.DNDarray (axis == a.split)
        repeats = ht.array([1, 2])
        repeats_np = repeats.numpy()
        with self.assertWarns(Warning):
            result = ht.repeat(a, repeats, 1)
        comparison = np.repeat(a_np, repeats_np, 1)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assertTrue((ht.array(comparison) == result).all())
        self.assertEqual(result.split, a.split)
        self.assertIsInstance(repeats, ht.DNDarray)
        self.assertEqual(repeats.split, None)

        # repeats = distributed ht.DNDarray (axis != a.split)
        repeats = ht.array([1, 2, 0], split=0)
        with self.assertWarns(Warning):
            result = ht.repeat(a, repeats, 2)
        comparison = np.repeat(a_np, repeats.numpy(), 2)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assertTrue((ht.array(comparison) == result).all())
        self.assertEqual(result.split, a.split)
        self.assertIsInstance(repeats, ht.DNDarray)
        self.assertEqual(repeats.split, 0)

        # repeats = distributed ht.DNDarray (axis == a.split)
        repeats = ht.array([1, 2], split=0)
        result = ht.repeat(a, repeats, 1)
        comparison = np.repeat(a_np, repeats.numpy(), 1)

        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.shape, comparison.shape)
        self.assertTrue((ht.array(comparison) == result).all())
        self.assertEqual(result.split, a.split)
        self.assertIsInstance(repeats, ht.DNDarray)
        self.assertEqual(repeats.split, 0)

    def test_reshape(self):
        # split = None
        a = ht.zeros((3, 4), device=self.device)
        result = ht.zeros((2, 6), device=self.device)
        reshaped = ht.reshape(a, (2, 6))

        self.assertEqual(reshaped.size, result.size)
        self.assertEqual(reshaped.shape, result.shape)
        self.assertTrue(ht.equal(reshaped, result))
        self.assertEqual(reshaped.device, result.device)

        b = ht.arange(4 * 5 * 6, dtype=ht.float64)
        # test *shape input
        reshaped = b.reshape(4, 5, 6)
        self.assertTrue(reshaped.gshape == (4, 5, 6))
        self.assertEqual(reshaped.dtype, b.dtype)
        self.assertEqual(reshaped.split, b.split)
        self.assertEqual(reshaped.device, b.device)
        # test new_split not None
        reshaped = b.reshape(4, 5, -1, new_split=-1)
        self.assertTrue(reshaped.gshape == (4, 5, 6))
        self.assertEqual(reshaped.dtype, b.dtype)
        self.assertEqual(reshaped.split, 2)
        self.assertEqual(reshaped.device, b.device)
        self.assertEqual(reshaped.balanced, b.is_balanced(force_check=True))
        # test shape types
        reshaped = b.reshape(ht.array([4, 5, 6], dtype=ht.int, device=self.device, split=None))
        self.assertTrue(reshaped.gshape == (4, 5, 6))
        self.assertEqual(reshaped.dtype, b.dtype)
        self.assertEqual(reshaped.split, b.split)
        self.assertEqual(reshaped.device, b.device)
        reshaped = b.reshape(ht.array([4, 5, 6], dtype=ht.int, device=self.device, split=0))
        self.assertTrue(reshaped.gshape == (4, 5, 6))
        self.assertEqual(reshaped.dtype, b.dtype)
        self.assertEqual(reshaped.split, b.split)
        self.assertEqual(reshaped.device, b.device)
        reshaped = b.reshape(
            torch.as_tensor([4, 5, 6], dtype=torch.int32, device=self.device.torch_device)
        )
        self.assertTrue(reshaped.gshape == (4, 5, 6))
        self.assertEqual(reshaped.dtype, b.dtype)
        self.assertEqual(reshaped.split, b.split)
        self.assertEqual(reshaped.device, b.device)
        reshaped = b.reshape(np.asarray([4, 5, 6], dtype=np.int32))
        self.assertTrue(reshaped.gshape == (4, 5, 6))
        self.assertEqual(reshaped.dtype, b.dtype)
        self.assertEqual(reshaped.split, b.split)
        self.assertEqual(reshaped.device, b.device)
        # shape = -1
        result = ht.zeros(12, device=self.device)
        reshaped = ht.reshape(a, -1)

        self.assertEqual(reshaped.size, result.size)
        self.assertEqual(reshaped.shape, result.shape)
        self.assertTrue(ht.equal(reshaped, result))
        self.assertEqual(reshaped.device, result.device)

        # 1-dim distributed vector
        a = ht.arange(8, dtype=ht.float64, split=0, device=self.device)
        result = ht.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=ht.float64, split=0)
        reshaped = ht.reshape(a, (2, 2, 2))

        self.assertEqual(reshaped.size, result.size)
        self.assertEqual(reshaped.shape, result.shape)
        self.assertTrue(ht.equal(reshaped, result))
        self.assertEqual(reshaped.device, result.device)

        a = ht.linspace(0, 14, 8, split=0, device=self.device)
        result = ht.array(
            [[0, 2, 4, 6], [8, 10, 12, 14]], dtype=ht.float32, split=0, device=self.device
        )
        reshaped = ht.reshape(a, (2, 4))

        self.assertEqual(reshaped.size, result.size)
        self.assertEqual(reshaped.shape, result.shape)
        self.assertTrue(ht.equal(reshaped, result))
        self.assertEqual(reshaped.device, result.device)

        # Matrix
        a = ht.zeros((4, 3), dtype=ht.int32, split=0, device=self.device)
        result = ht.zeros((3, 4), dtype=ht.int32, split=0, device=self.device)
        reshaped = ht.reshape(a, (3, 4))

        self.assertEqual(reshaped.size, result.size)
        self.assertEqual(reshaped.shape, result.shape)
        self.assertTrue(ht.equal(reshaped, result))
        self.assertEqual(reshaped.device, result.device)

        a = ht.arange(16, split=0, device=self.device)
        result = ht.array(
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], device=self.device
        )
        reshaped = a.reshape((4, 4))

        self.assertEqual(reshaped.size, result.size)
        self.assertEqual(reshaped.shape, result.shape)
        self.assertTrue(ht.equal(reshaped, result))
        self.assertEqual(reshaped.device, result.device)

        a = reshaped
        result = ht.array(
            [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]], split=0, device=self.device
        )
        reshaped = a.reshape((2, 8))

        self.assertEqual(reshaped.size, result.size)
        self.assertEqual(reshaped.shape, result.shape)
        self.assertTrue(ht.equal(reshaped, result))
        self.assertEqual(reshaped.device, result.device)

        a = ht.array(
            torch.arange(3 * 4 * 5, device=self.device.torch_device).reshape((3, 4, 5)), split=1
        )
        result = ht.array(
            torch.arange(4 * 5 * 3, device=self.device.torch_device).reshape((4, 5, 3)), split=1
        )
        reshaped = a.reshape((4, 5, 3))

        self.assertEqual(reshaped.size, result.size)
        self.assertEqual(reshaped.shape, result.shape)
        self.assertTrue(ht.equal(reshaped, result))
        self.assertEqual(reshaped.device, result.device)

        a = ht.array(
            torch.arange(6 * 4 * 8, device=self.device.torch_device).reshape([6, 4, 8]), split=2
        )
        result = ht.array(
            torch.arange(4 * 12 * 4, device=self.device.torch_device).reshape([4, 12, 4]), split=2
        )
        reshaped = ht.reshape(a, [4, 12, 4])
        self.assertEqual(reshaped.size, result.size)
        self.assertEqual(reshaped.shape, result.shape)
        self.assertTrue(ht.equal(reshaped, result))
        self.assertEqual(reshaped.device, result.device)

        a = ht.array(
            torch.arange(3 * 4 * 5, device=self.device.torch_device).reshape([3, 4, 5]), split=2
        )
        result = ht.array(
            torch.arange(4 * 5 * 3, device=self.device.torch_device).reshape([4, 5, 3]), split=1
        )
        reshaped = ht.reshape(a, [4, 5, 3], new_split=1)
        self.assertEqual(reshaped.size, result.size)
        self.assertEqual(reshaped.shape, result.shape)
        self.assertEqual(reshaped.split, 1)
        self.assertTrue(ht.equal(reshaped, result))
        self.assertEqual(reshaped.device, result.device)

        a = ht.array(
            torch.arange(3 * 4 * 5, device=self.device.torch_device).reshape([3, 4, 5]), split=1
        )
        result = ht.array(
            torch.arange(4 * 5 * 3, device=self.device.torch_device).reshape([4 * 5, 3]), split=0
        )
        reshaped = ht.reshape(a, [4 * 5, 3], new_split=0)
        self.assertEqual(reshaped.size, result.size)
        self.assertEqual(reshaped.shape, result.shape)
        self.assertEqual(reshaped.split, 0)
        self.assertTrue(ht.equal(reshaped, result))
        self.assertEqual(reshaped.device, result.device)

        a = ht.array(
            torch.arange(3 * 4 * 5, device=self.device.torch_device).reshape([3, 4, 5]), split=0
        )
        result = ht.array(
            torch.arange(4 * 5 * 3, device=self.device.torch_device).reshape([4, 5 * 3]), split=1
        )
        reshaped = ht.reshape(a, [4, 5 * 3], new_split=1)
        self.assertEqual(reshaped.size, result.size)
        self.assertEqual(reshaped.shape, result.shape)
        self.assertEqual(reshaped.split, 1)
        self.assertTrue(ht.equal(reshaped, result))
        self.assertEqual(reshaped.device, result.device)

        # unknown dimension
        a = ht.ones((4, 4, 4), split=0, device=self.device)

        result = ht.ones((64), split=0, device=self.device)
        reshaped = ht.reshape(a, -1)
        self.assertEqual(reshaped.size, result.size)
        self.assertEqual(reshaped.shape, result.shape)
        self.assertTrue(ht.equal(reshaped, result))
        self.assertEqual(reshaped.device, result.device)

        result = ht.ones((4, 16), split=0, device=self.device)
        reshaped = ht.reshape(a, (4, -1))
        self.assertEqual(reshaped.size, result.size)
        self.assertEqual(reshaped.shape, result.shape)
        self.assertTrue(ht.equal(reshaped, result))
        self.assertEqual(reshaped.device, result.device)

        result = ht.ones((2, 4, 8), split=0, device=self.device)
        reshaped = ht.reshape(a, (2, -1, 8))
        self.assertEqual(reshaped.size, result.size)
        self.assertEqual(reshaped.shape, result.shape)
        self.assertTrue(ht.equal(reshaped, result))
        self.assertEqual(reshaped.device, result.device)

        result = ht.ones((4, 2, 4, 2), split=0, device=self.device)
        reshaped = ht.reshape(a, (-1, 2, 4, 2))
        self.assertEqual(reshaped.size, result.size)
        self.assertEqual(reshaped.shape, result.shape)
        self.assertTrue(ht.equal(reshaped, result))
        self.assertEqual(reshaped.device, result.device)

        # booleans
        a = ht.arange(4, split=0, dtype=ht.bool, device=self.device)
        result = ht.array([[False, True], [True, True]], split=0, dtype=ht.bool, device=self.device)
        reshaped = a.reshape((2, 2))

        self.assertEqual(reshaped.size, result.size)
        self.assertEqual(reshaped.shape, result.shape)
        self.assertEqual(reshaped.device, result.device)
        self.assertTrue(ht.equal(reshaped, result))

        # exceptions
        with self.assertRaises(ValueError):
            ht.reshape(ht.zeros((4, 3)), (5, 7))
        with self.assertRaises(TypeError):
            ht.reshape("ht.zeros((4, 3)), (5, 7)", (2, 3))
        with self.assertRaises(TypeError):
            ht.reshape(ht.zeros((4, 3)), "(5, 7)")
        with self.assertRaises(ValueError):
            ht.reshape(ht.zeros((4, 3)), (-1, -1, 3))
        with self.assertRaises(ValueError):
            ht.reshape(ht.zeros((4, 3)), (5, -1))
        # with self.assertRaises(ValueError):   actually this works in numpy
        #    ht.reshape(ht.zeros((4, 3)), (3, -2))
        with self.assertRaises(TypeError):
            ht.reshape(ht.zeros((4, 3)), (3.4, 3.2))

    def test_roll(self):
        # no split
        # vector
        a = ht.arange(5)
        rolled = ht.roll(a, 1)
        compare = ht.array([4, 0, 1, 2, 3])

        self.assertEqual(rolled.device, a.device)
        self.assertEqual(rolled.size, a.size)
        self.assertEqual(rolled.dtype, a.dtype)
        self.assertEqual(rolled.split, a.split)
        self.assertTrue(ht.equal(rolled, compare))

        rolled = ht.roll(a, -1)
        compare = ht.array([1, 2, 3, 4, 0])

        self.assertEqual(rolled.device, a.device)
        self.assertEqual(rolled.size, a.size)
        self.assertEqual(rolled.dtype, a.dtype)
        self.assertEqual(rolled.split, a.split)
        self.assertTrue(ht.equal(rolled, compare))

        # matrix
        a = ht.arange(20.0).reshape((4, 5))

        rolled = ht.roll(a, -1)
        compare = torch.roll(a.larray, -1)
        self.assertEqual(rolled.device, a.device)
        self.assertEqual(rolled.size, a.size)
        self.assertEqual(rolled.dtype, a.dtype)
        self.assertEqual(rolled.split, a.split)
        self.assertTrue(torch.equal(rolled.larray, compare))

        rolled = ht.roll(a, 1, 0)
        compare = torch.roll(a.larray, 1, 0)
        self.assertEqual(rolled.device, a.device)
        self.assertEqual(rolled.size, a.size)
        self.assertEqual(rolled.dtype, a.dtype)
        self.assertEqual(rolled.split, a.split)
        self.assertTrue(torch.equal(rolled.larray, compare))

        rolled = ht.roll(a, -2, (0, 1))
        compare = np.roll(a.larray.cpu().numpy(), -2, (0, 1))
        self.assertEqual(rolled.device, a.device)
        self.assertEqual(rolled.size, a.size)
        self.assertEqual(rolled.dtype, a.dtype)
        self.assertEqual(rolled.split, a.split)
        self.assertTrue(np.array_equal(rolled.larray.cpu().numpy(), compare))

        rolled = ht.roll(a, (1, 2, 1), (0, 1, -2))
        compare = torch.roll(a.larray, (1, 2, 1), (0, 1, -2))
        self.assertEqual(rolled.device, a.device)
        self.assertEqual(rolled.size, a.size)
        self.assertEqual(rolled.dtype, a.dtype)
        self.assertEqual(rolled.split, a.split)
        self.assertTrue(torch.equal(rolled.larray, compare))

        # split
        # vector
        a = ht.arange(5, dtype=ht.uint8, split=0)
        rolled = ht.roll(a, 1)
        compare = ht.array([4, 0, 1, 2, 3], dtype=ht.uint8, split=0)

        self.assertEqual(rolled.device, a.device)
        self.assertEqual(rolled.size, a.size)
        self.assertEqual(rolled.dtype, a.dtype)
        self.assertEqual(rolled.split, a.split)
        self.assertTrue(ht.equal(rolled, compare))

        rolled = ht.roll(a, -1)
        compare = ht.array([1, 2, 3, 4, 0], ht.uint8, split=0)
        self.assertEqual(rolled.device, a.device)
        self.assertEqual(rolled.size, a.size)
        self.assertEqual(rolled.dtype, a.dtype)
        self.assertEqual(rolled.split, a.split)
        self.assertTrue(ht.equal(rolled, compare))

        # matrix
        a = ht.arange(20).reshape((4, 5), dtype=ht.int16, new_split=0)

        rolled = ht.roll(a, -1)
        compare = np.roll(a.numpy(), -1)
        self.assertEqual(rolled.device, a.device)
        self.assertEqual(rolled.size, a.size)
        self.assertEqual(rolled.dtype, a.dtype)
        self.assertEqual(rolled.split, a.split)
        self.assertTrue(np.array_equal(rolled.numpy(), compare))

        rolled = ht.roll(a, 1, 0)
        compare = np.roll(a.numpy(), 1, 0)
        self.assertEqual(rolled.device, a.device)
        self.assertEqual(rolled.size, a.size)
        self.assertEqual(rolled.dtype, a.dtype)
        self.assertEqual(rolled.split, a.split)
        self.assertTrue(np.array_equal(rolled.numpy(), compare))

        rolled = ht.roll(a, -2, (0, 1))
        compare = np.roll(a.numpy(), -2, (0, 1))
        self.assertEqual(rolled.device, a.device)
        self.assertEqual(rolled.size, a.size)
        self.assertEqual(rolled.dtype, a.dtype)
        self.assertEqual(rolled.split, a.split)
        self.assertTrue(np.array_equal(rolled.numpy(), compare))

        rolled = ht.roll(a, (1, 2, 1), (0, 1, -2))
        compare = np.roll(a.numpy(), (1, 2, 1), (0, 1, -2))
        self.assertEqual(rolled.device, a.device)
        self.assertEqual(rolled.size, a.size)
        self.assertEqual(rolled.dtype, a.dtype)
        self.assertEqual(rolled.split, a.split)
        self.assertTrue(np.array_equal(rolled.numpy(), compare))

        a = ht.arange(20, dtype=ht.complex64).reshape((4, 5), new_split=1)

        rolled = ht.roll(a, -1)
        compare = np.roll(a.numpy(), -1)
        self.assertEqual(rolled.device, a.device)
        self.assertEqual(rolled.size, a.size)
        self.assertEqual(rolled.dtype, a.dtype)
        self.assertEqual(rolled.split, a.split)
        self.assertTrue(np.array_equal(rolled.numpy(), compare))

        rolled = ht.roll(a, 1, 0)
        compare = np.roll(a.numpy(), 1, 0)
        self.assertEqual(rolled.device, a.device)
        self.assertEqual(rolled.size, a.size)
        self.assertEqual(rolled.dtype, a.dtype)
        self.assertEqual(rolled.split, a.split)
        self.assertTrue(np.array_equal(rolled.numpy(), compare))

        rolled = ht.roll(a, -2, [0, 1])
        compare = np.roll(a.numpy(), -2, [0, 1])
        self.assertEqual(rolled.device, a.device)
        self.assertEqual(rolled.size, a.size)
        self.assertEqual(rolled.dtype, a.dtype)
        self.assertEqual(rolled.split, a.split)
        self.assertTrue(np.array_equal(rolled.numpy(), compare))

        rolled = ht.roll(a, [1, 2, 1], [0, 1, -2])
        compare = np.roll(a.numpy(), [1, 2, 1], [0, 1, -2])
        self.assertEqual(rolled.device, a.device)
        self.assertEqual(rolled.size, a.size)
        self.assertEqual(rolled.dtype, a.dtype)
        self.assertEqual(rolled.split, a.split)
        self.assertTrue(np.array_equal(rolled.numpy(), compare))

        # added 3D test, only a quick test for functionality
        a = ht.arange(4 * 5 * 6, dtype=ht.complex64).reshape((4, 5, 6), new_split=2)

        rolled = ht.roll(a, -1)
        compare = np.roll(a.numpy(), -1)
        self.assertEqual(rolled.device, a.device)
        self.assertEqual(rolled.size, a.size)
        self.assertEqual(rolled.dtype, a.dtype)
        self.assertEqual(rolled.split, a.split)
        self.assertTrue(np.array_equal(rolled.numpy(), compare))

        rolled = ht.roll(a, 1, 0)
        compare = np.roll(a.numpy(), 1, 0)
        self.assertEqual(rolled.device, a.device)
        self.assertEqual(rolled.size, a.size)
        self.assertEqual(rolled.dtype, a.dtype)
        self.assertEqual(rolled.split, a.split)
        self.assertTrue(np.array_equal(rolled.numpy(), compare))

        rolled = ht.roll(a, -2, [0, 1])
        compare = np.roll(a.numpy(), -2, [0, 1])
        self.assertEqual(rolled.device, a.device)
        self.assertEqual(rolled.size, a.size)
        self.assertEqual(rolled.dtype, a.dtype)
        self.assertEqual(rolled.split, a.split)
        self.assertTrue(np.array_equal(rolled.numpy(), compare))

        rolled = ht.roll(a, [1, 2, 1], [0, 1, -2])
        compare = np.roll(a.numpy(), [1, 2, 1], [0, 1, -2])
        self.assertEqual(rolled.device, a.device)
        self.assertEqual(rolled.size, a.size)
        self.assertEqual(rolled.dtype, a.dtype)
        self.assertEqual(rolled.split, a.split)
        self.assertTrue(np.array_equal(rolled.numpy(), compare))

        with self.assertRaises(TypeError):
            ht.roll(a, 1.0, 0)
        with self.assertRaises(TypeError):
            ht.roll(a, 1, 1.0)
        with self.assertRaises(TypeError):
            ht.roll(a, 1, (1.0, 0.0))
        with self.assertRaises(TypeError):
            ht.roll(a, (-1, 1), 0.0)
        with self.assertRaises(TypeError):
            ht.roll(a, (-1.0, 1.0), (0, 0))
        with self.assertRaises(ValueError):
            ht.roll(a, [1, 1, 1], [0, 0])

    def test_rot90(self):
        size = ht.MPI_WORLD.size
        m = ht.arange(size**3, dtype=ht.int).reshape((size, size, size))

        self.assertTrue(ht.equal(ht.rot90(m, 0), m))
        self.assertTrue(ht.equal(ht.rot90(m, 4), m))
        self.assertTrue(ht.equal(ht.rot90(ht.rot90(m, 1), 1, (1, 0)), m))

        a = ht.resplit(m, 0)

        self.assertTrue(ht.equal(ht.rot90(a, 0), a))
        self.assertTrue(ht.equal(ht.rot90(a), ht.resplit(ht.rot90(m), 1)))
        self.assertTrue(ht.equal(ht.rot90(a, 2), ht.resplit(ht.rot90(m, 2), 0)))
        self.assertTrue(ht.equal(ht.rot90(a, 3, (1, 2)), ht.resplit(ht.rot90(m, 3, (1, 2)), 0)))

        m = ht.arange(size**3, dtype=ht.float).reshape((size, size, size))
        a = ht.resplit(m, 1)

        self.assertTrue(ht.equal(ht.rot90(a, 0), a))
        self.assertTrue(ht.equal(ht.rot90(a), ht.resplit(ht.rot90(m), 0)))
        self.assertTrue(ht.equal(ht.rot90(a, 2), ht.resplit(ht.rot90(m, 2), 1)))
        self.assertTrue(ht.equal(ht.rot90(a, 3, (1, 2)), ht.resplit(ht.rot90(m, 3, (1, 2)), 2)))

        a = ht.resplit(m, 2)

        self.assertTrue(ht.equal(ht.rot90(a, 0), a))
        self.assertTrue(ht.equal(ht.rot90(a), ht.resplit(ht.rot90(m), 2)))
        self.assertTrue(ht.equal(ht.rot90(a, 2), ht.resplit(ht.rot90(m, 2), 2)))
        self.assertTrue(ht.equal(ht.rot90(a, 3, (1, 2)), ht.resplit(ht.rot90(m, 3, (1, 2)), 1)))

        with self.assertRaises(ValueError):
            ht.rot90(ht.ones((2, 3)), 1, (0, 1, 2))
        with self.assertRaises(TypeError):
            ht.rot90(torch.tensor((2, 3)))
        with self.assertRaises(ValueError):
            ht.rot90(ht.zeros((2, 2)), 1, (0, 0))
        with self.assertRaises(ValueError):
            ht.rot90(ht.zeros((2, 2)), 1, (-3, 1))
        with self.assertRaises(ValueError):
            ht.rot90(ht.zeros((2, 2)), 1, (4, 1))
        with self.assertRaises(ValueError):
            ht.rot90(ht.zeros((2, 2)), 1, (0, -2))
        with self.assertRaises(ValueError):
            ht.rot90(ht.zeros((2, 2)), 1, (0, 3))
        with self.assertRaises(TypeError):
            ht.rot90(ht.zeros((2, 3)), "k", (0, 1))

    def test_row_stack(self):
        # test local row_stack, 2-D arrays
        a = np.arange(10, dtype=np.float32).reshape(2, 5)
        b = np.arange(15, dtype=np.float32).reshape(3, 5)
        np_rstack = np.row_stack((a, b))
        ht_a = ht.array(a)
        ht_b = ht.array(b)
        ht_rstack = ht.row_stack((ht_a, ht_b))
        self.assertTrue((np_rstack == ht_rstack.numpy()).all())

        # 2-D and 1-D arrays
        c = np.arange(5, dtype=np.float32)
        np_rstack = np.row_stack((a, b, c))
        ht_c = ht.array(c)
        ht_rstack = ht.row_stack((ht_a, ht_b, ht_c))
        self.assertTrue((np_rstack == ht_rstack.numpy()).all())

        # 2-D and 1-D arrays, distributed
        c = np.arange(5, dtype=np.float32)
        np_rstack = np.row_stack((a, b, c))
        ht_a = ht.array(a, split=0)
        ht_b = ht.array(b, split=0)
        ht_c = ht.array(c, split=0)
        ht_rstack = ht.row_stack((ht_a, ht_b, ht_c))
        self.assertTrue((ht_rstack.numpy() == np_rstack).all())
        self.assertTrue(ht_rstack.split == 0)

        # 1-D arrays, distributed, different dtypes
        d = np.arange(10).astype(np.float32)
        e = np.arange(10)
        np_rstack = np.row_stack((d, e))
        ht_d = ht.array(d, split=0)
        ht_e = ht.array(e, split=0)
        ht_rstack = ht.row_stack((ht_d, ht_e))
        self.assertTrue((ht_rstack.numpy() == np_rstack).all())
        self.assertTrue(ht_rstack.dtype == ht.float32)
        self.assertTrue(ht_rstack.split == 1)

        # test exceptions
        f = ht.random.randn(4, 5, 2, split=1)
        with self.assertRaises(ValueError):
            ht.row_stack((a, b, f))

    def test_shape(self):
        x = ht.random.randn(3, 4, 5, split=2)
        self.assertEqual(ht.shape(x), (3, 4, 5))
        self.assertEqual(ht.shape(x), x.shape)

        # test exceptions
        x = torch.randn(3, 4, 5)
        with self.assertRaises(TypeError):
            ht.shape(x)

    def test_sort(self):
        size = ht.MPI_WORLD.size
        rank = ht.MPI_WORLD.rank
        tensor = (
            torch.arange(size, device=self.device.torch_device).repeat(size).reshape(size, size)
        )

        data = ht.array(tensor, split=None)
        result, result_indices = ht.sort(data, axis=0, descending=True)
        expected, exp_indices = torch.sort(tensor, dim=0, descending=True)
        self.assertTrue(torch.equal(result.larray, expected))
        self.assertTrue(torch.equal(result_indices.larray, exp_indices.int()))

        result, result_indices = ht.sort(data, axis=1, descending=True)
        expected, exp_indices = torch.sort(tensor, dim=1, descending=True)
        self.assertTrue(torch.equal(result.larray, expected))
        self.assertTrue(torch.equal(result_indices.larray, exp_indices.int()))

        data = ht.array(tensor, split=0)

        exp_axis_zero = torch.arange(size, device=self.device.torch_device).reshape(1, size)
        exp_indices = torch.tensor([[rank] * size], device=self.device.torch_device)
        result, result_indices = ht.sort(data, descending=True, axis=0)
        self.assertTrue(torch.equal(result.larray, exp_axis_zero))
        self.assertTrue(torch.equal(result_indices.larray, exp_indices.int()))

        exp_axis_one, exp_indices = (
            torch.arange(size, device=self.device.torch_device)
            .reshape(1, size)
            .sort(dim=1, descending=True)
        )
        result, result_indices = ht.sort(data, descending=True, axis=1)
        self.assertTrue(torch.equal(result.larray, exp_axis_one))
        self.assertTrue(torch.equal(result_indices.larray, exp_indices.int()))

        result1 = ht.sort(data, axis=1, descending=True)
        result2 = ht.sort(data, descending=True)
        self.assertTrue(ht.equal(result1[0], result2[0]))
        self.assertTrue(ht.equal(result1[1], result2[1]))

        data = ht.array(tensor, split=1)

        exp_axis_zero = (
            torch.tensor(rank, device=self.device.torch_device).repeat(size).reshape(size, 1)
        )
        indices_axis_zero = torch.arange(
            size, dtype=torch.int64, device=self.device.torch_device
        ).reshape(size, 1)
        result, result_indices = ht.sort(data, axis=0, descending=True)
        self.assertTrue(torch.equal(result.larray, exp_axis_zero))
        # comparison value is only true on CPU
        if result_indices.larray.is_cuda is False:
            self.assertTrue(torch.equal(result_indices.larray, indices_axis_zero.int()))

        exp_axis_one = (
            torch.tensor(size - rank - 1, device=self.device.torch_device)
            .repeat(size)
            .reshape(size, 1)
        )
        result, result_indices = ht.sort(data, descending=True, axis=1)
        self.assertTrue(torch.equal(result.larray, exp_axis_one))
        self.assertTrue(torch.equal(result_indices.larray, exp_axis_one.int()))

        tensor = torch.tensor(
            [
                [[2, 8, 5], [7, 2, 3]],
                [[6, 5, 2], [1, 8, 7]],
                [[9, 3, 0], [1, 2, 4]],
                [[8, 4, 7], [0, 8, 9]],
            ],
            dtype=torch.int32,
            device=self.device.torch_device,
        )

        data = ht.array(tensor, split=0)
        exp_axis_zero = torch.tensor(
            [[2, 3, 0], [0, 2, 3]], dtype=torch.int32, device=self.device.torch_device
        )
        if torch.cuda.is_available() and data.device == ht.gpu and size < 4:
            indices_axis_zero = torch.tensor(
                [[0, 2, 2], [3, 2, 0]], dtype=torch.int32, device=self.device.torch_device
            )
        else:
            indices_axis_zero = torch.tensor(
                [[0, 2, 2], [3, 0, 0]], dtype=torch.int32, device=self.device.torch_device
            )
        result, result_indices = ht.sort(data, axis=0)
        first = result[0].larray
        first_indices = result_indices[0].larray
        if rank == 0:
            self.assertTrue(torch.equal(first, exp_axis_zero))
            self.assertTrue(torch.equal(first_indices, indices_axis_zero))

        data = ht.array(tensor, split=1)
        exp_axis_one = torch.tensor([[2, 2, 3]], dtype=torch.int32, device=self.device.torch_device)
        indices_axis_one = torch.tensor(
            [[0, 1, 1]], dtype=torch.int32, device=self.device.torch_device
        )
        result, result_indices = ht.sort(data, axis=1)
        first = result[0].larray[:1]
        first_indices = result_indices[0].larray[:1]
        if rank == 0:
            self.assertTrue(torch.equal(first, exp_axis_one))
            self.assertTrue(torch.equal(first_indices, indices_axis_one))

        data = ht.array(tensor, split=2)
        exp_axis_two = torch.tensor([[2], [2]], dtype=torch.int32, device=self.device.torch_device)
        indices_axis_two = torch.tensor(
            [[0], [1]], dtype=torch.int32, device=self.device.torch_device
        )
        result, result_indices = ht.sort(data, axis=2)
        first = result[0].larray[:, :1]
        first_indices = result_indices[0].larray[:, :1]
        if rank == 0:
            self.assertTrue(torch.equal(first, exp_axis_two))
            self.assertTrue(torch.equal(first_indices, indices_axis_two))
        #
        out = ht.empty_like(data)
        indices = ht.sort(data, axis=2, out=out)
        self.assertTrue(ht.equal(out, result))
        self.assertTrue(ht.equal(indices, result_indices))

        with self.assertRaises(ValueError):
            ht.sort(data, axis=3)
        with self.assertRaises(TypeError):
            ht.sort(data, axis="1")

        rank = ht.MPI_WORLD.rank
        ht.random.seed(1)
        data = ht.random.randn(100, 1, split=0)
        result, _ = ht.sort(data, axis=0)
        counts, _, _ = ht.get_comm().counts_displs_shape(data.gshape, axis=0)
        for i, c in enumerate(counts):
            for idx in range(c - 1):
                if rank == i:
                    self.assertTrue(torch.lt(result.larray[idx], result.larray[idx + 1]).all())

    def test_split(self):
        # ====================================
        # UNDISTRIBUTED CASE
        # ====================================
        # axis = 0
        # ====================================
        data_ht = ht.arange(24).reshape((2, 3, 4))
        data_np = data_ht.numpy()

        # indices_or_sections = int
        result = ht.split(data_ht, 2)
        comparison = np.split(data_np, 2)

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = tuple
        result = ht.split(data_ht, (0, 1))
        comparison = np.split(data_np, (0, 1))

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = list
        result = ht.split(data_ht, [0, 1])
        comparison = np.split(data_np, [0, 1])

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = undistributed DNDarray
        result = ht.split(data_ht, ht.array([0, 1]))
        comparison = np.split(data_np, np.array([0, 1]))

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = distributed DNDarray
        with self.assertWarns(Warning):
            result = ht.split(data_ht, ht.array([0, 1], split=0))
        comparison = np.split(data_np, np.array([0, 1]))

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # ====================================
        # axis != 0 (2 in this case)
        # ====================================
        # indices_or_sections = int
        result = ht.split(data_ht, 2, 2)
        comparison = np.split(data_np, 2, 2)

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = tuple
        result = ht.split(data_ht, (0, 1))
        comparison = np.split(data_np, (0, 1))

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # exceptions
        with self.assertRaises(TypeError):
            ht.split([1, 2, 3, 4], 2)
        with self.assertRaises(TypeError):
            ht.split(data_ht, "2")
        with self.assertRaises(TypeError):
            ht.split(data_ht, 2, "0")
        with self.assertRaises(ValueError):
            ht.split(data_ht, 2, -1)
        with self.assertRaises(ValueError):
            ht.split(data_ht, 2, 3)
        with self.assertRaises(ValueError):
            ht.split(data_ht, 5)
        with self.assertRaises(ValueError):
            ht.split(data_ht, [[0, 1]])

        # ====================================
        # DISTRIBUTED CASE
        # ====================================
        # axis == ary.split
        # ====================================
        data_ht = ht.arange(120, split=0).reshape((4, 5, 6))
        data_np = data_ht.numpy()

        # indices = int
        result = ht.split(data_ht, 2)
        comparison = np.split(data_np, 2)

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assertTrue((ht.array(comparison[i]) == result[i]).all())

        # larger example
        data_ht_large = ht.arange(160, split=0).reshape((8, 5, 4))
        data_np_large = data_ht_large.numpy()

        # indices = int
        result = ht.split(data_ht_large, 2)
        comparison = np.split(data_np_large, 2)

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assertTrue((ht.array(comparison[i]) == result[i]).all())

        # indices_or_sections = tuple
        result = ht.split(data_ht, (1, 3, 5))
        comparison = np.split(data_np, (1, 3, 5))

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = list
        result = ht.split(data_ht, [1, 3, 5])
        comparison = np.split(data_np, [1, 3, 5])

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = undistributed DNDarray
        result = ht.split(data_ht, ht.array([1, 3, 5]))
        comparison = np.split(data_np, np.array([1, 3, 5]))

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = distributed DNDarray
        with self.assertWarns(Warning):
            result = ht.split(data_ht, ht.array([1, 3, 5], split=0))
        comparison = np.split(data_np, np.array([1, 3, 5]))

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # ====================================
        # axis != ary.split
        # ====================================
        # indices_or_sections = int
        result = ht.split(data_ht, 2, 2)
        comparison = np.split(data_np, 2, 2)

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = list
        result = ht.split(data_ht, [3, 4, 6], 2)
        comparison = np.split(data_np, [3, 4, 6], 2)

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = undistributed DNDarray
        result = ht.split(data_ht, ht.array([3, 4, 6]), 2)
        comparison = np.split(data_np, np.array([3, 4, 6]), 2)

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = distributed DNDarray
        indices = ht.array([3, 4, 6], split=0)
        with self.assertWarns(Warning):
            result = ht.split(data_ht, indices, 2)
        comparison = np.split(data_np, np.array([3, 4, 6]), 2)

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

    def test_resplit(self):
        if ht.MPI_WORLD.size > 1:
            # resplitting with same axis, should leave everything unchanged
            shape = (ht.MPI_WORLD.size, ht.MPI_WORLD.size)
            data = ht.zeros(shape, split=None)
            data2 = ht.resplit(data, None)

            self.assertIsInstance(data2, ht.DNDarray)
            self.assertEqual(data2.shape, shape)
            self.assertEqual(data2.lshape, shape)
            self.assertEqual(data2.split, None)

            # resplitting with same axis, should leave everything unchanged
            shape = (ht.MPI_WORLD.size, ht.MPI_WORLD.size)
            data = ht.zeros(shape, split=1)
            data2 = ht.resplit(data, 1)

            self.assertIsInstance(data2, ht.DNDarray)
            self.assertEqual(data2.shape, shape)
            self.assertEqual(data2.lshape, (data.comm.size, 1))
            self.assertEqual(data2.split, 1)

            # resplitting a non-distributed DNDarray with split not None
            if ht.MPI_WORLD.size == 1:
                data = ht.zeros(10, 10, split=0)
                data2 = ht.resplit(data, 1)
                data3 = ht.resplit(data, None)
                self.assertTrue((data == data2).all())
                self.assertTrue((data == data3).all())
                self.assertEqual(data2.split, 1)
                self.assertTrue(data3.split is None)

            # splitting an unsplit tensor should result in slicing the tensor locally
            shape = (ht.MPI_WORLD.size, ht.MPI_WORLD.size)
            data = ht.zeros(shape)
            data2 = ht.resplit(data, 1)

            self.assertIsInstance(data2, ht.DNDarray)
            self.assertEqual(data2.shape, shape)
            self.assertEqual(data2.lshape, (data.comm.size, 1))
            self.assertEqual(data2.split, 1)

            # unsplitting, aka gathering a tensor
            shape = (ht.MPI_WORLD.size + 1, ht.MPI_WORLD.size)
            data = ht.ones(shape, split=0)
            data2 = ht.resplit(data, None)

            self.assertIsInstance(data2, ht.DNDarray)
            self.assertEqual(data2.shape, shape)
            self.assertEqual(data2.lshape, shape)
            self.assertEqual(data2.split, None)

            # assign and entirely new split axis
            shape = (ht.MPI_WORLD.size + 2, ht.MPI_WORLD.size + 1)
            data = ht.ones(shape, split=0)
            data2 = ht.resplit(data, 1)

            self.assertIsInstance(data2, ht.DNDarray)
            self.assertEqual(data2.shape, shape)
            self.assertEqual(data2.lshape[0], ht.MPI_WORLD.size + 2)
            self.assertTrue(data2.lshape[1] == 1 or data2.lshape[1] == 2)
            self.assertEqual(data2.split, 1)

            # test sorting order of resplit

            N = ht.MPI_WORLD.size
            reference_tensor = ht.zeros((N, N + 1, 2 * N))
            for n in range(N):
                for m in range(N + 1):
                    reference_tensor[n, m, :] = ht.arange(0, 2 * N) + m * 10 + n * 100

            # split along axis = 0
            resplit_tensor = ht.resplit(reference_tensor, axis=0)
            local_shape = (1, N + 1, 2 * N)
            local_tensor = reference_tensor[ht.MPI_WORLD.rank, :, :]
            self.assertEqual(resplit_tensor.lshape, local_shape)
            self.assertTrue((resplit_tensor.larray == local_tensor.larray).all())

            # unsplit
            unsplit_tensor = ht.resplit(resplit_tensor, axis=None)
            self.assertTrue((unsplit_tensor.larray == reference_tensor.larray).all())

            # split along axis = 1
            resplit_tensor = ht.resplit(unsplit_tensor, axis=1)
            if ht.MPI_WORLD.rank == 0:
                local_shape = (N, 2, 2 * N)
                local_tensor = reference_tensor[:, 0:2, :]
            else:
                local_shape = (N, 1, 2 * N)
                local_tensor = reference_tensor[:, ht.MPI_WORLD.rank + 1 : ht.MPI_WORLD.rank + 2, :]

            self.assertEqual(resplit_tensor.lshape, local_shape)
            self.assertTrue((resplit_tensor.larray == local_tensor.larray).all())

            # unsplit
            unsplit_tensor = ht.resplit(resplit_tensor, axis=None)
            self.assertTrue((unsplit_tensor.larray == reference_tensor.larray).all())

            # split along axis = 2
            resplit_tensor = ht.resplit(unsplit_tensor, axis=2)
            local_shape = (N, N + 1, 2)
            local_tensor = reference_tensor[:, :, 2 * ht.MPI_WORLD.rank : 2 * ht.MPI_WORLD.rank + 2]

            self.assertEqual(resplit_tensor.lshape, local_shape)
            self.assertTrue((resplit_tensor.larray == local_tensor.larray).all())

            # order tests for resplit
            for dims in range(3, 5):
                length = torch.tensor(
                    [i + 20 for i in range(dims)], device=self.device.torch_device
                )
                test = torch.arange(torch.prod(length)).reshape(length.tolist())
                for sp1 in range(dims):
                    for sp2 in range(dims):
                        if sp1 != sp2:
                            a = ht.array(test, split=sp1)
                            resplit_a = ht.resplit(a, axis=sp2)
                            self.assertTrue(ht.equal(resplit_a, ht.array(test, split=sp2)))
                            self.assertEqual(resplit_a.split, sp2)
                            self.assertEqual(resplit_a.dtype, a.dtype)
                            del a
                            del resplit_a

    def test_squeeze(self):
        torch.manual_seed(1)
        data = ht.random.randn(1, 4, 5, 1)

        # 4D local tensor, no axis
        result = ht.squeeze(data)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.float32)
        self.assertEqual(result.larray.dtype, torch.float32)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result.larray == data.larray.squeeze()).all())

        # 4D local tensor, major axis
        result = ht.squeeze(data, axis=0)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.float32)
        self.assertEqual(result.larray.dtype, torch.float32)
        self.assertEqual(result.shape, (4, 5, 1))
        self.assertEqual(result.lshape, (4, 5, 1))
        self.assertEqual(result.split, None)
        self.assertTrue((result.larray == data.larray.squeeze(0)).all())

        # 4D local tensor, minor axis
        result = ht.squeeze(data, axis=-1)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.float32)
        self.assertEqual(result.larray.dtype, torch.float32)
        self.assertEqual(result.shape, (1, 4, 5))
        self.assertEqual(result.lshape, (1, 4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result.larray == data.larray.squeeze(-1)).all())

        # 4D local tensor, tuple axis
        result = data.squeeze(axis=(0, -1))
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.float32)
        self.assertEqual(result.larray.dtype, torch.float32)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result.larray == data.larray.squeeze()).all())

        # 4D split tensor, along the axis
        data = ht.array(ht.random.randn(1, 4, 5, 1), split=1)
        result = ht.squeeze(data, axis=-1)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.float32)
        self.assertEqual(result.larray.dtype, torch.float32)
        self.assertEqual(result.shape, (1, 4, 5))
        self.assertEqual(result.split, 1)

        # 4D split tensor, axis = split
        data = ht.array(ht.random.randn(3, 1, 5, 6), split=1)
        result = ht.squeeze(data, axis=1)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.float32)
        self.assertEqual(result.larray.dtype, torch.float32)
        self.assertEqual(result.shape, (3, 5, 6))
        self.assertEqual(result.split, None)

        # 4D split tensor, axis = split = last dimension
        data = ht.array(ht.random.randn(3, 6, 5, 1), split=-1)
        result = ht.squeeze(data, axis=-1)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.float32)
        self.assertEqual(result.larray.dtype, torch.float32)
        self.assertEqual(result.shape, (3, 6, 5))
        self.assertEqual(result.split, None)

        # 3D split tensor, across the axis
        size = ht.MPI_WORLD.size
        data = ht.triu(ht.ones((1, size * 2, size), split=1), k=1)

        result = ht.squeeze(data, axis=0)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.float32)
        self.assertEqual(result.larray.dtype, torch.float32)
        self.assertEqual(result.shape, (size * 2, size))
        self.assertEqual(result.lshape, (2, size))
        self.assertEqual(result.split, 0)

        # check exceptions
        with self.assertRaises(TypeError):
            data.squeeze(axis=1.1)
        with self.assertRaises(TypeError):
            data.squeeze(axis="y")
        with self.assertRaises(ValueError):
            ht.squeeze(data, axis=-4)
        with self.assertRaises(ValueError):
            ht.squeeze(data, axis=1)

    def test_stack(self):
        a = np.arange(20, dtype=np.float32).reshape(5, 4)
        b = np.arange(20, 40, dtype=np.float32).reshape(5, 4)
        c = np.arange(40, 60, dtype=np.float32).reshape(5, 4)
        axis = 0
        d = np.stack((a, b, c), axis=axis)

        # test stack on non-distributed DNDarrays
        ht_a = ht.array(a)
        ht_b = ht.array(b)
        ht_c = ht.array(c)
        ht_d = ht.stack((ht_a, ht_b, ht_c), axis=axis)
        self.assertTrue(ht_d.shape == (3, 5, 4))
        self.assertTrue((d == ht_d.numpy()).all())

        # test stack on distributed DNDarrays, split/axis combinations
        axis = 1
        split = 0
        d = np.stack((a, b, c), axis=axis)
        ht_a_split = ht.array(a, split=split)
        ht_b_split = ht.array(b, split=split)
        ht_c_split = ht.array(c, split=split)
        ht_d_split = ht.stack((ht_a_split, ht_b_split, ht_c_split), axis=axis)
        self.assertTrue(ht_d_split.shape == (5, 3, 4))
        self.assertTrue(ht_d_split.split == split)
        self.assertTrue((d == ht_d_split.numpy()).all())

        axis = 1
        split = 1
        ht_a_split = ht.array(a, split=split)
        ht_b_split = ht.array(b, split=split)
        ht_c_split = ht.array(c, split=split)
        ht_d_split = ht.stack((ht_a_split, ht_b_split, ht_c_split), axis=axis)
        self.assertTrue(ht_d_split.shape == (5, 3, 4))
        self.assertTrue(ht_d_split.split == split + 1)
        self.assertTrue((d == ht_d_split.numpy()).all())

        # different dtypes
        axis = -1
        split = 0
        d = np.stack((a, b, c), axis=axis)
        ht_a_split = ht.array(a, dtype=ht.int32, split=split)
        ht_b_split = ht.array(b, split=split)
        ht_c_split = ht.array(c, split=split)
        ht_d_split = ht.stack((ht_a_split, ht_b_split, ht_c_split), axis=axis)
        self.assertTrue(ht_d_split.shape == (5, 4, 3))
        self.assertTrue(ht_d_split.dtype == ht.float32)
        self.assertTrue(ht_d_split.split == split)
        self.assertTrue((d == ht_d_split.numpy()).all())

        # test out buffer
        out = ht.empty((5, 4, 3), dtype=ht.float32, split=0)
        ht.stack((ht_a_split, ht_b_split, ht_c_split), axis=axis, out=out)
        self.assertTrue((out == ht_d_split).all())

        # test exceptions
        with self.assertRaises(TypeError):
            ht.stack((ht_a, b, ht_c))
        with self.assertRaises(TypeError):
            ht.stack((ht_a))
        with self.assertRaises(ValueError):
            ht.stack((ht_a,))
        ht_c_wrong_shape = ht.array(c.reshape(2, 10))
        with self.assertRaises(ValueError):
            ht.stack((ht_a, ht_b, ht_c_wrong_shape))
        if ht_a.comm.size > 1:
            ht_b_wrong_split = ht.array(b, split=1)
            with self.assertRaises(ValueError):
                ht.stack((ht_a_split, ht_b_wrong_split, ht_c_split))
            with self.assertRaises(ValueError):
                ht.stack((ht_a_split, ht_b.resplit(1), ht_c_split))
        out_wrong_type = torch.empty((3, 5, 4), dtype=torch.float32)
        with self.assertRaises(TypeError):
            ht.stack((ht_a_split, ht_b_split, ht_c_split), out=out_wrong_type)
        out_wrong_shape = ht.empty((2, 5, 4), dtype=ht.float32, split=1)
        with self.assertRaises(ValueError):
            ht.stack((ht_a_split, ht_b_split, ht_c_split), out=out_wrong_shape)
        out_wrong_split = ht.empty((3, 5, 4), dtype=ht.float32, split=0)
        with self.assertRaises(ValueError):
            ht.stack((ht_a_split, ht_b_split, ht_c_split), out=out_wrong_split)

    def test_swapaxes(self):
        x = ht.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
        swapped = ht.swapaxes(x, 0, 1)

        self.assertTrue(
            ht.equal(swapped, ht.array([[[0, 1], [4, 5]], [[2, 3], [6, 7]]], dtype=ht.int64))
        )

        with self.assertRaises(TypeError):
            ht.swapaxes(x, 4.9, "abc")

    def test_tile(self):
        # test local tile, tuple reps
        x = ht.arange(12).reshape((4, 3))
        reps = (2, 1)
        ht_tiled = ht.tile(x, reps)
        np_tiled = np.tile(x.numpy(), reps)
        self.assertTrue((np_tiled == ht_tiled.numpy()).all())
        self.assertTrue(ht_tiled.dtype is x.dtype)

        # test scalar DNDarray x
        x = ht.array(9.0)
        reps = (2, 1)
        ht_tiled = ht.tile(x, reps)
        np_tiled = np.tile(x.numpy(), reps)
        self.assertTrue((np_tiled == ht_tiled.numpy()).all())
        self.assertTrue(ht_tiled.dtype is x.dtype)

        # test scalar x
        x = 10
        reps = (2, 1)
        ht_tiled = ht.tile(x, reps)
        np_tiled = np.tile(x, reps)
        self.assertTrue((np_tiled == ht_tiled.numpy()).all())
        self.assertTrue(ht_tiled.dtype is ht.int64)

        # test distributed tile along split axis
        # len(reps) > x.ndim
        split = 1
        x = ht.random.randn(4, 3, split=split)
        reps = ht.random.randint(2, 10, size=(4,))
        tiled_along_split = ht.tile(x, reps)
        np_tiled_along_split = np.tile(x.numpy(), reps.tolist())
        self.assertTrue((tiled_along_split.numpy() == np_tiled_along_split).all())
        self.assertTrue(tiled_along_split.dtype is x.dtype)

        # test distributed tile along non-zero split axis
        # len(reps) > x.ndim
        split = 0
        x = ht.random.randn(4, 3, split=split)
        reps = ht.random.randint(2, 10, size=(4,))
        tiled_along_split = ht.tile(x, reps)
        np_tiled_along_split = np.tile(x.numpy(), reps.numpy())
        self.assertTrue((tiled_along_split.numpy() == np_tiled_along_split).all())
        self.assertTrue(tiled_along_split.dtype is x.dtype)

        # test distributed tile() on imbalanced DNDarray
        x = ht.random.randn(100, split=0)
        x = x[ht.where(x > 0)]
        reps = 5
        imbalanced_tiled_along_split = ht.tile(x, reps)
        np_imbalanced_tiled_along_split = np.tile(x.numpy(), reps)
        self.assertTrue(
            (imbalanced_tiled_along_split.numpy() == np_imbalanced_tiled_along_split).all()
        )
        self.assertTrue(imbalanced_tiled_along_split.dtype is x.dtype)
        self.assertTrue(imbalanced_tiled_along_split.is_balanced(force_check=True))

        # test tile along non-split axis
        # len(reps) < x.ndim
        np_x = np.random.randn(4, 5, 3, 10).astype(np.float32)
        split = 1
        x = ht.array(np_x, dtype=ht.float32, split=split)
        reps = (2, 2)
        tiled_along_non_split = ht.tile(x, reps)
        np_tiled_along_non_split = np.tile(np_x, reps)
        _, _, global_slice = tiled_along_non_split.comm.chunk(
            tiled_along_non_split.shape, tiled_along_non_split.split
        )
        self.assertTrue(
            (
                tiled_along_non_split.larray.cpu().numpy() == np_tiled_along_non_split[global_slice]
            ).all()
        )
        self.assertTrue(tiled_along_non_split.dtype is x.dtype)

        # test tile along split axis
        # len(reps) = x.ndim
        split = 1
        x = ht.random.randn(3, 3, dtype=ht.float64, split=split)
        reps = (2, 3)
        tiled_along_split = ht.tile(x, reps)
        np_tiled_along_split = np.tile(x.numpy(), reps)
        self.assertTrue((tiled_along_split.numpy() == np_tiled_along_split).all())
        self.assertTrue(tiled_along_split.dtype is x.dtype)

        # test exceptions
        float_reps = (1, 2, 2, 1.5)
        with self.assertRaises(TypeError):
            tiled_along_split = ht.tile(x, float_reps)
        arraylike_float_reps = torch.tensor(float_reps)
        with self.assertRaises(TypeError):
            tiled_along_split = ht.tile(x, arraylike_float_reps)

    def test_topk(self):
        size = ht.MPI_WORLD.size
        if size == 1:
            size = 4

        torch_array = torch.arange(size, dtype=torch.int32, device=self.device.torch_device).expand(
            size, size
        )
        split_zero = ht.array(torch_array, split=0)
        split_one = ht.array(torch_array, split=1)

        res, indcs = ht.topk(split_zero, 2, sorted=True)
        exp_zero = ht.array([[size - 1, size - 2] for i in range(size)], dtype=ht.int32, split=0)
        exp_zero_indcs = ht.array(
            [[size - 1, size - 2] for i in range(size)], dtype=ht.int64, split=0
        )
        self.assertTrue((res.larray == exp_zero.larray).all())
        self.assertTrue((indcs.larray == exp_zero.larray).all())
        self.assertTrue(indcs.larray.dtype == exp_zero_indcs.larray.dtype)

        res, indcs = ht.topk(split_one, 2, sorted=True)
        exp_one = ht.array([[size - 1, size - 2] for i in range(size)], dtype=ht.int32, split=1)
        exp_one_indcs = ht.array(
            [[size - 1, size - 2] for i in range(size)], dtype=ht.int64, split=1
        )
        self.assertTrue((res.larray == exp_one.larray).all())
        self.assertTrue((indcs.larray == exp_one_indcs.larray).all())
        self.assertTrue(indcs.larray.dtype == exp_one_indcs.larray.dtype)

        res, indcs = ht.topk(split_zero, 2, sorted=True, largest=False)
        exp_zero = ht.array([[0, 1] for i in range(size)], dtype=ht.int32, split=0)
        exp_zero_indcs = ht.array([[0, 1] for i in range(size)], dtype=ht.int64, split=0)
        self.assertTrue((res.larray == exp_zero.larray).all())
        self.assertTrue((indcs.larray == exp_zero.larray).all())
        self.assertTrue(indcs.larray.dtype == exp_zero_indcs.larray.dtype)

        exp_zero = ht.array([[0, 1] for i in range(size)], dtype=ht.int32, split=0)
        exp_zero_indcs = ht.array([[0, 1] for i in range(size)], dtype=ht.int64, split=0)
        out = (ht.empty_like(exp_zero), ht.empty_like(exp_zero_indcs))
        res, indcs = ht.topk(split_zero, 2, sorted=True, largest=False, out=out)

        self.assertTrue((res.larray == exp_zero.larray).all())
        self.assertTrue((indcs.larray == exp_zero.larray).all())
        self.assertTrue(indcs.larray.dtype == exp_zero_indcs.larray.dtype)

        self.assertTrue((out[0].larray == exp_zero.larray).all())
        self.assertTrue((out[1].larray == exp_zero.larray).all())
        self.assertTrue(out[1].larray.dtype == exp_zero_indcs.larray.dtype)

        torch_array = torch.arange(
            size, dtype=torch.float64, device=self.device.torch_device
        ).expand(size, size)
        split_zero = ht.array(torch_array, split=0)
        split_one = ht.array(torch_array, split=1)

        res, indcs = ht.topk(split_zero, 2, sorted=True)
        exp_zero = ht.array([[size - 1, size - 2] for i in range(size)], dtype=ht.float64, split=0)
        exp_zero_indcs = ht.array(
            [[size - 1, size - 2] for i in range(size)], dtype=ht.int64, split=0
        )
        self.assertTrue((res.larray == exp_zero.larray).all())
        self.assertTrue((indcs.larray == exp_zero_indcs.larray).all())
        self.assertTrue(indcs.larray.dtype == exp_zero_indcs.larray.dtype)

        res, indcs = ht.topk(split_one, 2, sorted=True)
        exp_one = ht.array([[size - 1, size - 2] for i in range(size)], dtype=ht.float64, split=1)
        exp_one_indcs = ht.array(
            [[size - 1, size - 2] for i in range(size)], dtype=ht.int64, split=1
        )
        self.assertTrue((res.larray == exp_one.larray).all())
        self.assertTrue((indcs.larray == exp_one_indcs.larray).all())
        self.assertTrue(indcs.larray.dtype == exp_one_indcs.larray.dtype)

        with self.assertRaises(RuntimeError):
            exp_zero = ht.array([[0, 1] for i in range(size)], dtype=ht.int32, split=0)
            exp_zero_indcs = ht.array([[0, 1] for i in range(size)], dtype=ht.int64, split=0)
            out = (ht.empty_like(exp_zero), ht.empty_like(exp_zero_indcs))
            res, indcs = ht.topk(split_zero, 2, sorted=True, largest=False, out=out)
        with self.assertRaises(RuntimeError):
            exp_zero = ht.array([[0, 1] for i in range(size)], dtype=ht.float64, split=0)
            exp_zero_indcs = ht.array([[0, 1] for i in range(size)], dtype=ht.int16, split=0)
            out = (ht.empty_like(exp_zero), ht.empty_like(exp_zero_indcs))
            res, indcs = ht.topk(split_zero, 2, sorted=True, largest=False, out=out)

    def test_unique(self):
        size = ht.MPI_WORLD.size
        rank = ht.MPI_WORLD.rank
        torch_array = torch.arange(size, dtype=torch.int32, device=self.device.torch_device).expand(
            size, size
        )
        split_zero = ht.array(torch_array, split=0)

        exp_axis_none = ht.array([rank], dtype=ht.int32)
        res = split_zero.unique(sorted=True)
        self.assertTrue((res.larray == exp_axis_none.larray).all())

        exp_axis_zero = ht.arange(size, dtype=ht.int32).expand_dims(0)
        res = ht.unique(split_zero, sorted=True, axis=0)
        self.assertTrue((res.larray == exp_axis_zero.larray).all())

        exp_axis_one = ht.array([rank], dtype=ht.int32).expand_dims(0)
        split_zero_transposed = ht.array(torch_array.transpose(0, 1), split=0)
        res = ht.unique(split_zero_transposed, sorted=False, axis=1)
        self.assertTrue((res.larray == exp_axis_one.larray).all())

        split_one = ht.array(torch_array, dtype=ht.int32, split=1)

        exp_axis_none = ht.arange(size, dtype=ht.int32)
        res = ht.unique(split_one, sorted=True)
        self.assertTrue((res.larray == exp_axis_none.larray).all())

        exp_axis_zero = ht.array([rank], dtype=ht.int32).expand_dims(0)
        res = ht.unique(split_one, sorted=False, axis=0)
        self.assertTrue((res.larray == exp_axis_zero.larray).all())

        exp_axis_one = ht.array([rank] * size, dtype=ht.int32).expand_dims(1)
        res = ht.unique(split_one, sorted=True, axis=1)
        self.assertTrue((res.larray == exp_axis_one.larray).all())

        torch_array = torch.tensor(
            [[1, 2], [2, 3], [1, 2], [2, 3], [1, 2]],
            dtype=torch.int32,
            device=self.device.torch_device,
        )
        data = ht.array(torch_array, split=0)

        res, inv = ht.unique(data, return_inverse=True, axis=0)
        _, exp_inv = torch_array.unique(dim=0, return_inverse=True, sorted=True)
        self.assertTrue(torch.equal(inv, exp_inv.to(dtype=inv.dtype)))

        res, inv = ht.unique(data, return_inverse=True, axis=1)
        _, exp_inv = torch_array.unique(dim=1, return_inverse=True, sorted=True)
        self.assertTrue(torch.equal(inv, exp_inv.to(dtype=inv.dtype)))

        torch_array = torch.tensor(
            [[1, 1, 2], [1, 2, 2], [2, 1, 2], [1, 3, 2], [0, 1, 2]],
            dtype=torch.int32,
            device=self.device.torch_device,
        )
        exp_res, exp_inv = torch_array.unique(return_inverse=True, sorted=True)

        data_split_none = ht.array(torch_array)
        res = ht.unique(data_split_none, sorted=True)
        self.assertIsInstance(res, ht.DNDarray)
        self.assertEqual(res.split, None)
        self.assertEqual(res.dtype, data_split_none.dtype)
        self.assertEqual(res.device, data_split_none.device)
        res, inv = ht.unique(data_split_none, return_inverse=True, sorted=True)
        self.assertIsInstance(inv, ht.DNDarray)
        self.assertEqual(inv.split, None)
        self.assertEqual(inv.dtype, data_split_none.dtype)
        self.assertEqual(inv.device, data_split_none.device)
        self.assertTrue(torch.equal(inv.larray, exp_inv.int()))

        data_split_zero = ht.array(torch_array, split=0)
        res, inv = ht.unique(data_split_zero, return_inverse=True, sorted=True)
        self.assertTrue(torch.equal(inv, exp_inv.to(dtype=inv.dtype)))

    def test_vsplit(self):
        # for further testing, see test_split
        data_ht = ht.arange(24).reshape((4, 3, 2))
        data_np = data_ht.numpy()

        # indices_or_sections = int
        result = ht.vsplit(data_ht, 2)
        comparison = np.vsplit(data_np, 2)

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = tuple
        result = ht.vsplit(data_ht, (0, 1))
        comparison = np.vsplit(data_np, (0, 1))

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = list
        result = ht.vsplit(data_ht, [0, 1])
        comparison = np.vsplit(data_np, [0, 1])

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = undistributed DNDarray
        result = ht.vsplit(data_ht, ht.array([0, 1]))
        comparison = np.vsplit(data_np, np.array([0, 1]))

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

        # indices_or_sections = distributed DNDarray
        with self.assertWarns(Warning):
            result = ht.vsplit(data_ht, ht.array([0, 1], split=0))
        comparison = np.vsplit(data_np, np.array([0, 1]))

        self.assertTrue(len(result) == len(comparison))

        for i in range(len(result)):
            self.assertIsInstance(result[i], ht.DNDarray)
            self.assert_array_equal(result[i], comparison[i])

    def test_vstack(self):
        # cases to test:
        # MM===================================
        # NN,
        a = ht.ones((10, 12), split=None)
        b = ht.ones((10, 12), split=None)
        res = ht.vstack((a, b))
        self.assertEqual(res.shape, (20, 12))
        # 11,
        a = ht.ones((10, 12), split=1)
        b = ht.ones((10, 12), split=1)
        res = ht.vstack((a, b))
        self.assertEqual(res.shape, (20, 12))

        # VM===================================
        # NN,
        a = ht.ones((10,), split=None)
        b = ht.ones((12, 10), split=None)
        res = ht.vstack((a, b))
        self.assertEqual(res.shape, (13, 10))
        # 00
        a = ht.ones((10,), split=0)
        b = ht.ones((12, 10), split=0)
        res = ht.vstack((a, b))
        self.assertEqual(res.shape, (13, 10))

        # MV===================================
        # NN,
        a = ht.ones((12, 10), split=None)
        b = ht.ones((10,), split=None)
        res = ht.vstack((a, b))
        self.assertEqual(res.shape, (13, 10))
        # 00
        a = ht.ones((12, 10), split=0)
        b = ht.ones((10,), split=0)
        res = ht.vstack((a, b))
        self.assertEqual(res.shape, (13, 10))

        # VV===================================
        # NN,
        a = ht.ones((12,), split=None)
        b = ht.ones((12,), split=None)
        res = ht.vstack((a, b))
        self.assertEqual(res.shape, (2, 12))
        # 00
        a = ht.ones((12,), split=0)
        b = ht.ones((12,), split=0)
        res = ht.vstack((a, b))
        self.assertEqual(res.shape, (2, 12))

    def test_unfold(self):
        dtypes = (ht.int, ht.float)

        for dtype in dtypes:  # test with different datatypes
            # exceptions
            n = 1000
            x = ht.arange(n, dtype=dtype)
            with self.assertRaises(ValueError):  # size too small
                ht.unfold(x, 0, 1, 1)
            with self.assertRaises(ValueError):  # step too small
                ht.unfold(x, 0, 2, 0)
            x.resplit_(0)
            min_chunk_size = x.lshape_map[:, 0].min().item()
            if min_chunk_size + 2 > n:  # size too large
                with self.assertRaises(ValueError):
                    ht.unfold(x, 0, min_chunk_size + 2)
            else:  # size too large for chunk_size
                with self.assertRaises(RuntimeError):
                    ht.unfold(x, 0, min_chunk_size + 2)
            with self.assertRaises(ValueError):  # size too large
                ht.unfold(x, 0, n + 1, 1)
            ht.unfold(
                x, 0, min_chunk_size, min_chunk_size + 1
            )  # no fully local unfolds on some nodes

            # 2D sliding views
            n = 100

            x = torch.arange(n * n).reshape((n, n))
            y = ht.array(x, dtype)
            y.resplit_(0)

            u = x.unfold(0, 3, 3)
            u = u.unfold(1, 3, 3)
            u = ht.array(u)
            v = ht.unfold(y, 0, 3, 3)
            v = ht.unfold(v, 1, 3, 3)

            self.assertTrue(ht.equal(u, v))

            # more dimensions, different split axes
            n = 53
            k = 3  # number of dimensions
            shape = k * (n,)
            size = n**k

            x = torch.arange(size).reshape(shape)
            _y = x.clone().detach()
            y = ht.array(_y, dtype)

            for split in (None, *range(k)):
                y.resplit_(split)
                for size in range(2, 9):
                    for step in range(1, 21):
                        for dimension in range(k):
                            u = ht.array(x.unfold(dimension, size, step))
                            v = ht.unfold(y, dimension, size, step)

                            self.assertTrue(ht.equal(u, v))
