import numpy as np
import torch
import heat as ht
import os
from heat.core.tests.test_suites.basic_test import BasicTest

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


class TestManipulations(BasicTest):
    def test_concatenate(self):
        # cases to test:
        # Matrices / Vectors
        # s0    s1  axis
        # None None 0
        x = ht.zeros((16, 15), split=None, device=ht_device)
        y = ht.ones((16, 15), split=None, device=ht_device)
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
        x = ht.zeros((16, 15), split=None, device=ht_device)
        y = ht.ones((16, 15), split=0, device=ht_device)
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
        x = ht.zeros((16, 15), split=None, device=ht_device)
        y = ht.ones((16, 15), split=1, device=ht_device)
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
        x = ht.zeros((16, 15), split=None, device=ht_device)
        y = ht.ones((16, 15), split=1, device=ht_device)
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
        x = ht.zeros((16, 15), split=0, device=ht_device)
        y = ht.ones((16, 15), split=None, device=ht_device)
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
        x = ht.zeros((16, 15), split=1, device=ht_device)
        y = ht.ones((16, 15), split=None, device=ht_device)
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
        x = ht.zeros((16, 15), split=0, device=ht_device)
        y = ht.ones((16, 15), split=0, device=ht_device)
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
        x = ht.zeros((16, 15), split=1, device=ht_device)
        y = ht.ones((16, 15), split=1, device=ht_device)
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
        x = ht.zeros((16, 15, 14), split=2, device=ht_device)
        y = ht.ones((16, 15, 14), split=2, device=ht_device)
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
        y = ht.ones((16, 15, 14), split=None, device=ht_device)
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
        x = ht.zeros((16, 15, 14), split=None, device=ht_device)
        y = ht.ones((16, 15, 14), split=2, device=ht_device)
        # None 2 0
        res = ht.concatenate((x, y), axis=0)
        self.assertEqual(res.gshape, (32, 15, 14))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((32, 15, 14), res.split)
        lshape = [0, 0, 0]
        for i in range(3):
            lshape[i] = chk[i].stop - chk[i].start
        self.assertEqual(res.lshape, tuple(lshape))

        x = ht.zeros((16, 15, 14), split=None, device=ht_device)
        y = ht.ones((16, 15, 14), split=2, device=ht_device)
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
        x = ht.zeros((16,), split=None, device=ht_device)
        y = ht.ones((16,), split=None, device=ht_device)
        res = ht.concatenate((x, y), axis=0)
        self.assertEqual(res.gshape, (32,))
        self.assertEqual(res.dtype, ht.float)
        # None 0 0
        y = ht.ones((16,), split=0, device=ht_device)
        res = ht.concatenate((x, y), axis=0)
        self.assertEqual(res.gshape, (32,))
        self.assertEqual(res.dtype, ht.float)
        _, _, chk = res.comm.chunk((32,), res.split)
        lshape = [0]
        lshape[0] = chk[0].stop - chk[0].start
        self.assertEqual(res.lshape, tuple(lshape))

        # 0 0 0
        x = ht.ones((16,), split=0, dtype=ht.float64, device=ht_device)
        res = ht.concatenate((x, y), axis=0)
        self.assertEqual(res.gshape, (32,))
        self.assertEqual(res.dtype, ht.float64)
        _, _, chk = res.comm.chunk((32,), res.split)
        lshape = [0]
        lshape[0] = chk[0].stop - chk[0].start
        self.assertEqual(res.lshape, tuple(lshape))
        # 0 None 0
        x = ht.ones((16,), split=0, device=ht_device)
        y = ht.ones((16,), split=None, dtype=ht.int64, device=ht_device)
        res = ht.concatenate((x, y), axis=0)
        self.assertEqual(res.gshape, (32,))
        self.assertEqual(res.dtype, ht.float64)
        _, _, chk = res.comm.chunk((32,), res.split)
        lshape = [0]
        lshape[0] = chk[0].stop - chk[0].start
        self.assertEqual(res.lshape, tuple(lshape))

        # test raises
        with self.assertRaises(ValueError):
            ht.concatenate(
                (ht.zeros((6, 3, 5), device=ht_device), ht.zeros((4, 5, 1), device=ht_device))
            )
        with self.assertRaises(TypeError):
            ht.concatenate((x, "5"))
        with self.assertRaises(TypeError):
            ht.concatenate((x))
        with self.assertRaises(TypeError):
            ht.concatenate((x, x), axis=x)
        with self.assertRaises(RuntimeError):
            ht.concatenate((x, ht.zeros((2, 2), device=ht_device)), axis=0)
        with self.assertRaises(ValueError):
            ht.concatenate(
                (ht.zeros((12, 12), device=ht_device), ht.zeros((2, 2), device=ht_device)), axis=0
            )
        with self.assertRaises(RuntimeError):
            ht.concatenate(
                (
                    ht.zeros((2, 2), split=0, device=ht_device),
                    ht.zeros((2, 2), split=1, device=ht_device),
                ),
                axis=0,
            )

    def test_diag(self):
        size = ht.MPI_WORLD.size
        rank = ht.MPI_WORLD.rank

        data = torch.arange(size * 2, device=device)
        a = ht.array(data, device=ht_device)
        res = ht.diag(a)
        self.assertTrue(torch.equal(res._DNDarray__array, torch.diag(data)))

        res = ht.diag(a, offset=size)
        self.assertTrue(torch.equal(res._DNDarray__array, torch.diag(data, diagonal=size)))

        res = ht.diag(a, offset=-size)
        self.assertTrue(torch.equal(res._DNDarray__array, torch.diag(data, diagonal=-size)))

        a = ht.array(data, split=0, device=ht_device)
        res = ht.diag(a)
        self.assertEqual(res.split, a.split)
        self.assertEqual(res.shape, (size * 2, size * 2))
        self.assertEqual(res.lshape[res.split], 2)
        exp = torch.diag(data)
        for i in range(rank * 2, (rank + 1) * 2):
            self.assertTrue(torch.equal(res[i, i]._DNDarray__array, exp[i, i]))

        res = ht.diag(a, offset=size)
        self.assertEqual(res.split, a.split)
        self.assertEqual(res.shape, (size * 3, size * 3))
        self.assertEqual(res.lshape[res.split], 3)
        exp = torch.diag(data, diagonal=size)
        for i in range(rank * 3, min((rank + 1) * 3, a.shape[0])):
            self.assertTrue(torch.equal(res[i, i + size]._DNDarray__array, exp[i, i + size]))

        res = ht.diag(a, offset=-size)
        self.assertEqual(res.split, a.split)
        self.assertEqual(res.shape, (size * 3, size * 3))
        self.assertEqual(res.lshape[res.split], 3)
        exp = torch.diag(data, diagonal=-size)
        for i in range(max(size, rank * 3), (rank + 1) * 3):
            self.assertTrue(torch.equal(res[i, i - size]._DNDarray__array, exp[i, i - size]))

        self.assertTrue(ht.equal(ht.diag(ht.diag(a)), a))

        a = ht.random.rand(15, 20, 5, split=1, device=ht_device)
        res_1 = ht.diag(a)
        res_2 = ht.diagonal(a)
        self.assertTrue(ht.equal(res_1, res_2))

        with self.assertRaises(ValueError):
            ht.diag(data)

        with self.assertRaises(ValueError):
            ht.diag(a, offset=None)

        a = ht.arange(size, device=ht_device)
        with self.assertRaises(ValueError):
            ht.diag(a, offset="3")

        a = ht.empty([], device=ht_device)
        with self.assertRaises(ValueError):
            ht.diag(a)

        if rank == 0:
            data = torch.ones(size, dtype=torch.int32, device=device)
        else:
            data = torch.empty(0, dtype=torch.int32, device=device)
        a = ht.array(data, is_split=0, device=ht_device)
        res = ht.diag(a)
        self.assertTrue(
            torch.equal(
                res[rank, rank]._DNDarray__array, torch.tensor(1, dtype=torch.int32, device=device)
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
            (27,),
            heat_func=ht.diag,
            numpy_func=np.diag,
            heat_args={"offset": -3},
            numpy_args={"k": -3},
        )

    def test_diagonal(self):
        size = ht.MPI_WORLD.size
        rank = ht.MPI_WORLD.rank

        data = torch.arange(size, device=device).repeat(size).reshape(size, size)
        a = ht.array(data, device=ht_device)
        res = ht.diagonal(a)
        self.assertTrue(torch.equal(res._DNDarray__array, torch.arange(size, device=device)))
        self.assertEqual(res.split, None)

        a = ht.array(data, split=0, device=ht_device)
        res = ht.diagonal(a)
        self.assertTrue(torch.equal(res._DNDarray__array, torch.tensor([rank], device=device)))
        self.assertEqual(res.split, 0)

        a = ht.array(data, split=1, device=ht_device)
        res2 = ht.diagonal(a, dim1=1, dim2=0)
        self.assertTrue(ht.equal(res, res2))

        res = ht.diagonal(a)
        self.assertTrue(torch.equal(res._DNDarray__array, torch.tensor([rank], device=device)))
        self.assertEqual(res.split, 0)

        a = ht.array(data, split=0, device=ht_device)
        res2 = ht.diagonal(a, dim1=1, dim2=0)
        self.assertTrue(ht.equal(res, res2))

        data = torch.arange(size + 1, device=device).repeat(size + 1).reshape(size + 1, size + 1)
        a = ht.array(data, device=ht_device)
        res = ht.diagonal(a, offset=0)
        self.assertTrue(torch.equal(res._DNDarray__array, torch.arange(size + 1, device=device)))
        res = ht.diagonal(a, offset=1)
        self.assertTrue(torch.equal(res._DNDarray__array, torch.arange(1, size + 1, device=device)))
        res = ht.diagonal(a, offset=-1)
        self.assertTrue(torch.equal(res._DNDarray__array, torch.arange(0, size, device=device)))

        a = ht.array(data, split=0, device=ht_device)
        res = ht.diagonal(a, offset=1)
        res.balance_()
        self.assertTrue(torch.equal(res._DNDarray__array, torch.tensor([rank + 1], device=device)))
        res = ht.diagonal(a, offset=-1)
        res.balance_()
        self.assertTrue(torch.equal(res._DNDarray__array, torch.tensor([rank], device=device)))

        a = ht.array(data, split=1, device=ht_device)
        res = ht.diagonal(a, offset=1)
        res.balance_()
        self.assertTrue(torch.equal(res._DNDarray__array, torch.tensor([rank + 1], device=device)))
        res = ht.diagonal(a, offset=-1)
        res.balance_()
        self.assertTrue(torch.equal(res._DNDarray__array, torch.tensor([rank], device=device)))

        data = (
            torch.arange(size * 2 + 10, device=device)
            .repeat(size * 2 + 10)
            .reshape(size * 2 + 10, size * 2 + 10)
        )
        a = ht.array(data, device=ht_device)
        res = ht.diagonal(a, offset=10)
        self.assertTrue(
            torch.equal(res._DNDarray__array, torch.arange(10, 10 + size * 2, device=device))
        )
        res = ht.diagonal(a, offset=-10)
        self.assertTrue(torch.equal(res._DNDarray__array, torch.arange(0, size * 2, device=device)))

        a = ht.array(data, split=0, device=ht_device)
        res = ht.diagonal(a, offset=10)
        res.balance_()
        self.assertTrue(
            torch.equal(
                res._DNDarray__array, torch.tensor([10 + rank * 2, 11 + rank * 2], device=device)
            )
        )
        res = ht.diagonal(a, offset=-10)
        res.balance_()
        self.assertTrue(
            torch.equal(res._DNDarray__array, torch.tensor([rank * 2, 1 + rank * 2], device=device))
        )

        a = ht.array(data, split=1, device=ht_device)
        res = ht.diagonal(a, offset=10)
        res.balance_()
        self.assertTrue(
            torch.equal(
                res._DNDarray__array, torch.tensor([10 + rank * 2, 11 + rank * 2], device=device)
            )
        )
        res = ht.diagonal(a, offset=-10)
        res.balance_()
        self.assertTrue(
            torch.equal(res._DNDarray__array, torch.tensor([rank * 2, 1 + rank * 2], device=device))
        )

        data = (
            torch.arange(size + 1, device=device)
            .repeat((size + 1) * (size + 1))
            .reshape(size + 1, size + 1, size + 1)
        )
        a = ht.array(data, device=ht_device)
        res = ht.diagonal(a)
        self.assertTrue(
            torch.equal(
                res._DNDarray__array,
                torch.arange(size + 1, device=device)
                .repeat(size + 1)
                .reshape(size + 1, size + 1)
                .t(),
            )
        )
        res = ht.diagonal(a, offset=1)
        self.assertTrue(
            torch.equal(
                res._DNDarray__array,
                torch.arange(size + 1, device=device).repeat(size).reshape(size, size + 1).t(),
            )
        )
        res = ht.diagonal(a, offset=-1)
        self.assertTrue(
            torch.equal(
                res._DNDarray__array,
                torch.arange(size + 1, device=device).repeat(size).reshape(size, size + 1).t(),
            )
        )

        res = ht.diagonal(a, dim1=1, dim2=2)
        self.assertTrue(
            torch.equal(
                res._DNDarray__array,
                torch.arange(size + 1, device=device).repeat(size + 1).reshape(size + 1, size + 1),
            )
        )
        res = ht.diagonal(a, offset=1, dim1=1, dim2=2)
        self.assertTrue(
            torch.equal(
                res._DNDarray__array,
                torch.arange(1, size + 1, device=device).repeat(size + 1).reshape(size + 1, size),
            )
        )
        res = ht.diagonal(a, offset=-1, dim1=1, dim2=2)
        self.assertTrue(
            torch.equal(
                res._DNDarray__array,
                torch.arange(size, device=device).repeat(size + 1).reshape(size + 1, size),
            )
        )

        res = ht.diagonal(a, dim1=0, dim2=2)
        self.assertTrue(
            torch.equal(
                res._DNDarray__array,
                torch.arange(size + 1, device=device).repeat(size + 1).reshape(size + 1, size + 1),
            )
        )

        a = ht.array(data, split=0, device=ht_device)
        res = ht.diagonal(a, offset=1, dim1=0, dim2=1)
        res.balance_()
        self.assertTrue(
            torch.equal(
                res._DNDarray__array, torch.arange(size + 1, device=device).reshape(size + 1, 1)
            )
        )
        self.assertEqual(res.split, 1)

        res = ht.diagonal(a, offset=-1, dim1=0, dim2=1)
        res.balance_()
        self.assertTrue(
            torch.equal(
                res._DNDarray__array, torch.arange(size + 1, device=device).reshape(size + 1, 1)
            )
        )
        self.assertEqual(res.split, 1)

        res = ht.diagonal(a, offset=size + 1, dim1=0, dim2=1)
        res.balance_()
        self.assertTrue(
            torch.equal(
                res._DNDarray__array, torch.empty((size + 1, 0), dtype=torch.int64, device=device)
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

    def test_expand_dims(self):
        # vector data
        a = ht.arange(10, device=ht_device)
        b = ht.expand_dims(a, 0)

        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(len(b.shape), 2)

        self.assertEqual(b.shape[0], 1)
        self.assertEqual(b.shape[1], a.shape[0])

        self.assertEqual(b.lshape[0], 1)
        self.assertEqual(b.lshape[1], a.shape[0])

        self.assertIs(b.split, None)

        # vector data with out-of-bounds axis
        a = ht.arange(12, device=ht_device)
        b = a.expand_dims(1)

        self.assertIsInstance(b, ht.DNDarray)
        self.assertEqual(len(b.shape), 2)

        self.assertEqual(b.shape[0], a.shape[0])
        self.assertEqual(b.shape[1], 1)

        self.assertEqual(b.lshape[0], a.shape[0])
        self.assertEqual(b.lshape[1], 1)

        self.assertIs(b.split, None)

        # volume with intermediate axis
        a = ht.empty((3, 4, 5), device=ht_device)
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
        a = ht.empty((3, 4, 5), device=ht_device)
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
        a = ht.empty((3, 4, 5), split=1, device=ht_device)
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
        a = ht.empty((3, 4, 5), split=2, device=ht_device)
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
            ht.empty((3, 4, 5), device=ht_device).expand_dims("1")
        with self.assertRaises(ValueError):
            ht.empty((3, 4, 5), device=ht_device).expand_dims(4)
        with self.assertRaises(ValueError):
            ht.empty((3, 4, 5), device=ht_device).expand_dims(-5)

    def test_hstack(self):
        # cases to test:
        # MM===================================
        # NN,
        a = ht.ones((10, 12), split=None, device=ht_device)
        b = ht.ones((10, 12), split=None, device=ht_device)
        res = ht.hstack((a, b))
        self.assertEqual(res.shape, (10, 24))
        # 11,
        a = ht.ones((10, 12), split=1, device=ht_device)
        b = ht.ones((10, 12), split=1, device=ht_device)
        res = ht.hstack((a, b))
        self.assertEqual(res.shape, (10, 24))

        # VM===================================
        # NN,
        a = ht.ones((12,), split=None, device=ht_device)
        b = ht.ones((12, 10), split=None, device=ht_device)
        res = ht.hstack((a, b))
        self.assertEqual(res.shape, (12, 11))
        # 00
        a = ht.ones((12,), split=0, device=ht_device)
        b = ht.ones((12, 10), split=0, device=ht_device)
        res = ht.hstack((a, b))
        self.assertEqual(res.shape, (12, 11))

        # MV===================================
        # NN,
        a = ht.ones((12, 10), split=None, device=ht_device)
        b = ht.ones((12,), split=None, device=ht_device)
        res = ht.hstack((a, b))
        self.assertEqual(res.shape, (12, 11))
        # 00
        a = ht.ones((12, 10), split=0, device=ht_device)
        b = ht.ones((12,), split=0, device=ht_device)
        res = ht.hstack((a, b))
        self.assertEqual(res.shape, (12, 11))

        # VV===================================
        # NN,
        a = ht.ones((12,), split=None, device=ht_device)
        b = ht.ones((12,), split=None, device=ht_device)
        res = ht.hstack((a, b))
        self.assertEqual(res.shape, (24,))
        # 00
        a = ht.ones((12,), split=0, device=ht_device)
        b = ht.ones((12,), split=0, device=ht_device)
        res = ht.hstack((a, b))
        self.assertEqual(res.shape, (24,))

    def test_sort(self):
        size = ht.MPI_WORLD.size
        rank = ht.MPI_WORLD.rank
        tensor = torch.arange(size, device=device).repeat(size).reshape(size, size)

        data = ht.array(tensor, split=None, device=ht_device)
        result, result_indices = ht.sort(data, axis=0, descending=True)
        expected, exp_indices = torch.sort(tensor, dim=0, descending=True)
        self.assertTrue(torch.equal(result._DNDarray__array, expected))
        self.assertTrue(torch.equal(result_indices._DNDarray__array, exp_indices))

        result, result_indices = ht.sort(data, axis=1, descending=True)
        expected, exp_indices = torch.sort(tensor, dim=1, descending=True)
        self.assertTrue(torch.equal(result._DNDarray__array, expected))
        self.assertTrue(torch.equal(result_indices._DNDarray__array, exp_indices))

        data = ht.array(tensor, split=0, device=ht_device)

        exp_axis_zero = torch.arange(size, device=device).reshape(1, size)
        exp_indices = torch.tensor([[rank] * size], device=device)
        result, result_indices = ht.sort(data, descending=True, axis=0)
        self.assertTrue(torch.equal(result._DNDarray__array, exp_axis_zero))
        self.assertTrue(torch.equal(result_indices._DNDarray__array, exp_indices))

        exp_axis_one, exp_indices = (
            torch.arange(size, device=device).reshape(1, size).sort(dim=1, descending=True)
        )
        result, result_indices = ht.sort(data, descending=True, axis=1)
        self.assertTrue(torch.equal(result._DNDarray__array, exp_axis_one))
        self.assertTrue(torch.equal(result_indices._DNDarray__array, exp_indices))

        result1 = ht.sort(data, axis=1, descending=True)
        result2 = ht.sort(data, descending=True)
        self.assertTrue(ht.equal(result1[0], result2[0]))
        self.assertTrue(ht.equal(result1[1], result2[1]))

        data = ht.array(tensor, split=1, device=ht_device)

        exp_axis_zero = torch.tensor(rank, device=device).repeat(size).reshape(size, 1)
        indices_axis_zero = torch.arange(size, dtype=torch.int64, device=device).reshape(size, 1)
        result, result_indices = ht.sort(data, axis=0, descending=True)
        self.assertTrue(torch.equal(result._DNDarray__array, exp_axis_zero))
        # comparison value is only true on CPU
        if result_indices._DNDarray__array.is_cuda is False:
            self.assertTrue(torch.equal(result_indices._DNDarray__array, indices_axis_zero))

        exp_axis_one = torch.tensor(size - rank - 1, device=device).repeat(size).reshape(size, 1)
        result, result_indices = ht.sort(data, descending=True, axis=1)
        self.assertTrue(torch.equal(result._DNDarray__array, exp_axis_one))
        self.assertTrue(torch.equal(result_indices._DNDarray__array, exp_axis_one))

        tensor = torch.tensor(
            [
                [[2, 8, 5], [7, 2, 3]],
                [[6, 5, 2], [1, 8, 7]],
                [[9, 3, 0], [1, 2, 4]],
                [[8, 4, 7], [0, 8, 9]],
            ],
            dtype=torch.int32,
            device=device,
        )

        data = ht.array(tensor, split=0, device=ht_device)
        exp_axis_zero = torch.tensor([[2, 3, 0], [0, 2, 3]], dtype=torch.int32, device=device)
        if torch.cuda.is_available() and data.device == ht.gpu and size < 4:
            indices_axis_zero = torch.tensor(
                [[0, 2, 2], [3, 2, 0]], dtype=torch.int32, device=device
            )
        else:
            indices_axis_zero = torch.tensor(
                [[0, 2, 2], [3, 0, 0]], dtype=torch.int32, device=device
            )
        result, result_indices = ht.sort(data, axis=0)
        first = result[0]._DNDarray__array
        first_indices = result_indices[0]._DNDarray__array
        if rank == 0:
            self.assertTrue(torch.equal(first, exp_axis_zero))
            self.assertTrue(torch.equal(first_indices, indices_axis_zero))

        data = ht.array(tensor, split=1, device=ht_device)
        exp_axis_one = torch.tensor([[2, 2, 3]], dtype=torch.int32, device=device)
        indices_axis_one = torch.tensor([[0, 1, 1]], dtype=torch.int32, device=device)
        result, result_indices = ht.sort(data, axis=1)
        first = result[0]._DNDarray__array[:1]
        first_indices = result_indices[0]._DNDarray__array[:1]
        if rank == 0:
            self.assertTrue(torch.equal(first, exp_axis_one))
            self.assertTrue(torch.equal(first_indices, indices_axis_one))

        data = ht.array(tensor, split=2, device=ht_device)
        exp_axis_two = torch.tensor([[2], [2]], dtype=torch.int32, device=device)
        indices_axis_two = torch.tensor([[0], [1]], dtype=torch.int32, device=device)
        result, result_indices = ht.sort(data, axis=2)
        first = result[0]._DNDarray__array[:, :1]
        first_indices = result_indices[0]._DNDarray__array[:, :1]
        if rank == 0:
            self.assertTrue(torch.equal(first, exp_axis_two))
            self.assertTrue(torch.equal(first_indices, indices_axis_two))
        #
        out = ht.empty_like(data, device=ht_device)
        indices = ht.sort(data, axis=2, out=out)
        self.assertTrue(ht.equal(out, result))
        self.assertTrue(ht.equal(indices, result_indices))

        with self.assertRaises(ValueError):
            ht.sort(data, axis=3)
        with self.assertRaises(TypeError):
            ht.sort(data, axis="1")

        rank = ht.MPI_WORLD.rank
        data = ht.random.randn(100, 1, split=0, device=ht_device)
        result, _ = ht.sort(data, axis=0)
        counts, _, _ = ht.get_comm().counts_displs_shape(data.gshape, axis=0)
        for i, c in enumerate(counts):
            for idx in range(c - 1):
                if rank == i:
                    self.assertTrue(
                        torch.lt(
                            result._DNDarray__array[idx], result._DNDarray__array[idx + 1]
                        ).all()
                    )

    def test_squeeze(self):
        torch.manual_seed(1)
        data = ht.random.randn(1, 4, 5, 1, device=ht_device)

        # 4D local tensor, no axis
        result = ht.squeeze(data)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.float64)
        self.assertEqual(result._DNDarray__array.dtype, torch.float64)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == data._DNDarray__array.squeeze()).all())

        # 4D local tensor, major axis
        result = ht.squeeze(data, axis=0)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.float64)
        self.assertEqual(result._DNDarray__array.dtype, torch.float64)
        self.assertEqual(result.shape, (4, 5, 1))
        self.assertEqual(result.lshape, (4, 5, 1))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == data._DNDarray__array.squeeze(0)).all())

        # 4D local tensor, minor axis
        result = ht.squeeze(data, axis=-1)
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.float64)
        self.assertEqual(result._DNDarray__array.dtype, torch.float64)
        self.assertEqual(result.shape, (1, 4, 5))
        self.assertEqual(result.lshape, (1, 4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == data._DNDarray__array.squeeze(-1)).all())

        # 4D local tensor, tuple axis
        result = data.squeeze(axis=(0, -1))
        self.assertIsInstance(result, ht.DNDarray)
        self.assertEqual(result.dtype, ht.float64)
        self.assertEqual(result._DNDarray__array.dtype, torch.float64)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.lshape, (4, 5))
        self.assertEqual(result.split, None)
        self.assertTrue((result._DNDarray__array == data._DNDarray__array.squeeze()).all())

        # 4D split tensor, along the axis
        # TODO: reinstate this test of uneven dimensions distribution
        # after update to Allgatherv implementation (Issue  #273 depending on #233)
        # data = ht.array(ht.random.randn(1, 4, 5, 1), split=1)
        # result = ht.squeeze(data, axis=-1)
        # self.assertIsInstance(result, ht.DNDarray)
        # # TODO: the following works locally but not when distributed,
        # #self.assertEqual(result.dtype, ht.float32)
        # #self.assertEqual(result._DNDarray__array.dtype, torch.float32)
        # self.assertEqual(result.shape, (1, 12, 5))
        # self.assertEqual(result.lshape, (1, 12, 5))
        # self.assertEqual(result.split, 1)

        # 3D split tensor, across the axis
        size = ht.MPI_WORLD.size * 2
        data = ht.triu(ht.ones((1, size, size), split=1, device=ht_device), k=1)

        result = ht.squeeze(data, axis=0)
        self.assertIsInstance(result, ht.DNDarray)
        # TODO: the following works locally but not when distributed,
        # self.assertEqual(result.dtype, ht.float32)
        # self.assertEqual(result._DNDarray__array.dtype, torch.float32)
        self.assertEqual(result.shape, (size, size))
        self.assertEqual(result.lshape, (size, size))
        # self.assertEqual(result.split, None)

        # check exceptions
        with self.assertRaises(ValueError):
            data.squeeze(axis=(0, 1))
        with self.assertRaises(TypeError):
            data.squeeze(axis=1.1)
        with self.assertRaises(TypeError):
            data.squeeze(axis="y")
        with self.assertRaises(ValueError):
            ht.argmin(data, axis=-4)

    def test_unique(self):
        size = ht.MPI_WORLD.size
        rank = ht.MPI_WORLD.rank
        torch_array = torch.arange(size, dtype=torch.int32, device=device).expand(size, size)
        split_zero = ht.array(torch_array, split=0, device=ht_device)

        exp_axis_none = ht.array([rank], dtype=ht.int32, device=ht_device)
        res = split_zero.unique(sorted=True)
        self.assertTrue((res._DNDarray__array == exp_axis_none._DNDarray__array).all())

        exp_axis_zero = ht.arange(size, dtype=ht.int32, device=ht_device).expand_dims(0)
        res = ht.unique(split_zero, sorted=True, axis=0)
        self.assertTrue((res._DNDarray__array == exp_axis_zero._DNDarray__array).all())

        exp_axis_one = ht.array([rank], dtype=ht.int32, device=ht_device).expand_dims(0)
        split_zero_transposed = ht.array(torch_array.transpose(0, 1), split=0, device=ht_device)
        res = ht.unique(split_zero_transposed, sorted=False, axis=1)
        self.assertTrue((res._DNDarray__array == exp_axis_one._DNDarray__array).all())

        split_one = ht.array(torch_array, dtype=ht.int32, split=1, device=ht_device)

        exp_axis_none = ht.arange(size, dtype=ht.int32, device=ht_device)
        res = ht.unique(split_one, sorted=True)
        self.assertTrue((res._DNDarray__array == exp_axis_none._DNDarray__array).all())

        exp_axis_zero = ht.array([rank], dtype=ht.int32, device=ht_device).expand_dims(0)
        res = ht.unique(split_one, sorted=False, axis=0)
        self.assertTrue((res._DNDarray__array == exp_axis_zero._DNDarray__array).all())

        exp_axis_one = ht.array([rank] * size, dtype=ht.int32, device=ht_device).expand_dims(1)
        res = ht.unique(split_one, sorted=True, axis=1)
        self.assertTrue((res._DNDarray__array == exp_axis_one._DNDarray__array).all())

        torch_array = torch.tensor(
            [[1, 2], [2, 3], [1, 2], [2, 3], [1, 2]], dtype=torch.int32, device=device
        )
        data = ht.array(torch_array, split=0, device=ht_device)

        res, inv = ht.unique(data, return_inverse=True, axis=0)
        _, exp_inv = torch_array.unique(dim=0, return_inverse=True, sorted=True)
        self.assertTrue(torch.equal(inv, exp_inv.to(dtype=inv.dtype)))

        res, inv = ht.unique(data, return_inverse=True, axis=1)
        _, exp_inv = torch_array.unique(dim=1, return_inverse=True, sorted=True)
        self.assertTrue(torch.equal(inv, exp_inv.to(dtype=inv.dtype)))

        torch_array = torch.tensor(
            [[1, 1, 2], [1, 2, 2], [2, 1, 2], [1, 3, 2], [0, 1, 2]],
            dtype=torch.int32,
            device=device,
        )
        exp_res, exp_inv = torch_array.unique(return_inverse=True, sorted=True)

        data_split_none = ht.array(torch_array, device=ht_device)
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
        self.assertTrue(torch.equal(inv._DNDarray__array, exp_inv))

        data_split_zero = ht.array(torch_array, split=0, device=ht_device)
        res, inv = ht.unique(data_split_zero, return_inverse=True, sorted=True)
        self.assertTrue(torch.equal(inv, exp_inv.to(dtype=inv.dtype)))

    def test_resplit(self):
        # resplitting with same axis, should leave everything unchanged
        shape = (ht.MPI_WORLD.size, ht.MPI_WORLD.size)
        data = ht.zeros(shape, split=None, device=ht_device)
        data2 = ht.resplit(data, None)

        self.assertIsInstance(data2, ht.DNDarray)
        self.assertEqual(data2.shape, shape)
        self.assertEqual(data2.lshape, shape)
        self.assertEqual(data2.split, None)

        # resplitting with same axis, should leave everything unchanged
        shape = (ht.MPI_WORLD.size, ht.MPI_WORLD.size)
        data = ht.zeros(shape, split=1, device=ht_device)
        data2 = ht.resplit(data, 1)

        self.assertIsInstance(data2, ht.DNDarray)
        self.assertEqual(data2.shape, shape)
        self.assertEqual(data2.lshape, (data.comm.size, 1))
        self.assertEqual(data2.split, 1)

        # splitting an unsplit tensor should result in slicing the tensor locally
        shape = (ht.MPI_WORLD.size, ht.MPI_WORLD.size)
        data = ht.zeros(shape, device=ht_device)
        data2 = ht.resplit(data, 1)

        self.assertIsInstance(data2, ht.DNDarray)
        self.assertEqual(data2.shape, shape)
        self.assertEqual(data2.lshape, (data.comm.size, 1))
        self.assertEqual(data2.split, 1)

        # unsplitting, aka gathering a tensor
        shape = (ht.MPI_WORLD.size + 1, ht.MPI_WORLD.size)
        data = ht.ones(shape, split=0, device=ht_device)
        data2 = ht.resplit(data, None)

        self.assertIsInstance(data2, ht.DNDarray)
        self.assertEqual(data2.shape, shape)
        self.assertEqual(data2.lshape, shape)
        self.assertEqual(data2.split, None)

        # assign and entirely new split axis
        shape = (ht.MPI_WORLD.size + 2, ht.MPI_WORLD.size + 1)
        data = ht.ones(shape, split=0, device=ht_device)
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
        self.assertTrue((resplit_tensor._DNDarray__array == local_tensor._DNDarray__array).all())

        # unsplit
        unsplit_tensor = ht.resplit(resplit_tensor, axis=None)
        self.assertTrue(
            (unsplit_tensor._DNDarray__array == reference_tensor._DNDarray__array).all()
        )

        # split along axis = 1
        resplit_tensor = ht.resplit(unsplit_tensor, axis=1)
        if ht.MPI_WORLD.rank == 0:
            local_shape = (N, 2, 2 * N)
            local_tensor = reference_tensor[:, 0:2, :]
        else:
            local_shape = (N, 1, 2 * N)
            local_tensor = reference_tensor[:, ht.MPI_WORLD.rank + 1 : ht.MPI_WORLD.rank + 2, :]

        self.assertEqual(resplit_tensor.lshape, local_shape)
        self.assertTrue((resplit_tensor._DNDarray__array == local_tensor._DNDarray__array).all())

        # unsplit
        unsplit_tensor = ht.resplit(resplit_tensor, axis=None)
        self.assertTrue(
            (unsplit_tensor._DNDarray__array == reference_tensor._DNDarray__array).all()
        )

        # split along axis = 2
        resplit_tensor = ht.resplit(unsplit_tensor, axis=2)
        local_shape = (N, N + 1, 2)
        local_tensor = reference_tensor[:, :, 2 * ht.MPI_WORLD.rank : 2 * ht.MPI_WORLD.rank + 2]

        self.assertEqual(resplit_tensor.lshape, local_shape)
        self.assertTrue((resplit_tensor._DNDarray__array == local_tensor._DNDarray__array).all())

    def test_vstack(self):
        # cases to test:
        # MM===================================
        # NN,
        a = ht.ones((10, 12), split=None, device=ht_device)
        b = ht.ones((10, 12), split=None, device=ht_device)
        res = ht.vstack((a, b))
        self.assertEqual(res.shape, (20, 12))
        # 11,
        a = ht.ones((10, 12), split=1, device=ht_device)
        b = ht.ones((10, 12), split=1, device=ht_device)
        res = ht.vstack((a, b))
        self.assertEqual(res.shape, (20, 12))

        # VM===================================
        # NN,
        a = ht.ones((10,), split=None, device=ht_device)
        b = ht.ones((12, 10), split=None, device=ht_device)
        res = ht.vstack((a, b))
        self.assertEqual(res.shape, (13, 10))
        # 00
        a = ht.ones((10,), split=0, device=ht_device)
        b = ht.ones((12, 10), split=0, device=ht_device)
        res = ht.vstack((a, b))
        self.assertEqual(res.shape, (13, 10))

        # MV===================================
        # NN,
        a = ht.ones((12, 10), split=None, device=ht_device)
        b = ht.ones((10,), split=None, device=ht_device)
        res = ht.vstack((a, b))
        self.assertEqual(res.shape, (13, 10))
        # 00
        a = ht.ones((12, 10), split=0, device=ht_device)
        b = ht.ones((10,), split=0, device=ht_device)
        res = ht.vstack((a, b))
        self.assertEqual(res.shape, (13, 10))

        # VV===================================
        # NN,
        a = ht.ones((12,), split=None, device=ht_device)
        b = ht.ones((12,), split=None, device=ht_device)
        res = ht.vstack((a, b))
        self.assertEqual(res.shape, (2, 12))
        # 00
        a = ht.ones((12,), split=0, device=ht_device)
        b = ht.ones((12,), split=0, device=ht_device)
        res = ht.vstack((a, b))
        self.assertEqual(res.shape, (2, 12))
