import torch
import os
import unittest
import heat as ht
import numpy as np

from ...tests.test_suites.basic_test import TestCase


class TestSolver(TestCase):
    def test_cg(self):
        size = ht.communication.MPI_WORLD.size * 3
        b = ht.arange(1, size + 1, dtype=ht.float32, split=0)
        A = ht.manipulations.diag(b)
        x0 = ht.random.rand(size, dtype=b.dtype, split=b.split)

        x = ht.ones(b.shape, dtype=b.dtype, split=b.split)

        res = ht.linalg.cg(A, b, x0)
        self.assertTrue(ht.allclose(x, res, atol=1e-3))

        b_np = np.arange(1, size + 1)
        with self.assertRaises(TypeError):
            ht.linalg.cg(A, b_np, x0)

        with self.assertRaises(RuntimeError):
            ht.linalg.cg(A, A, x0)
        with self.assertRaises(RuntimeError):
            ht.linalg.cg(b, b, x0)
        with self.assertRaises(RuntimeError):
            ht.linalg.cg(A, b, A)

    def test_lanczos(self):
        # single precision tolerance for torch.inv() is pretty bad
        tolerance = 1e-3

        dtype, atol = (ht.float32, tolerance) if self.is_mps else (ht.float64, 1e-12)

        # define positive definite matrix (n,n), split = 0
        n = 100
        A = ht.random.randn(n, n, dtype=dtype, split=0)
        B = A @ A.T
        # Lanczos decomposition with iterations m = n
        V, T = ht.lanczos(B, m=n)
        self.assertTrue(V.dtype is B.dtype)
        self.assertTrue(T.dtype is B.dtype)
        # V must be unitary
        V_inv = ht.linalg.inv(V)
        self.assertTrue(ht.allclose(V_inv, V.T, atol=atol))
        # V T V.T must be = B, V transposed = V inverse
        lanczos_B = V @ T @ V_inv
        self.assertTrue(ht.allclose(lanczos_B, B, atol=atol))

        # complex128, output buffers
        if not self.is_mps:
            A = ht.random.rand(n, n, dtype=ht.complex128, split=0)
            A_conj = ht.conj(A)
            B = A @ A_conj.T
            m = n
            V_out = ht.zeros((n, m), dtype=B.dtype, split=B.split, device=B.device, comm=B.comm)
            T_out = ht.zeros((m, m), dtype=ht.float64, device=B.device, comm=B.comm)
            # Lanczos decomposition with iterations m = n
            ht.lanczos(B, m=m, V_out=V_out, T_out=T_out)
            # V must be unitary
            V_inv = ht.linalg.inv(V_out)
            self.assertTrue(ht.allclose(V_inv, ht.conj(V_out).T))
            # V T V* must be = B, V conjugate transpose = V inverse
            lanczos_B = V_out @ T_out @ V_inv
            self.assertTrue(ht.allclose(lanczos_B, B))

        # float32, pre_defined v0, split mismatch
        A = ht.random.randn(n, n, dtype=ht.float32, split=0)
        B = A @ A.T
        v0 = ht.random.randn(n, device=A.device, split=None)
        v0 = v0 / ht.norm(v0)
        # Lanczos decomposition with iterations m = n
        V, T = ht.lanczos(B, m=n, v0=v0)
        self.assertTrue(V.dtype is B.dtype)
        self.assertTrue(T.dtype is B.dtype)
        # # skipping the following tests as torch.inv on float32 is too imprecise
        # # V must be unitary
        # V_inv = ht.linalg.inv(V)
        # self.assertTrue(ht.allclose(V_inv, V.T, atol=atol))
        # # V T V.T must be = B, V transposed = V inverse
        # lanczos_B = V @ T @ V_inv
        # self.assertTrue(ht.allclose(lanczos_B, B, atol=atol))

        # complex64
        if not self.is_mps:
            # in principle, MPS supports complex64, but many operations are not implemented, e.g. matmul, div
            A = ht.random.randn(n, n, dtype=ht.complex64, split=0)
            A_conj = ht.conj(A)
            B = A @ A_conj.T
            # Lanczos decomposition with iterations m = n
            V, T = ht.lanczos(B, m=n)
            # V must be unitary
            # V T V* must be = B, V conjugate transpose = V inverse
            V_conj = ht.conj(V)
            lanczos_B = V @ T @ V_conj.T
            self.assertTrue(ht.allclose(lanczos_B, B, atol=tolerance))

        # non-distributed
        A = ht.random.randn(n, n, dtype=dtype, split=None)
        B = A @ A.T
        # Lanczos decomposition with iterations m = n
        m = n
        V_out = ht.zeros((n, m), dtype=B.dtype, split=B.split, device=B.device, comm=B.comm)
        T_out = ht.zeros((m, m), dtype=dtype, device=B.device, comm=B.comm)
        ht.lanczos(B, m=m, V_out=V_out, T_out=T_out)
        self.assertTrue(V_out.dtype is B.dtype)
        self.assertTrue(T_out.dtype is B.real.dtype)
        # V must be unitary
        V_inv = ht.linalg.inv(V_out)
        self.assertTrue(ht.allclose(V_inv, V_out.T, atol=atol))
        # without output buffers
        V, T = ht.lanczos(B, m=m)
        # V T V.T must be = B, V transposed = V inverse
        lanczos_B = V @ T @ V.T
        self.assertTrue(ht.allclose(lanczos_B, B, atol=atol))

        with self.assertRaises(TypeError):
            V, T = ht.lanczos(B, m="3")
        with self.assertRaises(TypeError):
            A = ht.random.randint(0, 5, (10, 10))
            V, T = ht.lanczos(A, m=3)
        with self.assertRaises(TypeError):
            A = torch.randn(10, 10)
            V, T = ht.lanczos(A, m=3)
        with self.assertRaises(TypeError):
            A = ht.random.randn(10, 12)
            V, T = ht.lanczos(A, m=3)
        with self.assertRaises(RuntimeError):
            A = ht.random.randn(10, 12, 12)
            V, T = ht.lanczos(A, m=3)
        with self.assertRaises(NotImplementedError):
            A = ht.random.randn(10, 10, split=1)
            V, T = ht.lanczos(A, m=3)

    def test_solve(self):
        torch.manual_seed(42 + ht.communication.MPI_WORLD.rank)

        types = [ht.float32, ht.float64, ht.complex64, ht.complex128]
        s = ht.communication.MPI_WORLD.size
        n = s * 2
        k = s * 3

        A_shapes = [(n, n), (k, n, n), (n, n), (k, n, n), (k, n, n), (n, k, n, n)]
        b_shapes = [(n,), (k, n), (n, k), (n,), (n, k), (n, k, n)]

        for dtype in types:
            for A_shape, b_shape in zip(A_shapes, b_shapes):
                for split in [None] + [i for i in range(len(A_shape) - 2)]:
                    with self.subTest(f'{dtype=} {A_shape=} {b_shape=} {split=}'):
                        A = ht.random.randn(*A_shape, dtype=dtype, split=split)
                        b = ht.random.randn(*b_shape, dtype=dtype, split=split if b_shape[0] == A_shape[0] else None)

                        x = ht.linalg.solve(A, b)

                        try:
                            _b = A @ x
                        except NotImplementedError:  # batched matrix-vector not implemented
                            _x = ht.expand_dims(x, -1)
                            _b = (A@_x)[..., 0]

                        error = ht.linalg.norm(_b - b)
                        thresh = 1e5 * ht.finfo(dtype).eps
                        self.assertTrue(error < thresh, f'Error {float(error):.2e} > than threshold of {thresh:.2e}')

                        # test that the solving works with passing output array
                        y = ht.empty_like(x)
                        ht.linalg.solve(A, b, out=y)
                        self.assertTrue(ht.allclose(x, y))

        # test a few additionally allowed splits
        if ht.communication.MPI_WORLD.size > 1:
            A = ht.random.randn(s, s, split=None)
            b = ht.random.randn(s, 2*s, split=1)
            x = ht.linalg.solve(A, b)
            error = float(ht.linalg.norm(A@x - b))
            self.assertTrue(error < 1e-3, f'Error {error}')

            A = ht.random.randn(2*s, s, s, split=None)
            b = ht.random.randn(2*s, s, 2*s, split=2)
            x = ht.linalg.solve(A, b)
            error = float(ht.linalg.norm(A@x - b))
            self.assertTrue(error < 1e-3, f'Error {error}')


        # test separately with ints
        for dtype in [ht.int8, ht.int16, ht.int32, ht.int64]:
            with self.subTest(f'{dtype=}'):
                A = ht.array([[1, 2], [3, 5]], dtype=dtype)
                b = ht.array([1, 2], dtype=dtype)
                x = ht.linalg.solve(A, b)
                self.assertTrue(ht.allclose(A@x, b))
                self.assertTrue(x.dtype == dtype)

                out = ht.empty_like(x).astype(dtype, copy=False)
                ht.linalg.solve(A, b, out=out)
                self.assertTrue(ht.allclose(x, out))
                self.assertTrue(x.dtype == dtype)

        # --- test catching all the things that are not supposed to work ---

        # out argument is used incorrectly
        A = ht.random.randn(2*s, s, s)
        b = ht.random.randn(s)
        out = ht.empty(shape = (3*s, 4*s, 5*s))
        with self.assertRaises(ValueError):  # out has totally wrong shape
            ht.linalg.solve(A, b, out=out)
        out = ht.empty(shape = (2*s, s), split=1)
        with self.assertRaises(ValueError):  # out is split incorrectly
            ht.linalg.solve(A, b, out=out)
        out = ht.empty_like(b)
        with self.assertRaises(ValueError):  # need to expand out
            ht.linalg.solve(A, b, out=out)
        out = torch.zeros(size=(2*s, s))
        with self.assertRaises(TypeError):  # out has wrong datatype
            ht.linalg.solve(A, b, out=out)

        # input is torch tensor instead of DNDarray
        A = torch.ones(size=(s, s))
        b = ht.ones(s)
        with self.assertRaises(TypeError):
            ht.linalg.solve(A, b)

        A = ht.ones((s, s))
        b = torch.ones(s)
        with self.assertRaises(TypeError):
            ht.linalg.solve(A, b)

        # input has wrong shape
        A = ht.random.randn(2*s, s)
        b = ht.random.randn(s)
        with self.assertRaises(RuntimeError):
            ht.linalg.solve(A, b)

        A = ht.random.randn(s, s)
        b = ht.random.randn(2*s)
        with self.assertRaises(ValueError):
            ht.linalg.solve(A, b)

        A = ht.random.randn(s, s)
        b = ht.random.randn(4*s, 3*s, 2*s)
        with self.assertRaises(ValueError):
            ht.linalg.solve(A, b)

        A = ht.random.randn(s, s)
        b = ht.random.randn(4*s, 3*s, s)
        with self.assertRaises(ValueError):
            ht.linalg.solve(A, b)

        if ht.communication.MPI_WORLD.size > 1:
            # A is split in non-batched dimension
            for split in [0, 1]:
                A = ht.random.randn(s, s, split=split)
                b = ht.ones(s, split=None)
                with self.assertRaises(NotImplementedError):
                    ht.linalg.solve(A, b)

            A = ht.random.randn(2*s, s, s, split=1)
            b = ht.ones((2*s, s), split=None)
            with self.assertRaises(NotImplementedError):
                ht.linalg.solve(A, b)

            # b is split in non-batched dimension
            A = ht.random.randn(s, s, split=None)
            b = ht.ones((s,), split=0)
            with self.assertRaises(NotImplementedError):
                ht.linalg.solve(A, b)

            A = ht.random.randn(2*s, s, s, split=0)
            b = ht.ones((2*s, s), split=1)
            with self.assertRaises(NotImplementedError):
                ht.linalg.solve(A, b)

            # A and b are split in incompatible fashion
            A = ht.random.randn(3*s, s, s, split=0)
            b = ht.ones((s, 2*s), split=1)
            with self.assertRaises(ValueError):
                ht.linalg.solve(A, b)





    def test_solve_triangular(self):
        torch.manual_seed(42)
        tdev = ht.get_device().torch_device

        # non-batched tests
        k = 100  # data dimension size

        # random triangular matrix inversion
        at = torch.rand((k, k))
        # at += torch.eye(k)
        at += 1e2 * torch.ones_like(at)  # make gaussian elimination more stable
        at = torch.triu(at).to(tdev)

        ct = torch.linalg.solve_triangular(at, torch.eye(k, device=tdev), upper=True)

        a = ht.factories.asarray(at, copy=True)
        c = ht.factories.asarray(ct, copy=True)
        b = ht.eye(k)

        # exceptions
        with self.assertRaises(TypeError):  # invalid datatype for b
            ht.linalg.solve_triangular(a, 42)

        with self.assertRaises(ValueError):  # a no matrix, not enough dimensions
            ht.linalg.solve_triangular(a[1], b)

        with self.assertRaises(ValueError):  # a and b different number of dimensions
            ht.linalg.solve_triangular(a, b[1])

        with self.assertRaises(ValueError):  # a no square matrix
            ht.linalg.solve_triangular(a[1:, ...], b)

        with self.assertRaises(ValueError):  # split=1 for b
            b.resplit_(-1)
            ht.linalg.solve_triangular(a, b)

        b.resplit_(0)
        with self.assertRaises(ValueError):  # dimension mismatch
            ht.linalg.solve_triangular(a, b[1:, ...])

        for s0, s1 in (None, None), (-2, -2), (-1, -2), (-2, None), (-1, None), (None, -2):
            a.resplit_(s0)
            b.resplit_(s1)

            res = ht.linalg.solve_triangular(a, b)
            self.assertTrue(ht.allclose(res, c))

        # triangular ones inversion
        # for this test case, the results should be exact
        at = torch.triu(torch.ones_like(at)).to(tdev)
        ct = torch.linalg.solve_triangular(at, torch.eye(k, device=tdev), upper=True)

        a = ht.factories.asarray(at, copy=True)
        c = ht.factories.asarray(ct, copy=True)

        for s0, s1 in (None, None), (-2, -2), (-1, -2), (-2, None), (-1, None), (None, -2):
            a.resplit_(s0)
            b.resplit_(s1)

            res = ht.linalg.solve_triangular(a, b)
            self.assertTrue(ht.equal(res, c))

        # batched tests
        if self.is_mps:
            # reduction ops on tensors with ndim > 4 are not supported on MPS
            # see e.g. https://github.com/pytorch/pytorch/issues/129960
            # fmt: off
            batch_shapes = [(10,),]
            # fmt: on
        else:
            # fmt: off
            batch_shapes = [(10,), (4, 4, 4, 20,),]
            # fmt: on
        m = 100  # data dimension size

        # exceptions
        batch_shape = batch_shapes[-1]

        at = torch.rand((*batch_shape, m, m))
        # at += torch.eye(k)
        at += 1e2 * torch.ones_like(at)  # make gaussian elimination more stable
        at = torch.triu(at).to(tdev)
        bt = torch.eye(m).expand((*batch_shape, -1, -1)).to(tdev)

        ct = torch.linalg.solve_triangular(at, bt, upper=True)

        a = ht.factories.asarray(at, copy=True)
        c = ht.factories.asarray(ct, copy=True)
        b = ht.factories.asarray(bt, copy=True)

        with self.assertRaises(ValueError):  # batch dimensions of different shapes
            ht.linalg.solve_triangular(a[1:, ...], b)

        with self.assertRaises(ValueError):  # different batched split dimensions
            a.resplit_(0)
            b.resplit_(1)
            ht.linalg.solve_triangular(a, b)

        for batch_shape in batch_shapes:
            # batch_shape = tuple() # no batch dimensions
            at = torch.rand((*batch_shape, m, m))
            # at += torch.eye(k)
            at += 1e2 * torch.ones_like(at)  # make gaussian elimination more stable
            at = torch.triu(at).to(tdev)
            bt = torch.eye(m).expand((*batch_shape, -1, -1)).to(tdev)

            ct = torch.linalg.solve_triangular(at, bt, upper=True)

            a = ht.factories.asarray(at, copy=True)
            c = ht.factories.asarray(ct, copy=True)
            b = ht.factories.asarray(bt, copy=True)

            # split in linalg dimension or none
            for s0, s1 in (None, None), (-2, -2), (-1, -2), (-2, None), (-1, None), (None, -2):
                a.resplit_(s0)
                b.resplit_(s1)

                res = ht.linalg.solve_triangular(a, b)
                self.assertTrue(ht.allclose(c, res))

            # split in batch dimension
            s = len(batch_shape) - 1
            a.resplit_(s)
            b.resplit_(s)
            c.resplit_(s)

            res = ht.linalg.solve_triangular(a, b)
            self.assertTrue(ht.allclose(c, res))
