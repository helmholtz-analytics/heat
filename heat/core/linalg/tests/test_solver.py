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
