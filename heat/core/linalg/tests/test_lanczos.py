import torch
import os
import unittest
import heat as ht
import numpy as np

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


class TestLinalgLanczos(unittest.TestCase):
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

    def test_norm(self):
        a = ht.arange(9, dtype=ht.float32, split=0) - 4
        self.assertTrue(
            ht.allclose(ht.linalg.norm(a), ht.float32(np.linalg.norm(a.numpy())).item(), atol=1e-5)
        )
        a.resplit_(axis=None)
        self.assertTrue(
            ht.allclose(ht.linalg.norm(a), ht.float32(np.linalg.norm(a.numpy())).item(), atol=1e-5)
        )

        b = ht.array([[-4.0, -3.0, -2.0], [-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], split=0)
        self.assertTrue(
            ht.allclose(ht.linalg.norm(b), ht.float32(np.linalg.norm(b.numpy())).item(), atol=1e-5)
        )
        b.resplit_(axis=1)
        self.assertTrue(
            ht.allclose(ht.linalg.norm(b), ht.float32(np.linalg.norm(b.numpy())).item(), atol=1e-5)
        )

        with self.assertRaises(TypeError):
            c = np.arange(9) - 4
            ht.linalg.norm(c)

    def test_projection(self):
        a = ht.arange(1, 4, dtype=ht.float32, split=None)
        e1 = ht.array([1, 0, 0], dtype=ht.float32, split=None)
        self.assertTrue(ht.equal(ht.linalg.projection(a, e1), e1))

        a.resplit_(axis=0)
        self.assertTrue(ht.equal(ht.linalg.projection(a, e1), e1))

        e2 = ht.array([0, 1, 0], dtype=ht.float32, split=0)
        self.assertTrue(ht.equal(ht.linalg.projection(a, e2), e2 * 2))

        a = ht.arange(1, 4, dtype=ht.float32, split=None)
        e3 = ht.array([0, 0, 1], dtype=ht.float32, split=0)
        self.assertTrue(ht.equal(ht.linalg.projection(a, e3), e3 * 3))

        a = np.arange(1, 4)
        with self.assertRaises(TypeError):
            ht.linalg.projection(a, e1)

        a = ht.array([[1], [2], [3]], dtype=ht.float32, split=None)
        with self.assertRaises(RuntimeError):
            ht.linalg.projection(a, e1)
