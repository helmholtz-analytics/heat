import heat as ht
import torch

from .test_suites.basic_test import TestCase


class TestVmap(TestCase):
    def test_vmap(self):
        # two inputs (both split), two outputs, including keyword arguments that are not vmapped
        # inputs split along different axes, output split along same axis (one of them different to input split)
        x0 = ht.random.randn(5 * ht.MPI_WORLD.size, 10, 10, split=0)
        x1 = ht.random.randn(10, 5 * ht.MPI_WORLD.size, split=1)
        out_dims = (0, 0)

        def func(x0, x1, k=2, scale=1e-2):
            return torch.topk(torch.linalg.svdvals(x0), k)[0] ** 2, scale * x0 @ x1

        vfunc = ht.vmap(func, out_dims)
        y0, y1 = vfunc(x0, x1, k=2, scale=2.2)

        # compare with torch
        x0_torch = x0.resplit(None).larray
        x1_torch = x1.resplit(None).larray
        vfunc_torch = torch.vmap(func, (0, 1), (0, 0))
        y0_torch, y1_torch = vfunc_torch(x0_torch, x1_torch, k=2, scale=2.2)

        self.assertTrue(torch.allclose(y0.resplit(None).larray, y0_torch))
        self.assertTrue(torch.allclose(y1.resplit(None).larray, y1_torch))

        # two inputs (only one of them split), two outputs, including keyword arguments that are not vmapped
        # output split along different axis, one output has different data type than input
        x0 = ht.random.randn(5 * ht.MPI_WORLD.size, 10, 10, split=0)
        x1 = ht.random.randn(10, 5 * ht.MPI_WORLD.size, split=None)
        out_dims = (0, 1)

        def func(x0, x1, k=2, scale=1e-2):
            return torch.topk(torch.linalg.svdvals(x0), k)[0] ** 2, (scale * x0 @ x1).int()

        vfunc = ht.vmap(func, out_dims)
        y0, y1 = vfunc(x0, x1, k=2, scale=2.2)

        # compare with torch
        x0_torch = x0.resplit(None).larray
        x1_torch = x1.resplit(None).larray
        vfunc_torch = torch.vmap(func, (0, None), (0, 1))
        y0_torch, y1_torch = vfunc_torch(x0_torch, x1_torch, k=2, scale=2.2)

        self.assertTrue(torch.allclose(y0.resplit(None).larray, y0_torch))
        self.assertTrue(torch.allclose(y1.resplit(None).larray, y1_torch))

    def test_vmap_with_chunks(self):
        # same as before but now with prescribed chunk sizes for the vmap
        x0 = ht.random.randn(5 * ht.MPI_WORLD.size, 10, 10, split=0)
        x1 = ht.random.randn(10, 5 * ht.MPI_WORLD.size, split=1)
        out_dims = (0, 0)

        def func(x0, x1, k=2, scale=1e-2):
            return torch.topk(torch.linalg.svdvals(x0), k)[0] ** 2, scale * x0 @ x1

        vfunc = ht.vmap(func, out_dims, chunk_size=2)
        y0, y1 = vfunc(x0, x1, k=2, scale=-2.2)

        # compare with torch
        x0_torch = x0.resplit(None).larray
        x1_torch = x1.resplit(None).larray
        vfunc_torch = torch.vmap(func, (0, 1), (0, 0))
        y0_torch, y1_torch = vfunc_torch(x0_torch, x1_torch, k=2, scale=-2.2)

        self.assertTrue(torch.allclose(y0.resplit(None).larray, y0_torch))
        self.assertTrue(torch.allclose(y1.resplit(None).larray, y1_torch))

        # two inputs (only one of them split), two outputs, including keyword arguments that are not vmapped
        # output split along different axis
        x0 = ht.random.randn(5 * ht.MPI_WORLD.size, 10, 10, split=0)
        x1 = ht.random.randn(10, 5 * ht.MPI_WORLD.size, split=None)
        out_dims = (0, 1)

        def func(x0, x1, k=2, scale=1e-2):
            return torch.topk(torch.linalg.svdvals(x0), k)[0] ** 2, scale * x0 @ x1

        vfunc = ht.vmap(func, out_dims, chunk_size=1)
        y0, y1 = vfunc(x0, x1, k=5, scale=2.2)

        # compare with torch
        x0_torch = x0.resplit(None).larray
        x1_torch = x1.resplit(None).larray
        vfunc_torch = torch.vmap(func, (0, None), (0, 1))
        y0_torch, y1_torch = vfunc_torch(x0_torch, x1_torch, k=5, scale=2.2)

        self.assertTrue(torch.allclose(y0.resplit(None).larray, y0_torch))
        self.assertTrue(torch.allclose(y1.resplit(None).larray, y1_torch))

    def test_vmap_catch_errors(self):
        # not a callable
        with self.assertRaises(TypeError):
            ht.vmap(1)
        # invalid randomness
        with self.assertRaises(ValueError):
            ht.vmap(lambda x: x, randomness="random")
        # invalid chunk_size
        with self.assertRaises(TypeError):
            ht.vmap(lambda x: x, chunk_size="1")
        with self.assertRaises(ValueError):
            ht.vmap(lambda x: x, chunk_size=0)
        # not all inputs are DNDarrays
        with self.assertRaises(TypeError):
            ht.vmap(lambda x: x, out_dims=0)(ht.ones(10), 2)
        # number of output DNDarrays does not match number of split dimensions
        with self.assertRaises(ValueError):
            ht.vmap(lambda x: x, out_dims=(0, 1))(ht.ones(10))
