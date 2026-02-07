import torch
import numpy as np
import scipy
import heat as ht

from ...tests.test_suites.basic_test import TestCase


class TestMatrixExp(TestCase):
    def test_against_pytorch(self):
        torch.manual_seed(42)

        size = ht.communication.MPI_WORLD.size * 2
        shapes = [(size, size), (2*size, size, size), (2*size, 3*size, size, size)]
        dtypes = [ht.float32, ht.float64, ht.complex64, ht.complex128]
        for shape in shapes:
            for dtype in dtypes:
                for split in [None] + [i for i in range(len(shape)-2)]:
                    with self.subTest(f'{shape=} {dtype=} {split=}'):
                        A = ht.random.randn(*shape, dtype=dtype, split=split)

                        get = ht.linalg.matrix_exp(A)
                        expect = torch.linalg.matrix_exp(A.resplit(None).larray)

                        self.assertTrue(torch.allclose(get.resplit(None).larray, expect))

                        if ht.communication.MPI_WORLD.size > 1 and split is not None:
                            self.assertTrue(get.is_distributed())
                        else:
                            self.assertFalse(get.is_distributed())


    def test_against_scipy(self):
        torch.manual_seed(42)

        size = ht.communication.MPI_WORLD.size * 2
        shapes = [(size, size), (2*size, size, size), (2*size, 3*size, size, size)]
        dtypes = [ht.float64, ht.complex128]
        for shape in shapes:
            for dtype in dtypes:
                for split in [None] + [i for i in range(len(shape)-2)]:
                    with self.subTest(f'{shape=} {dtype=} {split=}'):
                        A = ht.random.randn(*shape, dtype=dtype, split=split)

                        get = ht.linalg.expm(A)
                        expect = scipy.linalg.expm(A.resplit(None).larray.cpu())

                        self.assertTrue(np.allclose(get.resplit(None).larray.cpu(), expect))

                        if ht.communication.MPI_WORLD.size > 1 and split is not None:
                            self.assertTrue(get.is_distributed())
                        else:
                            self.assertFalse(get.is_distributed())


    def test_errors(self):
        A = ht.random.randn(4, 3, split=None)
        with self.assertRaises(RuntimeError):
            ht.linalg.matrix_exp(A)

        with self.assertRaises(TypeError):
            ht.linalg.matrix_exp(A.larray)

        if ht.communication.MPI_WORLD.size > 1:
            for split in [0, 1]:
                A = ht.random.randn(ht.communication.MPI_WORLD.size, ht.communication.MPI_WORLD.size, split=split)
                with self.assertRaises(ValueError):
                    ht.linalg.matrix_exp(A)
            A = ht.random.randn(2, ht.communication.MPI_WORLD.size, ht.communication.MPI_WORLD.size, split=1)
            with self.assertRaises(ValueError):
                ht.linalg.matrix_exp(A)
