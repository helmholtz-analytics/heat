import heat as ht
import numpy as np

from ...tests.test_suites.basic_test import TestCase


class TestTallSkinnySVD(TestCase):
    def test_tallskinny_split0(self):
        if self.is_mps:
            dtypes = [ht.float32]
        else:
            dtypes = [ht.float32, ht.float64]
        for dtype in dtypes:
            for n_merge in [0, None]:
                tol = 1e-5 if dtype == ht.float32 else 1e-10
                X = ht.random.randn(ht.MPI_WORLD.size * 10 + 3, 10, split=0, dtype=dtype)
                if n_merge == 0:
                    U, S, V = ht.linalg.svd(X, qr_procs_to_merge=n_merge)
                else:
                    U, S, V = ht.linalg.svd(X)
                if ht.MPI_WORLD.size > 1:
                    self.assertTrue(U.split == 0)
                self.assertTrue(S.split is None)
                self.assertTrue(V.split is None)
                self.assertTrue(
                    ht.allclose(U.T @ U, ht.eye(U.shape[1], dtype=dtype), rtol=tol, atol=tol)
                )
                self.assertTrue(
                    ht.allclose(V.T @ V, ht.eye(V.shape[1], dtype=dtype), rtol=tol, atol=tol)
                )
                self.assertTrue(ht.allclose(U @ ht.diag(S) @ V.T, X, rtol=tol, atol=tol))
                self.assertTrue(ht.all(S >= 0))

    def test_shortfat_split1(self):
        if self.is_mps:
            dtypes = [ht.float32]
        else:
            dtypes = [ht.float32, ht.float64]
        for dtype in dtypes:
            tol = 1e-5 if dtype == ht.float32 else 1e-10
            X = ht.random.randn(10, ht.MPI_WORLD.size * 10 + 3, split=1, dtype=dtype)
            U, S, V = ht.linalg.svd(X)
            self.assertTrue(U.split is None)
            self.assertTrue(S.split is None)
            if ht.MPI_WORLD.size > 1:
                self.assertTrue(V.split == 0)
            self.assertTrue(
                ht.allclose(U.T @ U, ht.eye(U.shape[1], dtype=dtype), rtol=tol, atol=tol)
            )
            self.assertTrue(
                ht.allclose(V.T @ V, ht.eye(V.shape[1], dtype=dtype), rtol=tol, atol=tol)
            )
            self.assertTrue(ht.allclose(U @ ht.diag(S) @ V.T, X, rtol=tol, atol=tol))
            self.assertTrue(ht.all(S >= 0))

    def test_singvals_only(self):
        if self.is_mps:
            dtypes = [ht.float32]
        else:
            dtypes = [ht.float32, ht.float64]
        for dtype in dtypes:
            tol = 1e-5 if dtype == ht.float32 else 1e-10
            for split in [0, 1]:
                shape = (
                    (ht.MPI_WORLD.size * 10 + 3, 10)
                    if split == 0
                    else (10, ht.MPI_WORLD.size * 10 + 3)
                )
                X = ht.random.randn(*shape, split=split, dtype=dtype)
                S = ht.linalg.svd(X, compute_uv=False)
                self.assertTrue(S.split is None)
                self.assertTrue(S.shape[0] == min(shape))
                self.assertTrue(S.ndim == 1)
                X_np = X.numpy()
                self.assertTrue(
                    np.allclose(
                        np.linalg.svd(X_np, compute_uv=False), S.numpy(), rtol=tol, atol=tol
                    )
                )

    def test_wrong_inputs(self):
        # full_matrices = True
        X = ht.random.rand(10 * ht.MPI_WORLD.size, 5, split=0)
        with self.assertRaises(NotImplementedError):
            ht.linalg.svd(X, full_matrices=True)
        # not a DNDarray
        with self.assertRaises(TypeError):
            ht.linalg.svd("abc")
        # qr_procs_to_merge not an int
        with self.assertRaises(TypeError):
            ht.linalg.svd(X, qr_procs_to_merge="def")
        # qr_procs_to_merge = 1
        with self.assertRaises(ValueError):
            ht.linalg.svd(X, qr_procs_to_merge=1)
        # wrong dimension
        X = ht.random.randn(10, 10, 10, split=0)
        with self.assertRaises(ValueError):
            ht.linalg.svd(X)
        # not a float dtype
        X = ht.ones((10 * ht.MPI_WORLD.size, 10), split=0, dtype=ht.int32)
        with self.assertRaises(TypeError):
            ht.linalg.svd(X)


class TestZoloSVD(TestCase):
    def test_full_svd(self):
        shapes = [(100, 100), (117, 100), (100, 103)]
        splits = [None, 0, 1]
        dtypes = [ht.float32, ht.float64]
        for shape in shapes:
            for split in splits:
                for dtype in dtypes:
                    with self.subTest(shape=shape, split=split, dtype=dtype):
                        tol = 1e-2 if dtype == ht.float32 else 1e-2
                        X = ht.random.randn(*shape, split=split, dtype=dtype)
                        if split is not None and ht.MPI_WORLD.size > 1:
                            with self.assertWarns(UserWarning):
                                U, S, V = ht.linalg.svd(X)
                        else:
                            U, S, V = ht.linalg.svd(X)
                        self.assertTrue(
                            ht.allclose(
                                U.T @ U, ht.eye(U.shape[1], dtype=dtype), rtol=tol, atol=tol
                            )
                        )
                        self.assertTrue(
                            ht.allclose(
                                V.T @ V, ht.eye(V.shape[1], dtype=dtype), rtol=tol, atol=tol
                            )
                        )
                        self.assertTrue(ht.allclose(U @ ht.diag(S) @ V.T, X, rtol=tol, atol=tol))
                        self.assertTrue(ht.all(S >= 0))

    def test_options_full_svd(self):
        # only singular values
        X = ht.random.rand(101, 100, split=0, dtype=ht.float32)
        S = ht.linalg.svd(X, compute_uv=False)

        # prescribed r_max_zolopd
        U, S, V = ht.linalg.svd(X, r_max_zolopd=1)

        # catch error if r_max_zolopd is not provided properly
        if X.is_distributed():
            with self.assertRaises(ValueError):
                ht.linalg.svd(X, r_max_zolopd=0)
