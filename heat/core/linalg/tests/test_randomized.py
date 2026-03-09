import heat as ht

from ...tests.test_suites.basic_test import TestCase


class TestRSVD(TestCase):
    def test_rsvd(self):
        if self.is_mps:
            dtypes = [ht.float32]
        else:
            dtypes = [ht.float32, ht.float64]
        for dtype in dtypes:
            dtype_tol = 1e-4 if dtype == ht.float32 else 1e-10
            for split in [0, 1, None]:
                X = ht.random.randn(200, 200, dtype=dtype, split=split)
                for rank in [ht.MPI_WORLD.size, 10]:
                    for n_oversamples in [5, 10]:
                        for power_iter in [0, 1, 2, 3]:
                            with self.subTest(
                                dtype=dtype,
                                split=split,
                                rank=rank,
                                n_oversamples=n_oversamples,
                                power_iter=power_iter,
                            ):
                                U, S, Vt = ht.linalg.rsvd(
                                    X, rank, n_oversamples=n_oversamples,
                                    power_iter=power_iter
                                )
                                V = Vt.T
                                self.assertEqual(U.shape, (X.shape[0], rank))
                                self.assertEqual(S.shape, (rank,))
                                self.assertEqual(V.shape, (X.shape[1], rank))
                                self.assertTrue(ht.all(S >= 0))
                                self.assertTrue(
                                    ht.allclose(
                                        U.T @ U,
                                        ht.eye(rank, dtype=U.dtype, split=U.split),
                                        rtol=dtype_tol,
                                        atol=dtype_tol,
                                    )
                                )
                                self.assertTrue(
                                    ht.allclose(
                                        V.T @ V,
                                        ht.eye(rank, dtype=V.dtype, split=V.split),
                                        rtol=dtype_tol,
                                        atol=dtype_tol,
                                    )
                                )

    def test_rsvd_catch_wrong_inputs(self):
        X = ht.random.randn(10, 10)
        # wrong dtype for rank
        with self.assertRaises(TypeError):
            ht.linalg.rsvd(X, "a")
        # rank zero
        with self.assertRaises(ValueError):
            ht.linalg.rsvd(X, 0)
        # wrong dtype for n_oversamples
        with self.assertRaises(TypeError):
            ht.linalg.rsvd(X, 10, n_oversamples="a")
        # n_oversamples negative
        with self.assertRaises(ValueError):
            ht.linalg.rsvd(X, 10, n_oversamples=-1)
        # wrong dtype for power_iter
        with self.assertRaises(TypeError):
            ht.linalg.rsvd(X, 10, power_iter="a")
        # power_iter negative
        with self.assertRaises(ValueError):
            ht.linalg.rsvd(X, 10, power_iter=-1)


class TestREIGH(TestCase):
    def test_reigh(self):
        if self.is_mps:
            dtypes = [ht.float32]
        else:
            dtypes = [ht.float32, ht.float64]
        for dtype in dtypes:
            dtype_tol = 1e-4 if dtype == ht.float32 else 1e-10
            for split in [0, 1, None]:
                # Create a symmetric positive definite matrix
                X = ht.random.randn(100, 100, dtype=dtype, split=split)
                # This ensures A is symmetric positive semi-definite
                A = X @ X.T
                for rank in [5, 10]:
                    for n_oversamples in [5, 10]:
                        for power_iter in [0, 1, 2]:
                            with self.subTest(
                                dtype=dtype,
                                split=split,
                                rank=rank,
                                n_oversamples=n_oversamples,
                                power_iter=power_iter,
                            ):
                                S, V = ht.linalg.reigh(
                                    A, rank, n_oversamples=n_oversamples,
                                    power_iter=power_iter
                                )
                                self.assertEqual(V.shape, (A.shape[0], rank))
                                self.assertEqual(S.shape, (rank,))
                                # Check eigenvalues are real (symmetric matrices)
                                self.assertTrue(ht.all(ht.isreal(S)))
                                # Check that eigenvalues are in descending order
                                self.assertTrue(ht.all(S[:-1] >= S[1:]))
                                # Check orthogonality of eigenvectors
                                V.resplit_(None)
                                self.assertTrue(
                                    ht.allclose(
                                        V.T @ V,
                                        ht.eye(rank, dtype=V.dtype, split=V.split),
                                        rtol=dtype_tol,
                                        atol=dtype_tol,
                                    )
                                )

    def test_reigh_catch_wrong_inputs(self):
        # Create a symmetric matrix for testing
        X = ht.random.randn(10, 10)
        A = X @ X.T

        # wrong dtype for rank
        with self.assertRaises(TypeError):
            ht.linalg.reigh(A, "a")
        # rank zero
        with self.assertRaises(ValueError):
            ht.linalg.reigh(A, 0)
        # wrong dtype for n_oversamples
        with self.assertRaises(TypeError):
            ht.linalg.reigh(A, 5, n_oversamples="a")
        # n_oversamples negative
        with self.assertRaises(ValueError):
            ht.linalg.reigh(A, 5, n_oversamples=-1)
        # wrong dtype for power_iter
        with self.assertRaises(TypeError):
            ht.linalg.reigh(A, 5, power_iter="a")
        # power_iter negative
        with self.assertRaises(ValueError):
            ht.linalg.reigh(A, 5, power_iter=-1)
