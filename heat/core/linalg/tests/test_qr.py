import heat as ht
import unittest
import torch

from ...tests.test_suites.basic_test import TestCase


@unittest.skipIf(torch.cuda.is_available() and torch.version.hip, "not supported for HIP")
class TestQR(TestCase):
    def test_qr(self):
        for split in [1, None]:
            for mode in ["reduced", "r"]:
                for shape in [
                    (2 * ht.MPI_WORLD.size, 4 * ht.MPI_WORLD.size),
                    (2 * ht.MPI_WORLD.size, 2 * ht.MPI_WORLD.size),
                    (4 * ht.MPI_WORLD.size, 2 * ht.MPI_WORLD.size),
                ]:
                    for dtype in [ht.float32, ht.float64]:
                        dtypetol = 1e-3 if dtype == ht.float32 else 1e-6
                        mat = ht.random.randn(*shape, dtype=dtype, split=split)
                        qr = ht.linalg.qr(mat, mode=mode)

                        if mode == "reduced":
                            self.assertTrue(
                                ht.allclose(qr.Q @ qr.R, mat, atol=dtypetol, rtol=dtypetol)
                            )
                            self.assertIsInstance(qr.Q, ht.DNDarray)

                            # test if Q is orthogonal
                            self.assertTrue(
                                ht.allclose(
                                    qr.Q.T @ qr.Q,
                                    ht.eye(qr.Q.shape[1], dtype=dtype),
                                    atol=dtypetol,
                                    rtol=dtypetol,
                                )
                            )
                            # test correct shape of Q
                            self.assertEqual(qr.Q.shape, (shape[0], min(shape)))
                        else:
                            self.assertIsNone(qr.Q)

                        # test correct type and shape of R
                        self.assertIsInstance(qr.R, ht.DNDarray)
                        self.assertEqual(qr.R.shape, (min(shape), shape[1]))
