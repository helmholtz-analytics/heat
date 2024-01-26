import heat as ht
import unittest
import torch

from ...tests.test_suites.basic_test import TestCase


@unittest.skipIf(torch.cuda.is_available() and torch.version.hip, "not supported for HIP")
class TestQR(TestCase):
    def test_qr(self):
        for split in [1, None]:
            for calc_r in [True, False]:
                calc_qs = [True, False] if calc_r else [True]
                for calc_q in calc_qs:
                    for full_q in [True, False]:
                        for shape in [
                            (2 * ht.MPI_WORLD.size, 4 * ht.MPI_WORLD.size),
                            (2 * ht.MPI_WORLD.size, 2 * ht.MPI_WORLD.size),
                            (4 * ht.MPI_WORLD.size, 2 * ht.MPI_WORLD.size),
                        ]:
                            for dtype in [ht.float32, ht.float64]:
                                dtypetol = 1e-4 if dtype == ht.float32 else 1e-8
                                mat = ht.random.randn(*shape, dtype=dtype, split=split)
                                qr = ht.linalg.qr(mat, calc_r=calc_r, calc_q=calc_q, full_q=full_q)

                                # test if mat = Q*R
                                if calc_q and calc_r:
                                    self.assertTrue(ht.allclose(qr.Q @ qr.R, mat, atol=dtypetol))

                                if calc_q:
                                    # test if Q is orthogonal
                                    self.assertTrue(
                                        ht.allclose(
                                            qr.Q.T @ qr.Q,
                                            ht.eye(qr.Q.shape[1], dtype=dtype),
                                            atol=dtypetol,
                                        )
                                    )
                                    # test correct shape of Q
                                    if not full_q:
                                        self.assertEqual(qr.Q.shape, (shape[0], min(shape)))
                                    else:
                                        print(shape, full_q, qr.Q.shape, split)
                                        self.assertEqual(qr.Q.shape, (shape[0], shape[0]))
                                if calc_r:
                                    # test correct shape of R
                                    if not full_q:
                                        self.assertEqual(qr.R.shape, (min(shape), shape[1]))
                                    else:
                                        self.assertEqual(qr.R.shape, shape)
