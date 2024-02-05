import heat as ht
import unittest
import torch
import numpy as np

from ...tests.test_suites.basic_test import TestCase


class TestQR(TestCase):
    def test_qr_split1orNone(self):
        for split in [1, None]:
            for mode in ["reduced", "r"]:
                # note that split = 1 can be handeled for arbitrary shapes
                for shape in [
                    (20 * ht.MPI_WORLD.size + 1, 40 * ht.MPI_WORLD.size),
                    (20 * ht.MPI_WORLD.size, 20 * ht.MPI_WORLD.size),
                    (40 * ht.MPI_WORLD.size - 1, 20 * ht.MPI_WORLD.size),
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

                        # compare with torch qr, due to different signs we can only compare absolute values
                        mat_t = mat.resplit_(None).larray
                        q_t, r_t = torch.linalg.qr(mat_t, mode=mode)
                        r_ht = qr.R.resplit_(None).larray
                        self.assertTrue(
                            torch.allclose(
                                torch.abs(r_t), torch.abs(r_ht), atol=dtypetol, rtol=dtypetol
                            )
                        )
                        if mode == "reduced":
                            q_ht = qr.Q.resplit_(None).larray
                            self.assertTrue(
                                torch.allclose(
                                    torch.abs(q_t), torch.abs(q_ht), atol=dtypetol, rtol=dtypetol
                                )
                            )

    def test_qr_split0(self):
        split = 0
        for procs_to_merge in [0, 2, 3]:
            for mode in ["reduced", "r"]:
                # split = 0 can be handeled only for tall skinny matrices s.t. the local chunks are at least square too
                for shape in [(40 * ht.MPI_WORLD.size + 1, 40), (40 * ht.MPI_WORLD.size, 20)]:
                    for dtype in [ht.float32, ht.float64]:
                        dtypetol = 1e-3 if dtype == ht.float32 else 1e-6
                        mat = ht.random.randn(*shape, dtype=dtype, split=split)

                        qr = ht.linalg.qr(mat, mode=mode, procs_to_merge=procs_to_merge)

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

                        # compare with torch qr, due to different signs we can only compare absolute values
                        mat_t = mat.resplit_(None).larray
                        q_t, r_t = torch.linalg.qr(mat_t, mode=mode)
                        r_ht = qr.R.resplit_(None).larray
                        self.assertTrue(
                            torch.allclose(
                                torch.abs(r_t), torch.abs(r_ht), atol=dtypetol, rtol=dtypetol
                            )
                        )
                        if mode == "reduced":
                            q_ht = qr.Q.resplit_(None).larray
                            self.assertTrue(
                                torch.allclose(
                                    torch.abs(q_t), torch.abs(q_ht), atol=dtypetol, rtol=dtypetol
                                )
                            )

    def test_wronginputs(self):
        # test wrong input type
        with self.assertRaises(TypeError):
            ht.linalg.qr([1, 2, 3])
        # wrong data type for mode
        with self.assertRaises(TypeError):
            ht.linalg.qr(ht.zeros((10, 10)), mode=1)
        # test wrong mode
        with self.assertRaises(ValueError):
            ht.linalg.qr(ht.zeros((10, 10)), mode="full")
        # wrong dtype for procs_to_merge
        with self.assertRaises(TypeError):
            ht.linalg.qr(ht.zeros((10, 10)), procs_to_merge="abc")
        # test wrong procs_to_merge
        with self.assertRaises(ValueError):
            ht.linalg.qr(ht.zeros((10, 10)), procs_to_merge=1)
        # test wrong shape
        with self.assertRaises(ValueError):
            ht.linalg.qr(ht.zeros((10, 10, 10)))
        # test wrong dtype
        with self.assertRaises(TypeError):
            ht.linalg.qr(ht.zeros((10, 10), dtype=ht.int32))
        # test wrong shape for split=0
        if ht.MPI_WORLD.size > 1:
            with self.assertRaises(ValueError):
                ht.linalg.qr(ht.zeros((10, 10), split=0))
