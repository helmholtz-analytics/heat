import heat as ht
import unittest
import torch
import numpy as np

from ...tests.test_suites.basic_test import TestCase


class TestQR(TestCase):
    def test_qr_split1orNone(self):
        if self.is_mps:
            dtypes = [ht.float32]
        else:
            dtypes = [ht.float32, ht.float64, ht.complex64, ht.complex128]

        for split in [1, None]:
            for mode in ["reduced", "r"]:
                # note that split = 1 can be handled for arbitrary shapes
                for shape in [
                    (20 * ht.MPI_WORLD.size + 1, 40 * ht.MPI_WORLD.size),
                    (20 * ht.MPI_WORLD.size, 20 * ht.MPI_WORLD.size),
                    (40 * ht.MPI_WORLD.size - 1, 20 * ht.MPI_WORLD.size),
                ]:
                    for dtype in dtypes:
                        with self.subTest(f'{dtype=} {shape=} {mode=}'):
                            dtypetol = 1e-3 if dtype in [ht.float32, ht.complex64] else 1e-6
                            mat = ht.random.randn(*shape, dtype=dtype, split=split)
                            qr = ht.linalg.qr(mat, mode=mode)

                            if mode == "reduced":
                                self.assertTrue(
                                    ht.allclose(qr.Q @ qr.R, mat, atol=dtypetol, rtol=dtypetol)
                                )
                                self.assertIsInstance(qr.Q, ht.DNDarray)

                                # test if Q is orthogonal / hermitian
                                self.assertTrue(
                                    ht.allclose(
                                        ht.conj(qr.Q).T @ qr.Q,
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
        if self.is_mps:
            dtypes = [ht.float32]
        else:
            dtypes = [ht.float32, ht.float64, ht.complex64, ht.complex128]
        split = 0
        for procs_to_merge in [0, 2, 3]:
            for mode in ["reduced", "r"]:
                # split = 0 can be handled only for tall skinny matrices s.t. the local chunks are at least square too
                for shape in [
                    (20 * ht.MPI_WORLD.size + 1, 40 * ht.MPI_WORLD.size),
                    (20 * ht.MPI_WORLD.size, 20 * ht.MPI_WORLD.size),
                    (40 * ht.MPI_WORLD.size - 1, 20 * ht.MPI_WORLD.size),
                ]:
                    for dtype in dtypes:
                        with self.subTest(f'{dtype=} {shape=} {mode=} {procs_to_merge=}'):
                            dtypetol = 1e-3 if dtype in [ht.float32, ht.complex64] else 1e-6
                            mat = ht.random.randn(*shape, dtype=dtype, split=split)

                            qr = ht.linalg.qr(mat, mode=mode, procs_to_merge=procs_to_merge)

                            if mode == "reduced":
                                self.assertTrue(
                                    ht.allclose(qr.Q @ qr.R, mat, atol=dtypetol, rtol=dtypetol)
                                )
                                self.assertIsInstance(qr.Q, ht.DNDarray)

                                # test if Q is orthogonal / hermitian
                                self.assertTrue(
                                    ht.allclose(
                                        ht.conj(qr.Q).T @ qr.Q,
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

    def test_batched_qr_splitNone(self):
        # two batch dimensions, float64 data type, "split = None" (split batch axis)
        x = ht.random.rand(2, 2 * ht.MPI_WORLD.size, 10, 9, dtype=ht.float32, split=1)
        _, r = ht.linalg.qr(x, mode="r")
        self.assertEqual(r.shape, (2, 2 * ht.MPI_WORLD.size, 9, 9))
        self.assertEqual(r.split, 1)

    def test_batched_qr_split1(self):
        # skip float64 tests on MPS
        if not self.is_mps:
            # two batch dimensions, float64 data type, "split = 1" (last dimension)
            x = ht.random.rand(3, 2, 50, ht.MPI_WORLD.size * 5 + 3, dtype=ht.float64, split=3)
            q, r = ht.linalg.qr(x)
            batched_id = ht.stack([ht.eye(q.shape[3], dtype=ht.float64) for _ in range(6)]).reshape(
                3, 2, q.shape[3], q.shape[3]
            )

            self.assertTrue(
                ht.allclose(q.transpose([0, 1, 3, 2]) @ q, batched_id, atol=1e-6, rtol=1e-6)
            )
            self.assertTrue(ht.allclose(q @ r, x, atol=1e-6, rtol=1e-6))

    def test_batched_qr_split0(self):
        # one batch dimension, float32 data type, "split = 0" (second last dimension)
        x = ht.random.randn(
            8, ht.MPI_WORLD.size * 10 + 3, ht.MPI_WORLD.size * 10 - 1, dtype=ht.float32, split=1
        )
        q, r = ht.linalg.qr(x)
        batched_id = ht.stack([ht.eye(q.shape[2], dtype=ht.float32) for _ in range(q.shape[0])])

        self.assertTrue(ht.allclose(q.transpose([0, 2, 1]) @ q, batched_id, atol=1e-3, rtol=1e-3))
        self.assertTrue(ht.allclose(q @ r, x, atol=1e-3, rtol=1e-3))

    def test_wronginputs(self):
        # test wrong input type
        with self.assertRaises(TypeError):
            ht.linalg.qr([1, 2, 3])
        # wrong data type for mode
        with self.assertRaises(TypeError):
            ht.linalg.qr(ht.zeros((10, 10)), mode=1)
        # test wrong mode (such mode is not available for Torch)
        with self.assertRaises(ValueError):
            ht.linalg.qr(ht.zeros((10, 10)), mode="full")
        # test mode that is available for Torch but not for Heat
        with self.assertRaises(NotImplementedError):
            ht.linalg.qr(ht.zeros((10, 10)), mode="complete")
        with self.assertRaises(NotImplementedError):
            ht.linalg.qr(ht.zeros((10, 10)), mode="raw")
        # wrong dtype for procs_to_merge
        with self.assertRaises(TypeError):
            ht.linalg.qr(ht.zeros((10, 10)), procs_to_merge="abc")
        # test wrong procs_to_merge
        with self.assertRaises(ValueError):
            ht.linalg.qr(ht.zeros((10, 10)), procs_to_merge=1)
        # test wrong dtype
        with self.assertRaises(TypeError):
            ht.linalg.qr(ht.zeros((10, 10), dtype=ht.int32))
