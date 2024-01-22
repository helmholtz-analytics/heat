import heat as ht
import unittest
import torch

from ...tests.test_suites.basic_test import TestCase


@unittest.skipIf(torch.cuda.is_available() and torch.version.hip, "not supported for HIP")
class TestQR(TestCase):
    for split in [1, None]:
        for calc_r in [True, False]:
            for calc_q in [True, False]:
                for full_q in [True, False]:
                    for shape in [
                        (2 * ht.MPI_WORLD.size, 4 * ht.MPI_WORLD.size),
                        (2 * ht.MPI_WORLD.size, 2 * ht.MPI_WORLD.size),
                        (4 * ht.MPI_WORLD.size, 2 * ht.MPI_WORLD.size),
                    ]:
                        for dtype in [ht.float32, ht.float64]:
                            mat = ht.random.randn(*shape, dtype=dtype, split=split)
                            qr = ht.linalg.qr(mat, calc_r=calc_r, calc_q=calc_q, full_q=full_q)
                            print(qr)
