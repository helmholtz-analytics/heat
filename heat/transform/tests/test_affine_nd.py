import unittest
import numpy as np
import heat as ht
import torch
from heat.transform.affine import affine_transform


class TestAffineND(unittest.TestCase):

    # ============================================================
    # 2D BASIC TESTS
    # ============================================================

    def test_identity_2d(self):
        x = ht.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])

        M = torch.tensor([[1, 0, 0],
                          [0, 1, 0]], dtype=torch.float32)

        y = affine_transform(x, M)
        self.assertTrue(ht.allclose(x, y))

    def test_translation_2d(self):
        x = ht.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])

        # shift right by +1 pixel
        M = torch.tensor([[1, 0, 1],
                          [0, 1, 0]], dtype=torch.float32)

        y = affine_transform(x, M)

        expected = ht.array([[1., 1., 2.],
                             [4., 4., 5.],
                             [7., 7., 8.]])

        self.assertTrue(ht.allclose(y, expected))

    # ============================================================
    # 2D SCALING + ROTATION
    # ============================================================

    def test_scale_2d(self):
        x = ht.array([[1., 2.],
                      [3., 4.]])

        # scale by factor 2 (nearest → same)
        M = torch.tensor([[2., 0., 0.],
                          [0., 2., 0.]], dtype=torch.float32)

        y = affine_transform(x, M)

        expected = ht.array([[1., 2.],
                             [3., 4.]])

        self.assertTrue(ht.allclose(y, expected))

    def test_rotation_2d_90deg(self):
        x = ht.array([[0, 1, 0],
                      [0, 1, 0],
                      [0, 1, 0]], dtype=ht.float32)

        M = torch.tensor([
            [0., -1., 0.],
            [1.,  0., 0.]
        ], dtype=torch.float32)

        y = affine_transform(x, M)

        expected = ht.array([[0, 0, 0],
                             [1, 1, 1],
                             [0, 0, 0]])

        self.assertTrue(ht.allclose(y, expected))

    # ============================================================
    # 3D BASIC TESTS
    # ============================================================

    def test_identity_3d(self):
        x = ht.arange(27, dtype=ht.float32).reshape((3, 3, 3))

        M = torch.tensor([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0]], dtype=torch.float32)

        y = affine_transform(x, M)
        self.assertTrue(ht.allclose(x, y))

    def test_translation_3d(self):
        x = ht.arange(27, dtype=ht.float32).reshape((3, 3, 3))

        M = torch.tensor([[1, 0, 0, 1],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0]], dtype=torch.float32)

        y = affine_transform(x, M)

        expected = x.copy()
        expected[0] = x[0]
        expected[1] = x[0]
        expected[2] = x[1]

        self.assertTrue(ht.allclose(y, expected))

    # ============================================================
    # ND 4D TEST
    # ============================================================

    def test_translation_4d(self):
        x_np = np.arange(18).reshape((1, 2, 3, 3))
        x = ht.array(x_np)

        M = torch.tensor([
            [1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ], dtype=torch.float32)

        y = affine_transform(x, M)

        expected = x.copy()
        expected.larray[:, 1] = expected.larray[:, 0]

        self.assertTrue(ht.allclose(y, expected))

    # ============================================================
    # BATCH + CHANNEL TESTS
    # ============================================================

    def test_batch_channel_2d(self):
        x_np = np.array([
            [[1., 2.],
             [3., 4.]],

            [[10., 20.],
             [30., 40.]]
        ])  # (N=2, H, W)

        x = ht.array(x_np)

        M = torch.tensor([[1, 0, 0],
                          [0, 1, 0]], dtype=torch.float32)

        y = affine_transform(x, M)

        self.assertTrue(ht.allclose(x, y))

    # ============================================================
    # DISTRIBUTED TEST
    # ============================================================

    def test_distributed_identity(self):
        x = ht.arange(16).reshape((4, 4), split=0).astype(ht.float32)

        M = torch.tensor([[1, 0, 0],
                          [0, 1, 0]], dtype=torch.float32)

        y = affine_transform(x, M)

        self.assertTrue(ht.allclose(x, y))


if __name__ == "__main__":
    unittest.main()
