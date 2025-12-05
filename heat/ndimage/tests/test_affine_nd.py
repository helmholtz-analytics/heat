import unittest
import heat as ht
import numpy as np
from heat.ndimage.affine import affine_transform
"""
Unit tests for heat.ndimage.affine_transform.
These tests validate 2D/3D affine transforms, interpolation, and padding modes.
"""

# ============================================================
# BACKWARD-WARP NEAREST 2D
# ============================================================

class TestAffineNearest2D(unittest.TestCase):

    def test_identity(self):
        x = ht.array([[1.,2.,3.],
                      [4.,5.,6.],
                      [7.,8.,9.]])

        M = [[1,0,0],[0,1,0]]

        y = affine_transform(x, M, order=0)
        self.assertTrue(ht.allclose(y, x))

    def test_translation_right(self):
        x = ht.array([[1.,2.,3.],
                      [4.,5.,6.],
                      [7.,8.,9.]])

        # shift RIGHT by +1
        M = [[1,0,1],
             [0,1,0]]

        y = affine_transform(x, M, order=0)

        expected = ht.array([
            [1.,1.,2.],
            [4.,4.,5.],
            [7.,7.,8.]
        ])  # backward warp

        self.assertTrue(ht.allclose(y, expected))

    def test_translation_down(self):
        x = ht.array([[1.,2.,3.],
                      [4.,5.,6.],
                      [7.,8.,9.]])

        # shift DOWN by +1
        M = [[1,0,0],
             [0,1,1]]

        y = affine_transform(x, M, order=0)

        expected = ht.array([
            [1.,2.,3.],
            [1.,2.,3.],
            [4.,5.,6.]
        ])  # backward warp

        self.assertTrue(ht.allclose(y, expected))

    def test_rotation_90(self):
        x = ht.array([
            [0.,1.,0.],
            [0.,1.,0.],
            [0.,1.,0.]
        ])

        # CCW 90° (standard rotation matrix)
        M = [[0,-1,2],
             [1, 0,0]]

        y = affine_transform(x, M, order=0)

        expected = ht.array([
            [0.,0.,0.],
            [1.,1.,1.],
            [0.,0.,0.]
        ])

        self.assertTrue(ht.allclose(y, expected))

    def test_scale_2x(self):
        x = ht.array([[1.,2.],
                      [3.,4.]])

        M = [[2,0,0],
             [0,2,0]]

        y = affine_transform(x, M, order=0)

        expected = ht.array([
            [1.,1.],
            [1.,1.]
        ])

        self.assertTrue(ht.allclose(y, expected))


# ============================================================
# BILINEAR 2D
# ============================================================

class TestAffineBilinear2D(unittest.TestCase):

    def test_identity(self):
        x = ht.array([[1.,2.],
                      [3.,4.]])
        M = [[1,0,0],[0,1,0]]

        y = affine_transform(x, M, order=1)
        self.assertTrue(ht.allclose(x, y))

    def test_half_pixel_shift(self):
        x = ht.array([[1.,2.],
                      [3.,4.]])

        M = [[1,0,0.5],
             [0,1,0.5]]

        y = affine_transform(x, M, order=1)

        # center bilinear interpolation
        self.assertAlmostEqual(float(y[1,1]), 2.5, places=2)

    def test_smooth_gradient(self):
        x_np = np.linspace(0,1,9).reshape(3,3)
        x = ht.array(x_np)

        M = [[1,0,0.2],
             [0,1,0.3]]

        y = affine_transform(x, M, order=1)

        self.assertTrue(float(y[1,1]) > float(y[0,0]))


# ============================================================
# PADDING TESTS
# ============================================================

class TestAffinePadding(unittest.TestCase):

    def test_constant(self):
        x = ht.array([[1.,2.],[3.,4.]])
        M = [[1,0,1],[0,1,0]]

        y = affine_transform(x, M, order=0, mode="constant", constant_value=-1)

        expected = ht.array([
            [-1.,1.],
            [-1.,3.]
        ])

        self.assertTrue(ht.allclose(y, expected))

    def test_wrap(self):
        x = ht.array([[1.,2.],[3.,4.]])
        M = [[1,0,1],[0,1,0]]

        y = affine_transform(x, M, order=0, mode="wrap")

        expected = ht.array([
            [2.,1.],
            [4.,3.]
        ])

        self.assertTrue(ht.allclose(y, expected))

    def test_reflect(self):
        x = ht.array([[1.,2.],[3.,4.]])
        M = [[1,0,-1],[0,1,0]]

        y = affine_transform(x, M, order=0, mode="reflect")

        expected = ht.array([
            [2.,1.],
            [4.,3.]
        ])

        self.assertTrue(ht.allclose(y, expected))


# ============================================================
# EXPAND=True
# ============================================================

class TestAffineExpand(unittest.TestCase):

    def test_expand_rotation(self):
        x = ht.array([
            [1.,0.],
            [0.,1.]
        ])

        M = [[0,-1,0],
             [1,0,0]]

        y = affine_transform(x, M, expand=True)

        self.assertTrue(y.shape[0] >= 2 and y.shape[1] >= 2)


# ============================================================
# 3D TESTS (correct backward warp)
# ============================================================

class TestAffine3D(unittest.TestCase):

    def test_identity_3d(self):
        x = ht.arange(27).reshape((3,3,3)).astype(ht.float32)
        M = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]

        y = affine_transform(x, M)
        self.assertTrue(ht.allclose(x, y))

    def test_translate_3d_x(self):
        x = ht.arange(27).reshape((3,3,3)).astype(ht.float32)

        M = [[1,0,0,1],
             [0,1,0,0],
             [0,0,1,0]]

        y = affine_transform(x, M)

        # backward warp
        expected = ht.zeros((3,3,3), dtype=ht.float32)
        expected[:,:,1:] = x[:,:,:-1]

        self.assertTrue(ht.allclose(y, expected))


if __name__ == "__main__":
    unittest.main()
