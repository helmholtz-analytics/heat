"""
Unit tests for heat.ndimage.affine_transform.

This test suite verifies the correctness of the affine_transform function
under a variety of conditions relevant to the Heat library.

Covered aspects:
- 2D affine transforms with nearest-neighbor interpolation
- 2D affine transforms with bilinear interpolation
- Boundary handling via different padding modes
- 3D affine transforms (volumetric data)
- Correct behavior for distributed arrays using different split configurations

Important notes:
- All transforms are tested using *backward warping* semantics
- All tests are executed for split=None and valid split axes to ensure
  distributed correctness
- expand=True is currently a no-op and only shape behavior is validated
"""

import unittest
import numpy as np
import heat as ht
from heat.ndimage.affine import affine_transform


# ============================================================
# Helper: valid split configurations
# ============================================================

def valid_splits(ndim):
    """
    Yield all valid Heat split configurations for an array of dimension `ndim`.

    - None  → non-distributed (local) array
    - 0..ndim-1 → distributed along the corresponding axis
    """
    yield None
    for s in range(ndim):
        yield s


# ============================================================
# BACKWARD-WARP NEAREST 2D
# ============================================================

class TestAffineNearest2D(unittest.TestCase):
    """
    Tests for 2D affine transforms using nearest-neighbor interpolation (order=0).

    These tests focus on:
    - identity transforms
    - pure translations
    - correct backward-warp behavior
    """

    def test_identity(self):
        """
        Identity transform should return the input unchanged.

        This verifies:
        - basic correctness of the affine pipeline
        - behavior under all split configurations
        """
        data = [
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]
        ]

        M = [[1, 0, 0],
             [0, 1, 0]]

        for split in valid_splits(2):
            x = ht.array(data, split=split)
            y = affine_transform(x, M, order=0)
            self.assertTrue(ht.allclose(x, y))

    def test_translation_right(self):
        """
        Translation by +1 in x-direction.

        Because affine_transform uses backward warping and nearest padding,
        values are copied from the left neighbor and clamped at the boundary.
        """
        data = [
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]
        ]

        M = [[1, 0, 1],
             [0, 1, 0]]

        expected = ht.array([
            [1., 1., 2.],
            [4., 4., 5.],
            [7., 7., 8.]
        ])

        for split in valid_splits(2):
            x = ht.array(data, split=split)
            y = affine_transform(x, M, order=0)
            self.assertTrue(ht.allclose(y, expected))

    def test_translation_down(self):
        """
        Translation by +1 in y-direction.

        Verifies:
        - correct backward warp semantics
        - nearest-neighbor clamping at the top boundary
        """
        data = [
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]
        ]

        M = [[1, 0, 0],
             [0, 1, 1]]

        expected = ht.array([
            [1., 2., 3.],
            [1., 2., 3.],
            [4., 5., 6.]
        ])

        for split in valid_splits(2):
            x = ht.array(data, split=split)
            y = affine_transform(x, M, order=0)
            self.assertTrue(ht.allclose(y, expected))


# ============================================================
# BILINEAR 2D
# ============================================================

class TestAffineBilinear2D(unittest.TestCase):
    """
    Tests for 2D affine transforms using bilinear interpolation (order=1).

    These tests verify:
    - identity behavior
    - smooth interpolation
    - non-nearest interpolation effects
    """

    def test_identity(self):
        """
        Identity transform with bilinear interpolation.

        Should exactly reproduce the input values.
        """
        data = [
            [1., 2.],
            [3., 4.]
        ]

        M = [[1, 0, 0],
             [0, 1, 0]]

        for split in valid_splits(2):
            x = ht.array(data, split=split)
            y = affine_transform(x, M, order=1)
            self.assertTrue(ht.allclose(x, y))

    def test_half_pixel_shift(self):
        """
        Half-pixel translation in both x and y.

        This test verifies:
        - bilinear interpolation is applied
        - the center pixel is averaged correctly
        """
        data = [
            [1., 2.],
            [3., 4.]
        ]

        M = [[1, 0, 0.5],
             [0, 1, 0.5]]

        for split in valid_splits(2):
            x = ht.array(data, split=split)
            y = affine_transform(x, M, order=1)
            self.assertAlmostEqual(float(y[0, 1, 1]), 2.5, places=2)

    def test_smooth_gradient(self):
        """
        Small translation applied to a smooth gradient image.

        Verifies that interpolation preserves ordering and smoothness
        rather than copying nearest values.
        """
        x_np = np.linspace(0, 1, 9).reshape(3, 3)

        M = [[1, 0, 0.2],
             [0, 1, 0.3]]

        for split in valid_splits(2):
            x = ht.array(x_np, split=split)
            y = affine_transform(x, M, order=1)
            self.assertTrue(float(y[0, 1, 1]) > float(y[0, 0, 0]))


# ============================================================
# PADDING MODES
# ============================================================

class TestAffinePadding(unittest.TestCase):
    """
    Tests for different padding modes applied during sampling.

    Verifies:
    - constant padding
    - wrap padding
    - reflect padding
    """

    def test_constant(self):
        """
        Constant padding fills out-of-bounds values with a fixed constant.
        """
        data = [
            [1., 2.],
            [3., 4.]
        ]

        M = [[1, 0, 1],
             [0, 1, 0]]

        expected = ht.array([
            [-1., 1.],
            [-1., 3.]
        ])

        for split in valid_splits(2):
            x = ht.array(data, split=split)
            y = affine_transform(
                x, M, order=0, mode="constant", constant_value=-1
            )
            self.assertTrue(ht.allclose(y, expected))

    def test_wrap(self):
        """
        Wrap padding maps out-of-bounds indices cyclically.
        """
        data = [
            [1., 2.],
            [3., 4.]
        ]

        M = [[1, 0, 1],
             [0, 1, 0]]

        expected = ht.array([
            [2., 1.],
            [4., 3.]
        ])

        for split in valid_splits(2):
            x = ht.array(data, split=split)
            y = affine_transform(x, M, order=0, mode="wrap")
            self.assertTrue(ht.allclose(y, expected))

    def test_reflect(self):
        """
        Reflect padding mirrors indices at the boundary.
        """
        data = [
            [1., 2.],
            [3., 4.]
        ]

        M = [[1, 0, -1],
             [0, 1, 0]]

        expected = ht.array([
            [2., 1.],
            [4., 3.]
        ])

        for split in valid_splits(2):
            x = ht.array(data, split=split)
            y = affine_transform(x, M, order=0, mode="reflect")
            self.assertTrue(ht.allclose(y, expected))


# ============================================================
# EXPAND FLAG
# ============================================================

class TestAffineExpand(unittest.TestCase):
    """
    Tests for expand=True.

    Note:
    - expand is currently implemented as a no-op
    - only shape behavior is verified
    """

    def test_expand_rotation(self):
        x = ht.array([
            [1., 0.],
            [0., 1.]
        ])

        M = [[0, -1, 0],
             [1,  0, 0]]

        y = affine_transform(x, M, expand=True)

        self.assertEqual(y.shape, (1, *x.shape))
        self.assertTrue(y.shape[1] >= 2)


# ============================================================
# 3D AFFINE TRANSFORMS
# ============================================================

class TestAffine3D(unittest.TestCase):
    """
    Tests for 3D affine transforms (volumetric data).

    Verifies:
    - identity transform in 3D
    - translation along x-axis
    - correct backward-warp behavior
    """

    def test_identity_3d(self):
        """
        Identity transform on a 3D volume.

        Ensures:
        - no data corruption
        - correctness under all split configurations
        """
        M = [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0]]

        for split in valid_splits(3):
            x = ht.arange(27).reshape((3, 3, 3)).astype(ht.float32)
            x = ht.array(x, split=split)
            y = affine_transform(x, M)
            self.assertTrue(ht.allclose(x, y))

    def test_translate_3d_x(self):
        """
        Translation by +1 along the x-axis in 3D.

        Because backward warping with nearest padding is used:
        - values are copied from the left neighbor
        - the boundary is clamped
        """
        M = [[1, 0, 0, 1],
             [0, 1, 0, 0],
             [0, 0, 1, 0]]

        x = ht.arange(27).reshape((3, 3, 3)).astype(ht.float32)

        expected = ht.zeros((3, 3, 3), dtype=ht.float32)
        expected[:, :, 0] = x[:, :, 0]
        expected[:, :, 1:] = x[:, :, :-1]

        for split in valid_splits(3):
            x_split = ht.array(x, split=split)
            y = affine_transform(x_split, M)
            self.assertTrue(ht.allclose(y[0], expected))


if __name__ == "__main__":
    unittest.main()
