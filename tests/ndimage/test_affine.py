import pytest
import scipy.ndimage as ndimg
import numpy as np
import heat as ht
import heat.ndimage.affine as affine
from heat.testing.basic_test import TestCase
from heat.ndimage.util import create_checker


class TestAffine:

    @classmethod
    def setup_class(cls):
        TestCase.setUpClass()

        BULK = 5
        # matrix needs identity on color axis. color is last dimension
        cls.offset_2d = ht.random.random((3), dtype=ht.float32)
        cls.offset_2d[2] = 0
        cls.offset_3d = ht.random.random((4), dtype=ht.float32)
        cls.offset_3d[3] = 0
        cls.offset_3d_bulk = ht.random.random((BULK, 4), dtype=ht.float32)
        cls.offset_3d_bulk[:, 3] = 0

        cls.image_2d_1 = create_checker((64, 64), 8, min_value=50, max_value=200)
        cls.image_2d_2 = create_checker((64, 64), 4, min_value=50, max_value=200)
        cls.image_2d_bulk = ht.stack((cls.image_2d_1, cls.image_2d_2))
        cls.image_3d_1 = create_checker((32, 64, 64), 8, min_value=50, max_value=200)
        cls.image_3d_2 = create_checker((32, 64, 64), 4, min_value=50, max_value=200)
        cls.image_3d_bulk = create_checker(
            (BULK, 32, 32, 64), 8, min_value=50, max_value=200
        )

        cls.matrix_2d = ht.random.random((3, 3), dtype=ht.float32)
        cls.matrix_2d[:, 2] = ht.array([0, 0, 1], dtype=ht.float32)
        cls.matrix_2d[2, :] = ht.array([0, 0, 1], dtype=ht.float32)

        cls.matrix_3d = ht.random.random((4, 4), dtype=ht.float32)
        cls.matrix_3d[:, 3] = ht.array([0, 0, 0, 1], dtype=ht.float32)
        cls.matrix_3d[3, :] = ht.array([0, 0, 0, 1], dtype=ht.float32)

        cls.matrix_3d_bulk = ht.random.random((BULK, 4, 4), dtype=ht.float32)
        cls.matrix_3d_bulk[:, :, 3] = ht.array([0, 0, 0, 1], dtype=ht.float32)
        cls.matrix_3d_bulk[:, 3, :] = ht.array([0, 0, 0, 1], dtype=ht.float32)

    @staticmethod
    def default_testing_setup(image, matrix, offset, order, mode):
        matrix_affine = ht.hstack((matrix, offset[:, None]))

        with_offset_result: ht.DNDarray = affine.affine_transform(
            image,
            matrix,
            offset=offset,
            order=order,
            mode=mode,
        )
        combined_result: ht.DNDarray = affine.affine_transform(
            image, matrix_affine, order=order, mode=mode
        )
        assert ht.equal(with_offset_result, combined_result)

        with_offset_comparison = ndimg.affine_transform(
            image.numpy(), matrix.numpy(), offset=offset.numpy(), order=order, mode=mode
        )
        combined_comparison = ndimg.affine_transform(
            image.numpy(), matrix_affine.numpy(), order=order, mode=mode
        )

        assert np.allclose(
            with_offset_result.numpy(), with_offset_comparison, rtol=0, atol=0.01
        )
        assert np.allclose(
            combined_result.numpy(), combined_comparison, rtol=0, atol=0.01
        )

    @staticmethod
    def bulk_testing_setup(image, matrix, offset, order, mode):

        offset_stack = ht.expand_dims(offset, offset.ndim)
        # offset.resplit_(matrix.split)
        matrix_affine = ht.concatenate([matrix, offset_stack], axis=(matrix.ndim - 1))

        with_offset_result: ht.DNDarray = affine.affine_transform(
            image,
            matrix,
            offset=offset,
            order=order,
            mode=mode,
        )
        combined_result: ht.DNDarray = affine.affine_transform(
            image, matrix_affine, order=order, mode=mode
        )
        assert ht.equal(with_offset_result, combined_result)

        with_offset_comparison = [
            ndimg.affine_transform(
                img.numpy(), mat.numpy(), offset=off.numpy(), order=order, mode=mode
            )
            for img, mat, off in zip(image, matrix, offset)
        ]
        combined_comparison = [
            ndimg.affine_transform(img.numpy(), mat.numpy(), order=order, mode=mode)
            for img, mat in zip(image, matrix_affine)
        ]

        # visual_compare_2d(combined_result[:, 32, :], combined_comparison[:, 32, :])
        for res, comp in zip(with_offset_result, with_offset_comparison):
            assert np.allclose(res.numpy(), comp, rtol=0, atol=0.01)

        for res, comp in zip(combined_result, combined_comparison):
            assert np.allclose(res.numpy(), comp, rtol=0, atol=0.01)

    @pytest.mark.parametrize("order", [0, 1])
    @pytest.mark.parametrize("mode", ["grid-constant", "mirror", "nearest"])
    def test_affine_2d(self, order, mode):
        image = self.image_2d_1
        matrix = self.matrix_2d
        offset = self.offset_2d

        TestAffine.default_testing_setup(image, matrix, offset, order, mode)

    @pytest.mark.parametrize("order", [0, 1])
    @pytest.mark.parametrize("mode", ["grid-constant", "mirror", "nearest"])
    def test_affine_3d(self, order, mode):
        image = self.image_3d_1
        matrix = self.matrix_3d
        offset = self.offset_3d

        TestAffine.default_testing_setup(image, matrix, offset, order, mode)

    @pytest.mark.parametrize("order", [0, 1])
    @pytest.mark.parametrize("mode", ["grid-constant", "mirror", "nearest"])
    def test_affine_3d_bulk(self, order, mode):
        image = self.image_3d_bulk
        matrix = self.matrix_3d_bulk
        offset = self.offset_3d_bulk

        TestAffine.bulk_testing_setup(image, matrix, offset, order, mode)

    def test_affine_split(self):

        matrix = ht.array(
            (
                [[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0.2, 0]],
                [[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0]],
            ),
            split=None,
        )
        # only split=3 or split=0 should not error because other axes get changed
        # split=3 and not split=1, because the input gets permuted internally so the axis assignment
        # matches the result from scipy

        rnd_image_base = ht.random.random((2, 64, 64, 3), dtype=ht.float32, split=None)
        split_none = affine.affine_transform(rnd_image_base, matrix)

        rnd_image = ht.resplit(rnd_image_base, 0)
        split_0 = affine.affine_transform(rnd_image, matrix)
        assert ht.equal(split_none, split_0)

        rnd_image_1 = ht.resplit(rnd_image_base, 3)
        split_1 = affine.affine_transform(rnd_image_1, matrix)
        assert ht.equal(split_none, split_1)

        rnd_image = ht.resplit(rnd_image, 2)
        with pytest.raises(RuntimeError):
            affine.affine_transform(rnd_image, matrix)

        rnd_image = ht.resplit(rnd_image, 1)
        with pytest.raises(RuntimeError):
            affine.affine_transform(rnd_image, matrix)
