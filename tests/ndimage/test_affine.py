import pytest
import scipy.ndimage as ndimg
import numpy as np
import heat as ht
import heat.ndimage.affine as affine
from heat.testing.basic_test import TestCase
from heat.ndimage.util import create_checker, root_mean_square_error

class TestAffine:

    @classmethod
    def setup_class(cls):
        TestCase.setUpClass()

        BULK = 5
        ht.random.seed = 149156
        # matrix needs identity on color axis. color is last dimension
        # cls.offset_2d = ht.random.random((3), dtype=ht.float32) * 10 - 5
        # cls.offset_2d[2] = 0
        cls.offset_2d = ht.array([ 4.4599, -1.9924,  0.0000], dtype=ht.float32)
        # cls.offset_3d = ht.random.random((4), dtype=ht.float32) * 10 - 5
        # cls.offset_3d[3] = 0
        cls.offset_3d = ht.array([ 4.9231, -2.4895, -3.5396,  0.0000], dtype=ht.float32)
        cls.offset_3d_bulk = ht.random.random((BULK, 4), dtype=ht.float32) * 10 - 5
        cls.offset_3d_bulk[:, 3] = 0
        cls.offset_3d_bulk = ht.array([[-2.89095, -3.22055,  3.23815,  0.0000],
          [ 0.25775,  2.41225, -0.09195,  0.0000],
          [ 4.27525, -2.41565,  0.50525,  0.0000],
          [ 3.94495,  4.38435,  1.40695,  0.0000],
          [-2.55725, -4.18895,  3.75415,  0.0000]],
                                        dtype=ht.float32)

        cls.image_2d_1 = create_checker((64, 64), 8, min_value=50, max_value=200)
        cls.image_2d_2 = create_checker((64, 64), 4, min_value=50, max_value=200)
        cls.image_3d_1 = create_checker((32, 64, 64), 8, min_value=50, max_value=200)
        cls.image_3d_2 = create_checker((32, 64, 64), 4, min_value=50, max_value=200)
        cls.image_3d_bulk = create_checker(
            (BULK, 32, 32, 64), 8, min_value=50, max_value=200
        )

        # cls.matrix_2d = ht.random.random((3, 3), dtype=ht.float32) * 4 - 2
        # cls.matrix_2d[:, 2] = ht.array([0, 0, 1], dtype=ht.float32)
        # cls.matrix_2d[2, :] = ht.array([0, 0, 1], dtype=ht.float32)
        cls.matrix_2d = ht.array([[0.6564, 0.9997, 0.0000],
                                  [1.8381, 0.0900, 0.0000],
                                  [0.0000, 0.0000, 1.0000]],
                                 dtype=ht.float32)

        # cls.matrix_3d = ht.random.random((4, 4), dtype=ht.float32) * 4 - 2
        # cls.matrix_3d[:, 3] = ht.array([0, 0, 0, 1], dtype=ht.float32)
        # cls.matrix_3d[3, :] = ht.array([0, 0, 0, 1], dtype=ht.float32)
        cls.matrix_3d = ht.array([[ 1.2317,  1.5504, -1.2923,  0.0000],
                                  [ 1.2198, -1.1036,  0.2422,  0.0000],
                                  [ 0.4301, -0.2433,  0.1062,  0.0000],
                                  [ 0.0000,  0.0000,  0.0000,  1.0000]],
                                 dtype=ht.float32)

        # cls.matrix_3d_bulk = ht.random.random((BULK, 4, 4), dtype=ht.float32) * 4 - 2
        # cls.matrix_3d_bulk[:, :, 3] = ht.array([0, 0, 0, 1], dtype=ht.float32)
        # cls.matrix_3d_bulk[:, 3, :] = ht.array([0, 0, 0, 1], dtype=ht.float32)
        cls.matrix_3d_bulk = ht.array([[[ 0.7074,  1.0873, -1.1428,  0.0000],
                                        [ 0.5287,  0.3751, -0.4493,  0.0000],
                                        [-0.6325, -0.8122,  0.2367,  0.0000],
                                        [ 0.0000,  0.0000,  0.0000,  1.0000]],

                                        [[-1.2879,  1.2259, -0.7978,  0.0000],
                                        [-0.2755,  0.2205,  1.7386,  0.0000],
                                        [-1.7849,  0.1429, -1.0597,  0.0000],
                                        [ 0.0000,  0.0000,  0.0000,  1.0000]],

                                        [[ 0.2888, -1.0211, -1.6024,  0.0000],
                                        [ 1.0784, -1.7459,  1.4299,  0.0000],
                                        [ 0.6882,  1.4669, -1.9000,  0.0000],
                                        [ 0.0000,  0.0000,  0.0000,  1.0000]],

                                        [[ 1.5793,  1.1889,  0.7961,  0.0000],
                                        [-0.5300,  1.4195, -0.7138,  0.0000],
                                        [-0.1458, -0.7458,  1.2369,  0.0000],
                                        [ 0.0000,  0.0000,  0.0000,  1.0000]],

                                        [[-1.6955, -0.2650,  0.2087,  0.0000],
                                        [-1.3514,  1.9030,  1.5575,  0.0000],
                                        [-0.8669,  0.5857,  0.7428,  0.0000],
                                        [ 0.0000,  0.0000,  0.0000,  1.0000]]],
                                     dtype=ht.float32)

    @staticmethod
    def default_testing_setup(image, matrix, offset, tol, **kwargs):
        matrix_affine = ht.hstack((matrix, offset[:, None]))

        with_offset_result: ht.DNDarray = affine.affine_transform(
            image,
            matrix,
            offset=offset,
            **kwargs
        )
        combined_result: ht.DNDarray = affine.affine_transform(
            image, matrix_affine, **kwargs
        )
        assert ht.equal(with_offset_result, combined_result)

        with_offset_comparison = ndimg.affine_transform(
            image.numpy(), matrix.numpy(), offset=offset.numpy(), **kwargs
        )
        combined_comparison = ndimg.affine_transform(
            image.numpy(), matrix_affine.numpy(), **kwargs
        )
        assert np.allclose(
            with_offset_result.numpy(), with_offset_comparison, rtol=0, atol=tol
        )
        assert np.allclose(
            combined_result.numpy(), combined_comparison, rtol=0, atol=tol
        )


    @staticmethod
    def rmse_testing_setup(image, matrix, offset, tol, **kwargs):
        matrix_affine = ht.hstack((matrix, offset[:, None]))

        result: ht.DNDarray = affine.affine_transform(
            image, matrix_affine, **kwargs
        )


        comparison = ndimg.affine_transform(
            image.numpy(), matrix.numpy(), offset=offset.numpy(), **kwargs
        )
        error = root_mean_square_error(result.numpy(), comparison)
        assert error < tol


    @staticmethod
    def bulk_testing_setup(image, matrix, offset, tol, **kwargs):

        offset_stack = ht.expand_dims(offset, offset.ndim)
        # offset.resplit_(matrix.split)
        matrix_affine = ht.concatenate([matrix, offset_stack], axis=(matrix.ndim - 1))

        with_offset_result: ht.DNDarray = affine.affine_transform(
            image,
            matrix,
            offset=offset,
            **kwargs
        )
        combined_result: ht.DNDarray = affine.affine_transform(
            image, matrix_affine, **kwargs
        )
        assert ht.equal(with_offset_result, combined_result)

        with_offset_comparison = [
            ndimg.affine_transform(
                img.numpy(), mat.numpy(), offset=off.numpy(), **kwargs
            )
            for img, mat, off in zip(image, matrix, offset)
        ]
        combined_comparison = [
            ndimg.affine_transform(img.numpy(), mat.numpy(), **kwargs)
            for img, mat in zip(image, matrix_affine)
        ]

        for res, comp in zip(with_offset_result, with_offset_comparison):
            assert np.allclose(res.numpy(), comp, rtol=0, atol=tol)

        for res, comp in zip(combined_result, combined_comparison):
            assert np.allclose(res.numpy(), comp, rtol=0, atol=tol)


    @pytest.mark.parametrize("order", [0, 1])
    @pytest.mark.parametrize("mode", ["grid-constant", "mirror", "nearest"])
    def test_affine_2d(self, order, mode):
        image = self.image_2d_1
        matrix = self.matrix_2d
        offset = self.offset_2d
        TestAffine.default_testing_setup(image, matrix, offset, 0.05, order=order, mode=mode)


    @pytest.mark.parametrize("order", [0, 1])
    @pytest.mark.parametrize("mode", ["grid-constant", "mirror", "nearest"])
    def test_affine_3d(self, order, mode):
        image = self.image_3d_1
        matrix = self.matrix_3d
        offset = self.offset_3d
        TestAffine.default_testing_setup(image, matrix, offset, 0.05, order=order, mode=mode)


    @pytest.mark.parametrize("order", [0, 1])
    @pytest.mark.parametrize("mode", ["grid-constant", "mirror", "nearest"])
    def test_affine_3d_bulk(self, order, mode):
        image = self.image_3d_bulk
        matrix = self.matrix_3d_bulk
        offset = self.offset_3d_bulk
        TestAffine.bulk_testing_setup(image, matrix, offset, 0.05, order=order, mode=mode)

    @pytest.mark.parametrize("order", [3])
    @pytest.mark.parametrize("mode", ["grid-constant", "mirror", "nearest"])
    def test_affine_2d_rmse(self, order, mode):
        image = self.image_2d_1
        matrix = self.matrix_2d
        offset = self.offset_2d
        TestAffine.rmse_testing_setup(image, matrix, offset, tol=3, order=order, mode=mode)


    @pytest.mark.parametrize("order", [0, 1])
    @pytest.mark.parametrize("mode", ["grid-constant", "mirror", "nearest"])
    def test_affine_split(self, order, mode):

        matrix = ht.array(
            (
                [[1, 0, 0, 0, 0],
                 [0, 1, 1, 0, 1],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0]],

                [[1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0],
                 [0, 0, 1, 0, 1],
                 [0, 0, 0, 1, 0]],
            ),
            split=None,
            dtype=ht.float32
        )
        # because bulk are completly seperate, split 0 should always work on bulked arrays.
        # split 1 should not error because it is identity
        # split 2, 3 should error because the axis influence each other
        # split 4 should always work right now, because color axis is ignored
        # matches the result from scipy

        img_split_none = ht.random.random((2, 32, 32, 16, 3), dtype=ht.float32, split=None) * 255
        split_none = affine.affine_transform(img_split_none, matrix, order=order, mode=mode)

        img_split_0 = ht.resplit(img_split_none, 0)
        split_0 = affine.affine_transform(img_split_0, matrix, order=order, mode=mode)
        print(f"0 split{ht.equal(split_none, split_0)}")

        img_split_1 = ht.resplit(img_split_none, 1)
        split_1 = affine.affine_transform(img_split_1, matrix, order=order, mode=mode)
        assert np.allclose(
            split_1.numpy(), split_none.numpy(), rtol=0, atol=0.0005
        )

        img_split_2 = ht.resplit(img_split_none, 2)
        with pytest.raises(RuntimeError):
            affine.affine_transform(img_split_2, matrix, order=order, mode=mode)

        img_split_3 = ht.resplit(img_split_none, 3)
        with pytest.raises(RuntimeError):
            affine.affine_transform(img_split_3, matrix, order=order, mode=mode)

        img_split_4 = ht.resplit(img_split_none, 4)
        split_4 = affine.affine_transform(img_split_4, matrix, order=order, mode=mode)
        print(f"4 split{ht.equal(split_none, split_4)}")


    @pytest.mark.parametrize("order", [0,1])
    @pytest.mark.parametrize("cval", [128,255])
    def test_cval(self, order, cval):
        image = self.image_2d_1
        matrix = self.matrix_2d
        offset = self.offset_2d

        mode = "grid-constant" #cval only has effect in this mode
        TestAffine.rmse_testing_setup(image, matrix, offset, 0.01, cval=cval, mode=mode, order=order)
        TestAffine.default_testing_setup(image, matrix, offset, 0.01, cval=cval, mode=mode, order=order)
