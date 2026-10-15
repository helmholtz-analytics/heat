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

        # expected results for splits for the specific matrix used:
        # because bulk are completly seperate, split 0 should always work on bulked arrays.
        # split 1 should not error because the axis does not get changed by matrix
        # split 2, 3 should error because the axis have influence on other axes or get influenced by other axes
        # split 4 should always work right now, because color axis is ignored
        cls.matrix_test_split = ht.array(([[1, 0, 0, 0, 0],
                                [0, 1, 1, 0, 1],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0]],

                                [[1, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0],
                                [0, 0, 1, 0, 1],
                                [0, 0, 0, 1, 0]],),
                                split=None,
                                dtype=ht.float32)

        cls.image_test_split = ht.random.random((2, 32, 32, 16, 3), dtype=ht.float32, split=None) * 255


    @staticmethod
    def default_testing_setup(image, matrix, offset, tol, **kwargs):

        with_offset_result: ht.DNDarray = affine.affine_transform(
            image,
            matrix,
            offset=offset,
            **kwargs
        )

        if not "mode" in kwargs:
            kwargs["mode"] = "grid-constant" # because default mode is different for heat method

        numpy_image = image.numpy()
        numpy_matrix = matrix.numpy()
        if isinstance(offset, ht.DNDarray):
            numpy_offset = offset.numpy()
        elif offset is None:
            numpy_offset = 0.0
        else:
            numpy_offset = offset
        with_offset_comparison = ndimg.affine_transform(
            numpy_image, numpy_matrix, offset=numpy_offset, **kwargs
        )

        assert np.allclose(
            with_offset_result.numpy(), with_offset_comparison, rtol=0, atol=tol
        )

        if not offset is None:

            matrix_affine = ht.hstack((matrix, offset[:, None]))
            combined_result: ht.DNDarray = affine.affine_transform(
                image, matrix_affine, **kwargs
            )
            combined_comparison = ndimg.affine_transform(
                image.numpy(), matrix_affine.numpy(), **kwargs
            )

            assert ht.equal(with_offset_result, combined_result)

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

        with_offset_result: ht.DNDarray = affine.affine_transform(
            image,
            matrix,
            offset=offset,
            **kwargs
        )
        comp_kwargs = kwargs.copy()
        out_shape_no_bulk = None
        if "output_shape" in comp_kwargs:
            output_shape_whole = comp_kwargs.pop("output_shape")
            if  not output_shape_whole is None:
                out_shape_no_bulk = output_shape_whole[1:]

        if not offset is None:
            with_offset_comparison = [
                ndimg.affine_transform(
                    img.numpy(), mat.numpy(), offset=off.numpy(), output_shape=out_shape_no_bulk, **comp_kwargs
                )
                for img, mat, off in zip(image, matrix, offset)
            ]
        else:
            with_offset_comparison = [
                ndimg.affine_transform(
                    img.numpy(), mat.numpy(), output_shape=out_shape_no_bulk, **comp_kwargs
                )
                for img, mat in zip(image, matrix)
            ]

        for i in range(0, with_offset_result.shape[0]):
            assert np.allclose(with_offset_result[i].numpy(), with_offset_comparison[i], rtol=0, atol=tol)

        if not offset is None:
            offset_stack = ht.expand_dims(offset, offset.ndim)
            # offset.resplit_(matrix.split)
            matrix_affine = ht.concatenate([matrix, offset_stack], axis=(matrix.ndim - 1))

            combined_result: ht.DNDarray = affine.affine_transform(
                image, matrix_affine, **kwargs
            )
            assert ht.equal(with_offset_result, combined_result)

            combined_comparison = [
                ndimg.affine_transform(img.numpy(), mat.numpy(), output_shape=out_shape_no_bulk, **comp_kwargs)
                for img, mat in zip(image, matrix_affine)
            ]

            for res, comp in zip(combined_result, combined_comparison):
                assert np.allclose(res.numpy(), comp, rtol=0, atol=tol)


    @pytest.mark.parametrize("out_shape", [None, (128,32,3)])
    @pytest.mark.parametrize("order", [0, 1])
    @pytest.mark.parametrize("mode", ["grid-constant", "mirror", "nearest"])
    def test_affine_2d(self, order, mode, out_shape):
        image = self.image_2d_1
        matrix = self.matrix_2d
        offset = self.offset_2d
        TestAffine.default_testing_setup(image, matrix, offset, 0.05, order=order, mode=mode, output_shape=out_shape)


    @pytest.mark.parametrize("order", [0, 1])
    @pytest.mark.parametrize("mode", ["grid-constant", "mirror", "nearest"])
    def test_affine_3d(self, order, mode):
        image = self.image_3d_1
        matrix = self.matrix_3d
        offset = self.offset_3d
        TestAffine.default_testing_setup(image, matrix, offset, 0.05, order=order, mode=mode)


    @pytest.mark.parametrize("out_shape", [None, (5,16,64,128,3)])
    @pytest.mark.parametrize("order", [0, 1])
    @pytest.mark.parametrize("mode", ["grid-constant", "mirror", "nearest"])
    def test_affine_3d_bulk(self, order, mode, out_shape):
        image = self.image_3d_bulk
        matrix = self.matrix_3d_bulk
        offset = self.offset_3d_bulk
        TestAffine.bulk_testing_setup(image, matrix, offset, 0.05, order=order, mode=mode, output_shape=out_shape)


    @pytest.mark.parametrize("order", [3])
    @pytest.mark.parametrize("mode", ["grid-constant", "mirror", "nearest"])
    def test_affine_2d_rmse(self, order, mode):
        image = self.image_2d_1
        matrix = self.matrix_2d
        offset = self.offset_2d
        TestAffine.rmse_testing_setup(image, matrix, offset, tol=3, order=order, mode=mode)


    @pytest.mark.parametrize("split", [2, 3])
    @pytest.mark.parametrize("order", [0, 1])
    @pytest.mark.parametrize("mode", ["grid-constant"])
    def test_affine_split_invalid(self, order, mode, split):

        matrix = self.matrix_test_split
        img_split_none = self.image_test_split

        img_split = ht.resplit(img_split_none, split)
        with pytest.raises(RuntimeError):
            affine.affine_transform(img_split, matrix, order=order, mode=mode)


    @pytest.mark.parametrize("split", [0, 1, 4])
    @pytest.mark.parametrize("order", [0, 1])
    @pytest.mark.parametrize("mode", ["grid-constant"])
    def test_affine_split_valid(self, order, mode, split):

        matrix = self.matrix_test_split
        img_split_none = self.image_test_split

        split_none = affine.affine_transform(img_split_none, matrix, order=order, mode=mode)
        img_split = ht.resplit(img_split_none, split)
        split_result = affine.affine_transform(img_split, matrix, order=order, mode=mode)
        assert np.allclose(split_none.numpy(), split_result.numpy(), rtol=0, atol=0.001)


    @pytest.mark.parametrize("order", [0,1])
    @pytest.mark.parametrize("cval", [128,255])
    def test_cval(self, order, cval):
        input = self.image_2d_1
        matrix = self.matrix_2d
        offset = self.offset_2d

        mode = "grid-constant" #cval only has effect in this mode
        TestAffine.rmse_testing_setup(input, matrix, offset, 0.005, cval=cval, mode=mode, order=order)
        TestAffine.default_testing_setup(input, matrix, offset, 0.005, cval=cval, mode=mode, order=order)


    def test_split_edge_cases(self):

        # should not throw error, because matrix gets resplit to None
        input = create_checker((10,10), 2, dtype=ht.float32)
        matrix = ht.array([[0, 1, 0, 0], [-1, 0, 0, 9], [0, 0, 1, 0]], dtype=ht.float32, split=0)
        TestAffine.default_testing_setup(input, matrix, offset=None, tol = 0.005)

        # should not throw error, matrix gets resplit to None
        input = ht.random.random((10,10,3), dtype=ht.float32, split=1)
        matrix = ht.array([[1,0,0],[0,1,0],[0,0,1]], dtype=ht.float32, split=0)
        TestAffine.default_testing_setup(input, matrix, offset=None, tol = 0.005)

        # matrix split along bulk, input not
        input = create_checker((1, 10,10), 2, dtype=ht.float32)
        matrix = ht.array([[[0, 1, 0, 0], [-1, 0, 0, 9], [0, 0, 1, 0]]], dtype=ht.float32, split=0)
        TestAffine.bulk_testing_setup(input, matrix, offset=None, tol = 0.005)

        # output_shape differs from input_shape along split axis
        input = ht.empty((10,10,3),dtype=ht.float32, split=0)
        matrix = ht.empty((3,4), dtype=ht.float32, split=0)
        with pytest.raises(RuntimeError):
            affine.affine_transform(input, matrix, output_shape=(15,10,3))


    def test_unsupported_shapes(self):
        # shape missmatch - matrix has bulk, input not
        input = ht.empty((10,10,3),dtype=ht.float32)
        matrix = ht.empty((2,3,4), dtype=ht.float32)
        with pytest.raises(ValueError):
            affine.affine_transform(input, matrix)


        # shape missmatch - matrix to big
        matrix = ht.empty((4,5), dtype=ht.float32)
        with pytest.raises(ValueError):
            affine.affine_transform(input, matrix)

        # shape missmatch - matrix to small
        matrix = ht.empty((2,3), dtype=ht.float32)
        with pytest.raises(ValueError):
            affine.affine_transform(input, matrix)

        # shape missmatch - matrix too many dimensions (max is 3)
        input = ht.empty((1,10,10,3),dtype=ht.float32)
        matrix = ht.empty((1,2,3,4), dtype=ht.float32)
        with pytest.raises(ValueError):
            affine.affine_transform(input, matrix)

        # shape missmatch - matrix has no bulk, but input
        matrix = ht.empty((3,4), dtype=ht.float32)
        with pytest.raises(ValueError):
            affine.affine_transform(input, matrix)


    def test_unsupported_modes_order(self):
        # invalid mode
        input = ht.empty((1, 10,10,3),dtype=ht.float32)
        matrix = ht.empty((1,3,4), dtype=ht.float32)
        with pytest.raises(ValueError):
            affine.affine_transform(input, matrix, mode="test")
        with pytest.raises(NotImplementedError):
            affine.affine_transform(input, matrix, mode="constant")

        # invalid order
        with pytest.raises(NotImplementedError):
            affine.affine_transform(input, matrix, order=2)
        with pytest.raises(ValueError):
            affine.affine_transform(input, matrix, order=8)


    def test_invalid_offset(self):
        # invalid offset shape
        input = ht.empty((10,10,3),dtype=ht.float32)
        matrix = ht.empty((3,3), dtype=ht.float32)
        offset = ht.empty((2,), dtype=ht.float32)
        with pytest.raises(ValueError):
            affine.affine_transform(input, matrix, offset=offset)

        # invalid offset shape with matrix including offset, only warn user
        input = ht.empty((10,10,3),dtype=ht.float32)
        matrix = ht.empty((3,4), dtype=ht.float32)
        offset = ht.empty((3,), dtype=ht.float32)
        with pytest.warns(UserWarning, match="offset is not used, since matrix provides offset information in its rightmost column"):
            affine.affine_transform(input, matrix, offset=offset)


    def test_invalid_output_shape(self):
        # invalid output_shape
        input = ht.empty((1, 10, 10, 3),dtype=ht.float32)
        matrix = ht.empty((1, 3, 4), dtype=ht.float32)
        with pytest.raises(ValueError):
            affine.affine_transform(input, matrix, output_shape=(2, 10, 10, 3))
        with pytest.raises(ValueError):
            affine.affine_transform(input, matrix, output_shape=(1, 10, 10, 5))
        input = ht.resplit(input, 1)
        # extension of dimensions in split axis direction
        with pytest.raises(RuntimeError):
            affine.affine_transform(input, matrix, output_shape=(1, 15, 10, 3))
        # more dimensions that input dimensiont (could broadcast in the future?)
        with pytest.raises(ValueError):
            affine.affine_transform(input, matrix, output_shape=(17, 1, 10, 10, 3))
