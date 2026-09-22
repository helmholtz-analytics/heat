import pytest
import scipy.ndimage as ndimg
import numpy as np
import heat as ht
import heat.ndimage.affine as affine
from heat.testing.basic_test import TestCase
from heat.ndimage.util import create_checker, root_mean_square_error, center_transform
from heat.ndimage.affine import affine_transform
from examples.ndimages.affine_helpers_DONOTCOMMIT import visual_compare

class TestUtil:
    def test_center_transform(self):
        matrix = ht.array([[ 0, 1, 0],
                           [-1, 0, 0],
                           [ 0, 0, 1]],
                           dtype=ht.float32)
        result = center_transform(matrix, (10,10,3))
        compare = ht.array([[0, 1, 0, 0],
                            [-1, 0, 0, 9],
                            [0, 0, 1, 0]],
                           dtype=ht.float32)
        assert ht.equal(result, compare)


        matrix = ht.array([[[ 0, 1, 0],
                            [-1, 0, 0],
                            [ 0, 0, 1]],

                            [[-1,  0, 0],
                             [ 0, -1, 0],
                             [ 0,  0, 1]]],
                           dtype=ht.float32)
        result = center_transform(matrix, (2,10,10,3))
        compare = ht.array([[[ 0, 1, 0, 0],
                             [-1, 0, 0, 9],
                             [ 0, 0, 1, 0]],

                            [[-1,  0, 0, 9],
                             [ 0, -1, 0, 9],
                             [ 0,  0, 1, 0]]],
                           dtype=ht.float32)
        assert ht.equal(result, compare)
        # img = create_checker((10,10),4)
        # vis_result = affine_transform(img, result)
        # vis_compare = affine_transform(img, compare)
        # visual_compare(vis_result, vis_compare.numpy(), has_bulk=False)

        with pytest.raises(ValueError):
            center_transform(matrix, (10,10,3))

        #invalid inputs
        matrix = ht.empty((1,2,3,3), dtype=ht.float32)
        with pytest.raises(ValueError):
            center_transform(matrix, (1,2,10,10,3))
        matrix = ht.empty((3,), dtype=ht.float32)
        with pytest.raises(ValueError):
            center_transform(matrix, (10,10,3))


    def test_create_checker(self):
        min_val = 3
        max_val = 117
        checker = create_checker((4,4), 2, min_value=min_val, max_value=max_val)
        assert checker.shape == (4,4,3)
        assert ht.min(checker) >= min_val
        assert ht.max(checker) < max_val


    def test_rmse(self):
        input = np.random.random((10,10,3))
        assert root_mean_square_error(input, input) == 0
        input = np.array([1,0])
        input2 = np.array([1,1])
        assert np.isclose(root_mean_square_error(input, input2), np.sqrt(1/2))
