import pytest
import numpy as np
import heat as ht
from heat.ndimage.util import create_checker, root_mean_square_error, center_transform

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
