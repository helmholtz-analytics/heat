import numpy as np
import torch
import pytest
import os
import heat as ht
from heat.testing.basic_test import TestCase

NUMPY_HAS_NO_DESCENDING_KWARG = np.lib.NumpyVersion(np.__version__) < np.lib.NumpyVersion("2.5.0")

class TestSorting:
    @classmethod
    def setup_class(cls):
        TestCase.setUpClass()

    @pytest.mark.parametrize("split", [None, 0, 1, 2])
    @pytest.mark.parametrize("descending", [None, True, False])
    @pytest.mark.parametrize("axis", [None, 0, 1, 2])
    def test_sort(self, axis, descending, split):
        kwargs = {"axis": axis}

        if descending in [True, False]:
            if NUMPY_HAS_NO_DESCENDING_KWARG:
                pytest.skip(f"NumPy {np.__version__} does not support the 'descending' keyword.")
            else:
                kwargs["descending"] = descending

        data = ht.random.rand(2, 3, 4, split=split)
        result, idx = ht.sort(data, return_sort_indices=True, **kwargs)
        exp = np.sort(data.numpy(), **kwargs)
        exp_idx = np.argsort(data.numpy(), **kwargs)

        assert np.allclose(result.numpy(), exp)
        assert np.allclose(idx.numpy(), exp_idx)

    @pytest.mark.parametrize("split", [None, 0, 1, 2])
    @pytest.mark.parametrize("descending", [None, True, False])
    @pytest.mark.parametrize("axis", [None, 0, 1, 2])
    def test_argsort_random(self, axis, descending, split):
        kwargs = {"axis": axis}

        if descending in [True, False]:
            if NUMPY_HAS_NO_DESCENDING_KWARG:
                pytest.skip(f"NumPy {np.__version__} does not support the 'descending' keyword.")
            else:
                kwargs["descending"] = descending

        data = ht.random.rand(2, 3, 4, split=split)
        result_indices = ht.argsort(data, **kwargs)
        exp_indices = np.argsort(data.numpy(), **kwargs)
        assert np.allclose(result_indices.numpy(), exp_indices)

    @pytest.mark.parametrize("descending", [False, True])
    @pytest.mark.parametrize("stable", [False, True])
    @pytest.mark.parametrize("axis", [0, 1, -1])
    @pytest.mark.parametrize("split", [None, 0, 1])
    @pytest.mark.parametrize("orig_shape", [(10, 1), (1, 10), (10, 10), (20, 5, 10), (5, 10, 30, 2)])
    def test_vectorized_sort_multi_dim(self, orig_shape, split, axis, stable, descending):
        a = ht.random.randn(*orig_shape, split=split)
        arr = np.swapaxes(a.numpy(), 0, axis)
        shape = arr.shape
        arr = arr.reshape(arr.shape[0], -1)

        # Numpy Lexsort uses the last key as the primary key
        keys = tuple(arr[:, i] for i in range(arr.shape[1] - 1, -1, -1))
        if descending:
            keys = tuple(-k for k in keys)

        sort_idx = np.lexsort(keys)
        expected_res = arr[sort_idx].reshape(shape).swapaxes(0, axis)

        res = ht.vectorized_sort(a, axis=axis, stable=stable, descending=descending).numpy()

        assert np.isclose(res, expected_res).all()

    @pytest.mark.parametrize("descending", [False, True])
    @pytest.mark.parametrize("stable", [False, True])
    @pytest.mark.parametrize("axis", [0, -1])
    @pytest.mark.parametrize("split", [None, 0])
    def test_vectorized_sort_one_dim(self, split, axis, stable, descending):
        a = ht.random.randn(10, split=split)
        a_np = a.numpy()

        res = ht.vectorized_sort(a, axis=axis, stable=stable, descending=descending).numpy()

        if NUMPY_HAS_NO_DESCENDING_KWARG:
            if descending:
                a_np *= -1

            expected_res = np.sort(a_np, axis=axis, stable=stable)

            if descending:
                expected_res *= -1
        else:
            expected_res = np.sort(a_np, axis=axis, stable=stable, descending=descending)
        assert np.isclose(res, expected_res).all()
