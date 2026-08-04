import numpy as np
import torch
import pytest
import os
import heat as ht
from heat.testing.basic_test import TestCase


class TestSorting:
    def __init__(self, *args, **kwargs):
        TestCase.setUpClass()

    @pytest.mark.parametrize("split", [None, 0, 1, 2])
    @pytest.mark.parametrize("descending", [None, True, False])
    @pytest.mark.parametrize("axis", [None, 0, 1, 2])
    def test_sort(self, axis, descending, split):
        kwargs = {"axis": axis}

        if descending in [True, False]:
            if np.lib.NumpyVersion(np.__version__) < np.lib.NumpyVersion("2.5.0"):
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
            if np.lib.NumpyVersion(np.__version__) < np.lib.NumpyVersion("2.5.0"):
                pytest.skip(f"NumPy {np.__version__} does not support the 'descending' keyword.")
            else:
                kwargs["descending"] = descending

        data = ht.random.rand(2, 3, 4, split=split)
        result_indices = ht.argsort(data, **kwargs)
        exp_indices = np.argsort(data.numpy(), **kwargs)
        assert np.allclose(result_indices.numpy(), exp_indices)
