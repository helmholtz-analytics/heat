import numpy as np
import torch
import pytest
import os
import heat as ht
import itertools
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
    @pytest.mark.parametrize("axis", [0, 1, -1])
    @pytest.mark.parametrize("split", [None, 0, 1])
    @pytest.mark.parametrize("orig_shape", [(10, 1), (1, 10), (10, 10), (20, 5, 10), (5, 10, 30, 2)])
    def test_vectorized_sort_multi_dim(self, orig_shape, split, axis, descending):
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

        res = ht.vectorized_sort(a, axis=axis, descending=descending)
        res_idxs = ht.vectorized_sort(a, axis=axis, descending=descending, return_sort_indices_instead=True)

        assert np.isclose(res.numpy(), expected_res).all()
        assert a.device == res.device
        assert np.equal(sort_idx, res_idxs.numpy()).all()
        assert a.device == res_idxs.device

    @pytest.mark.parametrize("descending", [False, True])
    @pytest.mark.parametrize("axis", [0, -1])
    @pytest.mark.parametrize("split", [None, 0])
    def test_vectorized_sort_one_dim(self, split, axis, descending):
        a = ht.random.randn(10, split=split)
        a_np = a.numpy()

        res = ht.vectorized_sort(a, axis=axis, descending=descending).numpy()
        res_idxs = ht.vectorized_sort(a, axis=axis, descending=descending, return_sort_indices_instead=True).numpy()

        if NUMPY_HAS_NO_DESCENDING_KWARG:
            expected_res_idxs = np.argsort(a_np, axis=axis, stable=True)
            if descending:
                expected_res_idxs = np.flip(expected_res_idxs)
        else:
            expected_res_idxs = np.argsort(a_np, axis=axis, stable=True, descending=descending)

        expected_res = a_np[expected_res_idxs]

        assert np.isclose(res, expected_res).all()
        assert np.equal(expected_res_idxs, res_idxs).all()

    @staticmethod
    def _generate_shape_axis_split_cases():
        shapes = [(10,), (5, 10, 30, 2)]
        for shape in shapes:
            ndims = len(shape)
            # Fixed the empty list bug here
            axiss = list(range(ndims)) + list(range(-ndims, 0))
            splits = list(range(ndims))

            for axis, split in itertools.product(axiss, splits):
                yield shape, axis, split

    @pytest.mark.parametrize("descending", [False, True])
    @pytest.mark.parametrize("shape, axis, split", list(_generate_shape_axis_split_cases()))
    def test_sort_complex(self, shape, axis, split, descending):
        b = ht.random.randn(*shape, dtype=ht.float64, split=split)
        c = ht.random.randn(*shape, dtype=ht.float64, split=split)
        a = b + c * 1j
        arr = a.numpy()

        res, res_idx = ht.sort_complex(a, axis=axis, descending=descending, return_sort_indices=True)
        exp_res = np.sort(arr, axis=axis, stable=True, descending=descending)
        exp_res_idx = np.argsort(arr, axis=axis, stable=True, descending=descending)

        assert (res.numpy() == exp_res).all()
        assert (res_idx.numpy() == exp_res_idx).all()

        assert a.device == res.device
        assert res.device == res_idx.device

    @staticmethod
    def _generate_reorder_params():
        shapes = [(10, ), (20, 30), (10, 2, 40, 3)]

        comm = ht.get_comm()

        for shape in shapes:
            for axis, n in enumerate(shape):
                for split in range(len(shape)):
                    permutation = torch.randperm(n)

                    comm.Bcast(permutation)

                    yield shape, axis, permutation, split

    @pytest.mark.parametrize("resplit_result", [False, True])
    @pytest.mark.parametrize("shape, axis, permutation, split", list(_generate_reorder_params()))
    def test_reorder(self, shape, axis, permutation, split, resplit_result):
        a = ht.random.randn(*shape, split=split)
        arr = torch.from_numpy(a.numpy())

        res = ht.reorder(a, indices=permutation, axis=axis, resplit_result=resplit_result)
        exp_res = arr.transpose(0, axis)[permutation].transpose(0, axis).numpy()

        assert np.isclose(res.numpy(), exp_res).all()
        assert not resplit_result or a.split == res.split

        assert a.device == res.device
