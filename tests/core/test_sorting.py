import numpy as np
import torch
import pytest

import heat as ht


class TestSorting:
    @pytest.mark.parametrize("device", ["cpu", "gpu"])
    @pytest.mark.parametrize("split", [None, 0, 1, 2])
    @pytest.mark.parametrize("descending", [None, True, False])
    @pytest.mark.parametrize("axis", [None, 0, 1, 2])
    def test_sort(self, axis, descending, split, device):
        if not torch.cuda.is_available() and device == "gpu":
            pytest.skip("No gpu available for testing.")

        if np.lib.NumpyVersion(np.__version__) < '2.5.0' and descending is not None:
            pytest.skip(f"NumPy {np.__version__} does not support the 'descending' keyword.")

        data = ht.random.rand(2, 3, 4, split=split)
        result, idx = ht.sort(data, axis=axis, descending=descending, return_sort_indices=True)
        exp = np.sort(data.numpy(), axis=axis, descending=descending)
        exp_idx = np.argsort(data.numpy(), axis=axis, descending=descending)

        assert np.allclose(result.numpy(), exp)
        assert np.allclose(idx.numpy(), exp_idx)

    @pytest.mark.parametrize("device", ["cpu", "gpu"])
    @pytest.mark.parametrize("split", [None, 0, 1, 2])
    @pytest.mark.parametrize("descending", [None, True, False])
    @pytest.mark.parametrize("axis", [None, 0, 1, 2])
    def test_argsort_random(self, axis, descending, split, device):
        if not torch.cuda.is_available() and device == "gpu":
            pytest.skip("No gpu available for testing.")

        if np.lib.NumpyVersion(np.__version__) < '2.5.0' and descending is not None:
            pytest.skip(f"NumPy {np.__version__} does not support the 'descending' keyword.")

        data = ht.random.rand(2, 3, 4, split=split, device=device)
        result_indices = ht.argsort(data, axis=axis, descending=descending)
        exp_indices = np.argsort(data.numpy(), axis=axis, descending=descending)
        assert np.allclose(result_indices.numpy(), exp_indices)

    @pytest.mark.parametrize("device", ["cpu", "gpu"])
    def test_argsort_specific(self, device):
        if not torch.cuda.is_available() and device == "gpu":
            pytest.skip("No gpu available for testing.")

        size = ht.MPI_WORLD.size
        rank = ht.MPI_WORLD.rank

        tensor = torch.tensor(
            [
                [[2, 8, 5], [7, 2, 3]],
                [[6, 5, 2], [1, 8, 7]],
                [[9, 3, 0], [1, 2, 4]],
                [[8, 4, 7], [0, 8, 9]],
            ],
            dtype=torch.int32,
            device=device,
        )

        data = ht.array(tensor, split=0)
        if torch.cuda.is_available() and data.device == ht.gpu and size < 4:
            indices_axis_zero = torch.tensor(
                [[0, 2, 2], [3, 2, 0]], dtype=torch.int32, device=device
            )
        else:
            indices_axis_zero = torch.tensor(
                [[0, 2, 2], [3, 0, 0]], dtype=torch.int32, device=device
            )
        result_indices = ht.argsort(data, axis=0)
        first_indices = result_indices[0].larray
        if rank == 0:
            assert torch.equal(first_indices, indices_axis_zero)

        data = ht.array(tensor, split=1)
        indices_axis_one = torch.tensor(
            [[0, 1, 1]], dtype=torch.int32, device=device
        )
        result_indices = ht.argsort(data, axis=1)
        first_indices = result_indices[0].larray[:1]
        if rank == 0:
            assert torch.equal(first_indices, indices_axis_one)

        data = ht.array(tensor, split=2)
        indices_axis_two = torch.tensor(
            [[0], [1]], dtype=torch.int32, device=device
        )
        result_indices = ht.argsort(data, axis=2)
        first_indices = result_indices[0].larray[:, :1]
        if rank == 0:
            assert torch.equal(first_indices, indices_axis_two)

        # test exceptions
        with pytest.raises(ValueError):
            ht.argsort(data, axis=3)
        with pytest.raises(TypeError):
            ht.argsort(data, axis="1")

        rank = ht.MPI_WORLD.rank
        ht.random.seed(1)
        data = ht.random.randn(100, 1, split=0, device=device)
        indices = ht.argsort(data, axis=0)
        result = ht.resplit(data, None)[indices]
        arr = ht.resplit(result.flatten(), axis=None)

        assert (arr.larray[:-1] <= arr.larray[1:]).all()
