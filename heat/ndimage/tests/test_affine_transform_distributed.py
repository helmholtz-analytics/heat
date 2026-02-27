import numpy as np
import pytest
import heat as ht
from mpi4py import MPI

from heat.ndimage.affine import affine_transform

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@pytest.mark.mpi
def test_undistributed_affine_translation_backward():
    """
    Backward warping with nearest padding.

    out[z, y, x] = in[z, y, x - 1]
    with x < 0 clamped to 0.
    """
    data = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
    x = ht.array(data, split=None)

    M = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=np.float64,
    )

    y = affine_transform(x, M, order=0, mode="nearest").numpy()

    # correct backward-warp reference
    ref = np.zeros_like(data)
    ref[:, :, 0] = data[:, :, 0]
    ref[:, :, 1] = data[:, :, 0]

    assert np.allclose(y, ref)


@pytest.mark.mpi
def test_distributed_non_split_axis_translation_matches_undistributed():
    data = np.arange(48, dtype=np.float32).reshape(6, 4, 2)

    M = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=np.float64,
    )

    x_full = ht.array(data, split=None)
    y_ref = affine_transform(x_full, M, order=0).numpy()

    x_dist = ht.array(data, split=0)
    y_dist = affine_transform(x_dist, M, order=0)

    assert y_dist.split == 0
    assert np.allclose(y_dist.resplit(None).numpy(), y_ref)


@pytest.mark.mpi
def test_split_axis_translation_supported_via_resplit():
    data = np.zeros((8, 4, 4), dtype=np.float32)
    if rank == 0:
        data[1, 2, 2] = 1.0
    data = comm.bcast(data, root=0)

    x = ht.array(data, split=0)

    # translate +3 along z
    M = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 3],
        ],
        dtype=np.float64,
    )

    y = affine_transform(x, M, order=0)

    ref = affine_transform(ht.array(data, split=None), M, order=0).numpy()
    got = y.resplit(None).numpy()

    assert np.allclose(got, ref)


@pytest.mark.mpi
def test_distributed_vs_undistributed_equivalence():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(8, 5, 4)).astype(np.float32)

    M = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=np.float64,
    )

    y_ref = affine_transform(ht.array(data, split=None), M, order=0).numpy()
    y_dist = affine_transform(ht.array(data, split=0), M, order=0)

    assert np.allclose(y_dist.resplit(None).numpy(), y_ref)
