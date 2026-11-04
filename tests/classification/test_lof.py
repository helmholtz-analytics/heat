"""Tests for heat.classification.LocalOutlierFactor."""

import warnings

import numpy as np
import pytest

import heat as ht
from heat.classification.localoutlierfactor import LocalOutlierFactor
from heat.testing.distribution import LAYOUTS, assert_distribution_consistent

SPLITS = [None, 0, 1]
DTYPES = [ht.float32, ht.float64, ht.int32, ht.int64]
FULLY_DISTRIBUTED = [False, True]
METRICS = ["euclidean", "manhattan"]


def dtype_id(dtype):
    """Test id for a Heat dtype."""
    return dtype.__name__


def cast_data(x_np, dtype):
    """Data as it is passed to Heat; integer types get scaled coordinates to avoid ties."""
    if ht.types.heat_type_is_exact(dtype):
        return np.rint(x_np * 1e4).astype(np.int64)
    return x_np


def result_dtype(dtype):
    """Floating-point dtype the LOF scores are expected in for input of type ``dtype``."""
    return ht.types.promote_types(dtype, ht.float32)


def tolerance(dtype):
    """(rtol, atol) for comparing LOF scores computed in ``result_dtype(dtype)`` with sklearn."""
    if result_dtype(dtype) == ht.float64:
        return 1e-6, 1e-6
    return 1e-4, 1e-5


def skip_unsupported(dtype, is_mps):
    """Skip float64 computations on MPS."""
    if is_mps and result_dtype(dtype) == ht.float64:
        pytest.skip("MPS does not support float64")


def fit_lof(X, **kwargs):
    """Create a LocalOutlierFactor with ``kwargs`` and fit it to ``X``."""
    lof = LocalOutlierFactor(**kwargs)
    lof.fit(X)
    return lof


def local_sizes(x):
    """Local sizes along axis 0 on all processes (collective, so asserts agree on every rank)."""
    return x.comm.allgather(x.larray.shape[0] if x.larray.ndim > 0 else 0)


def assert_aligned_with_input(result, X):
    """``result`` is per sample of ``X`` and must be distributed like the samples of ``X``."""
    assert_distribution_consistent(result)
    assert result.shape[0] == X.shape[0]
    if X.split == 0:
        assert result.split == 0
        assert local_sizes(result) == local_sizes(X)
    elif X.split is None:
        assert result.split is None


def assert_neighbors_equal(idx, expected):
    """Compare neighbor sets per sample (the order among neighbors is not compared)."""
    idx_np = idx.numpy()
    assert idx_np.shape == expected.shape
    np.testing.assert_array_equal(np.sort(idx_np, axis=1), np.sort(expected, axis=1))


def assert_self_not_neighbor(idx):
    """No sample is listed as its own neighbor."""
    idx_np = idx.numpy()
    rows = np.arange(idx_np.shape[0])[:, None]
    self_hits = np.nonzero(np.any(idx_np == rows, axis=1))[0]
    assert self_hits.size == 0, f"samples {self_hits.tolist()} are their own neighbor"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"binary_decision": None},
        {"binary_decision": "foo"},
        {"binary_decision": "top_n", "top_n": None},
        {"binary_decision": "top_n", "top_n": 0},
        {"binary_decision": "top_n", "top_n": -1},
        {"binary_decision": "top_n", "top_n": 2.5},
        {"threshold": 0.5},
        {"threshold": 1.0},
        {"threshold": None},
        {"n_neighbors": 0},
        {"n_neighbors": -3},
        {"n_neighbors": 2.5},
        {"n_neighbors": "10"},
        {"metric": None},
        {"metric": "cosine"},
        {"metric": 3},
        {"chunks": 0},
        {"chunks": -1},
        {"chunks": 1.5},
    ],
    ids=str,
)
def test_invalid_parameters(kwargs):
    """Invalid constructor arguments are rejected."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises((TypeError, ValueError)):
            LocalOutlierFactor(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_neighbors": 5},
        {"n_neighbors": 101},
        {"binary_decision": "top_n", "top_n": 3, "threshold": 2.0},
        {"binary_decision": "threshold", "top_n": 3},
    ],
    ids=str,
)
def test_parameter_warnings(kwargs):
    """Unusual or ignored constructor arguments raise a warning."""
    with pytest.warns(UserWarning):
        LocalOutlierFactor(**kwargs)


@pytest.mark.parametrize("split", SPLITS)
def test_fit_invalid_shape(split, device):
    """Only 2-D data (samples x features) is accepted."""
    lof = LocalOutlierFactor()
    with pytest.raises(ValueError):
        lof.fit(ht.zeros((55,), split=None if split == 1 else split, device=device))
    with pytest.raises(ValueError):
        lof.fit(ht.zeros((55, 2, 2), split=split, device=device))


@pytest.mark.parametrize("split", SPLITS)
def test_fit_too_few_samples(split, device):
    """Data sets with at most 10 samples are rejected."""
    X = ht.random.randn(10, 2, split=split, device=device)
    with pytest.raises(ValueError):
        LocalOutlierFactor().fit(X)


@pytest.mark.parametrize("X", [np.zeros((55, 2)), "data"], ids=["numpy", "str"])
def test_fit_not_a_dndarray(X):
    """Input that is not a DNDarray is rejected."""
    with pytest.raises((TypeError, ValueError)):
        LocalOutlierFactor().fit(X)


@pytest.mark.parametrize("split", SPLITS)
def test_top_n_larger_than_samples(split, lof_data, make_array):
    """``top_n`` larger than the number of samples raises a ValueError mentioning ``top_n``."""
    X = make_array(lof_data, split=split)
    lof = LocalOutlierFactor(n_neighbors=10, binary_decision="top_n", top_n=lof_data.shape[0] + 1)
    with pytest.raises(ValueError, match="top_n"):
        lof.fit(X)


# ---------------------------------------------------------------------------
# Correctness against scikit-learn
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("fully_distributed", FULLY_DISTRIBUTED)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("split", SPLITS)
def test_lof_matches_sklearn(
    split, dtype, layout, fully_distributed, metric, lof_data, make_array, sklearn_lof, is_mps
):
    """LOF scores and neighbors match scikit-learn for every split, dtype, layout and mode."""
    skip_unsupported(dtype, is_mps)
    x_np = cast_data(lof_data, dtype)
    X = make_array(x_np, dtype=dtype, split=split, layout=layout)
    lshape_before = X.lshape
    ref = sklearn_lof(x_np, n_neighbors=10, metric=metric)

    lof = fit_lof(X, n_neighbors=10, metric=metric, fully_distributed=fully_distributed)

    # input is untouched
    assert X.dtype == dtype
    assert X.lshape == lshape_before
    np.testing.assert_array_equal(X.numpy(), x_np.astype(X.numpy().dtype))

    # scores
    assert lof.n_neighbors_ == ref.n_neighbors
    assert lof.lof_scores.dtype == result_dtype(dtype)
    assert_aligned_with_input(lof.lof_scores, X)
    rtol, atol = tolerance(dtype)
    np.testing.assert_allclose(lof.lof_scores.numpy(), ref.scores, rtol=rtol, atol=atol)

    # neighbors
    assert lof.idx_n_neighbors.shape == (X.shape[0], ref.n_neighbors)
    assert ht.types.heat_type_is_exact(lof.idx_n_neighbors.dtype)
    assert_aligned_with_input(lof.idx_n_neighbors, X)
    assert_self_not_neighbor(lof.idx_n_neighbors)
    assert_neighbors_equal(lof.idx_n_neighbors, ref.neighbors)

    # binary decision
    assert_aligned_with_input(lof.anomaly, X)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("n_neighbors", [1, 10, 20, 54, 55, 100])
def test_lof_n_neighbors(n_neighbors, split, layout, lof_data, make_array, sklearn_lof):
    """Small and large ``n_neighbors``; ``n_neighbors >= n_samples`` uses ``n_samples - 1``."""
    X = make_array(lof_data, split=split, layout=layout)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ref = sklearn_lof(lof_data, n_neighbors=n_neighbors)
        lof = fit_lof(X, n_neighbors=n_neighbors)

    assert lof.n_neighbors_ == ref.n_neighbors == min(n_neighbors, X.shape[0] - 1)
    np.testing.assert_allclose(lof.lof_scores.numpy(), ref.scores, rtol=1e-6, atol=1e-6)
    assert_self_not_neighbor(lof.idx_n_neighbors)
    assert_neighbors_equal(lof.idx_n_neighbors, ref.neighbors)


@pytest.mark.parametrize("fully_distributed", FULLY_DISTRIBUTED)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("chunks", [1, 2, 3, 7, 60])
def test_lof_chunks(chunks, split, layout, fully_distributed, lof_data, make_array, sklearn_lof):
    """The result does not depend on ``chunks``, also when it exceeds the local number of rows."""
    X = make_array(lof_data, split=split, layout=layout)
    ref = sklearn_lof(lof_data, n_neighbors=10)
    with warnings.catch_warnings():
        # chunks larger than the number of local rows
        warnings.simplefilter("ignore", UserWarning)
        lof = fit_lof(X, n_neighbors=10, chunks=chunks, fully_distributed=fully_distributed)

    np.testing.assert_allclose(lof.lof_scores.numpy(), ref.scores, rtol=1e-6, atol=1e-6)
    assert_neighbors_equal(lof.idx_n_neighbors, ref.neighbors)


@pytest.mark.parametrize("fully_distributed", FULLY_DISTRIBUTED)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("threshold", [1.25, 3.0])
def test_lof_threshold_decision(
    threshold, split, layout, fully_distributed, lof_data, lof_n_outliers, make_array, sklearn_lof
):
    """Samples with a score of at least ``threshold`` are flagged as outliers."""
    X = make_array(lof_data, split=split, layout=layout)
    ref = sklearn_lof(lof_data, n_neighbors=10)
    lof = fit_lof(
        X,
        n_neighbors=10,
        binary_decision="threshold",
        threshold=threshold,
        fully_distributed=fully_distributed,
    )

    assert_aligned_with_input(lof.anomaly, X)
    anomaly = lof.anomaly.numpy()
    np.testing.assert_array_equal(anomaly, np.where(ref.scores >= threshold, 1, -1))
    if threshold == 3.0:
        # exactly the explicit outliers
        assert np.all(anomaly[:-lof_n_outliers] == -1)
        assert np.all(anomaly[-lof_n_outliers:] == 1)


@pytest.mark.parametrize("fully_distributed", FULLY_DISTRIBUTED)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("dtype", [ht.float32, ht.float64], ids=dtype_id)
@pytest.mark.parametrize("top_n", [1, 3, 5, 20, 55])
def test_lof_top_n_decision(
    top_n, dtype, split, layout, fully_distributed, lof_data, make_array, sklearn_lof, is_mps
):
    """Exactly the ``top_n`` samples with the largest scores are flagged as outliers."""
    skip_unsupported(dtype, is_mps)
    X = make_array(lof_data, dtype=dtype, split=split, layout=layout)
    ref = sklearn_lof(lof_data, n_neighbors=10)
    lof = fit_lof(
        X,
        n_neighbors=10,
        binary_decision="top_n",
        top_n=top_n,
        fully_distributed=fully_distributed,
    )

    assert_aligned_with_input(lof.anomaly, X)
    anomaly = lof.anomaly.numpy()
    assert np.count_nonzero(anomaly == 1) == top_n
    assert np.count_nonzero(anomaly == -1) == X.shape[0] - top_n
    expected = np.sort(np.argsort(-ref.scores)[:top_n])
    np.testing.assert_array_equal(np.nonzero(anomaly == 1)[0], expected)


# ---------------------------------------------------------------------------
# Numerically difficult data
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fully_distributed", FULLY_DISTRIBUTED)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("split", SPLITS)
def test_lof_float32_large_offset(
    split, layout, fully_distributed, lof_data, make_array, sklearn_lof
):
    """More than 25 rows of float32 data far from the origin (torch.cdist's matmul path)."""
    x_np = (lof_data + 1000.0).astype(np.float32)
    X = make_array(x_np, dtype=ht.float32, split=split, layout=layout)
    ref = sklearn_lof(x_np, n_neighbors=10)
    lof = fit_lof(X, n_neighbors=10, fully_distributed=fully_distributed)

    assert_self_not_neighbor(lof.idx_n_neighbors)
    np.testing.assert_allclose(lof.lof_scores.numpy(), ref.scores, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("fully_distributed", FULLY_DISTRIBUTED)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("split", SPLITS)
def test_lof_duplicates(split, layout, fully_distributed, lof_data, make_array, sklearn_lof):
    """Duplicated samples: the copy is a neighbor at distance 0, the sample itself is not."""
    duplicated = [3, 17, 42]
    x_np = np.vstack([lof_data, lof_data[duplicated]])
    copies = np.arange(lof_data.shape[0], x_np.shape[0])
    X = make_array(x_np, split=split, layout=layout)
    # identical points have identical LOF quantities, so ties between copies don't change the scores
    ref = sklearn_lof(x_np, n_neighbors=10, check_ties=False)
    lof = fit_lof(X, n_neighbors=10, fully_distributed=fully_distributed)

    assert_self_not_neighbor(lof.idx_n_neighbors)
    idx_np = lof.idx_n_neighbors.numpy()
    for original, copy in zip(duplicated, copies):
        assert copy in idx_np[original]
        assert original in idx_np[copy]
    np.testing.assert_allclose(lof.lof_scores.numpy(), ref.scores, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("fully_distributed", FULLY_DISTRIBUTED)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("split", [0, 1])
def test_lof_ties_independent_of_distribution(
    split, layout, fully_distributed, lof_n_outliers, make_array
):
    """On a regular grid (many equal distances) the result must not depend on the distribution."""
    grid = np.stack(np.meshgrid(np.arange(7.0), np.arange(7.0)), axis=-1).reshape(-1, 2)
    outliers = 20.0 + 5.0 * np.arange(2 * lof_n_outliers, dtype=np.float64).reshape(-1, 2)
    x_np = np.vstack([grid, outliers])

    expected = fit_lof(make_array(x_np), n_neighbors=10, fully_distributed=fully_distributed)
    lof = fit_lof(
        make_array(x_np, split=split, layout=layout),
        n_neighbors=10,
        fully_distributed=fully_distributed,
    )

    np.testing.assert_allclose(
        lof.lof_scores.numpy(), expected.lof_scores.numpy(), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("split", SPLITS)
def test_lof_gaussian_metric_neighbors(split, layout, lof_data, make_array, sklearn_lof):
    """The gaussian kernel is a similarity: the nearest neighbors are the most similar points."""
    X = make_array(lof_data, split=split, layout=layout)
    ref = sklearn_lof(lof_data, n_neighbors=10, metric="euclidean")
    lof = fit_lof(X, n_neighbors=10, metric="gaussian")

    assert_self_not_neighbor(lof.idx_n_neighbors)
    assert_neighbors_equal(lof.idx_n_neighbors, ref.neighbors)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fully_distributed", FULLY_DISTRIBUTED)
@pytest.mark.parametrize("idx_layout", LAYOUTS)
@pytest.mark.parametrize("a_layout", LAYOUTS)
@pytest.mark.parametrize("split", [None, 0])
def test_advanced_indexing(split, a_layout, idx_layout, fully_distributed, make_array):
    """``A[idx]`` with distributed ``A`` and ``idx``, aligned with ``idx``."""
    n, k = 55, 10
    a_np = 1.5 * np.arange(n, dtype=np.float64)
    idx_np = np.random.RandomState(7).randint(0, n, size=(n, k)).astype(np.int64)
    A = make_array(a_np, split=split, layout=a_layout)
    idx = make_array(idx_np, split=split, layout=idx_layout)

    lof = LocalOutlierFactor(fully_distributed=fully_distributed)
    result = lof._advanced_indexing(A, idx)

    assert result.shape == idx.shape
    assert result.split == idx.split
    assert_distribution_consistent(result)
    assert local_sizes(result) == local_sizes(idx)
    np.testing.assert_array_equal(result.numpy(), a_np[idx_np])
