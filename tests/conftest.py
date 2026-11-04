"""Shared pytest fixtures for the Heat test suite (pytest-style counterpart of ``heat.testing.TestCase``)."""

import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pytest
import torch

import heat as ht
from heat.testing.distribution import make_unbalanced


# ---------------------------------------------------------------------------
# Generic setup (mirrors heat.testing.basic_test.TestCase)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DeviceConfig:
    """Devices selected through the environment variable ``HEAT_TEST_USE_DEVICE``."""

    device: ht.Device
    other_device: ht.Device
    envar: str
    is_mps: bool


@pytest.fixture(scope="session")
def device_config() -> DeviceConfig:
    """
    Read the environment variable ``HEAT_TEST_USE_DEVICE`` ("cpu" (default) or "gpu") and select the
    Heat device used for testing, like ``TestCase.setUpClass``.
    """
    envar = os.getenv("HEAT_TEST_USE_DEVICE", "cpu")
    mps_available = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    is_mps = False

    if envar == "cpu":
        ht.use_device("cpu")
        device = ht.cpu
        if torch.cuda.is_available():
            torch.cuda.set_device(torch.device(ht.gpu.torch_device))
            other_device = ht.gpu
        elif mps_available:
            other_device = ht.gpu
        else:
            other_device = ht.cpu
    elif envar == "gpu":
        if torch.cuda.is_available():
            ht.use_device("gpu")
            torch.cuda.set_device(torch.device(ht.gpu.torch_device))
        elif mps_available:
            ht.use_device("gpu")
            is_mps = True
        else:
            raise RuntimeError(
                f'Device {envar} from environment variable "HEAT_TEST_USE_DEVICE" not found! '
                "Neither CUDA, ROCm, nor MPS is available"
            )
        device, other_device = ht.gpu, ht.cpu
    else:
        raise RuntimeError(
            f"Value '{envar}' of environment variable 'HEAT_TEST_USE_DEVICE' is unsupported"
        )
    return DeviceConfig(device, other_device, envar, is_mps)


@pytest.fixture(scope="session")
def device(device_config: DeviceConfig) -> ht.Device:
    """The Heat device to run the tests on."""
    return device_config.device


@pytest.fixture(scope="session")
def other_device(device_config: DeviceConfig) -> ht.Device:
    """A second device (the GPU when testing on CPU and vice versa), if available."""
    return device_config.other_device


@pytest.fixture(scope="session")
def is_mps(device_config: DeviceConfig) -> bool:
    """Whether the tests run on Apple MPS (no float64 support)."""
    return device_config.is_mps


@pytest.fixture(scope="session")
def comm() -> ht.MPICommunication:
    """The MPI communicator."""
    return ht.MPI_WORLD


@pytest.fixture(scope="session")
def rank(comm: ht.MPICommunication) -> int:
    """The MPI rank of this process."""
    return comm.rank


@pytest.fixture(scope="session")
def size(comm: ht.MPICommunication) -> int:
    """The number of MPI processes."""
    return comm.size


@pytest.fixture
def seed() -> int:
    """Set Heat's random seed to 42 before the test, like ``TestCase.setUp``."""
    ht.random.seed(42)
    return 42


@pytest.fixture
def make_array(device: ht.Device, size: int) -> Callable[..., ht.DNDarray]:
    """
    Factory creating a DNDarray from a NumPy array with a given split and distribution layout.

    ``make_array(x_np, dtype=None, split=None, layout="balanced")``: ``layout`` is one of
    :data:`heat.testing.distribution.LAYOUTS`. Non-balanced layouts are skipped when the array is
    not split or when running on a single process, where they cannot differ from "balanced".
    """

    def _make(
        x_np: np.ndarray,
        dtype: Optional[type[ht.datatype]] = None,
        split: Optional[int] = None,
        layout: str = "balanced",
    ) -> ht.DNDarray:
        if layout != "balanced" and (split is None or size < 2):
            pytest.skip(f"layout '{layout}' needs a split array and at least 2 processes")
        x = ht.array(x_np, dtype=dtype, split=split, device=device)
        if layout != "balanced":
            make_unbalanced(x, layout)
        return x

    return _make


# ---------------------------------------------------------------------------
# Local Outlier Factor
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def lof_n_outliers() -> int:
    """Number of outliers at the end of :func:`lof_data`."""
    return 5


@pytest.fixture(scope="session")
def lof_data(lof_n_outliers: int) -> np.ndarray:
    """
    Data set for LOF tests, shape (55, 2), float64.

    - 50 inliers: Gaussian cluster around (0, 0), no exact duplicate distances
    - 5 clearly separated outliers far away from the cluster (the last rows)
    """
    rng = np.random.RandomState(123)
    inliers = rng.normal(loc=0.0, scale=0.8, size=(50, 2))
    outliers = np.array(
        [[6.0, 6.0], [-6.5, 6.0], [6.5, -6.0], [-6.0, -6.5], [0.0, 8.0]], dtype=np.float64
    )
    assert outliers.shape[0] == lof_n_outliers
    return np.vstack([inliers, outliers])


@dataclass(frozen=True)
class LOFReference:
    """Reference results of ``sklearn.neighbors.LocalOutlierFactor``."""

    scores: np.ndarray
    """LOF score per sample (``-negative_outlier_factor_``)."""
    neighbors: np.ndarray
    """Indices of the ``n_neighbors_`` nearest neighbors per sample, the sample itself excluded."""
    n_neighbors: int
    """Number of neighbors actually used (``n_neighbors_``)."""


@pytest.fixture(scope="session")
def sklearn_lof() -> Callable[..., LOFReference]:
    """
    Factory computing the scikit-learn LOF reference
    ``sklearn_lof(x_np, n_neighbors, metric="euclidean", check_ties=True)``.

    The computation is done in float64 with brute-force neighbor search. ``n_neighbors >= n_samples``
    is clamped to ``n_samples - 1``, like in Heat. With ``check_ties``, the factory asserts that the
    neighborhoods are unambiguous (no tie at the k-th neighbor), so the reference does not depend on
    how ties are broken.
    """
    neighbors_mod = pytest.importorskip("sklearn.neighbors")
    cache: dict = {}

    def _reference(
        x_np: np.ndarray, n_neighbors: int, metric: str = "euclidean", check_ties: bool = True
    ) -> LOFReference:
        x64 = np.ascontiguousarray(x_np, dtype=np.float64)
        n_neighbors = min(n_neighbors, x64.shape[0] - 1)
        key = (x64.tobytes(), x64.shape, n_neighbors, metric, check_ties)
        if key not in cache:
            lof = neighbors_mod.LocalOutlierFactor(
                n_neighbors=n_neighbors, metric=metric, algorithm="brute"
            )
            lof.fit(x64)
            k = lof.n_neighbors_
            # check for ties between the k-th and (k+1)-th neighbor (self excluded)
            if check_ties and k + 1 < x64.shape[0]:
                dist, _ = lof.kneighbors(n_neighbors=k + 1)
                gap = dist[:, k] - dist[:, k - 1]
                assert np.all(gap > 1e-6 * np.maximum(dist[:, k], 1.0)), (
                    "LOF reference data has ties at the k-th neighbor; the reference is ambiguous"
                )
            _, idx = lof.kneighbors()
            cache[key] = LOFReference(-lof.negative_outlier_factor_, idx, k)
        return cache[key]

    return _reference
